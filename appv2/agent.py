"""Compact goal planner and sequential model-directed tool loop."""

import inspect
import json
from collections.abc import Callable
from typing import Any

from jsonschema.exceptions import ValidationError
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from app.agent.chat_models import create_chat_model
from app.agent.llm import DATA_POLICY, UNTRUSTED_DATA_POLICY, load_prompt
from app.db.session import Database
from app.security import TokenAuthenticator
from app.settings import Settings
from app.tools.paypal import PayPalSandbox, catalog_adapters
from app.tools.system_search import SystemSearch
from appv2.catalog import Catalog
from appv2.config import local_principal
from appv2.tools import ToolDispatcher, stop_after_result


class ChatPlan(BaseModel):
    user_goal: str = Field(min_length=1)
    tool_actions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


def load_model(settings: Settings):
    return create_chat_model(settings)


def langfuse_callbacks(settings):
    if not (
        settings.langfuse_tracing_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
        and settings.langfuse_base_url
    ):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


async def run_chat(
    request: str,
    settings: Settings,
    log_event: Callable[[dict[str, Any]], Any],
    *,
    run_id,
    conversation_id,
    conversation_history: list[dict[str, str]] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    principal, _token = local_principal(settings)
    results: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    max_calls = 20
    tool_limit_reached = False
    paypal = None
    catalog = None
    database = None

    async def record(event: dict[str, Any]) -> None:
        events.append(event)
        logged = log_event(event)
        if inspect.isawaitable(logged):
            await logged

    try:
        paypal = (
            PayPalSandbox(settings)
            if (settings.paypal_sandbox_client_id and settings.paypal_sandbox_client_secret)
            else None
        )
        adapters = catalog_adapters(paypal) if paypal else {}
        database = Database(settings)
        await database.open(require_checkpoints=False)
        adapters["system_search"] = SystemSearch(
            database,
            TokenAuthenticator(settings),
            allowed_adapters=set(adapters) | {"rag_pipeline", "system_search"},
        )
        catalog = Catalog(settings, set(adapters) | {"rag_pipeline"})
        model = load_model(settings)
        callbacks = langfuse_callbacks(settings)
        dispatcher = ToolDispatcher(
            settings,
            catalog,
            principal,
            adapters,
            record,
            run_id=run_id,
            conversation_id=conversation_id,
        )
        classify_system = (
            load_prompt("v2", "classify")
            + " If the request can be answered conversationally without a tool, return an empty tool_actions list. "
            + DATA_POLICY
        )
        classify_user = json.dumps(
            {"request": request, "recent_conversation": conversation_history or []},
            ensure_ascii=False,
        )
        await record(
            {
                "event": "llm_started",
                "purpose": "classify",
                "model": settings.ollama_model_name
                if settings.llm_provider == "ollama"
                else settings.gemini_model_name,
                "system_instruction": classify_system,
                "prompt": classify_user,
            }
        )
        try:
            plan = await model.with_structured_output(ChatPlan).ainvoke(
                [SystemMessage(classify_system), HumanMessage(classify_user)],
                config={"callbacks": callbacks} if callbacks else None,
            )
        except Exception as error:
            await record(
                {
                    "event": "llm_failed",
                    "purpose": "classify",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        await record(
            {
                "event": "llm_finished",
                "purpose": "classify",
                "response": plan.model_dump(mode="json"),
            }
        )
        if len(plan.tool_actions) > settings.max_plan_steps:
            raise ValueError("The model returned an invalid goal plan")
        goal = {
            "user_goal": plan.user_goal,
            "tool_actions": plan.tool_actions,
            "missing_information": plan.missing_information,
        }
        await record({"event": "goal_plan", **goal})
        if progress:
            progress({"event": "goal_plan", **goal})
        if plan.missing_information:
            return {
                "summary": "I need more information: " + ", ".join(plan.missing_information),
                "goal_plan": goal,
                "results": {},
                "events": events,
            }

        for action_index, action in enumerate(plan.tool_actions):
            query = f"{plan.user_goal} {action}"
            await record({"event": "tool_retrieval", "action": action, "query": query})
            cards = await catalog.tools(query, principal, limit=5)
            await record(
                {
                    "event": "tool_shortlist",
                    "action": action,
                    "tools": [
                        {
                            "name": card.name,
                            "description": card.description,
                            "tool_id": card.tool_id,
                        }
                        for card in cards
                    ],
                }
            )
            if not cards:
                results[f"action_{action_index + 1}"] = {
                    "error": "No authorized tool matched this action."
                }
                continue
            declarations, by_name = dispatcher.declarations(cards)
            bound_model = model.bind_tools(declarations)
            selection_system = load_prompt("v2", "select_tool") + " " + DATA_POLICY
            messages = [
                SystemMessage(selection_system),
                HumanMessage(
                    json.dumps(
                        {
                            "original_request": request,
                            "recent_conversation": conversation_history or [],
                            "user_goal": plan.user_goal,
                            "current_action": action,
                            "verified_prior_results": results,
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
            action_results: list[Any] = []
            while dispatcher.tool_call_count < max_calls:
                await record(
                    {
                        "event": "llm_started",
                        "purpose": "tool_selection",
                        "action": action,
                        "model": settings.ollama_model_name
                        if settings.llm_provider == "ollama"
                        else settings.gemini_model_name,
                        "messages": [
                            {"role": "system", "content": messages[0].content},
                            {"role": "user", "content": messages[1].content},
                            *[
                                {
                                    "role": getattr(item, "type", "message"),
                                    "content": getattr(item, "content", ""),
                                    **(
                                        {"tool_calls": getattr(item, "tool_calls", [])}
                                        if getattr(item, "tool_calls", None)
                                        else {}
                                    ),
                                    **(
                                        {"tool_call_id": item.tool_call_id}
                                        if isinstance(item, ToolMessage)
                                        else {}
                                    ),
                                }
                                for item in messages[2:]
                            ],
                        ],
                        "tools": declarations,
                    }
                )
                try:
                    response = await bound_model.ainvoke(
                        messages, config={"callbacks": callbacks} if callbacks else None
                    )
                except Exception as error:
                    await record(
                        {
                            "event": "llm_failed",
                            "purpose": "tool_selection",
                            "action": action,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    raise
                await record(
                    {
                        "event": "model_response",
                        "purpose": "tool_selection",
                        "action": action,
                        "content": message_text(response.content),
                        "tool_calls": response.tool_calls,
                        "usage": getattr(response, "usage_metadata", None),
                    }
                )
                messages.append(response)
                calls = response.tool_calls or []
                if len(calls) > 1:
                    await record(
                        {
                            "event": "tool_validation",
                            "accepted": False,
                            "reason": "multiple_calls_in_one_response",
                        }
                    )
                    raise ValueError("Model proposed multiple tools at once; call rejected")
                if not calls:
                    break
                call = calls[0]
                if call["name"] not in by_name:
                    raise ValueError("Model selected a function outside the shortlist")
                card = by_name[call["name"]]
                try:
                    outcome = await dispatcher.execute(
                        call["name"],
                        call.get("args") or {},
                        cards,
                        request_text=request,
                        verified_results=results,
                    )
                except ValidationError as error:
                    path = ".".join(str(part) for part in error.absolute_path) or "arguments"
                    validation_result = {
                        "state": "invalid_arguments",
                        "error": f"Invalid {path}: {error.message}",
                    }
                    await record(
                        {
                            "event": "tool_validation",
                            "accepted": False,
                            "tool_id": card.tool_id,
                            "path": path,
                            "reason": error.message,
                        }
                    )
                    messages.append(
                        ToolMessage(
                            content=json.dumps(validation_result, ensure_ascii=False),
                            tool_call_id=call["id"],
                            status="error",
                        )
                    )
                    continue
                except Exception as error:
                    await record(
                        {
                            "event": "tool_validation_or_execution_failed",
                            "tool_id": card.tool_id,
                            "arguments": call.get("args") or {},
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    raise
                if outcome.state != "succeeded":
                    result = {
                        "state": outcome.state,
                        "error_code": outcome.error_code,
                        "data": outcome.data,
                    }
                    action_results.append(result)
                else:
                    result = {"state": outcome.state, "data": outcome.data}
                    action_results.append(outcome.data)
                messages.append(
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False, default=str),
                        tool_call_id=call["id"],
                    )
                )
                if stop_after_result(card, outcome.state):
                    results[f"action_{action_index + 1}"] = result
                    raise RuntimeError(
                        "The tool outcome is pending, uncertain, or failed. Inspect the saved exchange before retrying."
                    )
            if dispatcher.tool_call_count >= max_calls:
                tool_limit_reached = True
            results[f"action_{action_index + 1}"] = action_results
            if dispatcher.tool_call_count >= max_calls:
                break

        evidence = {
            "original_question": request,
            "recent_conversation": conversation_history or [],
            "user_goal": plan.user_goal,
            "verified_results": results,
            "tool_call_limit_reached": tool_limit_reached,
        }
        if plan.tool_actions:
            grounding = (
                "Use only verified_results as evidence. State clearly when requested data is "
                "missing; do not invent facts or recalculate values."
            )
        else:
            grounding = (
                "Answer ordinary conversational requests naturally. If the request needs "
                "private or external facts and no tool was used, say that verified information "
                "is unavailable instead of guessing."
            )
        final_system = (
            "Write a concise, well-formatted answer that directly answers original_question. "
            "Use recent_conversation to resolve follow-up references. Preserve the requested "
            "fields, order, and limits. "
            + grounding
            + " If tool_call_limit_reached is true, say the result may be incomplete. "
            + UNTRUSTED_DATA_POLICY
            + " Return the answer as readable Markdown, not a JSON contract or a copy of "
            "the input evidence. Select only the records and fields needed to answer the "
            "original question; when one record is requested, show one verified record."
        )
        final_input = json.dumps(evidence, ensure_ascii=False, default=str)
        await record(
            {
                "event": "llm_started",
                "purpose": "respond",
                "system_instruction": final_system,
                "prompt": final_input,
            }
        )
        try:
            final = await model.ainvoke(
                [
                    SystemMessage(final_system),
                    HumanMessage(final_input),
                ],
                config={"callbacks": callbacks} if callbacks else None,
            )
        except Exception as error:
            await record(
                {
                    "event": "llm_failed",
                    "purpose": "respond",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        summary = message_text(final.content).strip()
        if not summary:
            error = "The model returned an empty final answer. Tool results may already exist; inspect them before retrying."
            await record(
                {"event": "llm_failed", "purpose": "respond", "error_type": "EmptyFinalResponse", "error": error}
            )
            raise ValueError(error)
        final_event = {
            "event": "final_response",
            "response": summary,
            "usage": getattr(final, "usage_metadata", None),
        }
        # Show the answer before event/history persistence or resource cleanup can delay it.
        if progress:
            progress(final_event)
        await record(final_event)
        return {"summary": summary, "goal_plan": goal, "results": results, "events": events}
    finally:
        if catalog is not None:
            await catalog.close()
        if database is not None:
            await database.close()
        if paypal:
            await paypal.aclose()


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")
