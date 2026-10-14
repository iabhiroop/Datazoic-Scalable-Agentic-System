"""Generic sequential read nodes; identity comes from the persisted run."""

import json
from typing import Any
from uuid import UUID, uuid4

from fastapi.encoders import jsonable_encoder
from langgraph.types import interrupt

from app.agent.gateway import LLMError
from app.agent.debug import without_credentials
from app.agent.planning import resolve_arguments, validate_path_argument_provenance
from app.agent.state import (
    AgentState,
    FunctionCall,
    GoalPlan,
    Intent,
    LiteralBinding,
    Plan,
    PlanStep,
    ResumeEnvelope,
    ToolDeclaration,
    WaitRequest,
)
from app.db.models import WorkflowEvent, WorkflowRun
from app.observability import SecretRedactor
from app.security import Principal
from app.tools.contracts import ToolCard, ToolContext


class AgentNodes:
    def __init__(
        self,
        *,
        database,
        authenticator,
        model,
        retrieval,
        context_builder,
        registry,
        validator,
        execution,
        settings,
        ownership_check=None,
        telemetry=None,
    ):
        self.database, self.authenticator, self.model = database, authenticator, model
        self.retrieval, self.context_builder, self.registry = retrieval, context_builder, registry
        self.validator, self.execution, self.settings = validator, execution, settings
        self.ownership_check = ownership_check
        self.telemetry = telemetry

    async def classify_v3(self, state: AgentState, principal: Principal) -> AgentState:
        plan = await self.model.classify_goal(state["request_text"], state.get("clarifications", []))
        if len(plan.tool_actions) > self.settings.max_plan_steps:
            raise LLMError("invalid_goal_plan", "The request exceeds the supported action limit")
        await self.record_debug_event(
            state, principal, "debug_goal_plan",
            {"user_goal": plan.user_goal, "tool_actions": plan.tool_actions,
             "missing_information": plan.missing_information},
        )
        return {
            "goal_plan": plan.model_dump(mode="json"),
            "goal_action_index": 0,
            "goal_action_results": {},
            "routing_decision": "prepare_wait" if plan.missing_information else "retrieve",
        }

    async def retrieve_v3(self, state: AgentState, principal: Principal) -> AgentState:
        plan = GoalPlan.model_validate(state["goal_plan"])
        index = state.get("goal_action_index", 0)
        if index >= len(plan.tool_actions):
            return {"business_status": "completed", "routing_decision": "respond"}
        if state.get("tool_call_count", 0) >= self.settings.max_plan_steps:
            return {"observation": {"error_code": "tool_call_limit_reached"},
                    "business_status": "failed", "routing_decision": "respond"}
        action = plan.tool_actions[index]
        query = f"{plan.user_goal} {action}"
        await self.record_debug_event(state, principal, "debug_tool_retrieval",
                                      {"action_index": index, "query": query})
        cards = await self.retrieval.tools(
            query, [], principal, self.settings.candidate_tools_per_subgoal
        )
        await self.record_debug_event(
            state, principal, "debug_tool_shortlist",
            {"action_index": index, "user_goal": plan.user_goal, "action": action,
             "query": query, "tools": [{"tool_id": card.tool_id, "schema_version": card.schema_version,
                 "name": card.name, "description": card.description} for card in cards]},
        )
        if not cards:
            return {"observation": {"error_code": "no_authorized_tools", "action": action},
                    "business_status": "failed", "routing_decision": "respond"}
        return {"candidate_tools": [card.model_dump(mode="json") for card in cards],
                "routing_decision": "select"}

    async def select_v3(self, state: AgentState, principal: Principal) -> AgentState:
        plan = GoalPlan.model_validate(state["goal_plan"])
        index = state.get("goal_action_index", 0)
        action = plan.tool_actions[index]
        if state.get("tool_call_count", 0) >= self.settings.max_plan_steps:
            return {"observation": {"error_code": "tool_call_limit_reached"},
                    "business_status": "failed", "routing_decision": "respond"}
        cards = [ToolCard.model_validate(item) for item in state.get("candidate_tools", [])]
        declarations = [ToolDeclaration(name=card.tool_id, description=card.description,
                                        parameters=card.input_schema) for card in cards]
        shortlist = [{"name": card.name, "description": card.description} for card in cards]
        prior_results = state.get("goal_action_results", {})
        try:
            turn = await self.model.select_tool(plan.model_dump(mode="json"), action, shortlist,
                                                declarations, prior_results)
        except LLMError as error:
            return {"observation": {"error_code": error.code, "error_detail": error.detail},
                    "business_status": "failed", "routing_decision": "respond"}
        await self.record_debug_event(
            state, principal, "debug_tool_selection",
            {"action_index": index, "function_calls": [call.model_dump(mode="json") for call in turn.calls],
             "model_response": turn.text},
        )
        if len(turn.calls) != 1:
            await self.record_debug_event(state, principal, "debug_tool_validation",
                {"accepted": False, "reason": "expected_one_function_call"})
            return {"observation": {"error_code": "function_call_not_proposed" if not turn.calls else "multiple_function_calls_rejected"},
                    "business_status": "failed", "routing_decision": "respond"}
        call = turn.calls[0]
        card = next((candidate for candidate in cards if candidate.tool_id == call.name), None)
        if card is None or not isinstance(call.arguments, dict) or not call.call_id:
            await self.record_debug_event(state, principal, "debug_tool_validation",
                {"accepted": False, "reason": "unselected_or_invalid_function", "function_call": call.model_dump(mode="json")})
            return {"observation": {"error_code": "unselected_or_invalid_function"},
                    "business_status": "failed", "routing_decision": "respond"}
        step_id = f"action-{index + 1}-{card.tool_id}"
        step = PlanStep(step_id=step_id, tool_id=card.tool_id, schema_version=card.schema_version,
                        arguments={name: LiteralBinding(literal=value) for name, value in call.arguments.items()})
        plan_value = Plan(plan_version=1, steps=[step])
        try:
            await self.validator.validate(plan_value, principal)
            args = resolve_arguments(step, card, state.get("results", {}))
            validate_path_argument_provenance(card, args, state["request_text"],
                                               state.get("results", {}), state.get("clarifications", []))
        except (ValueError, TypeError) as error:
            await self.record_debug_event(state, principal, "debug_tool_validation",
                {"tool_id": card.tool_id, "arguments": call.arguments, "accepted": False, "error_detail": str(error)})
            return {"observation": {"error_code": "invalid_function_arguments", "reason_code": self.plan_validation_reason(error)},
                    "business_status": "failed", "routing_decision": "respond"}
        await self.record_debug_event(state, principal, "debug_tool_validation",
            {"tool_id": card.tool_id, "arguments": call.arguments, "accepted": True})
        return {"plan": plan_value.model_dump(mode="json"), "step_index": 0,
                "pending_function_call": call.model_dump(mode="json"), "tool_call_count": state.get("tool_call_count", 0) + 1,
                "routing_decision": "prepare_step"}

    async def validate_v3(self, state: AgentState, principal: Principal) -> AgentState:
        try:
            plan = Plan.model_validate(state["plan"])
            available = {(item["tool_id"], item["schema_version"]) for item in state.get("candidate_tools", [])}
            if len(plan.steps) != 1 or any((step.tool_id, step.schema_version) not in available for step in plan.steps):
                raise ValueError("Selected tool is outside the current shortlist")
            await self.validator.validate(plan, principal)
            step = plan.steps[0]
            card = await self.registry.get(step.tool_id, step.schema_version, principal)
            args = resolve_arguments(step, card, state.get("results", {}))
            validate_path_argument_provenance(card, args, state["request_text"],
                                               state.get("results", {}), state.get("clarifications", []))
        except (ValueError, TypeError) as error:
            await self.record_debug_event(state, principal, "debug_tool_validation",
                {"stage": "pre_dispatch", "accepted": False, "error_detail": str(error)})
            return {"observation": {"error_code": "unselected_or_invalid_function"}, "business_status": "failed", "routing_decision": "respond"}
        await self.record_debug_event(state, principal, "debug_tool_validation",
            {"stage": "pre_dispatch", "accepted": True})
        return {"routing_decision": "prepare_step"}

    async def prepare_step_v3(self, state: AgentState, principal: Principal) -> AgentState:
        result = await self.prepare_step_v2(state, principal)
        if result.get("routing_decision") == "wait":
            request = WaitRequest.model_validate(result["wait_request"])
            request.resume_route = "execute"
            result["wait_request"] = request.model_dump(mode="json")
        return result

    async def verify_v3(self, state: AgentState, principal: Principal) -> AgentState:
        observed = state["observation"]
        if observed.get("state") != "succeeded":
            step = Plan.model_validate(state["plan"]).steps[0]
            card = await self.registry.get(step.tool_id, step.schema_version, principal)
            return {"business_status": "needs_review" if card.operation_type == "write" and observed.get("state") in {"unknown", "pending"} else "failed",
                    "routing_decision": "respond"}
        result_key = f"action-{state.get('goal_action_index', 0) + 1}"
        data = observed["data"]
        step = Plan.model_validate(state["plan"]).steps[0]
        from app.db.models import WorkflowEvent

        async with self.database.session() as session, session.begin():
            session.add(WorkflowEvent(
                tenant_id=principal.tenant_id, run_id=UUID(state["run_id"]),
                event_type="tool_verified",
                evidence={"step_id": step.step_id, "tool_id": step.tool_id,
                          "schema_version": step.schema_version,
                          "source_ids": data.get("source_ids", [])},
            ))
        return {"results": {**state.get("results", {}), result_key: data},
                "goal_action_results": {**state.get("goal_action_results", {}), result_key: data},
                "goal_action_index": state.get("goal_action_index", 0) + 1,
                "candidate_tools": [], "pending_function_call": {},
                "provider_continuation": [],
                "business_status": "completed" if state.get("goal_action_index", 0) + 1 >= len(GoalPlan.model_validate(state["goal_plan"]).tool_actions) else state.get("business_status"),
                "routing_decision": "respond" if state.get("goal_action_index", 0) + 1 >= len(GoalPlan.model_validate(state["goal_plan"]).tool_actions) else "retrieve"}

    async def prepare_wait_v3(self, state: AgentState, principal: Principal) -> AgentState:
        plan = GoalPlan.model_validate(state["goal_plan"])
        request = WaitRequest(wait_token=str(uuid4()), kind="clarification",
            reason="Clarification required: " + ", ".join(plan.missing_information),
            resume_route="classify_v3", expected_payload_schema={"type":"object", "properties":{"answer":{"type":"string","minLength":1}}, "required":["answer"], "additionalProperties":False})
        return {"wait_request": request.model_dump(mode="json"), "routing_decision": "wait"}

    async def wait_v3(self, state: AgentState, principal: Principal) -> AgentState:
        result = await self.wait(state, principal)
        if result.get("routing_decision") == "classify":
            result["routing_decision"] = "classify"
        return result

    async def respond_v3(self, state: AgentState, principal: Principal) -> AgentState:
        status = state.get("business_status", "failed")
        results = state.get("results", {})
        sources = sorted({source for value in results.values() for source in value.get("source_ids", [])})
        if status != "completed":
            response = {"business_status": status, "summary": self.failure_summary(state),
                        "source_ids": sources, "results": results,
                        "error_code": state.get("observation", {}).get("error_code")}
            return {"final_result": without_credentials(response, self.settings)}
        evidence = {"business_status": status, "user_goal": state.get("goal_plan", {}).get("user_goal"),
                    "results": results, "source_ids": sources}
        try:
            response = await self.model.respond_goal(evidence)
        except LLMError:
            response = {"business_status": status, "summary": "The requested actions completed. Summary unavailable.", "source_ids": sources}
        response["results"] = results
        return {"final_result": SecretRedactor(self.settings)(response)}

    @staticmethod
    def failure_summary(state: AgentState) -> str:
        code = state.get("observation", {}).get("error_code")
        if code == "no_authorized_tools":
            return "No suitable authorized tool was found for this action."
        return "The request could not be completed."

    async def record_debug_event(
        self, state: AgentState, principal: Principal, event_type: str, evidence: dict[str, Any]
    ) -> None:
        async with self.database.session() as session, session.begin():
            session.add(
                WorkflowEvent(
                    tenant_id=principal.tenant_id,
                    run_id=UUID(state["run_id"]),
                    event_type=event_type,
                    evidence=without_credentials(jsonable_encoder(evidence), self.settings),
                )
            )

    async def principal(self, state, config):
        if self.ownership_check:
            await self.ownership_check()
        trusted = config["configurable"]
        if state["run_id"] != trusted["run_id"]:
            raise ValueError("Checkpoint run identity mismatch")
        async with self.database.session() as session:
            run = await session.get(WorkflowRun, UUID(trusted["run_id"]))
            if run is None or run.worker_id != trusted["worker_id"]:
                raise ValueError("Worker no longer owns this run")
            principal = Principal(run.tenant_id, run.user_id, frozenset())
        principal = await self.authenticator.refresh(principal)
        if "runs:write" not in principal.scopes:
            from fastapi import HTTPException

            raise HTTPException(403, "Run execution permission has been revoked")
        for step in Plan.model_validate(state["plan"]).steps if state.get("plan") else []:
            if step.step_id in state.get("results", {}):
                card = await self.registry.get(step.tool_id, step.schema_version, principal)
                arguments = resolve_arguments(step, card, state["results"])
                await self.registry.adapters[card.adapter_key].authorize_result(
                    arguments,
                    state["results"][step.step_id],
                    ToolContext(
                        principal.tenant_id,
                        principal.user_id,
                        run.run_id,
                        uuid4(),
                        None,
                        run.conversation_id,
                    ),
                )
        return principal

    async def classify(self, state: AgentState, principal: Principal) -> AgentState:
        from app.db.repositories import CatalogRepository

        async with self.database.session() as session:
            cards = await CatalogRepository(
                session, allowed_adapters=set(self.registry.adapters)
            ).eligible_tools(principal)
        intent = await self.model.classify(
            state["request_text"],
            {
                "category_ids": sorted({c for card in cards for c in card["category_ids"]}),
                "clarifications": state.get("clarifications", []),
            },
        )
        if not intent.subgoals:
            from app.agent.state import Subgoal

            intent.subgoals = [Subgoal(id="s1", action="complete", entity=intent.goal)]
        identities = [goal.id for goal in intent.subgoals]
        if any(not identity.strip() for identity in identities) or len(set(identities)) != len(
            identities
        ):
            raise LLMError("invalid_intent", "The model returned invalid subgoal identities")
        return {
            "intent": intent.model_dump(),
            "routing_decision": "prepare_wait" if intent.missing_fields else "retrieve",
        }

    async def cancelled(self, state):
        async with self.database.session() as session:
            run = await session.get(WorkflowRun, UUID(state["run_id"]))
            return run.cancel_requested

    async def retrieve(self, state: AgentState, principal: Principal) -> AgentState:
        if state.get("graph_version") in {"v2", "v3"}:
            return await self.retrieve_v2(state, principal)
        intent = Intent.model_validate(state["intent"])
        if len(intent.subgoals) > self.settings.max_plan_steps:
            raise ValueError("Goal exceeds the configured sequential planning limit")
        queries = [f"{intent.goal} {goal.action} {goal.entity}" for goal in intent.subgoals] or [
            intent.goal
        ]
        candidates = {}
        for query in queries:
            for card in await self.retrieval.tools(
                query, intent.category_ids, principal, self.settings.candidate_tools_per_subgoal
            ):
                candidates.setdefault((card.tool_id, card.schema_version), card)
        cards = list(candidates.values())
        return {
            "candidate_tools": [c.model_dump(mode="json") for c in cards],
            "routing_decision": "plan",
        }

    async def retrieve_v2(self, state: AgentState, principal: Principal) -> AgentState:
        intent = Intent.model_validate(state["intent"])
        if len(intent.subgoals) > self.settings.max_plan_steps:
            raise ValueError("Goal exceeds the configured sequential planning limit")
        subgoal_tools: dict[str, list[dict[str, Any]]] = {}
        for subgoal in intent.subgoals:
            query = f"{intent.goal} {subgoal.action} {subgoal.entity}"
            cards = await self.retrieval.tools(
                query, intent.category_ids, principal, self.settings.candidate_tools_per_subgoal
            )
            subgoal_tools[subgoal.id] = [card.model_dump(mode="json") for card in cards]
            await self.record_debug_event(
                state,
                principal,
                "debug_tool_shortlist",
                {
                    "subgoal": subgoal.model_dump(mode="json"),
                    "query": query,
                    "tools": [
                        {
                            "tool_id": card.tool_id,
                            "schema_version": card.schema_version,
                            "name": card.name,
                            "description": card.description,
                        }
                        for card in cards
                    ],
                },
            )
        return {
            "subgoal_tools": subgoal_tools,
            "subgoal_index": 0,
            "routing_decision": "plan",
        }

    async def plan_v2(self, state: AgentState, principal: Principal) -> AgentState:
        intent = Intent.model_validate(state["intent"])
        suggestions = [
            {
                "step_id": goal.id,
                "action": goal.action,
                "entity": goal.entity,
                "tools": [
                    {
                        "tool_id": card["tool_id"],
                        "name": card["name"],
                        "description": card["description"],
                    }
                    for card in state.get("subgoal_tools", {}).get(goal.id, [])
                ],
            }
            for goal in intent.subgoals
        ]
        if intent.execution_mode == "plan_only":
            return {
                "final_result": {
                    "business_status": "completed",
                    "execution_mode": "plan_only",
                    "summary": "Here is a suggested plan. No tools were executed.",
                    "steps": suggestions,
                    "results": {},
                    "source_ids": [],
                },
                "business_status": "completed",
                "routing_decision": "plan_only_done",
            }
        index = state.get("subgoal_index", 0)
        if index >= len(intent.subgoals):
            return {"business_status": "completed", "routing_decision": "respond"}
        if state.get("tool_call_count", 0) >= self.settings.max_plan_steps:
            return {
                "observation": {"error_code": "tool_call_limit_reached"},
                "business_status": "failed",
                "routing_decision": "respond",
            }
        goal = intent.subgoals[index]
        candidates = [
            ToolCard.model_validate(item)
            for item in state.get("subgoal_tools", {}).get(goal.id, [])
        ]
        if not candidates:
            return {
                "observation": {"error_code": "no_authorized_tools"},
                "business_status": "failed",
                "routing_decision": "respond",
            }
        declarations = [
            ToolDeclaration(
                name=card.tool_id,
                description=(
                    f"{card.name}. Action: {card.action}. "
                    f"Operation: {card.operation_type}. {card.description}"
                ),
                parameters=card.input_schema,
            )
            for card in candidates
        ]
        prompt = json.dumps(
            {
                "request": state["request_text"],
                "subgoal": goal.model_dump(),
                "completed_results": state.get("results", {}),
            },
            ensure_ascii=False,
        )
        try:
            turn = await self.model.select_function(
                prompt,
                declarations,
                state.get("provider_continuation", []),
                force_tool_call=(
                    not state.get("provider_continuation")
                    or bool(state.get("selection_retries", 0))
                ),
            )
        except LLMError as error:
            return {
                "observation": {"error_code": error.code, "error_detail": error.detail},
                "business_status": "failed",
                "routing_decision": "respond",
            }
        await self.record_debug_event(
            state,
            principal,
            "debug_tool_selection",
            {
                "subgoal_id": goal.id,
                "function_calls": [call.model_dump(mode="json") for call in turn.calls],
                "model_response": turn.text,
            },
        )
        if len(turn.calls) > 1:
            await self.record_debug_event(
                state, principal, "debug_tool_validation",
                {"accepted": False, "reason": "multiple_function_calls_rejected"},
            )
            return {
                "observation": {"error_code": "multiple_function_calls_rejected"},
                "business_status": "failed",
                "routing_decision": "respond",
            }
        if not turn.calls:
            # A tool-result continuation with no further call completes this subgoal.
            if not state.get("provider_continuation") or state.get("selection_retries", 0):
                await self.record_debug_event(
                    state, principal, "debug_tool_validation",
                    {"accepted": False, "reason": "function_call_not_proposed"},
                )
                return {
                    "observation": {"error_code": "function_call_not_proposed"},
                    "business_status": "failed",
                    "routing_decision": "respond",
                }
            return {
                "subgoal_index": index + 1,
                "provider_continuation": [],
                "selection_retries": 0,
                "pending_function_call": {},
                "subgoal_summaries": {
                    **state.get("subgoal_summaries", {}),
                    **({goal.id: turn.text} if turn.text.strip() else {}),
                },
                "routing_decision": "plan",
            }
        call = turn.calls[0]
        card = next((candidate for candidate in candidates if candidate.tool_id == call.name), None)
        if card is None or not isinstance(call.arguments, dict) or not call.call_id:
            await self.record_debug_event(
                state, principal, "debug_tool_validation",
                {
                    "accepted": False,
                    "reason": "unselected_or_invalid_function",
                    "function_call": call.model_dump(mode="json"),
                },
            )
            return {
                "observation": {"error_code": "unselected_or_invalid_function"},
                "business_status": "failed",
                "routing_decision": "respond",
            }
        # Repeated write proposals reuse one operation identity. Reads get distinct
        # result slots so a prior observation cannot be mistaken for the new call.
        step_id = (
            f"{goal.id}-{card.tool_id}"
            if card.operation_type == "write"
            else f"{goal.id}-{card.tool_id}-{state.get('tool_call_count', 0) + 1}"
        )
        step = PlanStep(
            step_id=step_id,
            tool_id=card.tool_id,
            schema_version=card.schema_version,
            arguments={
                name: LiteralBinding(literal=value) for name, value in call.arguments.items()
            },
            action_ref=None,
        )
        plan = Plan(plan_version=state.get("plan", {}).get("plan_version", 0) + 1, steps=[step])
        try:
            await self.validator.validate(plan, principal)
            arguments = resolve_arguments(step, card, state.get("results", {}))
            validate_path_argument_provenance(
                card, arguments, state["request_text"],
                state.get("results", {}), state.get("clarifications", []),
            )
        except ValueError as error:
            await self.record_debug_event(
                state, principal, "debug_tool_validation",
                {"tool_id": card.tool_id, "arguments": call.arguments, "accepted": False, "error_detail": str(error)},
            )
            if state.get("selection_retries", 0) < 1:
                rejection = {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.call_id,
                    "result": [{"type": "text", "text": json.dumps({
                        "error": "The proposed tool call was rejected before execution.",
                        "reason": str(error),
                        "instruction": "Choose a suitable tool from the same shortlist with real arguments. Do not invent resource IDs.",
                    })}],
                }
                return await self.plan_v2(
                    {
                        **state,
                        "provider_continuation": [*turn.continuation, rejection],
                        "selection_retries": state.get("selection_retries", 0) + 1,
                    },
                    principal,
                )
            return {
                "observation": {
                    "error_code": "invalid_function_arguments",
                    "reason_code": self.plan_validation_reason(error),
                },
                "business_status": "failed",
                "routing_decision": "respond",
            }
        await self.record_debug_event(
            state, principal, "debug_tool_validation",
            {"tool_id": card.tool_id, "arguments": call.arguments, "accepted": True},
        )
        return {
            "plan": plan.model_dump(),
            "results": {
                key: value for key, value in state.get("results", {}).items() if key != step_id
            },
            "candidate_tools": [card.model_dump(mode="json") for card in candidates],
            "pending_function_call": call.model_dump(mode="json"),
            "provider_continuation": turn.continuation,
            "selection_retries": 0,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "step_index": 0,
            "routing_decision": "prepare_step",
        }

    async def validate_v2(self, state: AgentState, principal: Principal) -> AgentState:
        try:
            plan = Plan.model_validate(state["plan"])
            available = {(c["tool_id"], c["schema_version"]) for c in state["candidate_tools"]}
            if len(plan.steps) != 1 or any(
                (s.tool_id, s.schema_version) not in available for s in plan.steps
            ):
                raise ValueError("Function call selected outside current shortlist")
            await self.validator.validate(plan, principal)
            step = plan.steps[0]
            card = await self.registry.get(step.tool_id, step.schema_version, principal)
            arguments = resolve_arguments(step, card, state.get("results", {}))
            validate_path_argument_provenance(
                card, arguments, state["request_text"],
                state.get("results", {}), state.get("clarifications", []),
            )
        except ValueError as error:
            await self.record_debug_event(
                state, principal, "debug_tool_validation",
                {"stage": "pre_dispatch", "accepted": False, "error_detail": str(error)},
            )
            return {
                "observation": {
                    "error_code": "invalid_function_arguments",
                    "reason_code": self.plan_validation_reason(error),
                },
                "business_status": "failed",
                "routing_decision": "respond",
            }
        await self.record_debug_event(
            state, principal, "debug_tool_validation",
            {"stage": "pre_dispatch", "accepted": True},
        )
        return {"routing_decision": "prepare_step"}

    async def prepare_step_v2(self, state: AgentState, principal: Principal) -> AgentState:
        async with self.database.session() as session:
            run = await session.get(WorkflowRun, UUID(state["run_id"]))
            cancelled = run.cancel_requested
        if cancelled:
            return {"business_status": "cancelled", "routing_decision": "respond"}
        step = Plan.model_validate(state["plan"]).steps[0]
        card = await self.registry.get(step.tool_id, step.schema_version, principal)
        if card.operation_type == "write":
            try:
                operation = await self.execution.prepare_write(step, card, state, principal)
            except (ValueError, TypeError):
                return {
                    "observation": {"error_code": "write_precondition_failed"},
                    "business_status": "failed",
                    "routing_decision": "respond",
                }
            from app.db.models import OperationState

            if operation.status == OperationState.SUCCEEDED:
                return {"routing_decision": "execute"}
            if operation.status == OperationState.FAILED:
                return {
                    "observation": {"error_code": "write_failed"},
                    "business_status": "failed",
                    "routing_decision": "respond",
                }
            if operation.status != OperationState.PREPARED:
                return {
                    "observation": {"error_code": "write_requires_review"},
                    "business_status": "needs_review",
                    "routing_decision": "respond",
                }
            if state.get("approved_operation_id") != str(operation.operation_id):
                arguments = resolve_arguments(step, card, state.get("results", {}))
                request = WaitRequest(
                    wait_token=str(uuid4()),
                    kind="approval",
                    reason="Approve sandbox action: "
                    + card.name
                    + ". "
                    + self.approval_preview(arguments),
                    resume_route="execute",
                    expected_payload_schema={
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}},
                        "required": ["approved"],
                        "additionalProperties": False,
                    },
                    operation_id=str(operation.operation_id),
                    input_hash=operation.input_hash,
                )
                return {"wait_request": request.model_dump(), "routing_decision": "wait"}
        return {"routing_decision": "execute"}

    async def verify_v2(self, state: AgentState, principal: Principal) -> AgentState:
        observed = state["observation"]
        call = FunctionCall.model_validate(state["pending_function_call"])
        if observed.get("state") != "succeeded":
            step = Plan.model_validate(state["plan"]).steps[0]
            card = await self.registry.get(step.tool_id, step.schema_version, principal)
            return {
                "business_status": "needs_review"
                if card.operation_type == "write"
                and observed.get("state") in {"unknown", "pending"}
                else "failed",
                "routing_decision": "respond",
            }
        step = Plan.model_validate(state["plan"]).steps[0]
        from app.db.models import WorkflowEvent

        async with self.database.session() as session, session.begin():
            session.add(
                WorkflowEvent(
                    tenant_id=principal.tenant_id,
                    run_id=UUID(state["run_id"]),
                    event_type="tool_verified",
                    evidence={
                        "step_id": step.step_id,
                        "tool_id": step.tool_id,
                        "schema_version": step.schema_version,
                        "source_ids": observed["data"].get("source_ids", []),
                    },
                )
            )
        history = list(state.get("provider_continuation", []))
        history.append(
            {
                "type": "function_result",
                "name": call.name,
                "call_id": call.call_id,
                "result": [
                    {"type": "text", "text": json.dumps(observed["data"], ensure_ascii=False)}
                ],
            }
        )
        return {
            "results": {**state.get("results", {}), step.step_id: observed["data"]},
            "provider_continuation": history,
            "pending_function_call": {},
            "routing_decision": "plan",
        }

    async def plan(self, state: AgentState, principal: Principal) -> AgentState:
        from app.tools.contracts import ToolCard

        cards = [ToolCard.model_validate(c) for c in state.get("candidate_tools", [])]
        if not cards:
            return {
                "observation": {"error_code": "no_authorized_tools"},
                "routing_decision": "recover",
            }
        context = await self.context_builder.build(cards, [], [])
        context["failure"] = state.get("observation", {})
        proposed = await self.model.plan(Intent.model_validate(state["intent"]), context)
        proposed.plan_version = state.get("plan", {}).get("plan_version", 0) + 1
        return {
            "plan": proposed.model_dump(),
            "candidate_tools": [card.model_dump(mode="json") for card in cards],
            "step_index": 0,
            "results": {},
            "routing_decision": "validate",
        }

    async def validate(self, state: AgentState, principal: Principal) -> AgentState:
        try:
            available = {
                (card["tool_id"], card["schema_version"]) for card in state["candidate_tools"]
            }
            if any(
                (step.tool_id, step.schema_version) not in available
                for step in Plan.model_validate(state["plan"]).steps
            ):
                raise ValueError("Plan selected a tool outside supplied candidates")
            await self.validator.validate(Plan.model_validate(state["plan"]), principal)
        except ValueError as error:
            return {
                "observation": {
                    "error_code": "invalid_plan",
                    "reason_code": self.plan_validation_reason(error),
                },
                "routing_decision": "recover",
            }
        return {"routing_decision": "prepare_step"}

    @staticmethod
    def plan_validation_reason(error: ValueError) -> str:
        """Map validator errors to safe, stable feedback codes; never expose input values."""
        return {
            "Resolved tool input does not match its schema": "tool_input_schema_mismatch",
            "Output path must be a JSON Pointer": "invalid_json_pointer",
            "Output binding must name an earlier declared dependency": "invalid_output_dependency",
        }.get(str(error), "plan_contract_validation_failed")

    async def prepare_step(self, state: AgentState, principal: Principal) -> AgentState:
        async with self.database.session() as session:
            run = await session.get(WorkflowRun, UUID(state["run_id"]))
            cancelled = run.cancel_requested
        if cancelled:
            return {"business_status": "cancelled", "routing_decision": "respond"}
        if state.get("step_index", 0) >= len(state["plan"]["steps"]):
            return {"business_status": "completed", "routing_decision": "respond"}
        step = Plan.model_validate(state["plan"]).steps[state.get("step_index", 0)]
        card = await self.registry.get(step.tool_id, step.schema_version, principal)
        if card.operation_type == "write":
            try:
                operation = await self.execution.prepare_write(step, card, state, principal)
            except (ValueError, TypeError):
                return {
                    "observation": {"error_code": "write_precondition_failed"},
                    "business_status": "failed",
                    "routing_decision": "respond",
                }
            from app.db.models import OperationState

            if operation.status == OperationState.SUCCEEDED:
                return {"routing_decision": "execute"}
            if operation.status == OperationState.FAILED:
                return {
                    "observation": {"error_code": "write_failed"},
                    "business_status": "failed",
                    "routing_decision": "respond",
                }
            if operation.status != OperationState.PREPARED:
                return {
                    "observation": {"error_code": "write_requires_review"},
                    "business_status": "needs_review",
                    "routing_decision": "respond",
                }
            if state.get("approved_operation_id") != str(operation.operation_id):
                arguments = resolve_arguments(step, card, state.get("results", {}))
                request = WaitRequest(
                    wait_token=str(uuid4()),
                    kind="approval",
                    reason="Approve sandbox action: "
                    + card.name
                    + ". "
                    + self.approval_preview(arguments),
                    resume_route="execute",
                    expected_payload_schema={
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}},
                        "required": ["approved"],
                        "additionalProperties": False,
                    },
                    operation_id=str(operation.operation_id),
                    input_hash=operation.input_hash,
                )
                return {"wait_request": request.model_dump(), "routing_decision": "wait"}
        return {"routing_decision": "execute"}

    @staticmethod
    def approval_preview(arguments: dict) -> str:
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    async def execute(self, state: AgentState, principal: Principal) -> AgentState:
        step = Plan.model_validate(state["plan"]).steps[state.get("step_index", 0)]
        call = state.get("pending_function_call", {})
        await self.record_debug_event(
            state, principal, "debug_tool_dispatch",
            {
                "tool_id": step.tool_id,
                "schema_version": step.schema_version,
                "step_id": step.step_id,
                "arguments": call.get("arguments") or {
                    name: binding.model_dump(mode="json")
                    for name, binding in step.arguments.items()
                },
                "function_call": call,
            },
        )
        try:
            outcome = await self.execution.execute_step(step, state, principal)
        except (ValueError, LLMError) as error:
            result = {
                "observation": {
                    "error_code": "execution_failed",
                    "error_detail": getattr(error, "detail", None) or str(error),
                },
                "routing_decision": "recover",
            }
            await self.record_debug_event(
                state, principal, "debug_tool_result",
                {"tool_id": step.tool_id, "step_id": step.step_id, "observation": result["observation"]},
            )
            if state.get("graph_version") in {"v2", "v3"}:
                result["llm_activity"] = [
                    SecretRedactor(self.settings)(
                        {
                            "purpose": "tool_result",
                            "function_call": call,
                            "response": result["observation"],
                        }
                    )
                ]
            return result
        except Exception as error:  # noqa: BLE001 -- preserve worker failure handling
            await self.record_debug_event(
                state, principal, "debug_tool_result",
                {
                    "tool_id": step.tool_id,
                    "step_id": step.step_id,
                    "error_type": type(error).__name__,
                    "error_detail": str(error),
                },
            )
            raise
        result = {
            "observation": {
                "state": outcome.state,
                "data": outcome.data,
                "error_code": outcome.error_code,
            },
            "routing_decision": "verify",
        }
        await self.record_debug_event(
            state, principal, "debug_tool_result",
            {"tool_id": step.tool_id, "step_id": step.step_id, "observation": result["observation"]},
        )
        if state.get("graph_version") in {"v2", "v3"}:
            result["llm_activity"] = [
                SecretRedactor(self.settings)(
                    {
                        "purpose": "tool_result",
                        "function_call": call,
                        "response": result["observation"],
                    }
                )
            ]
        return result

    async def verify(self, state: AgentState, principal: Principal) -> AgentState:
        observed = state["observation"]
        if observed.get("error_code") == "cancelled":
            return {"business_status": "cancelled", "routing_decision": "respond"}
        if observed.get("state") != "succeeded":
            step = state["plan"]["steps"][state.get("step_index", 0)]
            card = await self.registry.get(step["tool_id"], step["schema_version"], principal)
            if card.operation_type == "write":
                return {
                    "business_status": "needs_review"
                    if observed.get("state") in {"unknown", "pending"}
                    else "failed",
                    "routing_decision": "respond",
                }
            return {"routing_decision": "recover"}
        step = state["plan"]["steps"][state.get("step_index", 0)]
        from app.db.models import WorkflowEvent

        async with self.database.session() as session, session.begin():
            session.add(
                WorkflowEvent(
                    tenant_id=principal.tenant_id,
                    run_id=UUID(state["run_id"]),
                    event_type="tool_verified",
                    evidence={
                        "step_id": step["step_id"],
                        "tool_id": step["tool_id"],
                        "schema_version": step["schema_version"],
                        "source_ids": observed["data"].get("source_ids", []),
                    },
                )
            )
        return {
            "results": {**state.get("results", {}), step["step_id"]: observed["data"]},
            "step_index": state.get("step_index", 0) + 1,
            "routing_decision": "prepare_step",
        }

    async def recover(self, state: AgentState, principal: Principal) -> AgentState:
        count = state.get("replan_count", 0)
        if (
            state.get("observation", {}).get("error_code") == "invalid_plan"
            and count < self.settings.max_replans
        ):
            return {"replan_count": count + 1, "routing_decision": "plan"}
        return {"business_status": "failed", "routing_decision": "respond"}

    async def prepare_wait(self, state: AgentState, principal: Principal) -> AgentState:
        intent = Intent.model_validate(state["intent"])
        request = WaitRequest(
            wait_token=str(uuid4()),
            kind="clarification",
            reason="Clarification required: " + ", ".join(intent.missing_fields),
            resume_route="classify",
            expected_payload_schema={
                "type": "object",
                "properties": {"answer": {"type": "string", "minLength": 1}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
        return {"wait_request": request.model_dump(), "routing_decision": "wait"}

    async def wait(self, state: AgentState, principal: Principal) -> AgentState:
        from jsonschema import Draft202012Validator

        request = WaitRequest.model_validate(state["wait_request"])
        resumed = ResumeEnvelope.model_validate(interrupt(request.model_dump()))
        if resumed.wait_token != request.wait_token or resumed.kind != request.kind:
            raise ValueError("Resume does not match active wait")
        Draft202012Validator(request.expected_payload_schema).validate(resumed.payload)
        if request.kind == "approval":
            if not resumed.payload["approved"]:
                return {
                    "last_input_id": resumed.input_id,
                    "business_status": "cancelled",
                    "routing_decision": "respond",
                    "wait_request": {},
                }
            return {
                "last_input_id": resumed.input_id,
                "approved_operation_id": request.operation_id,
                "routing_decision": "execute",
                "wait_request": {},
            }
        return {
            "clarifications": [*state.get("clarifications", []), resumed.payload],
            "last_input_id": resumed.input_id,
            "routing_decision": "classify_v3" if state.get("graph_version") == "v3" else "classify",
            "wait_request": {},
        }

    async def reconcile(self, state: AgentState, principal: Principal) -> AgentState:
        raise NotImplementedError("Write reconciliation is unavailable in the read-only graph")

    async def respond(self, state: AgentState, principal: Principal) -> AgentState:
        status = state.get("business_status", "failed")
        results = state.get("results", {})
        sources = sorted(
            {source for value in results.values() for source in value.get("source_ids", [])}
        )
        evidence = {
            "business_status": status,
            "results": results,
            "subgoal_summaries": state.get("subgoal_summaries", {}),
            "source_ids": sources,
            "error_code": state.get("observation", {}).get("error_code"),
            "failure_reason": state.get("observation", {}).get("reason_code"),
            "error_detail": state.get("observation", {}).get("error_detail"),
        }
        if status != "completed":
            code = evidence["error_code"]
            if status == "needs_review":
                summary = "A write outcome needs review before this request can be retried."
            elif status == "cancelled":
                summary = "Request cancelled."
            elif code in {"provider_busy", "provider_unavailable", "provider_connection_error"}:
                summary = "The model provider is temporarily unavailable. Please try again later."
            elif code == "provider_request_rejected":
                summary = "Gemini rejected the tool request. Check the run trace before retrying."
            else:
                summary = "The request could not be completed."
            response = {"business_status": status, "summary": summary, "source_ids": sources}
            response["results"] = results
            if code:
                response["error_code"] = code
            if evidence["failure_reason"]:
                response["failure_reason"] = evidence["failure_reason"]
            if evidence["error_detail"]:
                response["error_detail"] = evidence["error_detail"]
            return {"final_result": without_credentials(response, self.settings)}
        try:
            response = await self.model.respond(evidence)
        except LLMError:
            response = {
                "business_status": status,
                "summary": f"Workflow status: {status}. Summary unavailable.",
                "source_ids": sources,
            }
        response["results"] = results
        response["business_status"] = status
        if evidence["error_code"]:
            response["error_code"] = evidence["error_code"]
        if evidence["failure_reason"]:
            response["failure_reason"] = evidence["failure_reason"]
        return {"final_result": SecretRedactor(self.settings)(response)}
