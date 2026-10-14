"""Small Streamlit chat backed by the existing Postgres conversation tables."""

import asyncio
import logging
from uuid import UUID

import streamlit as st
from fastapi import HTTPException
from langfuse import get_client

from app.db.session import event_loop_factory
from app.observability import SecretRedactor
from app.settings import Settings
from appv2.agent import run_chat
from appv2.config import ROOT, local_principal
from appv2.persistence import ChatStore

LOGGER = logging.getLogger("appv2")


def new_loop():
    return event_loop_factory()


def run(coro):
    with asyncio.Runner(loop_factory=new_loop) as runner:
        return runner.run(coro)


async def load_conversations(settings, principal):
    store = ChatStore(settings)
    try:
        return await store.conversations(principal)
    finally:
        await store.close()


async def load_history(settings, principal, conversation_id):
    store = ChatStore(settings)
    try:
        return await store.history(principal, conversation_id)
    finally:
        await store.close()


async def process_turn(settings, principal, conversation_id, prompt, progress):
    store = ChatStore(settings)
    actual_conversation_id = None
    run_id = None
    try:
        recent_messages = await store.recent_messages(principal, conversation_id)
        actual_conversation_id, run_id = await store.start_turn(principal, conversation_id, prompt)

        async def record(event):
            log_terminal_event(settings, event)
            await store.log_event(principal, run_id, event)

        result = await run_chat(
            prompt,
            settings,
            record,
            run_id=run_id,
            conversation_id=actual_conversation_id,
            conversation_history=recent_messages,
            progress=progress,
        )
        await store.finish_turn(
            principal,
            actual_conversation_id,
            run_id,
            response=result["summary"],
            result=result,
        )
        return actual_conversation_id, result
    except Exception as error:
        if run_id is not None and actual_conversation_id is not None:
            detail = str(SecretRedactor(settings)(f"{type(error).__name__}: {error}"))
            await store.log_event(principal, run_id, {"event": "app_error", "detail": detail})
            await store.finish_turn(
                principal,
                actual_conversation_id,
                run_id,
                response=detail,
                error=detail,
            )
        raise
    finally:
        await store.close()


def log_terminal_event(settings: Settings, event: dict) -> None:
    """Print progress without dumping prompts, results, or credentials."""
    kind = event.get("event", "event")
    purpose = event.get("purpose")
    action = event.get("action")
    context = f" purpose={purpose}" if purpose else ""
    if action:
        context += f" action={action!r}"
    if kind == "llm_started":
        LOGGER.info("LLM call started:%s model=%s", context, event.get("model", "configured"))
    elif kind in {"llm_finished", "model_response"}:
        calls = event.get("tool_calls") or []
        LOGGER.info("LLM call finished:%s response_chars=%d selected=%s", context,
                    len(str(event.get("response", event.get("content", "")))),
                    [call.get("name", "unknown") for call in calls])
    elif kind == "tool_shortlist":
        LOGGER.info("Tool shortlist:%s %s", context,
                    [item.get("name", "unknown") for item in event.get("tools", [])])
    elif kind in {"tool_dispatch_started", "tool_dispatch_finished"}:
        LOGGER.info("%s tool_id=%s state=%s", kind, event.get("tool_id"), event.get("state"))
    elif kind in {"llm_failed", "tool_validation_or_execution_failed", "app_error"}:
        detail = SecretRedactor(settings)(str(event.get("error", event.get("detail", ""))))
        LOGGER.error("%s:%s %s", kind, context, detail)
    elif kind == "goal_plan":
        LOGGER.info("Goal plan ready: goal=%r actions=%s", event.get("user_goal"),
                    event.get("tool_actions"))
    elif kind == "final_response":
        LOGGER.info("Final response ready: chars=%d", len(str(event.get("response", ""))))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    st.set_page_config(page_title="Agent chat", page_icon="💬", layout="centered")
    st.title("Agent chat")
    st.caption("Find information or complete an action using your authorized tools.")

    try:
        settings = Settings(_env_file=ROOT / ".env")
        principal, _token = local_principal(settings)
        if not settings.gemini_api_key or not settings.embedding_model:
            raise ValueError("Gemini embedding settings are required for semantic search")
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required")
    except (ValueError, OSError, HTTPException) as error:
        st.error(f"Configuration issue: {getattr(error, 'detail', str(error))}")
        return

    try:
        conversations = run(load_conversations(settings, principal))
    except Exception as error:  # noqa: BLE001 -- present owner-visible startup failure
        st.error(f"PostgreSQL is unavailable: {SecretRedactor(settings)(str(error))}")
        return

    if "appv2_conversation_id" not in st.session_state:
        st.session_state.appv2_conversation_id = None
    with st.sidebar:
        st.subheader("Conversations")
        if st.button("＋ New chat", use_container_width=True):
            st.session_state.appv2_conversation_id = None
            st.rerun()
        choices = {str(row.conversation_id): row.title for row in conversations}
        selected = st.selectbox(
            "History",
            options=[""] + list(choices),
            format_func=lambda key: "New conversation" if not key else choices[key],
            index=(list(choices).index(str(st.session_state.appv2_conversation_id)) + 1
                   if st.session_state.appv2_conversation_id is not None
                   and str(st.session_state.appv2_conversation_id) in choices else 0),
            label_visibility="collapsed",
        )
        selected_id = UUID(selected) if selected else None
        if selected_id != st.session_state.appv2_conversation_id:
            st.session_state.appv2_conversation_id = selected_id
            st.rerun()

    messages = []
    if st.session_state.appv2_conversation_id is not None:
        try:
            messages = run(load_history(settings, principal, st.session_state.appv2_conversation_id))
        except Exception as error:  # noqa: BLE001 -- show scoped history failure
            st.error(f"Could not load conversation: {SecretRedactor(settings)(str(error))}")

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("goal_plan"):
                render_goal(message["goal_plan"])
            if message.get("results"):
                with st.expander("Tool results"):
                    st.json(message["results"])
            if message.get("events"):
                with st.expander("LLM and tool activity"):
                    st.json(message["events"])

    prompt = st.chat_input("Ask a question or request an action")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        output = st.empty()
        try:
            conversation_id, result = run(
                process_turn(
                    settings,
                    principal,
                    st.session_state.appv2_conversation_id,
                    prompt,
                    progress=lambda event: output.markdown(
                        f"**Goal:** {event['user_goal']}\n\n**Actions:** "
                        + "; ".join(event.get("tool_actions", []))
                    ) if event.get("event") == "goal_plan" else None,
                )
            )
            st.session_state.appv2_conversation_id = conversation_id
            output.markdown(result["summary"])
            if result.get("goal_plan"):
                render_goal(result["goal_plan"])
            if result.get("results"):
                with st.expander("Tool results", expanded=True):
                    st.json(result["results"])
        except Exception as error:  # noqa: BLE001 -- show owner-visible application error
            detail = str(SecretRedactor(settings)(f"{type(error).__name__}: {error}"))
            st.error(detail)
        finally:
            if settings.langfuse_tracing_enabled and settings.langfuse_public_key:
                try:
                    get_client().flush()
                except Exception as error:  # noqa: BLE001 -- telemetry must not mask chat result
                    LOGGER.warning("Langfuse flush failed (%s)", type(error).__name__)


def render_goal(plan: dict) -> None:
    st.markdown(f"**Goal:** {plan.get('user_goal', '')}")
    actions = plan.get("tool_actions") or []
    if actions:
        st.markdown("**Actions:** " + "; ".join(actions))
    missing = plan.get("missing_information") or []
    if missing:
        st.info("Need clarification: " + ", ".join(missing))


if __name__ == "__main__":
    main()
