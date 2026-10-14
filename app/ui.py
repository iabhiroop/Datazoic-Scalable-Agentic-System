"""Local Streamlit chat for the read-only agent workflow."""

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TOKEN_FILE = ROOT / "runtime" / "local-auth.json"
FINAL_STATUSES = {"completed", "failed", "needs_review", "cancelled"}
STATUS_LABELS = {
    "queued": "Queued",
    "running": "Working",
    "waiting_for_input": "Needs your input",
    "waiting_for_retry": "Waiting to retry",
    "waiting_for_reconciliation": "Checking outcome",
    "completed": "Complete",
    "failed": "Failed",
    "needs_review": "Needs review",
    "cancelled": "Cancelled",
}


class UIError(Exception):
    """A short, safe message for the local interface."""


def local_api_url() -> str:
    url = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")
    parts = urlsplit(url)
    if (
        parts.scheme != "http"
        or parts.hostname not in {"127.0.0.1", "localhost"}
        or parts.username
        or parts.password
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
    ):
        raise UIError("AGENT_API_URL must be a local HTTP address for this demo.")
    return url


def local_token() -> str:
    try:
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        token = data["bearer_token"]
        if not isinstance(token, str) or not token:
            raise ValueError("Missing bearer token")
        return token
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise UIError("Local demo identity is missing. Set up runtime/local-auth.json.") from error


class AgentAPI:
    def __init__(self, base_url: str, token: str, *, transport=None):
        self.client = httpx.Client(
            base_url=base_url,
            timeout=10,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
            headers={"Authorization": f"Bearer {token}"},
        )

    def close(self) -> None:
        self.client.close()

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.RequestError as error:
            raise UIError("Cannot reach the agent API. Start the API and try again.") from error
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = {}
            message = body.get("message") if isinstance(body, dict) else None
            if response.status_code == 401:
                message = (
                    "The local demo identity was rejected. Restart the API after configuring it."
                )
            raise UIError(message or f"Agent API returned HTTP {response.status_code}.")
        try:
            return response.json()
        except ValueError as error:
            raise UIError("Agent API returned an invalid response.") from error

    def conversations(self) -> list[dict[str, Any]]:
        return self.request("GET", "/conversations")

    def conversation(self, conversation_id: str) -> dict[str, Any]:
        return self.request("GET", f"/conversations/{conversation_id}")

    def debug_events(self, run_id: str, cursor: str | None = None) -> dict[str, Any]:
        return self.request(
            "GET", f"/runs/{run_id}/debug", params={"cursor": cursor} if cursor else None
        )

    def submit(self, message: str, conversation_id: str | None) -> dict[str, Any]:
        return self.request(
            "POST",
            "/runs",
            headers={"Idempotency-Key": str(uuid4())},
            json={"message": message, "conversation_id": conversation_id},
        )

    def clarify(self, run_id: str, wait_token: str, answer: str) -> None:
        self.request(
            "POST",
            f"/runs/{run_id}/input",
            json={
                "wait_token": wait_token,
                "kind": "clarification",
                "payload": {"answer": answer},
            },
        )

    def approve(self, run_id: str, wait_token: str, approved: bool) -> None:
        self.request(
            "POST",
            f"/runs/{run_id}/input",
            json={
                "wait_token": wait_token,
                "kind": "approval",
                "payload": {"approved": approved},
            },
        )

    def cancel(self, run_id: str) -> None:
        self.request("POST", f"/runs/{run_id}/cancel")


def submit_chat_message(
    api: AgentAPI, message: str, conversation_id: str | None, runs: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Route an explicit chat approval to its bound wait; other text starts a run."""
    approvals = [
        item["run"]
        for item in runs
        if item["run"]["status"] == "waiting_for_input"
        and item["run"].get("wait_kind") == "approval"
    ]
    decision = message.strip().lower()
    if decision in {"approve", "reject"}:
        if not approvals:
            raise UIError("No action is waiting for approval in this conversation.")
        if len(approvals) != 1:
            raise UIError("More than one action awaits approval. Use the button on the intended run.")
        run = approvals[0]
        api.approve(run["run_id"], run["wait_token"], decision == "approve")
        return None
    return api.submit(message, conversation_id)


def source_passages(result: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    cited = set(result.get("source_ids") or [])
    passages: list[dict[str, Any]] = []
    for step in result.get("results", {}).values():
        if not isinstance(step, dict):
            continue
        for passage in step.get("passages", []):
            if not isinstance(passage, dict) or not isinstance(passage.get("chunk_id"), str):
                continue
            if cited and passage["chunk_id"] not in cited:
                continue
            if passage["chunk_id"] not in seen:
                seen.add(passage["chunk_id"])
                passages.append(passage)
    return passages


def render_sources(result: dict[str, Any]) -> None:
    passages = source_passages(result)
    if passages:
        with st.expander(f"Sources ({len(passages)})"):
            for passage in passages:
                st.write(f"{passage.get('title') or 'Document'} · {passage.get('locator') or ''}")
                st.caption(f"Source ID: {passage['chunk_id']}")
                st.write(passage.get("text") or "")
        return
    source_ids = result.get("source_ids") or []
    if source_ids:
        with st.expander(f"Source references ({len(source_ids)})"):
            for source_id in source_ids:
                st.code(str(source_id))


def render_llm_activity(activity: list[dict[str, Any]]) -> None:
    if not activity:
        return
    totals: dict[str, int] = {}
    for call in activity:
        for name, value in (call.get("usage") or {}).items():
            if isinstance(value, int):
                totals[name] = totals.get(name, 0) + value
    with st.expander(f"LLM activity ({len(activity)} calls)"):
        if totals:
            st.caption(
                "Run token usage · "
                + " · ".join(
                    f"{name.replace('_', ' ')}: {value:,}" for name, value in totals.items()
                )
            )
        for index, call in enumerate(activity, start=1):
            purpose = str(call.get("purpose") or "LLM call").replace(".", " · ")
            model = " / ".join(
                str(value) for value in (call.get("provider"), call.get("model")) if value
            )
            st.markdown(f"**{index}. {purpose}**")
            st.caption(" · ".join(value for value in (model, call.get("thinking_level")) if value))
            usage = call.get("usage") or {}
            if usage:
                st.caption(
                    " · ".join(
                        f"{name.replace('_', ' ')}: {value:,}"
                        for name, value in usage.items()
                        if isinstance(value, int)
                    )
                )
            for summary in call.get("thought_summaries", []):
                if summary:
                    st.markdown("**Gemini thought summary**")
                    st.write(summary)
            for function_call in call.get("function_calls", []):
                if isinstance(function_call, dict):
                    st.markdown("**Selected tool**")
                    st.code(function_call.get("name", "unknown"))
                    if function_call.get("arguments"):
                        st.json(function_call["arguments"])
            response = call.get("response")
            if response:
                st.markdown("**Tool result**" if purpose == "tool_result" else "**Model response**")
                if isinstance(response, (dict, list)):
                    st.json(response)
                else:
                    st.code(
                        str(response),
                        language="json" if str(response).lstrip().startswith(("{", "[")) else None,
                    )
            citations = call.get("citations") or []
            for citation in citations:
                if isinstance(citation, dict):
                    label = citation.get("title") or citation.get("url")
                    if label:
                        st.caption(f"Citation: {label} · {citation.get('url', '')}")


def render_debug_timeline(api: AgentAPI, run_id: str) -> bool:
    cache = st.session_state.setdefault("run_debug_timeline", {})
    entry = cache.setdefault(run_id, {"events": [], "cursor": None})
    try:
        while True:
            page = api.debug_events(run_id, entry["cursor"])
            new_events = page.get("events") or []
            entry["events"].extend(new_events)
            entry["cursor"] = page.get("next_cursor") or entry["cursor"]
            if not page.get("has_more") or not new_events:
                break
    except UIError as error:
        st.error(f"Could not load agent exchange: {error}")
    events = entry["events"]
    if not events:
        return False
    goal_event = next((event for event in events if event.get("event_type") == "debug_goal_plan"), None)
    if goal_event:
        evidence = goal_event.get("evidence") or {}
        st.markdown(f"**Goal:** {evidence.get('user_goal', '')}")
        actions = evidence.get("tool_actions") or []
        if actions:
            st.markdown("**Actions:**")
            for index, action in enumerate(actions, start=1):
                st.write(f"{index}. {action}")
        missing = evidence.get("missing_information") or []
        if missing:
            st.caption("Need clarification: " + ", ".join(missing))
    with st.expander(f"Agent exchange ({len(events)} events)", expanded=True):
        for index, event in enumerate(events, start=1):
            label = str(event.get("event_type", "event")).removeprefix("debug_").replace("_", " ")
            st.markdown(f"**{index}. {label.title()}**")
            st.caption(str(event.get("created_at") or ""))
            st.json(event.get("evidence") or {})
    return True


def render_run(api: AgentAPI, item: dict[str, Any]) -> None:
    run = item["run"]
    run_id = run["run_id"]
    status = run["status"]
    with st.chat_message("user"):
        st.write(item["message"])
    with st.chat_message("assistant"):
        st.caption(STATUS_LABELS.get(status, status.replace("_", " ").title()))
        result = run.get("result") or {}
        summary = result.get("summary")
        if summary:
            st.write(summary)
        suggested_steps = result.get("steps") or []
        if suggested_steps:
            with st.expander("Suggested steps and tools", expanded=True):
                for index, step in enumerate(suggested_steps, start=1):
                    st.markdown(
                        f"**{index}. {step.get('action', 'Step')} {step.get('entity', '')}**"
                    )
                    tools = step.get("tools") or []
                    if not tools:
                        st.caption("No eligible tools were found for this step.")
                    for tool in tools:
                        st.write(
                            f"{tool.get('name', tool.get('tool_id', 'Tool'))}: {tool.get('description', '')}"
                        )
        elif status == "completed":
            st.write("Done.")
        elif status == "failed":
            st.write("The request could not be completed.")
            code = run.get("error_code") or result.get("error_code")
            reason = result.get("failure_reason")
            if code or reason or run.get("error_type"):
                detail = " · ".join(
                    str(value) for value in (code, reason, run.get("error_type")) if value
                )
                st.caption(f"Failure detail: {detail}")
        elif status == "needs_review":
            st.write("A write outcome or run recovery needs investigation before retrying.")
            code = run.get("error_code") or result.get("error_code")
            if code:
                st.caption(f"Review reason: {code}")
        elif status == "cancelled":
            st.write("Request cancelled.")
        elif status == "waiting_for_input":
            st.write(run.get("wait_reason") or "Please provide more detail.")
        else:
            st.write("The agent is working on this request.")

        if status == "failed" and result.get("error_detail"):
            st.markdown("**Provider error**")
            st.code(str(result["error_detail"]))

        if status == "waiting_for_input" and run.get("wait_kind") == "clarification":
            with st.form(f"clarify-{run_id}"):
                answer = st.text_area("Your answer", key=f"answer-{run_id}")
                if st.form_submit_button("Send answer", type="primary"):
                    if not answer.strip():
                        st.warning("Enter an answer first.")
                    else:
                        try:
                            api.clarify(run_id, run["wait_token"], answer.strip())
                            st.rerun()
                        except UIError as error:
                            st.error(str(error))
        if status == "waiting_for_input" and run.get("wait_kind") == "approval":
            st.caption("You can type **approve** or **reject** in the chat after reviewing this action.")
            left, right = st.columns(2)
            if left.button("Approve sandbox action", key=f"approve-{run_id}", type="primary"):
                try:
                    api.approve(run_id, run["wait_token"], True)
                    st.rerun()
                except UIError as error:
                    st.error(str(error))
            if right.button("Reject", key=f"reject-{run_id}"):
                try:
                    api.approve(run_id, run["wait_token"], False)
                    st.rerun()
                except UIError as error:
                    st.error(str(error))
        render_sources(result)
        if not render_debug_timeline(api, run_id):
            render_llm_activity(run.get("llm_activity") or [])
        with st.expander("Run details"):
            st.write(f"Run ID: {run_id}")
            st.write(f"Status checked: {run['as_of']}")
            if status not in FINAL_STATUSES and st.button("Cancel request", key=f"cancel-{run_id}"):
                try:
                    api.cancel(run_id)
                    st.rerun()
                except UIError as error:
                    st.error(str(error))


def main() -> None:
    st.set_page_config(page_title="Agent chat", page_icon="💬", layout="centered")
    st.title("Agent chat")
    st.caption("Ask about available tools or your imported documents.")
    try:
        api = AgentAPI(local_api_url(), local_token())
        conversations = api.conversations()
    except UIError as error:
        st.error(str(error))
        st.stop()

    try:
        with st.sidebar:
            st.header("Conversations")
            if st.button("New chat", type="primary", use_container_width=True):
                st.session_state.conversation_id = None
                st.session_state.poll_active = False
                st.rerun()
            for conversation in conversations:
                if st.button(
                    conversation["title"] or "Conversation",
                    key=f"conversation-{conversation['conversation_id']}",
                    use_container_width=True,
                ):
                    st.session_state.conversation_id = conversation["conversation_id"]
                    st.session_state.poll_active = True
                    st.rerun()

        conversation_id = st.session_state.get("conversation_id")
        if conversation_id is None:
            st.info("Start a new conversation below, or open a recent one.")
        else:
            poll_every = "2s" if st.session_state.get("poll_active", True) else None

            @st.fragment(run_every=poll_every)
            def conversation_panel() -> None:
                panel_api = AgentAPI(local_api_url(), local_token())
                try:
                    try:
                        detail = panel_api.conversation(conversation_id)
                    except UIError as error:
                        st.error(str(error))
                        return
                    active = any(
                        run["run"]["status"] not in FINAL_STATUSES for run in detail["runs"]
                    )
                    for run in detail["runs"]:
                        render_run(panel_api, run)
                    if st.session_state.get("poll_active", True) and not active:
                        st.session_state.poll_active = False
                        st.rerun()
                finally:
                    panel_api.close()

            conversation_panel()

        message = st.chat_input("Ask the agent or approve a pending action", max_chars=4000)
        if message and message.strip():
            try:
                runs = api.conversation(conversation_id)["runs"] if conversation_id else []
                accepted = submit_chat_message(api, message.strip(), conversation_id, runs)
            except UIError as error:
                st.error(str(error))
            else:
                if accepted:
                    st.session_state.conversation_id = accepted["conversation_id"]
                st.session_state.poll_active = True
                st.rerun()
    finally:
        api.close()


if __name__ == "__main__":
    main()
