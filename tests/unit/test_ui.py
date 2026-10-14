"""The local chat client uses backend contracts without exposing its token."""

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from app import ui
from app.ui import AgentAPI, UIError, local_api_url, source_passages, submit_chat_message


def test_local_chat_client_submits_and_clarifies_without_token_in_body():
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/runs":
            return httpx.Response(
                202,
                json={
                    "run_id": "run-id",
                    "conversation_id": "conversation-id",
                    "status": "queued",
                    "status_url": "/runs/run-id",
                },
            )
        return httpx.Response(202, json={"status": "queued"})

    api = AgentAPI("http://127.0.0.1:8000", "private-token", transport=httpx.MockTransport(respond))
    try:
        accepted = api.submit("Hello", None)
        api.clarify(accepted["run_id"], "wait-id", "More detail")
        assert accepted["conversation_id"] == "conversation-id"
        assert all(
            request.headers["Authorization"] == "Bearer private-token" for request in requests
        )
        assert requests[0].headers["Idempotency-Key"]
        assert b"private-token" not in requests[0].content
        assert b'"answer":"More detail"' in requests[1].content
    finally:
        api.close()


def test_local_chat_client_hides_unstructured_server_errors():
    api = AgentAPI(
        "http://127.0.0.1:8000",
        "private-token",
        transport=httpx.MockTransport(lambda _: httpx.Response(503, text="private-token")),
    )
    try:
        with pytest.raises(UIError) as error:
            api.conversations()
        assert "private-token" not in str(error.value)
    finally:
        api.close()


def test_demo_api_address_is_local(monkeypatch):
    monkeypatch.setenv("AGENT_API_URL", "https://example.com")
    with pytest.raises(UIError, match="local HTTP"):
        local_api_url()


def test_source_passages_deduplicates_citations():
    result = {
        "results": {
            "first": {"passages": [{"chunk_id": "a", "title": "Policy"}]},
            "second": {"passages": [{"chunk_id": "a"}, {"chunk_id": "b"}]},
        }
    }
    assert [passage["chunk_id"] for passage in source_passages(result)] == ["a", "b"]


def test_chat_approval_uses_bound_wait_instead_of_creating_a_run():
    api = type("API", (), {"approve": lambda self, *args: calls.append(args),
                            "submit": lambda self, *_args: pytest.fail("new run submitted")})()
    calls = []
    runs = [{"run": {"run_id": "run-1", "status": "waiting_for_input",
                      "wait_kind": "approval", "wait_token": "wait-1"}}]
    assert submit_chat_message(api, "approve", "conversation-1", runs) is None
    assert calls == [("run-1", "wait-1", True)]


def test_chat_approval_rejects_ambiguous_waits():
    api = type("API", (), {})()
    runs = [{"run": {"run_id": str(index), "status": "waiting_for_input",
                      "wait_kind": "approval", "wait_token": str(index)}}
            for index in range(2)]
    with pytest.raises(UIError, match="More than one"):
        submit_chat_message(api, "approve", "conversation-1", runs)


def test_chat_approval_rejects_terminal_review_run():
    api = type("API", (), {})()
    runs = [{"run": {"run_id": "run-1", "status": "needs_review"}}]
    with pytest.raises(UIError, match="No action is waiting"):
        submit_chat_message(api, "approve", "conversation-1", runs)


def test_streamlit_chat_submits_and_renders_run(monkeypatch):
    submitted = []

    class FakeAPI:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

        def conversations(self):
            return []

        def submit(self, message, conversation_id):
            submitted.append((message, conversation_id))
            return {"conversation_id": "conversation-id", "run_id": "run-id"}

        def conversation(self, _conversation_id):
            return {
                "runs": [
                    {
                        "message": "What is the return window?",
                        "run": {
                            "run_id": "run-id",
                            "status": "completed",
                            "as_of": "2026-01-01T00:00:00Z",
                            "result": {
                                "summary": "Returns are accepted within 14 days.",
                                "results": {},
                            },
                        },
                    }
                ]
            }

    monkeypatch.setattr(ui, "AgentAPI", FakeAPI)
    monkeypatch.setattr(ui, "local_token", lambda: "synthetic-token")
    app = AppTest.from_string("import app.ui as ui\nui.main()")
    app.run()
    assert not app.exception
    app.chat_input[0].set_value("What is the return window?").run()
    assert submitted == [("What is the return window?", None)]
    assert not app.exception
    assert any("Returns are accepted within 14 days." in item.value for item in app.markdown)


def test_llm_activity_panel_shows_call_output_usage_and_summaries(monkeypatch):
    rendered = []

    class Expander:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def expander(label, **kwargs):
        rendered.append(("expander", label, kwargs))
        return Expander()

    for name in ("caption", "markdown", "write", "code"):
        monkeypatch.setattr(
            ui.st, name, lambda value, _name=name, **_kwargs: rendered.append((_name, value))
        )
    monkeypatch.setattr(ui.st, "expander", expander)
    ui.render_llm_activity(
        [
            {
                "purpose": "rag.answer",
                "provider": "gemini",
                "model": "gemini-test",
                "thinking_level": "medium",
                "usage": {"input_tokens": 10, "output_tokens": 4},
                "response": '{"answer":"Returns are accepted"}',
                "thought_summaries": ["Checking the supplied source."],
            }
        ]
    )
    assert rendered[0] == ("expander", "LLM activity (1 calls)", {})
    assert any("input tokens: 10" in str(entry) for entry in rendered)
    assert any("Checking the supplied source." in str(entry) for entry in rendered)
