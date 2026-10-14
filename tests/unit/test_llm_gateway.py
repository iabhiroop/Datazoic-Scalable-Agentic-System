"""Provider swapping, structured output and safe failure boundaries."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from app.agent.gateway import (
    PROVIDER_FACTORIES,
    LLMError,
    LLMGateway,
    LLMResponse,
    append_llm_trace,
    capture_llm_activity,
    create_llm_gateway,
    current_llm_trace,
)
from app.agent.gemini import GeminiProvider
from app.agent.llm import ModelClient
from app.main import create_app
from app.settings import Settings


class Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str


def settings(**changes):
    return Settings(
        _env_file=None,
        gemini_model_name="models/gemini-3.8-flash",
        gemini_api_key="test-secret",
        **changes,
    )


def fake_provider(text='{"message":"hello"}'):
    return SimpleNamespace(
        generate=AsyncMock(return_value=LLMResponse(text, "fake", "test-model")),
        aclose=AsyncMock(),
    )


@pytest.fixture
def sdk(monkeypatch):
    interaction = SimpleNamespace(
        status="completed",
        output_text="hello",
        id="interaction-1",
        steps=[],
        usage=SimpleNamespace(total_input_tokens=10, total_output_tokens=2),
    )
    client = SimpleNamespace(
        aio=SimpleNamespace(
            interactions=SimpleNamespace(create=AsyncMock(return_value=interaction)),
            aclose=AsyncMock(),
        ),
        close=Mock(),
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr("app.agent.gemini.genai.Client", factory)
    return client, factory


@pytest.mark.asyncio
async def test_structured_response_validated_locally():
    provider = fake_provider()
    gateway = LLMGateway(provider)
    assert (await gateway.generate_structured("reply", Reply)).message == "hello"
    request = provider.generate.call_args.args[0]
    assert request.response_schema["additionalProperties"] is False
    assert request.web_search is False


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["not json", '{"message":"ok","unexpected":"secret"}'])
async def test_bad_structured_output_is_rejected_without_echoing_payload(text):
    gateway = LLMGateway(fake_provider(text))
    with pytest.raises(LLMError) as error:
        await gateway.generate_structured("reply", Reply)
    assert error.value.code == "invalid_output"
    assert text not in str(error.value)


@pytest.mark.asyncio
async def test_empty_and_timed_out_outputs_are_not_success():
    with pytest.raises(LLMError, match="no text"):
        await LLMGateway(fake_provider(" ")).generate("reply")

    async def slow(request):
        await asyncio.sleep(1)

    provider = fake_provider()
    provider.generate.side_effect = slow
    with pytest.raises(LLMError) as error:
        await LLMGateway(provider, timeout_seconds=0.01).generate("reply")
    assert error.value.code == "timeout"
    assert error.value.retryable is True


@pytest.mark.asyncio
async def test_provider_can_be_swapped_without_changing_model_client(monkeypatch):
    provider = fake_provider()
    factory = Mock(return_value=provider)
    monkeypatch.setitem(PROVIDER_FACTORIES, "alternative", factory)
    gateway = create_llm_gateway(settings(llm_provider="alternative"))
    assert (await gateway.generate("reply")).provider == "fake"
    await gateway.aclose()
    provider.aclose.assert_awaited_once()


def test_unknown_provider_fails_without_fallback():
    with pytest.raises(ValueError, match="not registered"):
        create_llm_gateway(settings(llm_provider="missing"))


def test_gemini_requires_both_explicit_settings():
    for config in (
        Settings(_env_file=None, gemini_model_name="", gemini_api_key="test"),
        Settings(_env_file=None, gemini_model_name="model", gemini_api_key=""),
    ):
        with pytest.raises(ValueError, match="GEMINI_MODEL_NAME"):
            GeminiProvider(config)
    assert "test-secret" not in repr(settings())


@pytest.mark.asyncio
async def test_gemini_uses_async_interactions_explicit_key_and_json_schema(sdk):
    client, factory = sdk
    provider = GeminiProvider(settings())
    assert factory.call_args.kwargs["api_key"] == "test-secret"
    assert factory.call_args.kwargs["vertexai"] is False
    assert factory.call_args.kwargs["http_options"].timeout == 60000
    gateway = LLMGateway(provider)
    response = await gateway.generate("reply", response_schema=Reply.model_json_schema())
    arguments = client.aio.interactions.create.call_args.kwargs
    assert arguments["model"] == "gemini-3.8-flash"
    assert arguments["store"] is False
    assert arguments["stream"] is False
    assert "tools" not in arguments
    assert arguments["generation_config"] == {
        "thinking_level": "medium",
        "thinking_summaries": "auto",
    }
    assert arguments["response_format"]["mime_type"] == "application/json"
    assert response.usage == {"input_tokens": 10, "output_tokens": 2}
    await gateway.aclose()
    client.aio.aclose.assert_awaited_once()
    client.close.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_function_call_and_stateless_continuation(sdk):
    from app.agent.state import ToolDeclaration

    client, _ = sdk
    step = SimpleNamespace(
        type="function_call",
        name="get_invoice",
        arguments={"invoice_id": "INV-1"},
        id="call-1",
        model_dump=Mock(
            return_value={
                "type": "function_call",
                "name": "get_invoice",
                "id": "call-1",
                "arguments": {"invoice_id": "INV-1"},
                "signature": "private",
            }
        ),
    )
    interaction = client.aio.interactions.create.return_value
    interaction.steps = [step]
    interaction.output_text = ""
    history = [{"type": "user_input", "content": [{"type": "text", "text": "get an invoice"}]}]
    gateway = LLMGateway(GeminiProvider(settings()))
    response = await gateway.generate(
        "ignored after initial turn",
        tool_declarations=[
            ToolDeclaration(name="get_invoice", description="Get", parameters={"type": "object"})
        ],
        continuation=history,
    )
    request = client.aio.interactions.create.call_args.kwargs
    assert request["store"] is False
    assert request["input"] == history
    assert request["tools"][0]["name"] == "get_invoice"
    assert response.function_calls[0].call_id == "call-1"
    assert "signature" in response.continuation[-1]
    assert "private" not in repr(response.function_calls)


@pytest.mark.asyncio
async def test_google_search_is_explicit_and_preserves_citations(sdk):
    client, _ = sdk
    citation = SimpleNamespace(
        type="url_citation",
        model_dump=Mock(
            return_value={
                "type": "url_citation",
                "url": "https://example.com",
                "title": "Example",
            }
        ),
    )
    client.aio.interactions.create.return_value.steps = [
        SimpleNamespace(type="thought"),
        SimpleNamespace(
            type="model_output",
            content=[
                SimpleNamespace(type="text", annotations=[citation]),
            ],
        ),
    ]
    response = await LLMGateway(GeminiProvider(settings())).generate("reply", web_search=True)
    assert client.aio.interactions.create.call_args.kwargs["tools"] == [{"type": "google_search"}]
    assert response.citations[0]["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_gemini_returns_only_text_thought_summaries_and_never_signatures(sdk):
    client, _ = sdk
    client.aio.interactions.create.return_value.steps = [
        SimpleNamespace(
            type="thought",
            summary=[
                SimpleNamespace(type="text", text="A brief provider summary."),
                SimpleNamespace(type="image", text="ignored"),
            ],
            signature="private-signature",
        ),
        SimpleNamespace(type="thought", summary=None, signature="another-signature"),
    ]
    response = await LLMGateway(GeminiProvider(settings())).generate("reply")
    assert response.thought_summaries == ["A brief provider summary."]
    assert "private-signature" not in repr(response)


@pytest.mark.asyncio
async def test_activity_capture_keeps_redacted_output_even_when_structured_validation_fails():
    output = '{"unexpected":"user@example.com"}'
    gateway = LLMGateway(fake_provider(output))
    from app.observability import SecretRedactor

    with (
        capture_llm_activity(SecretRedactor(Settings(_env_file=None))) as captured,
        pytest.raises(LLMError, match="expected contract"),
    ):
        await gateway.generate_structured("private prompt", Reply, observation_name="classify")
    assert len(captured) == 1
    call = captured[0]
    assert call["purpose"] == "classify"
    assert "[REDACTED]" in call["response"]
    assert "private prompt" not in repr(call)


@pytest.mark.asyncio
async def test_large_prompt_is_sent_and_provider_usage_is_recorded():
    model = fake_provider()
    model.generate.return_value = LLMResponse(
        '{"message":"hello"}',
        "fake",
        "test-model",
        usage={"input_tokens": 12_000, "output_tokens": 20, "total_tokens": 12_020},
    )
    from app.observability import SecretRedactor

    with capture_llm_activity(SecretRedactor(Settings(_env_file=None))) as captured:
        await LLMGateway(model).generate("x" * 50_000, observation_name="plan")
    assert len(model.generate.call_args.args[0].prompt) == 50_000
    assert captured[0]["usage"]["total_tokens"] == 12_020


@pytest.mark.asyncio
async def test_request_response_and_tool_selection_are_saved_as_redacted_jsonl(tmp_path):
    from app.agent.state import FunctionCall, ToolDeclaration
    from app.observability import SecretRedactor

    call = FunctionCall(
        name="invoice_create", arguments={"email": "user@example.com"}, call_id="call-1"
    )
    provider = fake_provider("")
    provider.generate.return_value = LLMResponse(
        "",
        "gemini",
        "test-model",
        function_calls=[call],
        usage={"total_tokens": 9},
        continuation=[
            {"type": "user_input", "content": [{"type": "text", "text": "query"}]},
            {"type": "function_call", "name": "invoice_create", "signature": "private-signature"},
        ],
    )
    with capture_llm_activity(SecretRedactor(Settings(_env_file=None))) as activity:
        response = await LLMGateway(provider).generate(
            '{"request":"user@example.com"}',
            observation_name="tool_selection",
            tool_declarations=[
                ToolDeclaration(
                    name="invoice_create", description="Create", parameters={"type": "object"}
                )
            ],
        )
        traces = current_llm_trace()
    assert response.function_calls[0].name == "invoice_create"
    assert "[REDACTED]" in str(activity[0]["function_calls"])
    path = tmp_path / "run.jsonl"
    append_llm_trace(path, traces)
    saved = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert saved["request"]["tools"][0]["name"] == "invoice_create"
    assert saved["function_calls"][0]["name"] == "invoice_create"
    assert "[REDACTED]" in str(saved["function_calls"])
    assert saved["usage"]["total_tokens"] == 9
    assert "private-signature" not in str(saved["provider_steps"])


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "failed", "requires_action", "in_progress"])
async def test_noncompleted_interaction_is_never_returned(sdk, status):
    client, _ = sdk
    client.aio.interactions.create.return_value.status = status
    with pytest.raises(LLMError) as error:
        await LLMGateway(GeminiProvider(settings())).generate("reply")
    assert error.value.code == "incomplete_output"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, code", [
        (400, "provider_request_rejected"),
        (403, "authentication"),
        (429, "rate_limited"),
        (500, "provider_unavailable"),
        (503, "provider_busy"),
    ]
)
async def test_sdk_errors_do_not_expose_keys_or_prompts(sdk, status, code):
    client, _ = sdk
    error = RuntimeError("test-secret and confidential prompt")
    error.code = status
    client.aio.interactions.create.side_effect = error
    with pytest.raises(LLMError) as result:
        await LLMGateway(GeminiProvider(settings())).generate("reply")
    assert result.value.code == code
    assert "test-secret" not in str(result.value)
    assert "confidential" not in str(result.value)


@pytest.mark.asyncio
async def test_interactions_sdk_status_code_is_used_for_busy_retries(sdk):
    client, _ = sdk
    error = RuntimeError("private provider response")
    error.status_code = 503
    client.aio.interactions.create.side_effect = error
    with pytest.raises(LLMError) as result:
        await LLMGateway(GeminiProvider(settings())).generate("reply")
    assert result.value.code == "provider_busy"
    assert result.value.retryable is True
    assert "private" not in str(result.value)


@pytest.mark.asyncio
async def test_cancelled_request_stays_cancelled(sdk):
    client, _ = sdk
    client.aio.interactions.create.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await LLMGateway(GeminiProvider(settings())).generate("reply")


@pytest.mark.asyncio
async def test_rag_rejects_fabricated_sources_and_skips_empty_evidence():
    provider = fake_provider(
        json.dumps(
            {
                "sufficient_evidence": True,
                "answer": "Invented",
                "source_ids": ["outside"],
            }
        )
    )
    model = ModelClient(LLMGateway(provider))
    assert (await model.answer("question", []))["sufficient_evidence"] is False
    provider.generate.assert_not_awaited()
    with pytest.raises(LLMError, match="outside"):
        await model.answer("question", [{"chunk_id": "allowed", "text": "evidence"}])


@pytest.mark.asyncio
async def test_grounded_response_preserves_application_status():
    model = ModelClient(LLMGateway(fake_provider('{"summary":"Pending", "source_ids":[]}')))
    result = await model.respond({"business_status": "pending"})
    assert result["business_status"] == "pending"
    with pytest.raises(ValueError, match="business_status"):
        await model.respond({"business_status": "unverified-success"})


@pytest.mark.asyncio
async def test_classify_uses_intent_contract_and_plan_discards_model_action_slots():
    intent_payload = {
        "goal": "Read payment",
        "subgoals": [],
        "category_ids": [],
        "entities": {},
        "constraints": {},
        "missing_fields": [],
        "confidence": 0.8,
    }
    provider = fake_provider(json.dumps(intent_payload))
    model = ModelClient(LLMGateway(provider))
    intent = await model.classify("Read payment", {})
    assert intent.goal == "Read payment"
    assert (
        intent.execution_mode == "plan_only"
    )  # Missing mode defaults to the safe no-dispatch path.
    provider.generate.return_value = LLMResponse(
        json.dumps(
            {
                "plan_version": 1,
                "steps": [
                    {
                        "step_id": "s1",
                        "tool_id": "get_payment",
                        "schema_version": "1",
                        "arguments": {"payment_id": {"literal": "P1"}},
                        "action_ref": "model-chosen-action",
                    }
                ],
            }
        ),
        "fake",
        "model",
    )
    plan = await model.plan(intent, {})
    assert plan.steps[0].action_ref is None
    request = provider.generate.call_args.args[0]
    assert request.web_search is False
    assert "RFC 6901" in request.system_instruction
    assert "never JSON-encode them into strings" in request.system_instruction


@pytest.mark.asyncio
async def test_classify_never_requests_server_managed_credentials_or_configuration():
    provider = fake_provider(
        json.dumps(
            {
                "goal": "List invoices",
                "subgoals": [],
                "category_ids": ["paypal"],
                "entities": {},
                "constraints": {},
                "missing_fields": [
                    "authentication_credentials",
                    "sandbox_api_configuration",
                    "invoice_number",
                ],
                "confidence": 0.8,
            }
        )
    )
    intent = await ModelClient(LLMGateway(provider)).classify("List invoices", {})
    assert intent.missing_fields == ["invoice_number"]


def test_api_lifecycle_constructs_one_gateway_without_calling_provider(monkeypatch):
    provider = fake_provider()
    gateway = LLMGateway(provider)
    factory = Mock(return_value=gateway)
    monkeypatch.setattr("app.main.create_llm_gateway", factory)
    config = settings(database_url=None, checkpoint_database_url=None)
    application = create_app(config)
    with TestClient(application) as client:
        assert application.state.model_client.gateway is gateway
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/live").status_code == 200
        factory.assert_called_once()
        provider.generate.assert_not_awaited()
    provider.aclose.assert_awaited_once()
