"""Redaction and optional telemetry must not alter model/business outcomes."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, ConfigDict

from app.agent.gateway import LLMError, LLMGateway, LLMResponse
from app.observability import REDACTED, SecretRedactor, Telemetry, TraceContext
from app.settings import Settings


def config(**changes):
    return Settings(
        _env_file=None,
        gemini_api_key="configured-gemini-secret",
        langfuse_secret_key="configured-langfuse-secret",
        **changes,
    )


def fake_telemetry(**changes):
    client = MagicMock()
    observation = client.start_as_current_observation.return_value.__enter__.return_value
    observation.trace_id = "1" * 32
    observation.id = "2" * 16
    return Telemetry(config(**changes), client), client, observation


def provider(text="hello"):
    return SimpleNamespace(
        model="fake-model",
        max_output_tokens=2000,
        thinking_level="medium",
        generate=AsyncMock(
            return_value=LLMResponse(
                text,
                "fake",
                "fake-model",
                usage={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            )
        ),
        aclose=AsyncMock(),
    )


def test_redaction_handles_nested_serialized_data_without_mutating_original():
    mask = SecretRedactor(config())
    data = {
        "password": "unknown-password",
        "nested": [
            {"text": "configured-gemini-secret", "email": "person@example.com"},
            {"serialized": json.dumps({"api_key": "unknown-api-key", "safe": "keep"})},
        ],
    }
    original = json.dumps(data)
    masked = json.dumps(mask(data))
    assert json.dumps(data) == original
    for value in (
        "unknown-password",
        "configured-gemini-secret",
        "person@example.com",
        "unknown-api-key",
    ):
        assert value not in masked
    assert "keep" in masked


def test_plain_text_secrets_and_url_credentials_are_masked_before_truncation():
    mask = SecretRedactor(config(langfuse_max_content_chars=200))
    text = "Bearer unlisted-token password=unlisted-password postgresql://user:private@db person@example.com configured-langfuse-secret"
    result = mask(text)
    for value in (
        "unlisted-token",
        "unlisted-password",
        "private",
        "person@example.com",
        "configured-langfuse-secret",
    ):
        assert value not in result
    mask.limit = 10
    assert mask("configured-gemini-secret").startswith(REDACTED)


@pytest.mark.asyncio
async def test_generation_captures_redacted_content_usage_and_preserves_provider_input():
    telemetry, client, _ = fake_telemetry()
    model = provider("configured-langfuse-secret")
    gateway = LLMGateway(model, telemetry=telemetry)
    with telemetry.bind_context(TraceContext(request_id="request", run_id="run")):
        result = await gateway.generate("configured-gemini-secret")
    assert result.text == "configured-langfuse-secret"
    assert model.generate.call_args.args[0].prompt == "configured-gemini-secret"
    arguments = client.start_as_current_observation.call_args.kwargs
    assert arguments["name"] == "agent.llm.generate"
    assert arguments["input"]["prompt"] == REDACTED
    assert arguments["metadata"]["run_id"] == "run"
    updates = client.start_as_current_observation.return_value.__enter__.return_value.update.call_args_list
    assert updates[-1].kwargs["output"] == REDACTED
    assert updates[-1].kwargs["usage_details"] == {"input": 10, "output": 3, "total": 13}


@pytest.mark.asyncio
async def test_metadata_only_mode_excludes_text():
    telemetry, client, _ = fake_telemetry(langfuse_capture_content=False)
    await LLMGateway(provider("private-response"), telemetry=telemetry).generate("private-prompt")
    assert client.start_as_current_observation.call_args.kwargs["input"] is None
    updates = client.start_as_current_observation.return_value.__enter__.return_value.update.call_args_list
    assert all("output" not in call.kwargs for call in updates)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["start", "update", "end"])
async def test_telemetry_failures_do_not_reexecute_or_change_success(failure):
    telemetry, client, observation = fake_telemetry()
    manager = client.start_as_current_observation.return_value
    if failure == "start":
        client.start_as_current_observation.side_effect = RuntimeError("SDK secret")
    elif failure == "update":
        observation.update.side_effect = RuntimeError("SDK secret")
    else:
        manager.__exit__.side_effect = RuntimeError("SDK secret")
    model = provider()
    assert (await LLMGateway(model, telemetry=telemetry).generate("request")).text == "hello"
    model.generate.assert_awaited_once()
    assert telemetry.status == "degraded"


@pytest.mark.asyncio
async def test_original_model_error_is_preserved_without_raw_exception_export():
    telemetry, client, observation = fake_telemetry()
    model = provider()
    error = LLMError("rate_limited", "private details", retryable=True)
    model.generate.side_effect = error
    with pytest.raises(LLMError) as result:
        await LLMGateway(model, telemetry=telemetry).generate("request")
    assert result.value is error
    assert observation.update.call_args.kwargs["status_message"] == "rate_limited"
    assert "private details" not in str(observation.update.call_args_list)
    client.start_as_current_observation.return_value.__exit__.assert_called_with(None, None, None)


@pytest.mark.asyncio
async def test_correlation_context_isolated_across_concurrent_tasks():
    telemetry = Telemetry(config())

    async def task(request_id):
        with telemetry.bind_context(TraceContext(request_id=request_id)):
            await asyncio.sleep(0)
            return telemetry.trace_context()["request_id"]

    assert await asyncio.gather(task("one"), task("two")) == ["one", "two"]
    assert telemetry.trace_context() == {}


@pytest.mark.asyncio
async def test_flush_shutdown_failures_are_nonfatal_and_close_is_idempotent():
    telemetry, client, _ = fake_telemetry()
    client.flush.side_effect = RuntimeError("unavailable")
    client.shutdown.side_effect = RuntimeError("unavailable")
    await telemetry.flush()
    await telemetry.aclose()
    await telemetry.aclose()
    client.shutdown.assert_called_once()
    assert telemetry.client is None
    assert telemetry.status == "degraded"


def test_disabled_or_partial_configuration_does_not_construct_sdk(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr("langfuse.Langfuse", factory)
    disabled = Telemetry.from_settings(config(langfuse_tracing_enabled=False))
    assert disabled.status == "disabled"
    incomplete = Telemetry.from_settings(config())
    assert incomplete.status == "degraded"
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_actual_sdk_produces_nested_spans_masks_content_and_tracks_validation_error():
    settings = config(langfuse_public_key="memory-" + uuid4().hex)
    mask = SecretRedactor(settings)
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key="memory-secret",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
        mask=mask.sdk_mask,
        mask_otel_spans=mask.mask_otel_spans,
    )
    telemetry = Telemetry(settings, client)

    class Expected(BaseModel):
        model_config = ConfigDict(extra="forbid")
        answer: str

    try:
        with (
            telemetry.bind_context(TraceContext(request_id="real-request")),
            telemetry.observe("root"),
        ):
            assert telemetry.trace_context()["traceparent"].startswith("00-")
            with pytest.raises(LLMError) as error:
                await LLMGateway(
                    provider("configured-gemini-secret"), telemetry=telemetry
                ).generate_structured(
                    "password=unlisted-secret",
                    Expected,
                    observation_name="classify",
                )
            assert error.value.code == "invalid_output"
        await telemetry.flush()
        spans = exporter.get_finished_spans()
        assert sorted(span.name for span in spans) == [
            "agent.classify",
            "agent.classify",
            "agent.root",
        ]
        root_span = next(span for span in spans if span.name == "agent.root")
        classify = next(
            span
            for span in spans
            if span.name == "agent.classify" and span.parent.span_id == root_span.context.span_id
        )
        generation = next(
            span
            for span in spans
            if span.name == "agent.classify" and span.parent.span_id == classify.context.span_id
        )
        assert classify.parent.span_id == root_span.context.span_id
        assert generation.parent.span_id == classify.context.span_id
        serialized = json.dumps([dict(span.attributes) for span in spans])
        assert "configured-gemini-secret" not in serialized
        assert "unlisted-secret" not in serialized
        assert "invalid_output" in serialized
        assert "real-request" in serialized
        assert not any(span.events for span in spans)
    finally:
        await telemetry.aclose()
