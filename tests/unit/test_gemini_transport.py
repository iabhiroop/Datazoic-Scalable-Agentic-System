"""Exercise the actual Google SDK serialization over an in-memory HTTP transport."""

import json

import httpx
import pytest
from google import genai

from app.agent.gateway import LLMGateway
from app.agent.gemini import GeminiProvider
from app.settings import Settings


@pytest.mark.asyncio
async def test_actual_sdk_serializes_interaction_and_extracts_final_text(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "wire-test",
                "status": "completed",
                "model": "gemini-3.8-flash",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "OK"}]}],
                "usage": {"total_input_tokens": 3, "total_output_tokens": 1},
            },
        )

    real_factory = genai.Client

    def factory(**kwargs):
        kwargs["http_options"].httpx_async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        )
        return real_factory(**kwargs)

    monkeypatch.setattr("app.agent.gemini.genai.Client", factory)
    config = Settings(
        _env_file=None,
        gemini_model_name="models/gemini-3.8-flash",
        gemini_api_key="wire-test-secret",
    )
    gateway = LLMGateway(GeminiProvider(config))
    try:
        reply = await gateway.generate(
            "Reply OK.",
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        )
        assert reply.text == "OK"
        assert reply.interaction_id == "wire-test"
        assert len(requests) == 1
        request = requests[0]
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path.endswith("/interactions")
        assert request.headers["x-goog-api-key"] == "wire-test-secret"
        body = json.loads(request.content)
        assert body["model"] == "gemini-3.8-flash"
        assert body["store"] is False
        assert body["generation_config"]["thinking_summaries"] == "auto"
        assert body["response_format"]["mime_type"] == "application/json"
        assert "api_key" not in body
    finally:
        await gateway.aclose()
