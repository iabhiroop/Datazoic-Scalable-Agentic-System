"""Provider wire format, vector validation and deterministic rank fusion."""

import json

import httpx
import pytest
from google import genai

from app.agent.gateway import LLMError
from app.agent.gemini import GeminiEmbeddingProvider
from app.knowledge.retrieval import RetrievalService, fuse_ranks
from app.settings import Settings


def provider(monkeypatch, payload):
    requests = []
    real_factory = genai.Client

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    def factory(**kwargs):
        kwargs["http_options"].httpx_async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        )
        return real_factory(**kwargs)

    monkeypatch.setattr("app.agent.gemini.genai.Client", factory)
    settings = Settings(
        _env_file=None,
        gemini_api_key="synthetic-key",
        embedding_model="gemini-embedding-2",
        embedding_version="test-v1",
        embedding_dimension=768,
    )
    return GeminiEmbeddingProvider(settings), requests


async def test_actual_sdk_embedding_request_uses_matching_retrieval_formats(monkeypatch):
    adapter, requests = provider(monkeypatch, {"embeddings": [{"values": [1.0] * 768}]})
    try:
        service = RetrievalService(adapter)
        assert len(await service.embed("Return window?")) == 768
        assert len(await service.embed("14 days", task="document", title="Policy")) == 768
        for request in requests:
            assert request.url.path.endswith("gemini-embedding-2:batchEmbedContents")
            body = json.loads(request.content)
            assert len(body["requests"]) == 1
            assert body["requests"][0]["outputDimensionality"] == 768
            assert "api_key" not in body
        query, document = [json.loads(request.content)["requests"][0] for request in requests]
        assert query["content"]["parts"][0]["text"] == "task: search result | query: Return window?"
        assert document["content"]["parts"][0]["text"] == "title: Policy | text: 14 days"
    finally:
        await adapter.aclose()


@pytest.mark.parametrize(
    "payload",
    [
        {"embeddings": []},
        {"embeddings": [{"values": [1.0]}]},
        {"embeddings": [{"values": [0.0] * 768}]},
        {"embeddings": [{"values": [1.0] * 768}, {"values": [1.0] * 768}]},
    ],
)
async def test_invalid_embedding_rejected(monkeypatch, payload):
    adapter, _ = provider(monkeypatch, payload)
    try:
        with pytest.raises(LLMError) as error:
            await adapter.embed("question")
        assert error.value.code == "invalid_embedding"
    finally:
        await adapter.aclose()


def test_fusion_deduplicates_and_ties_are_stable():
    assert fuse_ranks([], []) == []
    assert fuse_ranks(["b", "b", "a"], ["a", "b"]) == [
        ("a", 1 / 61 + 1 / 62),
        ("b", 1 / 61 + 1 / 62),
    ]
