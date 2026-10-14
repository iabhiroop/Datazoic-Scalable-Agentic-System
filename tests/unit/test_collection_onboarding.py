"""Draft staging cannot enable an adapter; bulk imports reuse valid versions."""

import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.knowledge.taxonomy import TaxonomyCluster
from app.security import Principal
from app.settings import Settings
from app.tools.contracts import EndpointContract, ToolCard
from app.tools.onboarding import (
    _cached_or_new_review,
    _evaluation_vectors,
    choose_version,
    onboard_collection,
)
from app.tools.registry import ToolRegistry


def card():
    path = Path(__file__).resolve().parents[2] / "resources/tool_cards/system_search.json"
    return ToolCard.model_validate_json(path.read_text())


def record(tool, **changes):
    values = {
        "card": tool.model_dump(),
        "schema_version": tool.schema_version,
        "tool_id": tool.tool_id,
        "status": "published",
        "is_current": True,
        "embedding": [1, 0, 0],
        "embedding_version": "v1",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def config():
    return Settings(
        _env_file=None, embedding_model="test", embedding_version="v1", embedding_dimension=3
    )


async def test_draft_staging_still_blocks_unconfigured_publication():
    registry = ToolRegistry(set(), database=None, verification_keys=set())
    draft = card().model_copy(update={"adapter_key": "unconfigured"})
    assert registry.validate_draft(draft) == draft
    with pytest.raises(ValueError, match="adapter"):
        await registry.publish(draft, Principal(uuid4(), "admin", frozenset({"tools:manage"})))
    with pytest.raises(ValueError, match="External"):
        registry.validate_draft(
            draft.model_copy(update={"input_schema": {"$ref": "https://example.invalid"}})
        )


def test_version_reuse_requires_matching_contract_and_embedding():
    stored = card().model_copy(
        update={"schema_version": "2", "embedding_model": "test", "embedding_version": "v1"}
    )
    candidate, reused = choose_version(card(), [record(stored)], config())
    assert reused and candidate.schema_version == "2"
    changed = card().model_copy(update={"description": "Changed contract"})
    candidate, reused = choose_version(changed, [record(stored)], config())
    assert not reused and candidate.schema_version == "3"
    candidate, reused = choose_version(card(), [record(stored, status="revoked")], config())
    assert not reused and candidate.schema_version == "3"


async def test_batch_uses_standard_import_and_reuses_published_version(tmp_path):
    original = card()
    stored = original.model_copy(update={"embedding_model": "test", "embedding_version": "v1"})
    another = original.model_copy(update={"tool_id": "other", "adapter_key": "unconfigured"})

    class Database:
        @asynccontextmanager
        async def session(self):
            yield SimpleNamespace(scalars=AsyncMock(return_value=[record(stored)]))

    service = SimpleNamespace(
        database=Database(),
        settings=config(),
        import_tool=AsyncMock(return_value=another),
    )
    inventory = {
        "endpoints": [
            {
                "tool_id": tool.tool_id,
                "method": "GET",
                "path": "/v1/resource",
                "needs_manual_mapping": False,
            }
            for tool in (original, another)
        ]
    }
    principal = Principal(uuid4(), "admin", frozenset({"tools:manage"}))
    rows = await onboard_collection([original, another], inventory, tmp_path, service, principal)
    assert [(row["outcome"], row["status"]) for row in rows] == [
        ("reused", "published"),
        ("imported", "draft"),
    ]
    service.import_tool.assert_awaited_once_with(
        tmp_path / "other.json", principal, draft_only=True
    )
    report = json.loads((tmp_path / "onboarding-report.json").read_text())
    assert report["counts"] == {"reused": 1, "imported": 1}


async def test_manual_review_avoids_gemini_and_approves_clear_descriptive_metadata(tmp_path):
    tool = card().model_copy(
        update={
            "endpoint": EndpointContract(
                method="GET", path="/v2/invoicing/invoices", schema_source="official-spec"
            )
        }
    )
    cluster = TaxonomyCluster("invoice-tools", (tool.tool_id,), (tool.tool_id,))
    review, source = await _cached_or_new_review(
        tmp_path / "taxonomy-review.json",
        [cluster],
        [tool],
        None,
        "manual",
        {tool.tool_id: "/v2/invoicing/invoices"},
    )
    assert source == "manual"
    assert review.labels[0].label == "Invoices"
    assert review.tools[0].approved
    assert '"paypal"' not in (tmp_path / "taxonomy-review.json").read_text()


async def test_manual_review_keeps_ambiguous_token_actions_in_draft(tmp_path):
    tool = card().model_copy(update={"name": "Create terminate", "action": "create"})
    cluster = TaxonomyCluster("token-tools", (tool.tool_id,), (tool.tool_id,))
    review, _ = await _cached_or_new_review(
        tmp_path / "taxonomy-review.json",
        [cluster],
        [tool],
        None,
        "manual",
        {tool.tool_id: "/v1/oauth2/token/terminate"},
    )
    assert not review.tools[0].approved
    assert review.tools[0].reason_code == "unclear_metadata"


async def test_manual_threshold_evaluation_uses_cache_without_embedding_calls(tmp_path):
    evaluation = tmp_path / "queries.json"
    cache = tmp_path / "vectors.json"
    cases = [{"query": "find invoices", "expected_tool_ids": ["tool-a"]}]
    evaluation.write_text(json.dumps(cases))
    key = hashlib.sha256(b"find invoices").hexdigest()
    fingerprint_payload = {
        "model": "test",
        "version": "v1",
        "dimension": 3,
        "cases": [{"query": "find invoices", "tool_ids": ["tool-a"]}],
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    cache.write_text(json.dumps({"fingerprint": fingerprint, "vectors": {key: [1, 0, 0]}}))
    service = SimpleNamespace(
        settings=config(), retrieval=SimpleNamespace(query_vector=AsyncMock())
    )
    result = await _evaluation_vectors(evaluation, cache, {"tool-a"}, service, allow_embedding=False)
    assert result == [([1, 0, 0], {"tool-a"})]
    service.retrieval.query_vector.assert_not_awaited()
