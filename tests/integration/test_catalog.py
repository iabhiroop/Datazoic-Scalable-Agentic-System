"""Publication, pinned versions and access filters against real PostgreSQL."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.db.models import KnowledgeDocument, PayloadReference, SystemVersions
from app.db.repositories import CatalogRepository
from app.db.session import Database, event_loop_factory
from app.knowledge.ingestion import IngestionService
from app.knowledge.retrieval import RetrievalService
from app.security import Principal
from app.tools.contracts import ToolCard
from app.tools.registry import ToolRegistry


def test_registry_lifecycle_and_scoped_catalog(config):
    config.embedding_model = "synthetic"
    config.embedding_version = "synthetic-v1"
    admin = Principal(uuid4(), "admin", frozenset({"tools:manage", "system:read"}))
    reader = Principal(admin.tenant_id, "reader", frozenset({"system:read"}))
    other = Principal(uuid4(), "other", reader.scopes)
    path = Path(__file__).resolve().parents[2] / "resources/tool_cards/system_search.json"
    card = ToolCard.model_validate(json.loads(path.read_text())).model_copy(
        update={
            "embedding_model": config.embedding_model,
            "embedding_version": config.embedding_version,
        }
    )
    vector = [1.0] + [0.0] * (config.embedding_dimension - 1)

    async def check():
        database = Database(config)
        try:
            await database.open()
            registry = ToolRegistry(
                {"system_search": object()},
                database=database,
                verification_keys={"system_search_result"},
            )
            await registry.save_draft(card, admin, vector)
            with pytest.raises(HTTPException) as draft:
                await registry.get(card.tool_id, "1", reader)
            assert draft.value.status_code == 404
            await registry.publish(card, admin)
            await registry.publish(card, admin)
            with pytest.raises(HTTPException) as duplicate:
                await registry.save_draft(card, admin, vector)
            assert duplicate.value.status_code == 409
            with pytest.raises(HTTPException) as denied:
                await registry.revoke(card.tool_id, "1", reader)
            assert denied.value.status_code == 403
            with pytest.raises(HTTPException) as cross_tenant:
                await registry.get(card.tool_id, "1", other)
            assert cross_tenant.value.status_code == 404
            with pytest.raises(HTTPException) as no_scope:
                await registry.get(
                    card.tool_id, "1", Principal(admin.tenant_id, "none", frozenset())
                )
            assert no_scope.value.status_code == 403
            async with database.session() as session:
                catalog = CatalogRepository(session, allowed_adapters={"system_search"})
                assert len(await catalog.eligible_tools(reader)) == 1
                assert await catalog.eligible_tools(other) == []
                assert (
                    await catalog.eligible_tools(Principal(admin.tenant_id, "none", frozenset()))
                    == []
                )
                assert (
                    await CatalogRepository(session, allowed_adapters=set()).eligible_tools(reader)
                    == []
                )
            newer = card.model_copy(update={"schema_version": "2"})
            await registry.save_draft(newer, admin, vector)
            await registry.publish(newer, admin)
            await registry.publish(card, admin)  # Replay must not restore old discovery version.
            assert (await registry.get(card.tool_id, "1", reader)).schema_version == "1"
            async with database.session() as session:
                cards = await CatalogRepository(
                    session, allowed_adapters={"system_search"}
                ).eligible_tools(reader)
                assert [item["schema_version"] for item in cards] == ["2"]
            await registry.revoke(card.tool_id, "1", admin)
            await registry.revoke(card.tool_id, "1", admin)
            with pytest.raises(HTTPException):
                await registry.get(card.tool_id, "1", reader)
            with pytest.raises(HTTPException):
                await registry.publish(card, admin)
            async with database.session() as session:
                revision = await session.scalar(
                    select(SystemVersions).where(SystemVersions.tenant_id == admin.tenant_id)
                )
                assert revision.registry_revision == 3
        finally:
            await database.close()

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        runner.run(check())


def test_tool_import_validates_before_embedding_and_creates_only_drafts(config, tmp_path):
    config.embedding_model = "synthetic"
    config.embedding_version = "synthetic-v1"
    admin = Principal(uuid4(), "admin", frozenset({"tools:manage", "system:read"}))
    source = Path(__file__).resolve().parents[2] / "resources/tool_cards/system_search.json"
    payload = json.loads(source.read_text())
    imported = tmp_path / "tool.json"
    vector = [1.0] + [0.0] * (config.embedding_dimension - 1)
    provider = SimpleNamespace(
        model="synthetic",
        version="synthetic-v1",
        dimension=config.embedding_dimension,
        embed=AsyncMock(return_value=vector),
    )

    async def check():
        database = Database(config)
        try:
            await database.open()
            registry = ToolRegistry(
                {"system_search"}, database=database, verification_keys={"system_search_result"}
            )
            service = IngestionService(
                database, None, RetrievalService(provider), config, registry=registry
            )
            imported.write_text(json.dumps({**payload, "adapter_key": "unapproved"}))
            with pytest.raises(ValueError):
                await service.import_tool(imported, admin)
            provider.embed.assert_not_awaited()
            imported.write_text(json.dumps(payload))
            card = await service.import_tool(imported, admin)
            assert card.status == "draft"
            with pytest.raises(HTTPException):
                await registry.get(card.tool_id, card.schema_version, admin)
            await registry.publish(card, admin)
            assert (
                await registry.get(card.tool_id, card.schema_version, admin)
            ).status == "published"
            with pytest.raises(HTTPException) as duplicate:
                await service.import_tool(imported, admin)
            assert duplicate.value.status_code == 409
            await registry.revoke(card.tool_id, card.schema_version, admin)
            with pytest.raises(HTTPException):
                await registry.get(card.tool_id, card.schema_version, admin)
        finally:
            await database.close()

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        runner.run(check())


def test_document_acl_corpus_and_effective_dates(config):
    principal = Principal(uuid4(), "reader", frozenset({"knowledge:read"}))
    now = datetime.now(UTC)

    async def check():
        database = Database(config)
        try:
            await database.open()
            async with database.session() as session, session.begin():
                for label, tenant, scopes, active, start, end in [
                    ("visible", principal.tenant_id, ["knowledge:read"], True, None, None),
                    ("denied", principal.tenant_id, ["restricted"], True, None, None),
                    ("inactive", principal.tenant_id, [], False, None, None),
                    ("future", principal.tenant_id, [], True, now + timedelta(days=1), None),
                    ("expired", principal.tenant_id, [], True, None, now - timedelta(days=1)),
                    ("other", uuid4(), [], True, None, None),
                ]:
                    payload_id = uuid4()
                    session.add(
                        PayloadReference(
                            payload_id=payload_id,
                            tenant_id=tenant,
                            user_id="reader",
                            object_key=str(payload_id),
                            content_type="text/plain",
                            size=1,
                            checksum="synthetic",
                        )
                    )
                    await session.flush()
                    session.add(
                        KnowledgeDocument(
                            tenant_id=tenant,
                            corpus_id="test",
                            source_key=label,
                            version="1",
                            payload_ref=payload_id,
                            title=label,
                            required_scopes=scopes,
                            checksum="synthetic",
                            active=active,
                            effective_from=start,
                            effective_until=end,
                        )
                    )
            async with database.session() as session:
                catalog = CatalogRepository(session, allowed_adapters=set())
                assert [doc.title for doc in await catalog.active_documents(principal, [])] == [
                    "visible"
                ]
                assert await catalog.active_documents(principal, ["different"]) == []
        finally:
            await database.close()

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        runner.run(check())
