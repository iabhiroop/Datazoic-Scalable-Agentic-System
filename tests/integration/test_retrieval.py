"""Both search branches apply the same access and version filters before ranking."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import update

from app.db.models import KnowledgeChunk, KnowledgeDocument, PayloadReference, RegistryTool
from app.db.session import Database, event_loop_factory
from app.knowledge.retrieval import RetrievalService
from app.security import Principal
from app.tools.contracts import ToolCard


class Embeddings:
    model = "synthetic"
    version = "synthetic-v1"

    def __init__(self, dimension):
        self.dimension = dimension

    async def embed(self, text, **kwargs):
        return [1.0] + [0.0] * (self.dimension - 1)


def test_hybrid_access_versions_corpora_and_stale_evidence(config):
    principal = Principal(uuid4(), "reader", frozenset({"knowledge:read", "system:read"}))
    provider = Embeddings(config.embedding_dimension)
    now = datetime.now(UTC)
    path = Path(__file__).resolve().parents[2] / "resources/tool_cards/system_search.json"
    card = ToolCard.model_validate(json.loads(path.read_text())).model_copy(
        update={
            "embedding_model": "synthetic",
            "embedding_version": "synthetic-v1",
            "status": "published",
        }
    )

    async def check():
        db = Database(config)
        try:
            await db.open()
            async with db.session() as session, session.begin():
                for label, tenant, scopes, active, start, end, corpus, version in [
                    ("visible", principal.tenant_id, [], True, None, None, "test", "synthetic-v1"),
                    (
                        "restricted",
                        principal.tenant_id,
                        ["restricted"],
                        True,
                        None,
                        None,
                        "test",
                        "synthetic-v1",
                    ),
                    (
                        "inactive",
                        principal.tenant_id,
                        [],
                        False,
                        None,
                        None,
                        "test",
                        "synthetic-v1",
                    ),
                    (
                        "future",
                        principal.tenant_id,
                        [],
                        True,
                        now + timedelta(days=1),
                        None,
                        "test",
                        "synthetic-v1",
                    ),
                    (
                        "expired",
                        principal.tenant_id,
                        [],
                        True,
                        None,
                        now - timedelta(days=1),
                        "test",
                        "synthetic-v1",
                    ),
                    ("other", uuid4(), [], True, None, None, "test", "synthetic-v1"),
                    (
                        "other-corpus",
                        principal.tenant_id,
                        [],
                        True,
                        None,
                        None,
                        "another",
                        "synthetic-v1",
                    ),
                    (
                        "incompatible",
                        principal.tenant_id,
                        [],
                        True,
                        None,
                        None,
                        "test",
                        "old-version",
                    ),
                ]:
                    payload, document = uuid4(), uuid4()
                    session.add(
                        PayloadReference(
                            payload_id=payload,
                            tenant_id=tenant,
                            user_id="admin",
                            object_key=str(payload),
                            content_type="text/plain",
                            size=1,
                            checksum="test",
                        )
                    )
                    await session.flush()
                    session.add(
                        KnowledgeDocument(
                            document_id=document,
                            tenant_id=tenant,
                            corpus_id=corpus,
                            source_key=label,
                            version="1",
                            payload_ref=payload,
                            title=label,
                            required_scopes=scopes,
                            checksum="test",
                            active=active,
                            effective_from=start,
                            effective_until=end,
                        )
                    )
                    await session.flush()
                    session.add(
                        KnowledgeChunk(
                            tenant_id=tenant,
                            document_id=document,
                            chunk_index=0,
                            text="Returns have an example limit.",
                            embedding=await provider.embed(""),
                            embedding_version=version,
                            locator="lines:1-1",
                        )
                    )
                for label, tenant, scopes, current, status, adapter in [
                    (
                        "visible",
                        principal.tenant_id,
                        ["system:read"],
                        True,
                        "published",
                        "system_search",
                    ),
                    (
                        "restricted",
                        principal.tenant_id,
                        ["restricted"],
                        True,
                        "published",
                        "system_search",
                    ),
                    ("other", uuid4(), [], True, "published", "system_search"),
                    ("revoked", principal.tenant_id, [], False, "revoked", "system_search"),
                    ("older", principal.tenant_id, [], False, "published", "system_search"),
                    ("unconfigured", principal.tenant_id, [], True, "published", "unconfigured"),
                ]:
                    record = card.model_copy(
                        update={
                            "tool_id": label,
                            "required_scopes": scopes,
                            "adapter_key": adapter,
                            "status": status,
                        }
                    )
                    session.add(
                        RegistryTool(
                            tenant_id=tenant,
                            tool_id=label,
                            schema_version="1",
                            card=record.model_dump(mode="json"),
                            searchable_text="request search",
                            embedding=await provider.embed(""),
                            embedding_version=provider.version,
                            is_current=current,
                            status=status,
                        )
                    )
            retrieval = RetrievalService(
                provider, database=db, settings=config, allowed_adapters={"system_search"}
            )
            passages = await retrieval.knowledge("returns", ["test"], principal, 4)
            assert [p.title for p in passages] == ["visible"]
            assert await retrieval.knowledge("returns", ["missing"], principal, 4) == []
            assert await retrieval.validate_evidence(passages, principal, ["test"])
            cards = await retrieval.tools("request", [], principal, 5)
            assert [c.tool_id for c in cards] == ["visible"]
            async with db.session() as session, session.begin():
                await session.execute(
                    update(KnowledgeDocument)
                    .where(
                        KnowledgeDocument.tenant_id == principal.tenant_id,
                        KnowledgeDocument.source_key == "visible",
                    )
                    .values(active=False)
                )
            assert not await retrieval.validate_evidence(passages, principal, ["test"])
        finally:
            await db.close()

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        runner.run(check())
