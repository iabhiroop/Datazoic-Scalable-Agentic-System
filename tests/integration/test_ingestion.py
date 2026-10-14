"""Atomic document activation and authorized payload access with a fake S3 transport."""

import asyncio
import io
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.db.models import KnowledgeChunk, KnowledgeDocument, SystemVersions
from app.db.session import Database, event_loop_factory
from app.knowledge.ingestion import IngestionService
from app.knowledge.retrieval import RetrievalService
from app.security import Principal
from app.storage import PayloadStore, StorageError


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.fail = False
        self.reads = 0

    def put_object(self, **kwargs):
        if self.fail:
            raise StorageError("Synthetic storage failure")
        assert kwargs["IfNoneMatch"] == "*"
        assert kwargs["Key"] not in self.objects
        self.objects[kwargs["Key"]] = kwargs

    def head_object(self, **kwargs):
        stored = self.objects[kwargs["Key"]]
        return {"ContentLength": len(stored["Body"]), "Metadata": stored["Metadata"]}

    def get_object(self, **kwargs):
        self.reads += 1
        return {"Body": io.BytesIO(self.objects[kwargs["Key"]]["Body"])}

    def generate_presigned_url(self, **kwargs):
        assert kwargs["ExpiresIn"] == 300
        return "https://synthetic.invalid/scoped"

    def close(self):
        pass


class Embeddings:
    def __init__(self, dimension):
        self.model = "synthetic"
        self.version = "synthetic-v1"
        self.dimension = dimension
        self.calls = 0
        self.fail = False

    async def embed(self, text, **kwargs):
        assert kwargs["task"] == "document"
        self.calls += 1
        if self.fail:
            raise ValueError("Synthetic embedding failure")
        return [1.0] + [0.0] * (self.dimension - 1)


def test_document_activation_failures_idempotency_acl_and_payloads(config, tmp_path):
    config.embedding_model = "synthetic"
    config.embedding_version = "synthetic-v1"
    principal = Principal(uuid4(), "admin", frozenset({"knowledge:manage"}))
    reader = Principal(principal.tenant_id, "reader", frozenset({"knowledge:read", "restricted"}))
    denied = Principal(principal.tenant_id, "denied", frozenset({"knowledge:read"}))
    source = tmp_path / "policy.md"
    source.write_text("# Policy\nReturns: 14 days.\n", encoding="utf-8")

    async def check():
        database = Database(config)
        memory = MemoryS3()
        store = PayloadStore(database, config, client=memory)
        embeddings = Embeddings(config.embedding_dimension)
        ingestion = IngestionService(database, store, RetrievalService(embeddings), config)
        try:
            await database.open()
            first = await ingestion.import_document(
                source, "test", principal, required_scopes=["restricted"]
            )
            calls = embeddings.calls
            assert await ingestion.import_document(source, "test", principal) == first
            assert embeddings.calls == calls and len(memory.objects) == 1
            source.write_text("# Policy\nReturns: 30 days.\n", encoding="utf-8")
            embeddings.fail = True
            with pytest.raises(ValueError):
                await ingestion.import_document(source, "test", principal)
            embeddings.fail = False
            memory.fail = True
            with pytest.raises(StorageError):
                await ingestion.import_document(source, "test", principal)
            async with database.session() as session:
                active = await session.scalar(
                    select(KnowledgeDocument).where(
                        KnowledgeDocument.tenant_id == principal.tenant_id,
                        KnowledgeDocument.active.is_(True),
                    )
                )
                assert active.document_id == UUID(first)
            memory.fail = False
            second = await ingestion.import_document(source, "test", principal)
            assert second != first
            async with database.session() as session:
                active = await session.scalar(
                    select(KnowledgeDocument).where(
                        KnowledgeDocument.tenant_id == principal.tenant_id,
                        KnowledgeDocument.active.is_(True),
                    )
                )
                assert active.document_id == UUID(second)
                assert active.required_scopes == ["restricted"]
                chunks = list(
                    await session.scalars(
                        select(KnowledgeChunk).where(
                            KnowledgeChunk.document_id == active.document_id
                        )
                    )
                )
                assert "30 days" in "".join(chunk.text for chunk in chunks)
                revision = await session.get(SystemVersions, principal.tenant_id)
                assert revision.knowledge_revision == 2
                payload = active.payload_ref
            for who in (denied, Principal(uuid4(), "other", reader.scopes)):
                reads = memory.reads
                with pytest.raises(HTTPException):
                    await store.retrieve(payload, who)
                assert memory.reads == reads
            assert await store.retrieve(payload, reader) == source.read_bytes()
            assert await store.signed_url(payload, reader) == "https://synthetic.invalid/scoped"
            for stored in memory.objects.values():
                if stored["Body"] == source.read_bytes():
                    stored["Body"] = b"corrupted"
            with pytest.raises(StorageError):
                await store.retrieve(payload, reader)
        finally:
            await store.aclose()
            await database.close()

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        runner.run(check())
