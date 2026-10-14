"""Administrative import flows: failed publication must preserve the previous active version."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.db.models import KnowledgeChunk, KnowledgeDocument, SystemVersions
from app.db.session import Database
from app.knowledge.retrieval import RetrievalService
from app.observability import SecretRedactor
from app.security import Principal
from app.settings import Settings
from app.storage import PayloadStore
from app.tools.contracts import ToolCard
from app.tools.registry import ToolRegistry


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    locator: str


class IngestionService:
    def __init__(
        self,
        database: Database,
        store: PayloadStore,
        retrieval: RetrievalService,
        settings: Settings,
        *,
        registry: ToolRegistry | None = None,
    ) -> None:
        if settings.document_chunk_overlap >= settings.document_chunk_chars:
            raise ValueError("Chunk overlap must be smaller than the chunk size")
        provider = retrieval.embedding_provider
        if (provider.model, provider.version, provider.dimension) != (
            settings.embedding_model,
            settings.embedding_version,
            settings.embedding_dimension,
        ):
            raise ValueError("Embedding provider does not match the document configuration")
        self.database, self.store, self.retrieval, self.settings = (
            database,
            store,
            retrieval,
            settings,
        )
        self.registry = registry

    async def import_tool(
        self, path: Path, principal: Principal, *, draft_only: bool = False
    ) -> ToolCard:
        if "tools:manage" not in principal.scopes:
            raise HTTPException(403, "Tool management permission is required")
        if self.registry is None:
            raise ValueError("Tool registry is not configured")
        if path.suffix.lower() != ".json":
            raise ValueError("Tool cards must be JSON files")
        with path.open("rb") as source:
            data = source.read(self.settings.document_max_bytes + 1)
        if len(data) > self.settings.document_max_bytes:
            raise ValueError("Tool card exceeds the configured import size limit")
        card = ToolCard.model_validate_json(data).model_copy(
            update={
                "status": "draft",
                "embedding_model": self.settings.embedding_model,
                "embedding_version": self.settings.embedding_version,
            }
        )
        serialized = card.model_dump_json()
        if any(secret in serialized for secret in SecretRedactor(self.settings).secrets):
            raise ValueError("Tool card contains configured credentials")
        card = (
            self.registry.validate_draft(card)
            if draft_only
            else await self.registry.validate_import(card)
        )
        description = "\n".join(
            [card.name, card.description, *card.synonyms, *card.entities, card.action]
        )
        vector = await self.retrieval.embed(description, task="document", title=card.name)
        await self.registry.save_draft(card, principal, vector, draft_only=draft_only)
        return card

    async def import_document(
        self,
        path: Path,
        corpus_id: str,
        principal: Principal,
        *,
        source_key: str | None = None,
        required_scopes: list[str] | None = None,
    ) -> str:
        if "knowledge:manage" not in principal.scopes:
            raise HTTPException(403, "Document management permission is required")
        if not corpus_id.strip() or (source_key is not None and not source_key.strip()):
            raise ValueError("Corpus and source identities must not be empty")
        data = self.read_source(path)
        text = self.decode(data)
        chunks = self.chunk(text)
        if len(chunks) > self.settings.document_max_chunks:
            raise ValueError("Document exceeds the configured chunk limit")
        if required_scopes is not None and any(not scope.strip() for scope in required_scopes):
            raise ValueError("Document permission names must not be empty")
        title = next(
            (line.lstrip("#").strip() for line in text.splitlines() if line.startswith("#")),
            path.name,
        )
        source = source_key or path.name
        version = hashlib.sha256(
            data + self.retrieval.embedding_provider.version.encode()
        ).hexdigest()
        async with self.database.session() as session:
            existing = await session.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.tenant_id == principal.tenant_id,
                    KnowledgeDocument.corpus_id == corpus_id,
                    KnowledgeDocument.source_key == source,
                    KnowledgeDocument.version == version,
                )
            )
            if existing is not None and existing.active:
                if required_scopes is not None and set(required_scopes) != set(
                    existing.required_scopes
                ):
                    raise HTTPException(409, "Existing version has different document permissions")
                return str(existing.document_id)
        vectors = []
        for chunk in chunks:
            vector = await self.retrieval.embed(chunk.text, task="document", title=title)
            if (
                len(vector) != self.settings.embedding_dimension
                or not all(math.isfinite(value) for value in vector)
                or not any(vector)
            ):
                raise ValueError("Document embedding failed vector validation")
            vectors.append(vector)
        # No DB transaction remains open during embedding or object-storage requests.
        payload = await self.store.upload(
            data, "text/markdown" if path.suffix.lower() == ".md" else "text/plain", None, principal
        )
        async with self.database.session() as session, session.begin():
            await session.execute(
                insert(SystemVersions)
                .values(tenant_id=principal.tenant_id)
                .on_conflict_do_nothing()
            )
            revisions = await session.scalar(
                select(SystemVersions)
                .where(SystemVersions.tenant_id == principal.tenant_id)
                .with_for_update()
            )
            current = await session.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.tenant_id == principal.tenant_id,
                    KnowledgeDocument.corpus_id == corpus_id,
                    KnowledgeDocument.source_key == source,
                    KnowledgeDocument.active.is_(True),
                )
            )
            existing = await session.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.tenant_id == principal.tenant_id,
                    KnowledgeDocument.corpus_id == corpus_id,
                    KnowledgeDocument.source_key == source,
                    KnowledgeDocument.version == version,
                )
            )
            scopes = (
                required_scopes
                if required_scopes is not None
                else (current.required_scopes if current is not None else ["knowledge:read"])
            )
            if existing is not None and set(existing.required_scopes) != set(scopes):
                raise HTTPException(409, "Existing version has different document permissions")
            if existing is not None and existing.active:
                return str(existing.document_id)
            await session.execute(
                update(KnowledgeDocument)
                .where(
                    KnowledgeDocument.tenant_id == principal.tenant_id,
                    KnowledgeDocument.corpus_id == corpus_id,
                    KnowledgeDocument.source_key == source,
                    KnowledgeDocument.active.is_(True),
                )
                .values(active=False)
            )
            if existing is None:
                existing = KnowledgeDocument(
                    document_id=uuid4(),
                    tenant_id=principal.tenant_id,
                    corpus_id=corpus_id,
                    source_key=source,
                    version=version,
                    payload_ref=payload.payload_id,
                    title=title,
                    required_scopes=scopes,
                    checksum=hashlib.sha256(data).hexdigest(),
                    active=True,
                    effective_from=current.effective_from if current else None,
                    effective_until=current.effective_until if current else None,
                )
                session.add(existing)
                await session.flush()
                session.add_all(
                    [
                        KnowledgeChunk(
                            tenant_id=principal.tenant_id,
                            document_id=existing.document_id,
                            chunk_index=chunk.index,
                            text=chunk.text,
                            embedding=vector,
                            embedding_version=self.retrieval.embedding_provider.version,
                            locator=chunk.locator,
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ]
                )
            else:
                existing.active = True
            revisions.knowledge_revision += 1
            await session.flush()
            identity = str(existing.document_id)
        return identity

    def read_source(self, path: Path) -> bytes:
        if path.suffix.lower() not in (".md", ".txt"):
            raise ValueError("Only Markdown and plain text documents are supported")
        with path.open("rb") as source:
            data = source.read(self.settings.document_max_bytes + 1)
        if not data or len(data) > self.settings.document_max_bytes:
            raise ValueError("Document is empty or exceeds the configured size limit")
        return data

    @staticmethod
    def decode(data: bytes) -> str:
        try:
            text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeError:
            raise ValueError("Documents must be UTF-8 encoded") from None
        if not text.strip() or "\x00" in text:
            raise ValueError("Document must contain nonempty text without NUL bytes")
        return text

    def extract(self, path: Path) -> str:
        return self.decode(self.read_source(path))

    def chunk(self, text: str) -> list[Chunk]:
        if not text.strip():
            raise ValueError("Document text must not be empty")
        chunks = []
        start = 0
        size, overlap = self.settings.document_chunk_chars, self.settings.document_chunk_overlap
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind("\n", start + size // 2, end)
                if boundary != -1:
                    end = boundary + 1
            if end - start <= overlap and end < len(text):
                end = min(start + size, len(text))
            value = text[start:end]
            if value.strip():
                first = text.count("\n", 0, start) + 1
                last = text.count("\n", 0, end - 1) + 1
                chunks.append(Chunk(len(chunks), value, f"lines:{first}-{last}"))
            if end == len(text):
                break
            start = end - overlap
        return chunks
