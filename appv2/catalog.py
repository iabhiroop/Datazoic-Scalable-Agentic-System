"""Semantic lookup over the existing published tool and document tables."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.gemini import GeminiEmbeddingProvider
from app.db.models import KnowledgeChunk, KnowledgeDocument, RegistryTool, SystemVersions
from app.db.repositories import CatalogRepository
from app.security import Principal
from app.settings import Settings
from app.tools.contracts import ToolCard


class Catalog:
    def __init__(self, settings: Settings, allowed_adapters: set[str]):
        if settings.database_url is None:
            raise ValueError("DATABASE_URL is required")
        self.settings = settings
        self.embeddings = GeminiEmbeddingProvider(settings)
        self.engine = create_async_engine(
            settings.database_url.get_secret_value(), pool_pre_ping=True
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.allowed_adapters = allowed_adapters

    async def close(self) -> None:
        await self.embeddings.aclose()
        await self.engine.dispose()

    async def tools(self, query: str, principal: Principal, limit: int = 5) -> list[ToolCard]:
        vector = await self.embeddings.embed(query, task="query")
        async with self.sessions() as session:
            eligible = (
                CatalogRepository(session, allowed_adapters=self.allowed_adapters)
                .tool_query(principal)
                .where(
                    RegistryTool.embedding_version == self.embeddings.version,
                    RegistryTool.card["embedding_model"].astext == self.embeddings.model,
                    RegistryTool.embedding.is_not(None),
                )
            )
            rows = await session.scalars(
                eligible.order_by(
                    RegistryTool.embedding.cosine_distance(vector), RegistryTool.tool_id
                ).limit(limit)
            )
            return [
                ToolCard.model_validate(row.card).model_copy(update={"status": row.status})
                for row in rows
            ]

    async def documents(
        self,
        query: str,
        principal: Principal,
        *,
        corpus_ids: list[str] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        vector = await self.embeddings.embed(query, task="query")
        async with self.sessions() as session:
            eligible_documents = CatalogRepository(session, allowed_adapters=set()).document_query(
                principal, corpus_ids or []
            )
            rows = await session.execute(
                select(KnowledgeChunk, KnowledgeDocument)
                .join(
                    KnowledgeDocument,
                    (KnowledgeDocument.tenant_id == KnowledgeChunk.tenant_id)
                    & (KnowledgeDocument.document_id == KnowledgeChunk.document_id),
                )
                .where(
                    KnowledgeChunk.tenant_id == principal.tenant_id,
                    KnowledgeChunk.embedding_version == self.embeddings.version,
                    KnowledgeDocument.document_id.in_(
                        eligible_documents.with_only_columns(KnowledgeDocument.document_id)
                    ),
                )
                .order_by(KnowledgeChunk.embedding.cosine_distance(vector), KnowledgeChunk.chunk_id)
                .limit(limit)
            )
            return [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(document.document_id),
                    "document_version": document.version,
                    "title": document.title,
                    "locator": chunk.locator,
                    "text": chunk.text,
                }
                for chunk, document in rows
            ]

    async def knowledge_revision(self, principal: Principal) -> int:
        async with self.sessions() as session:
            revision = await session.scalar(
                select(SystemVersions.knowledge_revision).where(
                    SystemVersions.tenant_id == principal.tenant_id
                )
            )
            return revision or 0
