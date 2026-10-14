"""Authorized lexical/vector retrieval; start with exact search and rank fusion."""

import json
import math
from typing import Any, Protocol

from sqlalchemy import func, select

from app.db.models import KnowledgeChunk, KnowledgeDocument, RegistryTool, SystemVersions
from app.db.repositories import CatalogRepository
from app.db.session import Database
from app.security import Principal
from app.settings import Settings
from app.tools.contracts import ToolCard
from app.tools.rag_pipeline import Passage


class EmbeddingProvider(Protocol):
    model: str
    version: str
    dimension: int

    async def embed(
        self, text: str, *, task: str = "query", title: str | None = None
    ) -> list[float]: ...

    async def aclose(self) -> None: ...


class RetrievalService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        database: Database | None = None,
        settings: Settings | None = None,
        allowed_adapters: set[str] | None = None,
    ) -> None:
        # The process owns this provider's lifecycle; alternative providers can be injected.
        self.embedding_provider = embedding_provider
        self.database = database
        self.settings = settings or Settings(_env_file=None)
        self.allowed_adapters = set(allowed_adapters or ())

    def require_database(self) -> Database:
        if self.database is None:
            raise ValueError("Retrieval database is not configured")
        return self.database

    async def query_vector(self, query: str, limit: int) -> list[float]:
        if not query.strip() or not 1 <= limit <= self.settings.retrieval_branch_limit:
            raise ValueError("Invalid retrieval query or limit")
        vector = await self.embed(query)
        if (
            len(vector) != self.embedding_provider.dimension
            or not all(math.isfinite(value) for value in vector)
            or not any(vector)
        ):
            raise ValueError("Query embedding is invalid")
        return vector

    async def tools(
        self, query: str, category_ids: list[str], principal: Principal, limit: int
    ) -> list[ToolCard]:
        database = self.require_database()
        vector = await self.query_vector(query, limit)
        async with database.session() as session:
            eligible = (
                CatalogRepository(session, allowed_adapters=self.allowed_adapters)
                .tool_query(principal)
                .where(
                    RegistryTool.embedding_version == self.embedding_provider.version,
                    RegistryTool.card["embedding_model"].astext == self.embedding_provider.model,
                )
            )
            search = func.websearch_to_tsquery("simple", query)
            lexical = list(
                await session.scalars(
                    eligible.where(RegistryTool.search_vector.bool_op("@@")(search))
                    .order_by(
                        func.ts_rank_cd(RegistryTool.search_vector, search).desc(),
                        RegistryTool.tool_id,
                    )
                    .limit(self.settings.retrieval_branch_limit)
                )
            )
            semantic = list(
                await session.scalars(
                    eligible.where(RegistryTool.embedding.is_not(None))
                    .order_by(RegistryTool.embedding.cosine_distance(vector), RegistryTool.tool_id)
                    .limit(self.settings.retrieval_branch_limit)
                )
            )

            def identity(row):
                return json.dumps([row.tool_id, row.schema_version])

            rows = {identity(row): row for row in lexical + semantic}
            fused = fuse_ranks(
                [identity(row) for row in lexical],
                [identity(row) for row in semantic],
            )
            categories = set(category_ids)
            ranked = sorted(
                fused,
                key=lambda item: (
                    -(
                        item[1]
                        + (
                            1 / 6000
                            if categories.intersection(rows[item[0]].card.get("category_ids", []))
                            else 0
                        )
                    ),
                    item[0],
                ),
            )
            return [
                ToolCard.model_validate(rows[identity].card).model_copy(
                    update={"status": "published"}
                )
                for identity, _ in ranked[:limit]
            ]

    async def knowledge(
        self, query: str, corpus_ids: list[str], principal: Principal, limit: int
    ) -> list[Passage]:
        database = self.require_database()
        vector = await self.query_vector(query, limit)
        async with database.session() as session:
            documents = (
                CatalogRepository(session, allowed_adapters=set())
                .document_query(principal, corpus_ids)
                .with_only_columns(KnowledgeDocument.document_id)
            )
            eligible = (
                select(KnowledgeChunk, KnowledgeDocument)
                .join(
                    KnowledgeDocument,
                    (KnowledgeDocument.tenant_id == KnowledgeChunk.tenant_id)
                    & (KnowledgeDocument.document_id == KnowledgeChunk.document_id),
                )
                .where(
                    KnowledgeChunk.tenant_id == principal.tenant_id,
                    KnowledgeChunk.document_id.in_(documents),
                    KnowledgeChunk.embedding_version == self.embedding_provider.version,
                )
            )
            search = func.websearch_to_tsquery("simple", query)
            lexical = list(
                (
                    await session.execute(
                        eligible.where(KnowledgeChunk.search_vector.bool_op("@@")(search))
                        .order_by(
                            func.ts_rank_cd(KnowledgeChunk.search_vector, search).desc(),
                            KnowledgeChunk.chunk_id,
                        )
                        .limit(self.settings.retrieval_branch_limit)
                    )
                ).all()
            )
            semantic = list(
                (
                    await session.execute(
                        eligible.order_by(
                            KnowledgeChunk.embedding.cosine_distance(vector),
                            KnowledgeChunk.chunk_id,
                        ).limit(self.settings.retrieval_branch_limit)
                    )
                ).all()
            )
            rows = {
                str(chunk.chunk_id): (chunk, document) for chunk, document in lexical + semantic
            }
            fused = fuse_ranks(
                [str(chunk.chunk_id) for chunk, _ in lexical],
                [str(chunk.chunk_id) for chunk, _ in semantic],
            )
            result = []
            for identity, score in fused[:limit]:
                chunk, document = rows[identity]
                result.append(
                    Passage(
                        chunk_id=identity,
                        document_id=str(document.document_id),
                        document_version=document.version,
                        title=document.title,
                        locator=chunk.locator,
                        text=chunk.text,
                        score=score,
                    )
                )
            return result

    async def knowledge_revision(self, principal: Principal) -> int:
        async with self.require_database().session() as session:
            value = await session.scalar(
                select(SystemVersions.knowledge_revision).where(
                    SystemVersions.tenant_id == principal.tenant_id
                )
            )
            return value or 0

    async def validate_evidence(
        self, passages: list[Passage], principal: Principal, corpus_ids: list[str]
    ) -> bool:
        async with self.require_database().session() as session:
            documents = await CatalogRepository(session, allowed_adapters=set()).active_documents(
                principal, corpus_ids
            )
            allowed = {(str(doc.document_id), doc.version) for doc in documents}
            return all(
                (passage.document_id, passage.document_version) in allowed for passage in passages
            )

    async def trajectories(
        self, goal: str, versions: dict[str, str], principal: Principal, limit: int
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Reviewed compatible summaries; no old IDs/approvals reused")

    async def embed(
        self, text: str, *, task: str = "query", title: str | None = None
    ) -> list[float]:
        return await self.embedding_provider.embed(text, task=task, title=title)


def fuse_ranks(lexical: list[str], vector: list[str]) -> list[tuple[str, float]]:
    """Equal-weight reciprocal-rank fusion; stable identity tie-breaks, branch deduplication."""
    scores: dict[str, float] = {}
    for branch in (lexical, vector):
        for rank, identity in enumerate(dict.fromkeys(branch), start=1):
            scores[identity] = scores.get(identity, 0.0) + 1 / (60 + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
