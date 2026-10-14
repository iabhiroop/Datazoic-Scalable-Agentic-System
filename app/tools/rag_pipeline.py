"""Read-only RAG tool. Reuses retrieval without recursively invoking the agent graph."""

from typing import TYPE_CHECKING, Any, Literal

from fastapi import HTTPException
from pydantic import Field, ValidationError

from app.agent.gateway import LLMError
from app.agent.llm import ModelClient
from app.observability import Telemetry
from app.security import Principal, TokenAuthenticator
from app.tools.contracts import Contract, ToolContext, ToolOutcome

if TYPE_CHECKING:
    from app.knowledge.context import ContextBuilder
    from app.knowledge.retrieval import RetrievalService


class RagArguments(Contract):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["context", "answer"] = "context"
    corpus_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=6)


class Passage(Contract):
    chunk_id: str
    document_id: str
    document_version: str
    title: str
    locator: str
    text: str
    score: float


class RagResult(Contract):
    query: str
    sufficient_evidence: bool
    answer: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    passages: list[Passage]
    knowledge_revision: int


class RagPipeline:
    async def authorize_result(self, arguments, data, context) -> None:
        principal = await self.principal(context)
        result = RagResult.model_validate(data)
        if result.knowledge_revision != await self.retrieval.knowledge_revision(
            principal
        ) or not await self.retrieval.validate_evidence(
            result.passages, principal, arguments.get("corpus_ids", [])
        ):
            raise HTTPException(403, "Knowledge evidence is no longer authorized/current")

    def __init__(
        self,
        retrieval: "RetrievalService",
        context_builder: "ContextBuilder",
        authenticator: TokenAuthenticator,
        model: ModelClient | None = None,
        *,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.retrieval, self.context_builder, self.authenticator, self.model = (
            retrieval,
            context_builder,
            authenticator,
            model,
        )
        self.telemetry = telemetry

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutcome:
        if self.telemetry is None:
            return await self._invoke(arguments, context)
        with self.telemetry.observe(
            "tool.rag",
            input=arguments,
            metadata={"run_id": str(context.run_id), "operation_id": str(context.operation_id)},
        ) as observation:
            result = await self._invoke(arguments, context)
            observation.update(output=result.data, metadata={"state": result.state})
            if result.error_code:
                observation.error(result.error_code)
            return result

    async def principal(self, context: ToolContext) -> Principal:
        principal = await self.authenticator.refresh(
            Principal(context.tenant_id, context.user_id, frozenset())
        )
        if "knowledge:read" not in principal.scopes:
            raise HTTPException(403, "Knowledge read permission is required")
        return principal

    async def _invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutcome:
        try:
            request = RagArguments.model_validate(arguments)
            if not request.query.strip():
                raise ValueError("Empty query")
            principal = await self.principal(context)
            revision = await self.retrieval.knowledge_revision(principal)
            passages = await self.retrieval.knowledge(
                request.query, request.corpus_ids, principal, request.top_k
            )
            bounded = await self.context_builder.build(
                [],
                passages,
                [],
            )
            passages = [Passage.model_validate(item) for item in bounded["passages"]]
            principal = await self.principal(context)
            if revision != await self.retrieval.knowledge_revision(
                principal
            ) or not await self.retrieval.validate_evidence(
                passages, principal, request.corpus_ids
            ):
                return ToolOutcome(state="failed", data={}, error_code="stale_evidence")
            answer = {
                "sufficient_evidence": bool(passages),
                "answer": None,
                "source_ids": [p.chunk_id for p in passages],
            }
            if request.mode == "answer" and passages:
                if self.model is None:
                    return ToolOutcome(state="failed", data={}, error_code="model_unavailable")
                answer = await self.model.answer(request.query, [p.model_dump() for p in passages])
            principal = await self.principal(context)
            if revision != await self.retrieval.knowledge_revision(
                principal
            ) or not await self.retrieval.validate_evidence(
                passages, principal, request.corpus_ids
            ):
                return ToolOutcome(state="failed", data={}, error_code="stale_evidence")
            result = RagResult(
                query=request.query, passages=passages, knowledge_revision=revision, **answer
            )
            return ToolOutcome(state="succeeded", data=result.model_dump(mode="json"))
        except HTTPException:
            return ToolOutcome(
                state="failed", data={}, error_code="forbidden", effect_known_absent=True
            )
        except (ValidationError, ValueError):
            return ToolOutcome(
                state="failed",
                data={},
                error_code="invalid_input_or_context",
                effect_known_absent=True,
            )
        except LLMError as error:
            return ToolOutcome(
                state="failed",
                data={},
                error_code=error.code,
                retryable=error.retryable,
                effect_known_absent=True,
            )

    async def reconcile(self, operation: dict[str, Any], context: ToolContext) -> ToolOutcome:
        return ToolOutcome(
            state="failed",
            data={},
            error_code="reconciliation_unsupported",
            effect_known_absent=True,
        )
