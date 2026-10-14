"""Scoped persistence stubs. All mutations must use short, atomic transactions."""

import base64
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import case, func, or_, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.checkpoints import wait_request_from_snapshot
from app.db.models import (
    KnowledgeDocument,
    RegistryTool,
    RunState,
    ToolOperation,
    WorkflowEvent,
    WorkflowInput,
    WorkflowRun,
)
from app.security import Principal, redact
from app.tools.contracts import ToolCard


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def scoped(principal: Principal):
        query = select(WorkflowRun).where(WorkflowRun.tenant_id == principal.tenant_id)
        if "runs:manage" not in principal.scopes:
            query = query.where(WorkflowRun.user_id == principal.user_id)
        return query

    async def get(self, run_id: UUID, principal: Principal) -> WorkflowRun:
        run = await self.session.scalar(self.scoped(principal).where(WorkflowRun.run_id == run_id))
        if run is None:
            raise HTTPException(404, "Run not found")
        return run

    async def by_submission(self, principal: Principal, key: str) -> WorkflowRun | None:
        return await self.session.scalar(
            select(WorkflowRun).where(
                WorkflowRun.tenant_id == principal.tenant_id,
                WorkflowRun.user_id == principal.user_id,
                WorkflowRun.submission_key == key,
            )
        )

    async def claim_next(self, worker_id: str) -> WorkflowRun | None:
        """Trusted singleton worker only; claim with SKIP LOCKED, commit before graph call."""
        run = await self.session.scalar(
            select(WorkflowRun)
            .where(
                WorkflowRun.status == "queued",
                WorkflowRun.cancel_requested.is_(False),
            )
            .order_by(WorkflowRun.created_at, WorkflowRun.run_id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if run is not None:
            from datetime import datetime

            run.status = "running"
            run.worker_id = worker_id
            run.heartbeat_at = datetime.now(UTC)
            run.row_version += 1
            await self.session.flush()
        # The caller owns the short transaction and must commit before invoking the graph.
        return run

    async def last(
        self, principal: Principal, conversation_id: UUID | None, exclude_run_id: UUID
    ) -> WorkflowRun | None:
        # "last" refers to the caller's history even when management access is present.
        query = select(WorkflowRun).where(
            WorkflowRun.tenant_id == principal.tenant_id,
            WorkflowRun.user_id == principal.user_id,
            WorkflowRun.run_id != exclude_run_id,
        )
        if conversation_id is not None:
            query = query.where(WorkflowRun.conversation_id == conversation_id)
        return await self.session.scalar(
            query.order_by(WorkflowRun.created_at.desc(), WorkflowRun.run_id.desc()).limit(1)
        )

    async def project_status(
        self, tenant_id: UUID, run_id: UUID, snapshot: Any, *, worker_id: str
    ) -> None:
        run = await self.session.scalar(
            select(WorkflowRun)
            .where(WorkflowRun.tenant_id == tenant_id, WorkflowRun.run_id == run_id)
            .with_for_update()
        )
        if run is None or run.worker_id != worker_id:
            raise ValueError("Run ownership changed")
        values = snapshot.values or {}
        final = values.get("final_result")
        if final and not snapshot.next:
            status = {
                "completed": RunState.COMPLETED,
                "failed": RunState.FAILED,
                "cancelled": RunState.CANCELLED,
                "needs_review": RunState.NEEDS_REVIEW,
            }.get(final.get("business_status"))
            if status is None:
                raise ValueError("Invalid terminal checkpoint status")
            if run.cancel_requested:
                from app.db.models import OperationState

                unresolved = await self.session.scalar(
                    select(ToolOperation.operation_id)
                    .where(
                        ToolOperation.tenant_id == tenant_id,
                        ToolOperation.run_id == run_id,
                        ToolOperation.status.in_(
                            [
                                OperationState.DISPATCHING,
                                OperationState.PENDING,
                                OperationState.UNKNOWN,
                            ]
                        ),
                    )
                    .limit(1)
                )
                if unresolved:
                    status = RunState.NEEDS_REVIEW
                    final = {**final, "business_status": "needs_review"}
                else:
                    status = RunState.CANCELLED
                    final = {
                        **final,
                        "business_status": "cancelled",
                        "summary": "Request cancelled.",
                    }
            run.status = status
            run.result_payload = {
                **final,
                "_llm_activity": values.get("llm_activity", []),
            }
            run.wait_token = run.wait_kind = run.due_at = None
        elif any(task.interrupts for task in snapshot.tasks):
            wait = wait_request_from_snapshot(snapshot)
            if wait is None or wait.kind not in {"clarification", "approval"}:
                raise ValueError("Unsupported wait policy")
            run.status = RunState.CANCELLED if run.cancel_requested else RunState.WAITING_FOR_INPUT
            run.wait_token = UUID(wait.wait_token)
            run.wait_kind = wait.kind
            run.result_payload = {
                "wait_request": wait.model_dump(mode="json"),
                "_llm_activity": values.get("llm_activity", []),
            }
        else:
            return
        run.row_version += 1
        self.session.add(
            WorkflowEvent(
                tenant_id=tenant_id,
                run_id=run_id,
                event_type="checkpoint_projected",
                evidence={"status": run.status.value},
            )
        )
        await self.session.flush()


class OperationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, tenant_id: UUID, operation_id: UUID) -> ToolOperation:
        operation = await self.session.scalar(
            select(ToolOperation).where(
                ToolOperation.tenant_id == tenant_id,
                ToolOperation.operation_id == operation_id,
            )
        )
        if operation is None:
            raise HTTPException(404, "Operation not found")
        return operation

    async def prepare(self, principal: Principal, operation: ToolOperation) -> ToolOperation:
        existing = await self.session.scalar(
            select(ToolOperation)
            .where(
                ToolOperation.tenant_id == principal.tenant_id,
                ToolOperation.run_id == operation.run_id,
                ToolOperation.action_ref == operation.action_ref,
            )
            .with_for_update()
        )
        if existing:
            if (
                existing.input_hash != operation.input_hash
                or existing.tool_id != operation.tool_id
                or existing.schema_version != operation.schema_version
            ):
                raise HTTPException(409, "Business action changed after preparation")
            return existing
        self.session.add(operation)
        await self.session.flush()
        self.session.add(
            WorkflowEvent(
                tenant_id=principal.tenant_id,
                run_id=operation.run_id,
                operation_id=operation.operation_id,
                event_type="write_prepared",
                evidence={"tool_id": operation.tool_id, "action_ref": operation.action_ref},
            )
        )
        return operation

    async def mark_dispatching(self, operation: ToolOperation, expected_version: int) -> bool:
        from app.db.models import OperationState

        changed = await self.session.execute(
            update(ToolOperation)
            .where(
                ToolOperation.tenant_id == operation.tenant_id,
                ToolOperation.operation_id == operation.operation_id,
                ToolOperation.status == OperationState.PREPARED,
                ToolOperation.row_version == expected_version,
            )
            .values(
                status=OperationState.DISPATCHING,
                attempt_count=ToolOperation.attempt_count + 1,
                row_version=ToolOperation.row_version + 1,
            )
        )
        return changed.rowcount == 1

    async def save_outcome(self, operation: ToolOperation, outcome: Any) -> None:
        from app.db.models import OperationState

        if operation.status != OperationState.DISPATCHING:
            raise ValueError("Only a dispatching operation can store a provider outcome")
        operation.status = OperationState(outcome.state)
        operation.provider_operation_id = outcome.provider_operation_id
        if outcome.state == "succeeded":
            operation.result_payload = outcome.data
            operation.verified_at = datetime.now(UTC)
        else:
            operation.error_payload = {"code": outcome.error_code or "provider_outcome_unknown"}
        operation.row_version += 1
        self.session.add(
            WorkflowEvent(
                tenant_id=operation.tenant_id,
                run_id=operation.run_id,
                operation_id=operation.operation_id,
                event_type="write_" + outcome.state,
                evidence={"tool_id": operation.tool_id, "state": outcome.state},
            )
        )
        await self.session.flush()


class InputRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def pending(
        self, tenant_id: UUID, run_id: UUID, wait_token: UUID
    ) -> WorkflowInput | None:
        return await self.session.scalar(
            select(WorkflowInput).where(
                WorkflowInput.tenant_id == tenant_id,
                WorkflowInput.run_id == run_id,
                WorkflowInput.wait_token == wait_token,
                WorkflowInput.status == "pending",
            )
        )

    async def acknowledge(self, tenant_id: UUID, input_id: UUID, *, run_id: UUID) -> None:
        await self.session.execute(
            update(WorkflowInput)
            .where(
                WorkflowInput.tenant_id == tenant_id,
                WorkflowInput.input_id == input_id,
                WorkflowInput.run_id == run_id,
                WorkflowInput.status == "pending",
            )
            .values(status="applied", applied_at=func.now())
        )


class CatalogRepository:
    def __init__(self, session: AsyncSession, *, allowed_adapters: set[str]) -> None:
        self.session = session
        # Environment eligibility is the server-configured adapter allowlist.
        self.allowed_adapters = frozenset(allowed_adapters)

    def tool_query(self, principal: Principal):
        return select(RegistryTool).where(
            RegistryTool.tenant_id == principal.tenant_id,
            RegistryTool.status == "published",
            RegistryTool.is_current.is_(True),
            RegistryTool.card["required_scopes"].contained_by(sorted(principal.scopes)),
            RegistryTool.card["adapter_key"].astext.in_(self.allowed_adapters),
        )

    async def eligible_tools(self, principal: Principal) -> list[dict[str, Any]]:
        rows = await self.session.scalars(
            self.tool_query(principal).order_by(RegistryTool.tool_id, RegistryTool.schema_version)
        )
        return [
            ToolCard.model_validate(row.card)
            .model_copy(update={"status": row.status})
            .model_dump(mode="json")
            for row in rows
        ]

    async def search_tools(
        self, query: str, principal: Principal, *, limit: int = 5, category: str | None = None
    ) -> list[ToolCard]:
        if not 1 <= limit <= 20:
            raise ValueError("Tool search limit must be between 1 and 20")
        search = func.websearch_to_tsquery("simple", query)
        rank = func.ts_rank_cd(RegistryTool.search_vector, search)
        if category:
            rank = rank + case(
                (RegistryTool.card["category_ids"].contains([category]), 0.05), else_=0.0
            )
        rows = await self.session.scalars(
            self.tool_query(principal)
            .where(RegistryTool.search_vector.bool_op("@@")(search))
            .order_by(rank.desc(), RegistryTool.tool_id, RegistryTool.schema_version)
            .limit(limit)
        )
        return [
            ToolCard.model_validate(row.card).model_copy(update={"status": row.status})
            for row in rows
        ]

    def document_query(self, principal: Principal, corpus_ids: list[str]):
        query = select(KnowledgeDocument).where(
            KnowledgeDocument.tenant_id == principal.tenant_id,
            KnowledgeDocument.active.is_(True),
            KnowledgeDocument.required_scopes.bool_op("<@")(sorted(principal.scopes)),
            or_(
                KnowledgeDocument.effective_from.is_(None),
                KnowledgeDocument.effective_from <= func.now(),
            ),
            or_(
                KnowledgeDocument.effective_until.is_(None),
                KnowledgeDocument.effective_until > func.now(),
            ),
        )
        if corpus_ids:
            query = query.where(KnowledgeDocument.corpus_id.in_(corpus_ids))
        return query

    async def active_documents(self, principal: Principal, corpus_ids: list[str]) -> list[Any]:
        query = self.document_query(principal, corpus_ids)
        return list(await self.session.scalars(query.order_by(KnowledgeDocument.document_id)))

    async def published_trajectories(
        self, principal: Principal, versions: dict[str, str]
    ) -> list[Any]:
        raise NotImplementedError("Compatible reviewed examples only; exclude unresolved successes")


class EventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def cursor(event: dict[str, Any]) -> str:
        value = json.dumps([event["run_id"], event["created_at"], event["event_id"]])
        return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")

    async def history(
        self, run_id: UUID, principal: Principal, limit: int, cursor: str | None,
        *, debug_only: bool = False, redact_evidence: bool = True,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 101:
            raise ValueError("History limit must be between 1 and 101")
        await RunRepository(self.session).get(run_id, principal)
        query = select(WorkflowEvent).where(
            WorkflowEvent.tenant_id == principal.tenant_id, WorkflowEvent.run_id == run_id
        )
        if debug_only:
            query = query.where(WorkflowEvent.event_type.like("debug\\_%", escape="\\"))
        if cursor:
            try:
                if len(cursor) > 512:
                    raise ValueError
                payload = json.loads(
                    base64.b64decode(
                        cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True
                    )
                )
                if (
                    not isinstance(payload, list)
                    or len(payload) != 3
                    or not all(isinstance(value, str) for value in payload)
                ):
                    raise ValueError
                bound_run, timestamp, event_id = payload
                if UUID(bound_run) != run_id:
                    raise ValueError
                timestamp = datetime.fromisoformat(timestamp)
                if timestamp.tzinfo is None:
                    raise ValueError
                event_id = UUID(event_id)
            except (ValueError, TypeError, UnicodeError):
                raise HTTPException(422, "Invalid history cursor") from None
            query = query.where(
                tuple_(WorkflowEvent.created_at, WorkflowEvent.event_id)
                > tuple_(timestamp, event_id)
            )
        rows = await self.session.scalars(
            query.order_by(WorkflowEvent.created_at, WorkflowEvent.event_id).limit(limit)
        )
        return [
            {
                "run_id": str(run_id),
                "event_id": str(event.event_id),
                "created_at": event.created_at.isoformat(),
                "event_type": event.event_type,
                "operation_id": str(event.operation_id) if event.operation_id else None,
                "evidence": redact(event.evidence) if redact_evidence else event.evidence,
            }
            for event in rows
        ]
