"""Short application transactions; no graph/provider call occurs inside these boundaries."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.agent.checkpoints import wait_request_from_checkpoint
from app.api.schemas import (
    AcceptedRun,
    ConversationDetail,
    ConversationRun,
    ConversationSummary,
    RunDebugPage,
    RunInput,
    RunStatus,
    SubmitRun,
)
from app.db.models import (
    Conversation,
    ConversationMessage,
    OperationState,
    RunState,
    ToolOperation,
    WorkflowEvent,
    WorkflowInput,
    WorkflowRun,
)
from app.db.repositories import EventRepository, RunRepository
from app.db.session import Database
from app.observability import Telemetry
from app.security import AuthorizationService, Principal
from app.settings import Settings

FINAL_STATES = {RunState.COMPLETED, RunState.FAILED, RunState.NEEDS_REVIEW, RunState.CANCELLED}
UNRESOLVED_STATES = {OperationState.DISPATCHING, OperationState.PENDING, OperationState.UNKNOWN}


class RunService:
    def __init__(
        self,
        database: Database | None = None,
        settings: Settings | None = None,
        authorization: AuthorizationService | None = None,
    ) -> None:
        self.database = database
        self.settings = settings or Settings()
        self.authorization = authorization

    async def _authorize(self, principal: Principal, scope: str) -> Principal:
        if self.authorization is not None:
            principal = await self.authorization.refresh(principal)
            await self.authorization.require_scopes(principal, [scope])
        elif scope not in principal.scopes:
            raise HTTPException(403, "Required permissions are missing")
        if self.database is None:
            raise HTTPException(503, "Database is not configured")
        return principal

    @staticmethod
    def _accepted(run: WorkflowRun) -> AcceptedRun:
        return AcceptedRun(
            run_id=run.run_id,
            conversation_id=run.conversation_id,
            status=run.status,
            status_url=f"/runs/{run.run_id}",
        )

    @staticmethod
    def _status(
        run: WorkflowRun, *, checkpoint_activity: list[dict[str, Any]] | None = None
    ) -> RunStatus:
        payload = run.result_payload or {}
        result = {key: value for key, value in payload.items() if key != "_llm_activity"}
        return RunStatus(
            run_id=run.run_id,
            conversation_id=run.conversation_id,
            status=run.status,
            cancel_requested=run.cancel_requested,
            wait_token=run.wait_token,
            wait_kind=run.wait_kind,
            due_at=run.due_at,
            wait_reason=payload.get("wait_request", {}).get("reason") or run.wait_kind,
            result=result or None,
            llm_activity=checkpoint_activity
            if checkpoint_activity is not None
            else payload.get("_llm_activity", []),
            error_code=(getattr(run, "error_payload", None) or {}).get("code"),
            error_type=(getattr(run, "error_payload", None) or {}).get("error_type"),
            as_of=datetime.now(UTC),
        )

    async def _checkpoint_activity(self, run: WorkflowRun) -> list[dict[str, Any]] | None:
        """Read only the activity channel; never return the rest of graph state."""
        if "_llm_activity" in (run.result_payload or {}):
            return None
        saver = await self.database.checkpointer()
        checkpoint = await saver.aget_tuple({"configurable": {"thread_id": run.thread_id}})
        if checkpoint is None:
            return []
        values = checkpoint.checkpoint.get("channel_values", {})
        activity = values.get("llm_activity", [])
        return activity if isinstance(activity, list) else []

    async def submit(
        self,
        request: SubmitRun,
        principal: Principal,
        submission_key: str,
        trace_context: dict[str, Any] | None = None,
    ) -> AcceptedRun:
        principal = await self._authorize(principal, "runs:write")
        payload = request.model_dump(mode="json")
        request_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        async with self.database.session() as session:
            try:
                async with session.begin():
                    existing = await RunRepository(session).by_submission(principal, submission_key)
                    if existing is not None:
                        return self._replay(existing, request_hash)
                    if request.conversation_id is None:
                        conversation = Conversation(
                            tenant_id=principal.tenant_id,
                            conversation_id=uuid4(),
                            user_id=principal.user_id,
                            title=" ".join(request.message.split())[:80] or "Conversation",
                        )
                        session.add(conversation)
                        await session.flush()
                    else:
                        conversation = await session.scalar(
                            select(Conversation).where(
                                Conversation.tenant_id == principal.tenant_id,
                                Conversation.user_id == principal.user_id,
                                Conversation.conversation_id == request.conversation_id,
                            )
                        )
                        if conversation is None:
                            raise HTTPException(404, "Conversation not found")
                        conversation.updated_at = datetime.now(UTC)
                    run = WorkflowRun(
                        tenant_id=principal.tenant_id,
                        run_id=uuid4(),
                        user_id=principal.user_id,
                        conversation_id=conversation.conversation_id,
                        thread_id=str(uuid4()),
                        submission_key=submission_key,
                        request_hash=request_hash,
                        request_payload=payload,
                        graph_version=self.settings.graph_version,
                        prompt_version=self.settings.prompt_version,
                        status=RunState.QUEUED,
                        trace_context=trace_context,
                    )
                    session.add(run)
                    await session.flush()
                    session.add_all(
                        [
                            ConversationMessage(
                                tenant_id=principal.tenant_id,
                                conversation_id=conversation.conversation_id,
                                run_id=run.run_id,
                                role="user",
                                text=request.message,
                            ),
                            WorkflowEvent(
                                tenant_id=principal.tenant_id,
                                run_id=run.run_id,
                                event_type="run_submitted",
                                evidence={"graph_version": run.graph_version},
                            ),
                        ]
                    )
                # Return only after the transaction committed.
                Telemetry.update_current(
                    run_id=str(run.run_id), conversation_id=str(run.conversation_id)
                )
                return self._accepted(run)
            except IntegrityError as error:
                # Roll back the whole attempt, including any new conversation/message.
                await session.rollback()
                if (
                    getattr(getattr(error.orig, "diag", None), "constraint_name", None)
                    != "uq_run_submission"
                ):
                    raise
                existing = await RunRepository(session).by_submission(principal, submission_key)
                if existing is None:
                    raise
                return self._replay(existing, request_hash)

    def _replay(self, existing: WorkflowRun, request_hash: str) -> AcceptedRun:
        if existing.request_hash != request_hash:
            raise HTTPException(409, "Idempotency key was used with different request content")
        Telemetry.update_current(
            run_id=str(existing.run_id), conversation_id=str(existing.conversation_id)
        )
        return self._accepted(existing)

    async def get_status(self, run_id: UUID, principal: Principal) -> RunStatus:
        principal = await self._authorize(principal, "runs:read")
        async with self.database.session() as session:
            run = await RunRepository(session).get(run_id, principal)
            activity = await self._checkpoint_activity(run)
            return self._status(run, checkpoint_activity=activity)

    async def get_debug_events(
        self, run_id: UUID, principal: Principal, cursor: str | None = None
    ) -> RunDebugPage:
        principal = await self._authorize(principal, "runs:read")
        async with self.database.session() as session:
            owner_run = await session.scalar(
                select(WorkflowRun.run_id).where(
                    WorkflowRun.run_id == run_id,
                    WorkflowRun.tenant_id == principal.tenant_id,
                    WorkflowRun.user_id == principal.user_id,
                )
            )
            if owner_run is None:
                raise HTTPException(404, "Run not found")
            repository = EventRepository(session)
            rows = await repository.history(
                run_id, principal, 101, cursor, debug_only=True, redact_evidence=False
            )
        events = rows[:100]
        return RunDebugPage(
            events=events,
            next_cursor=repository.cursor(events[-1]) if events else cursor,
            has_more=len(rows) > 100,
        )

    async def list_conversations(self, principal: Principal) -> list[ConversationSummary]:
        principal = await self._authorize(principal, "runs:read")
        async with self.database.session() as session:
            rows = (
                await session.scalars(
                    select(Conversation)
                    .where(
                        Conversation.tenant_id == principal.tenant_id,
                        Conversation.user_id == principal.user_id,
                    )
                    .order_by(Conversation.updated_at.desc(), Conversation.conversation_id.desc())
                    .limit(50)
                )
            ).all()
            return [
                ConversationSummary(
                    conversation_id=row.conversation_id,
                    title=row.title,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]

    async def get_conversation(
        self, conversation_id: UUID, principal: Principal
    ) -> ConversationDetail:
        principal = await self._authorize(principal, "runs:read")
        async with self.database.session() as session:
            conversation = await session.scalar(
                select(Conversation).where(
                    Conversation.tenant_id == principal.tenant_id,
                    Conversation.user_id == principal.user_id,
                    Conversation.conversation_id == conversation_id,
                )
            )
            if conversation is None:
                raise HTTPException(404, "Conversation not found")
            runs = (
                await session.scalars(
                    select(WorkflowRun)
                    .where(
                        WorkflowRun.tenant_id == principal.tenant_id,
                        WorkflowRun.user_id == principal.user_id,
                        WorkflowRun.conversation_id == conversation_id,
                    )
                    .order_by(WorkflowRun.created_at.desc(), WorkflowRun.run_id.desc())
                    .limit(100)
                )
            ).all()
            activity_by_run = {}
            runs_missing_activity = [
                run for run in runs if "_llm_activity" not in (run.result_payload or {})
            ]
            if runs_missing_activity:
                saver = await self.database.checkpointer()
                for run in runs_missing_activity:
                    checkpoint = await saver.aget_tuple(
                        {"configurable": {"thread_id": run.thread_id}}
                    )
                    values = checkpoint.checkpoint.get("channel_values", {}) if checkpoint else {}
                    activity = values.get("llm_activity", [])
                    activity_by_run[run.run_id] = activity if isinstance(activity, list) else []
            return ConversationDetail(
                conversation_id=conversation.conversation_id,
                title=conversation.title,
                runs=[
                    ConversationRun(
                        message=run.request_payload["message"],
                        run=self._status(run, checkpoint_activity=activity_by_run.get(run.run_id)),
                    )
                    for run in reversed(runs)
                ],
            )

    async def accept_input(
        self, run_id: UUID, response: RunInput, principal: Principal
    ) -> AcceptedRun:
        principal = await self._authorize(principal, "runs:write")
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError

        from app.agent.planning import canonical_payload_hash

        digest = canonical_payload_hash({"kind": response.kind, "payload": response.payload})
        async with self.database.session() as session, session.begin():
            run = await session.scalar(
                RunRepository.scoped(principal)
                .where(WorkflowRun.run_id == run_id)
                .with_for_update()
            )
            if run is None:
                raise HTTPException(404, "Run not found")
            existing = await session.scalar(
                select(WorkflowInput).where(
                    WorkflowInput.tenant_id == principal.tenant_id,
                    WorkflowInput.run_id == run_id,
                    WorkflowInput.wait_token == response.wait_token,
                )
            )
            if existing is not None:
                if existing.actor_id != principal.user_id or existing.payload_hash != digest:
                    raise HTTPException(409, "Wait input conflicts with accepted input")
                return self._accepted(run)
            if response.kind == "approval" and (
                run.user_id != principal.user_id or "paypal:write" not in principal.scopes
            ):
                raise HTTPException(
                    403, "Sandbox action approval requires its owner and write scope"
                )
            if (
                run.cancel_requested
                or run.status != RunState.WAITING_FOR_INPUT
                or run.wait_token != response.wait_token
            ):
                raise HTTPException(409, "Run is not waiting for this input")
            saver = await self.database.checkpointer()
            checkpoint = await saver.aget_tuple({"configurable": {"thread_id": run.thread_id}})
            wait = wait_request_from_checkpoint(checkpoint)
            if (
                not checkpoint
                or not any(
                    channel == "__interrupt__" for _, channel, _ in checkpoint.pending_writes
                )
                or wait is None
                or wait.wait_token != str(response.wait_token)
                or wait.kind != response.kind
            ):
                raise HTTPException(409, "Checkpoint does not match an active input interrupt")
            if response.kind == "approval":
                operation = await session.scalar(
                    select(ToolOperation).where(
                        ToolOperation.tenant_id == principal.tenant_id,
                        ToolOperation.run_id == run_id,
                        ToolOperation.operation_id == UUID(wait.operation_id),
                    )
                )
                if (
                    operation is None
                    or operation.status != OperationState.PREPARED
                    or operation.input_hash != wait.input_hash
                ):
                    raise HTTPException(409, "Approved business action has changed")
            try:
                Draft202012Validator(wait.expected_payload_schema).validate(response.payload)
            except ValidationError:
                raise HTTPException(422, "Input does not match the expected schema") from None
            session.add(
                WorkflowInput(
                    input_id=uuid4(),
                    tenant_id=principal.tenant_id,
                    run_id=run_id,
                    wait_token=response.wait_token,
                    actor_id=principal.user_id,
                    kind=response.kind,
                    payload=response.payload,
                    payload_hash=digest,
                    status="pending",
                )
            )
            run.status = RunState.QUEUED
            run.wait_token = run.wait_kind = run.due_at = None
            run.result_payload = None
            run.row_version += 1
            session.add(
                WorkflowEvent(
                    tenant_id=principal.tenant_id,
                    run_id=run_id,
                    event_type="input_accepted",
                    evidence={"kind": response.kind, "wait_token": str(response.wait_token)},
                )
            )
        return self._accepted(run)

    async def request_cancel(self, run_id: UUID, principal: Principal) -> RunStatus:
        principal = await self._authorize(principal, "runs:write")
        async with self.database.session() as session:
            async with session.begin():
                query = (
                    RunRepository.scoped(principal)
                    .where(WorkflowRun.run_id == run_id)
                    .with_for_update()
                )
                run = await session.scalar(query)
                if run is None:
                    raise HTTPException(404, "Run not found")
                if run.status in FINAL_STATES or run.cancel_requested:
                    return self._status(run)
                run.cancel_requested = True
                run.row_version += 1
                unresolved = await session.scalar(
                    select(ToolOperation.operation_id)
                    .where(
                        ToolOperation.tenant_id == principal.tenant_id,
                        ToolOperation.run_id == run_id,
                        ToolOperation.operation_type == "write",
                        ToolOperation.status.in_(UNRESOLVED_STATES),
                    )
                    .limit(1)
                )
                if (
                    run.status in {RunState.QUEUED, RunState.WAITING_FOR_INPUT}
                    and unresolved is None
                ):
                    run.status = RunState.CANCELLED
                session.add(
                    WorkflowEvent(
                        tenant_id=principal.tenant_id,
                        run_id=run_id,
                        event_type="cancel_requested",
                        evidence={"status": run.status},
                    )
                )
            return self._status(run)
