"""Small tenant-scoped Postgres store for the standalone appv2 chat."""

import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Conversation, ConversationMessage, RunState, WorkflowEvent, WorkflowRun
from app.observability import SecretRedactor
from app.security import Principal
from app.settings import Settings


class ChatStore:
    def __init__(self, settings: Settings):
        if settings.database_url is None:
            raise ValueError("DATABASE_URL is required")
        self.settings = settings
        self.engine = create_async_engine(
            settings.database_url.get_secret_value(), pool_pre_ping=True, connect_args={"connect_timeout": 5}
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.redact = SecretRedactor(settings)

    async def close(self) -> None:
        await self.engine.dispose()

    async def conversations(self, principal: Principal) -> list[Conversation]:
        async with self.sessions() as session:
            rows = await session.scalars(
                select(Conversation)
                .where(
                    Conversation.tenant_id == principal.tenant_id,
                    Conversation.user_id == principal.user_id,
                )
                .order_by(Conversation.updated_at.desc(), Conversation.conversation_id)
            )
            return list(rows)

    async def history(self, principal: Principal, conversation_id: UUID) -> list[dict]:
        async with self.sessions() as session:
            conversation = await session.scalar(
                select(Conversation).where(
                    Conversation.tenant_id == principal.tenant_id,
                    Conversation.user_id == principal.user_id,
                    Conversation.conversation_id == conversation_id,
                )
            )
            if conversation is None:
                raise PermissionError("Conversation not found")
            rows = await session.execute(
                select(ConversationMessage, WorkflowRun)
                .outerjoin(
                    WorkflowRun,
                    (WorkflowRun.tenant_id == ConversationMessage.tenant_id)
                    & (WorkflowRun.run_id == ConversationMessage.run_id),
                )
                .where(
                    ConversationMessage.tenant_id == principal.tenant_id,
                    ConversationMessage.conversation_id == conversation_id,
                )
                .order_by(ConversationMessage.created_at, ConversationMessage.message_id)
            )
            messages = []
            for message, run in rows:
                item = {"role": message.role, "content": message.text or ""}
                if run is not None and message.role == "assistant":
                    item["results"] = (run.result_payload or {}).get("results", {})
                    item["goal_plan"] = (run.result_payload or {}).get("goal_plan")
                    events = await session.scalars(
                        select(WorkflowEvent)
                        .where(
                            WorkflowEvent.tenant_id == principal.tenant_id,
                            WorkflowEvent.run_id == run.run_id,
                        )
                        .order_by(WorkflowEvent.created_at, WorkflowEvent.event_id)
                    )
                    item["events"] = [
                        {"event": event.event_type, **(event.evidence or {})} for event in events
                    ]
                messages.append(item)
            return messages

    async def recent_messages(
        self, principal: Principal, conversation_id: UUID | None, *, limit: int = 20
    ) -> list[dict[str, str]]:
        if conversation_id is None:
            return []
        async with self.sessions() as session:
            rows = await session.scalars(
                select(ConversationMessage)
                .join(
                    Conversation,
                    (Conversation.tenant_id == ConversationMessage.tenant_id)
                    & (Conversation.conversation_id == ConversationMessage.conversation_id),
                )
                .where(
                    Conversation.tenant_id == principal.tenant_id,
                    Conversation.user_id == principal.user_id,
                    Conversation.conversation_id == conversation_id,
                    ConversationMessage.text.is_not(None),
                )
                .order_by(
                    ConversationMessage.created_at.desc(), ConversationMessage.message_id.desc()
                )
                .limit(limit)
            )
            return [
                {"role": message.role, "content": message.text or ""}
                for message in reversed(list(rows))
            ]


    async def start_turn(
        self, principal: Principal, conversation_id: UUID | None, request: str
    ) -> tuple[UUID, UUID]:
        run_id = uuid4()
        now = datetime.now(UTC)
        async with self.sessions.begin() as session:
            conversation = None
            if conversation_id is not None:
                conversation = await session.scalar(
                    select(Conversation).where(
                        Conversation.tenant_id == principal.tenant_id,
                        Conversation.user_id == principal.user_id,
                        Conversation.conversation_id == conversation_id,
                    )
                )
                if conversation is None:
                    raise PermissionError("Conversation not found")
            else:
                conversation = Conversation(
                    tenant_id=principal.tenant_id,
                    user_id=principal.user_id,
                    title=request.strip().replace("\n", " ")[:100] or "New conversation",
                )
                session.add(conversation)
                await session.flush()
            run = WorkflowRun(
                run_id=run_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                conversation_id=conversation.conversation_id,
                thread_id=f"appv2:{run_id}",
                submission_key=str(run_id),
                request_hash=hashlib.sha256(request.encode()).hexdigest(),
                request_payload={"request": self.redact(request)},
                graph_version="appv2",
                prompt_version=self.settings.prompt_version,
                status=RunState.RUNNING,
                created_at=now,
                updated_at=now,
            )
            conversation.updated_at = now
            session.add(run)
            # These tables have a composite FK; persist the run before its message/event.
            await session.flush([run])
            session.add(
                ConversationMessage(
                    tenant_id=principal.tenant_id,
                    conversation_id=conversation.conversation_id,
                    run_id=run_id,
                    role="user",
                    text=request,
                )
            )
            session.add(
                WorkflowEvent(
                    tenant_id=principal.tenant_id,
                    run_id=run_id,
                    event_type="user_request",
                    evidence={"request": self.redact(request)},
                )
            )
        return conversation.conversation_id, run_id

    async def log_event(self, principal: Principal, run_id: UUID, event: dict) -> None:
        safe = self.redact(event)
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(WorkflowRun).where(
                    WorkflowRun.tenant_id == principal.tenant_id,
                    WorkflowRun.user_id == principal.user_id,
                    WorkflowRun.run_id == run_id,
                )
            )
            if run is None:
                raise PermissionError("Chat turn not found")
            session.add(
                WorkflowEvent(
                    tenant_id=principal.tenant_id,
                    run_id=run_id,
                    event_type=str(event.get("event", "event"))[:100],
                    evidence=safe,
                )
            )

    async def finish_turn(
        self,
        principal: Principal,
        conversation_id: UUID,
        run_id: UUID,
        *,
        response: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(WorkflowRun).where(
                    WorkflowRun.tenant_id == principal.tenant_id,
                    WorkflowRun.user_id == principal.user_id,
                    WorkflowRun.run_id == run_id,
                )
            )
            if run is None:
                raise PermissionError("Chat turn not found")
            run.status = RunState.FAILED if error else RunState.COMPLETED
            run.result_payload = self.redact(
                {key: value for key, value in (result or {}).items() if key != "events"}
            )
            run.error_payload = {"detail": self.redact(error)} if error else None
            run.updated_at = datetime.now(UTC)
            session.add(
                ConversationMessage(
                    tenant_id=principal.tenant_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    role="assistant",
                    text=response,
                )
            )
            session.add(
                WorkflowEvent(
                    tenant_id=principal.tenant_id,
                    run_id=run_id,
                    event_type="turn_failed" if error else "turn_completed",
                    evidence={"detail": self.redact(error)} if error else {},
                )
            )
