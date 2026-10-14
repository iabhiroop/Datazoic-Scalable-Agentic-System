"""Application ORM mappings. Composite foreign keys preserve tenant boundaries."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Enum,
    ForeignKeyConstraint,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_RETRY = "waiting_for_retry"
    WAITING_FOR_RECONCILIATION = "waiting_for_reconciliation"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    CANCELLED = "cancelled"


class OperationState(StrEnum):
    PREPARED = "prepared"
    DISPATCHING = "dispatching"
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


def lifecycle(enum: type[StrEnum], name: str) -> Enum:
    return Enum(
        enum,
        values_callable=lambda values: [v.value for v in values],
        native_enum=False,
        create_constraint=True,
        name=name,
    )


class TenantRecord:
    tenant_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)


class Timestamps:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Conversation(TenantRecord, Timestamps, Base):
    __tablename__ = "conversations"
    __table_args__ = (UniqueConstraint("tenant_id", "conversation_id"),)
    conversation_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)


class WorkflowRun(TenantRecord, Timestamps, Base):
    __tablename__ = "workflow_runs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "run_id"),
        UniqueConstraint("tenant_id", "user_id", "submission_key", name="uq_run_submission"),
        ForeignKeyConstraint(
            ["tenant_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.conversation_id"],
        ),
        CheckConstraint("row_version >= 0", name="ck_run_row_version"),
        Index("workflow_runnable_idx", "status", "due_at", "created_at"),
        Index(
            "workflow_user_history_idx",
            "tenant_id",
            "user_id",
            text("created_at DESC"),
            text("run_id DESC"),
        ),
    )
    run_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(Text)
    conversation_id: Mapped[UUID] = mapped_column(Uuid)
    thread_id: Mapped[str] = mapped_column(Text, unique=True)
    submission_key: Mapped[str] = mapped_column(Text)
    request_hash: Mapped[str] = mapped_column(Text)
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    graph_version: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text)
    status: Mapped[RunState] = mapped_column(lifecycle(RunState, "ck_run_status"))
    wait_token: Mapped[UUID | None] = mapped_column(Uuid)
    wait_kind: Mapped[str | None] = mapped_column(Text)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_requested: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    worker_id: Mapped[str | None] = mapped_column(Text)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    trace_context: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    row_version: Mapped[int] = mapped_column(Integer, server_default=text("0"))


class PayloadReference(TenantRecord, Base):
    __tablename__ = "payload_refs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "payload_id"),
        UniqueConstraint("tenant_id", "object_key"),
        ForeignKeyConstraint(
            ["tenant_id", "run_id"], ["workflow_runs.tenant_id", "workflow_runs.run_id"]
        ),
        CheckConstraint("size >= 0", name="ck_payload_size"),
    )
    payload_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(Text)
    run_id: Mapped[UUID | None] = mapped_column(Uuid)
    object_key: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(Text)
    size: Mapped[int] = mapped_column(Integer)
    checksum: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RegistryTool(TenantRecord, Base):
    __tablename__ = "tool_registry"
    __table_args__ = (
        CheckConstraint("status IN ('draft','published','revoked')", name="ck_registry_status"),
        CheckConstraint("NOT is_current OR status = 'published'", name="ck_current_published"),
        CheckConstraint(
            "status != 'published' OR (embedding IS NOT NULL AND embedding_version IS NOT NULL)",
            name="ck_published_embedding",
        ),
        Index(
            "registry_current_idx",
            "tenant_id",
            "tool_id",
            unique=True,
            postgresql_where=text("is_current AND status = 'published'"),
        ),
        Index("registry_text_idx", "search_vector", postgresql_using="gin"),
    )
    tenant_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    tool_id: Mapped[str] = mapped_column(Text, primary_key=True)
    schema_version: Mapped[str] = mapped_column(Text, primary_key=True)
    card: Mapped[dict[str, Any]] = mapped_column(JSONB)
    searchable_text: Mapped[str] = mapped_column(Text)
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('simple'::regconfig, searchable_text)", persisted=True)
    )
    # Dimension is fixed by the initial migration and checked at startup.
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
    embedding_version: Mapped[str | None] = mapped_column(Text)
    is_current: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    status: Mapped[str] = mapped_column(Text, server_default=text("'draft'"))


class ToolOperation(TenantRecord, Timestamps, Base):
    __tablename__ = "tool_operations"
    __table_args__ = (
        UniqueConstraint("tenant_id", "operation_id"),
        UniqueConstraint("tenant_id", "run_id", "action_ref", name="uq_operation_action"),
        UniqueConstraint(
            "tenant_id", "provider", "provider_account_key", "tool_id", "idempotency_key"
        ),
        ForeignKeyConstraint(
            ["tenant_id", "run_id"], ["workflow_runs.tenant_id", "workflow_runs.run_id"]
        ),
        ForeignKeyConstraint(
            ["tenant_id", "tool_id", "schema_version"],
            ["tool_registry.tenant_id", "tool_registry.tool_id", "tool_registry.schema_version"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "payload_ref"], ["payload_refs.tenant_id", "payload_refs.payload_id"]
        ),
        CheckConstraint("operation_type IN ('read','write')", name="ck_operation_type"),
        CheckConstraint("attempt_count >= 0 AND row_version >= 0", name="ck_operation_counters"),
        Index("operation_pending_idx", "status", "next_check_at"),
    )
    operation_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(Uuid)
    action_ref: Mapped[str] = mapped_column(Text)
    tool_id: Mapped[str] = mapped_column(Text)
    schema_version: Mapped[str] = mapped_column(Text)
    operation_type: Mapped[str] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(Text)
    provider_account_key: Mapped[str] = mapped_column(Text)
    input_hash: Mapped[str] = mapped_column(Text)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[OperationState] = mapped_column(lifecycle(OperationState, "ck_operation_status"))
    idempotency_key: Mapped[str | None] = mapped_column(Text)
    idempotency_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provider_operation_id: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    payload_ref: Mapped[UUID | None] = mapped_column(Uuid)
    error_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_check_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    row_version: Mapped[int] = mapped_column(Integer, server_default=text("0"))


class WorkflowInput(TenantRecord, Base):
    __tablename__ = "workflow_inputs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "input_id"),
        UniqueConstraint("tenant_id", "run_id", "wait_token", name="uq_input_wait"),
        ForeignKeyConstraint(
            ["tenant_id", "run_id"], ["workflow_runs.tenant_id", "workflow_runs.run_id"]
        ),
        CheckConstraint("kind IN ('clarification','approval','timer')", name="ck_input_kind"),
        CheckConstraint("status IN ('pending','applied','rejected')", name="ck_input_status"),
    )
    input_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(Uuid)
    wait_token: Mapped[UUID] = mapped_column(Uuid)
    actor_id: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    payload_hash: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConversationMessage(TenantRecord, Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.conversation_id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "run_id"], ["workflow_runs.tenant_id", "workflow_runs.run_id"]
        ),
        ForeignKeyConstraint(
            ["tenant_id", "input_id"], ["workflow_inputs.tenant_id", "workflow_inputs.input_id"]
        ),
        ForeignKeyConstraint(
            ["tenant_id", "payload_ref"], ["payload_refs.tenant_id", "payload_refs.payload_id"]
        ),
        CheckConstraint("role IN ('user','assistant','system','tool')", name="ck_message_role"),
        CheckConstraint("text IS NOT NULL OR payload_ref IS NOT NULL", name="ck_message_content"),
        Index("conversation_messages_time_idx", "tenant_id", "conversation_id", "created_at"),
    )
    message_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(Uuid)
    role: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    text: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[UUID | None] = mapped_column(Uuid)
    input_id: Mapped[UUID | None] = mapped_column(Uuid)
    payload_ref: Mapped[UUID | None] = mapped_column(Uuid)


class TaxonomyCategory(TenantRecord, Base):
    __tablename__ = "taxonomy_categories"
    __table_args__ = (
        UniqueConstraint("tenant_id", "category_id", "depth"),
        ForeignKeyConstraint(
            ["tenant_id", "parent_id", "parent_depth"],
            [
                "taxonomy_categories.tenant_id",
                "taxonomy_categories.category_id",
                "taxonomy_categories.depth",
            ],
        ),
        CheckConstraint(
            "parent_depth = 1 AND ((depth = 1 AND parent_id IS NULL) OR (depth = 2 AND parent_id IS NOT NULL))",
            name="ck_taxonomy_depth",
        ),
    )
    tenant_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    category_id: Mapped[str] = mapped_column(Text, primary_key=True)
    label: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    revision: Mapped[int] = mapped_column(Integer)
    parent_id: Mapped[str | None] = mapped_column(Text)
    depth: Mapped[int] = mapped_column(Integer, server_default=text("1"))
    parent_depth: Mapped[int] = mapped_column(Integer, server_default=text("1"))


class KnowledgeDocument(TenantRecord, Base):
    __tablename__ = "knowledge_documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "document_id"),
        UniqueConstraint("tenant_id", "corpus_id", "source_key", "version"),
        ForeignKeyConstraint(
            ["tenant_id", "payload_ref"], ["payload_refs.tenant_id", "payload_refs.payload_id"]
        ),
        CheckConstraint(
            "effective_until IS NULL OR effective_from IS NULL OR effective_until > effective_from",
            name="ck_document_dates",
        ),
        Index(
            "document_active_idx",
            "tenant_id",
            "corpus_id",
            "source_key",
            unique=True,
            postgresql_where=text("active"),
        ),
    )
    document_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    corpus_id: Mapped[str] = mapped_column(Text)
    source_key: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(Text)
    payload_ref: Mapped[UUID] = mapped_column(Uuid)
    title: Mapped[str] = mapped_column(Text)
    required_scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    checksum: Mapped[str] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    effective_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeChunk(TenantRecord, Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("tenant_id", "document_id", "chunk_index"),
        ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["knowledge_documents.tenant_id", "knowledge_documents.document_id"],
        ),
        CheckConstraint("chunk_index >= 0", name="ck_chunk_index"),
        Index("knowledge_text_idx", "search_vector", postgresql_using="gin"),
    )
    chunk_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(Uuid)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('simple'::regconfig, text)", persisted=True)
    )
    embedding: Mapped[list[float]] = mapped_column(Vector())
    embedding_version: Mapped[str] = mapped_column(Text)
    locator: Mapped[str] = mapped_column(Text)


class TrajectorySummary(TenantRecord, Base):
    __tablename__ = "trajectory_summaries"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "source_run_id"], ["workflow_runs.tenant_id", "workflow_runs.run_id"]
        ),
        CheckConstraint(
            "review_status IN ('pending','approved','rejected')", name="ck_trajectory_review"
        ),
        CheckConstraint(
            "review_status != 'approved' OR outcome NOT IN ('pending','unknown')",
            name="ck_trajectory_outcome",
        ),
        Index("trajectory_text_idx", "search_vector", postgresql_using="gin"),
    )
    trajectory_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(Text)
    source_run_id: Mapped[UUID] = mapped_column(Uuid)
    goal: Mapped[str] = mapped_column(Text)
    plan_summary: Mapped[dict[str, Any]] = mapped_column(JSONB)
    outcome: Mapped[str] = mapped_column(Text)
    versions: Mapped[dict[str, str]] = mapped_column(JSONB)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    embedding_version: Mapped[str] = mapped_column(Text)
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('simple'::regconfig, goal)", persisted=True)
    )
    review_status: Mapped[str] = mapped_column(Text, server_default=text("'pending'"))
    required_scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))


class WorkflowEvent(TenantRecord, Base):
    __tablename__ = "workflow_events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "run_id"], ["workflow_runs.tenant_id", "workflow_runs.run_id"]
        ),
        ForeignKeyConstraint(
            ["tenant_id", "operation_id"],
            ["tool_operations.tenant_id", "tool_operations.operation_id"],
        ),
        Index("workflow_events_time_idx", "tenant_id", "run_id", "created_at", "event_id"),
    )
    event_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(Uuid)
    event_type: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    operation_id: Mapped[UUID | None] = mapped_column(Uuid)


class SystemVersions(Base):
    __tablename__ = "system_versions"
    __table_args__ = (
        CheckConstraint(
            "registry_revision >= 0 AND taxonomy_revision >= 0 AND knowledge_revision >= 0 AND acl_revision >= 0",
            name="ck_system_revisions",
        ),
    )
    tenant_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    registry_revision: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    taxonomy_revision: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    knowledge_revision: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    acl_revision: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
