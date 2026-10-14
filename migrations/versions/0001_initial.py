"""Initial application schema. Dimension is explicitly selected at first deployment.

This revision is a frozen schema snapshot, independent of future ORM changes.
LangGraph checkpoint tables are initialized by a separate CLI operation.
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    embedding_dimension = op.get_context().config.attributes["embedding_dimension"]
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("conversation_id", name=op.f("pk_conversations")),
        sa.UniqueConstraint(
            "tenant_id", "conversation_id", name=op.f("uq_conversations_tenant_id_conversation_id")
        ),
    )
    op.create_table(
        "system_versions",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("registry_revision", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("taxonomy_revision", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("knowledge_revision", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("acl_revision", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "registry_revision >= 0 AND taxonomy_revision >= 0 AND knowledge_revision >= 0 AND acl_revision >= 0",
            name="ck_system_revisions",
        ),
        sa.PrimaryKeyConstraint("tenant_id", name=op.f("pk_system_versions")),
    )
    op.create_table(
        "taxonomy_categories",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("category_id", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("parent_id", sa.Text(), nullable=True),
        sa.Column("depth", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("parent_depth", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.CheckConstraint(
            "parent_depth = 1 AND ((depth = 1 AND parent_id IS NULL) OR (depth = 2 AND parent_id IS NOT NULL))",
            name="ck_taxonomy_depth",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "parent_id", "parent_depth"],
            [
                "taxonomy_categories.tenant_id",
                "taxonomy_categories.category_id",
                "taxonomy_categories.depth",
            ],
            name=op.f(
                "fk_taxonomy_categories_tenant_id_parent_id_parent_depth_taxonomy_categories"
            ),
        ),
        sa.PrimaryKeyConstraint("tenant_id", "category_id", name=op.f("pk_taxonomy_categories")),
        sa.UniqueConstraint(
            "tenant_id",
            "category_id",
            "depth",
            name=op.f("uq_taxonomy_categories_tenant_id_category_id_depth"),
        ),
    )
    op.create_table(
        "tool_registry",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("tool_id", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("card", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("searchable_text", sa.Text(), nullable=False),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('simple'::regconfig, searchable_text)", persisted=True),
            nullable=False,
        ),
        sa.Column("embedding", Vector(embedding_dimension), nullable=True),
        sa.Column("embedding_version", sa.Text(), nullable=True),
        sa.Column("is_current", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("status", sa.Text(), server_default=sa.text("'draft'"), nullable=False),
        sa.CheckConstraint("NOT is_current OR status = 'published'", name="ck_current_published"),
        sa.CheckConstraint(
            "status != 'published' OR (embedding IS NOT NULL AND embedding_version IS NOT NULL)",
            name="ck_published_embedding",
        ),
        sa.CheckConstraint("status IN ('draft','published','revoked')", name="ck_registry_status"),
        sa.PrimaryKeyConstraint(
            "tenant_id", "tool_id", "schema_version", name=op.f("pk_tool_registry")
        ),
    )
    op.create_index(
        "registry_current_idx",
        "tool_registry",
        ["tenant_id", "tool_id"],
        unique=True,
        postgresql_where=sa.text("is_current AND status = 'published'"),
    )
    op.create_index(
        "registry_text_idx",
        "tool_registry",
        ["search_vector"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_table(
        "workflow_runs",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("submission_key", sa.Text(), nullable=False),
        sa.Column("request_hash", sa.Text(), nullable=False),
        sa.Column("request_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("graph_version", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("status", sa.String(26), nullable=False),
        sa.Column("wait_token", sa.Uuid(), nullable=True),
        sa.Column("wait_kind", sa.Text(), nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "cancel_requested", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("worker_id", sa.Text(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("trace_context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("row_version", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'waiting_for_input', 'waiting_for_retry', 'waiting_for_reconciliation', 'completed', 'failed', 'needs_review', 'cancelled')",
            name="ck_run_status",
        ),
        sa.CheckConstraint("row_version >= 0", name="ck_run_row_version"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.conversation_id"],
            name=op.f("fk_workflow_runs_tenant_id_conversation_id_conversations"),
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_workflow_runs")),
        sa.UniqueConstraint("tenant_id", "run_id", name=op.f("uq_workflow_runs_tenant_id_run_id")),
        sa.UniqueConstraint("tenant_id", "user_id", "submission_key", name="uq_run_submission"),
        sa.UniqueConstraint("thread_id", name=op.f("uq_workflow_runs_thread_id")),
    )
    op.create_index(
        "workflow_runnable_idx", "workflow_runs", ["status", "due_at", "created_at"], unique=False
    )
    op.create_index(
        "workflow_user_history_idx",
        "workflow_runs",
        [
            "tenant_id",
            "user_id",
            sa.literal_column("created_at DESC"),
            sa.literal_column("run_id DESC"),
        ],
        unique=False,
    )
    op.create_table(
        "payload_refs",
        sa.Column("payload_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("checksum", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint("size >= 0", name="ck_payload_size"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "run_id"],
            ["workflow_runs.tenant_id", "workflow_runs.run_id"],
            name=op.f("fk_payload_refs_tenant_id_run_id_workflow_runs"),
        ),
        sa.PrimaryKeyConstraint("payload_id", name=op.f("pk_payload_refs")),
        sa.UniqueConstraint(
            "tenant_id", "object_key", name=op.f("uq_payload_refs_tenant_id_object_key")
        ),
        sa.UniqueConstraint(
            "tenant_id", "payload_id", name=op.f("uq_payload_refs_tenant_id_payload_id")
        ),
    )
    op.create_table(
        "trajectory_summaries",
        sa.Column("trajectory_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("source_run_id", sa.Uuid(), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("plan_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("versions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("embedding", Vector(embedding_dimension), nullable=False),
        sa.Column("embedding_version", sa.Text(), nullable=False),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('simple'::regconfig, goal)", persisted=True),
            nullable=False,
        ),
        sa.Column("review_status", sa.Text(), server_default=sa.text("'pending'"), nullable=False),
        sa.Column(
            "required_scopes", sa.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "review_status != 'approved' OR outcome NOT IN ('pending','unknown')",
            name="ck_trajectory_outcome",
        ),
        sa.CheckConstraint(
            "review_status IN ('pending','approved','rejected')", name="ck_trajectory_review"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "source_run_id"],
            ["workflow_runs.tenant_id", "workflow_runs.run_id"],
            name=op.f("fk_trajectory_summaries_tenant_id_source_run_id_workflow_runs"),
        ),
        sa.PrimaryKeyConstraint("trajectory_id", name=op.f("pk_trajectory_summaries")),
    )
    op.create_index(
        "trajectory_text_idx",
        "trajectory_summaries",
        ["search_vector"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_table(
        "workflow_inputs",
        sa.Column("input_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("wait_token", sa.Uuid(), nullable=False),
        sa.Column("actor_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("payload_hash", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint("kind IN ('clarification','approval','timer')", name="ck_input_kind"),
        sa.CheckConstraint("status IN ('pending','applied','rejected')", name="ck_input_status"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "run_id"],
            ["workflow_runs.tenant_id", "workflow_runs.run_id"],
            name=op.f("fk_workflow_inputs_tenant_id_run_id_workflow_runs"),
        ),
        sa.PrimaryKeyConstraint("input_id", name=op.f("pk_workflow_inputs")),
        sa.UniqueConstraint(
            "tenant_id", "input_id", name=op.f("uq_workflow_inputs_tenant_id_input_id")
        ),
        sa.UniqueConstraint("tenant_id", "run_id", "wait_token", name="uq_input_wait"),
    )
    op.create_table(
        "conversation_messages",
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("input_id", sa.Uuid(), nullable=True),
        sa.Column("payload_ref", sa.Uuid(), nullable=True),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint("role IN ('user','assistant','system','tool')", name="ck_message_role"),
        sa.CheckConstraint(
            "text IS NOT NULL OR payload_ref IS NOT NULL", name="ck_message_content"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.conversation_id"],
            name=op.f("fk_conversation_messages_tenant_id_conversation_id_conversations"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "input_id"],
            ["workflow_inputs.tenant_id", "workflow_inputs.input_id"],
            name=op.f("fk_conversation_messages_tenant_id_input_id_workflow_inputs"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "payload_ref"],
            ["payload_refs.tenant_id", "payload_refs.payload_id"],
            name=op.f("fk_conversation_messages_tenant_id_payload_ref_payload_refs"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "run_id"],
            ["workflow_runs.tenant_id", "workflow_runs.run_id"],
            name=op.f("fk_conversation_messages_tenant_id_run_id_workflow_runs"),
        ),
        sa.PrimaryKeyConstraint("message_id", name=op.f("pk_conversation_messages")),
    )
    op.create_index(
        "conversation_messages_time_idx",
        "conversation_messages",
        ["tenant_id", "conversation_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "knowledge_documents",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("corpus_id", sa.Text(), nullable=False),
        sa.Column("source_key", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("payload_ref", sa.Uuid(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column(
            "required_scopes", sa.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column("checksum", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("effective_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "effective_until IS NULL OR effective_from IS NULL OR effective_until > effective_from",
            name="ck_document_dates",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "payload_ref"],
            ["payload_refs.tenant_id", "payload_refs.payload_id"],
            name=op.f("fk_knowledge_documents_tenant_id_payload_ref_payload_refs"),
        ),
        sa.PrimaryKeyConstraint("document_id", name=op.f("pk_knowledge_documents")),
        sa.UniqueConstraint(
            "tenant_id",
            "corpus_id",
            "source_key",
            "version",
            name=op.f("uq_knowledge_documents_tenant_id_corpus_id_source_key_version"),
        ),
        sa.UniqueConstraint(
            "tenant_id", "document_id", name=op.f("uq_knowledge_documents_tenant_id_document_id")
        ),
    )
    op.create_index(
        "document_active_idx",
        "knowledge_documents",
        ["tenant_id", "corpus_id", "source_key"],
        unique=True,
        postgresql_where=sa.text("active"),
    )
    op.create_table(
        "tool_operations",
        sa.Column("operation_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("action_ref", sa.Text(), nullable=False),
        sa.Column("tool_id", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("operation_type", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("provider_account_key", sa.Text(), nullable=False),
        sa.Column("input_hash", sa.Text(), nullable=False),
        sa.Column("input_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(11), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column("idempotency_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider_operation_id", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("result_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("payload_ref", sa.Uuid(), nullable=True),
        sa.Column("error_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_check_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("row_version", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("operation_type IN ('read','write')", name="ck_operation_type"),
        sa.CheckConstraint(
            "status IN ('prepared', 'dispatching', 'pending', 'succeeded', 'failed', 'unknown')",
            name="ck_operation_status",
        ),
        sa.CheckConstraint("attempt_count >= 0 AND row_version >= 0", name="ck_operation_counters"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "payload_ref"],
            ["payload_refs.tenant_id", "payload_refs.payload_id"],
            name=op.f("fk_tool_operations_tenant_id_payload_ref_payload_refs"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "run_id"],
            ["workflow_runs.tenant_id", "workflow_runs.run_id"],
            name=op.f("fk_tool_operations_tenant_id_run_id_workflow_runs"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "tool_id", "schema_version"],
            ["tool_registry.tenant_id", "tool_registry.tool_id", "tool_registry.schema_version"],
            name=op.f("fk_tool_operations_tenant_id_tool_id_schema_version_tool_registry"),
        ),
        sa.PrimaryKeyConstraint("operation_id", name=op.f("pk_tool_operations")),
        sa.UniqueConstraint(
            "tenant_id", "operation_id", name=op.f("uq_tool_operations_tenant_id_operation_id")
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_account_key",
            "tool_id",
            "idempotency_key",
            name=op.f(
                "uq_tool_operations_tenant_id_provider_provider_account_key_tool_id_idempotency_key"
            ),
        ),
        sa.UniqueConstraint("tenant_id", "run_id", "action_ref", name="uq_operation_action"),
    )
    op.create_index(
        "operation_pending_idx", "tool_operations", ["status", "next_check_at"], unique=False
    )
    op.create_table(
        "knowledge_chunks",
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('simple'::regconfig, text)", persisted=True),
            nullable=False,
        ),
        sa.Column("embedding", Vector(embedding_dimension), nullable=False),
        sa.Column("embedding_version", sa.Text(), nullable=False),
        sa.Column("locator", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint("chunk_index >= 0", name="ck_chunk_index"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["knowledge_documents.tenant_id", "knowledge_documents.document_id"],
            name=op.f("fk_knowledge_chunks_tenant_id_document_id_knowledge_documents"),
        ),
        sa.PrimaryKeyConstraint("chunk_id", name=op.f("pk_knowledge_chunks")),
        sa.UniqueConstraint(
            "tenant_id",
            "document_id",
            "chunk_index",
            name=op.f("uq_knowledge_chunks_tenant_id_document_id_chunk_index"),
        ),
    )
    op.create_index(
        "knowledge_text_idx",
        "knowledge_chunks",
        ["search_vector"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_table(
        "workflow_events",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("operation_id", sa.Uuid(), nullable=True),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "operation_id"],
            ["tool_operations.tenant_id", "tool_operations.operation_id"],
            name=op.f("fk_workflow_events_tenant_id_operation_id_tool_operations"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "run_id"],
            ["workflow_runs.tenant_id", "workflow_runs.run_id"],
            name=op.f("fk_workflow_events_tenant_id_run_id_workflow_runs"),
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_workflow_events")),
    )
    op.create_index(
        "workflow_events_time_idx",
        "workflow_events",
        ["tenant_id", "run_id", "created_at", "event_id"],
        unique=False,
    )
    op.execute(
        "\n        CREATE FUNCTION reject_workflow_event_mutation() RETURNS trigger\n        LANGUAGE plpgsql AS $$\n        BEGIN\n            RAISE EXCEPTION 'workflow_events are append-only' USING ERRCODE = '42501';\n        END; $$\n    "
    )
    op.execute(
        "\n        CREATE TRIGGER workflow_events_append_only\n        BEFORE UPDATE OR DELETE ON workflow_events\n        FOR EACH ROW EXECUTE FUNCTION reject_workflow_event_mutation()\n    "
    )


def downgrade():
    op.drop_table("workflow_events")
    op.drop_table("knowledge_chunks")
    op.drop_table("tool_operations")
    op.drop_table("knowledge_documents")
    op.drop_table("conversation_messages")
    op.drop_table("workflow_inputs")
    op.drop_table("trajectory_summaries")
    op.drop_table("payload_refs")
    op.drop_table("workflow_runs")
    op.drop_table("tool_registry")
    op.drop_table("taxonomy_categories")
    op.drop_table("system_versions")
    op.drop_table("conversations")
    op.execute("DROP FUNCTION reject_workflow_event_mutation()")
