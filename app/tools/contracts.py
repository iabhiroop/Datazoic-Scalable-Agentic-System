"""Typed tool metadata and adapter protocol. Credentials are never part of a card."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetryPolicy(Contract):
    max_attempts: int = Field(default=3, ge=1)
    safe_error_codes: list[str] = Field(default_factory=list)
    write_transport_retry_allowed: bool = False


class IdempotencyPolicy(Contract):
    supported: bool = False
    header_or_field: str | None = None
    retention_seconds: int | None = Field(default=None, gt=0)
    request_consistency_rules: str | None = None


class EndpointContract(Contract):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    schema_source: str
    query_array_explode: dict[str, bool] = Field(default_factory=dict)
    prefer_representation: bool = False
    request_id_header: bool = False


class ToolCard(Contract):
    tool_id: str
    schema_version: str
    adapter_key: str
    name: str
    description: str
    examples: list[dict[str, Any]] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    category_ids: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    action: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    operation_type: Literal["read", "write"]
    required_scopes: list[str]
    precondition_key: str | None
    verification_key: str
    timeout_seconds: float = Field(default=30, gt=0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    idempotency: IdempotencyPolicy = Field(default_factory=IdempotencyPolicy)
    reconciliation_key: str | None = None
    reconciliation_deadline_seconds: int | None = Field(default=None, gt=0)
    max_status_checks: int | None = Field(default=None, gt=0)
    provider: str
    provider_account_key: str
    status: Literal["draft", "published", "revoked"] = "draft"
    embedding_model: str | None = None
    embedding_version: str | None = None
    endpoint: EndpointContract | None = None


@dataclass(frozen=True)
class ToolContext:
    tenant_id: UUID
    user_id: str
    run_id: UUID
    operation_id: UUID
    idempotency_key: str | None
    conversation_id: UUID | None = None


@dataclass
class ToolOutcome:
    state: Literal["succeeded", "pending", "failed", "unknown"]
    data: dict[str, Any]
    provider_operation_id: str | None = None
    error_code: str | None = None
    retryable: bool = False
    effect_known_absent: bool = False


class ToolAdapter(Protocol):
    async def authorize_result(
        self, arguments: dict[str, Any], data: dict[str, Any], context: ToolContext
    ) -> None: ...

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutcome: ...

    async def reconcile(self, operation: dict[str, Any], context: ToolContext) -> ToolOutcome: ...
