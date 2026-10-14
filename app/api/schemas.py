"""Public clients may submit clarification/approval, never internal timer evidence."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubmitRun(Contract):
    message: str = Field(min_length=1)
    conversation_id: UUID | None = None


class AcceptedRun(Contract):
    run_id: UUID
    conversation_id: UUID
    status: str
    status_url: str


class RunStatus(Contract):
    run_id: UUID
    conversation_id: UUID
    status: str
    cancel_requested: bool
    wait_reason: str | None = None
    wait_token: UUID | None = None
    wait_kind: str | None = None
    due_at: datetime | None = None
    result: dict[str, Any] | None = None
    llm_activity: list[dict[str, Any]] = Field(default_factory=list)
    error_code: str | None = None
    error_type: str | None = None
    as_of: datetime


class RunDebugPage(Contract):
    events: list[dict[str, Any]] = Field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


class ConversationSummary(Contract):
    conversation_id: UUID
    title: str
    updated_at: datetime


class ConversationDetail(Contract):
    conversation_id: UUID
    title: str
    runs: list["ConversationRun"]


class ConversationRun(Contract):
    message: str
    run: RunStatus


class RunInput(Contract):
    wait_token: UUID
    kind: Literal["clarification", "approval"]
    payload: dict[str, Any]


class ErrorResponse(Contract):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)
