"""Read-only structured system queries; no model-generated SQL or arbitrary log access."""

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import Field, TypeAdapter, ValidationError, model_validator
from sqlalchemy import select

from app.db.models import ToolOperation, WorkflowEvent
from app.db.repositories import CatalogRepository, EventRepository, RunRepository
from app.db.session import Database
from app.observability import Telemetry
from app.security import Principal, TokenAuthenticator
from app.tools.contracts import Contract, ToolContext, ToolOutcome


class CapabilitiesArguments(Contract):
    mode: Literal["capabilities"]
    query: str = Field(min_length=1, max_length=2000)
    category: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class RunSelector(Contract):
    run_id: UUID | None = None
    selector: Literal["last"] | None = None

    @model_validator(mode="after")
    def exclusive_selector(self) -> "RunSelector":
        if self.run_id is not None and self.selector is not None:
            raise ValueError("Choose run_id or selector, not both")
        if self.run_id is None and self.selector is None:
            self.selector = "last"
        return self


class StatusArguments(RunSelector):
    mode: Literal["request_status"]


class HistoryArguments(RunSelector):
    mode: Literal["execution_history"]
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = None


SearchArguments = Annotated[
    CapabilitiesArguments | StatusArguments | HistoryArguments, Field(discriminator="mode")
]
SEARCH_ARGUMENTS = TypeAdapter(SearchArguments)


class Capability(Contract):
    tool_id: str
    name: str
    description: str
    operation_type: Literal["read", "write"]
    category_ids: list[str]


class CapabilitiesResult(Contract):
    capabilities: list[Capability]


class OperationSummary(Contract):
    operation_id: UUID
    state: str


class StatusResult(Contract):
    run_id: UUID
    run_created_at: datetime
    status: str
    summary: str
    operations: list[OperationSummary]
    as_of: datetime


class HistoryResult(Contract):
    run_id: UUID
    run_created_at: datetime
    events: list[dict[str, Any]]
    next_cursor: str | None = None


SEARCH_RESULTS = TypeAdapter(CapabilitiesResult | StatusResult | HistoryResult)


class SystemSearch:
    async def authorize_result(self, arguments, data, context) -> None:
        request = SEARCH_ARGUMENTS.validate_python(arguments)
        principal = await self.authenticator.refresh(
            Principal(context.tenant_id, context.user_id, frozenset())
        )
        if "system:read" not in principal.scopes:
            raise HTTPException(403, "System read permission is required")
        async with self.database.session() as session:
            if isinstance(request, CapabilitiesArguments):
                eligible = await CatalogRepository(
                    session, allowed_adapters=set(self.allowed_adapters)
                ).eligible_tools(principal)
                if not {item["tool_id"] for item in data["capabilities"]}.issubset(
                    {card["tool_id"] for card in eligible}
                ):
                    raise HTTPException(403, "Capabilities are no longer authorized")
            else:
                if "runs:read" not in principal.scopes:
                    raise HTTPException(403, "Run read permission is required")
                await RunRepository(session).get(UUID(data["run_id"]), principal)

    def __init__(
        self,
        database: Database,
        authenticator: TokenAuthenticator,
        *,
        allowed_adapters: set[str],
        telemetry: Telemetry | None = None,
    ) -> None:
        self.database = database
        self.authenticator = authenticator
        self.allowed_adapters = frozenset(allowed_adapters)
        self.telemetry = telemetry

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutcome:
        """Resolve last within owner/conversation, excluding this run; status bypasses caches."""
        if self.telemetry is None:
            return await self._invoke(arguments, context)
        with self.telemetry.observe(
            "tool.system_search",
            input=arguments,
            metadata={
                "run_id": str(context.run_id),
                "operation_id": str(context.operation_id),
            },
        ) as observation:
            result = await self._invoke(arguments, context)
            observation.update(output=result.data, metadata={"state": result.state})
            if result.error_code:
                observation.error(result.error_code)
            return result

    async def _invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutcome:
        try:
            request = SEARCH_ARGUMENTS.validate_python(arguments)
            principal = await self.authenticator.refresh(
                Principal(context.tenant_id, context.user_id, frozenset())
            )
            required = {"system:read"}
            if not isinstance(request, CapabilitiesArguments):
                required.add("runs:read")
            if not required.issubset(principal.scopes):
                raise HTTPException(403, "Required permissions are missing")
            async with self.database.session() as session:
                runs = RunRepository(session)
                current = await runs.get(context.run_id, principal)
                if current.user_id != context.user_id or (
                    context.conversation_id is not None
                    and context.conversation_id != current.conversation_id
                ):
                    raise HTTPException(403, "Tool context does not match run ownership")
                if isinstance(request, CapabilitiesArguments):
                    cards = await CatalogRepository(
                        session, allowed_adapters=set(self.allowed_adapters)
                    ).search_tools(
                        request.query, principal, limit=request.limit, category=request.category
                    )
                    result = CapabilitiesResult(
                        capabilities=[
                            Capability(
                                tool_id=card.tool_id,
                                name=card.name,
                                description=card.description,
                                operation_type=card.operation_type,
                                category_ids=card.category_ids,
                            )
                            for card in cards
                        ]
                    )
                else:
                    run = (
                        await runs.get(request.run_id, principal)
                        if request.run_id
                        else await runs.last(principal, context.conversation_id, context.run_id)
                    )
                    if run is None:
                        raise HTTPException(404, "No previous request found")
                    if isinstance(request, StatusArguments):
                        operations = await session.scalars(
                            select(ToolOperation)
                            .where(
                                ToolOperation.tenant_id == principal.tenant_id,
                                ToolOperation.run_id == run.run_id,
                            )
                            .order_by(ToolOperation.created_at, ToolOperation.operation_id)
                        )
                        operation_summaries = {
                            str(op.operation_id): OperationSummary(
                                operation_id=op.operation_id, state=op.status.value
                            )
                            for op in operations
                        }
                        appv2_events = await session.scalars(
                            select(WorkflowEvent)
                            .where(
                                WorkflowEvent.tenant_id == principal.tenant_id,
                                WorkflowEvent.run_id == run.run_id,
                                WorkflowEvent.event_type == "tool_dispatch_finished",
                            )
                            .order_by(WorkflowEvent.created_at, WorkflowEvent.event_id)
                        )
                        for event in appv2_events:
                            operation_id = (event.evidence or {}).get("operation_id")
                            state = (event.evidence or {}).get("state")
                            if operation_id and state:
                                operation_summaries[operation_id] = OperationSummary(
                                    operation_id=UUID(operation_id), state=state
                                )
                        result = StatusResult(
                            run_id=run.run_id,
                            run_created_at=run.created_at,
                            status=run.status.value,
                            summary=f"Request state: {run.status.value}.",
                            operations=list(operation_summaries.values()),
                            as_of=datetime.now(UTC),
                        )
                    else:
                        events = EventRepository(session)
                        page = await events.history(
                            run.run_id, principal, request.limit + 1, request.cursor
                        )
                        result = HistoryResult(
                            run_id=run.run_id,
                            run_created_at=run.created_at,
                            events=page[: request.limit],
                            next_cursor=events.cursor(page[request.limit - 1])
                            if len(page) > request.limit
                            else None,
                        )
            return ToolOutcome(state="succeeded", data=result.model_dump(mode="json"))
        except ValidationError:
            return ToolOutcome(
                state="failed", data={}, error_code="invalid_arguments", effect_known_absent=True
            )
        except HTTPException as error:
            code = {403: "forbidden", 404: "not_found", 422: "invalid_cursor"}.get(
                error.status_code
            )
            if code is None:
                raise
            return ToolOutcome(state="failed", data={}, error_code=code, effect_known_absent=True)

    async def reconcile(self, operation: dict[str, Any], context: ToolContext) -> ToolOutcome:
        return ToolOutcome(
            state="failed",
            data={},
            error_code="reconciliation_unsupported",
            effect_known_absent=True,
        )
