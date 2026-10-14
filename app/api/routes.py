"""API wiring exists; business endpoints deliberately fail until implemented."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.api.runs import RunService
from app.api.schemas import (
    AcceptedRun,
    ConversationDetail,
    ConversationSummary,
    RunInput,
    RunDebugPage,
    RunStatus,
    SubmitRun,
)
from app.observability import Telemetry
from app.security import Principal, authenticated_principal

router = APIRouter()
Identity = Annotated[Principal, Depends(authenticated_principal)]


def run_service(request: Request) -> RunService:
    return request.app.state.run_service


Service = Annotated[RunService, Depends(run_service)]


@router.post("/runs", status_code=202, response_model=AcceptedRun)
async def submit_run(
    request: Request,
    body: SubmitRun,
    principal: Identity,
    service: Service,
    submission_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1)],
) -> AcceptedRun:
    accepted = await service.submit(
        body, principal, submission_key, trace_context=request.state.trace_context
    )
    Telemetry.update_current(run_id=str(accepted.run_id))
    return accepted


@router.get("/runs/{run_id}", response_model=RunStatus)
async def get_run(run_id: UUID, principal: Identity, service: Service) -> RunStatus:
    return await service.get_status(run_id, principal)


@router.get("/runs/{run_id}/debug", response_model=RunDebugPage)
async def get_run_debug(
    run_id: UUID, principal: Identity, service: Service, cursor: str | None = None
) -> RunDebugPage:
    return await service.get_debug_events(run_id, principal, cursor)


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(principal: Identity, service: Service) -> list[ConversationSummary]:
    return await service.list_conversations(principal)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: UUID, principal: Identity, service: Service
) -> ConversationDetail:
    return await service.get_conversation(conversation_id, principal)


@router.post("/runs/{run_id}/input", status_code=202, response_model=AcceptedRun)
async def input_run(
    run_id: UUID, body: RunInput, principal: Identity, service: Service
) -> AcceptedRun:
    return await service.accept_input(run_id, body, principal)


@router.post("/runs/{run_id}/cancel", response_model=RunStatus)
async def cancel_run(run_id: UUID, principal: Identity, service: Service) -> RunStatus:
    return await service.request_cancel(run_id, principal)


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready")
async def ready(request: Request) -> dict[str, str]:
    database = request.app.state.database
    if (
        database is None
        or not request.app.state.authenticator.configured
        or not await database.ready()
    ):
        raise HTTPException(
            status_code=503, detail="Required database/checkpoint/auth resources unavailable"
        )
    return {"status": "ready", "langfuse": request.app.state.telemetry.status}
