"""Inspect actual checkpoints and never restart an existing run from its request."""

from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from langgraph.types import Command
from sqlalchemy import select, update

from app.agent.checkpoints import wait_request_from_snapshot
from app.agent.state import ResumeEnvelope
from app.db.models import ToolOperation, WorkflowRun
from app.db.repositories import InputRepository, RunRepository


@dataclass(frozen=True)
class InvocationDecision:
    mode: Literal["initial", "continue", "resume", "repair_wait", "repair_final", "review"]
    resume: ResumeEnvelope | None = None


class RecoveryService:
    def __init__(self, database, graph, settings, worker_id):
        self.database = database
        self.graphs = (
            graph if isinstance(graph, dict) else {settings.graph_version: graph, "v1": graph}
        )
        self.settings = settings
        self.worker_id = str(worker_id)

    def config(self, run):
        return {
            "configurable": {
                "thread_id": run.thread_id,
                "run_id": str(run.run_id),
                "worker_id": self.worker_id,
            },
            "recursion_limit": self.settings.graph_recursion_limit,
        }

    async def repair_startup(self, terminated_worker_id: UUID):
        # Caller must obtain the singleton lock and explicit operator confirmation first.
        async with self.database.session() as session, session.begin():
            rows = await session.scalars(
                select(WorkflowRun)
                .where(
                    WorkflowRun.worker_id == str(terminated_worker_id),
                    WorkflowRun.status == "running",
                )
                .with_for_update()
            )
            for run in rows:
                unresolved_write = await session.scalar(
                    select(ToolOperation.operation_id)
                    .where(
                        ToolOperation.tenant_id == run.tenant_id,
                        ToolOperation.run_id == run.run_id,
                        ToolOperation.operation_type == "write",
                        ToolOperation.status.in_(["dispatching", "pending", "unknown"]),
                    )
                    .limit(1)
                )
                if unresolved_write:
                    run.status = "needs_review"
                    run.error_payload = {"code": "write_recovery_not_implemented"}
                elif run.cancel_requested:
                    run.status = "cancelled"
                else:
                    run.status = "queued"
                run.worker_id = None
                run.row_version += 1

    async def choose_invocation(self, run, snapshot: Any):
        async with self.database.session() as session:
            unresolved_write = await session.scalar(
                select(ToolOperation.operation_id)
                .where(
                    ToolOperation.tenant_id == run.tenant_id,
                    ToolOperation.run_id == run.run_id,
                    ToolOperation.operation_type == "write",
                    ToolOperation.status.in_(["dispatching", "pending", "unknown"]),
                )
                .limit(1)
            )
        if (
            unresolved_write
            or run.graph_version not in self.graphs
            or (run.graph_version, run.prompt_version)
            not in {
                ("v1", "v1"), ("v2", "v1"), ("v2", "v2"), ("v3", "v2"), ("v4", "v2")
            }
        ):
            return InvocationDecision("review")
        values = snapshot.values or {}
        if not values:
            return InvocationDecision("initial")
        if values.get("final_result") and not snapshot.next:
            return InvocationDecision("repair_final")
        if any(task.interrupts for task in snapshot.tasks):
            wait = wait_request_from_snapshot(snapshot)
            if wait is None:
                return InvocationDecision("review")
            if wait.kind not in {"clarification", "approval"}:
                return InvocationDecision("review")
            async with self.database.session() as session:
                response = await InputRepository(session).pending(
                    run.tenant_id, run.run_id, UUID(wait.wait_token)
                )
                if response:
                    return InvocationDecision(
                        "resume",
                        ResumeEnvelope(
                            input_id=str(response.input_id),
                            wait_token=wait.wait_token,
                            kind=response.kind,
                            payload=response.payload,
                        ),
                    )
            return InvocationDecision("repair_wait")
        return InvocationDecision("continue" if snapshot.next else "review")

    async def acknowledge_consumed_input(self, run, snapshot):
        identity = (snapshot.values or {}).get("last_input_id")
        if identity:
            async with self.database.session() as session, session.begin():
                await InputRepository(session).acknowledge(
                    run.tenant_id, UUID(identity), run_id=run.run_id
                )

    async def invoke(self, run, decision):
        if decision.mode == "review":
            async with self.database.session() as session, session.begin():
                await session.execute(
                    update(WorkflowRun)
                    .where(
                        WorkflowRun.run_id == run.run_id, WorkflowRun.worker_id == self.worker_id
                    )
                    .values(
                        status="needs_review",
                        error_payload={"code": "unsupported_recovery"},
                        row_version=WorkflowRun.row_version + 1,
                    )
                )
            return
        config = self.config(run)
        if decision.mode in ("initial", "continue", "resume"):
            value = {
                "run_id": str(run.run_id),
                "worker_id": self.worker_id,
                "thread_id": run.thread_id,
                "conversation_id": str(run.conversation_id),
                "request_text": run.request_payload["message"],
                "graph_version": run.graph_version,
            }
            if decision.mode == "continue":
                value = None
            elif decision.mode == "resume":
                value = Command(resume=decision.resume.model_dump())
            await self.graphs[run.graph_version].ainvoke(value, config, durability="sync")
        graph = self.graphs[run.graph_version]
        snapshot = await graph.aget_state(config)
        await self.acknowledge_consumed_input(run, snapshot)
        async with self.database.session() as session, session.begin():
            await RunRepository(session).project_status(
                run.tenant_id, run.run_id, snapshot, worker_id=self.worker_id
            )
