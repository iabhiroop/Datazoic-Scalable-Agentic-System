"""One fenced worker with versioned graph execution and explicit prior-owner recovery."""

import argparse
import asyncio
import os
from contextlib import AsyncExitStack
from uuid import UUID, uuid4

import psycopg
from sqlalchemy import select, update

from app.agent.debug import mask_known_credentials
from app.agent.gateway import create_llm_gateway
from app.agent.gemini import GeminiEmbeddingProvider
from app.agent.graph import build_graph
from app.agent.llm import ModelClient
from app.agent.nodes import AgentNodes
from app.agent.planning import PlanValidator
from app.db.models import WorkflowEvent, WorkflowRun
from app.db.session import Database, event_loop_factory
from app.knowledge.context import ContextBuilder
from app.knowledge.retrieval import RetrievalService
from app.observability import Telemetry, TraceContext
from app.security import TokenAuthenticator
from app.settings import Settings
from app.tools.gateway import ExecutionService
from app.tools.paypal import PayPalSandbox, catalog_adapters
from app.tools.rag_pipeline import RagPipeline
from app.tools.registry import ToolRegistry
from app.tools.system_search import SystemSearch
from app.worker.recovery import RecoveryService
from app.worker.scheduler import Scheduler

LOCK_KEY = 1794312057


class PriorOwnerConfirmationRequired(ValueError):
    def __init__(self, owners: set[str]):
        self.owners = sorted(owners)
        super().__init__("Prior worker confirmation is required")


class Worker:
    def __init__(self, settings=None, *, recover_workers=(), once=False):
        self.settings = settings or Settings()
        self.worker_id = uuid4()
        self.recover_workers = set(recover_workers)
        self.once = once
        self.process_id = os.getpid()
        self.stopping = asyncio.Event()
        self.lock_connection = None
        self.database = None

    async def acquire_ownership(self):
        if not self.settings.checkpoint_database_url:
            raise ValueError("Worker database is not configured")
        self.lock_connection = await psycopg.AsyncConnection.connect(
            self.settings.checkpoint_database_url.get_secret_value(),
            autocommit=True,
            connect_timeout=5,
            application_name=(
                f"dzw:{self.worker_id}:{self.process_id}:"
                f"{self.settings.graph_version}:{self.settings.prompt_version}"
            ),
        )
        result = await self.lock_connection.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
        if not (await result.fetchone())[0]:
            await self.lock_connection.close()
            self.lock_connection = None
            raise ValueError("Another worker owns the database lock")
        return self.worker_id

    async def ensure_ownership(self):
        if self.stopping.is_set():
            raise asyncio.CancelledError
        if self.lock_connection is None or self.lock_connection.closed:
            raise RuntimeError("Worker ownership connection is unavailable")
        result = await self.lock_connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE "
            "locktype='advisory' AND pid=pg_backend_pid() AND classid=0 AND objid=%s AND granted)",
            (LOCK_KEY,),
        )
        if not (await result.fetchone())[0]:
            raise RuntimeError("Worker ownership lock was lost")

    async def run(self):
        if (self.settings.graph_version, self.settings.prompt_version) not in {
            ("v1", "v1"), ("v2", "v1"), ("v2", "v2"), ("v3", "v2"), ("v4", "v2")
        }:
            raise ValueError("No handler is installed for these executable versions")
        async with AsyncExitStack() as resources:
            try:
                await self.acquire_ownership()
                resources.push_async_callback(self.lock_connection.close)
                self.database = Database(self.settings)
                resources.push_async_callback(self.database.close)
                await self.database.open()
                telemetry = Telemetry.from_settings(self.settings)
                resources.push_async_callback(telemetry.aclose)
                auth = TokenAuthenticator(self.settings)
                embeddings = GeminiEmbeddingProvider(self.settings)
                resources.push_async_callback(embeddings.aclose)
                gateway = create_llm_gateway(self.settings, telemetry=telemetry)
                resources.push_async_callback(gateway.aclose)
                model = ModelClient(gateway)
                allowed = {"system_search", "rag_pipeline"}
                retrieval = RetrievalService(
                    embeddings,
                    database=self.database,
                    settings=self.settings,
                    allowed_adapters=allowed,
                )
                context = ContextBuilder(self.settings)
                adapters = {
                    "system_search": SystemSearch(
                        self.database, auth, allowed_adapters=allowed, telemetry=telemetry
                    ),
                    "rag_pipeline": RagPipeline(
                        retrieval, context, auth, model, telemetry=telemetry
                    ),
                }
                if (
                    self.settings.paypal_sandbox_client_id
                    and self.settings.paypal_sandbox_client_secret
                    and self.settings.paypal_sandbox_tenant_id
                ):
                    paypal = PayPalSandbox(self.settings)
                    resources.push_async_callback(paypal.aclose)
                    adapters.update(
                        {
                            key: adapter
                            for key, adapter in catalog_adapters(paypal).items()
                            if adapter.endpoint.operation_type == "read"
                            or self.settings.paypal_sandbox_writes_enabled
                        }
                    )
                    allowed.update(adapters)
                    retrieval.allowed_adapters = frozenset(allowed)
                    adapters["system_search"].allowed_adapters = frozenset(allowed)
                registry = ToolRegistry(
                    adapters,
                    database=self.database,
                    verification_keys={"system_search_result", "rag_result", "paypal_response"},
                    precondition_keys={"paypal_reviewed_write", "sandbox_approved_write"}
                    if self.settings.paypal_sandbox_writes_enabled
                    else set(),
                )
                nodes = AgentNodes(
                    database=self.database,
                    authenticator=auth,
                    model=model,
                    retrieval=retrieval,
                    context_builder=context,
                    registry=registry,
                    validator=PlanValidator(registry, max_steps=self.settings.max_plan_steps),
                    execution=ExecutionService(self.database, registry, auth),
                    settings=self.settings,
                    ownership_check=self.ensure_ownership,
                    telemetry=telemetry,
                )
                checkpointer = await self.database.checkpointer()
                graphs = {
                    "v1": build_graph(checkpointer, nodes, version="v1"),
                    "v2": build_graph(checkpointer, nodes, version="v2"),
                    "v3": build_graph(checkpointer, nodes, version="v3"),
                }
                from app.agent.chat_models import create_chat_model
                from app.agent.graph_v4 import AgentGraphV4

                graphs["v4"] = AgentGraphV4(
                    nodes, self.settings, create_chat_model(self.settings)
                ).build(checkpointer)
                recovery = RecoveryService(self.database, graphs, self.settings, self.worker_id)
                scheduler = Scheduler(self.database)
                async with self.database.session() as session:
                    owners = set(
                        await session.scalars(
                            select(WorkflowRun.worker_id).where(WorkflowRun.status == "running")
                        )
                    )
                unconfirmed = owners - {str(owner) for owner in self.recover_workers}
                if unconfirmed:
                    raise PriorOwnerConfirmationRequired(unconfirmed)
                for owner in self.recover_workers:
                    await recovery.repair_startup(owner)
                print(
                    f"Agent worker started (graph {self.settings.graph_version}):", self.worker_id
                )
                while not self.stopping.is_set():
                    await self.ensure_ownership()
                    run = await scheduler.claim(self.worker_id)
                    if run is None:
                        if self.once:
                            break
                        try:
                            await asyncio.wait_for(
                                self.stopping.wait(), self.settings.worker_poll_seconds
                            )
                        except TimeoutError:
                            pass
                        continue
                    heartbeat = asyncio.create_task(self.beat(scheduler, run.run_id))
                    try:
                        with (
                            telemetry.bind_context(
                                TraceContext(
                                    run_id=str(run.run_id),
                                    conversation_id=str(run.conversation_id),
                                    traceparent=(run.trace_context or {}).get("traceparent"),
                                )
                            ),
                            telemetry.observe(
                                "worker.run", metadata={"worker_id": str(self.worker_id)}
                            ),
                        ):
                            graph = recovery.graphs.get(run.graph_version, recovery.graphs["v1"])
                            snapshot = await graph.aget_state(recovery.config(run))
                            decision = await recovery.choose_invocation(run, snapshot)
                            await recovery.invoke(run, decision)
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:  # noqa: BLE001 -- expose owner-visible debug details
                        await self.ensure_ownership()
                        async with self.database.session() as session, session.begin():
                            session.add(
                                WorkflowEvent(
                                    tenant_id=run.tenant_id,
                                    run_id=run.run_id,
                                    event_type="debug_worker_failed",
                                    evidence={
                                        "error_type": type(error).__name__,
                                        "error_detail": mask_known_credentials(
                                            str(error), self.settings
                                        ),
                                    },
                                )
                            )
                            await session.execute(
                                update(WorkflowRun)
                                .where(
                                    WorkflowRun.run_id == run.run_id,
                                    WorkflowRun.worker_id == str(self.worker_id),
                                    WorkflowRun.status == "running",
                                )
                                .values(
                                    status="failed",
                                    error_payload={
                                        "code": "worker_invocation_failed",
                                        "error_type": type(error).__name__,
                                    },
                                    row_version=WorkflowRun.row_version + 1,
                                )
                            )
                    finally:
                        heartbeat.cancel()
                        await asyncio.gather(heartbeat, return_exceptions=True)
                    if self.once:
                        break
            finally:
                self.lock_connection = None

    async def beat(self, scheduler, run_id):
        while True:
            await asyncio.sleep(self.settings.worker_heartbeat_seconds)
            await self.ensure_ownership()
            await scheduler.heartbeat(self.worker_id, run_id)

    async def stop(self):
        self.stopping.set()


def main():
    parser = argparse.ArgumentParser(description="Single read-only agent worker")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--recover-worker",
        type=UUID,
        action="append",
        default=[],
        help="Confirm this previous worker process has stopped before recovery",
    )
    args = parser.parse_args()
    try:
        with asyncio.Runner(loop_factory=event_loop_factory) as runner:
            runner.run(Worker(recover_workers=args.recover_worker, once=args.once).run())
    except PriorOwnerConfirmationRequired as error:
        parser.exit(
            status=2,
            message="Prior-owner recovery required: " + ", ".join(error.owners) + "\n",
        )
    except ValueError:
        parser.exit(
            status=2,
            message="Worker startup failed; verify configuration and prior-owner recovery confirmation.\n",
        )


if __name__ == "__main__":
    main()
