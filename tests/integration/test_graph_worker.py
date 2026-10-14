"""Real PostgreSQL checkpoints with fake model decisions and registered read execution."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select, update

from app.agent.graph import build_graph
from app.agent.nodes import AgentNodes
from app.agent.planning import PlanValidator
from app.agent.state import Intent, LiteralBinding, Plan, PlanStep
from app.db.models import WorkflowInput, WorkflowRun
from app.db.session import Database, event_loop_factory
from app.knowledge.context import ContextBuilder
from app.main import create_app
from app.security import TokenAuthenticator
from app.tools.contracts import ToolCard
from app.tools.gateway import ExecutionService
from app.tools.registry import ToolRegistry
from app.tools.system_search import SystemSearch
from app.worker.main import Worker
from app.worker.recovery import RecoveryService
from app.worker.scheduler import Scheduler


@pytest.mark.parametrize("clarify,cancel", [(False, False), (True, False), (False, True)])
def test_queue_graph_clarification_resume_and_projection_repair(config, clarify, cancel):
    bindings = json.loads(config.auth_bindings_json.get_secret_value())
    bindings[0]["scopes"] += ["system:read", "tools:manage"]
    config.auth_bindings_json = SecretStr(json.dumps(bindings))
    config.embedding_model = "synthetic"
    config.embedding_version = "synthetic-v1"
    auth = TokenAuthenticator(config)
    principal = auth.authenticate("owner")
    path = Path(__file__).resolve().parents[2] / "resources/tool_cards/system_search.json"
    card = ToolCard.model_validate(json.loads(path.read_text())).model_copy(
        update={"embedding_model": "synthetic", "embedding_version": "synthetic-v1"}
    )
    intent = Intent(
        goal="request capabilities",
        subgoals=[],
        category_ids=["system"],
        entities={},
        constraints={},
        missing_fields=[],
        confidence=1,
    )
    model = AsyncMock()
    model.classify.side_effect = (
        [intent.model_copy(update={"missing_fields": ["details"]}), intent] if clarify else [intent]
    )
    model.plan.return_value = Plan(
        plan_version=1,
        steps=[
            PlanStep(
                step_id="lookup",
                tool_id=card.tool_id,
                schema_version="1",
                arguments={
                    "mode": LiteralBinding(literal="capabilities"),
                    "query": LiteralBinding(literal="request"),
                },
            )
        ],
    )
    model.respond.return_value = {
        "business_status": "completed",
        "summary": "Capabilities found.",
        "source_ids": [],
    }
    retrieval = AsyncMock()
    retrieval.tools.return_value = [card]
    owner = uuid4()
    headers = {"Authorization": "Bearer owner"}
    with TestClient(
        create_app(config), backend_options={"loop_factory": event_loop_factory}
    ) as client:
        submitted = client.post(
            "/runs",
            headers={**headers, "Idempotency-Key": str(uuid4())},
            json={"message": "Describe available capabilities."},
        )
        assert submitted.status_code == 202
        run_id = UUID(submitted.json()["run_id"])

        async def check():
            database = Database(config)
            try:
                await database.open()
                adapter = SystemSearch(database, auth, allowed_adapters={"system_search"})
                registry = ToolRegistry(
                    {"system_search": adapter},
                    database=database,
                    verification_keys={"system_search_result"},
                )
                await registry.save_draft(
                    card, principal, [1.0] + [0.0] * (config.embedding_dimension - 1)
                )
                await registry.publish(card, principal)
                nodes = AgentNodes(
                    database=database,
                    authenticator=auth,
                    model=model,
                    retrieval=retrieval,
                    context_builder=ContextBuilder(config),
                    registry=registry,
                    validator=PlanValidator(registry),
                    execution=ExecutionService(database, registry, auth),
                    settings=config,
                )
                graph = build_graph(await database.checkpointer(), nodes)
                recovery = RecoveryService(database, graph, config, owner)
                # Earlier independent tests may leave rows: select only this fixture for claiming.
                async with database.session() as session, session.begin():
                    run = await session.get(WorkflowRun, run_id, with_for_update=True)
                    run.worker_id = str(owner)
                    run.status = "running"
                if cancel:
                    assert client.post(f"/runs/{run_id}/cancel", headers=headers).status_code == 200
                snapshot = await graph.aget_state(recovery.config(run))
                decision = await recovery.choose_invocation(run, snapshot)
                assert decision.mode == "initial"
                await recovery.invoke(run, decision)
                if cancel:
                    result = client.get(f"/runs/{run_id}", headers=headers).json()
                    assert result["status"] == "cancelled"
                    model.classify.assert_not_awaited()
                    return
                if clarify:
                    status = client.get(f"/runs/{run_id}", headers=headers).json()
                    assert status["status"] == "waiting_for_input"
                    body = {
                        "wait_token": status["wait_token"],
                        "kind": "clarification",
                        "payload": {"answer": "List capabilities."},
                    }
                    assert (
                        client.post(
                            f"/runs/{run_id}/input", headers=headers, json={**body, "payload": {}}
                        ).status_code
                        == 422
                    )
                    assert (
                        client.post(f"/runs/{run_id}/input", headers=headers, json=body).status_code
                        == 202
                    )
                    assert (
                        client.post(f"/runs/{run_id}/input", headers=headers, json=body).status_code
                        == 202
                    )
                    assert (
                        client.post(
                            f"/runs/{run_id}/input",
                            headers=headers,
                            json={**body, "payload": {"answer": "Different"}},
                        ).status_code
                        == 409
                    )
                    async with database.session() as session, session.begin():
                        run = await session.get(WorkflowRun, run_id, with_for_update=True)
                        run.status = "running"
                        run.worker_id = str(owner)
                    snapshot = await graph.aget_state(recovery.config(run))
                    decision = await recovery.choose_invocation(run, snapshot)
                    assert decision.mode == "resume"
                    await recovery.invoke(run, decision)
                    async with database.session() as session:
                        response = await session.scalar(
                            select(WorkflowInput).where(WorkflowInput.run_id == run_id)
                        )
                        assert response.status == "applied"
                result = client.get(f"/runs/{run_id}", headers=headers).json()
                assert result["status"] == "completed"
                assert (
                    result["result"]["results"]["lookup"]["capabilities"][0]["tool_id"]
                    == "system_search"
                )
                # Simulate lost final projection, then repair without invoking model/tools again.
                calls = model.plan.await_count
                async with database.session() as session, session.begin():
                    await session.execute(
                        update(WorkflowRun)
                        .where(WorkflowRun.run_id == run_id)
                        .values(status="running", result_payload=None)
                    )
                snapshot = await graph.aget_state(recovery.config(run))
                decision = await recovery.choose_invocation(run, snapshot)
                assert decision.mode == "repair_final"
                await recovery.invoke(run, decision)
                assert model.plan.await_count == calls
                assert (
                    client.get(f"/runs/{run_id}", headers=headers).json()["status"] == "completed"
                )
            finally:
                await database.close()

        with asyncio.Runner(loop_factory=event_loop_factory) as runner:
            runner.run(check())


def test_worker_database_lock_excludes_second_worker(config):
    async def check():
        first, second = Worker(config), Worker(config)
        try:
            await first.acquire_ownership()
            await first.ensure_ownership()
            with pytest.raises(ValueError, match="Another worker"):
                await second.acquire_ownership()
        finally:
            if first.lock_connection:
                await first.lock_connection.close()
            if second.lock_connection:
                await second.lock_connection.close()

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        runner.run(check())


def test_scheduler_commits_claim_and_fences_heartbeat(config):
    from datetime import UTC, datetime

    from app.db.models import Conversation

    async def check():
        database = Database(config)
        owner = uuid4()
        tenant, conversation, identity = uuid4(), uuid4(), uuid4()
        try:
            await database.open()
            async with database.session() as session, session.begin():
                session.add(
                    Conversation(
                        tenant_id=tenant, conversation_id=conversation, user_id="test", title="test"
                    )
                )
                await session.flush()
                session.add(
                    WorkflowRun(
                        tenant_id=tenant,
                        run_id=identity,
                        user_id="test",
                        conversation_id=conversation,
                        thread_id=str(identity),
                        submission_key=str(identity),
                        request_hash="test",
                        request_payload={"message": "test"},
                        graph_version="v1",
                        prompt_version="v1",
                        status="queued",
                        created_at=datetime(1970, 1, 1, tzinfo=UTC),
                    )
                )
            scheduler = Scheduler(database)
            run = await scheduler.claim(owner)
            assert run.run_id == identity and run.worker_id == str(owner)
            assert await scheduler.heartbeat(owner, identity)
            assert not await scheduler.heartbeat(uuid4(), identity)
            async with database.session() as session:
                stored = await session.get(WorkflowRun, identity)
                assert stored.status == "running" and stored.heartbeat_at is not None
            recovery = RecoveryService(database, None, config, uuid4())
            # Only the caller-confirmed owner is eligible for startup repair.
            await recovery.repair_startup(uuid4())
            async with database.session() as session:
                assert (await session.get(WorkflowRun, identity)).status == "running"
            await recovery.repair_startup(owner)
            async with database.session() as session:
                assert (await session.get(WorkflowRun, identity)).status == "queued"
        finally:
            await database.close()

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        runner.run(check())


def test_worker_once_composes_services_and_releases_lock(config, monkeypatch):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    import app.worker.main as worker_module

    bindings = json.loads(config.auth_bindings_json.get_secret_value())
    bindings[0]["scopes"] += ["system:read", "tools:manage"]
    config.auth_bindings_json = SecretStr(json.dumps(bindings))
    config.embedding_model = "synthetic"
    config.embedding_version = "synthetic-v1"
    auth = TokenAuthenticator(config)
    principal = auth.authenticate("owner")
    provider = SimpleNamespace(
        model="synthetic",
        version="synthetic-v1",
        dimension=config.embedding_dimension,
        embed=AsyncMock(return_value=[1.0] + [0.0] * (config.embedding_dimension - 1)),
        aclose=AsyncMock(),
    )
    gateway = SimpleNamespace(aclose=AsyncMock())
    model = AsyncMock()
    model.classify.return_value = Intent(
        goal="request capabilities",
        subgoals=[],
        category_ids=[],
        entities={},
        constraints={},
        missing_fields=[],
        confidence=1,
    )
    model.plan.return_value = Plan(
        plan_version=1,
        steps=[
            PlanStep(
                step_id="lookup",
                tool_id="system_search",
                schema_version="1",
                arguments={
                    "mode": LiteralBinding(literal="capabilities"),
                    "query": LiteralBinding(literal="request"),
                },
            )
        ],
    )
    model.respond.return_value = {
        "summary": "Capabilities found.",
        "source_ids": [],
        "business_status": "completed",
    }
    monkeypatch.setattr(worker_module, "GeminiEmbeddingProvider", lambda _: provider)
    monkeypatch.setattr(worker_module, "create_llm_gateway", lambda *args, **kwargs: gateway)
    monkeypatch.setattr(worker_module, "ModelClient", lambda _: model)
    with TestClient(
        create_app(config), backend_options={"loop_factory": event_loop_factory}
    ) as client:
        identity = UUID(
            client.post(
                "/runs",
                headers={"Authorization": "Bearer owner", "Idempotency-Key": str(uuid4())},
                json={"message": "List capabilities."},
            ).json()["run_id"]
        )

        async def check():
            database = Database(config)
            await database.open()
            try:
                card = ToolCard.model_validate(
                    json.loads(
                        (
                            Path(__file__).resolve().parents[2]
                            / "resources/tool_cards/system_search.json"
                        ).read_text()
                    )
                ).model_copy(
                    update={"embedding_model": "synthetic", "embedding_version": "synthetic-v1"}
                )
                registry = ToolRegistry(
                    {"system_search"}, database=database, verification_keys={"system_search_result"}
                )
                await registry.save_draft(
                    card, principal, [1.0] + [0.0] * (config.embedding_dimension - 1)
                )
                await registry.publish(card, principal)

                async def fixture_claim(scheduler, owner):
                    async with scheduler.database.session() as session, session.begin():
                        run = await session.get(WorkflowRun, identity, with_for_update=True)
                        assert run.status == "queued"
                        run.status = "running"
                        run.worker_id = str(owner)
                        run.heartbeat_at = datetime.now(UTC)
                        await session.flush()
                        return run

                monkeypatch.setattr(Scheduler, "claim", fixture_claim)
                worker = Worker(config, once=True)
                await worker.run()
                assert worker.database.engine is None and worker.lock_connection is None
                gateway.aclose.assert_awaited_once()
                provider.aclose.assert_awaited_once()
                assert (
                    client.get(
                        f"/runs/{identity}", headers={"Authorization": "Bearer owner"}
                    ).json()["status"]
                    == "completed"
                )
                next_worker = Worker(config)
                await next_worker.acquire_ownership()
                await next_worker.lock_connection.close()
            finally:
                await database.close()

        with asyncio.Runner(loop_factory=event_loop_factory) as runner:
            runner.run(check())
