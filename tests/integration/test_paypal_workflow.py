"""Sandbox invoice flow with real checkpoints and a mocked PayPal HTTP server."""

import asyncio
import json
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select, update

from app.agent.graph import build_graph
from app.agent.nodes import AgentNodes
from app.agent.planning import PlanValidator
from app.agent.state import Intent, LiteralBinding, OutputBinding, Plan, PlanStep
from app.db.models import ToolOperation, WorkflowRun
from app.db.session import Database, event_loop_factory
from app.knowledge.context import ContextBuilder
from app.main import create_app
from app.security import TokenAuthenticator
from app.tools.contracts import ToolCard
from app.tools.gateway import ExecutionService
from app.tools.paypal import PayPalSandbox, adapter_key, invoice_adapters
from app.tools.registry import ToolRegistry
from app.worker.recovery import RecoveryService


def invoice_card(method, path, *, write):
    properties = {"body": {"type": "object"}} if write else {}
    if "{invoice_id}" in path:
        properties["invoice_id"] = {"type": "string", "minLength": 1}
    return ToolCard(
        tool_id=adapter_key(method, path),
        schema_version="1",
        adapter_key=adapter_key(method, path),
        name=f"PayPal {method} {path}",
        description=f"Sandbox invoice {method} {path}",
        action=method.lower(),
        input_schema={
            "type": "object",
            "properties": properties,
            "required": ["invoice_id"] if "{invoice_id}" in path else [],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"http_status": {"type": "integer"}, "response": {"type": "object"}},
            "required": ["http_status", "response"],
            "additionalProperties": False,
        },
        operation_type="write" if write else "read",
        required_scopes=["paypal:write" if write else "paypal:read"],
        precondition_key="paypal_reviewed_write" if write else None,
        verification_key="paypal_response",
        provider="paypal",
        provider_account_key="local_sandbox",
        embedding_model="synthetic",
        embedding_version="synthetic-v1",
    )


@pytest.mark.parametrize("send_timeout", [False, True])
def test_invoice_create_and_send_each_require_approval_and_dispatch_once(config, send_timeout):
    bindings = json.loads(config.auth_bindings_json.get_secret_value())
    bindings[0]["scopes"] += ["paypal:write", "paypal:read", "tools:manage"]
    config.auth_bindings_json = SecretStr(json.dumps(bindings))
    config.embedding_model = "synthetic"
    config.embedding_version = "synthetic-v1"
    config.paypal_sandbox_tenant_id = UUID(bindings[0]["tenant_id"])
    config.paypal_sandbox_client_id = SecretStr("mock-client")
    config.paypal_sandbox_client_secret = SecretStr("mock-secret")
    config.paypal_sandbox_writes_enabled = True
    auth = TokenAuthenticator(config)
    principal = auth.authenticate("owner")
    create = invoice_card("POST", "/v2/invoicing/invoices", write=True)
    send = invoice_card("POST", "/v2/invoicing/invoices/{invoice_id}/send", write=True)
    model = AsyncMock()
    model.classify.return_value = Intent(
        goal="create and send invoice",
        subgoals=[],
        category_ids=["paypal"],
        entities={},
        constraints={},
        missing_fields=[],
        confidence=1,
    )
    model.plan.return_value = Plan(
        plan_version=1,
        steps=[
            PlanStep(
                step_id="create",
                tool_id=create.tool_id,
                schema_version="1",
                arguments={
                    "body": LiteralBinding(
                        literal={
                            "primary_recipients": [
                                {"billing_info": {"email_address": "buyer@example.test"}}
                            ],
                            "items": [
                                {
                                    "name": "Widget",
                                    "quantity": "1",
                                    "unit_amount": {"currency_code": "USD", "value": "10.00"},
                                }
                            ],
                        }
                    ),
                },
            ),
            PlanStep(
                step_id="send",
                tool_id=send.tool_id,
                schema_version="1",
                depends_on=["create"],
                arguments={
                    "invoice_id": OutputBinding(from_step="create", path="/response/id"),
                },
            ),
        ],
    )
    model.respond.return_value = {
        "business_status": "completed",
        "summary": "Invoice sent.",
        "source_ids": [],
    }
    retrieval = AsyncMock()
    retrieval.tools.return_value = [create, send]
    calls = {"create": 0, "send": 0}

    def paypal_response(request):
        assert request.url.host == "api-m.sandbox.paypal.com"
        if request.url.path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "mock-token", "expires_in": 3600})
        assert request.headers.get("PayPal-Request-Id") or request.method == "GET"
        if request.method == "GET":
            return httpx.Response(200, json={"id": "INV2-TEST", "status": "DRAFT"})
        if request.url.path == "/v2/invoicing/invoices":
            calls["create"] += 1
            return httpx.Response(201, json={"id": "INV2-TEST", "status": "DRAFT"})
        calls["send"] += 1
        if send_timeout:
            raise httpx.ReadTimeout("synthetic uncertain send")
        return httpx.Response(202)

    headers = {"Authorization": "Bearer owner"}
    with TestClient(
        create_app(config), backend_options={"loop_factory": event_loop_factory}
    ) as api:
        submitted = api.post(
            "/runs",
            headers={**headers, "Idempotency-Key": str(uuid4())},
            json={"message": "Create and send a sandbox invoice."},
        )
        assert submitted.status_code == 202
        run_id = UUID(submitted.json()["run_id"])

        async def check():
            database = Database(config)
            client = httpx.AsyncClient(transport=httpx.MockTransport(paypal_response))
            try:
                await database.open()
                transport = PayPalSandbox(config, client)
                adapters = invoice_adapters(transport)
                registry = ToolRegistry(
                    adapters,
                    database=database,
                    verification_keys={"paypal_response"},
                    precondition_keys={"paypal_reviewed_write"},
                )
                for card in (create, send):
                    await registry.save_draft(card, principal, [1.0, 0.0, 0.0])
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
                owner = uuid4()
                graph = build_graph(await database.checkpointer(), nodes)
                recovery = RecoveryService(database, graph, config, owner)

                async def claim():
                    async with database.session() as session, session.begin():
                        run = await session.get(WorkflowRun, run_id, with_for_update=True)
                        run.worker_id = str(owner)
                        run.status = "running"
                        return run

                run = await claim()
                await recovery.invoke(
                    run,
                    await recovery.choose_invocation(
                        run, await graph.aget_state(recovery.config(run))
                    ),
                )
                first = api.get(f"/runs/{run_id}", headers=headers).json()
                assert first["wait_kind"] == "approval"
                assert calls == {"create": 0, "send": 0}
                assert (
                    api.post(
                        f"/runs/{run_id}/input",
                        headers=headers,
                        json={
                            "wait_token": first["wait_token"],
                            "kind": "approval",
                            "payload": {"approved": True},
                        },
                    ).status_code
                    == 202
                )
                run = await claim()
                await recovery.invoke(
                    run,
                    await recovery.choose_invocation(
                        run, await graph.aget_state(recovery.config(run))
                    ),
                )
                second = api.get(f"/runs/{run_id}", headers=headers).json()
                assert second["wait_kind"] == "approval"
                assert calls == {"create": 1, "send": 0}
                assert (
                    api.post(
                        f"/runs/{run_id}/input",
                        headers=headers,
                        json={
                            "wait_token": second["wait_token"],
                            "kind": "approval",
                            "payload": {"approved": True},
                        },
                    ).status_code
                    == 202
                )
                run = await claim()
                await recovery.invoke(
                    run,
                    await recovery.choose_invocation(
                        run, await graph.aget_state(recovery.config(run))
                    ),
                )
                assert api.get(f"/runs/{run_id}", headers=headers).json()["status"] == (
                    "needs_review" if send_timeout else "completed"
                )
                assert calls == {"create": 1, "send": 1}
                async with database.session() as session:
                    operations = list(
                        await session.scalars(
                            select(ToolOperation).where(ToolOperation.run_id == run_id)
                        )
                    )
                    assert len(operations) == 2
                    assert {operation.status.value for operation in operations} == (
                        {"succeeded", "unknown"} if send_timeout else {"succeeded"}
                    )
                if send_timeout:
                    return  # Ambiguous write requires review; never auto-dispatch again.
                async with database.session() as session, session.begin():
                    await session.execute(
                        update(WorkflowRun)
                        .where(WorkflowRun.run_id == run_id)
                        .values(status="running", result_payload=None)
                    )
                run = await claim()
                await recovery.invoke(
                    run,
                    await recovery.choose_invocation(
                        run, await graph.aget_state(recovery.config(run))
                    ),
                )
                assert calls == {"create": 1, "send": 1}
            finally:
                await client.aclose()
                await database.close()

        with asyncio.Runner(loop_factory=event_loop_factory) as runner:
            runner.run(check())
