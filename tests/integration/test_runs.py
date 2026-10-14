"""Real transaction, ownership and cancellation checks against PostgreSQL."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.db.repositories import RunRepository
from app.db.session import Database, event_loop_factory
from app.main import create_app
from app.observability import Telemetry
from app.security import TokenAuthenticator


def headers(token="owner", key=None):
    values = {"Authorization": f"Bearer {token}"}
    if key is not None:
        values["Idempotency-Key"] = key
    return values


def submit(client, key="example", token="owner", **body):
    return client.post("/runs", headers=headers(token, key), json={"message": "Hello", **body})


def test_submission_replay_and_conflict(client, connection):
    first = submit(client)
    assert first.status_code == 202
    run_id = first.json()["run_id"]
    assert submit(client).json()["run_id"] == run_id
    assert submit(client, message="Changed").status_code == 409
    result = connection.execute(
        "SELECT count(*) FROM conversation_messages WHERE run_id = %s", (run_id,)
    ).fetchone()
    assert result[0] == 1
    assert (
        connection.execute(
            "SELECT count(*) FROM workflow_events WHERE run_id = %s", (run_id,)
        ).fetchone()[0]
        == 1
    )
    status = client.get(f"/runs/{run_id}", headers=headers()).json()
    assert status["status"] == "queued"
    assert client.get("/health/ready").status_code == 200


def test_concurrent_duplicate_has_one_conversation_message_and_run(client, connection, config):
    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(lambda _: submit(client, key="concurrent"), range(8)))
    assert {response.status_code for response in responses} == {202}
    assert len({response.json()["run_id"] for response in responses}) == 1
    tenant = TokenAuthenticator(config).authenticate("owner").tenant_id
    for table in ("conversations", "workflow_runs", "conversation_messages", "workflow_events"):
        # Table identifiers are a fixed test allowlist, never supplied by a model/client.
        assert (
            connection.execute(
                f"SELECT count(*) FROM {table} WHERE tenant_id = %s", (tenant,)
            ).fetchone()[0]
            == 1
        )


def test_concurrent_conflicting_body_rolls_back_losing_conversation(client, connection, config):
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(
            executor.map(lambda message: submit(client, key="race", message=message), ["A", "B"])
        )
    assert sorted(response.status_code for response in responses) == [202, 409]
    tenant = TokenAuthenticator(config).authenticate("owner").tenant_id
    assert (
        connection.execute(
            "SELECT count(*) FROM conversations WHERE tenant_id = %s", (tenant,)
        ).fetchone()[0]
        == 1
    )


def test_owner_and_tenant_boundaries_even_with_management_scope(client):
    run_id = submit(client).json()["run_id"]
    assert client.get(f"/runs/{run_id}", headers=headers("peer")).status_code == 404
    assert client.get(f"/runs/{run_id}", headers=headers("outsider")).status_code == 404
    assert client.get(f"/runs/{run_id}", headers=headers("manager")).status_code == 200
    assert submit(client, token="reader").status_code == 403
    assert client.post(f"/runs/{run_id}/cancel", headers=headers("peer")).status_code == 404


def test_existing_conversation_must_belong_to_submitter(client, connection):
    run_id = submit(client).json()["run_id"]
    conversation = connection.execute(
        "SELECT conversation_id FROM workflow_runs WHERE run_id = %s", (run_id,)
    ).fetchone()[0]
    assert (
        submit(
            client, key="peer-request", token="peer", conversation_id=str(conversation)
        ).status_code
        == 404
    )
    assert submit(client, key="owner-next", conversation_id=str(conversation)).status_code == 202


def test_cancellation_is_idempotent_for_queued_run(client, connection):
    run_id = submit(client).json()["run_id"]
    for _ in range(2):
        response = client.post(f"/runs/{run_id}/cancel", headers=headers())
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        assert response.json()["cancel_requested"] is True
    count = connection.execute(
        "SELECT count(*) FROM workflow_events WHERE run_id = %s AND event_type = 'cancel_requested'",
        (run_id,),
    ).fetchone()[0]
    assert count == 1


def test_cancellation_preserves_pending_write_evidence(client, connection):
    run_id = submit(client).json()["run_id"]
    tenant = connection.execute(
        "SELECT tenant_id FROM workflow_runs WHERE run_id = %s", (run_id,)
    ).fetchone()[0]
    connection.execute(
        """INSERT INTO tool_registry (tenant_id,tool_id,schema_version,card,searchable_text)
                          VALUES (%s,'refund','1','{}','refund')""",
        (tenant,),
    )
    operation = uuid4()
    connection.execute(
        """INSERT INTO tool_operations
        (operation_id,tenant_id,run_id,action_ref,tool_id,schema_version,operation_type,
         provider,provider_account_key,input_hash,input_payload,status)
        VALUES (%s,%s,%s,'refund_primary','refund','1','write','test','test','hash','{}','pending')""",
        (operation, tenant, run_id),
    )
    response = client.post(f"/runs/{run_id}/cancel", headers=headers())
    assert response.json()["status"] == "queued"
    assert response.json()["cancel_requested"] is True
    assert (
        connection.execute(
            "SELECT status FROM tool_operations WHERE operation_id = %s", (operation,)
        ).fetchone()[0]
        == "pending"
    )


def test_approval_input_remains_blocked_until_operation_binding(client):
    run_id = submit(client).json()["run_id"]
    response = client.post(
        f"/runs/{run_id}/input",
        headers=headers(),
        json={
            "wait_token": str(uuid4()),
            "kind": "approval",
            "payload": {},
        },
    )
    assert response.status_code == 409


def test_last_request_excludes_current_query_and_other_owners(client, config):
    previous = submit(client, key="previous").json()["run_id"]
    current = submit(client, key="current").json()["run_id"]
    submit(client, key="peer-latest", token="peer")

    async def lookup():
        database = Database(config)
        try:
            await database.open()
            principal = TokenAuthenticator(config).authenticate("owner")
            async with database.session() as session:
                run = await RunRepository(session).last(principal, None, UUID(current))
                return str(run.run_id)
        finally:
            await database.close()

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        assert runner.run(lookup()) == previous


def test_submission_persists_traceparent_without_exporting_to_external_service(
    config, connection, monkeypatch
):
    sdk = MagicMock()
    span = sdk.start_as_current_observation.return_value.__enter__.return_value
    span.trace_id = "a" * 32
    span.id = "b" * 16
    telemetry = Telemetry(config, sdk)
    monkeypatch.setattr("app.main.initialize_telemetry", AsyncMock(return_value=telemetry))
    with TestClient(
        create_app(config), backend_options={"loop_factory": event_loop_factory}
    ) as client:
        response = submit(client, key="traced")
        assert response.status_code == 202
        context = connection.execute(
            "SELECT trace_context FROM workflow_runs WHERE run_id = %s",
            (response.json()["run_id"],),
        ).fetchone()[0]
        assert context["request_id"] == response.headers["X-Request-ID"]
        assert context["traceparent"] == f"00-{'a' * 32}-{'b' * 16}-01"
    sdk.shutdown.assert_called_once()
