"""Standalone System Search uses fresh authorization and scoped persisted evidence."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import SecretStr
from sqlalchemy import update

from app.db.models import Conversation, WorkflowEvent, WorkflowRun
from app.db.session import Database, event_loop_factory
from app.security import TokenAuthenticator
from app.tools.contracts import ToolCard, ToolContext
from app.tools.registry import ToolRegistry
from app.tools.system_search import SEARCH_RESULTS, SystemSearch


def test_search_scope_status_history_pagination_and_revocation(config):
    bindings = json.loads(config.auth_bindings_json.get_secret_value())
    bindings[0]["scopes"] += ["system:read", "tools:manage"]
    config.auth_bindings_json = SecretStr(json.dumps(bindings))
    auth = TokenAuthenticator(config)
    principal = auth.authenticate("owner")
    config.embedding_model = "synthetic"
    config.embedding_version = "synthetic-v1"
    current_id, previous_id, peer_id, other_id, wide_id = [uuid4() for _ in range(5)]
    conversation, wide_conversation, other_conversation = [uuid4() for _ in range(3)]
    other_tenant = uuid4()
    now = datetime.now(UTC)
    context = ToolContext(
        principal.tenant_id, principal.user_id, current_id, uuid4(), None, conversation
    )
    path = Path(__file__).resolve().parents[2] / "resources/tool_cards/system_search.json"
    card = ToolCard.model_validate(json.loads(path.read_text())).model_copy(
        update={"embedding_model": "synthetic", "embedding_version": "synthetic-v1"}
    )

    async def check():
        database = Database(config)
        try:
            await database.open()
            registry = ToolRegistry(
                {"system_search": object()},
                database=database,
                verification_keys={"system_search_result"},
            )
            await registry.save_draft(
                card, principal, [1.0] + [0.0] * (config.embedding_dimension - 1)
            )
            await registry.publish(card, principal)
            async with database.session() as session, session.begin():
                session.add_all(
                    [
                        Conversation(
                            tenant_id=principal.tenant_id,
                            conversation_id=conversation,
                            user_id="owner",
                            title="test",
                        ),
                        Conversation(
                            tenant_id=principal.tenant_id,
                            conversation_id=wide_conversation,
                            user_id="owner",
                            title="test",
                        ),
                        Conversation(
                            tenant_id=other_tenant,
                            conversation_id=other_conversation,
                            user_id="other",
                            title="test",
                        ),
                    ]
                )
                await session.flush()
                for identity, tenant, user, convo, age in [
                    (previous_id, principal.tenant_id, "owner", conversation, 4),
                    (wide_id, principal.tenant_id, "owner", wide_conversation, 3),
                    (peer_id, principal.tenant_id, "peer", conversation, 2),
                    (other_id, other_tenant, "other", other_conversation, 1),
                    (current_id, principal.tenant_id, "owner", conversation, 0),
                ]:
                    session.add(
                        WorkflowRun(
                            run_id=identity,
                            tenant_id=tenant,
                            user_id=user,
                            conversation_id=convo,
                            thread_id=str(identity),
                            submission_key=str(identity),
                            request_hash="synthetic",
                            request_payload={},
                            graph_version="v1",
                            prompt_version="v1",
                            status="queued",
                            created_at=now - timedelta(minutes=age),
                        )
                    )
                await session.flush()
                for index in range(3):
                    session.add(
                        WorkflowEvent(
                            tenant_id=principal.tenant_id,
                            run_id=previous_id,
                            event_type="test",
                            created_at=now,
                            evidence={
                                "index": index,
                                "api_key": "synthetic-private",
                                "message": "Bearer synthetic-private",
                            },
                        )
                    )
            tool = SystemSearch(database, auth, allowed_adapters={"system_search"})
            status = await tool.invoke({"mode": "request_status"}, context)
            assert status.state == "succeeded"
            SEARCH_RESULTS.validate_python(status.data)
            assert status.data["run_id"] == str(previous_id)
            wide_context = ToolContext(principal.tenant_id, "owner", current_id, uuid4(), None)
            wide = await tool.invoke({"mode": "request_status"}, wide_context)
            assert wide.data["run_id"] == str(wide_id)
            for target in (peer_id, other_id):
                denied = await tool.invoke(
                    {"mode": "request_status", "run_id": str(target)}, context
                )
                assert denied.error_code == "not_found"
            async with database.session() as session, session.begin():
                await session.execute(
                    update(WorkflowRun)
                    .where(WorkflowRun.run_id == previous_id)
                    .values(status="needs_review")
                )
            status = await tool.invoke({"mode": "request_status"}, context)
            assert status.data["status"] == "needs_review"
            assert "needs_review" in status.data["summary"]
            first = await tool.invoke({"mode": "execution_history", "limit": 2}, context)
            SEARCH_RESULTS.validate_python(first.data)
            assert len(first.data["events"]) == 2 and first.data["next_cursor"]
            second = await tool.invoke(
                {"mode": "execution_history", "limit": 2, "cursor": first.data["next_cursor"]},
                context,
            )
            assert len(second.data["events"]) == 1 and second.data["next_cursor"] is None
            events = first.data["events"] + second.data["events"]
            assert len({event["event_id"] for event in events}) == 3
            assert "synthetic-private" not in json.dumps(events)
            assert {event["evidence"]["index"] for event in events} == {0, 1, 2}
            invalid = await tool.invoke({"mode": "execution_history", "cursor": "invalid"}, context)
            assert invalid.error_code == "invalid_cursor"
            wrong_run_cursor = await tool.invoke(
                {
                    "mode": "execution_history",
                    "run_id": str(wide_id),
                    "cursor": first.data["next_cursor"],
                },
                context,
            )
            assert wrong_run_cursor.error_code == "invalid_cursor"
            capabilities = await tool.invoke({"mode": "capabilities", "query": "request"}, context)
            assert [item["tool_id"] for item in capabilities.data["capabilities"]] == [
                "system_search"
            ]
            forged = ToolContext(principal.tenant_id, "owner", peer_id, uuid4(), None, conversation)
            assert (await tool.invoke({"mode": "request_status"}, forged)).error_code == "not_found"
            ambiguous = await tool.invoke(
                {"mode": "request_status", "run_id": str(previous_id), "selector": "last"}, context
            )
            assert ambiguous.error_code == "invalid_arguments"
            auth.bindings[0].scopes = auth.bindings[0].scopes | {"runs:manage"}
            managed = await tool.invoke({"mode": "request_status", "run_id": str(peer_id)}, context)
            assert managed.data["run_id"] == str(peer_id)
            assert (await tool.invoke({"mode": "request_status"}, context)).data["run_id"] == str(
                previous_id
            )
            assert (
                await tool.invoke({"mode": "request_status", "run_id": str(other_id)}, context)
            ).error_code == "not_found"
            assert (await tool.reconcile({}, context)).error_code == "reconciliation_unsupported"
            auth.bindings[0].scopes = frozenset({"runs:read"})
            assert (
                await tool.invoke({"mode": "request_status"}, context)
            ).error_code == "forbidden"
        finally:
            await database.close()

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        runner.run(check())
