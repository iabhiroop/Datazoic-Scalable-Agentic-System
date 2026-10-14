"""Both graph versions require operation-bound approval before a write."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.agent.planning import canonical_payload_hash
from app.agent.state import LiteralBinding, PlanStep
from app.db.models import OperationState
from app.db.session import event_loop_factory
from app.security import Principal
from app.tools.contracts import ToolCard, ToolOutcome
from app.tools.gateway import ExecutionService


class Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def begin(self):
        return self


def run(coroutine):
    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        return runner.run(coroutine)


def test_write_requires_approval_in_both_graph_versions(monkeypatch):
    async def check():
        run_id = uuid4()
        tenant_id = uuid4()
        principal = Principal(tenant_id, "developer", frozenset({"runs:write", "payments:write"}))
        card = ToolCard(
            tool_id="create_record",
            schema_version="1",
            adapter_key="create_record",
            name="Create record",
            description="Create one record",
            action="create",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            operation_type="write",
            required_scopes=["payments:write"],
            precondition_key="sandbox_approved_write",
            verification_key="result",
            provider="sandbox",
            provider_account_key="sandbox-account",
        )
        step = PlanStep(
            step_id="s1-create_record",
            tool_id=card.tool_id,
            schema_version=card.schema_version,
            arguments={"value": LiteralBinding(literal="synthetic")},
        )
        arguments = {"value": "synthetic"}
        action_ref, operation_id = ExecutionService.write_identity(step, run_id)
        operation = SimpleNamespace(
            tenant_id=tenant_id,
            action_ref=action_ref,
            input_hash=canonical_payload_hash(arguments),
            status=OperationState.PREPARED,
            operation_id=operation_id,
            row_version=1,
            result_payload=None,
        )
        monkeypatch.setattr(
            "app.tools.gateway.OperationRepository.get", AsyncMock(return_value=operation)
        )
        monkeypatch.setattr(
            "app.tools.gateway.OperationRepository.mark_dispatching", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("app.tools.gateway.OperationRepository.save_outcome", AsyncMock())
        database = SimpleNamespace(session=Session)
        registry = SimpleNamespace(get=AsyncMock(return_value=card))
        service = ExecutionService(
            database, registry, SimpleNamespace(refresh=AsyncMock(return_value=principal))
        )
        service.check_preconditions = AsyncMock()
        successful = ToolOutcome(state="succeeded", data={"created": True})
        service.verify = AsyncMock(return_value=successful)
        adapter = SimpleNamespace(invoke=AsyncMock(return_value=successful))
        state = {"run_id": str(run_id), "graph_version": "v2", "results": {}}

        for version in ("v1", "v2"):
            unapproved = await service._execute_write(
                step, card, arguments, {**state, "graph_version": version}, principal, adapter, None
            )
            assert unapproved.error_code == "approval_required"
            adapter.invoke.assert_not_awaited()

        approved = await service._execute_write(
            step,
            card,
            arguments,
            {**state, "approved_operation_id": str(operation_id)},
            principal,
            adapter,
            None,
        )
        assert approved.state == "succeeded"
        adapter.invoke.assert_awaited_once()

    run(check())
