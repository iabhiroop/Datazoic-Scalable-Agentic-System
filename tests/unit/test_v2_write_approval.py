"""A model-proposed write must pause before dispatch, even in graph v2."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agent.nodes import AgentNodes
from app.agent.state import LiteralBinding, Plan, PlanStep
from app.db.models import OperationState


@pytest.mark.asyncio
async def test_v2_prepared_write_waits_for_bound_approval():
    run_id = uuid4()
    operation_id = uuid4()
    step = PlanStep(
        step_id="invoice-create",
        tool_id="invoice_create",
        schema_version="v1",
        arguments={"amount": LiteralBinding(literal="50.00")},
    )
    card = SimpleNamespace(
        tool_id="invoice_create",
        schema_version="v1",
        operation_type="write",
        input_schema={"type": "object", "properties": {"amount": {"type": "string"}}},
        name="Create invoice",
    )

    @asynccontextmanager
    async def session():
        yield SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(cancel_requested=False)))

    execution = SimpleNamespace(
        prepare_write=AsyncMock(
            return_value=SimpleNamespace(
                operation_id=operation_id, status=OperationState.PREPARED, input_hash="hash"
            )
        )
    )
    nodes = AgentNodes(
        database=SimpleNamespace(session=session),
        authenticator=None,
        model=None,
        retrieval=None,
        context_builder=None,
        registry=SimpleNamespace(get=AsyncMock(return_value=card)),
        validator=None,
        execution=execution,
        settings=None,
    )
    state = {"run_id": str(run_id), "plan": Plan(plan_version=1, steps=[step]).model_dump()}
    result = await nodes.prepare_step_v2(state, None)
    assert result["routing_decision"] == "wait"
    assert result["wait_request"]["kind"] == "approval"
    assert result["wait_request"]["operation_id"] == str(operation_id)
    assert result["wait_request"]["input_hash"] == "hash"
    execution.prepare_write.assert_awaited_once()

    approved = await nodes.prepare_step_v2(
        {**state, "approved_operation_id": str(operation_id)}, None
    )
    assert approved["routing_decision"] == "execute"
