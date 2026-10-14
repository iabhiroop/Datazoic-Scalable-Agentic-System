"""Routing remains application-owned; cancellation avoids later model/tool nodes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.gateway import LLMGateway, LLMResponse
from app.agent.graph import NODE_NAMES, build_graph
from app.settings import Settings


def services(cancelled=False):
    value = SimpleNamespace(
        principal=AsyncMock(return_value=None), cancelled=AsyncMock(return_value=cancelled)
    )
    for name in NODE_NAMES:
        setattr(value, name, AsyncMock(return_value={}))
    value.respond.return_value = {"final_result": {"business_status": "cancelled"}}
    return value


async def test_unknown_route_is_rejected():
    value = services()
    value.classify.return_value = {"routing_decision": "unregistered_node"}
    graph = build_graph(InMemorySaver(), value)
    with pytest.raises(KeyError):
        await graph.ainvoke(
            {"run_id": str(uuid4())},
            {"configurable": {"thread_id": str(uuid4())}},
            durability="sync",
        )


async def test_cancelled_run_skips_classification_and_execution():
    value = services(cancelled=True)
    graph = build_graph(InMemorySaver(), value)
    result = await graph.ainvoke(
        {"run_id": str(uuid4())}, {"configurable": {"thread_id": str(uuid4())}}, durability="sync"
    )
    assert result["final_result"]["business_status"] == "cancelled"
    value.classify.assert_not_awaited()
    value.execute.assert_not_awaited()


async def test_graph_checkpoints_redacted_model_activity_per_run_node():
    value = services()
    value.settings = Settings(_env_file=None)
    gateway = LLMGateway(
        SimpleNamespace(
            generate=AsyncMock(
                return_value=LLMResponse(
                    '{"answer":"contact user@example.com"}',
                    "fake",
                    "test-model",
                    usage={"total_tokens": 5},
                )
            )
        )
    )

    async def classify(*_args):
        await gateway.generate("request", observation_name="classify")
        return {"routing_decision": "retrieve"}

    value.classify.side_effect = classify
    value.retrieve.return_value = {"routing_decision": "plan"}
    value.plan.return_value = {"routing_decision": "validate"}
    value.validate.return_value = {"routing_decision": "prepare_step"}
    value.prepare_step.return_value = {"routing_decision": "respond"}
    value.respond.return_value = {"final_result": {"business_status": "completed"}}
    graph = build_graph(InMemorySaver(), value)
    result = await graph.ainvoke(
        {"run_id": str(uuid4())},
        {"configurable": {"thread_id": str(uuid4())}},
        durability="sync",
    )
    assert len(result["llm_activity"]) == 1
    assert result["llm_activity"][0]["purpose"] == "classify"
    assert "[REDACTED]" in result["llm_activity"][0]["response"]
