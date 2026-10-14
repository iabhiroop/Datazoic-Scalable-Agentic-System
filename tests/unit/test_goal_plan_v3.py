"""The v3 contract is small and actions are independently retrieved."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agent.nodes import AgentNodes
from app.agent.state import GoalPlan, ToolDeclaration


def test_goal_plan_contract_is_small_and_rejects_extra_fields():
    plan = GoalPlan.model_validate({
        "user_goal": "List invoices related to the user",
        "tool_actions": ["list invoices"],
        "missing_information": [],
    })
    assert plan.tool_actions == ["list invoices"]
    with pytest.raises(ValueError):
        GoalPlan.model_validate({**plan.model_dump(), "execution_mode": "execute"})


def make_nodes():
    nodes = AgentNodes.__new__(AgentNodes)
    nodes.model = SimpleNamespace(classify_goal=AsyncMock(return_value=GoalPlan(
        user_goal="Find invoices", tool_actions=["list invoices"], missing_information=[])))
    nodes.settings = SimpleNamespace(max_plan_steps=10, candidate_tools_per_subgoal=5)
    nodes.record_debug_event = AsyncMock()
    nodes._test_record_debug_event = nodes.record_debug_event
    nodes.retrieval = SimpleNamespace(tools=AsyncMock(return_value=[]))
    return nodes


async def test_classification_emits_goal_plan_before_retrieval_and_missing_info_waits():
    nodes = make_nodes()
    nodes._record_debug_event = AsyncMock()
    nodes.record_debug_event = nodes._record_debug_event
    result = await nodes.classify_v3({"request_text": "list invoices", "run_id": str(uuid4())}, None)
    assert result["routing_decision"] == "retrieve"
    nodes._record_debug_event.assert_awaited_once()
    nodes.model.classify_goal.return_value = GoalPlan(
        user_goal="Find invoice", tool_actions=["get invoice"], missing_information=["invoice ID"])
    result = await nodes.classify_v3({"request_text": "show invoice", "run_id": str(uuid4())}, None)
    assert result["routing_decision"] == "prepare_wait"


async def test_retrieval_uses_goal_and_current_action_only():
    nodes = make_nodes()
    nodes.record_debug_event = AsyncMock()
    nodes.goal = None
    nodes.retrieval.tools.return_value = []
    state = {"goal_plan": {"user_goal": "Find invoices", "tool_actions": ["list invoices"], "missing_information": []},
             "goal_action_index": 0, "run_id": str(uuid4())}
    result = await nodes.retrieve_v3(state, None)
    args = nodes.retrieval.tools.await_args.args
    assert args[0] == "Find invoices list invoices"
    assert result["observation"]["error_code"] == "no_authorized_tools"


async def test_tool_selection_has_names_and_descriptions_in_prompt_and_schemas_as_declarations():
    from app.agent.state import FunctionCall
    nodes = make_nodes()
    card = SimpleNamespace(tool_id="tool.list", schema_version="1", name="List invoices",
        description="Find invoices", input_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
        operation_type="read", action="list", model_dump=lambda **_: {})
    nodes.registry = SimpleNamespace()
    nodes.settings.max_plan_steps = 5
    nodes.model.select_tool = AsyncMock(return_value=SimpleNamespace(
        calls=[FunctionCall(name="tool.list", arguments={"limit": 5}, call_id="call-1")],
        text="", continuation=[]))
    # Stop at the existing plan validator boundary to inspect the model inputs.
    nodes.validator = SimpleNamespace(validate=AsyncMock(side_effect=ValueError("stop after selection")))
    nodes.record_debug_event = AsyncMock()
    # Ensure card parsing and call mapping use the same simple card representation.
    import app.agent.nodes as module
    original = module.ToolCard
    module.ToolCard = SimpleNamespace(model_validate=lambda _: card)
    try:
        await nodes.select_v3({
            "goal_plan": {"user_goal": "Find invoices", "tool_actions": ["list invoices"], "missing_information": []},
            "goal_action_index": 0, "candidate_tools": [{"tool_id": "tool.list", "schema_version": "1"}],
            "request_text": "list invoices", "run_id": str(uuid4()), "results": {},
        }, None)
        args, declarations, prior = nodes.model.select_tool.await_args.args
    finally:
        module.ToolCard = original
    assert args["user_goal"] == "Find invoices"
    assert args["suggested_tools"] == [{"name": "List invoices", "description": "Find invoices"}]
    assert declarations[0].parameters == card.input_schema
    assert prior == {}
