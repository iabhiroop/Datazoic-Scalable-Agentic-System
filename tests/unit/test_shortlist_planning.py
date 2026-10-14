"""Planning receives only retrieved tools and never dispatches invalid calls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.agent.gateway import LLMError
from app.agent.nodes import AgentNodes
from app.agent.planning import PlanValidator
from app.agent.state import FunctionCall, FunctionTurn, Intent, Plan, Subgoal
from app.db.session import event_loop_factory
from app.knowledge.context import ContextBuilder
from app.security import Principal
from app.settings import Settings
from app.tools.contracts import ToolCard


def tool_card():
    return ToolCard(
        tool_id="lookup",
        schema_version="1",
        adapter_key="lookup",
        name="Lookup",
        description="Read one record",
        action="read",
        input_schema={
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "properties": {"record": {"type": "object"}}},
        operation_type="read",
        required_scopes=["records:read"],
        precondition_key=None,
        verification_key="lookup_result",
        provider="internal",
        provider_account_key="internal",
    )


def intent():
    return Intent(
        goal="Read a record",
        subgoals=[],
        category_ids=[],
        entities={},
        constraints={},
        missing_fields=[],
        confidence=1,
    )


def nodes(settings, *, database=None, retrieval=None, model=None, context_builder=None):
    instance = AgentNodes.__new__(AgentNodes)
    instance.settings = settings
    instance.database = database
    instance.registry = SimpleNamespace(adapters={"lookup": object()})
    instance.retrieval = retrieval
    instance.model = model
    instance.context_builder = context_builder
    return instance


def run(coroutine):
    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        return runner.run(coroutine)


def test_retrieval_keeps_candidate_limit():
    async def check():
        card = tool_card()
        retrieval = SimpleNamespace(tools=AsyncMock(return_value=[card]))
        configured = Settings(
            _env_file=None,
            candidate_tools_per_subgoal=3,
        )
        node = nodes(configured, retrieval=retrieval)
        principal = Principal(uuid4(), "reader", frozenset())

        result = await node.retrieve({"intent": intent().model_dump()}, principal)

        retrieval.tools.assert_awaited_once_with("Read a record", [], principal, 3)
        assert result["candidate_tools"] == [card.model_dump(mode="json")]

    run(check())


def test_planner_receives_retrieved_schemas_and_provider_failure_is_safe():
    async def check():
        card = tool_card()
        model = SimpleNamespace(plan=AsyncMock(return_value=Plan(plan_version=1, steps=[])))
        configured = Settings(_env_file=None)
        node = nodes(
            configured,
            model=model,
            context_builder=ContextBuilder(configured),
        )
        state = {
            "intent": intent().model_dump(),
            "candidate_tools": [card.model_dump(mode="json")],
        }
        await node.plan(state, Principal(uuid4(), "reader", frozenset()))
        model.plan.assert_awaited_once()
        context = model.plan.await_args.args[1]
        assert context["tool_cards"][0]["input_schema"] == card.input_schema
        assert context["tool_cards"][0]["output_schema"] == card.output_schema

        model.plan.side_effect = LLMError("provider_error", "safe adapter error")
        from pytest import raises

        with raises(LLMError, match="safe adapter error"):
            await node.plan(state, Principal(uuid4(), "reader", frozenset()))

    run(check())


def test_v2_retrieves_each_subgoal_separately_and_plan_only_never_calls_model():
    async def check():
        card = tool_card()
        goals = [
            Subgoal(id="s1", action="find", entity="record"),
            Subgoal(id="s2", action="summarize", entity="record"),
        ]
        request_intent = Intent(
            goal="Find and summarize a record",
            subgoals=goals,
            category_ids=[],
            entities={},
            constraints={},
            missing_fields=[],
            confidence=1,
            execution_mode="plan_only",
        )
        retrieval = SimpleNamespace(tools=AsyncMock(side_effect=[[card], []]))
        model = SimpleNamespace(select_function=AsyncMock())
        node = nodes(Settings(_env_file=None), retrieval=retrieval, model=model)
        retrieved = await node.retrieve_v2(
            {"intent": request_intent.model_dump()}, Principal(uuid4(), "reader", frozenset())
        )
        assert retrieval.tools.await_count == 2
        assert retrieved["subgoal_tools"]["s1"][0]["tool_id"] == "lookup"
        assert retrieved["subgoal_tools"]["s2"] == []

        planned = await node.plan_v2(
            {"intent": request_intent.model_dump(), **retrieved},
            Principal(uuid4(), "reader", frozenset()),
        )
        assert planned["routing_decision"] == "plan_only_done"
        assert planned["final_result"]["steps"][0]["tools"][0]["tool_id"] == "lookup"
        assert planned["final_result"]["steps"][1]["tools"] == []
        model.select_function.assert_not_awaited()

    run(check())


def test_v2_accepts_only_one_function_call_from_current_shortlist():
    async def check():
        card = tool_card()
        request_intent = Intent(
            goal="Read a record",
            subgoals=[Subgoal(id="s1", action="read", entity="record")],
            category_ids=[],
            entities={},
            constraints={},
            missing_fields=[],
            confidence=1,
            execution_mode="execute",
        )
        turn = FunctionTurn(
            calls=[FunctionCall(name="lookup", arguments={"record_id": "R-1"}, call_id="c1")]
        )
        model = SimpleNamespace(select_function=AsyncMock(return_value=turn))
        node = nodes(Settings(_env_file=None), model=model)
        registry = SimpleNamespace(get=AsyncMock(return_value=card))
        node.validator = PlanValidator(registry)
        principal = Principal(uuid4(), "reader", frozenset({"records:read"}))
        state = {
            "intent": request_intent.model_dump(),
            "request_text": "read record R-1",
            "subgoal_tools": {"s1": [card.model_dump(mode="json")]},
        }
        proposed = await node.plan_v2(state, principal)
        assert proposed["routing_decision"] == "prepare_step"
        assert proposed["plan"]["steps"][0]["tool_id"] == "lookup"
        assert proposed["plan"]["steps"][0]["arguments"]["record_id"] == {"literal": "R-1"}

        model.select_function.return_value = FunctionTurn(
            calls=[FunctionCall(name="not_selected", arguments={"record_id": "R-1"}, call_id="c2")]
        )
        unknown = await node.plan_v2(state, principal)
        assert unknown["observation"]["error_code"] == "unselected_or_invalid_function"

        model.select_function.return_value = FunctionTurn(
            calls=[FunctionCall(name="lookup", arguments={"unexpected": "x"}, call_id="c3")]
        )
        invalid = await node.plan_v2(state, principal)
        assert invalid["observation"]["error_code"] == "invalid_function_arguments"

        model.select_function.return_value = FunctionTurn(
            calls=[
                FunctionCall(name="lookup", arguments={"record_id": "R-1"}, call_id="c1"),
                FunctionCall(name="lookup", arguments={"record_id": "R-2"}, call_id="c2"),
            ]
        )
        rejected = await node.plan_v2(state, principal)
        assert rejected["observation"]["error_code"] == "multiple_function_calls_rejected"
        assert registry.get.await_count == 2

    run(check())
