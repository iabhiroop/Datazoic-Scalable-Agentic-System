"""Binding and hashing invariants used before tool dispatch."""

import math
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agent.nodes import AgentNodes
from app.agent.planning import (
    PlanValidator,
    canonical_payload_hash,
    resolve_arguments,
    resolve_binding,
)
from app.agent.state import LiteralBinding, OutputBinding, Plan, PlanStep
from app.security import Principal
from app.tools.contracts import ToolCard


def test_pointer_escapes_arrays_and_copy():
    results = {"s1": {"a/b": {"~key": [{"value": "14"}]}}}
    value = resolve_binding(OutputBinding(from_step="s1", path="/a~1b/~0key/0"), results)
    assert value == {"value": "14"}
    value["value"] = "changed"
    assert results["s1"]["a/b"]["~key"][0]["value"] == "14"
    assert resolve_binding(OutputBinding(from_step="s1", path=""), results) == results["s1"]


def test_plan_failure_feedback_is_stable_and_contains_no_input_values():
    assert (
        AgentNodes.plan_validation_reason(
            ValueError("Resolved tool input does not match its schema")
        )
        == "tool_input_schema_mismatch"
    )
    assert (
        AgentNodes.plan_validation_reason(ValueError("Output path must be a JSON Pointer"))
        == "invalid_json_pointer"
    )
    assert AgentNodes.plan_validation_reason(ValueError("recipient@example.com")) == (
        "plan_contract_validation_failed"
    )


@pytest.mark.parametrize(
    "path", ["a", "/missing", "/list/-", "/list/01", "/list/2", "/bad~2escape"]
)
def test_bad_pointer_rejected(path):
    with pytest.raises(ValueError):
        resolve_binding(OutputBinding(from_step="s1", path=path), {"s1": {"list": [0]}})


def test_missing_output_and_literal_copy():
    with pytest.raises(ValueError):
        resolve_binding(OutputBinding(from_step="missing", path=""), {})
    binding = LiteralBinding(literal={"values": [1]})
    resolve_binding(binding, {})["values"].append(2)
    assert binding.literal == {"values": [1]}


def test_hash_order_independent_preserves_decimal_strings():
    assert canonical_payload_hash({"currency": "INR", "amount": "20.00"}) == canonical_payload_hash(
        {"amount": "20.00", "currency": "INR"}
    )
    assert canonical_payload_hash({"amount": "20.00"}) != canonical_payload_hash({"amount": "20.0"})


@pytest.mark.parametrize(
    "payload",
    [
        {"value": math.nan},
        {"value": math.inf},
        {1: "bad"},
        {"nested": {1: "bad"}},
        {"tuple": (1, 2)},
    ],
)
def test_non_json_payload_rejected(payload):
    with pytest.raises(ValueError):
        canonical_payload_hash(payload)


def read_card():
    return ToolCard(
        tool_id="lookup",
        schema_version="1",
        adapter_key="lookup",
        name="Lookup",
        description="lookup",
        action="lookup",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string", "minLength": 1}},
            "required": ["id"],
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        operation_type="read",
        required_scopes=[],
        precondition_key=None,
        verification_key="lookup",
        provider="internal",
        provider_account_key="internal",
    )


def step(identity="s1", **updates):
    return PlanStep(
        step_id=identity,
        tool_id="lookup",
        schema_version="1",
        arguments={"id": LiteralBinding(literal="valid")},
    ).model_copy(update=updates)


async def test_plan_validation_and_runtime_output_schema():
    registry = AsyncMock()
    registry.get.return_value = read_card()
    validator = PlanValidator(registry)
    principal = Principal(uuid4(), "reader", frozenset())
    second = step(
        "s2", depends_on=["s1"], arguments={"id": OutputBinding(from_step="s1", path="/id")}
    )
    plan = Plan(plan_version=1, steps=[step(), second])
    assert await validator.validate(plan, principal) == plan
    assert resolve_arguments(second, read_card(), {"s1": {"id": "valid"}}) == {"id": "valid"}
    with pytest.raises(ValueError):
        resolve_arguments(second, read_card(), {"s1": {"id": 123}})


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [step(), step()],
        [step(depends_on=["s2"])],
        [step(action_ref="model-write")],
        [step(arguments={})],
        [step(arguments={"id": LiteralBinding(literal=3)})],
        [step(arguments={"id": OutputBinding(from_step="s2", path="/id")})],
        [
            step(),
            step(
                "s2",
                depends_on=["s1"],
                arguments={"id": OutputBinding(from_step="s1", path="/bad~2")},
            ),
        ],
    ],
)
async def test_invalid_plans_rejected(steps):
    registry = AsyncMock()
    registry.get.return_value = read_card()
    with pytest.raises(ValueError):
        await PlanValidator(registry).validate(
            Plan(plan_version=1, steps=steps), Principal(uuid4(), "reader", frozenset())
        )


async def test_plan_limits_and_writes_blocked():
    registry = AsyncMock()
    registry.get.return_value = read_card()
    principal = Principal(uuid4(), "reader", frozenset())
    with pytest.raises(ValueError):
        await PlanValidator(registry, max_steps=1).validate(
            Plan(plan_version=1, steps=[step(), step("s2")]), principal
        )
    registry.get.return_value = read_card().model_copy(update={"operation_type": "write"})
    with pytest.raises(ValueError, match="Write"):
        await PlanValidator(registry).validate(Plan(plan_version=1, steps=[step()]), principal)
