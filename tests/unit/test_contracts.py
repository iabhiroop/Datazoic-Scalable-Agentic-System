"""Boundary checks for untrusted input contracts; not implementation acceptance tests."""

import pytest
from pydantic import ValidationError

from app.agent.state import Plan
from app.api.schemas import RunInput
from app.tools.rag_pipeline import RagArguments
from app.tools.system_search import SEARCH_ARGUMENTS


def test_model_cannot_mix_literal_and_output_binding():
    with pytest.raises(ValidationError):
        Plan.model_validate(
            {
                "plan_version": 1,
                "steps": [
                    {
                        "step_id": "s2",
                        "tool_id": "refund",
                        "schema_version": "1",
                        "arguments": {
                            "amount": {"literal": "10.00", "from_step": "s1", "path": "/amount"}
                        },
                    }
                ],
            }
        )


def test_literal_null_and_decimal_string_remain_valid():
    plan = Plan.model_validate(
        {
            "plan_version": 1,
            "steps": [
                {
                    "step_id": "s1",
                    "tool_id": "example",
                    "schema_version": "1",
                    "arguments": {"optional": {"literal": None}, "amount": {"literal": "2000.00"}},
                }
            ],
        }
    )
    assert plan.steps[0].arguments["amount"].literal == "2000.00"
    assert plan.steps[0].arguments["optional"].literal is None


@pytest.mark.parametrize("extra", [{"tenant_id": "other"}, {"user_id": "other"}, {"endpoint": "x"}])
def test_rag_rejects_authority_arguments(extra):
    with pytest.raises(ValidationError):
        RagArguments.model_validate({"query": "policy", **extra})


@pytest.mark.parametrize("top_k", [0, 7])
def test_rag_evidence_limit_is_bounded(top_k):
    with pytest.raises(ValidationError):
        RagArguments(query="policy", top_k=top_k)


def test_status_mode_cannot_accept_capability_filters_or_sql():
    with pytest.raises(ValidationError):
        SEARCH_ARGUMENTS.validate_python({"mode": "request_status", "query": "select *"})


def test_explicit_run_and_last_selector_are_mutually_exclusive():
    with pytest.raises(ValidationError):
        SEARCH_ARGUMENTS.validate_python(
            {
                "mode": "execution_history",
                "selector": "last",
                "run_id": "00000000-0000-0000-0000-000000000001",
            }
        )


def test_public_input_cannot_supply_worker_timer():
    with pytest.raises(ValidationError):
        RunInput.model_validate(
            {
                "wait_token": "00000000-0000-0000-0000-000000000001",
                "kind": "timer",
                "payload": {},
            }
        )
