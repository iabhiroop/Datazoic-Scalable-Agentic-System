"""Small, focused checks for appv2's model-facing tool boundary."""

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from app.tools.contracts import ToolCard
from appv2.tools import (
    compact_schema,
    function_name,
    normalize_argument_values,
    normalize_invoice_status_argument,
    stop_after_result,
    validate_arguments,
    write_fingerprint,
)


def card():
    return ToolCard(
        tool_id="paypal_get_list",
        schema_version="4",
        adapter_key="paypal_get_list",
        name="List invoices",
        description="List invoices in the sandbox.",
        action="list",
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "object",
                    "properties": {"page": {"type": "integer", "minimum": 1}},
                }
            },
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        operation_type="read",
        required_scopes=["paypal:read"],
        precondition_key=None,
        verification_key="paypal_response",
        provider="paypal",
        provider_account_key="local_sandbox",
    )


def test_compact_schema_removes_annotations_and_keeps_constraints():
    result = compact_schema(
        {
            "title": "Request",
            "type": "object",
            "required": ["count"],
            "properties": {"count": {"title": "Count", "type": "integer", "minimum": 1}},
        }
    )
    assert "title" not in result
    assert result["required"] == ["count"]
    assert result["properties"]["count"] == {"type": "integer", "minimum": 1}


def test_function_names_are_readable_and_unique_within_shortlist():
    first, second = card(), card().model_copy(update={"name": "List invoices"})
    assert function_name(1, first) == "t1_list_invoices"
    assert function_name(2, second) == "t2_list_invoices"


def test_model_schema_exposes_actual_invoice_body_fields_and_preserves_types():
    original = {
        "type": "object",
        "properties": {"body": {"$ref": "#/$defs/search_data"}},
        "additionalProperties": False,
        "$defs": {
            "search_data": {
                "type": "object",
                "properties": {"status": {"$ref": "#/$defs/status_list"}},
            },
            "status_list": {
                "type": "array",
                "maxItems": 5,
                "items": {"enum": ["SENT", "UNPAID"]},
            },
        },
    }
    compact = compact_schema(original)
    assert "$defs" not in compact
    assert compact["properties"]["body"]["properties"]["status"]["type"] == "array"
    for schema in (original, compact):
        validator = Draft202012Validator(schema)
        validator.validate({"body": {"status": ["SENT"]}})
        with pytest.raises(ValidationError):
            validator.validate({"search_data": {"status": ["SENT"]}})
        with pytest.raises(ValidationError):
            validator.validate({"body": {"status": "SENT"}})
        with pytest.raises(ValidationError):
            validator.validate({"body": {"status": ["OUTSTANDING"]}})


def test_invalid_arguments_are_rejected_by_original_card_schema():
    with pytest.raises(ValidationError):
        validate_arguments(card(), {"query": {"page": 0}})


def test_invalid_invoice_id_array_is_rejected_by_tool_schema():
    invoice = card().model_copy(
        update={
            "input_schema": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
                "additionalProperties": False,
            }
        }
    )
    with pytest.raises(ValidationError, match="is not of type 'string'"):
        validate_arguments(invoice, {"invoice_id": []})


def test_array_and_enum_normalization_preserves_fields_and_rejects_unknown_values():
    tool = card().model_copy(
        update={
            "input_schema": {
                "type": "object",
                "properties": {
                    "body": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "array", "items": {"enum": ["SENT", "UNPAID"]}}
                        },
                    }
                },
            }
        }
    )
    original = {"body": {"status": "sent"}}
    normalized = normalize_argument_values(tool, original)
    assert normalized == {"body": {"status": ["SENT"]}}
    assert original == {"body": {"status": "sent"}}
    validate_arguments(tool, normalized)
    assert normalize_argument_values(tool, {}) == {}
    invalid = normalize_argument_values(tool, {"body": {"status": "outstanding"}})
    with pytest.raises(ValidationError):
        validate_arguments(tool, invalid)


def test_misplaced_invoice_status_is_moved_only_to_schema_defined_body_path():
    invoice_search = card().model_copy(
        update={
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "properties": {"page": {"type": "integer"}},
                        "additionalProperties": False,
                    },
                    "body": {"$ref": "#/$defs/search_body"},
                },
                "$defs": {
                    "search_body": {
                        "type": "object",
                        "properties": {"search_data": {"$ref": "#/$defs/search_data"}},
                    },
                    "search_data": {
                        "type": "object",
                        "properties": {"invoice_status": {"type": "string"}},
                    },
                },
            }
        }
    )
    supplied = {"query": {"invoice_status": "OUTSTANDING"}}

    result = normalize_invoice_status_argument(invoice_search, supplied)

    assert result == {"query": {}, "body": {"search_data": {"invoice_status": "OUTSTANDING"}}}
    assert supplied == {"query": {"invoice_status": "OUTSTANDING"}}


def test_uncertain_writes_stop_and_duplicate_proposals_have_stable_identity():
    write_card = card().model_copy(update={"operation_type": "write"})
    args = {"query": {"page": 1}}
    assert stop_after_result(write_card, "unknown")
    assert stop_after_result(write_card, "failed")
    assert not stop_after_result(write_card, "succeeded")
    assert write_fingerprint(write_card.tool_id, args) == write_fingerprint(
        write_card.tool_id, {"query": {"page": 1}}
    )
