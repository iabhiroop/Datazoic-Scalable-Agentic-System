"""Official contract mapping excludes examples and keeps schemas locally resolvable."""

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from app.tools.contracts import ToolCard
from app.tools.openapi import map_contract


def generated():
    return ToolCard(
        tool_id="paypal_patch_test",
        schema_version="1",
        adapter_key="paypal_patch_test",
        name="Update order",
        description="Update order",
        action="update",
        category_ids=["orders"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        operation_type="write",
        required_scopes=["paypal:write"],
        precondition_key=None,
        verification_key="paypal_response",
        provider="paypal",
        provider_account_key="local_sandbox",
    )


def match(media="application/json"):
    spec = {
        "components": {
            "schemas": {
                "patch": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["description"],
                        "properties": {
                            "description": {"type": "string", "example": "SAMPLE_CUSTOMER"},
                            "child": {"$ref": "#/components/schemas/child"},
                        },
                    },
                }
            },
            "child": {},
        }
    }
    spec["components"]["schemas"]["child"] = {
        "type": "string",
        "nullable": True,
        "default": "SECRET",
    }
    operation = {
        "summary": "Update PayPal order",
        "parameters": [
            {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
            {"name": "page", "in": "query", "schema": {"type": "integer", "minimum": 1}},
            {"name": "Prefer", "in": "header", "schema": {"type": "string"}},
        ],
        "requestBody": {
            "required": True,
            "content": {media: {"schema": {"$ref": "#/components/schemas/patch"}}},
        },
    }
    return (spec, "/v2/checkout/orders/{id}", operation, [], "https://example.invalid/pinned.json")


def test_schema_mapping_renames_path_arguments_and_preserves_patch_arrays():
    entry = {"method": "PATCH", "path": "/v2/checkout/orders/{order_id}", "body_modes": ["raw"]}
    mapped = map_contract(generated(), entry, match())
    assert mapped.name == "Update order"
    assert mapped.precondition_key == "sandbox_approved_write"
    assert mapped.endpoint.prefer_representation
    schema = mapped.input_schema
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(
        {"order_id": "O1", "query": {"page": 2}, "body": [{"description": "Updated"}]}
    )
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate({"order_id": "O1", "body": {}})
    assert "SAMPLE_CUSTOMER" not in str(schema) and "SECRET" not in str(schema)
    assert "description" in schema["$defs"]["patch"]["items"]["properties"]


def test_multipart_operations_remain_unsupported():
    with pytest.raises(ValueError, match="unsupported_request_media"):
        map_contract(
            generated(),
            {"method": "PATCH", "path": "/v2/checkout/orders/{order_id}", "body_modes": ["raw"]},
            match("multipart/form-data"),
        )
