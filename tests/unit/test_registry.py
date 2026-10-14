"""Imported schema and execution metadata must stay within approved policies."""

import json
from pathlib import Path

import pytest

from app.tools.contracts import ToolCard
from app.tools.registry import ToolRegistry


def card():
    path = Path(__file__).resolve().parents[2] / "resources/tool_cards/system_search.json"
    return ToolCard.model_validate(json.loads(path.read_text()))


def registry():
    return ToolRegistry(
        {"system_search": object()}, database=None, verification_keys={"system_search_result"}
    )


async def test_builtin_schemas_with_local_refs_accepted():
    original = card()
    result = await registry().validate_import(original)
    assert result == original
    assert result is not original


@pytest.mark.parametrize(
    "update",
    [
        {"adapter_key": "unconfigured"},
        {"verification_key": "unapproved"},
        {"operation_type": "write"},
        {"precondition_key": "unapproved"},
        {"reconciliation_key": "unapproved"},
        {"tool_id": " "},
    ],
)
async def test_unapproved_execution_metadata_rejected(update):
    with pytest.raises(ValueError):
        await registry().validate_import(card().model_copy(update=update))


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "https://example.invalid/schema"},
        {"$ref": "file:///private"},
        {"$ref": "#/$defs/missing"},
        {"type": "invalid"},
        {"$dynamicRef": "#node"},
        {"$id": "https://example.invalid/", "$ref": "#/properties/x"},
    ],
)
def test_unsafe_or_invalid_schema_rejected(schema):
    with pytest.raises(ValueError):
        ToolRegistry.validate_schema(schema)
