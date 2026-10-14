"""Deterministic validation precedes all dispatch. Money remains decimal strings."""

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from app.agent.state import Binding, LiteralBinding, Plan, PlanStep
from app.security import Principal
from app.tools.contracts import ToolCard
from app.tools.registry import ToolRegistry


class PlanValidator:
    def __init__(self, registry: ToolRegistry, *, max_steps: int = 10) -> None:
        if max_steps < 1:
            raise ValueError("Plan step limit must be positive")
        self.registry = registry
        self.max_steps = max_steps

    async def validate(self, plan: Plan, principal: Principal) -> Plan:
        """Validate sequential plans; Python assigns write identities at dispatch."""
        if not plan.steps or len(plan.steps) > self.max_steps:
            raise ValueError("Plan length exceeds configured limits")
        seen: set[str] = set()
        for step in plan.steps:
            if not step.step_id.strip() or step.step_id in seen:
                raise ValueError("Step identities must be nonempty and unique")
            if len(set(step.depends_on)) != len(step.depends_on) or not set(
                step.depends_on
            ).issubset(seen):
                raise ValueError("Dependencies must name distinct earlier steps")
            if step.action_ref is not None:
                raise ValueError("Model plans cannot assign business action identities")
            card = await self.registry.get(step.tool_id, step.schema_version, principal)
            if card.operation_type == "write" and card.precondition_key not in {
                "paypal_reviewed_write",
                "sandbox_approved_write",
            }:
                raise ValueError("Write tool has no executable precondition policy")
            dynamic = False
            for binding in step.arguments.values():
                if not isinstance(binding, LiteralBinding):
                    dynamic = True
                    if binding.from_step not in step.depends_on:
                        raise ValueError("Output binding must name an earlier declared dependency")
                    pointer_segments(binding.path)
            if not dynamic:
                resolve_arguments(step, card, {})
            else:
                # Values depending on tool outputs require full validation at dispatch.
                schema = card.input_schema
                if not set(schema.get("required", [])).issubset(step.arguments):
                    raise ValueError("Required tool arguments are missing")
                if schema.get("additionalProperties") is False and not set(step.arguments).issubset(
                    schema.get("properties", {})
                ):
                    raise ValueError("Unknown tool arguments")
            seen.add(step.step_id)
        return plan.model_copy(deep=True)

    async def assign_actions(self, plan: Plan, existing: dict[str, str]) -> dict[str, str]:
        raise NotImplementedError(
            "Persist stable write slots and retain completed effects on replan"
        )


def pointer_segments(path: str) -> list[str]:
    if path == "":
        return []
    if not path.startswith("/"):
        raise ValueError("Output path must be a JSON Pointer")
    segments = path[1:].split("/")
    if any(re.search(r"~(?![01])", segment) for segment in segments):
        raise ValueError("Invalid JSON Pointer escape")
    return [segment.replace("~1", "/").replace("~0", "~") for segment in segments]


def resolve_arguments(
    step: PlanStep, card: ToolCard, results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Always validate the complete resolved input immediately before dispatch."""
    if (step.tool_id, step.schema_version) != (card.tool_id, card.schema_version):
        raise ValueError("Tool contract does not match the planned version")
    arguments = {
        name: resolve_binding(binding, results) for name, binding in step.arguments.items()
    }
    try:
        Draft202012Validator(card.input_schema, format_checker=FormatChecker()).validate(arguments)
    except ValidationError:
        raise ValueError("Resolved tool input does not match its schema") from None
    return arguments


def validate_path_argument_provenance(
    card: ToolCard,
    arguments: dict[str, Any],
    request_text: str,
    results: dict[str, dict[str, Any]],
    clarifications: list[dict[str, Any]] | None = None,
) -> None:
    """Do not use a model-invented resource ID in an endpoint path."""
    if card.endpoint is None:
        return

    def result_contains(value: Any, expected: str) -> bool:
        if isinstance(value, str):
            return value == expected
        if isinstance(value, dict):
            return any(result_contains(item, expected) for item in value.values())
        if isinstance(value, list):
            return any(result_contains(item, expected) for item in value)
        return False

    def text_contains(text: str, expected: str) -> bool:
        return bool(re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(expected)}(?![A-Za-z0-9_])",
            text,
            flags=re.IGNORECASE,
        ))

    for name in re.findall(r"\{([^{}]+)\}", card.endpoint.path):
        value = arguments.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Path argument {name} must be a concrete string")
        normalized = value.strip().lower()
        if normalized in {
            "unknown", "none", "null", "n/a", "placeholder", "all", "any", "latest",
        } or (
            normalized.startswith("<") and normalized.endswith(">")
        ):
            raise ValueError(f"Path argument {name} is a placeholder")
        if text_contains(request_text, value):
            continue
        if result_contains(results, value) or result_contains(clarifications or [], value):
            continue
        raise ValueError(f"Path argument {name} was not supplied by the user or a prior result")


def resolve_binding(binding: Binding, results: dict[str, dict[str, Any]]) -> Any:
    """Resolve RFC 6901 pointers against already validated outputs; never evaluate code."""
    if isinstance(binding, LiteralBinding):
        return deepcopy(binding.literal)
    if binding.from_step not in results:
        raise ValueError("Referenced output is unavailable")
    value: Any = results[binding.from_step]
    for key in pointer_segments(binding.path):
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
            index = int(key)
            if index >= len(value):
                raise ValueError("Output path does not exist")
            value = value[index]
        else:
            raise ValueError("Output path does not exist")
    return deepcopy(value)


def canonical_payload_hash(arguments: dict[str, Any]) -> str:
    """Hash strict JSON after adapter normalization; preserve decimal strings exactly."""

    def check(value: Any) -> None:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("Payload object keys must be strings")
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("Payload must contain only JSON values")

    check(arguments)
    try:
        canonical = json.dumps(
            arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError):
        raise ValueError("Payload must contain only finite JSON values") from None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
