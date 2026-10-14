"""Small dispatcher for retrieved cards, with validation kept in Python."""

import hashlib
import inspect
import json
import re
from copy import deepcopy
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator, FormatChecker

from app.agent.planning import validate_path_argument_provenance
from app.db.models import RegistryTool
from app.db.repositories import CatalogRepository
from app.security import Principal, TokenAuthenticator
from app.tools.contracts import ToolCard, ToolContext, ToolOutcome


def compact_schema(value: Any) -> Any:
    """Inline local references for model readability without changing constraints."""
    annotations = {"title", "examples", "example", "deprecated", "readOnly", "writeOnly"}
    unresolved = False

    def expand(node: Any, active: frozenset[str] = frozenset()) -> Any:
        nonlocal unresolved
        if isinstance(node, list):
            return [expand(item, active) for item in node]
        if not isinstance(node, dict):
            return node
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/"):
            if reference not in active:
                target = value
                for part in reference[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
                resolved = expand(target, active | {reference})
                siblings = {key: item for key, item in node.items() if key != "$ref"}
                remainder = expand(siblings, active)
                return {"allOf": [resolved, remainder]} if remainder else resolved
            unresolved = True
        return {
            key: (
                {name: expand(item, active) for name, item in child.items()}
                if key in {"properties", "patternProperties", "dependentSchemas"}
                else expand(child, active)
            )
            for key, child in node.items()
            if key not in annotations | {"$defs", "definitions"}
        }

    result = expand(value)
    if unresolved:
        # Recursive schemas retain their definitions so surviving refs still resolve.
        for key in ("$defs", "definitions"):
            if key in value:
                result[key] = deepcopy(value[key])
    return result


def function_name(index: int, card: ToolCard) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", card.name.lower()).strip("_") or "tool"
    return f"t{index}_{slug}"[:64]


def validate_arguments(card: ToolCard, arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise TypeError("Tool arguments must be a JSON object")
    Draft202012Validator(card.input_schema, format_checker=FormatChecker()).validate(arguments)


def normalize_argument_values(card: ToolCard, arguments: dict[str, Any]) -> dict[str, Any]:
    """Adapt scalar array values and unambiguous enum casing; never add fields."""

    def adapt(value: Any, schema: dict[str, Any]) -> Any:
        for constraint in schema.get("allOf", []):
            value = adapt(value, constraint)
        if schema.get("type") == "array":
            if isinstance(value, (str, int, float, bool)):
                value = [value]
            if isinstance(value, list) and isinstance(schema.get("items"), dict):
                return [adapt(item, schema["items"]) for item in value]
        if isinstance(value, dict):
            properties = schema.get("properties", {})
            return {key: adapt(item, properties.get(key, {})) for key, item in value.items()}
        if isinstance(value, str) and "enum" in schema:
            matches = [
                item
                for item in schema["enum"]
                if isinstance(item, str) and item.casefold() == value.casefold()
            ]
            if len(matches) == 1:
                return matches[0]
        return value

    return adapt(arguments, compact_schema(card.input_schema))


def normalize_invoice_status_argument(card: ToolCard, arguments: dict[str, Any]) -> dict[str, Any]:
    """Move a misplaced invoice filter only when the published schema defines its target."""
    query = arguments.get("query")
    if not isinstance(query, dict) or "invoice_status" not in query:
        return arguments

    root = card.input_schema

    def resolve(schema: Any) -> Any:
        if not isinstance(schema, dict):
            return schema
        reference = schema.get("$ref")
        if not isinstance(reference, str) or not reference.startswith("#/"):
            return schema
        current = root
        for part in reference[2:].split("/"):
            current = current[part.replace("~1", "/").replace("~0", "~")]
        return current

    properties = resolve(root).get("properties", {})
    query_schema = resolve(properties.get("query", {}))
    body_schema = resolve(properties.get("body", {}))
    body_properties = body_schema.get("properties", {})
    search_schema = resolve(body_properties.get("search_data", {}))
    query_properties = query_schema.get("properties", {})
    search_properties = search_schema.get("properties", {})
    if "invoice_status" in query_properties or "invoice_status" not in search_properties:
        return arguments

    normalized = deepcopy(arguments)
    status = normalized["query"].pop("invoice_status")
    body = normalized.setdefault("body", {})
    search_data = body.setdefault("search_data", {})
    if "invoice_status" in search_data:
        return arguments
    search_data["invoice_status"] = status
    return normalized


def write_fingerprint(tool_id: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps([tool_id, arguments], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def stop_after_result(card: ToolCard, state: str) -> bool:
    return state in {"unknown", "pending"} or (
        card.operation_type == "write" and state != "succeeded"
    )


class ToolDispatcher:
    def __init__(
        self, settings, catalog, principal, adapters, log_event, *, run_id, conversation_id
    ):
        self.settings = settings
        self.catalog = catalog
        self.principal: Principal = principal
        self.adapters = adapters
        self.log_event = log_event
        self.authenticator = TokenAuthenticator(settings)
        self.run_id = run_id
        self.conversation_id = conversation_id
        self.write_fingerprints: set[str] = set()
        self.tool_call_count = 0

    async def _log(self, event: dict[str, Any]) -> None:
        result = self.log_event(event)
        if inspect.isawaitable(result):
            await result

    def declarations(self, cards: list[ToolCard]) -> tuple[list[dict], dict[str, ToolCard]]:
        declarations, by_name = [], {}
        for index, card in enumerate(cards, start=1):
            name = function_name(index, card)
            by_name[name] = card
            declarations.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"{card.name}: {card.description}",
                        "parameters": compact_schema(card.input_schema),
                    },
                }
            )
        return declarations, by_name

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        cards: list[ToolCard],
        *,
        request_text: str,
        verified_results: dict[str, Any],
    ) -> ToolOutcome:
        declarations, by_name = self.declarations(cards)
        del declarations
        card = by_name.get(name)
        if card is None:
            raise ValueError("Model selected a tool outside the current shortlist")
        # Count proposals, including invalid ones, so repeated malformed calls cannot
        # bypass the per-turn tool-call limit.
        self.tool_call_count += 1
        arguments = normalize_invoice_status_argument(card, arguments)
        arguments = normalize_argument_values(card, arguments)
        validate_arguments(card, arguments)
        self.principal = await self.authenticator.refresh(self.principal)
        async with self.catalog.sessions() as session:
            rows = await session.scalars(
                CatalogRepository(session, allowed_adapters=set(self.adapters) | {"rag_pipeline"})
                .tool_query(self.principal)
                .where(
                    RegistryTool.tool_id == card.tool_id,
                    RegistryTool.schema_version == card.schema_version,
                )
                .limit(1)
            )
            current = rows.first()
        if current is None or ToolCard.model_validate(current.card) != card:
            raise PermissionError("Tool is no longer published or authorized")
        validate_path_argument_provenance(card, arguments, request_text, verified_results)
        operation_id = uuid4()
        adapter = self.adapters.get(card.adapter_key)
        if card.adapter_key == "rag_pipeline":
            if "knowledge:read" not in self.principal.scopes:
                raise PermissionError("Knowledge read permission is missing")
            await self._log(
                {
                    "event": "tool_dispatch_started",
                    "tool_id": card.tool_id,
                    "operation_type": "read",
                    "operation_id": str(operation_id),
                    "arguments": arguments,
                }
            )
            passages = await self.catalog.documents(
                str(arguments["query"]),
                self.principal,
                corpus_ids=arguments.get("corpus_ids", []),
                limit=arguments.get("top_k", 4),
            )
            outcome = ToolOutcome(
                state="succeeded",
                data={
                    "query": arguments["query"],
                    "sufficient_evidence": bool(passages),
                    "answer": None,
                    "source_ids": [item["chunk_id"] for item in passages],
                    "passages": passages,
                    "knowledge_revision": await self.catalog.knowledge_revision(self.principal),
                },
            )
            await self._log(
                {
                    "event": "tool_dispatch_finished",
                    "tool_id": card.tool_id,
                    "operation_id": str(operation_id),
                    "state": outcome.state,
                    "result": outcome.data,
                }
            )
            return outcome
        if adapter is None:
            raise ValueError("Tool adapter is not available")

        fingerprint = write_fingerprint(card.tool_id, arguments)
        if card.operation_type == "write":
            if not self.settings.paypal_sandbox_writes_enabled:
                raise PermissionError("Sandbox writes are disabled")
            if fingerprint in self.write_fingerprints:
                raise ValueError("Duplicate write proposal was blocked")
            self.write_fingerprints.add(fingerprint)
        context = ToolContext(
            tenant_id=self.principal.tenant_id,
            user_id=self.principal.user_id,
            run_id=self.run_id,
            operation_id=operation_id,
            idempotency_key=str(operation_id) if card.operation_type == "write" else None,
            conversation_id=self.conversation_id,
        )
        await self._log(
            {
                "event": "tool_dispatch_started",
                "tool_id": card.tool_id,
                "schema_version": card.schema_version,
                "operation_type": card.operation_type,
                "operation_id": str(operation_id),
                "arguments": arguments,
            }
        )
        outcome = await adapter.invoke(arguments, context)
        if outcome.state == "succeeded":
            await adapter.authorize_result(arguments, outcome.data, context)
        await self._log(
            {
                "event": "tool_dispatch_finished",
                "tool_id": card.tool_id,
                "operation_id": str(operation_id),
                "state": outcome.state,
                "error_code": outcome.error_code,
                "result": outcome.data,
            }
        )
        return outcome
