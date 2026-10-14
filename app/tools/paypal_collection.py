"""Convert Postman requests into reviewable, inert tool drafts.

Only the collection's top-level items are buffered at once. Postman scripts,
saved responses, headers, bodies and variables are never copied into cards.
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.tools.contracts import ToolCard
from app.tools.paypal import REVIEWED_INVOICE_ENDPOINTS

SANDBOX_HOST = "api-m.sandbox.paypal.com"
METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")
PATH_VAR = re.compile(r"^\{([^{}]+)\}$|^:([A-Za-z][\w-]*)$")
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._~-]+$")
PROVIDER_LABEL = re.compile(r"(?i)\bpaypal\b")


def _category(value: str) -> str:
    """Use the service/resource label, not the common provider, for discovery."""
    return re.sub(r"[^a-z0-9]+", "_", PROVIDER_LABEL.sub("", value).lower()).strip("_")[:60]


def _singular(value: str) -> str:
    if value.endswith("ies"):
        return value[:-3] + "y"
    if value.endswith("s") and not value.endswith(("ss", "us")):
        return value[:-1]
    return value


def _summary(method: str, endpoint: str) -> tuple[str, str]:
    """Describe the endpoint's action and resource without a provider prefix."""
    parts = endpoint.strip("/").split("/")
    parts = parts[1:] if re.fullmatch(r"v\d+", parts[0]) else parts
    literals = [part for part in parts if not part.startswith("{")]
    if parts[-1].startswith("{"):
        resource = _singular(literals[-1])
        action = {
            "GET": "Retrieve",
            "POST": "Create",
            "PUT": "Update",
            "PATCH": "Update",
            "DELETE": "Delete",
        }[method]
    elif len(parts) >= 2 and parts[-2].startswith("{"):
        resource = _singular(literals[-2])
        action = parts[-1].replace("-", "_").replace("_", " ").title()
    else:
        resource = parts[-1]
        action = {
            "GET": "List",
            "POST": "Create",
            "PUT": "Update",
            "PATCH": "Update",
            "DELETE": "Delete",
        }[method]
        if method != "GET":
            resource = _singular(resource)
    return f"{action} {resource.replace('_', ' ')}", action.lower()


def collection_envelope_metadata(path: Path) -> tuple[str | None, list[str]]:
    """Read only envelope auth/variable names, never their values or scripts."""
    auth_type = None
    names: list[str] = []
    section = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith('  "auth": {'):
                section = "auth"
            elif line.startswith('  "variable": ['):
                section = "variable"
            elif line.startswith(("  }", "  ]")):
                section = None
            elif section == "auth":
                match = re.match(r'\s+"type":\s*"([A-Za-z0-9_-]+)"', line)
                if match and auth_type is None:
                    auth_type = match[1]
            elif section == "variable":
                match = re.match(r'\s+"key":\s*"([A-Za-z0-9_-]+)"', line)
                if match:
                    names.append(match[1])
    return auth_type, sorted(set(names))


def top_level_items(path: Path) -> Iterator[dict[str, Any]]:
    """Decode one top-level item at a time without loading the collection at once."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        position = 0
        eof = False

        def fill() -> bool:
            nonlocal buffer, position, eof
            if eof:
                return False
            chunk = stream.read(65536)
            buffer = buffer[position:] + chunk
            position = 0
            eof = not chunk
            return bool(chunk)

        def skip_space() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or not fill():
                    return

        # The Postman envelope has info before item. Scan JSON string tokens at
        # envelope depth, so an incidental "item" inside info cannot redirect us.
        depth = 0
        while True:
            skip_space()
            if position >= len(buffer):
                raise ValueError("Collection has no item array")
            char = buffer[position]
            if char == '"':
                while True:
                    try:
                        key, end = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError:
                        if not fill():
                            raise ValueError("Invalid collection JSON") from None
                position = end
                if depth == 1 and key == "item":
                    skip_space()
                    if buffer[position] != ":":
                        raise ValueError("Invalid item member")
                    position += 1
                    skip_space()
                    if buffer[position] != "[":
                        raise ValueError("Collection item must be an array")
                    position += 1
                    break
            elif char in "{[":
                depth += 1
                position += 1
            elif char in "}]":
                depth -= 1
                position += 1
            else:
                position += 1
        while True:
            skip_space()
            if position >= len(buffer):
                raise ValueError("Truncated collection")
            if buffer[position] == "]":
                return
            if buffer[position] == ",":
                position += 1
                skip_space()
            while True:
                try:
                    item, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if not fill():
                        raise ValueError("Invalid or truncated collection item") from None
            position = end
            if isinstance(item, dict):
                yield item


def _requests(item: dict[str, Any], parents: tuple[str, ...], inherited_auth: str | None):
    name = str(item.get("name", ""))[:100]
    folder = parents + ((name,) if name else ())
    auth = item.get("auth", {}).get("type") if isinstance(item.get("auth"), dict) else None
    auth = auth or inherited_auth
    if isinstance(item.get("item"), list):
        for child in item["item"]:
            if isinstance(child, dict):
                yield from _requests(child, folder, auth)
    elif isinstance(item.get("request"), dict):
        yield folder, item["request"], auth, bool(item.get("event"))


def _referenced_variables(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(PLACEHOLDER.findall(value))
    if isinstance(value, list):
        return set().union(*(_referenced_variables(item) for item in value))
    if isinstance(value, dict):
        return set().union(*(_referenced_variables(item) for item in value.values()))
    return set()


def _endpoint(request: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    method = str(request.get("method", "")).upper()
    if method not in METHODS:
        raise ValueError("unsupported_method")
    url = request.get("url")
    raw = url.get("raw", "") if isinstance(url, dict) else url
    if not isinstance(raw, str):
        raise TypeError("unsupported_url")
    if raw.startswith(("{{base_url}}", "{{baseUrl}}")):
        path = raw.split("}}", 1)[1].split("?", 1)[0]
    else:
        parsed = urlsplit(raw)
        if parsed.scheme != "https" or parsed.hostname != SANDBOX_HOST:
            raise ValueError("untrusted_host")
        path = parsed.path
    if not path.startswith("/") or "//" in path or ".." in path:
        raise ValueError("unsafe_path")
    variables: list[str] = []
    segments = []
    for segment in path.strip("/").split("/"):
        match = PLACEHOLDER.fullmatch(segment) or PATH_VAR.fullmatch(segment)
        if match:
            variable = next(group for group in match.groups() if group)
            if not re.fullmatch(r"[A-Za-z][\w-]*", variable):
                raise ValueError("unsafe_path_variable")
            variables.append(variable)
            segments.append("{" + variable + "}")
        elif SAFE_SEGMENT.fullmatch(segment):
            segments.append(segment)
        else:
            raise ValueError("unsafe_path_segment")
    path = "/" + "/".join(segments)
    if path == "/v1/oauth2/token":
        raise ValueError("internal_auth")
    query = []
    if isinstance(url, dict):
        for entry in url.get("query", []) or []:
            key = entry.get("key") if isinstance(entry, dict) else None
            if isinstance(key, str) and re.fullmatch(r"[A-Za-z][\w.-]*", key):
                query.append(key)
    return method, path, variables, sorted(set(query))


def convert(path: Path) -> tuple[list[ToolCard], dict[str, Any]]:
    """Return sanitized drafts and a metadata-only review inventory."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    folders: Counter[str] = Counter()
    auth_types: Counter[str] = Counter()
    variable_names: set[str] = set()
    collection_auth, collection_variables = collection_envelope_metadata(path)
    for top in top_level_items(path):
        for names, request, auth, has_scripts in _requests(top, (), collection_auth):
            counts["requests"] += 1
            folders[names[0] if names else "(root)"] += 1
            auth_types[auth or "inherited_or_none"] += 1
            variable_names.update(_referenced_variables(request))
            if has_scripts:
                counts["scripted_requests"] += 1
            try:
                method, endpoint, variables, query = _endpoint(request)
            except (ValueError, TypeError) as error:
                counts[str(error)] += 1
                continue
            identity = (method, endpoint)
            body = request.get("body")
            body_mode = body.get("mode", "none") if isinstance(body, dict) else "none"
            if body_mode == "raw" and isinstance(body, dict):
                options = body.get("options")
                raw_options = options.get("raw") if isinstance(options, dict) else None
                language = raw_options.get("language") if isinstance(raw_options, dict) else None
                if language not in (None, "json"):
                    body_mode = "raw_" + str(language)[:30]
            if body_mode not in {"none", "raw"}:
                counts["non_json_body_requests"] += 1
            if identity in found:
                counts["duplicate_variants"] += 1
                found[identity]["variants"] += 1
                found[identity]["query"].update(query)
                found[identity]["scripted"] |= has_scripts
                found[identity]["body_modes"].add(body_mode)
                found[identity]["auth_types"].add(auth or "none")
                continue
            found[identity] = {
                "folder": names[0] if names else "",
                "variables": variables,
                "query": set(query),
                "variants": 1,
                "scripted": has_scripts,
                "body_modes": {body_mode},
                "auth_types": {auth or "none"},
            }
    cards = []
    inventory = []
    for (method, endpoint), meta in sorted(found.items()):
        digest = hashlib.sha256(f"{method} {endpoint}".encode()).hexdigest()[:12]
        entity = endpoint.strip("/").split("/")[1:3]
        label = " ".join(entity) or "resource"
        properties: dict[str, Any] = {
            key: {"type": "string", "minLength": 1, "maxLength": 255} for key in meta["variables"]
        }
        if meta["query"]:
            properties["query"] = {
                "type": "object",
                "properties": {key: {"type": "string"} for key in sorted(meta["query"])},
                "additionalProperties": False,
            }
        if method != "GET":
            properties["body"] = {"type": "object"}
        operation_type = "read" if method == "GET" else "write"
        summary, action = _summary(method, endpoint)
        card = ToolCard(
            tool_id=f"paypal_{method.lower()}_{digest}",
            schema_version="1",
            adapter_key=f"paypal_{method.lower()}_{digest}",
            name=summary[:200],
            description=f"{summary} in sandbox via {method} {endpoint} for {label}.",
            category_ids=list(
                dict.fromkeys(
                    category
                    for category in (
                        _category(entity[0] if entity else ""),
                        _category(str(meta["folder"])),
                    )
                    if category
                )
            )
            or ["other"],
            entities=entity,
            action=action,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": meta["variables"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "http_status": {"type": "integer", "minimum": 200, "maximum": 299},
                    "response": {"type": "object"},
                },
                "required": ["http_status", "response"],
                "additionalProperties": False,
            },
            operation_type=operation_type,
            required_scopes=["paypal:read" if operation_type == "read" else "paypal:write"],
            precondition_key="paypal_reviewed_write"
            if operation_type == "write" and (method, endpoint) in REVIEWED_INVOICE_ENDPOINTS
            else None,
            verification_key="paypal_response",
            provider="paypal",
            provider_account_key="local_sandbox",
        )
        cards.append(card)
        inventory.append(
            {
                "tool_id": card.tool_id,
                "method": method,
                "path": endpoint,
                "variants": meta["variants"],
                "scripts_present": meta["scripted"],
                "body_modes": sorted(meta["body_modes"]),
                "auth_variants": sorted(meta["auth_types"]),
                "needs_manual_mapping": meta["scripted"]
                or meta["body_modes"] - {"none", "raw"} != set()
                or len(meta["body_modes"]) > 1,
            }
        )
    return cards, {
        "counts": dict(counts),
        "folders": dict(folders),
        "auth_types": dict(auth_types),
        "collection_auth_type": collection_auth,
        "collection_variable_names": collection_variables,
        "referenced_variable_names": sorted(variable_names),
        "endpoints": inventory,
    }
