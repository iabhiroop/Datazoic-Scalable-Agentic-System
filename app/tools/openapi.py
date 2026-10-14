"""Map collection endpoints to pinned official schemas without copying examples."""

import json
import re
from pathlib import Path
from typing import Any

import httpx

from app.tools.contracts import EndpointContract, ToolCard
from app.tools.paypal import adapter_key

PROVIDER = re.compile(r"(?i)\bpaypal\b")
METHODS = {"get", "post", "put", "patch", "delete"}


async def download_specifications(directory: Path) -> None:
    """Fetch data only from the provider's official repository, pinned to one commit."""
    directory.mkdir(parents=True, exist_ok=True)
    repository = "https://api.github.com/repos/paypal/paypal-rest-api-specifications"
    async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
        response = await client.get(repository + "/commits/main")
        response.raise_for_status()
        commit = response.json()["sha"]
        response = await client.get(repository + f"/git/trees/{commit}?recursive=1")
        response.raise_for_status()
        paths = [
            entry["path"]
            for entry in response.json()["tree"]
            if entry["path"].startswith("openapi/") and entry["path"].endswith(".json")
        ]
        for path in paths:
            response = await client.get(
                f"https://raw.githubusercontent.com/paypal/paypal-rest-api-specifications/{commit}/{path}"
            )
            response.raise_for_status()
            (directory / Path(path).name).write_text(json.dumps(response.json()), encoding="utf-8")
        (directory / "source.json").write_text(
            json.dumps(
                {
                    "repository": "https://github.com/paypal/paypal-rest-api-specifications",
                    "commit": commit,
                    "files": paths,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"Official API contracts pinned to {commit} ({len(paths)} files)", flush=True)


def normalize(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


def description(value: str) -> str:
    return " ".join(PROVIDER.sub("", re.sub(r"<[^>]+>", " ", value)).split())


class SchemaBuilder:
    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.definitions: dict[str, dict] = {}

    def resolve(self, value: dict) -> dict:
        if "$ref" not in value:
            return value
        reference = value["$ref"]
        if not reference.startswith("#/"):
            raise ValueError("external_reference")
        target = self.spec
        for segment in reference[2:].split("/"):
            target = target[segment.replace("~1", "/").replace("~0", "~")]
        return target

    def convert(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            reference = value["$ref"]
            key = reference.rsplit("/", 1)[-1]
            if key not in self.definitions:
                self.definitions[key] = {}
                self.definitions[key] = self.convert(self.resolve(value))
            return {"$ref": f"#/$defs/{key}"}
        result = {}
        for key, item in value.items():
            if key in {
                "example",
                "examples",
                "default",
                "xml",
                "nullable",
                "discriminator",
                "readOnly",
                "writeOnly",
                "$schema",
                "$id",
            } or key.startswith("x-"):
                continue
            if key == "description":
                # Descriptions can embed example data; field shapes carry the contract.
                continue
            if key in {"properties", "patternProperties", "$defs", "definitions"}:
                result[key] = {name: self.convert(child) for name, child in item.items()}
            elif key in {"enum", "const"}:
                result[key] = item
            else:
                result[key] = self.convert(item)
        for bound in ("Minimum", "Maximum"):
            exclusive = "exclusive" + bound
            inclusive = bound.lower()
            if isinstance(result.get(exclusive), bool):
                enabled = result.pop(exclusive)
                if enabled and inclusive in result:
                    result[exclusive] = result.pop(inclusive)
        if value.get("nullable"):
            result = {"anyOf": [result, {"type": "null"}]}
        return result

    def finish(self, schema: dict) -> dict:
        if self.definitions:
            schema["$defs"] = self.definitions.copy()
        return schema


def load_operations(directory: Path) -> dict:
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    result = {}
    for filename in source["files"]:
        spec = json.loads((directory / Path(filename).name).read_text(encoding="utf-8"))
        for path, methods in spec.get("paths", {}).items():
            for method in METHODS & methods.keys():
                result[(method.upper(), normalize(path))] = (
                    spec,
                    path,
                    methods[method],
                    methods.get("parameters", []),
                    f"{source['repository']}/blob/{source['commit']}/{filename}",
                )
    return result


def map_contract(card: ToolCard, entry: dict, match: tuple) -> ToolCard:
    spec, official_path, operation, inherited, source = match
    builder = SchemaBuilder(spec)
    body = builder.resolve(operation.get("requestBody", {}))
    content = body.get("content", {})
    if content and "application/json" not in content:
        raise ValueError("unsupported_request_media")
    if entry["body_modes"] and set(entry["body_modes"]) - {"none", "raw"}:
        raise ValueError("unsupported_collection_body")
    official_vars = re.findall(r"\{([^}]+)\}", official_path)
    local_vars = re.findall(r"\{([^}]+)\}", entry["path"])
    aliases = dict(zip(official_vars, local_vars, strict=True))
    properties, required, queries, query_required, explode = {}, [], {}, [], {}
    prefer, request_id = False, False
    for raw in inherited + operation.get("parameters", []):
        parameter = builder.resolve(raw)
        name = parameter["name"]
        location = parameter.get("in")
        schema = builder.convert(parameter.get("schema", {}))
        if location == "path":
            properties[aliases[name]] = schema
            required.append(aliases[name])
        elif location == "query":
            queries[name] = schema
            if parameter.get("required"):
                query_required.append(name)
            explode[name] = parameter.get("explode", True)
        elif location == "header":
            normalized = name.lower()
            prefer |= normalized == "prefer"
            request_id |= normalized == "paypal-request-id"
            if parameter.get("required") and normalized not in {
                "authorization",
                "content-type",
                "accept",
                "paypal-request-id",
            }:
                raise ValueError("required_infrastructure_header")
        else:
            raise ValueError("unsupported_parameter_location")
    if queries:
        properties["query"] = {
            "type": "object",
            "properties": queries,
            "additionalProperties": False,
        }
        if query_required:
            properties["query"]["required"] = query_required
            required.append("query")
    if content:
        properties["body"] = builder.convert(content["application/json"].get("schema", {}))
        if body.get("required"):
            required.append("body")
    schema = builder.finish(
        {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    )
    summary = description(operation.get("summary") or card.name)
    return card.model_copy(
        update={
            "name": summary,
            "description": f"{summary} in sandbox via {entry['method']} {entry['path']}.",
            "input_schema": schema,
            "precondition_key": card.precondition_key
            or ("sandbox_approved_write" if card.operation_type == "write" else None),
            "endpoint": EndpointContract(
                method=entry["method"],
                path=entry["path"],
                schema_source=source,
                query_array_explode=explode,
                prefer_representation=prefer,
                request_id_header=request_id,
            ),
        }
    )


def enrich(cards: list[ToolCard], inventory: dict, directory: Path) -> list[ToolCard]:
    operations = load_operations(directory)
    by_tool = {card.tool_id: card for card in cards}
    for entry in inventory["endpoints"]:
        match = operations.get((entry["method"], normalize(entry["path"])))
        entry["execution_supported"] = False
        if match is None:
            entry["review_reason"] = "official_contract_unavailable"
            continue
        try:
            mapped = map_contract(by_tool[entry["tool_id"]], entry, match)
            from app.tools.registry import ToolRegistry

            ToolRegistry.validate_draft(mapped)
            assert mapped.adapter_key == adapter_key(mapped.endpoint.method, mapped.endpoint.path)
        except (ValueError, KeyError) as error:
            code = str(error)
            entry["review_reason"] = (
                code
                if code
                in {
                    "unsupported_request_media",
                    "unsupported_collection_body",
                    "required_infrastructure_header",
                    "unsupported_parameter_location",
                    "external_reference",
                }
                else "contract_or_request_format_unsupported"
            )
        else:
            by_tool[entry["tool_id"]] = mapped
            entry["execution_supported"] = True
            entry["schema_source"] = mapped.endpoint.schema_source
    return [by_tool[card.tool_id] for card in cards]
