"""Fixed-host PayPal sandbox endpoint adapter. No Postman scripts are executed."""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from app.settings import Settings
from app.tools.contracts import EndpointContract, ToolCard, ToolContext, ToolOutcome

SANDBOX_ORIGIN = "https://api-m.sandbox.paypal.com"
PATH_PARAMETER = re.compile(r"\{([A-Za-z][\w-]*)\}")
REVIEWED_INVOICE_ENDPOINTS = (
    ("GET", "/v2/invoicing/invoices"),
    ("GET", "/v2/invoicing/invoices/{invoice_id}"),
    ("POST", "/v2/invoicing/invoices"),
    ("POST", "/v2/invoicing/invoices/{invoice_id}/send"),
)


def adapter_key(method: str, path: str) -> str:
    digest = hashlib.sha256(f"{method} {path}".encode()).hexdigest()[:12]
    return f"paypal_{method.lower()}_{digest}"


def invoice_adapters(transport: "PayPalSandbox") -> dict[str, "PayPalEndpointAdapter"]:
    return {
        adapter_key(method, path): PayPalEndpointAdapter(
            transport, PayPalEndpoint(method, path, "read" if method == "GET" else "write")
        )
        for method, path in REVIEWED_INVOICE_ENDPOINTS
    }


def validate_endpoint_card(card: ToolCard) -> None:
    endpoint = card.endpoint
    if endpoint is None:
        raise ValueError("Endpoint contract is unavailable")
    PayPalEndpoint(endpoint.method, endpoint.path, card.operation_type)
    identity = adapter_key(endpoint.method, endpoint.path)
    if (
        card.tool_id != identity
        or card.adapter_key != identity
        or card.provider != "paypal"
        or card.provider_account_key != "local_sandbox"
    ):
        raise ValueError("Endpoint contract does not match the adapter identity")


def load_endpoint_catalog(path: Path) -> dict[str, EndpointContract]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported endpoint catalog version")
    catalog = {}
    for raw in payload["endpoints"]:
        key = raw["adapter_key"]
        endpoint = EndpointContract.model_validate(raw["endpoint"])
        PayPalEndpoint(
            endpoint.method, endpoint.path, "read" if endpoint.method == "GET" else "write"
        )
        if key != adapter_key(endpoint.method, endpoint.path) or key in catalog:
            raise ValueError("Invalid endpoint catalog identity")
        catalog[key] = endpoint
    return catalog


def save_endpoint_catalog(path: Path, cards: list[ToolCard]) -> None:
    catalog = load_endpoint_catalog(path)
    for card in cards:
        if card.endpoint is None:
            continue
        validate_endpoint_card(card)
        catalog[card.adapter_key] = card.endpoint
    entries = [
        {"adapter_key": key, "endpoint": endpoint.model_dump(mode="json")}
        for key, endpoint in sorted(catalog.items())
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"version": 1, "endpoints": entries}, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def catalog_adapters(transport: "PayPalSandbox") -> dict[str, "PayPalEndpointAdapter"]:
    adapters = invoice_adapters(transport)
    for key, endpoint in load_endpoint_catalog(transport.settings.endpoint_catalog_path).items():
        if endpoint.method != "GET" and not transport.settings.paypal_sandbox_writes_enabled:
            continue
        adapters[key] = PayPalEndpointAdapter(
            transport,
            PayPalEndpoint(
                endpoint.method,
                endpoint.path,
                "read" if endpoint.method == "GET" else "write",
                endpoint.query_array_explode,
                endpoint.prefer_representation,
                endpoint.request_id_header,
            ),
        )
    return adapters


@dataclass(frozen=True)
class PayPalEndpoint:
    method: str
    path: str
    operation_type: str
    query_array_explode: dict[str, bool] | None = None
    prefer_representation: bool = False
    request_id_header: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("Unsupported PayPal method")
        if not self.path.startswith("/v") or "//" in self.path or ".." in self.path:
            raise ValueError("Unsafe PayPal path")
        if re.search(r"[^/A-Za-z0-9._{}~-]", self.path):
            raise ValueError("Unsafe PayPal path")
        if self.operation_type != ("read" if self.method == "GET" else "write"):
            raise ValueError("Incorrect PayPal operation type")


class PayPalSandbox:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.tool_timeout_seconds)
        self.owns_client = client is None
        self._token: str | None = None
        self._expires_at = 0.0
        self._token_lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self.owns_client:
            await self.client.aclose()

    async def token(self, *, force: bool = False) -> str:
        async with self._token_lock:
            if not force and self._token and time.monotonic() < self._expires_at:
                return self._token
            client_id = self.settings.paypal_sandbox_client_id
            secret = self.settings.paypal_sandbox_client_secret
            if not client_id or not secret:
                raise ValueError("PayPal sandbox credentials are not configured")
            response = await self.client.post(
                SANDBOX_ORIGIN + "/v1/oauth2/token",
                auth=(client_id.get_secret_value(), secret.get_secret_value()),
                data={"grant_type": "client_credentials"},
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
            token = payload.get("access_token")
            if not isinstance(token, str) or not token:
                raise ValueError("PayPal OAuth response has no access token")
            self._token = token
            self._expires_at = time.monotonic() + max(0, int(payload.get("expires_in", 0)) - 60)
            return token


class PayPalEndpointAdapter:
    def __init__(self, transport: PayPalSandbox, endpoint: PayPalEndpoint) -> None:
        self.transport = transport
        self.endpoint = endpoint

    async def authorize_result(
        self, arguments: dict[str, Any], data: dict[str, Any], context: ToolContext
    ) -> None:
        if context.tenant_id != self.transport.settings.paypal_sandbox_tenant_id:
            raise ValueError("PayPal result is outside the bound sandbox tenant")

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutcome:
        if (
            self.transport.settings.paypal_sandbox_tenant_id is None
            or context.tenant_id != self.transport.settings.paypal_sandbox_tenant_id
        ):
            return ToolOutcome(state="failed", data={}, error_code="paypal_account_unavailable")
        if self.endpoint.operation_type == "write" and not context.idempotency_key:
            return ToolOutcome(state="failed", data={}, error_code="missing_operation_identity")
        path = self.endpoint.path
        for variable in PATH_PARAMETER.findall(path):
            value = arguments.get(variable)
            if not isinstance(value, str) or not value or len(value) > 255:
                return ToolOutcome(state="failed", data={}, error_code="invalid_path_parameter")
            path = path.replace("{" + variable + "}", quote(value, safe=""))
        if "{" in path or "}" in path:
            return ToolOutcome(state="failed", data={}, error_code="invalid_path_parameter")
        query = arguments.get("query") or {}
        if not isinstance(query, dict) or any(
            not isinstance(value, (str, int, float, bool, list))
            or isinstance(value, list)
            and any(not isinstance(item, (str, int, float, bool)) for item in value)
            for value in query.values()
        ):
            return ToolOutcome(state="failed", data={}, error_code="invalid_query")
        params = []
        for name, value in query.items():
            values = value if isinstance(value, list) else [value]
            text = [str(item).lower() if isinstance(item, bool) else str(item) for item in values]
            if isinstance(value, list) and not (self.endpoint.query_array_explode or {}).get(
                name, True
            ):
                params.append((name, ",".join(text)))
            else:
                params.extend((name, item) for item in text)
        body = arguments.get("body")
        if body is not None and not isinstance(body, (dict, list)):
            return ToolOutcome(state="failed", data={}, error_code="invalid_body")
        try:
            access_token = await self.transport.token()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, ValueError):
            return ToolOutcome(
                state="failed",
                data={},
                error_code="paypal_auth_unavailable",
                effect_known_absent=True,
            )
        try:
            headers = {"Authorization": "Bearer " + access_token, "Accept": "application/json"}
            if context.idempotency_key and self.endpoint.request_id_header:
                headers["PayPal-Request-Id"] = context.idempotency_key
            if self.endpoint.prefer_representation:
                headers["Prefer"] = "return=representation"
            response = await self.transport.client.request(
                self.endpoint.method,
                SANDBOX_ORIGIN + path,
                params=params,
                json=body if body is not None else None,
                headers=headers,
            )
            if response.status_code == 401 and self.endpoint.operation_type == "read":
                access_token = await self.transport.token(force=True)
                headers["Authorization"] = "Bearer " + access_token
                response = await self.transport.client.request(
                    self.endpoint.method,
                    SANDBOX_ORIGIN + path,
                    params=params,
                    json=body if body is not None else None,
                    headers=headers,
                )
        except (httpx.TimeoutException, httpx.TransportError):
            return ToolOutcome(
                state="unknown" if self.endpoint.operation_type == "write" else "failed",
                data={},
                error_code="paypal_transport_error",
                retryable=self.endpoint.operation_type == "read",
            )
        except (ValueError, httpx.HTTPStatusError):
            return ToolOutcome(state="failed", data={}, error_code="paypal_auth_unavailable")
        if response.status_code >= 500:
            return ToolOutcome(
                state="unknown" if self.endpoint.operation_type == "write" else "failed",
                data={},
                error_code="paypal_server_error",
                retryable=self.endpoint.operation_type == "read",
            )
        if response.status_code >= 400:
            uncertain = self.endpoint.operation_type == "write" and response.status_code not in {
                400,
                401,
                403,
                404,
                422,
            }
            return ToolOutcome(
                state="unknown" if uncertain else "failed",
                data={"http_status": response.status_code},
                error_code="paypal_request_rejected",
                effect_known_absent=not uncertain,
            )
        try:
            payload = response.json() if response.content else {}
        except ValueError:
            return ToolOutcome(
                state="unknown" if self.endpoint.operation_type == "write" else "failed",
                data={"http_status": response.status_code},
                error_code="invalid_provider_response",
            )
        if not isinstance(payload, dict):
            payload = {"value": payload}
        asynchronous = self.endpoint.operation_type == "write" and (
            response.status_code == 202
            or payload.get("status") in {"PENDING", "PROCESSING"}
            or isinstance(payload.get("batch_header"), dict)
            and payload["batch_header"].get("batch_status") in {"PENDING", "PROCESSING"}
        )
        return ToolOutcome(
            state="pending" if asynchronous else "succeeded",
            data={"http_status": response.status_code, "response": payload},
            provider_operation_id=str(payload.get("id")) if payload.get("id") else None,
        )

    async def reconcile(self, operation: dict[str, Any], context: ToolContext) -> ToolOutcome:
        return ToolOutcome(state="unknown", data={}, error_code="manual_reconciliation_required")
