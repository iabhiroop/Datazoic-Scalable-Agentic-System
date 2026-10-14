from uuid import uuid4

import httpx
import pytest

from app.settings import Settings
from app.tools.contracts import ToolContext
from app.tools.paypal import PayPalEndpoint, PayPalEndpointAdapter, PayPalSandbox


@pytest.mark.asyncio
async def test_paypal_read_uses_sandbox_oauth_and_refreshes_expired_token():
    tenant = uuid4()
    requests = []

    def respond(request):
        requests.append(request)
        assert request.url.host == "api-m.sandbox.paypal.com"
        if request.url.path == "/v1/oauth2/token":
            return httpx.Response(
                200, json={"access_token": f"token-{len(requests)}", "expires_in": 3600}
            )
        if len(requests) == 2:
            return httpx.Response(401, json={})
        return httpx.Response(200, json={"id": "INV2-TEST"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    settings = Settings(
        _env_file=None,
        paypal_sandbox_client_id="client",
        paypal_sandbox_client_secret="secret",
        paypal_sandbox_tenant_id=tenant,
    )
    adapter = PayPalEndpointAdapter(
        PayPalSandbox(settings, client),
        PayPalEndpoint("GET", "/v2/invoicing/invoices/{invoice_id}", "read"),
    )
    context = ToolContext(tenant, "user", uuid4(), uuid4(), None)
    outcome = await adapter.invoke({"invoice_id": "INV2-TEST"}, context)
    assert outcome.state == "succeeded"
    assert outcome.data["response"]["id"] == "INV2-TEST"
    assert len(requests) == 4
    await client.aclose()


@pytest.mark.asyncio
async def test_paypal_write_requires_identity_and_marks_timeout_unknown():
    tenant = uuid4()

    def respond(request):
        if request.url.path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})
        assert request.headers["PayPal-Request-Id"] == "stable-key"
        raise httpx.ReadTimeout("synthetic timeout")

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    settings = Settings(
        _env_file=None,
        paypal_sandbox_client_id="client",
        paypal_sandbox_client_secret="secret",
        paypal_sandbox_tenant_id=tenant,
    )
    adapter = PayPalEndpointAdapter(
        PayPalSandbox(settings, client), PayPalEndpoint("POST", "/v2/invoicing/invoices", "write")
    )
    context = ToolContext(tenant, "user", uuid4(), uuid4(), None)
    assert (await adapter.invoke({"body": {}}, context)).error_code == "missing_operation_identity"
    context = ToolContext(tenant, "user", uuid4(), uuid4(), "stable-key")
    assert (await adapter.invoke({"body": {}}, context)).state == "unknown"
    await client.aclose()


def test_paypal_rejects_arbitrary_endpoint_host():
    with pytest.raises(ValueError):
        PayPalEndpoint("GET", "https://example.org/v2/invoicing/invoices", "read")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected",
    [(400, "failed"), (409, "unknown"), (500, "unknown")],
)
async def test_paypal_write_error_classification(status, expected):
    tenant = uuid4()

    def respond(request):
        if request.url.path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})
        return httpx.Response(status, json={"name": "synthetic error"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    settings = Settings(
        _env_file=None,
        paypal_sandbox_client_id="client",
        paypal_sandbox_client_secret="secret",
        paypal_sandbox_tenant_id=tenant,
    )
    adapter = PayPalEndpointAdapter(
        PayPalSandbox(settings, client), PayPalEndpoint("POST", "/v2/invoicing/invoices", "write")
    )
    result = await adapter.invoke(
        {"body": {}}, ToolContext(tenant, "user", uuid4(), uuid4(), "key")
    )
    assert result.state == expected
    await client.aclose()


@pytest.mark.asyncio
async def test_paypal_tenant_mismatch_never_uses_credentials():
    called = False

    def respond(request):
        nonlocal called
        called = True
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    settings = Settings(_env_file=None, paypal_sandbox_tenant_id=uuid4())
    adapter = PayPalEndpointAdapter(
        PayPalSandbox(settings, client), PayPalEndpoint("GET", "/v2/invoicing/invoices", "read")
    )
    result = await adapter.invoke({}, ToolContext(uuid4(), "user", uuid4(), uuid4(), None))
    assert result.error_code == "paypal_account_unavailable"
    assert not called
    await client.aclose()
