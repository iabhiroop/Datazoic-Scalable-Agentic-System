"""Authentication and unavailable-resource behavior without a live database."""

import hashlib
import json
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import create_app
from app.security import TokenAuthenticator
from app.settings import Settings


def settings():
    return Settings(
        _env_file=None,
        database_url=None,
        checkpoint_database_url=None,
        auth_bindings_json=json.dumps(
            [
                {
                    "token_sha256": hashlib.sha256(b"unit-token").hexdigest(),
                    "tenant_id": str(UUID(int=1)),
                    "user_id": "owner",
                    "scopes": ["runs:read", "runs:write"],
                }
            ]
        ),
    )


def test_live_but_not_ready():
    with TestClient(create_app(settings())) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        missing = client.get("/missing-route")
        assert missing.status_code == 404
        assert missing.json()["code"] == "not_found"


def test_bearer_required_and_invalid_token_rejected():
    with TestClient(create_app(settings())) as client:
        missing = client.post(
            "/runs", headers={"Idempotency-Key": "example"}, json={"message": "Hi"}
        )
        assert missing.status_code == 401
        assert missing.headers["WWW-Authenticate"] == "Bearer"
        invalid = client.get(
            "/runs/00000000-0000-0000-0000-000000000001", headers={"Authorization": "Bearer wrong"}
        )
        assert invalid.status_code == 401
        assert invalid.json()["request_id"] == invalid.headers["X-Request-ID"]


def test_valid_identity_cannot_submit_without_persistence():
    with TestClient(create_app(settings())) as client:
        headers = {"Authorization": "Bearer unit-token", "Idempotency-Key": "example"}
        assert client.post("/runs", headers=headers, json={"message": "Hi"}).status_code == 503
        bad = client.post("/runs", headers=headers, json={"message": "Hi", "tenant_id": "secret"})
        assert bad.status_code == 422
        assert "secret" not in bad.text


def test_scope_is_server_bound():
    authenticator = TokenAuthenticator(settings())
    principal = authenticator.authenticate("unit-token")
    assert principal.tenant_id == UUID(int=1)
    assert "runs:write" in principal.scopes
    with pytest.raises(HTTPException) as error:
        authenticator.authenticate("other-token")
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_refresh_detects_revoked_binding():
    authenticator = TokenAuthenticator(settings())
    principal = authenticator.authenticate("unit-token")
    authenticator.bindings.clear()
    with pytest.raises(HTTPException) as error:
        await authenticator.refresh(principal)
    assert error.value.status_code == 403


def test_duplicate_hash_cannot_choose_between_tenants():
    config = settings()
    values = json.loads(config.auth_bindings_json.get_secret_value())
    values.append({**values[0], "tenant_id": str(UUID(int=2))})
    config = Settings(_env_file=None, auth_bindings_json=json.dumps(values))
    with pytest.raises(ValueError, match="unique"):
        TokenAuthenticator(config)
