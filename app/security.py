"""Trusted server-side identities. Clients cannot select a tenant/user/scope."""

import hashlib
import hmac
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.settings import Settings

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    tenant_id: UUID
    user_id: str
    scopes: frozenset[str]


class IdentityBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tenant_id: UUID
    user_id: str = Field(min_length=1)
    scopes: frozenset[str]


class TokenAuthenticator:
    """Small initial auth backend. Replace with an identity provider when selected."""

    def __init__(self, settings: Settings) -> None:
        self.bindings: list[IdentityBinding] = []
        if settings.auth_bindings_json is not None:
            try:
                from pydantic import TypeAdapter

                self.bindings = TypeAdapter(list[IdentityBinding]).validate_json(
                    settings.auth_bindings_json.get_secret_value()
                )
            except ValidationError:
                raise ValueError("AUTH_BINDINGS_JSON is invalid") from None
            hashes = [binding.token_sha256 for binding in self.bindings]
            identities = [(binding.tenant_id, binding.user_id) for binding in self.bindings]
            if len(set(hashes)) != len(hashes) or len(set(identities)) != len(identities):
                raise ValueError("Auth token hashes and tenant/user bindings must be unique")

    @property
    def configured(self) -> bool:
        return bool(self.bindings)

    def authenticate(self, token: str) -> Principal:
        if not self.configured:
            raise HTTPException(503, "Authentication is not configured")
        digest = hashlib.sha256(token.encode()).hexdigest()
        for binding in self.bindings:
            if hmac.compare_digest(binding.token_sha256, digest):
                return Principal(binding.tenant_id, binding.user_id, binding.scopes)
        raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    async def refresh(self, principal: Principal) -> Principal:
        for binding in self.bindings:
            if (binding.tenant_id, binding.user_id) == (principal.tenant_id, principal.user_id):
                return Principal(binding.tenant_id, binding.user_id, binding.scopes)
        raise HTTPException(403, "Identity is no longer authorized")


async def authenticated_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    if credentials is None:
        raise HTTPException(
            401, "Bearer authentication required", headers={"WWW-Authenticate": "Bearer"}
        )
    return request.app.state.authenticator.authenticate(credentials.credentials)


class AuthorizationService:
    def __init__(self, authenticator: TokenAuthenticator, database=None) -> None:
        self.authenticator = authenticator
        self.database = database

    async def refresh(self, principal: Principal) -> Principal:
        return await self.authenticator.refresh(principal)

    async def require_owner(self, principal: Principal, run_id: UUID) -> None:
        if self.database is None:
            raise HTTPException(503, "Database is not configured")
        from app.db.repositories import RunRepository

        async with self.database.session() as session:
            await RunRepository(session).get(run_id, principal)

    async def require_scopes(self, principal: Principal, scopes: list[str]) -> None:
        if not set(scopes).issubset(principal.scopes):
            raise HTTPException(403, "Required permissions are missing")

    async def validate_approval(
        self, principal: Principal, operation_id: UUID, input_hash: str, evidence: dict[str, Any]
    ) -> None:
        raise NotImplementedError("Bind approval to target, amount, currency and payload hash")


def redact(data: dict[str, Any]) -> dict[str, Any]:
    """Redact exports, without erasing essential restricted recovery records."""
    from app.observability import SecretRedactor

    return SecretRedactor(Settings())(data)
