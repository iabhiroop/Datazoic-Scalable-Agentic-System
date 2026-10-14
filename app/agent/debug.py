"""Per-run application exchange events for the local chat debugger."""

from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic import SecretStr

from app.settings import Settings

DebugWriter = Callable[[str, dict[str, Any]], Awaitable[None]]
_writer: ContextVar[DebugWriter | None] = ContextVar("run_debug_writer", default=None)
_CREDENTIAL_KEYS = {
    "password", "passwd", "secret", "secretkey", "apikey", "authorization",
    "cookie", "setcookie", "token", "bearertoken", "accesstoken",
    "refreshtoken", "idtoken", "clientid", "clientsecret", "privatekey",
    "publickey", "accesskey", "credentials", "credential",
}


def without_credentials(value: Any, settings: Settings | None = None) -> Any:
    """Keep exchange content intact except credential-shaped mapping fields."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = "".join(character for character in str(key).lower() if character.isalnum())
            if normalized in _CREDENTIAL_KEYS or normalized.endswith(
                ("apikey", "secretkey", "clientsecret", "accesstoken", "refreshtoken", "password")
            ):
                result[key] = "[credential omitted]"
            else:
                result[key] = without_credentials(item, settings)
        return result
    if isinstance(value, list):
        return [without_credentials(item, settings) for item in value]
    if isinstance(value, str) and settings is not None:
        return mask_known_credentials(value, settings)
    return value


def mask_known_credentials(value: str, settings: Settings) -> str:
    """Preserve error text while removing configured credential values."""
    secrets = []
    for name in type(settings).model_fields:
        configured = getattr(settings, name)
        if isinstance(configured, SecretStr) and configured.get_secret_value():
            secrets.append(configured.get_secret_value())
    if settings.langfuse_public_key:
        secrets.append(settings.langfuse_public_key)
    for secret in sorted(set(secrets), key=len, reverse=True):
        value = value.replace(secret, "[credential omitted]")
    return value


@contextmanager
def capture_debug_events(writer: DebugWriter):
    token = _writer.set(writer)
    try:
        yield
    finally:
        _writer.reset(token)


async def record_debug_event(event_type: str, evidence: dict[str, Any]) -> None:
    writer = _writer.get()
    if writer is not None:
        await writer(event_type, evidence)
