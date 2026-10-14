"""Load the existing local settings and development identity."""

import json
from pathlib import Path

from app.security import Principal, TokenAuthenticator
from app.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def load_settings() -> Settings:
    return Settings(_env_file=ROOT / ".env")


def local_principal(settings: Settings) -> tuple[Principal, str]:
    token_path = ROOT / "runtime" / "local-auth.json"
    try:
        token = json.loads(token_path.read_text(encoding="utf-8"))["bearer_token"]
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("Local identity is missing; run the existing setup script first") from None
    principal = TokenAuthenticator(settings).authenticate(token)
    return principal, token
