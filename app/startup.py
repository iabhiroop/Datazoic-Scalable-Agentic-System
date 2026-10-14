"""Small, secret-safe checks invoked by the Windows startup script."""

import json
import sys
from pathlib import Path

import psycopg

from app.security import TokenAuthenticator
from app.settings import Settings
from app.worker.main import LOCK_KEY


def validate_local_configuration() -> int:
    try:
        settings = Settings()
        required = {
            "DATABASE_URL": settings.database_url,
            "CHECKPOINT_DATABASE_URL": settings.checkpoint_database_url,
            "AUTH_BINDINGS_JSON": settings.auth_bindings_json,
            "OBJECT_STORAGE_ENDPOINT": settings.object_storage_endpoint,
            "OBJECT_STORAGE_ACCESS_KEY": settings.object_storage_access_key,
            "OBJECT_STORAGE_SECRET_KEY": settings.object_storage_secret_key,
            "GEMINI_API_KEY": settings.gemini_api_key,
            "EMBEDDING_MODEL": settings.embedding_model,
            "EMBEDDING_VERSION": settings.embedding_version,
            "EMBEDDING_DIMENSION": settings.embedding_dimension,
        }
        if settings.llm_provider.strip().lower() == "gemini":
            required["GEMINI_MODEL_NAME"] = settings.gemini_model_name
        elif settings.llm_provider.strip().lower() == "ollama":
            required["OLLAMA_MODEL_NAME"] = settings.ollama_model_name
            required["OLLAMA_BASE_URL"] = settings.ollama_base_url
        else:
            raise ValueError("Unsupported local LLM provider")
        missing = [name for name, value in required.items() if not value]
        if missing:
            print("Missing local configuration: " + ", ".join(missing))
            return 2
        token = json.loads(Path("runtime/local-auth.json").read_text(encoding="utf-8"))[
            "bearer_token"
        ]
        principal = TokenAuthenticator(settings).authenticate(token)
        if not {"runs:read", "runs:write"}.issubset(principal.scopes):
            raise ValueError("Local identity lacks run permissions")
    except Exception:  # noqa: BLE001 -- configuration values must never reach the console
        print(
            "Local settings or development identity are invalid; check .env and runtime/local-auth.json."
        )
        return 2
    return 0


def worker_lock_state() -> int:
    try:
        url = Settings().checkpoint_database_url.get_secret_value()
        with psycopg.connect(url, autocommit=True, connect_timeout=5) as connection:
            available = connection.execute(
                "SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,)
            ).fetchone()[0]
            if available:
                connection.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
    except Exception:  # noqa: BLE001 -- DSNs and connection details stay private
        print("Could not check worker ownership.", file=sys.stderr)
        return 2
    print("available" if available else "busy")
    return 0


def worker_lock_owner() -> int:
    """Print safe owner metadata for the project worker advisory lock."""
    try:
        url = Settings().checkpoint_database_url.get_secret_value()
        with psycopg.connect(url, connect_timeout=5) as connection:
            row = connection.execute(
                """
                SELECT activity.application_name
                FROM pg_locks AS locks
                JOIN pg_stat_activity AS activity ON activity.pid = locks.pid
                WHERE locks.locktype = 'advisory'
                  AND locks.classid = 0
                  AND locks.objid = %s
                  AND locks.objsubid = 1
                  AND locks.granted
                LIMIT 1
                """,
                (LOCK_KEY,),
            ).fetchone()
    except Exception:  # noqa: BLE001 -- connection details must not reach the console
        print("Could not inspect worker ownership.", file=sys.stderr)
        return 2
    owner = parse_worker_application_name(row[0]) if row else None
    print(json.dumps({"locked": row is not None, "owner": owner}))
    return 0


def parse_worker_application_name(application_name: str) -> dict[str, str] | None:
    """Parse only the deliberately small, non-secret worker marker."""
    if not application_name.startswith("dzw:"):
        return None
    parts = application_name.split(":")
    if (
        len(parts) != 5
        or not parts[1]
        or not parts[2].isdigit()
        or parts[3] not in {"v1", "v2", "v3", "v4"}
        or parts[4] not in {"v1", "v2"}
    ):
        return None
    return {
        "worker_id": parts[1],
        "pid": parts[2],
        "graph_version": parts[3],
        "prompt_version": parts[4],
    }


def active_run_count() -> int:
    """Return only a count so shutdown can avoid orphaning an in-flight run."""
    try:
        url = Settings().checkpoint_database_url.get_secret_value()
        with psycopg.connect(url, connect_timeout=5) as connection:
            count = connection.execute(
                "SELECT count(*) FROM workflow_runs WHERE status = 'running'"
            ).fetchone()[0]
    except Exception:  # noqa: BLE001 -- DSNs and run data stay private
        print("Could not check active runs.", file=sys.stderr)
        return 2
    print(count)
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m app.startup validate|worker-lock|worker-owner|active-runs", file=sys.stderr)
        return 2
    if sys.argv[1] == "validate":
        return validate_local_configuration()
    if sys.argv[1] == "worker-lock":
        return worker_lock_state()
    if sys.argv[1] == "worker-owner":
        return worker_lock_owner()
    if sys.argv[1] == "active-runs":
        return active_run_count()
    print("Unknown startup check.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
