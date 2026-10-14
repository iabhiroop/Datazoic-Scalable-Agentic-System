"""Opt-in PostgreSQL tests. Point TEST_DATABASE_URL at a disposable migrated database."""

import hashlib
import json
import os
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.db.session import event_loop_factory
from app.main import create_app
from app.settings import Settings


@pytest.fixture
def database_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to an isolated PostgreSQL/pgvector database")
    return url


@pytest.fixture
def config(database_url):
    tenant = str(uuid4())
    bindings = [
        ("owner", tenant, "owner", ["runs:read", "runs:write"]),
        ("peer", tenant, "peer", ["runs:read", "runs:write"]),
        ("manager", tenant, "manager", ["runs:read", "runs:write", "runs:manage"]),
        ("outsider", str(uuid4()), "other", ["runs:read", "runs:write", "runs:manage"]),
        ("reader", tenant, "reader", ["runs:read"]),
    ]
    return Settings(
        _env_file=None,
        database_url=database_url.replace("postgresql://", "postgresql+psycopg://", 1),
        checkpoint_database_url=database_url,
        embedding_dimension=int(os.environ.get("TEST_EMBEDDING_DIMENSION", "3")),
        auth_bindings_json=json.dumps(
            [
                {
                    "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    "tenant_id": tenant_id,
                    "user_id": user,
                    "scopes": scopes,
                }
                for token, tenant_id, user, scopes in bindings
            ]
        ),
    )


@pytest.fixture
def client(config):
    with TestClient(
        create_app(config), backend_options={"loop_factory": event_loop_factory}
    ) as client:
        yield client


@pytest.fixture
def connection(database_url):
    with psycopg.connect(database_url, connect_timeout=5, autocommit=True) as connection:
        yield connection
