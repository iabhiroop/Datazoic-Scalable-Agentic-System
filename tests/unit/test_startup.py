"""Safe worker-lock reporting for the local PowerShell lifecycle script."""

import json
from types import SimpleNamespace

from app import startup


def test_parse_worker_application_name():
    owner = startup.parse_worker_application_name("dzw:abc-123:4567:v3:v2")
    assert owner == {
        "worker_id": "abc-123",
        "pid": "4567",
        "graph_version": "v3",
        "prompt_version": "v2",
    }


def test_parse_worker_application_name_rejects_unknown_or_malformed_values():
    assert startup.parse_worker_application_name("python") is None
    assert startup.parse_worker_application_name("dzw:id:not-a-pid:v3:v2") is None


def test_worker_lock_owner_reports_metadata_without_dsn(monkeypatch, capsys):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args):
            return self

        def fetchone(self):
            return ("dzw:worker-123:9876:v3:v2",)

    monkeypatch.setattr(
        startup,
        "Settings",
        lambda: SimpleNamespace(
            checkpoint_database_url=SimpleNamespace(get_secret_value=lambda: "hidden-dsn")
        ),
    )
    monkeypatch.setattr(startup.psycopg, "connect", lambda *_args, **_kwargs: FakeConnection())

    assert startup.worker_lock_owner() == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "locked": True,
        "owner": {
            "worker_id": "worker-123",
            "pid": "9876",
            "graph_version": "v3",
            "prompt_version": "v2",
        },
    }
