"""dotenv precedence and test isolation without real configuration or credentials."""

import os

import pytest

from app.settings import Settings


@pytest.fixture(autouse=True)
def restore_dotenv_values():
    names = ("GEMINI_API_KEY", "LLM_TIMEOUT_SECONDS", "PYTHON_DOTENV_DISABLED")
    original = {name: os.environ.get(name) for name in names}
    yield
    for name, value in original.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def prepare(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "GEMINI_API_KEY=synthetic-test-key\n"
        "LLM_TIMEOUT_SECONDS=12\n",
        encoding="utf-8",
    )
    return path


def test_dotenv_loads_and_pydantic_masks_secret(monkeypatch, tmp_path):
    path = prepare(monkeypatch, tmp_path)
    config = Settings(_env_file=path)
    assert config.gemini_api_key.get_secret_value() == "synthetic-test-key"
    assert config.llm_timeout_seconds == 12
    assert "synthetic-test-key" not in repr(config)


def test_process_environment_has_precedence(monkeypatch, tmp_path):
    path = prepare(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-process-key")
    config = Settings(_env_file=path)
    assert config.gemini_api_key.get_secret_value() == "synthetic-process-key"


def test_explicit_settings_have_precedence(monkeypatch, tmp_path):
    path = prepare(monkeypatch, tmp_path)
    config = Settings(_env_file=path, gemini_api_key="synthetic-explicit-key")
    assert config.gemini_api_key.get_secret_value() == "synthetic-explicit-key"


def test_no_env_file_skips_loader(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    assert Settings(_env_file=None).gemini_api_key is None


def test_disabled_dotenv_is_not_bypassed(monkeypatch, tmp_path):
    path = prepare(monkeypatch, tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    assert Settings(_env_file=path).gemini_api_key is None


def test_v4_is_default_and_unknown_graph_versions_fail_closed():
    config = Settings(_env_file=None)
    assert config.graph_version == "v4"
    assert config.prompt_version == "v2"
    with pytest.raises(ValueError, match="No handler"):
        Settings(_env_file=None, graph_version="v3", prompt_version="v1")


