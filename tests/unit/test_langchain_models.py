import sys
from types import ModuleType

from app.agent.chat_models import create_chat_model
from app.settings import Settings


def test_gemini_langchain_model_uses_existing_provider_settings(monkeypatch):
    calls = {}

    class ChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    module = ModuleType("langchain_google_genai")
    module.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
    monkeypatch.setitem(sys.modules, module.__name__, module)
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        gemini_api_key="test-key",
        gemini_model_name="gemini-test",
        llm_timeout_seconds=17,
        llm_max_attempts=4,
    )

    assert isinstance(create_chat_model(settings), ChatGoogleGenerativeAI)
    assert calls == {
        "model": "gemini-test",
        "google_api_key": "test-key",
        "vertexai": False,
        "temperature": 0,
        "timeout": 17,
        "max_retries": 4,
    }


def test_ollama_langchain_model_uses_existing_provider_settings(monkeypatch):
    calls = {}

    class ChatOllama:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    module = ModuleType("langchain_ollama")
    module.ChatOllama = ChatOllama
    monkeypatch.setitem(sys.modules, module.__name__, module)
    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        ollama_model_name="local-test",
        ollama_base_url="http://localhost:11434",
        llm_timeout_seconds=19,
    )

    assert isinstance(create_chat_model(settings), ChatOllama)
    assert calls == {
        "model": "local-test",
        "base_url": "http://localhost:11434",
        "temperature": 0,
        "client_kwargs": {"timeout": 19},
        "num_predict": 4096,
    }
