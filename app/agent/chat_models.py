"""LangChain chat model factory for the current agent graph."""

from app.settings import Settings


def create_chat_model(settings: Settings):
    """Create the configured provider model using its LangChain integration."""
    provider = settings.llm_provider.strip().lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.gemini_api_key or not settings.gemini_model_name:
            raise ValueError("GEMINI_MODEL_NAME and GEMINI_API_KEY are required")
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model_name,
            google_api_key=settings.gemini_api_key.get_secret_value(),
            vertexai=False,
            temperature=0,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_attempts,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model_name,
            base_url=settings.ollama_base_url,
            temperature=0,
            client_kwargs={"timeout": settings.llm_timeout_seconds},
            num_predict=4096,
        )
    raise ValueError(f"Unsupported LangChain chat provider: {provider}")
