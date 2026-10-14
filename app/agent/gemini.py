"""Google AI Studio Interactions adapter; provider-specific details stay here."""

import asyncio
import math
from typing import Any

from google import genai
from google.genai import types

from app.agent.gateway import LLMError, LLMRequest, LLMResponse, retry_llm_call
from app.agent.debug import mask_known_credentials
from app.settings import Settings


class GeminiProvider:
    provider_name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        model = (settings.gemini_model_name or "").strip().removeprefix("models/")
        key = settings.gemini_api_key
        if not model or key is None or not key.get_secret_value().strip():
            raise ValueError("GEMINI_MODEL_NAME and GEMINI_API_KEY are required")
        self.model = model
        self.thinking_level = settings.llm_thinking_level
        self.timeout = settings.llm_timeout_seconds
        self.client = genai.Client(
            vertexai=False,
            api_key=key.get_secret_value().strip(),
            http_options=types.HttpOptions(
                timeout=int(self.timeout * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        parameters: dict[str, Any] = {
            "model": self.model,
            "input": request.continuation
            or (
                [{"type": "user_input", "content": [{"type": "text", "text": request.prompt}]}]
                if request.tool_declarations
                else request.prompt
            ),
            "system_instruction": request.system_instruction,
            "generation_config": {
                "thinking_level": self.thinking_level,
                "thinking_summaries": "auto",
            },
            "store": False,
            "stream": False,
            "timeout": self.timeout,
        }
        if request.tool_declarations:
            parameters["tools"] = [
                {
                    "type": "function",
                    "name": declaration.name,
                    "description": declaration.description,
                    "parameters": declaration.parameters,
                }
                for declaration in request.tool_declarations
            ]
            if request.force_tool_call:
                parameters["generation_config"]["tool_choice"] = "any"
        if request.web_search:
            parameters["tools"] = [{"type": "google_search"}]
        if request.response_schema is not None:
            parameters["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": request.response_schema,
            }
        try:
            interaction = await self.client.aio.interactions.create(**parameters)
        except TimeoutError as error:
            detail = mask_known_credentials(
                f"{type(error).__name__}: {error or 'Gemini request timed out'}",
                self.settings,
            )
            raise LLMError(
                "timeout", "Gemini request timed out", retryable=True, detail=detail
            ) from None
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 -- narrow SDK boundary, discard unsafe SDK messages
            status = getattr(error, "status_code", None) or getattr(error, "code", None)
            detail = mask_known_credentials(f"{type(error).__name__}: {error}", self.settings)
            if status in (401, 403):
                raise LLMError(
                    "authentication", "Gemini authentication or access denied", detail=detail
                ) from None
            if status == 429:
                raise LLMError(
                    "rate_limited", "Gemini rate limit reached", retryable=True, detail=detail
                ) from None
            if status == 503:
                raise LLMError(
                    "provider_busy", "Gemini is temporarily busy", retryable=True, detail=detail
                ) from None
            if isinstance(status, int) and 500 <= status < 600:
                raise LLMError(
                    "provider_unavailable", "Gemini service is temporarily unavailable",
                    retryable=True, detail=detail,
                ) from None
            if status in (400, 413, 422):
                raise LLMError(
                    "provider_request_rejected", "Gemini rejected the model request",
                    detail=detail,
                ) from None
            if type(error).__name__ in {"APITimeoutError", "APIConnectionError"}:
                raise LLMError(
                    "provider_connection_error", "Gemini connection failed",
                    retryable=True, detail=detail,
                ) from None
            raise LLMError(
                "provider_error", "Gemini request failed", detail=detail
            ) from None
        if interaction.status != "completed":
            raise LLMError(
                "incomplete_output", "Gemini did not complete the response",
                detail=f"Gemini interaction status: {interaction.status}",
            )
        usage = interaction.usage
        counts = {}
        for source, target in (
            ("total_input_tokens", "input_tokens"),
            ("total_output_tokens", "output_tokens"),
            ("total_thought_tokens", "thinking_tokens"),
            ("total_cached_tokens", "cached_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = getattr(usage, source, None)
            if value is not None:
                counts[target] = value
        # Keep provider steps only in internal checkpoint continuation; strip signatures from UI.
        citations = []
        thought_summaries = []
        function_calls = []
        continuation = list(request.continuation)
        if request.tool_declarations and not continuation:
            continuation = [
                {"type": "user_input", "content": [{"type": "text", "text": request.prompt}]}
            ]
        for step in interaction.steps or []:
            raw_step = step.model_dump() if hasattr(step, "model_dump") else None
            if raw_step is not None:
                continuation.append(raw_step)
            if step.type == "thought":
                for content in getattr(step, "summary", None) or []:
                    if getattr(content, "type", None) == "text" and getattr(content, "text", None):
                        thought_summaries.append(content.text)
                continue
            if step.type == "function_call":
                try:
                    from app.agent.state import FunctionCall

                    arguments = step.arguments or {}
                    if not isinstance(arguments, dict):
                        raise TypeError
                    function_calls.append(
                        FunctionCall(name=step.name, arguments=arguments, call_id=step.id)
                    )
                except (TypeError, ValueError):
                    raise LLMError(
                        "invalid_function_call",
                        "Gemini returned an invalid function call",
                        detail=mask_known_credentials(
                            f"Invalid function call: name={step.name!r}, "
                            f"arguments={step.arguments!r}, call_id={step.id!r}",
                            self.settings,
                        ),
                    ) from None
                continue
            if step.type != "model_output":
                continue
            for content in step.content or []:
                if content.type != "text":
                    continue
                for annotation in content.annotations or []:
                    if annotation.type == "url_citation":
                        citations.append(annotation.model_dump(exclude_none=True))
        return LLMResponse(
            text=getattr(interaction, "output_text", "") or "",
            provider="gemini",
            model=self.model,
            interaction_id=interaction.id,
            usage=counts,
            citations=citations,
            thinking_level=self.thinking_level,
            thought_summaries=thought_summaries,
            function_calls=function_calls,
            continuation=continuation,
        )

    async def aclose(self) -> None:
        try:
            await self.client.aio.aclose()
        finally:
            self.client.close()


class GeminiEmbeddingProvider:
    """AI Studio embeddings with matching query/document retrieval formatting."""

    def __init__(self, settings: Settings) -> None:
        if settings.embedding_model != "gemini-embedding-2":
            raise ValueError("This embedding adapter requires gemini-embedding-2")
        if not settings.embedding_version or not settings.embedding_dimension:
            raise ValueError("Embedding version and dimension are required")
        key = settings.gemini_api_key
        if not key or not key.get_secret_value().strip():
            raise ValueError("Gemini embedding credentials are not configured")
        self.model = settings.embedding_model
        self.version = settings.embedding_version
        self.dimension = settings.embedding_dimension
        self.settings = settings
        self.client = genai.Client(
            api_key=key.get_secret_value().strip(),
            vertexai=False,
            http_options=types.HttpOptions(
                timeout=int(settings.llm_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    async def embed(
        self, text: str, *, task: str = "query", title: str | None = None
    ) -> list[float]:
        return await retry_llm_call(
            lambda: self._embed_once(text, task=task, title=title),
            max_attempts=self.settings.llm_max_attempts,
            timeout_seconds=self.settings.llm_timeout_seconds,
            initial_delay=self.settings.llm_retry_initial_seconds,
            max_delay=self.settings.llm_retry_max_seconds,
        )

    async def _embed_once(self, text: str, *, task: str, title: str | None) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding text must not be empty")
        if task == "query":
            content = f"task: search result | query: {text}"
        elif task == "document":
            content = f"title: {title or 'none'} | text: {text}"
        else:
            raise ValueError("Unsupported embedding task")
        try:
            result = await self.client.aio.models.embed_content(
                model=self.model,
                contents=content,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 -- discard unsafe provider messages
            status = getattr(error, "code", None) or getattr(error, "status_code", None)
            raise LLMError(
                "embedding_provider_error",
                "Gemini embedding request failed",
                retryable=status in (429, 500, 502, 503, 504),
            ) from None
        embeddings = result.embeddings or []
        vector = embeddings[0].values if len(embeddings) == 1 else None
        if (
            vector is None
            or len(vector) != self.dimension
            or not all(math.isfinite(value) for value in vector)
            or not any(vector)
        ):
            raise LLMError("invalid_embedding", "Embedding response failed vector validation")
        return list(vector)

    async def aclose(self) -> None:
        try:
            await self.client.aio.aclose()
        finally:
            self.client.close()
