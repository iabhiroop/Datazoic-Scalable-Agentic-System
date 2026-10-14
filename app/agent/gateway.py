"""Provider-neutral LLM gateway. Model output has no execution authority."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from app.agent.state import FunctionCall, ToolDeclaration
from app.agent.debug import mask_known_credentials, record_debug_event
from app.observability import Telemetry
from app.settings import Settings

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
_activity_capture: ContextVar[
    tuple[list[dict[str, Any]], list[dict[str, Any]], Callable[[Any], Any]] | None
] = ContextVar("llm_activity_capture", default=None)


@contextmanager
def capture_llm_activity(redactor: Callable[[Any], Any]):
    """Capture redacted model responses for the current graph-node context only."""
    records: list[dict[str, Any]] = []
    token = _activity_capture.set((records, [], redactor))
    try:
        yield records
    finally:
        _activity_capture.reset(token)


def current_llm_trace() -> list[dict[str, Any]]:
    capture = _activity_capture.get()
    return capture[1] if capture is not None else []


def _trace_safe(value: Any) -> Any:
    """Remove opaque provider signatures while retaining inspectable turn history."""
    if isinstance(value, dict):
        return {
            key: _trace_safe(item)
            for key, item in value.items()
            if key.lower() not in {"signature", "thought_signature", "thoughtsignature"}
        }
    if isinstance(value, list):
        return [_trace_safe(item) for item in value]
    return value


def _trace_request(
    request: "LLMRequest",
    thinking_level: str | None = None,
    timeout_seconds: float | None = None,
    provider_name: str | None = None,
) -> dict[str, Any]:
    provider_input: Any = request.prompt
    if request.tool_declarations:
        provider_input = request.continuation or [
            {"type": "user_input", "content": [{"type": "text", "text": request.prompt}]}
        ]
    return _trace_safe(
        {
            "application_prompt": request.prompt,
            "provider_input": provider_input,
            "system_instruction": request.system_instruction,
            "response_schema": request.response_schema,
            "tools": [item.model_dump(mode="json") for item in request.tool_declarations],
            "web_search": request.web_search,
            "force_tool_call": request.force_tool_call,
            "generation_config": {
                "thinking_level": thinking_level,
                **({"thinking_summaries": "auto"} if provider_name == "gemini" else {}),
                **(
                    {"tool_choice": "any"}
                    if provider_name == "gemini" and request.force_tool_call
                    else {}
                ),
            },
            "provider_options": (
                {"store": False, "stream": False, "timeout_seconds": timeout_seconds}
                if provider_name == "gemini"
                else {"endpoint": "/api/chat", "stream": False}
                if provider_name == "ollama"
                else {}
            ),
        }
    )


def append_llm_trace(path: Path, records: list[dict[str, Any]]) -> None:
    """Append already-redacted call and tool-result records as durable JSONL."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")


class LLMError(RuntimeError):
    """Safe summary plus an owner-visible, secret-redacted provider detail."""

    def __init__(
        self, code: str, message: str, *, retryable: bool = False, detail: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.detail = detail


async def retry_llm_call(
    operation: Callable[[], Awaitable[R]],
    *,
    max_attempts: int,
    timeout_seconds: float,
    initial_delay: float,
    max_delay: float,
    telemetry: Telemetry | None = None,
) -> R:
    """Bounded, cancellable retry for stateless model calls; never business dispatch."""
    if max_attempts < 1 or min(timeout_seconds, initial_delay, max_delay) <= 0:
        raise ValueError("Invalid model retry configuration")
    for attempt in range(1, max_attempts + 1):
        try:
            async with asyncio.timeout(timeout_seconds):
                return await operation()
        except TimeoutError:
            error = LLMError("timeout", "LLM request timed out", retryable=True)
        except LLMError as failure:
            error = failure
        if not error.retryable or attempt == max_attempts:
            raise error from None
        delay = min(initial_delay * 2 ** (attempt - 1), max_delay)
        if telemetry:
            with telemetry.observe(
                "llm.retry_wait",
                metadata={"attempt": attempt, "delay_seconds": delay, "error_code": error.code},
            ):
                await asyncio.sleep(delay)
        else:
            await asyncio.sleep(delay)
    raise RuntimeError("Unreachable retry state")


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system_instruction: str = ""
    response_schema: dict[str, Any] | None = None
    web_search: bool = False
    tool_declarations: list[ToolDeclaration] = field(default_factory=list)
    continuation: list[dict[str, Any]] = field(default_factory=list)
    force_tool_call: bool = False


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    interaction_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    thinking_level: str | None = None
    thought_summaries: list[str] = field(default_factory=list)
    function_calls: list[FunctionCall] = field(default_factory=list)
    continuation: list[dict[str, Any]] = field(default_factory=list)


class LLMProvider(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    async def aclose(self) -> None: ...


class LLMGateway:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        timeout_seconds: float = 60,
        telemetry: Telemetry | None = None,
        owns_telemetry: bool = False,
        max_attempts: int = 1,
        retry_initial_seconds: float = 10,
        retry_max_seconds: float = 30,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.telemetry = telemetry or Telemetry()
        self.owns_telemetry = owns_telemetry
        self.max_attempts = max_attempts
        self.retry_initial_seconds = retry_initial_seconds
        self.retry_max_seconds = retry_max_seconds

    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str = "",
        web_search: bool = False,
        response_schema: dict[str, Any] | None = None,
        observation_name: str = "llm.generate",
        tool_declarations: list[ToolDeclaration] | None = None,
        continuation: list[dict[str, Any]] | None = None,
        force_tool_call: bool = False,
    ) -> LLMResponse:
        if not prompt.strip():
            raise ValueError("LLM prompt cannot be empty")
        request = LLMRequest(
            prompt,
            system_instruction,
            response_schema,
            web_search,
            tool_declarations or [],
            continuation or [],
            force_tool_call,
        )
        return await retry_llm_call(
            lambda: self._generate_once(request, observation_name),
            max_attempts=self.max_attempts,
            # Let the inner provider timeout persist its failed-attempt event first.
            timeout_seconds=self.timeout_seconds + 5,
            initial_delay=self.retry_initial_seconds,
            max_delay=self.retry_max_seconds,
            telemetry=self.telemetry,
        )

    async def _generate_once(self, request: LLMRequest, observation_name: str) -> LLMResponse:
        prompt, system_instruction = request.prompt, request.system_instruction
        web_search, response_schema = request.web_search, request.response_schema
        attempt_id = str(uuid4())
        await record_debug_event(
            "debug_llm_started",
            {
                "attempt_id": attempt_id,
                "purpose": observation_name,
                "provider": getattr(self.provider, "provider_name", type(self.provider).__name__),
                "model": getattr(self.provider, "model", None),
                "request": _trace_request(
                    request,
                    getattr(self.provider, "thinking_level", None),
                    self.timeout_seconds,
                    getattr(self.provider, "provider_name", None),
                ),
            },
        )
        with self.telemetry.observe(
            observation_name,
            as_type="generation",
            input={"prompt": prompt, "system_instruction": system_instruction},
            model=getattr(self.provider, "model", None),
            metadata={"web_search": web_search, "structured_output": response_schema is not None},
            model_parameters={
                "thinking_level": getattr(self.provider, "thinking_level", None),
            },
        ) as observation:
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    response = await self.provider.generate(request)
            except TimeoutError:
                self._capture_failed_trace(request, observation_name, "timeout")
                await record_debug_event(
                    "debug_llm_failed",
                    {
                        "attempt_id": attempt_id,
                        "purpose": observation_name,
                        "error_code": "timeout",
                        "error_detail": "LLM request timed out",
                    },
                )
                raise LLMError(
                    "timeout", "LLM request timed out", retryable=True,
                    detail="LLM request timed out",
                ) from None
            except LLMError as error:
                self._capture_failed_trace(request, observation_name, error.code, error.detail)
                await record_debug_event(
                    "debug_llm_failed",
                    {
                        "attempt_id": attempt_id,
                        "purpose": observation_name,
                        "error_code": error.code,
                        "error_detail": error.detail or str(error),
                    },
                )
                raise
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 -- show redacted detail to run owner
                detail = mask_known_credentials(
                    f"{type(error).__name__}: {error}", Settings(_env_file=None)
                )
                self._capture_failed_trace(request, observation_name, "provider_error", detail)
                await record_debug_event(
                    "debug_llm_failed",
                    {
                        "attempt_id": attempt_id,
                        "purpose": observation_name,
                        "error_code": "provider_error",
                        "error_detail": detail,
                    },
                )
                raise LLMError(
                    "provider_error", "LLM provider request failed", detail=detail
                ) from None
            await record_debug_event(
                "debug_llm_finished",
                {
                    "attempt_id": attempt_id,
                    "purpose": observation_name,
                    "provider": response.provider,
                    "model": response.model,
                    "interaction_id": response.interaction_id,
                    "response": response.text,
                    "function_calls": [call.model_dump(mode="json") for call in response.function_calls],
                    "usage": response.usage,
                    "thinking_level": response.thinking_level or getattr(self.provider, "thinking_level", None),
                    "thought_summaries": response.thought_summaries,
                    "citations": response.citations,
                },
            )
            if not response.text.strip() and not response.function_calls:
                self._capture_failed_trace(request, observation_name, "empty_output")
                await record_debug_event(
                    "debug_llm_failed",
                    {
                        "attempt_id": attempt_id,
                        "purpose": observation_name,
                        "error_code": "empty_output",
                        "error_detail": "LLM returned no text",
                    },
                )
                raise LLMError(
                    "empty_output", "LLM returned no text", detail="LLM returned no text"
                )
            usage = {
                target: response.usage[source]
                for source, target in (
                    ("input_tokens", "input"),
                    ("output_tokens", "output"),
                    ("total_tokens", "total"),
                    ("thinking_tokens", "output_thought_tokens"),
                    ("cached_tokens", "input_cached_tokens"),
                )
                if source in response.usage
            }
            observation.update(
                output=response.text,
                model=response.model,
                usage_details=usage,
                metadata={"provider": response.provider, "interaction_id": response.interaction_id},
            )
            capture = _activity_capture.get()
            if capture is not None:
                records, traces, redact = capture
                safe_calls = [call.model_dump(mode="json") for call in response.function_calls]
                records.append(
                    redact(
                        {
                            "purpose": observation_name,
                            "provider": response.provider,
                            "model": response.model,
                            "interaction_id": response.interaction_id,
                            "completed_at": datetime.now(UTC).isoformat(),
                            "response": response.text,
                            "usage": response.usage,
                            "thinking_level": response.thinking_level
                            or getattr(self.provider, "thinking_level", None),
                            "thought_summaries": response.thought_summaries,
                            "citations": response.citations,
                            "function_calls": safe_calls,
                        }
                    )
                )
                traces.append(
                    redact(
                        {
                            "purpose": observation_name,
                            "provider": response.provider,
                            "model": response.model,
                            "request": _trace_request(
                                request,
                                getattr(self.provider, "thinking_level", None),
                                self.timeout_seconds,
                                getattr(self.provider, "provider_name", None),
                            ),
                            "response": response.text,
                            "function_calls": safe_calls,
                            "provider_steps": _trace_safe(
                                response.continuation[
                                    len(request.continuation)
                                    + (
                                        1
                                        if request.tool_declarations and not request.continuation
                                        else 0
                                    ) :
                                ]
                            ),
                            "thought_summaries": response.thought_summaries,
                            "citations": response.citations,
                            "usage": response.usage,
                            "interaction_id": response.interaction_id,
                            "completed_at": datetime.now(UTC).isoformat(),
                        }
                    )
                )
            return response

    def _capture_failed_trace(
        self, request: LLMRequest, purpose: str, error_code: str, detail: str | None = None
    ) -> None:
        capture = _activity_capture.get()
        if capture is None:
            return
        _, traces, redact = capture
        traces.append(
            redact(
                _trace_safe(
                    {
                        "purpose": purpose,
                        "provider": getattr(
                            self.provider, "provider_name", type(self.provider).__name__
                        ),
                        "model": getattr(self.provider, "model", None),
                        "request": _trace_request(
                            request,
                            getattr(self.provider, "thinking_level", None),
                            self.timeout_seconds,
                            getattr(self.provider, "provider_name", None),
                        ),
                        "error_code": error_code,
                        "error_detail": detail,
                        "completed_at": datetime.now(UTC).isoformat(),
                    }
                )
            )
        )

    async def generate_structured(
        self,
        prompt: str,
        response_type: type[T],
        *,
        system_instruction: str = "",
        observation_name: str = "llm.structured",
    ) -> T:
        with self.telemetry.observe(
            observation_name, metadata={"response_type": response_type.__name__}
        ) as observation:
            response = await self.generate(
                prompt,
                system_instruction=system_instruction,
                response_schema=response_type.model_json_schema(),
                observation_name=observation_name,
            )
            try:
                result = response_type.model_validate_json(response.text)
            except ValidationError as error:
                detail = mask_known_credentials(
                    f"{type(error).__name__}: {error}\nModel response:\n{response.text}",
                    Settings(_env_file=None),
                )
                await record_debug_event(
                    "debug_llm_output_rejected",
                    {
                        "purpose": observation_name,
                        "error_code": "invalid_output",
                        "error_detail": detail,
                        "response": response.text,
                    },
                )
                raise LLMError(
                    "invalid_output", "LLM response does not match the expected contract",
                    detail=detail,
                ) from None
            observation.update(metadata={"validation": "passed"})
            return result

    async def aclose(self) -> None:
        try:
            await self.provider.aclose()
        finally:
            if self.owns_telemetry:
                await self.telemetry.aclose()


def _gemini(settings: Settings) -> LLMProvider:
    # Provider dependency stays behind this factory; application callers import no Google SDK types.
    from app.agent.gemini import GeminiProvider

    return GeminiProvider(settings)


def _ollama(settings: Settings) -> LLMProvider:
    from app.agent.ollama import OllamaProvider

    return OllamaProvider(settings)


PROVIDER_FACTORIES: dict[str, Callable[[Settings], LLMProvider]] = {
    "gemini": _gemini,
    "ollama": _ollama,
}


def create_llm_gateway(settings: Settings, *, telemetry: Telemetry | None = None) -> LLMGateway:
    provider_name = settings.llm_provider.strip().lower()
    factory = PROVIDER_FACTORIES.get(provider_name)
    if factory is None:
        raise ValueError(f"LLM provider is not registered: {provider_name}")
    provider = factory(settings)
    owns_telemetry = telemetry is None
    telemetry = telemetry or Telemetry.from_settings(settings)
    return LLMGateway(
        provider,
        timeout_seconds=settings.llm_timeout_seconds,
        telemetry=telemetry,
        owns_telemetry=owns_telemetry,
        max_attempts=settings.llm_max_attempts,
        retry_initial_seconds=settings.llm_retry_initial_seconds,
        retry_max_seconds=settings.llm_retry_max_seconds,
    )
