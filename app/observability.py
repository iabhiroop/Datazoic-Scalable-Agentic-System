"""Optional Langfuse tracing with isolated context, redaction and non-fatal failures."""

import asyncio
import json
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, SecretStr

from app.settings import Settings

logger = logging.getLogger(__name__)
REDACTED = "[REDACTED]"


@dataclass(frozen=True)
class TraceContext:
    request_id: str | None = None
    conversation_id: str | None = None
    run_id: str | None = None
    operation_id: str | None = None
    traceparent: str | None = None


_context: ContextVar[TraceContext | None] = ContextVar("agent_trace_context", default=None)
_current: ContextVar[Any] = ContextVar("agent_observation", default=None)


class SecretRedactor:
    """Export-only redaction. Never modify the actual LLM request or recovery evidence."""

    sensitive_keys = frozenset(
        {
            "password",
            "passwd",
            "secret",
            "secretkey",
            "apikey",
            "authorization",
            "cookie",
            "setcookie",
            "token",
            "bearertoken",
            "accesstoken",
            "refreshtoken",
            "idtoken",
            "credential",
            "credentials",
            "clientsecret",
            "privatekey",
            "authbindingsjson",
            "tokensha256",
            "email",
            "useremail",
            "phone",
            "ssn",
            "cardnumber",
        }
    )

    def __init__(self, settings: Settings, *, truncate: bool = True) -> None:
        self.limit = settings.langfuse_max_content_chars if truncate else None
        values = []
        for name in type(settings).model_fields:
            value = getattr(settings, name)
            if isinstance(value, SecretStr) and value.get_secret_value():
                values.append(value.get_secret_value())
        if settings.langfuse_public_key:
            values.append(settings.langfuse_public_key)
        self.secrets = sorted(set(values), key=len, reverse=True)

    @classmethod
    def sensitive(cls, key: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        return normalized in cls.sensitive_keys or normalized.endswith(
            ("apikey", "secretkey", "password", "accesstoken", "refreshtoken")
        )

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SecretStr):
            return REDACTED
        if isinstance(data, BaseModel):
            return self(data.model_dump())
        if isinstance(data, dict):
            return {
                str(key): REDACTED if self.sensitive(str(key)) else self(value)
                for key, value in data.items()
            }
        if isinstance(data, (list, tuple)):
            return [self(value) for value in data]
        if isinstance(data, str):
            # LLM prompts are commonly JSON-serialized context: mask its fields structurally.
            if data.lstrip().startswith(("{", "[")):
                try:
                    masked = json.dumps(self(json.loads(data)), ensure_ascii=False)
                    return self._truncate(masked)
                except (ValueError, TypeError):
                    pass
            for secret in self.secrets:
                data = data.replace(secret, REDACTED)
            data = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer " + REDACTED, data)
            data = re.sub(r"\b(?:sk-|pk-lf-|AIza)[A-Za-z0-9_-]{10,}", REDACTED, data)
            data = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", REDACTED, data)
            data = re.sub(r"(://)[^/\s:@]+:[^/\s@]+@", r"\1" + REDACTED + "@", data)
            data = re.sub(
                r"(?i)\b(api[_-]?key|secret[_-]?key|password|access[_-]?token|client[_-]?secret)"
                r"(\s*[:=]\s*)[^\s,;]+",
                lambda match: match[1] + match[2] + REDACTED,
                data,
            )
            data = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", REDACTED, data)
            return self._truncate(data)
        if data is None or isinstance(data, (bool, int, float)):
            return data
        return "[UNSUPPORTED VALUE]"

    def _truncate(self, text: str) -> str:
        return (
            text
            if self.limit is None or len(text) <= self.limit
            else text[: self.limit] + " [TRUNCATED]"
        )

    def sdk_mask(self, *, data: Any, **kwargs: Any) -> Any:
        return self(data)

    def mask_otel_spans(self, *, params):
        from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

        patches = {}
        for identifier, span in params.spans.items():
            values = {}
            deletes = []
            for key, value in span.attributes.items():
                if self.sensitive(key) or key in {
                    "http.request.header.authorization",
                    "http.request.header.cookie",
                    "exception.message",
                    "exception.stacktrace",
                }:
                    deletes.append(key)
                    continue
                masked = self(value)
                if masked != value:
                    values[key] = tuple(masked) if isinstance(masked, list) else masked
            if values or deletes:
                patches[identifier] = OtelSpanPatch(
                    set_attributes=values, delete_attributes=tuple(deletes)
                )
        return MaskOtelSpansResult(span_patches=patches)


class Observation:
    def __init__(self, telemetry, observation=None, metadata=None) -> None:
        self.telemetry = telemetry
        self.observation = observation
        self.metadata = metadata or {}

    @property
    def traceparent(self) -> str | None:
        trace_id = getattr(self.observation, "trace_id", "")
        span_id = getattr(self.observation, "id", "")
        if (
            isinstance(trace_id, str)
            and re.fullmatch(r"[0-9a-f]{32}", trace_id)
            and trace_id != "0" * 32
            and isinstance(span_id, str)
            and re.fullmatch(r"[0-9a-f]{16}", span_id)
            and span_id != "0" * 16
        ):
            return f"00-{trace_id}-{span_id}-01"
        return None

    def update(self, *, output=None, metadata=None, usage_details=None, model=None) -> None:
        if self.observation is None:
            return
        try:
            self.metadata.update(metadata or {})
            values = {"metadata": self.telemetry.redactor(self.metadata)}
            if output is not None and self.telemetry.capture_content:
                values["output"] = self.telemetry.redactor(output)
            if usage_details:
                values["usage_details"] = usage_details
            if model:
                values["model"] = model
            self._update(**values)
        except Exception:  # noqa: BLE001 -- export formatting cannot affect application work
            self.telemetry.failed("format_update")

    def error(self, code: str, *, retryable: bool = False) -> None:
        code = (
            code if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_]{1,60}", code) else "error"
        )
        self.metadata.update({"error_code": code, "retryable": retryable})
        try:
            self._update(
                level="ERROR", status_message=code, metadata=self.telemetry.redactor(self.metadata)
            )
        except Exception:  # noqa: BLE001 -- preserve the original business exception
            self.telemetry.failed("format_error")

    def _update(self, **values) -> None:
        if self.observation is not None:
            try:
                self.observation.update(**values)
            except Exception:  # noqa: BLE001 -- isolate optional telemetry
                self.telemetry.failed("update")


class Telemetry:
    def __init__(self, settings: Settings | None = None, client=None) -> None:
        self.settings = settings or Settings(_env_file=None)
        self.redactor = SecretRedactor(self.settings)
        self.client = client
        self.capture_content = self.settings.langfuse_capture_content
        self.degraded = False

    @classmethod
    def from_settings(cls, settings: Settings):
        telemetry = cls(settings)
        if not settings.langfuse_tracing_enabled:
            return telemetry
        public = settings.langfuse_public_key
        secret = settings.langfuse_secret_key
        base_url = settings.langfuse_base_url
        if not any((public, secret and secret.get_secret_value(), base_url)):
            return telemetry
        if not all((public, secret and secret.get_secret_value(), base_url)):
            telemetry.failed("configuration")
            return telemetry
        try:
            from langfuse import Langfuse
            from opentelemetry.sdk.trace import TracerProvider

            telemetry.client = Langfuse(
                public_key=public,
                secret_key=secret.get_secret_value(),
                base_url=base_url,
                timeout=settings.langfuse_timeout_seconds,
                environment=settings.langfuse_environment,
                release=settings.graph_version,
                tracing_enabled=True,
                sample_rate=1.0,
                mask=telemetry.redactor.sdk_mask,
                mask_otel_spans=telemetry.redactor.mask_otel_spans,
                should_export_span=lambda span: (
                    span.instrumentation_scope.name == "langfuse-sdk"
                    and span.name.startswith("agent.")
                ),
                tracer_provider=TracerProvider(),
            )
        except Exception:  # noqa: BLE001 -- optional service must not break execution
            telemetry.failed("initialization")
        return telemetry

    @property
    def status(self) -> str:
        return "degraded" if self.degraded else "configured" if self.client else "disabled"

    def failed(self, operation: str) -> None:
        self.degraded = True
        logger.warning("Langfuse telemetry unavailable during %s", operation)

    @contextmanager
    def bind_context(self, context: TraceContext) -> Iterator[None]:
        combined = asdict(_context.get() or TraceContext())
        combined.update({key: value for key, value in asdict(context).items() if value is not None})
        token = _context.set(TraceContext(**combined))
        try:
            yield
        finally:
            _context.reset(token)

    @staticmethod
    def trace_context() -> dict[str, str]:
        observation = _current.get()
        values = {
            key: value
            for key, value in asdict(_context.get() or TraceContext()).items()
            if value is not None
        }
        if observation is not None and observation.traceparent:
            values["traceparent"] = observation.traceparent
        return values

    @staticmethod
    def update_current(**metadata) -> None:
        observation = _current.get()
        if observation is not None:
            observation.update(metadata=metadata)

    @contextmanager
    def observe(
        self,
        name: str,
        *,
        as_type: str = "span",
        input=None,
        metadata=None,
        model=None,
        model_parameters=None,
    ) -> Iterator[Observation]:
        context = asdict(_context.get() or TraceContext())
        parent = context.pop("traceparent")
        attributes = {
            "graph_version": self.settings.graph_version,
            "prompt_version": self.settings.prompt_version,
            "policy_version": self.settings.policy_version,
            **{key: value for key, value in context.items() if value is not None},
            **(metadata or {}),
        }
        manager = observation = None
        if self.client is not None:
            try:
                from opentelemetry import trace

                values = {
                    "name": "agent." + name,
                    "as_type": as_type,
                    "metadata": self.redactor(attributes),
                    "input": self.redactor(input) if self.capture_content else None,
                    "model": model,
                    "model_parameters": self.redactor(model_parameters),
                }
                if parent and not trace.get_current_span().get_span_context().is_valid:
                    match = re.fullmatch(r"00-([0-9a-f]{32})-([0-9a-f]{16})-(?:00|01)", parent)
                    if match and int(match[1], 16) and int(match[2], 16):
                        values["trace_context"] = {"trace_id": match[1], "parent_span_id": match[2]}
                manager = self.client.start_as_current_observation(**values)
                observation = manager.__enter__()
            except Exception:  # noqa: BLE001 -- never retry/reexecute business work
                self.failed("start_observation")
        handle = Observation(self, observation, attributes)
        token = _current.set(handle)
        try:
            yield handle
        except BaseException as error:
            code = getattr(error, "code", type(error).__name__)
            handle.error(code, retryable=getattr(error, "retryable", False))
            raise
        finally:
            _current.reset(token)
            if manager is not None and observation is not None:
                try:
                    # Record safe error codes ourselves; do not export exception messages/stacktraces.
                    manager.__exit__(None, None, None)
                except Exception:  # noqa: BLE001 -- non-fatal export boundary
                    self.failed("end_observation")

    async def auth_check(self) -> bool:
        if self.client is None:
            return False
        try:
            return await asyncio.to_thread(self.client.auth_check)
        except Exception:  # noqa: BLE001 -- credentials/response bodies must not escape
            self.failed("auth_check")
            return False

    async def flush(self) -> None:
        if self.client is not None:
            try:
                await asyncio.to_thread(self.client.flush)
            except Exception:  # noqa: BLE001 -- never affect execution evidence
                self.failed("flush")

    async def aclose(self) -> None:
        if self.client is not None:
            try:
                await asyncio.to_thread(self.client.shutdown)
            except Exception:  # noqa: BLE001 -- cleanup must remain non-fatal
                self.failed("shutdown")
            finally:
                self.client = None


async def initialize_telemetry(settings: Settings) -> Telemetry:
    return Telemetry.from_settings(settings)


async def close_telemetry(telemetry: Telemetry) -> None:
    await telemetry.aclose()
