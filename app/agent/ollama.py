"""Local Ollama chat adapter for the provider-neutral agent gateway."""

import json
from typing import Any

import httpx

from app.agent.debug import mask_known_credentials
from app.agent.gateway import LLMError, LLMRequest, LLMResponse
from app.agent.state import FunctionCall
from app.settings import Settings


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(part.get("text"))
            for part in value
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        )
    return ""


def _continuation_messages(continuation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept Ollama history and map older Gemini checkpoints to chat messages."""
    messages = []
    for step in continuation:
        if not isinstance(step, dict):
            raise LLMError("invalid_continuation", "Invalid model continuation")
        if step.get("role") in {"user", "assistant", "tool"}:
            message = {
                key: step[key]
                for key in ("role", "content", "tool_calls", "tool_name")
                if key in step
            }
            messages.append(message)
            continue
        kind = step.get("type")
        if kind == "user_input":
            messages.append({"role": "user", "content": _content_text(step.get("content"))})
        elif kind == "model_output":
            messages.append({"role": "assistant", "content": _content_text(step.get("content"))})
        elif kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": step.get("name"),
                                "arguments": step.get("arguments") or {},
                            },
                        }
                    ],
                }
            )
        elif kind == "function_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_name": step.get("name"),
                    "content": _content_text(step.get("result")),
                }
            )
        elif kind == "thought":
            continue
        else:
            raise LLMError(
                "invalid_continuation",
                "The saved model continuation cannot be used with Ollama",
                detail=f"Unsupported continuation step: {kind or 'unknown'}",
            )
    return messages


class OllamaProvider:
    provider_name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.ollama_model_name.strip()
        self.client = httpx.AsyncClient(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=settings.llm_timeout_seconds,
            trust_env=False,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.web_search:
            raise LLMError(
                "unsupported_feature", "Local Ollama does not provide web search"
            )
        history = (
            _continuation_messages(request.continuation)
            if request.continuation
            else [{"role": "user", "content": request.prompt}]
        )
        instruction = request.system_instruction.strip()
        if request.force_tool_call and request.tool_declarations:
            instruction += "\nCall exactly one supplied tool before responding."
        if request.response_schema is not None:
            instruction += "\nReturn only a JSON object matching the supplied schema."
        messages = ([{"role": "system", "content": instruction}] if instruction else []) + history
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if request.response_schema is not None:
            payload["format"] = request.response_schema
        if request.tool_declarations:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tool_declarations
            ]
        try:
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise LLMError(
                "timeout", "Ollama request timed out", retryable=True,
                detail=mask_known_credentials(
                    str(error) or "Ollama request timed out", self.settings
                ),
            ) from None
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            detail = mask_known_credentials(
                f"Ollama HTTP {status}: {error.response.text}", self.settings
            )
            if status == 429:
                code = "rate_limited"
            elif status == 503:
                code = "provider_busy"
            elif status in {400, 404, 422}:
                code = "provider_request_rejected"
            else:
                code = "provider_error"
            raise LLMError(
                code,
                "Ollama rejected the model request" if status < 500 else "Ollama failed",
                retryable=status == 429 or status >= 500,
                detail=detail,
            ) from None
        except httpx.RequestError as error:
            raise LLMError(
                "provider_connection_error", "Cannot reach local Ollama", retryable=True,
                detail=mask_known_credentials(f"{type(error).__name__}: {error}", self.settings),
            ) from None
        try:
            data = response.json()
            message = data["message"]
            if not isinstance(data, dict) or not isinstance(message, dict):
                raise ValueError("Missing Ollama message")
            if data.get("done") is not True:
                raise ValueError("Ollama response was incomplete")
            content = message.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("Ollama message content was not text")
            raw_calls = message.get("tool_calls") or []
            if not isinstance(raw_calls, list):
                raise ValueError("Ollama tool calls were not a list")
            calls = []
            for index, item in enumerate(raw_calls):
                function = item["function"]
                name = function["name"]
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(name, str) or not name or not isinstance(arguments, dict):
                    raise ValueError("Invalid Ollama function call")
                calls.append(
                    FunctionCall(
                        name=name,
                        arguments=arguments,
                        call_id=f"ollama-{len(history)}-{index}",
                    )
                )
            usage = {
                "input_tokens": int(data.get("prompt_eval_count") or 0),
                "output_tokens": int(data.get("eval_count") or 0),
            }
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
            if data.get("prompt_eval_cached_count") is not None:
                usage["cached_tokens"] = int(data["prompt_eval_cached_count"])
        except (ValueError, TypeError, KeyError) as error:
            raise LLMError(
                "invalid_output", "Ollama returned an invalid chat response",
                detail=mask_known_credentials(
                    f"{type(error).__name__}: {error}\nResponse: {response.text}",
                    self.settings,
                ),
            ) from None
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if raw_calls:
            assistant_message["tool_calls"] = raw_calls
        return LLMResponse(
            text=content,
            provider="ollama",
            model=data.get("model") or self.model,
            usage=usage,
            function_calls=calls,
            continuation=[*history, assistant_message],
        )

    async def aclose(self) -> None:
        await self.client.aclose()
