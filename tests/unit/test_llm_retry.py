"""Temporary model failures wait and retry; fatal errors and cancellation do not."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.agent.gateway import LLMError, retry_llm_call


async def call(operation, **options):
    return await retry_llm_call(
        operation,
        max_attempts=options.get("attempts", 3),
        timeout_seconds=1,
        initial_delay=10,
        max_delay=15,
    )


async def test_transient_failures_wait_then_succeed(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("app.agent.gateway.asyncio.sleep", sleep)
    operation = AsyncMock(
        side_effect=[
            LLMError("provider_error", "safe", retryable=True),
            LLMError("provider_error", "safe", retryable=True),
            "ok",
        ]
    )
    assert await call(operation) == "ok"
    assert operation.await_count == 3
    assert [entry.args[0] for entry in sleep.await_args_list] == [10, 15]


async def test_retry_budget_exhausted_and_fatal_error_not_retried(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("app.agent.gateway.asyncio.sleep", sleep)
    operation = AsyncMock(side_effect=LLMError("provider_error", "safe", retryable=True))
    with pytest.raises(LLMError):
        await call(operation, attempts=2)
    assert operation.await_count == 2 and sleep.await_count == 1
    operation = AsyncMock(side_effect=LLMError("authentication", "safe"))
    sleep.reset_mock()
    with pytest.raises(LLMError):
        await call(operation)
    operation.assert_awaited_once()
    sleep.assert_not_awaited()


async def test_cancellation_during_wait_propagates(monkeypatch):
    monkeypatch.setattr(
        "app.agent.gateway.asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError)
    )
    operation = AsyncMock(side_effect=LLMError("provider_error", "safe", retryable=True))
    with pytest.raises(asyncio.CancelledError):
        await call(operation)
    operation.assert_awaited_once()
