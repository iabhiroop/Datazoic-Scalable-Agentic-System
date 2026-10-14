"""Run API exposes only persisted or checkpointed LLM activity."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.api.runs import RunService
from app.db.models import RunState


@pytest.mark.asyncio
async def test_active_status_reads_only_activity_from_its_checkpoint():
    activity = [{"purpose": "classify", "response": "redacted"}]
    checkpoint = SimpleNamespace(
        checkpoint={
            "channel_values": {
                "request_text": "private request",
                "llm_activity": activity,
                "intent": {"private": "state"},
            }
        }
    )
    saver = SimpleNamespace(aget_tuple=AsyncMock(return_value=checkpoint))

    class FakeDatabase:
        async def checkpointer(self):
            return saver

    run = SimpleNamespace(status=RunState.RUNNING, thread_id="thread-id", result_payload={})
    result = await RunService(database=FakeDatabase())._checkpoint_activity(run)
    assert result == activity
    saver.aget_tuple.assert_awaited_once_with({"configurable": {"thread_id": "thread-id"}})


def test_status_projects_activity_out_of_result_payload():
    activity = [{"purpose": "respond", "response": "redacted"}]
    run = SimpleNamespace(
        run_id=uuid4(),
        conversation_id=uuid4(),
        status=RunState.COMPLETED,
        cancel_requested=False,
        wait_token=None,
        wait_kind=None,
        due_at=None,
        error_payload={"code": "worker_invocation_failed", "error_type": "RuntimeError"},
        result_payload={"summary": "done", "_llm_activity": activity},
    )
    status = RunService._status(run)
    assert status.llm_activity == activity
    assert status.result == {"summary": "done"}
    assert status.error_code == "worker_invocation_failed"
    assert status.error_type == "RuntimeError"
