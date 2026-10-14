"""Tool outcomes and safe error codes participate in shared telemetry."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.tools.contracts import ToolContext, ToolOutcome
from app.tools.system_search import SystemSearch


async def test_tool_failure_is_traced_without_reexecution():
    telemetry = MagicMock()
    tool = SystemSearch(None, None, allowed_adapters=set(), telemetry=telemetry)
    outcome = ToolOutcome(state="failed", data={}, error_code="forbidden")
    tool._invoke = AsyncMock(return_value=outcome)
    context = ToolContext(uuid4(), "reader", uuid4(), uuid4(), None)
    assert await tool.invoke({"mode": "request_status"}, context) is outcome
    tool._invoke.assert_awaited_once()
    observation = telemetry.observe.return_value.__enter__.return_value
    observation.error.assert_called_once_with("forbidden")
    observation.update.assert_called_once_with(output={}, metadata={"state": "failed"})
