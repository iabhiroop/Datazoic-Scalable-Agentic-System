from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.nodes import AgentNodes
from app.settings import Settings


@pytest.mark.asyncio
async def test_provider_failure_is_reported_without_another_model_call():
    model = SimpleNamespace(respond=AsyncMock())
    nodes = AgentNodes(
        database=None,
        authenticator=None,
        model=model,
        retrieval=None,
        context_builder=None,
        registry=None,
        validator=None,
        execution=None,
        settings=Settings(_env_file=None),
    )
    result = await nodes.respond(
        {
            "business_status": "failed",
            "observation": {"error_code": "provider_busy"},
            "results": {},
        },
        None,
    )
    assert result["final_result"]["error_code"] == "provider_busy"
    assert "temporarily unavailable" in result["final_result"]["summary"]
    model.respond.assert_not_awaited()
