"""Evidence-driven answers, permission refresh and complete selected context."""

from unittest.mock import AsyncMock
from uuid import uuid4

from app.agent.llm import KnowledgeAnswer, ModelClient
from app.knowledge.context import ContextBuilder
from app.security import Principal
from app.settings import Settings
from app.tools.contracts import ToolContext
from app.tools.rag_pipeline import Passage, RagPipeline


def passage():
    return Passage(
        chunk_id="source-1",
        document_id="doc-1",
        document_version="1",
        title="Sample",
        locator="lines:1-2",
        text="An example fact.",
        score=0.03,
    )


def pipeline(answer=None):
    retrieval = AsyncMock()
    retrieval.knowledge.return_value = [passage()]
    retrieval.knowledge_revision.return_value = 1
    retrieval.validate_evidence.return_value = True
    auth = AsyncMock()
    context = ToolContext(uuid4(), "reader", uuid4(), uuid4(), None)
    auth.refresh.return_value = Principal(
        context.tenant_id, context.user_id, frozenset({"knowledge:read"})
    )
    gateway = AsyncMock()
    gateway.generate_structured.return_value = answer or KnowledgeAnswer(
        sufficient_evidence=True, answer="An example fact.", source_ids=["source-1"]
    )
    tool = RagPipeline(
        retrieval, ContextBuilder(Settings(_env_file=None)), auth, ModelClient(gateway)
    )
    return tool, context, retrieval, auth, gateway


async def test_answer_uses_actual_source_validation():
    tool, context, _, _, gateway = pipeline()
    result = await tool.invoke({"query": "What is the fact?", "mode": "answer"}, context)
    assert result.state == "succeeded" and result.data["source_ids"] == ["source-1"]
    gateway.generate_structured.assert_awaited_once()
    tool, context, _, _, _ = pipeline(
        KnowledgeAnswer(sufficient_evidence=True, answer="Invented", source_ids=["invented"])
    )
    result = await tool.invoke({"query": "Question", "mode": "answer"}, context)
    assert result.error_code == "invalid_evidence" and result.data == {}


async def test_empty_evidence_skips_model_and_unsupported_answer_is_null():
    tool, context, retrieval, _, gateway = pipeline()
    retrieval.knowledge.return_value = []
    result = await tool.invoke({"query": "Question", "mode": "answer"}, context)
    assert not result.data["sufficient_evidence"] and result.data["answer"] is None
    gateway.generate_structured.assert_not_awaited()
    tool, context, _, _, _ = pipeline(
        KnowledgeAnswer(sufficient_evidence=False, answer="Unsupported", source_ids=[])
    )
    result = await tool.invoke({"query": "Question", "mode": "answer"}, context)
    assert not result.data["sufficient_evidence"] and result.data["answer"] is None


async def test_revision_change_and_revoked_scope_do_not_return_evidence():
    tool, context, retrieval, _, gateway = pipeline()
    retrieval.knowledge_revision.side_effect = [1, 2]
    assert (
        await tool.invoke({"query": "Question", "mode": "answer"}, context)
    ).error_code == "stale_evidence"
    gateway.generate_structured.assert_not_awaited()
    tool, context, _, auth, _ = pipeline()
    allowed = auth.refresh.return_value
    auth.refresh.side_effect = [
        allowed,
        allowed,
        Principal(context.tenant_id, "reader", frozenset()),
    ]
    result = await tool.invoke({"query": "Question", "mode": "answer"}, context)
    assert result.error_code == "forbidden" and result.data == {}


async def test_context_keeps_selected_evidence_without_a_token_cap():
    builder = ContextBuilder(Settings(_env_file=None))
    long = passage().model_copy(update={"text": "x" * 2000})
    result = await builder.build([], [long], [{"summary": "x" * 2000}])
    assert result["passages"][0]["text"] == long.text
    assert result["trajectories"] == [{"summary": "x" * 2000}]


async def test_selected_tool_schemas_are_preserved_whole():
    import json
    from pathlib import Path

    from app.tools.contracts import ToolCard

    path = Path(__file__).resolve().parents[2] / "resources/tool_cards/system_search.json"
    card = ToolCard.model_validate(json.loads(path.read_text()))
    builder = ContextBuilder(Settings(_env_file=None, langfuse_max_content_chars=100000))
    result = await builder.build([card], [], [{"summary": "x" * 50000}])
    assert result["trajectories"] == [{"summary": "x" * 50000}]
    assert result["tool_cards"][0]["input_schema"] == card.input_schema
    assert result["tool_cards"][0]["output_schema"] == card.output_schema
