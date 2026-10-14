"""Small contract checks for the standalone v2 chat path."""

from app.tools.contracts import ToolCard
from appv2.agent import ChatPlan
from appv2.tools import ToolDispatcher


def card(tool_id: str) -> ToolCard:
    return ToolCard(
        tool_id=tool_id,
        schema_version="1",
        adapter_key="lookup",
        name=tool_id,
        description="Read one record",
        action="read",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}},
        operation_type="read",
        required_scopes=["records:read"],
        precondition_key=None,
        verification_key="lookup_result",
        provider="internal",
        provider_account_key="internal",
        status="published",
    )


def test_chat_plan_allows_a_conversational_turn_without_tools():
    plan = ChatPlan(user_goal="Greet the user")

    assert plan.tool_actions == []
    assert plan.missing_information == []


def test_model_declarations_include_only_the_retrieved_shortlist():
    dispatcher = ToolDispatcher.__new__(ToolDispatcher)
    declarations, cards_by_name = dispatcher.declarations([card("authorized_lookup")])

    assert len(declarations) == 1
    assert "authorized_lookup" in declarations[0]["function"]["description"]
    assert [item.tool_id for item in cards_by_name.values()] == ["authorized_lookup"]
