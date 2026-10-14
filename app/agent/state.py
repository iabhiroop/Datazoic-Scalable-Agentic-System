"""Model contracts and small checkpoint state; no trusted identity or secrets."""

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Subgoal(Contract):
    id: str
    action: str
    entity: str


class Intent(Contract):
    goal: str
    subgoals: list[Subgoal]
    category_ids: list[str]
    entities: dict[str, Any]
    constraints: dict[str, Any]
    missing_fields: list[str]
    confidence: float = Field(ge=0, le=1)
    execution_mode: Literal["plan_only", "execute"] = "plan_only"


class GoalPlan(Contract):
    user_goal: str = Field(min_length=1)
    tool_actions: list[str] = Field(min_length=1)
    missing_information: list[str] = Field(default_factory=list)


class ToolDeclaration(Contract):
    name: str
    description: str
    parameters: dict[str, Any]


class FunctionCall(Contract):
    name: str
    arguments: dict[str, Any]
    call_id: str


class FunctionTurn(Contract):
    text: str = ""
    calls: list[FunctionCall] = Field(default_factory=list)
    continuation: list[dict[str, Any]] = Field(default_factory=list)


class LiteralBinding(Contract):
    literal: Any


class OutputBinding(Contract):
    from_step: str
    path: str  # JSON Pointer; interpreted by Python, never eval/template execution.


# The mutually exclusive required fields discriminate the LLD's binding shapes.
Binding = Annotated[LiteralBinding | OutputBinding, Field(union_mode="left_to_right")]


class PlanStep(Contract):
    step_id: str
    tool_id: str
    schema_version: str
    depends_on: list[str] = Field(default_factory=list)
    arguments: dict[str, Binding]
    action_ref: str | None = None  # Python must assign/validate; never trust model ownership.


class Plan(Contract):
    plan_version: int = Field(ge=1)
    steps: list[PlanStep]


class WaitRequest(Contract):
    wait_token: str
    kind: Literal["clarification", "approval", "retry", "reconciliation"]
    reason: str
    resume_route: str
    expected_payload_schema: dict[str, Any]
    due_at: str | None = None
    operation_id: str | None = None
    input_hash: str | None = None


class ResumeEnvelope(Contract):
    input_id: str
    wait_token: str
    kind: Literal["clarification", "approval", "timer"]
    payload: dict[str, Any]


class AgentState(TypedDict, total=False):
    run_id: str
    thread_id: str
    conversation_id: str
    request_text: str
    intent: dict[str, Any]
    candidate_tools: list[dict[str, Any]]
    knowledge_refs: list[str]
    plan: dict[str, Any]
    step_index: int
    results: dict[str, dict[str, Any]]
    action_map: dict[str, str]
    observation: dict[str, Any]
    wait_request: dict[str, Any]
    last_input_id: str
    retry_counts: dict[str, int]
    replan_count: int
    routing_decision: str
    final_result: dict[str, Any]
    llm_activity: list[dict[str, Any]]
    clarifications: list[dict[str, Any]]
    business_status: str
    approved_operation_id: str
    subgoal_index: int
    tool_call_count: int
    subgoal_summaries: dict[str, str]
    subgoal_tools: dict[str, list[dict[str, Any]]]
    provider_continuation: list[dict[str, Any]]
    selection_retries: int
    pending_function_call: dict[str, Any]
    goal_plan: dict[str, Any]
    goal_action_index: int
    goal_action_results: dict[str, dict[str, Any]]
