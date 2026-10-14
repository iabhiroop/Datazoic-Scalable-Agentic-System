"""Versioned sequential graph builders with synchronous checkpoint durability."""

from contextlib import nullcontext
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent.gateway import LLMError, append_llm_trace, capture_llm_activity
from app.agent.debug import capture_debug_events, record_debug_event
from app.agent.state import AgentState
from app.observability import SecretRedactor
from app.settings import Settings

NODE_NAMES = (
    "classify",
    "retrieve",
    "plan",
    "validate",
    "prepare_step",
    "execute",
    "verify",
    "recover",
    "prepare_wait",
    "wait",
    "reconcile",
    "respond",
)
V2_NODE_NAMES = (
    "classify",
    "retrieve_v2",
    "plan_v2",
    "validate_v2",
    "prepare_step_v2",
    "execute",
    "verify_v2",
    "prepare_wait",
    "wait",
    "respond",
)
V3_NODE_NAMES = (
    "classify_v3",
    "retrieve_v3",
    "select_v3",
    "validate_v3",
    "prepare_step_v3",
    "execute",
    "verify_v3",
    "prepare_wait_v3",
    "wait_v3",
    "respond_v3",
)


def build_graph(checkpointer: Any, services: Any, *, version: str = "v1") -> Any:
    """Wire allowlisted conditional routes; pin graph version and recursion/retry limits."""
    builder = StateGraph(AgentState)
    node_names = V3_NODE_NAMES if version == "v3" else V2_NODE_NAMES if version == "v2" else NODE_NAMES
    for name in node_names:

        def wrap(node_name):
            async def node(state: AgentState, config: RunnableConfig):
                principal = await services.principal(state, config)
                if not node_name.startswith("respond") and await services.cancelled(state):
                    return {"business_status": "cancelled", "routing_decision": "respond_v3" if version == "v3" else "respond"}
                telemetry = getattr(services, "telemetry", None)
                context = (
                    telemetry.observe("graph." + node_name, metadata={"run_id": state["run_id"]})
                    if telemetry and node_name != "wait"
                    else nullcontext()
                )
                redactor = SecretRedactor(getattr(services, "settings", Settings(_env_file=None)))

                async def write_debug(event_type, evidence):
                    writer = getattr(services, "record_debug_event", None)
                    if writer is not None:
                        await writer(state, principal, event_type, evidence)

                with (
                    capture_debug_events(write_debug),
                    capture_llm_activity(redactor) as activity,
                    context as observation,
                ):
                    try:
                        result = await getattr(services, node_name)(state, principal)
                    except LLMError as error:
                        result = {
                            "business_status": "failed",
                            "observation": {"error_code": error.code, "error_detail": error.detail},
                            "routing_decision": "respond",
                        }
                    if result.get("final_result"):
                        await record_debug_event("debug_final_response", result["final_result"])
                    node_activity = result.get("llm_activity", [])
                    if activity or node_activity:
                        result["llm_activity"] = [
                            *state.get("llm_activity", []),
                            *node_activity,
                            *activity,
                        ]
                    from app.agent.gateway import current_llm_trace

                    traces = list(current_llm_trace())
                    if node_name == "execute" and state.get("plan"):
                        step = state["plan"].get("steps", [{}])[state.get("step_index", 0)]
                        traces.append(
                            redactor(
                                {
                                    "event": "tool_result",
                                    "tool_id": step.get("tool_id"),
                                    "schema_version": step.get("schema_version"),
                                    "call_id": state.get("pending_function_call", {}).get(
                                        "call_id"
                                    ),
                                    "result": result.get("observation", {}),
                                }
                            )
                        )
                    if traces:
                        trace_path = services.settings.llm_trace_dir / f"{state['run_id']}.jsonl"
                        append_llm_trace(trace_path, traces)
                    await services.principal({**state, **result}, config)
                    if observation:
                        observation.update(
                            metadata={"route": result.get("routing_decision", "end")}
                        )
                return result

            return node

        builder.add_node(name, wrap(name))
    builder.add_edge(START, "classify_v3" if version == "v3" else "classify")
    routes = (
        {
            "classify_v3": {"retrieve": "retrieve_v3", "prepare_wait": "prepare_wait_v3"},
            "retrieve_v3": {"select": "select_v3", "respond": "respond_v3"},
            "select_v3": {"validate": "validate_v3", "respond": "respond_v3"},
            "validate_v3": {"prepare_step": "prepare_step_v3", "respond": "respond_v3"},
            "prepare_step_v3": {"execute": "execute", "wait": "wait_v3", "respond": "respond_v3"},
            "execute": {"verify": "verify_v3", "recover": "respond_v3"},
            "verify_v3": {"retrieve": "retrieve_v3", "respond": "respond_v3"},
            "prepare_wait_v3": {"wait": "wait_v3"},
            "wait_v3": {"classify": "classify_v3", "execute": "execute", "respond": "respond_v3"},
        }
        if version == "v3"
        else (
        {
            "classify": {"retrieve": "retrieve_v2", "prepare_wait": "prepare_wait"},
            "retrieve_v2": {"plan": "plan_v2"},
            "plan_v2": {
                "prepare_step": "validate_v2",
                "respond": "respond",
                "plan": "plan_v2",
                "plan_only_done": END,
            },
            "validate_v2": {"prepare_step": "prepare_step_v2", "respond": "respond"},
            "prepare_step_v2": {"execute": "execute", "wait": "wait", "respond": "respond"},
            "execute": {"verify": "verify_v2", "recover": "respond"},
            "verify_v2": {"plan": "plan_v2", "respond": "respond"},
            "prepare_wait": {"wait": "wait"},
            "wait": {"classify": "classify", "execute": "execute"},
        }
        if version == "v2"
        else {
            "classify": {"retrieve": "retrieve", "prepare_wait": "prepare_wait"},
            "retrieve": {"plan": "plan"},
            "plan": {"validate": "validate", "recover": "recover"},
            "validate": {"prepare_step": "prepare_step", "recover": "recover"},
            "prepare_step": {
                "execute": "execute",
                "wait": "wait",
                "respond": "respond",
                "recover": "recover",
            },
            "execute": {"verify": "verify", "recover": "recover"},
            "verify": {"prepare_step": "prepare_step", "respond": "respond", "recover": "recover"},
            "recover": {"plan": "plan", "respond": "respond"},
            "prepare_wait": {"wait": "wait"},
            "wait": {"classify": "classify", "execute": "execute"},
            "reconcile": {"verify": "verify", "recover": "recover"},
        }
    ))

    def route(state):
        return state["routing_decision"]

    for name, edges in routes.items():
        if version == "v3":
            edges.pop("respond", None)
        elif version != "v3":
            edges["respond"] = "respond"
        builder.add_conditional_edges(name, route, edges)
    builder.add_edge("respond_v3" if version == "v3" else "respond", END)
    return builder.compile(checkpointer=checkpointer)


def build_graph_v2(checkpointer: Any, services: Any) -> Any:
    return build_graph(checkpointer, services, version="v2")


def build_graph_v3(checkpointer: Any, services: Any) -> Any:
    return build_graph(checkpointer, services, version="v3")
