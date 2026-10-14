"""Read wait requests from both legacy state fields and LangGraph interrupts."""

from typing import Any

from app.agent.state import WaitRequest


def wait_request_from_value(value: Any) -> WaitRequest | None:
    if isinstance(value, WaitRequest):
        return value
    if isinstance(value, dict) and "wait_token" in value and "kind" in value:
        try:
            return WaitRequest.model_validate(value)
        except ValueError:
            return None
    nested = getattr(value, "value", None)
    return wait_request_from_value(nested) if nested is not None else None


def wait_request_from_snapshot(snapshot: Any) -> WaitRequest | None:
    values = getattr(snapshot, "values", {}) or {}
    wait = wait_request_from_value(values.get("wait_request"))
    if wait is not None:
        return wait
    for task in getattr(snapshot, "tasks", ()) or ():
        for interruption in getattr(task, "interrupts", ()) or ():
            wait = wait_request_from_value(interruption)
            if wait is not None:
                return wait
    return None


def wait_request_from_checkpoint(checkpoint: Any) -> WaitRequest | None:
    if checkpoint is None:
        return None
    values = checkpoint.checkpoint.get("channel_values", {}) or {}
    wait = wait_request_from_value(values.get("wait_request"))
    if wait is not None:
        return wait
    for _, channel, value in checkpoint.pending_writes or ():
        if channel != "__interrupt__":
            continue
        candidates = value if isinstance(value, (tuple, list)) else [value]
        for interruption in candidates:
            wait = wait_request_from_value(interruption)
            if wait is not None:
                return wait
    return None
