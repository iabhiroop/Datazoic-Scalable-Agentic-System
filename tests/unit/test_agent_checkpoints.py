from types import SimpleNamespace

from app.agent.checkpoints import wait_request_from_checkpoint, wait_request_from_snapshot


def wait_value():
    return {
        "wait_token": "token-1",
        "kind": "approval",
        "reason": "Approve action",
        "resume_route": "execute",
        "expected_payload_schema": {"type": "object"},
        "operation_id": "operation-1",
        "input_hash": "hash-1",
    }


def test_reads_wait_request_from_legacy_state():
    snapshot = SimpleNamespace(values={"wait_request": wait_value()}, tasks=[])
    wait = wait_request_from_snapshot(snapshot)
    assert wait is not None
    assert wait.wait_token == "token-1"


def test_reads_wait_request_from_langgraph_interrupt():
    interrupt = SimpleNamespace(value=wait_value())
    snapshot = SimpleNamespace(values={}, tasks=[SimpleNamespace(interrupts=[interrupt])])
    checkpoint = SimpleNamespace(
        checkpoint={"channel_values": {}}, pending_writes=[("task-1", "__interrupt__", interrupt)]
    )
    assert wait_request_from_snapshot(snapshot).operation_id == "operation-1"
    assert wait_request_from_checkpoint(checkpoint).input_hash == "hash-1"
