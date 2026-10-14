"""Conversation browsing keeps the same owner and tenant boundaries as runs."""

from uuid import uuid4


def test_chat_conversation_history_and_owner_scope(client):
    owner = {"Authorization": "Bearer owner"}
    first = client.post(
        "/runs",
        headers={**owner, "Idempotency-Key": str(uuid4())},
        json={"message": "What is in the policy?"},
    )
    assert first.status_code == 202
    conversation_id = first.json()["conversation_id"]
    assert first.json()["run_id"]
    second = client.post(
        "/runs",
        headers={**owner, "Idempotency-Key": str(uuid4())},
        json={"message": "And the return window?", "conversation_id": conversation_id},
    )
    assert second.status_code == 202
    assert second.json()["conversation_id"] == conversation_id

    history = client.get(f"/conversations/{conversation_id}", headers=owner)
    assert history.status_code == 200
    assert history.json()["title"] == "What is in the policy?"
    assert [item["message"] for item in history.json()["runs"]] == [
        "What is in the policy?",
        "And the return window?",
    ]
    assert all(item["run"]["conversation_id"] == conversation_id for item in history.json()["runs"])
    assert any(
        item["conversation_id"] == conversation_id
        for item in client.get("/conversations", headers=owner).json()
    )

    for token in ("peer", "outsider"):
        other = {"Authorization": f"Bearer {token}"}
        assert client.get(f"/conversations/{conversation_id}", headers=other).status_code == 404
        assert all(
            item["conversation_id"] != conversation_id
            for item in client.get("/conversations", headers=other).json()
        )
