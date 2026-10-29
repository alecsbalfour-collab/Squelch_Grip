from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from core.session_store import SessionSnapshotStore
from core.state_manager import SessionStateManager
from schemas.snapshot_models import RoomSnapshot


@pytest.mark.asyncio
async def test_export_room_deep_copies_state_and_nodes() -> None:
    state = SessionStateManager()
    room = await state.create_room("room-snap-01", "Root Topic")
    await room.branch_tangent("Tangent A", "usr-a")
    await room.record_utterance("usr-a", "hello")

    snapshot = SessionSnapshotStore().export_room(room)

    assert snapshot.state.room_id == "room-snap-01"
    assert set(snapshot.nodes) == set(room.nodes)
    assert len(snapshot.nodes[room.state.current_node_id].utterances) == 1

    room.current_node.utterances.clear()
    assert len(snapshot.nodes[room.state.current_node_id].utterances) == 1


@pytest.mark.asyncio
async def test_restore_room_round_trip() -> None:
    source = SessionStateManager()
    room = await source.create_room("room-snap-02", "Quarterly Review")
    tangent = await room.branch_tangent("Marketing", "usr-b")
    await room.record_utterance("usr-b", "spend is up")
    snapshot = SessionSnapshotStore().export_room(room)

    target = SessionStateManager()
    store = SessionSnapshotStore()
    restored = await store.restore_room(target, snapshot)

    assert target.get_room("room-snap-02") is restored
    assert restored.state.room_id == "room-snap-02"
    assert restored.state.current_node_id == tangent.node_id
    assert restored.current_node.topic_summary == "Marketing"
    assert restored.current_node.utterances[-1].text == "spend is up"
    assert set(restored.nodes) == set(room.nodes)

    # Restored graph is independent of the snapshot payload.
    snapshot.nodes[tangent.node_id].topic_summary = "Mutated"
    assert restored.current_node.topic_summary == "Marketing"


@pytest.mark.asyncio
async def test_restore_room_rejects_corrupted_snapshots() -> None:
    source = SessionStateManager()
    room = await source.create_room("room-snap-03", "Root")
    snapshot = SessionSnapshotStore().export_room(room)
    store = SessionSnapshotStore()

    bad_current = snapshot.model_copy(deep=True)
    bad_current.state.current_node_id = "node-missing"
    with pytest.raises(ValueError):
        await store.restore_room(SessionStateManager(), bad_current)

    bad_root = snapshot.model_copy(deep=True)
    bad_root.state.root_node_id = "node-missing"
    with pytest.raises(ValueError):
        await store.restore_room(SessionStateManager(), bad_root)


def test_rest_snapshot_export_and_restore() -> None:
    client = TestClient(create_app())
    client.post(
        "/rooms", json={"room_id": "room-rest-01", "initial_topic": "Sync"}
    )
    client.post(
        "/rooms/room-rest-01/turns",
        json={
            "room_id": "room-rest-01",
            "raw_utterance": "first point",
            "speaker_id_hint": "usr-a",
            "new_tangent_topic": "Tangent Topic",
        },
    )

    exported = client.get("/rooms/room-rest-01/snapshot")
    assert exported.status_code == 200
    body = exported.json()
    RoomSnapshot(**body)
    assert body["state"]["room_id"] == "room-rest-01"
    assert len(body["nodes"]) == 2
    assert "exported_at" in body

    assert client.get("/rooms/room-missing/snapshot").status_code == 404

    restored = client.post("/rooms/restore", json=body)
    assert restored.status_code == 200
    assert restored.json()["room_id"] == "room-rest-01"
    assert client.get("/rooms/room-rest-01").status_code == 200


def test_rest_restore_corrupted_snapshot_returns_400() -> None:
    client = TestClient(create_app())
    client.post(
        "/rooms", json={"room_id": "room-rest-02", "initial_topic": "Sync"}
    )
    body = client.get("/rooms/room-rest-02/snapshot").json()
    body["state"]["current_node_id"] = "node-missing"

    response = client.post("/rooms/restore", json=body)
    assert response.status_code == 400
