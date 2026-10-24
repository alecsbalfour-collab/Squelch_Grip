from __future__ import annotations

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from api.server import create_app


def test_rest_room_profile_lexicon_prompt_flow() -> None:
    client = TestClient(create_app())

    created = client.post(
        "/rooms",
        json={"room_id": "room-api-01", "initial_topic": "Design Review"},
    )
    assert created.status_code == 200
    assert created.json()["room_id"] == "room-api-01"
    assert created.json()["is_ai_speaking"] is False

    assert client.get("/rooms/room-api-01").json()["room_id"] == "room-api-01"
    assert client.get("/rooms/room-missing").status_code == 404

    profile = client.post(
        "/profiles",
        json={
            "user_id": "usr-alec",
            "display_name": "Alec",
            "tier": "PERMANENT",
            "embedding_vector": [1.0, 0.0],
        },
    )
    assert profile.status_code == 200
    assert profile.json()["tier"] == "PERMANENT"

    pack = client.post(
        "/lexicons",
        json={
            "pack_id": "lex-infra-01",
            "name": "Infrastructure",
            "domain": "engineering",
            "terms": [
                {
                    "canonical": "Kubernetes",
                    "aliases": ["k8s"],
                    "definition": "Container orchestration platform.",
                }
            ],
        },
    )
    assert pack.status_code == 200

    attached = client.post("/rooms/room-api-01/lexicons/lex-infra-01")
    assert attached.status_code == 200
    assert attached.json()["lexicon_pack_ids"] == ["lex-infra-01"]
    assert client.post("/rooms/room-api-01/lexicons/lex-missing").status_code == 404

    turn = client.post(
        "/rooms/room-api-01/turns",
        json={
            "room_id": "room-api-01",
            "raw_utterance": "we deploy on k8s",
            "speaker_id_hint": "usr-alec",
        },
    )
    assert turn.status_code == 200
    assert turn.json()["normalized_utterance"] == "we deploy on Kubernetes"
    assert turn.json()["resolved_speaker_id"] == "usr-alec"

    prompt = client.get("/rooms/room-api-01/prompt")
    assert prompt.status_code == 200
    body = prompt.json()
    assert body["active_topic"] == "Design Review"
    assert body["lineage_breadcrumbs"] == ["Design Review"]
    assert body["recent_utterances"] == ["usr-alec: we deploy on Kubernetes"]
    assert body["domain_glossary"] == {
        "Kubernetes": "Container orchestration platform."
    }
    assert "# Active Topic" in body["system_prompt_markdown"]


def test_rest_ai_speaking_flag_and_turn_404() -> None:
    client = TestClient(create_app())
    client.post("/rooms", json={"room_id": "room-api-02", "initial_topic": "Sync"})

    flagged = client.post(
        "/rooms/room-api-02/ai-speaking", json={"is_ai_speaking": True}
    )
    assert flagged.status_code == 200
    assert flagged.json()["is_ai_speaking"] is True

    missing = client.post(
        "/rooms/room-gone/turns",
        json={"room_id": "room-gone", "raw_utterance": "hi"},
    )
    assert missing.status_code == 404


def test_websocket_full_duplex_turns() -> None:
    client = TestClient(create_app())
    client.post("/rooms", json={"room_id": "room-ws-01", "initial_topic": "Roadmap"})

    with client.websocket_connect("/ws/rooms/room-ws-01") as ws:
        ws.send_json({"raw_utterance": "hello team", "speaker_id_hint": "usr-a"})
        normal = ws.receive_json()
        assert normal["room_id"] == "room-ws-01"
        assert normal["resolved_speaker_id"] == "usr-a"
        assert normal["barge_in_directive"] is None
        assert normal["normalized_utterance"] == "hello team"

        client.post(
            "/rooms/room-ws-01/ai-speaking", json={"is_ai_speaking": True}
        )

        ws.send_json({"raw_utterance": "yeah", "speaker_id_hint": "usr-b"})
        backchannel = ws.receive_json()
        assert backchannel["barge_in_directive"]["command"] == "CONTINUE"
        assert backchannel["barge_in_directive"]["action"] == "LOG_BACKCHANNEL"
        assert backchannel["barge_in_directive"]["is_ai_speaking"] is True

        ws.send_json({"raw_utterance": "[hvac hum]", "vad_confidence": 0.1})
        background = ws.receive_json()
        assert background["barge_in_directive"]["command"] == "IGNORE"

        ws.send_json({"raw_utterance": "wait, stop that", "speaker_id_hint": "usr-c"})
        interrupt = ws.receive_json()
        assert interrupt["barge_in_directive"]["command"] == "HALT"
        assert interrupt["barge_in_directive"]["is_ai_speaking"] is False

        state = client.get("/rooms/room-ws-01").json()
        assert state["is_ai_speaking"] is False
        assert "usr-c" in state["active_speakers"]


def test_websocket_unknown_room_closes() -> None:
    client = TestClient(create_app())
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/rooms/room-missing"):
            pass
