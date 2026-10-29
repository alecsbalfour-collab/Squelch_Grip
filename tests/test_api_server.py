from __future__ import annotations

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from api.server import create_app
from core.client_sdk import SquelchGripClient
from schemas.turn_models import AudioTurnEvent


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "squelch_grip_backend",
        "version": "1.0.0a1",
    }


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


def test_crosstalk_batch_endpoint_and_ws_broadcast() -> None:
    client = TestClient(create_app())
    client.post(
        "/rooms", json={"room_id": "room-xtalk-01", "initial_topic": "Design"}
    )

    with (
        client.websocket_connect("/ws/rooms/room-xtalk-01") as ws_audio,
        client.websocket_connect("/ws/rooms/room-xtalk-01") as ws_ui,
    ):
        ws_audio.send_json(
            {"raw_utterance": "starting discussion", "speaker_id_hint": "usr-1"}
        )
        msg_audio = ws_audio.receive_json()
        msg_ui = ws_ui.receive_json()
        assert msg_audio["resolved_speaker_id"] == "usr-1"
        assert msg_ui["resolved_speaker_id"] == "usr-1"

        xtalk = client.post(
            "/rooms/room-xtalk-01/crosstalk",
            json={
                "room_id": "room-xtalk-01",
                "frames": [
                    {
                        "speaker_id": "usr-guest",
                        "tier": "CONVERSATIONAL",
                        "utterance": "let me jump in",
                        "rms_db": -10.0,
                    },
                    {
                        "speaker_id": "usr-host",
                        "tier": "PERMANENT",
                        "utterance": "quick tangent on Auth Tokens",
                        "rms_db": -25.0,
                    },
                ],
            },
        )
        assert xtalk.status_code == 200
        body = xtalk.json()
        assert body["arbitration"]["winning_speaker_id"] == "usr-host"
        assert body["turn_result"]["resolved_speaker_id"] == "usr-host"
        assert body["turn_result"]["branched_tangent"] is True

        broadcast_audio = ws_audio.receive_json()
        broadcast_ui = ws_ui.receive_json()
        assert broadcast_audio["resolved_speaker_id"] == "usr-host"
        assert broadcast_ui["branched_tangent"] is True



def test_default_lexicons_room_creation_and_cors() -> None:
    client = TestClient(create_app())

    packs = client.get("/lexicons")
    assert packs.status_code == 200
    pack_ids = {p["pack_id"] for p in packs.json()}
    assert {"lex-f1ndr", "lex-dealr"} <= pack_ids

    created = client.post(
        "/rooms",
        json={
            "room_id": "room-int-01",
            "initial_topic": "Dealr Negotiation",
            "lexicon_pack_ids": ["lex-f1ndr", "lex-dealr"],
        },
    )
    assert created.status_code == 200
    assert created.json()["lexicon_pack_ids"] == ["lex-f1ndr", "lex-dealr"]

    bad = client.post(
        "/rooms",
        json={
            "room_id": "room-int-02",
            "initial_topic": "Bad Pack",
            "lexicon_pack_ids": ["lex-nope"],
        },
    )
    assert bad.status_code == 404

    turn = client.post(
        "/rooms/room-int-01/turns",
        json={
            "room_id": "room-int-01",
            "raw_utterance": "finder sent a counter offer to dealer",
            "speaker_id_hint": "usr-a",
        },
    )
    assert turn.status_code == 200
    assert turn.json()["normalized_utterance"] == (
        "F1ndr sent a counteroffer to Dealr"
    )

    cors = client.get("/health", headers={"Origin": "https://f1ndr.app"})
    assert cors.headers.get("access-control-allow-origin") in (
        "*",
        "https://f1ndr.app",
    )


@pytest.mark.asyncio
async def test_squelch_grip_client_bridge() -> None:
    client = SquelchGripClient(app=create_app())

    f1ndr_state = await client.create_f1ndr_room("room-sdk-f", "Discovery")
    assert "lex-f1ndr" in f1ndr_state.lexicon_pack_ids

    dealr_state = await client.create_dealr_room("room-sdk-d", "Negotiation")
    assert "lex-dealr" in dealr_state.lexicon_pack_ids

    f1ndr_turn = await client.send_turn(
        AudioTurnEvent(
            room_id="room-sdk-f",
            raw_utterance="finder is now live",
            speaker_id_hint="usr-a",
        )
    )
    assert f1ndr_turn.normalized_utterance == "F1ndr is now live"

    dealr_turn = await client.send_turn(
        AudioTurnEvent(
            room_id="room-sdk-d",
            raw_utterance="dealer counter offer received",
            speaker_id_hint="usr-b",
        )
    )
    assert dealr_turn.normalized_utterance == "Dealr counteroffer received"

    prompt = await client.get_prompt("room-sdk-f")
    assert prompt.room_id == "room-sdk-f"
    assert prompt.active_topic == "Discovery"
    assert prompt.domain_glossary["F1ndr"] == (
        "F1ndr discovery and matching platform."
    )

    delta = await client.get_prompt_delta("room-sdk-f")
    assert delta.room_id == "room-sdk-f"
    assert delta.topic_switched is False
    assert "usr-a: F1ndr is now live" in delta.delta_utterances


def test_api_key_authentication_rest_and_websocket() -> None:
    client = TestClient(create_app(api_key="secret-prod-key"))

    assert client.get("/health").status_code == 200

    unauthorized = client.post(
        "/rooms",
        json={"room_id": "room-secure", "initial_topic": "Secure Sync"},
    )
    assert unauthorized.status_code == 401

    wrong_key = client.post(
        "/rooms",
        json={"room_id": "room-secure", "initial_topic": "Secure Sync"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert wrong_key.status_code == 401

    authorized = client.post(
        "/rooms",
        json={"room_id": "room-secure", "initial_topic": "Secure Sync"},
        headers={"X-API-Key": "secret-prod-key"},
    )
    assert authorized.status_code == 200
    assert authorized.json()["room_id"] == "room-secure"

    assert client.get("/rooms/room-secure").status_code == 401
    assert (
        client.get(
            "/rooms/room-secure", headers={"X-API-Key": "secret-prod-key"}
        ).status_code
        == 200
    )

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/rooms/room-secure"):
            pass

    with client.websocket_connect(
        "/ws/rooms/room-secure?api_key=secret-prod-key"
    ) as ws:
        ws.send_json({"raw_utterance": "secure hello", "speaker_id_hint": "usr-s"})
        result = ws.receive_json()
        assert result["room_id"] == "room-secure"


@pytest.mark.asyncio
async def test_squelch_grip_client_with_api_key() -> None:
    secured = SquelchGripClient(
        app=create_app(api_key="secret-prod-key"),
        api_key="secret-prod-key",
    )
    state = await secured.create_f1ndr_room("room-sdk-sec", "Secured Discovery")
    assert state.room_id == "room-sdk-sec"
    assert "lex-f1ndr" in state.lexicon_pack_ids

    turn = await secured.send_turn(
        AudioTurnEvent(
            room_id="room-sdk-sec",
            raw_utterance="finder is locked down",
            speaker_id_hint="usr-a",
        )
    )
    assert turn.normalized_utterance == "F1ndr is locked down"

    no_key = SquelchGripClient(app=create_app(api_key="secret-prod-key"))
    try:
        await no_key.create_f1ndr_room("room-sdk-denied", "Denied")
        assert False, "Expected HTTPStatusError for missing API key"
    except Exception:
        pass
