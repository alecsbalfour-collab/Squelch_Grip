from __future__ import annotations

import time

from fastapi.testclient import TestClient

from api.server import create_app
from schemas.navigation_models import NavigationAction


def _turn(client: TestClient, room_id: str, **payload):
    payload.setdefault("room_id", room_id)
    return client.post(f"/rooms/{room_id}/turns", json=payload)


def test_multi_speaker_full_duplex_stress_simulation() -> None:
    client = TestClient(create_app())
    room_id = "room-stress-01"

    client.post(
        "/rooms",
        json={"room_id": room_id, "initial_topic": "Infrastructure Roadmap"},
    )
    for user_id, tier in (
        ("usr-perm", "PERMANENT"),
        ("usr-temp", "TEMPORARY"),
        ("usr-conv", "CONVERSATIONAL"),
    ):
        resp = client.post(
            "/profiles",
            json={
                "user_id": user_id,
                "display_name": user_id,
                "tier": tier,
                "embedding_vector": [1.0, 0.0],
            },
        )
        assert resp.status_code == 200

    client.post(
        "/lexicons",
        json={
            "pack_id": "lex-stress-01",
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
    attached = client.post(f"/rooms/{room_id}/lexicons/lex-stress-01")
    assert attached.json()["lexicon_pack_ids"] == ["lex-stress-01"]

    # --- Normal turns with ASR alias normalization ---
    t1 = _turn(
        client, room_id,
        raw_utterance="we deploy on k8s clusters",
        speaker_id_hint="usr-perm",
    ).json()
    assert t1["normalized_utterance"] == "we deploy on Kubernetes clusters"
    t2 = _turn(
        client, room_id,
        raw_utterance="the k8s ingress is flaky",
        speaker_id_hint="usr-temp",
    ).json()
    assert t2["normalized_utterance"] == "the Kubernetes ingress is flaky"
    _turn(client, room_id, raw_utterance="autoscaling looks fine", speaker_id_hint="usr-conv")

    # --- Full-duplex segment: backchannels, noise, hard interrupt ---
    client.post(f"/rooms/{room_id}/ai-speaking", json={"is_ai_speaking": True})
    bc1 = _turn(client, room_id, raw_utterance="yeah", speaker_id_hint="usr-temp").json()
    assert bc1["barge_in_directive"]["command"] == "CONTINUE"
    bc2 = _turn(client, room_id, raw_utterance="mm-hmm", speaker_id_hint="usr-conv").json()
    assert bc2["barge_in_directive"]["command"] == "CONTINUE"

    n1 = _turn(
        client, room_id,
        raw_utterance="[cough]",
        speaker_id_hint="usr-noise",
    ).json()
    assert n1["barge_in_directive"]["command"] == "IGNORE"
    n2 = _turn(client, room_id, raw_utterance="[door slam]").json()
    assert n2["barge_in_directive"]["command"] == "IGNORE"

    hi = _turn(
        client, room_id,
        raw_utterance="wait, stop the rollout",
        speaker_id_hint="usr-perm",
    ).json()
    assert hi["barge_in_directive"]["command"] == "HALT"
    assert client.get(f"/rooms/{room_id}").json()["is_ai_speaking"] is False

    # --- Crosstalk collision: 3 speakers + noise frame ---
    crosstalk = client.post(
        f"/rooms/{room_id}/crosstalk",
        json={
            "room_id": room_id,
            "frames": [
                {
                    "speaker_id": "usr-perm",
                    "tier": "PERMANENT",
                    "utterance": "actually the migration window matters",
                    "rms_db": -18.0,
                },
                {
                    "speaker_id": "usr-temp",
                    "tier": "TEMPORARY",
                    "utterance": "the latency budget needs work",
                    "rms_db": -12.0,
                },
                {
                    "speaker_id": "usr-conv",
                    "tier": "CONVERSATIONAL",
                    "utterance": "i want to discuss quotas",
                    "rms_db": -10.0,
                },
                {
                    "speaker_id": "usr-hvac",
                    "tier": "CONVERSATIONAL",
                    "utterance": "",
                    "vad_confidence": 0.1,
                    "rms_db": -60.0,
                },
            ],
            "current_floor_speaker_id": "usr-temp",
            "last_turn_timestamp": time.time() - 1.0,
        },
    )
    assert crosstalk.status_code == 200
    ct = crosstalk.json()
    assert ct["arbitration"]["winning_speaker_id"] == "usr-perm"
    assert ct["arbitration"]["dropped_speaker_ids"] == ["usr-hvac"]
    assert ct["turn_result"]["resolved_speaker_id"] == "usr-perm"

    # --- Spoken navigation cues ---
    br = _turn(
        client, room_id,
        raw_utterance="quick tangent on caching strategy",
        speaker_id_hint="usr-temp",
    ).json()
    assert br["branched_tangent"] is True
    caching_node_id = br["active_node_id"]

    _turn(
        client, room_id,
        raw_utterance="redis failover is risky here",
        speaker_id_hint="usr-perm",
    )
    _turn(
        client, room_id,
        raw_utterance="let's put a pin in this",
        speaker_id_hint="usr-conv",
    )
    state = client.get(f"/rooms/{room_id}").json()
    assert state["current_node_id"] != caching_node_id

    jm = _turn(
        client, room_id,
        raw_utterance="going back to caching strategy",
        speaker_id_hint="usr-perm",
    ).json()
    assert jm["active_node_id"] == caching_node_id

    pp = _turn(
        client, room_id,
        raw_utterance="anyway back to the main topic",
        speaker_id_hint="usr-temp",
    ).json()
    assert pp["active_node_id"] != caching_node_id

    br2 = _turn(
        client, room_id,
        raw_utterance="sidebar on tls rotation",
        speaker_id_hint="usr-conv",
    ).json()
    assert br2["branched_tangent"] is True
    tls_node_id = br2["active_node_id"]

    _turn(client, room_id, raw_utterance="cert expiry is close", speaker_id_hint="usr-temp")
    _turn(client, room_id, raw_utterance="monitoring dashboards need work", speaker_id_hint="usr-perm")
    _turn(client, room_id, raw_utterance="the k8s autoscaler config", speaker_id_hint="usr-conv")
    _turn(client, room_id, raw_utterance="budget review is next sprint", speaker_id_hint="usr-temp")

    # --- Diagnostics overlay ---
    diag = client.get(f"/rooms/{room_id}/diagnostics")
    assert diag.status_code == 200
    report = diag.json()
    assert report["room_id"] == room_id
    assert report["active_node_id"] == tls_node_id
    assert report["active_topic"] == "tls rotation"
    assert report["total_node_count"] == 3
    assert report["unresolved_tangent_ids"] == [tls_node_id]
    assert report["speaker_tiers"] == {
        "usr-perm": "PERMANENT",
        "usr-temp": "TEMPORARY",
        "usr-conv": "CONVERSATIONAL",
    }
    assert report["active_lexicon_packs"] == ["lex-stress-01"]

    nav_actions = {
        d["action"] for d in report["recent_navigation_decisions"]
    }
    assert nav_actions == {
        NavigationAction.BRANCH_TANGENT.value,
        NavigationAction.PARK_AND_POP.value,
        NavigationAction.JUMP_TO_NODE.value,
        NavigationAction.POP_TO_ANCHOR.value,
    }
    assert len(report["recent_arbitration_outcomes"]) == 1
    commands = [
        d["command"] for d in report["recent_barge_in_directives"]
    ]
    assert commands.count("CONTINUE") == 2
    assert commands.count("IGNORE") == 2
    assert commands.count("HALT") == 1
    assert client.get("/rooms/room-missing/diagnostics").status_code == 404

    # --- Delta prompt packet ---
    delta = client.get(f"/rooms/{room_id}/prompt/delta")
    assert delta.status_code == 200
    delta_body = delta.json()
    assert delta_body["active_topic"] == "tls rotation"
    assert delta_body["topic_switched"] is False
    assert delta_body["lineage_breadcrumbs"] == [
        "Infrastructure Roadmap",
        "tls rotation",
    ]

    delta_same = client.get(
        f"/rooms/{room_id}/prompt/delta",
        params={"previous_node_id": tls_node_id, "since_timestamp": 9999999999.0},
    )
    same_body = delta_same.json()
    assert same_body["topic_switched"] is False
    assert same_body["delta_utterances"] == []
    assert same_body["lineage_breadcrumbs"] == []

    delta_switch = client.get(
        f"/rooms/{room_id}/prompt/delta",
        params={"previous_node_id": caching_node_id},
    )
    assert delta_switch.json()["topic_switched"] is True

    # --- Prune resolved branches ---
    prune = client.post(f"/rooms/{room_id}/prune")
    assert prune.status_code == 200
    assert prune.json()["pruned_node_ids"] == [caching_node_id]

    snapshot = client.get(f"/rooms/{room_id}/snapshot").json()
    root_node = snapshot["nodes"][snapshot["state"]["root_node_id"]]
    assert root_node["metadata"]["archived_tangents"][0]["topic_summary"] == (
        "caching strategy"
    )

    diag2 = client.get(f"/rooms/{room_id}/diagnostics").json()
    assert diag2["total_node_count"] == 2
    assert diag2["unresolved_tangent_ids"] == [tls_node_id]
