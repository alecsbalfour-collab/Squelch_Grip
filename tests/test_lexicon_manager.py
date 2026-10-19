from __future__ import annotations

import pytest

from core.lexicon_manager import LexiconManager
from schemas.lexicon_models import LexiconPack, LexiconTerm
from schemas.session_models import RoomState


def _room_state(room_id: str = "room-lex-01") -> RoomState:
    return RoomState(
        room_id=room_id,
        root_node_id="node-root",
        current_node_id="node-root",
    )


def _medical_pack() -> LexiconPack:
    return LexiconPack(
        pack_id="lex-med-01",
        name="Clinical Vocabulary",
        domain="medical",
        terms=[
            LexiconTerm(
                canonical="hypertension",
                aliases=["high blood pressure", "hypertention"],
                boost_weight=2.0,
                definition="Chronically elevated arterial blood pressure.",
            ),
            LexiconTerm(
                canonical="acetaminophen",
                aliases=["tylenol"],
            ),
        ],
    )


def _infra_pack() -> LexiconPack:
    return LexiconPack(
        pack_id="lex-infra-01",
        name="Infrastructure Vocabulary",
        domain="engineering",
        terms=[
            LexiconTerm(canonical="Kubernetes", aliases=["k8s", "kuburnetes"]),
        ],
    )


def test_register_and_get_pack() -> None:
    manager = LexiconManager()
    pack = _medical_pack()

    manager.register_pack(pack)

    assert manager.get_pack("lex-med-01") is pack
    assert manager.get_pack("lex-missing") is None
    assert pack.terms[0].boost_weight == 2.0
    assert pack.terms[1].boost_weight == 1.5


def test_attach_pack_to_room() -> None:
    manager = LexiconManager()
    manager.register_pack(_medical_pack())
    room = _room_state()

    updated = manager.attach_pack_to_room(room, "lex-med-01")

    assert updated is room
    assert room.lexicon_pack_ids == ["lex-med-01"]

    manager.attach_pack_to_room(room, "lex-med-01")
    assert room.lexicon_pack_ids == ["lex-med-01"]

    with pytest.raises(KeyError):
        manager.attach_pack_to_room(room, "lex-unregistered")


def test_normalize_utterance_alias_to_canonical() -> None:
    manager = LexiconManager()
    manager.register_pack(_medical_pack())
    manager.register_pack(_infra_pack())

    normalized = manager.normalize_utterance(
        "The patient has high blood pressure, so we started tylenol.",
        ["lex-med-01"],
    )
    assert normalized == (
        "The patient has hypertension, so we started acetaminophen."
    )

    normalized_infra = manager.normalize_utterance(
        "We deploy on K8s, but the kuburnetes control plane is flaky.",
        ["lex-infra-01"],
    )
    assert normalized_infra == (
        "We deploy on Kubernetes, but the Kubernetes control plane is flaky."
    )


def test_normalize_utterance_whole_word_and_case_insensitive() -> None:
    manager = LexiconManager()
    manager.register_pack(_infra_pack())

    text = "The k8smaster node differs from k8s workers."
    normalized = manager.normalize_utterance(text, ["lex-infra-01"])

    assert normalized == "The k8smaster node differs from Kubernetes workers."
    assert manager.normalize_utterance(text, []) == text
    assert manager.normalize_utterance(text, ["lex-missing"]) == text


def test_normalize_utterance_later_pack_wins_alias_conflicts() -> None:
    manager = LexiconManager()
    general = LexiconPack(
        pack_id="lex-general",
        name="General",
        domain="general",
        terms=[LexiconTerm(canonical="doctor", aliases=["doc"])],
    )
    medical = LexiconPack(
        pack_id="lex-med-02",
        name="Clinical",
        domain="medical",
        terms=[LexiconTerm(canonical="doxorubicin", aliases=["doc"])],
    )
    manager.register_pack(general)
    manager.register_pack(medical)

    assert (
        manager.normalize_utterance("ask the doc", ["lex-general", "lex-med-02"])
        == "ask the doxorubicin"
    )
    assert (
        manager.normalize_utterance("ask the doc", ["lex-med-02", "lex-general"])
        == "ask the doctor"
    )
