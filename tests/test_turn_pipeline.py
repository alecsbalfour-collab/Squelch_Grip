from __future__ import annotations

import pytest

from core.lexicon_manager import LexiconManager
from core.state_manager import SessionStateManager
from core.turn_pipeline import TurnPipeline
from schemas.lexicon_models import LexiconPack, LexiconTerm
from schemas.session_models import AudioCommandType, VoiceProfile, VoiceTier
from schemas.turn_models import AudioTurnEvent


def _infra_lexicon() -> LexiconManager:
    manager = LexiconManager()
    manager.register_pack(
        LexiconPack(
            pack_id="lex-infra-01",
            name="Infrastructure",
            domain="engineering",
            terms=[LexiconTerm(canonical="Kubernetes", aliases=["k8s"])],
        )
    )
    return manager


@pytest.mark.asyncio
async def test_normal_turn_biometric_match_and_lexicon_normalization() -> None:
    state = SessionStateManager()
    lexicon = _infra_lexicon()
    room = await state.create_room("room-pipe-01", "Infra Review")
    lexicon.attach_pack_to_room(room.state, "lex-infra-01")

    await state.register_voice_profile(
        VoiceProfile(
            user_id="usr-alec",
            display_name="Alec",
            tier=VoiceTier.PERMANENT,
            embedding_vector=[1.0, 0.0],
        )
    )

    pipeline = TurnPipeline(state_manager=state, lexicon_manager=lexicon)
    result = await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-pipe-01",
            raw_utterance="we deploy on k8s",
            embedding_vector=[0.98, 0.02],
        )
    )

    assert result.resolved_speaker_id == "usr-alec"
    assert result.speaker_tier == VoiceTier.PERMANENT
    assert result.normalized_utterance == "we deploy on Kubernetes"
    assert result.barge_in_directive is None
    assert result.branched_tangent is False
    assert result.active_node_id == room.state.current_node_id
    assert room.current_node.utterances[-1].text == "we deploy on Kubernetes"


@pytest.mark.asyncio
async def test_unknown_speaker_auto_registers_as_guest() -> None:
    state = SessionStateManager()
    pipeline = TurnPipeline(state_manager=state, lexicon_manager=LexiconManager())
    await state.create_room("room-pipe-02", "Standup")

    guest_result = await pipeline.process_turn(
        AudioTurnEvent(room_id="room-pipe-02", raw_utterance="hello all")
    )
    assert guest_result.resolved_speaker_id.startswith("guest-")
    assert guest_result.speaker_tier == VoiceTier.CONVERSATIONAL
    assert (
        state.get_voice_profile(guest_result.resolved_speaker_id) is not None
    )

    hinted_result = await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-pipe-02",
            raw_utterance="hi",
            speaker_id_hint="usr-new",
        )
    )
    assert hinted_result.resolved_speaker_id == "usr-new"
    assert hinted_result.speaker_tier == VoiceTier.CONVERSATIONAL
    assert state.get_voice_profile("usr-new") is not None


@pytest.mark.asyncio
async def test_process_turn_unknown_room_raises_keyerror() -> None:
    pipeline = TurnPipeline(
        state_manager=SessionStateManager(), lexicon_manager=LexiconManager()
    )
    with pytest.raises(KeyError):
        await pipeline.process_turn(
            AudioTurnEvent(room_id="room-missing", raw_utterance="hello")
        )


@pytest.mark.asyncio
async def test_barge_in_backchannel_and_background_squelch() -> None:
    state = SessionStateManager()
    pipeline = TurnPipeline(state_manager=state, lexicon_manager=LexiconManager())
    room = await state.create_room("room-pipe-03", "Roadmap")
    room.state.is_ai_speaking = True

    backchannel = await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-pipe-03",
            raw_utterance="yeah",
            speaker_id_hint="usr-listener",
        )
    )
    assert backchannel.barge_in_directive is not None
    assert backchannel.barge_in_directive.command == AudioCommandType.CONTINUE
    assert room.state.is_ai_speaking is True

    background = await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-pipe-03",
            raw_utterance="",
            speaker_id_hint="usr-noise",
            vad_confidence=0.2,
        )
    )
    assert background.barge_in_directive is not None
    assert background.barge_in_directive.command == AudioCommandType.IGNORE
    assert "usr-noise" not in room.state.active_speakers
    assert state.get_voice_profile("usr-noise") is None


@pytest.mark.asyncio
async def test_barge_in_hard_interrupt_with_and_without_tangent() -> None:
    state = SessionStateManager()
    pipeline = TurnPipeline(state_manager=state, lexicon_manager=LexiconManager())
    room = await state.create_room("room-pipe-04", "Budget Review")
    room.state.is_ai_speaking = True
    root_id = room.state.current_node_id

    plain_interrupt = await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-pipe-04",
            raw_utterance="wait, go back",
            speaker_id_hint="usr-cfo",
        )
    )
    assert plain_interrupt.barge_in_directive is not None
    assert plain_interrupt.barge_in_directive.command == AudioCommandType.HALT
    assert plain_interrupt.branched_tangent is False
    assert room.state.is_ai_speaking is False
    assert room.state.current_node_id == root_id

    room.state.is_ai_speaking = True
    tangent_interrupt = await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-pipe-04",
            raw_utterance="stop, new topic",
            speaker_id_hint="usr-cfo",
            new_tangent_topic="Marketing Spend",
        )
    )
    assert tangent_interrupt.barge_in_directive is not None
    assert tangent_interrupt.barge_in_directive.command == AudioCommandType.HALT
    assert tangent_interrupt.branched_tangent is True
    assert room.current_node.topic_summary == "Marketing Spend"
    assert room.current_node.parent_id == root_id
    assert tangent_interrupt.active_node_id == room.current_node.node_id
    assert room.current_node.utterances[-1].text == "stop, new topic"


@pytest.mark.asyncio
async def test_normal_turn_with_new_tangent_branches_then_records() -> None:
    state = SessionStateManager()
    pipeline = TurnPipeline(state_manager=state, lexicon_manager=LexiconManager())
    room = await state.create_room("room-pipe-05", "Sprint Planning")

    result = await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-pipe-05",
            raw_utterance="let's discuss caching",
            speaker_id_hint="usr-maya",
            new_tangent_topic="Cache Strategy",
        )
    )

    assert result.branched_tangent is True
    assert result.barge_in_directive is None
    assert room.current_node.topic_summary == "Cache Strategy"
    assert room.current_node.utterances[-1].text == "let's discuss caching"
    assert room.current_node.utterances[-1].speaker_id == "usr-maya"
