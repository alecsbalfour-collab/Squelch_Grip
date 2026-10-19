from __future__ import annotations

import time
import pytest

from core.context_graph import ContextGraphOrchestrator
from core.state_manager import SessionStateManager
from schemas.session_models import (
    AudioCommandType,
    BargeInDirective,
    ContextNode,
    InterruptionType,
    NodeStatus,
    RoomState,
    VoiceProfile,
    VoiceTier,
)


@pytest.mark.asyncio
async def test_orchestrator_initialization_and_root_node() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-alpha-01",
        initial_topic="System Architecture Review",
    )

    assert orchestrator.state.room_id == "room-alpha-01"
    assert orchestrator.state.root_node_id == orchestrator.root_node.node_id
    assert orchestrator.state.current_node_id == orchestrator.root_node.node_id
    assert orchestrator.state.is_ai_speaking is False
    assert orchestrator.state.active_speakers == []
    assert orchestrator.state.lexicon_pack_ids == []

    root = orchestrator.root_node
    assert isinstance(root, ContextNode)
    assert root.parent_id is None
    assert root.topic_summary == "System Architecture Review"
    assert root.active_speaker_id is None
    assert root.child_nodes == []
    assert root.node_id in orchestrator.nodes


@pytest.mark.asyncio
async def test_branch_tangent_parent_child_relationships() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-alpha-02",
        initial_topic="Q4 Product Roadmap",
    )
    root_id = orchestrator.root_node.node_id

    tangent_node = await orchestrator.branch_tangent(
        new_topic="Latency Budget on Edge Nodes",
        speaker_id="speaker-alec",
    )

    assert isinstance(tangent_node, ContextNode)
    assert tangent_node.parent_id == root_id
    assert tangent_node.topic_summary == "Latency Budget on Edge Nodes"
    assert tangent_node.active_speaker_id == "speaker-alec"
    assert tangent_node.node_id in orchestrator.root_node.child_nodes
    assert orchestrator.state.current_node_id == tangent_node.node_id
    assert "speaker-alec" in orchestrator.state.active_speakers

    nested_tangent = await orchestrator.branch_tangent(
        new_topic="WebRTC Jitter Buffer Tuning",
        speaker_id="speaker-maya",
    )

    assert nested_tangent.parent_id == tangent_node.node_id
    assert nested_tangent.node_id in tangent_node.child_nodes
    assert orchestrator.state.current_node_id == nested_tangent.node_id
    assert orchestrator.state.active_speakers == ["speaker-alec", "speaker-maya"]


@pytest.mark.asyncio
async def test_pop_to_anchor_returns_parent_and_root() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-alpha-03",
        initial_topic="Main Deployment Pipeline",
    )
    root_id = orchestrator.root_node.node_id

    tangent_1 = await orchestrator.branch_tangent(
        new_topic="Docker Layer Caching",
        speaker_id="speaker-01",
    )
    tangent_2 = await orchestrator.branch_tangent(
        new_topic="Registry Mirror Auth",
        speaker_id="speaker-02",
    )
    assert orchestrator.state.current_node_id == tangent_2.node_id

    popped_once = await orchestrator.pop_to_anchor()
    assert popped_once.node_id == tangent_1.node_id
    assert orchestrator.state.current_node_id == tangent_1.node_id

    popped_twice = await orchestrator.pop_to_anchor()
    assert popped_twice.node_id == root_id
    assert orchestrator.state.current_node_id == root_id

    popped_at_root = await orchestrator.pop_to_anchor()
    assert popped_at_root.node_id == root_id
    assert orchestrator.state.current_node_id == root_id


@pytest.mark.asyncio
async def test_handle_barge_in_halts_ai_speech() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-alpha-04",
        initial_topic="Quarterly Financial Summary",
    )
    orchestrator.state.is_ai_speaking = True

    payload = await orchestrator.handle_barge_in(
        speaker_id="speaker-interrupter",
        utterance="Hold on, let's check the Q3 numbers first.",
    )

    assert orchestrator.state.is_ai_speaking is False
    assert payload["command"] == "HALT"
    assert payload["room_id"] == "room-alpha-04"
    assert payload["speaker_id"] == "speaker-interrupter"
    assert payload["utterance"] == "Hold on, let's check the Q3 numbers first."
    assert payload["interruption_type"] == InterruptionType.HARD_INTERRUPT.value
    assert payload["is_ai_speaking"] is False
    assert "speaker-interrupter" in orchestrator.state.active_speakers


@pytest.mark.asyncio
async def test_pydantic_schemas_and_state_manager() -> None:
    now = time.time()
    profile = VoiceProfile(
        user_id="usr-100",
        display_name="Alec",
        tier=VoiceTier.PERMANENT,
        embedding_vector=[0.12, -0.45, 0.88, 0.03],
        created_at=now,
    )
    assert profile.tier == VoiceTier.PERMANENT
    assert len(profile.embedding_vector) == 4

    room_state = RoomState(
        room_id="room-schema-test",
        root_node_id="node-root",
        current_node_id="node-root",
        active_speakers=["usr-100"],
        is_ai_speaking=True,
        lexicon_pack_ids=["lex-med-01"],
    )
    assert room_state.is_ai_speaking is True
    assert InterruptionType.BACKCHANNEL == "BACKCHANNEL"
    assert InterruptionType.BACKGROUND_EVENT == "BACKGROUND_EVENT"

    manager = SessionStateManager()
    await manager.register_voice_profile(profile)
    assert manager.get_voice_profile("usr-100") == profile

    room = await manager.create_room("room-managed", "Initial Topic")
    assert manager.get_room("room-managed") is room


@pytest.mark.asyncio
async def test_interruption_types_backchannel_and_background_squelch() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-alpha-05",
        initial_topic="Real-Time Audio Pipeline",
    )
    orchestrator.state.is_ai_speaking = True

    backchannel_directive = await orchestrator.handle_barge_in(
        speaker_id="speaker-listener",
        utterance="uh-huh, yeah",
        interruption_type=InterruptionType.BACKCHANNEL,
    )
    assert isinstance(backchannel_directive, BargeInDirective)
    assert orchestrator.state.is_ai_speaking is True
    assert backchannel_directive.command == AudioCommandType.CONTINUE
    assert backchannel_directive["command"] == "CONTINUE"
    assert backchannel_directive.action == "LOG_BACKCHANNEL"
    assert "speaker-listener" in orchestrator.state.active_speakers
    assert len(orchestrator.current_node.utterances) == 1

    bg_directive = await orchestrator.handle_barge_in(
        speaker_id="noise-source",
        utterance="[door slam]",
        interruption_type=InterruptionType.BACKGROUND_EVENT,
    )
    assert orchestrator.state.is_ai_speaking is True
    assert bg_directive.command == AudioCommandType.IGNORE
    assert bg_directive["command"] == "IGNORE"
    assert bg_directive.action == "SQUELCH_BACKGROUND"
    assert "noise-source" not in orchestrator.state.active_speakers
    assert len(orchestrator.current_node.utterances) == 1


@pytest.mark.asyncio
async def test_lineage_resolver_jump_and_node_lifecycle() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-alpha-06",
        initial_topic="Root Architecture",
    )
    root = orchestrator.root_node

    branch_a = await orchestrator.branch_tangent("Database Sharding", "speaker-a")
    branch_a1 = await orchestrator.branch_tangent("Consistent Hashing Ring", "speaker-b")

    lineage = orchestrator.get_active_lineage()
    assert [n.node_id for n in lineage] == [
        root.node_id,
        branch_a.node_id,
        branch_a1.node_id,
    ]

    await orchestrator.set_node_status(NodeStatus.PARKED, branch_a1.node_id)
    assert branch_a1.status == NodeStatus.PARKED

    await orchestrator.jump_to_node(root.node_id)
    assert orchestrator.state.current_node_id == root.node_id

    branch_b = await orchestrator.branch_tangent("API Rate Limiting", "speaker-c")
    assert branch_b.parent_id == root.node_id

    unresolved = orchestrator.get_unresolved_tangents()
    unresolved_ids = {n.node_id for n in unresolved}
    assert unresolved_ids == {branch_a.node_id, branch_a1.node_id, branch_b.node_id}

    popped = await orchestrator.pop_to_anchor(mark_resolved=True)
    assert popped.node_id == root.node_id
    assert branch_b.status == NodeStatus.RESOLVED

    remaining_unresolved_ids = {n.node_id for n in orchestrator.get_unresolved_tangents()}
    assert remaining_unresolved_ids == {branch_a.node_id, branch_a1.node_id}

