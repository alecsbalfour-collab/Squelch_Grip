from __future__ import annotations

import pytest

from core.context_graph import ContextGraphOrchestrator
from core.lexicon_manager import LexiconManager
from core.navigation_detector import NavigationIntentDetector
from core.state_manager import SessionStateManager
from core.turn_pipeline import TurnPipeline
from schemas.navigation_models import NavigationAction
from schemas.session_models import NodeStatus
from schemas.turn_models import AudioTurnEvent


def test_detect_branch_tangent_phrases() -> None:
    detector = NavigationIntentDetector()

    for utterance in (
        "quick tangent on Redis cluster failover",
        "TangENT about cache eviction",
        "sidebar on TLS rotation",
        "side note on billing",
    ):
        decision = detector.detect(utterance)
        assert decision.action == NavigationAction.BRANCH_TANGENT, utterance
        assert decision.target_topic, utterance

    decision = detector.detect("quick tangent on Redis cluster failover.")
    assert decision.target_topic == "Redis cluster failover"


def test_detect_park_and_pop() -> None:
    detector = NavigationIntentDetector()

    for utterance in (
        "let's park this",
        "let's park that for now",
        "let's put a pin in this for now",
        "table this for now",
    ):
        decision = detector.detect(utterance)
        assert decision.action == NavigationAction.PARK_AND_POP, utterance


def test_detect_pop_to_anchor_marks_resolved() -> None:
    detector = NavigationIntentDetector()

    for utterance in (
        "anyway, back to the main topic",
        "back to what we were saying",
        "that resolves that",
        "return to the anchor",
    ):
        decision = detector.detect(utterance)
        assert decision.action == NavigationAction.POP_TO_ANCHOR, utterance
        assert decision.mark_resolved is True, utterance


@pytest.mark.asyncio
async def test_detect_jump_to_node_fuzzy_match() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-nav-01", initial_topic="Sprint Planning"
    )
    redis_node = await orchestrator.branch_tangent(
        "Redis cluster failover", "usr-a"
    )

    detector = NavigationIntentDetector()
    decision = detector.detect(
        "going back to Redis cluster failover", orchestrator.nodes
    )

    assert decision.action == NavigationAction.JUMP_TO_NODE
    assert decision.target_node_id == redis_node.node_id
    assert decision.target_topic == "Redis cluster failover"

    token_match = detector.detect("let's revisit the failover bit", orchestrator.nodes)
    assert token_match.action == NavigationAction.JUMP_TO_NODE
    assert token_match.target_node_id == redis_node.node_id

    no_match = detector.detect("going back to quantum computing", orchestrator.nodes)
    assert no_match.action == NavigationAction.NONE


def test_detect_none_for_plain_speech() -> None:
    detector = NavigationIntentDetector()
    assert detector.detect("the weather is nice today").action == NavigationAction.NONE
    assert detector.detect("").action == NavigationAction.NONE


@pytest.mark.asyncio
async def test_pipeline_spoken_tangent_branches() -> None:
    state = SessionStateManager()
    pipeline = TurnPipeline(state_manager=state, lexicon_manager=LexiconManager())
    room = await state.create_room("room-nav-02", "Infra Review")

    result = await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-nav-02",
            raw_utterance="quick tangent on Redis cluster failover",
            speaker_id_hint="usr-a",
        )
    )

    assert result.branched_tangent is True
    assert room.current_node.topic_summary == "Redis cluster failover"
    assert room.current_node.parent_id == room.state.root_node_id


@pytest.mark.asyncio
async def test_pipeline_park_and_pop_marks_parked() -> None:
    state = SessionStateManager()
    pipeline = TurnPipeline(state_manager=state, lexicon_manager=LexiconManager())
    room = await state.create_room("room-nav-03", "Roadmap")
    tangent = await room.branch_tangent("Q3 Targets", "usr-a")

    await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-nav-03",
            raw_utterance="let's put a pin in this for now",
            speaker_id_hint="usr-a",
        )
    )

    assert tangent.status == NodeStatus.PARKED
    assert room.state.current_node_id == room.state.root_node_id


@pytest.mark.asyncio
async def test_pipeline_pop_to_anchor_marks_resolved() -> None:
    state = SessionStateManager()
    pipeline = TurnPipeline(state_manager=state, lexicon_manager=LexiconManager())
    room = await state.create_room("room-nav-04", "Budget")
    tangent = await room.branch_tangent("Marketing Spend", "usr-a")

    await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-nav-04",
            raw_utterance="anyway, back to the main topic",
            speaker_id_hint="usr-a",
        )
    )

    assert tangent.status == NodeStatus.RESOLVED
    assert room.state.current_node_id == room.state.root_node_id


@pytest.mark.asyncio
async def test_pipeline_spoken_jump_to_node() -> None:
    state = SessionStateManager()
    pipeline = TurnPipeline(state_manager=state, lexicon_manager=LexiconManager())
    room = await state.create_room("room-nav-05", "Infra Review")
    redis_node = await room.branch_tangent("Redis cluster failover", "usr-a")
    await room.jump_to_node(room.state.root_node_id)
    await room.branch_tangent("Billing Migration", "usr-b")

    result = await pipeline.process_turn(
        AudioTurnEvent(
            room_id="room-nav-05",
            raw_utterance="anyway, back to Redis cluster failover",
            speaker_id_hint="usr-b",
        )
    )

    assert room.state.current_node_id == redis_node.node_id
    assert result.active_node_id == redis_node.node_id
    assert redis_node.utterances[-1].text == "anyway, back to Redis cluster failover"
