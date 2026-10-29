from __future__ import annotations

import pytest

from core.context_graph import ContextGraphOrchestrator
from core.lexicon_manager import LexiconManager
from core.prompt_compiler import ContextPromptCompiler
from schemas.lexicon_models import LexiconPack, LexiconTerm
from schemas.session_models import NodeStatus


@pytest.mark.asyncio
async def test_lineage_breadcrumbs_across_nested_tangents() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-pc-01", initial_topic="Root Architecture"
    )
    await orchestrator.branch_tangent("Database Sharding", "usr-a")
    await orchestrator.branch_tangent("Consistent Hashing", "usr-b")

    packet = ContextPromptCompiler().compile(orchestrator)

    assert packet.room_id == "room-pc-01"
    assert packet.active_topic == "Consistent Hashing"
    assert packet.active_node_id == orchestrator.current_node.node_id
    assert packet.lineage_breadcrumbs == [
        "Root Architecture",
        "Database Sharding",
        "Consistent Hashing",
    ]


@pytest.mark.asyncio
async def test_recent_utterances_windowing() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-pc-02", initial_topic="Retrospective"
    )
    for i in range(5):
        await orchestrator.record_utterance("usr-x", f"message-{i}")

    packet = ContextPromptCompiler(max_recent_utterances=3).compile(orchestrator)

    assert packet.recent_utterances == [
        "usr-x: message-2",
        "usr-x: message-3",
        "usr-x: message-4",
    ]


@pytest.mark.asyncio
async def test_unresolved_tangents_exclude_active_node() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-pc-03", initial_topic="Root"
    )
    tangent_a = await orchestrator.branch_tangent("Tangent A", "usr-a")
    await orchestrator.jump_to_node(orchestrator.state.root_node_id)
    tangent_b = await orchestrator.branch_tangent("Tangent B", "usr-b")
    await orchestrator.set_node_status(NodeStatus.PARKED, tangent_a.node_id)

    packet = ContextPromptCompiler().compile(orchestrator)

    assert packet.active_node_id == tangent_b.node_id
    assert len(packet.unresolved_tangent_topics) == 1
    assert packet.unresolved_tangent_topics[0] == (
        f"[PARKED] Tangent A (node={tangent_a.node_id})"
    )
    assert all("Tangent B" not in t for t in packet.unresolved_tangent_topics)


@pytest.mark.asyncio
async def test_domain_glossary_collects_defined_terms() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-pc-04", initial_topic="Rounds"
    )
    lexicon = LexiconManager()
    lexicon.register_pack(
        LexiconPack(
            pack_id="lex-med-01",
            name="Clinical",
            domain="medical",
            terms=[
                LexiconTerm(
                    canonical="hypertension",
                    aliases=["high blood pressure"],
                    definition="Chronically elevated arterial blood pressure.",
                ),
                LexiconTerm(canonical="acetaminophen", aliases=["tylenol"]),
            ],
        )
    )
    lexicon.attach_pack_to_room(orchestrator.state, "lex-med-01")

    packet = ContextPromptCompiler().compile(orchestrator, lexicon)

    assert packet.domain_glossary == {
        "hypertension": "Chronically elevated arterial blood pressure."
    }
    assert "acetaminophen" not in packet.domain_glossary


@pytest.mark.asyncio
async def test_system_prompt_markdown_renders_all_sections() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-pc-05", initial_topic="Root Topic"
    )
    tangent = await orchestrator.branch_tangent("Side Quest", "usr-a")
    await orchestrator.record_utterance("usr-a", "first point")

    packet = ContextPromptCompiler().compile(orchestrator)
    markdown = packet.system_prompt_markdown

    assert "# Active Topic" in markdown
    assert "Side Quest" in markdown
    assert "## Lineage Breadcrumbs" in markdown
    assert "1. Root Topic" in markdown
    assert "## Recent Transcript" in markdown
    assert "- usr-a: first point" in markdown
    assert "## Unresolved / Parked Tangents" in markdown
    assert "- (none)" in markdown
    assert "## Domain Glossary" in markdown
    assert tangent.node_id == packet.active_node_id


@pytest.mark.asyncio
async def test_compile_delta_topic_switch_and_breadcrumbs() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-delta-01", initial_topic="Root Topic"
    )
    tangent = await orchestrator.branch_tangent("Side Topic", "usr-a")
    root_id = orchestrator.state.root_node_id

    compiler = ContextPromptCompiler()

    switched = compiler.compile_delta(orchestrator, previous_node_id=root_id)
    assert switched.topic_switched is True
    assert switched.active_node_id == tangent.node_id
    assert switched.lineage_breadcrumbs == ["Root Topic", "Side Topic"]

    same = compiler.compile_delta(
        orchestrator, previous_node_id=tangent.node_id
    )
    assert same.topic_switched is False
    assert same.lineage_breadcrumbs == []

    initial = compiler.compile_delta(orchestrator, previous_node_id=None)
    assert initial.topic_switched is False
    assert initial.lineage_breadcrumbs == ["Root Topic", "Side Topic"]


@pytest.mark.asyncio
async def test_compile_delta_utterance_window_and_tangents() -> None:
    orchestrator = ContextGraphOrchestrator(
        room_id="room-delta-02", initial_topic="Root Topic"
    )
    tangent = await orchestrator.branch_tangent("Active Side", "usr-a")
    await orchestrator.jump_to_node(orchestrator.state.root_node_id)
    sibling = await orchestrator.branch_tangent("Sibling Park", "usr-b")
    await orchestrator.set_node_status(NodeStatus.PARKED, sibling.node_id)
    await orchestrator.jump_to_node(tangent.node_id)

    await orchestrator.record_utterance("usr-a", "old message")
    await orchestrator.record_utterance("usr-a", "new message")
    orchestrator.current_node.utterances[0].timestamp = 100.0
    orchestrator.current_node.utterances[1].timestamp = 200.0

    packet = ContextPromptCompiler().compile_delta(
        orchestrator,
        previous_node_id=tangent.node_id,
        since_timestamp=150.0,
    )

    assert packet.delta_utterances == ["usr-a: new message"]
    assert packet.unresolved_tangent_topics == [
        f"[PARKED] Sibling Park (node={sibling.node_id})"
    ]
    assert "# Context Delta" in packet.delta_markdown
    assert "Topic unchanged" in packet.delta_markdown
    assert "- usr-a: new message" in packet.delta_markdown
