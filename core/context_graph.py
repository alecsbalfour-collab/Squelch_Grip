from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from schemas.session_models import (
    AudioCommandType,
    BargeInDirective,
    ContextNode,
    InterruptionType,
    NodeStatus,
    RoomState,
    Utterance,
)


class ContextGraphOrchestrator:
    """Manages non-linear conversational topic trees and barge-in state for a room."""

    def __init__(self, room_id: str, initial_topic: str) -> None:
        self._lock = asyncio.Lock()
        root_id = str(uuid.uuid4())
        root_node = ContextNode(
            node_id=root_id,
            parent_id=None,
            topic_summary=initial_topic,
            active_speaker_id=None,
            status=NodeStatus.OPEN,
            child_nodes=[],
            utterances=[],
            metadata={"is_root": True},
            timestamp=time.time(),
        )
        self.nodes: dict[str, ContextNode] = {root_id: root_node}
        self.state = RoomState(
            room_id=room_id,
            root_node_id=root_id,
            current_node_id=root_id,
            active_speakers=[],
            is_ai_speaking=False,
            lexicon_pack_ids=[],
        )

    @property
    def room_state(self) -> RoomState:
        return self.state

    @property
    def root_node(self) -> ContextNode:
        return self.nodes[self.state.root_node_id]

    @property
    def current_node(self) -> ContextNode:
        return self.nodes[self.state.current_node_id]

    def get_node(self, node_id: str) -> Optional[ContextNode]:
        return self.nodes.get(node_id)

    def get_active_lineage(self) -> list[ContextNode]:
        """Returns the ancestral path from root_node down to current_node."""
        lineage: list[ContextNode] = []
        visited: set[str] = set()
        cursor: Optional[ContextNode] = self.current_node

        while cursor is not None and cursor.node_id not in visited:
            visited.add(cursor.node_id)
            lineage.append(cursor)
            cursor = (
                self.nodes.get(cursor.parent_id)
                if cursor.parent_id is not None
                else None
            )

        lineage.reverse()
        return lineage

    def get_unresolved_tangents(self) -> list[ContextNode]:
        """Returns all non-root tangent nodes that are still OPEN or PARKED."""
        return [
            node
            for node_id, node in self.nodes.items()
            if node_id != self.state.root_node_id
            and node.status in (NodeStatus.OPEN, NodeStatus.PARKED)
        ]

    async def jump_to_node(self, node_id: str) -> ContextNode:
        """Non-linearly switches the active conversational focus to an existing node."""
        async with self._lock:
            target = self.nodes.get(node_id)
            if target is None:
                raise KeyError(f"ContextNode '{node_id}' does not exist in room '{self.state.room_id}'.")
            self.state.current_node_id = node_id
            return target

    async def set_node_status(
        self, status: NodeStatus, node_id: Optional[str] = None
    ) -> ContextNode:
        """Updates the lifecycle status (OPEN, RESOLVED, PARKED) of a node."""
        async with self._lock:
            target_id = node_id or self.state.current_node_id
            target = self.nodes.get(target_id)
            if target is None:
                raise KeyError(f"ContextNode '{target_id}' does not exist in room '{self.state.room_id}'.")
            target.status = status
            return target

    async def record_utterance(
        self,
        speaker_id: str,
        text: str,
        interruption_type: Optional[InterruptionType] = None,
    ) -> Utterance:
        """Appends a speaker utterance to the currently active ContextNode."""
        async with self._lock:
            entry = Utterance(
                speaker_id=speaker_id,
                text=text,
                interruption_type=interruption_type,
            )
            self.current_node.utterances.append(entry)
            self.current_node.active_speaker_id = speaker_id
            if speaker_id not in self.state.active_speakers:
                self.state.active_speakers.append(speaker_id)
            return entry

    async def branch_tangent(self, new_topic: str, speaker_id: str) -> ContextNode:
        async with self._lock:
            parent_node = self.current_node
            child_id = str(uuid.uuid4())
            child_node = ContextNode(
                node_id=child_id,
                parent_id=parent_node.node_id,
                topic_summary=new_topic,
                active_speaker_id=speaker_id,
                status=NodeStatus.OPEN,
                child_nodes=[],
                utterances=[],
                metadata={},
                timestamp=time.time(),
            )
            parent_node.child_nodes.append(child_id)
            self.nodes[child_id] = child_node
            self.state.current_node_id = child_id

            if speaker_id not in self.state.active_speakers:
                self.state.active_speakers.append(speaker_id)

            return child_node

    async def pop_to_anchor(self, mark_resolved: bool = False) -> ContextNode:
        async with self._lock:
            current = self.current_node
            if current.parent_id is not None and current.parent_id in self.nodes:
                if mark_resolved:
                    current.status = NodeStatus.RESOLVED
                self.state.current_node_id = current.parent_id
                return self.nodes[current.parent_id]

            self.state.current_node_id = self.state.root_node_id
            return self.root_node

    async def handle_barge_in(
        self,
        speaker_id: str,
        utterance: str,
        interruption_type: InterruptionType = InterruptionType.HARD_INTERRUPT,
    ) -> BargeInDirective:
        async with self._lock:
            if interruption_type == InterruptionType.HARD_INTERRUPT:
                self.state.is_ai_speaking = False
                if speaker_id not in self.state.active_speakers:
                    self.state.active_speakers.append(speaker_id)
                if utterance.strip():
                    self.current_node.utterances.append(
                        Utterance(
                            speaker_id=speaker_id,
                            text=utterance,
                            interruption_type=InterruptionType.HARD_INTERRUPT,
                        )
                    )
                    self.current_node.active_speaker_id = speaker_id
                command = AudioCommandType.HALT
                action = "HALT_AUDIO"

            elif interruption_type == InterruptionType.BACKCHANNEL:
                if speaker_id not in self.state.active_speakers:
                    self.state.active_speakers.append(speaker_id)
                if utterance.strip():
                    self.current_node.utterances.append(
                        Utterance(
                            speaker_id=speaker_id,
                            text=utterance,
                            interruption_type=InterruptionType.BACKCHANNEL,
                        )
                    )
                command = AudioCommandType.CONTINUE
                action = "LOG_BACKCHANNEL"

            else:
                command = AudioCommandType.IGNORE
                action = "SQUELCH_BACKGROUND"

            return BargeInDirective(
                command=command,
                action=action,
                room_id=self.state.room_id,
                speaker_id=speaker_id,
                utterance=utterance,
                interruption_type=interruption_type,
                current_node_id=self.state.current_node_id,
                is_ai_speaking=self.state.is_ai_speaking,
                timestamp=time.time(),
            )
