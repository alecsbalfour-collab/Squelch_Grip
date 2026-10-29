from __future__ import annotations

from core.context_graph import ContextGraphOrchestrator
from core.state_manager import SessionStateManager
from schemas.snapshot_models import RoomSnapshot


class SessionSnapshotStore:
    """Exports and restores full room context graphs for persistence and
    frontend graph visualizers."""

    def export_room(self, orchestrator: ContextGraphOrchestrator) -> RoomSnapshot:
        return RoomSnapshot(
            state=orchestrator.state.model_copy(deep=True),
            nodes={
                node_id: node.model_copy(deep=True)
                for node_id, node in orchestrator.nodes.items()
            },
        )

    async def restore_room(
        self,
        state_manager: SessionStateManager,
        snapshot: RoomSnapshot,
    ) -> ContextGraphOrchestrator:
        if snapshot.state.root_node_id not in snapshot.nodes:
            raise ValueError(
                f"Snapshot root_node_id '{snapshot.state.root_node_id}' "
                "is missing from nodes."
            )
        if snapshot.state.current_node_id not in snapshot.nodes:
            raise ValueError(
                f"Snapshot current_node_id '{snapshot.state.current_node_id}' "
                "is missing from nodes."
            )

        orchestrator = ContextGraphOrchestrator(
            room_id=snapshot.state.room_id,
            initial_topic=snapshot.nodes[snapshot.state.root_node_id].topic_summary,
        )
        orchestrator.state = snapshot.state.model_copy(deep=True)
        orchestrator.nodes = {
            node_id: node.model_copy(deep=True)
            for node_id, node in snapshot.nodes.items()
        }
        await state_manager.set_room(snapshot.state.room_id, orchestrator)
        return orchestrator
