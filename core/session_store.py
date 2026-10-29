from __future__ import annotations

from pathlib import Path

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
        for key, node in snapshot.nodes.items():
            if key != node.node_id:
                raise ValueError(
                    f"Snapshot node key '{key}' does not match node_id '{node.node_id}'."
                )
            if node.parent_id is not None and node.parent_id not in snapshot.nodes:
                raise ValueError(
                    f"Node '{key}' references missing parent_id '{node.parent_id}'."
                )
            for child_id in node.child_nodes:
                if child_id not in snapshot.nodes:
                    raise ValueError(
                        f"Node '{key}' references missing child_id '{child_id}'."
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

    def save_to_disk(
        self, orchestrator: ContextGraphOrchestrator, directory: str | Path
    ) -> Path:
        snapshot = self.export_room(orchestrator)
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = (
            target_dir / f"{orchestrator.state.room_id}.snapshot.json"
        )
        file_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        return file_path

    async def load_from_disk(
        self, state_manager: SessionStateManager, file_path: str | Path
    ) -> ContextGraphOrchestrator:
        snapshot = RoomSnapshot.model_validate_json(
            Path(file_path).read_text(encoding="utf-8")
        )
        return await self.restore_room(state_manager, snapshot)
