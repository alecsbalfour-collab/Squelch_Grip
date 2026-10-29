from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.lexicon_manager import LexiconManager
from core.prompt_compiler import ContextPromptCompiler
from core.session_store import SessionSnapshotStore
from core.state_manager import SessionStateManager
from core.turn_pipeline import TurnPipeline
from schemas.arbitration_models import CrosstalkBatchEvent, CrosstalkBatchResult
from schemas.diagnostics_models import RoomDiagnosticsReport
from schemas.lexicon_models import LexiconPack
from schemas.prompt_models import CompiledContextPacket, ContextDeltaPacket
from schemas.session_models import RoomState, VoiceProfile
from schemas.snapshot_models import RoomSnapshot
from schemas.turn_models import AudioTurnEvent, TurnProcessingResult


class _CreateRoomRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str = Field(..., min_length=1)
    initial_topic: str = Field(..., min_length=1)


class _AiSpeakingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_ai_speaking: bool


def create_app() -> FastAPI:
    app = FastAPI(title="SquelchGrip Backend")

    state_manager = SessionStateManager()
    lexicon_manager = LexiconManager()
    turn_pipeline = TurnPipeline(
        state_manager=state_manager, lexicon_manager=lexicon_manager
    )
    prompt_compiler = ContextPromptCompiler()
    snapshot_store = SessionSnapshotStore()
    ws_connections: dict[str, list[WebSocket]] = {}

    async def _broadcast_turn(room_id: str, result: TurnProcessingResult) -> None:
        payload = result.model_dump(mode="json")
        for peer in list(ws_connections.get(room_id, [])):
            try:
                await peer.send_json(payload)
            except Exception:
                pass

    def _require_room(room_id: str):
        orchestrator = state_manager.get_room(room_id)
        if orchestrator is None:
            raise HTTPException(
                status_code=404, detail=f"Room '{room_id}' not found."
            )
        return orchestrator

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "service": "squelch_grip_backend",
            "version": "1.0.0a1",
        }

    @app.post("/rooms", response_model=RoomState)
    async def create_room(payload: _CreateRoomRequest) -> RoomState:
        orchestrator = await state_manager.create_room(
            payload.room_id, payload.initial_topic
        )
        return orchestrator.state

    @app.get("/rooms/{room_id}", response_model=RoomState)
    async def get_room(room_id: str) -> RoomState:
        return _require_room(room_id).state

    @app.post("/rooms/{room_id}/ai-speaking", response_model=RoomState)
    async def set_ai_speaking(
        room_id: str, payload: _AiSpeakingRequest
    ) -> RoomState:
        orchestrator = _require_room(room_id)
        orchestrator.state.is_ai_speaking = payload.is_ai_speaking
        return orchestrator.state

    @app.get("/rooms/{room_id}/prompt", response_model=CompiledContextPacket)
    async def get_prompt(room_id: str) -> CompiledContextPacket:
        return prompt_compiler.compile(
            _require_room(room_id), lexicon_manager
        )

    @app.get("/rooms/{room_id}/prompt/delta", response_model=ContextDeltaPacket)
    async def get_prompt_delta(
        room_id: str,
        previous_node_id: Optional[str] = None,
        since_timestamp: float = 0.0,
    ) -> ContextDeltaPacket:
        return prompt_compiler.compile_delta(
            _require_room(room_id),
            previous_node_id=previous_node_id,
            since_timestamp=since_timestamp,
        )

    @app.post("/rooms/{room_id}/prune")
    async def prune_room(room_id: str) -> dict:
        orchestrator = _require_room(room_id)
        pruned = await orchestrator.prune_resolved_branches()
        return {"room_id": room_id, "pruned_node_ids": pruned}

    @app.get("/rooms/{room_id}/diagnostics", response_model=RoomDiagnosticsReport)
    async def get_diagnostics(room_id: str) -> RoomDiagnosticsReport:
        try:
            return turn_pipeline.get_room_diagnostics(room_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0]))

    @app.post("/profiles", response_model=VoiceProfile)
    async def register_profile(profile: VoiceProfile) -> VoiceProfile:
        return await state_manager.register_voice_profile(profile)

    @app.post("/lexicons", response_model=LexiconPack)
    async def register_lexicon(pack: LexiconPack) -> LexiconPack:
        lexicon_manager.register_pack(pack)
        return pack

    @app.post("/rooms/{room_id}/lexicons/{pack_id}", response_model=RoomState)
    async def attach_lexicon(room_id: str, pack_id: str) -> RoomState:
        orchestrator = _require_room(room_id)
        if lexicon_manager.get_pack(pack_id) is None:
            raise HTTPException(
                status_code=404, detail=f"LexiconPack '{pack_id}' not found."
            )
        return lexicon_manager.attach_pack_to_room(orchestrator.state, pack_id)

    @app.post("/rooms/{room_id}/turns", response_model=TurnProcessingResult)
    async def process_turn(
        room_id: str, event: AudioTurnEvent
    ) -> TurnProcessingResult:
        event = event.model_copy(update={"room_id": room_id})
        try:
            result = await turn_pipeline.process_turn(event)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0]))
        await _broadcast_turn(room_id, result)
        return result

    @app.post("/rooms/{room_id}/crosstalk", response_model=CrosstalkBatchResult)
    async def process_crosstalk(
        room_id: str, batch: CrosstalkBatchEvent
    ) -> CrosstalkBatchResult:
        batch = batch.model_copy(update={"room_id": room_id})
        try:
            result = await turn_pipeline.process_crosstalk_batch(batch)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0]))
        if result.turn_result is not None:
            await _broadcast_turn(room_id, result.turn_result)
        return result

    @app.get("/rooms/{room_id}/snapshot", response_model=RoomSnapshot)
    async def get_snapshot(room_id: str) -> RoomSnapshot:
        return snapshot_store.export_room(_require_room(room_id))

    @app.post("/rooms/restore", response_model=RoomState)
    async def restore_room(snapshot: RoomSnapshot) -> RoomState:
        try:
            orchestrator = await snapshot_store.restore_room(
                state_manager, snapshot
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return orchestrator.state

    @app.websocket("/ws/rooms/{room_id}")
    async def room_websocket(websocket: WebSocket, room_id: str) -> None:
        if state_manager.get_room(room_id) is None:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        ws_connections.setdefault(room_id, []).append(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                if not isinstance(data, dict):
                    await websocket.send_json(
                        {"detail": "Payload must be a JSON object."}
                    )
                    continue
                data.setdefault("room_id", room_id)
                try:
                    event = AudioTurnEvent(**data)
                except ValidationError as exc:
                    await websocket.send_json({"detail": exc.errors()})
                    continue
                result = await turn_pipeline.process_turn(event)
                await _broadcast_turn(room_id, result)
        except WebSocketDisconnect:
            pass
        finally:
            peers = ws_connections.get(room_id, [])
            if websocket in peers:
                peers.remove(websocket)

    return app


app = create_app()
