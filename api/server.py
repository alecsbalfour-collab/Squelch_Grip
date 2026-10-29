from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.lexicon_manager import LexiconManager
from core.prompt_compiler import ContextPromptCompiler
from core.state_manager import SessionStateManager
from core.turn_pipeline import TurnPipeline
from schemas.lexicon_models import LexiconPack
from schemas.prompt_models import CompiledContextPacket
from schemas.session_models import RoomState, VoiceProfile
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
            return await turn_pipeline.process_turn(event)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0]))

    @app.websocket("/ws/rooms/{room_id}")
    async def room_websocket(websocket: WebSocket, room_id: str) -> None:
        if state_manager.get_room(room_id) is None:
            await websocket.close(code=1008)
            return
        await websocket.accept()
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
                await websocket.send_json(result.model_dump(mode="json"))
        except WebSocketDisconnect:
            return

    return app


app = create_app()
