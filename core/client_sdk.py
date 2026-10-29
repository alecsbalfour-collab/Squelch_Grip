from __future__ import annotations

from typing import Any, Optional

import httpx

from schemas.prompt_models import CompiledContextPacket, ContextDeltaPacket
from schemas.session_models import RoomState
from schemas.turn_models import AudioTurnEvent, TurnProcessingResult


class SquelchGripClient:
    """Async client bridge so f1ndr/dealr can call a remote SquelchGrip
    server or an in-process ASGI app over one interface."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        app: Optional[Any] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = base_url
        self.app = app
        self.api_key = api_key

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=(
                httpx.ASGITransport(app=self.app)
                if self.app is not None
                else None
            ),
            base_url=self.base_url,
            headers=(
                {"X-API-Key": self.api_key} if self.api_key else None
            ),
        )

    async def create_f1ndr_room(
        self, room_id: str, initial_topic: str
    ) -> RoomState:
        return await self._create_room(room_id, initial_topic, ["lex-f1ndr"])

    async def create_dealr_room(
        self, room_id: str, initial_topic: str
    ) -> RoomState:
        return await self._create_room(room_id, initial_topic, ["lex-dealr"])

    async def send_turn(self, event: AudioTurnEvent) -> TurnProcessingResult:
        async with self._client() as client:
            response = await client.post(
                f"/rooms/{event.room_id}/turns",
                json=event.model_dump(mode="json"),
            )
            response.raise_for_status()
            return TurnProcessingResult(**response.json())

    async def get_prompt(self, room_id: str) -> CompiledContextPacket:
        async with self._client() as client:
            response = await client.get(f"/rooms/{room_id}/prompt")
            response.raise_for_status()
            return CompiledContextPacket(**response.json())

    async def get_prompt_delta(
        self,
        room_id: str,
        previous_node_id: Optional[str] = None,
        since_timestamp: float = 0.0,
    ) -> ContextDeltaPacket:
        params: dict[str, Any] = {"since_timestamp": since_timestamp}
        if previous_node_id is not None:
            params["previous_node_id"] = previous_node_id
        async with self._client() as client:
            response = await client.get(
                f"/rooms/{room_id}/prompt/delta",
                params=params,
            )
            response.raise_for_status()
            return ContextDeltaPacket(**response.json())

    async def _create_room(
        self, room_id: str, initial_topic: str, pack_ids: list[str]
    ) -> RoomState:
        async with self._client() as client:
            response = await client.post(
                "/rooms",
                json={
                    "room_id": room_id,
                    "initial_topic": initial_topic,
                    "lexicon_pack_ids": pack_ids,
                },
            )
            response.raise_for_status()
            return RoomState(**response.json())
