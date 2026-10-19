from __future__ import annotations

import asyncio
from typing import Optional

from core.context_graph import ContextGraphOrchestrator
from schemas.session_models import RoomState, VoiceProfile, VoiceTier


class SessionStateManager:
    """Registry for active room orchestrators and enrolled speaker voice profiles."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._rooms: dict[str, ContextGraphOrchestrator] = {}
        self._voice_profiles: dict[str, VoiceProfile] = {}

    async def create_room(
        self, room_id: str, initial_topic: str
    ) -> ContextGraphOrchestrator:
        async with self._lock:
            orchestrator = ContextGraphOrchestrator(
                room_id=room_id,
                initial_topic=initial_topic,
            )
            self._rooms[room_id] = orchestrator
            return orchestrator

    def get_room(self, room_id: str) -> Optional[ContextGraphOrchestrator]:
        return self._rooms.get(room_id)

    def get_room_state(self, room_id: str) -> Optional[RoomState]:
        orchestrator = self._rooms.get(room_id)
        return orchestrator.state if orchestrator is not None else None

    async def remove_room(self, room_id: str) -> bool:
        async with self._lock:
            return self._rooms.pop(room_id, None) is not None

    async def register_voice_profile(self, profile: VoiceProfile) -> VoiceProfile:
        async with self._lock:
            self._voice_profiles[profile.user_id] = profile
            return profile

    def get_voice_profile(self, user_id: str) -> Optional[VoiceProfile]:
        return self._voice_profiles.get(user_id)

    async def purge_non_permanent_profiles(self) -> int:
        async with self._lock:
            to_remove = [
                uid
                for uid, profile in self._voice_profiles.items()
                if profile.tier != VoiceTier.PERMANENT
            ]
            for uid in to_remove:
                del self._voice_profiles[uid]
            return len(to_remove)
