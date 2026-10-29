from __future__ import annotations

import re
from typing import Optional

from schemas.lexicon_models import LexiconPack
from schemas.session_models import RoomState


class LexiconManager:
    """Registry for hot-swappable domain lexicon packs and alias normalization."""

    def __init__(self) -> None:
        self._packs: dict[str, LexiconPack] = {}

    def register_pack(self, pack: LexiconPack) -> None:
        self._packs[pack.pack_id] = pack

    def get_pack(self, pack_id: str) -> Optional[LexiconPack]:
        return self._packs.get(pack_id)

    def list_packs(self) -> list[LexiconPack]:
        return list(self._packs.values())

    def attach_pack_to_room(
        self, room_state: RoomState, pack_id: str
    ) -> RoomState:
        if pack_id not in self._packs:
            raise KeyError(f"LexiconPack '{pack_id}' is not registered.")
        if pack_id not in room_state.lexicon_pack_ids:
            room_state.lexicon_pack_ids.append(pack_id)
        return room_state

    def normalize_utterance(self, text: str, pack_ids: list[str]) -> str:
        """Replaces aliases with canonical terms, case-insensitive whole-word."""
        alias_map: dict[str, str] = {}
        for pack_id in pack_ids:
            pack = self._packs.get(pack_id)
            if pack is None:
                continue
            for term in pack.terms:
                for alias in term.aliases:
                    alias_map[alias.lower()] = term.canonical

        if not alias_map:
            return text

        pattern = re.compile(
            r"(?<!\w)(?:"
            + "|".join(
                re.escape(alias)
                for alias in sorted(alias_map, key=len, reverse=True)
            )
            + r")(?!\w)",
            re.IGNORECASE,
        )
        return pattern.sub(lambda match: alias_map[match.group(0).lower()], text)
