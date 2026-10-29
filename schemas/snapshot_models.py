from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from schemas.session_models import ContextNode, RoomState


class RoomSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: RoomState
    nodes: dict[str, ContextNode]
    exported_at: float = Field(default_factory=time.time)
