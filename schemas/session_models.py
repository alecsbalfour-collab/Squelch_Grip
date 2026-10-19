from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class VoiceTier(str, Enum):
    PERMANENT = "PERMANENT"
    TEMPORARY = "TEMPORARY"
    CONVERSATIONAL = "CONVERSATIONAL"


class InterruptionType(str, Enum):
    BACKCHANNEL = "BACKCHANNEL"
    HARD_INTERRUPT = "HARD_INTERRUPT"
    BACKGROUND_EVENT = "BACKGROUND_EVENT"


class NodeStatus(str, Enum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    PARKED = "PARKED"


class AudioCommandType(str, Enum):
    HALT = "HALT"
    CONTINUE = "CONTINUE"
    IGNORE = "IGNORE"


class Utterance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    speaker_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    interruption_type: Optional[InterruptionType] = None
    timestamp: float = Field(default_factory=time.time)


class BargeInDirective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: AudioCommandType
    action: str = Field(..., min_length=1)
    room_id: str = Field(..., min_length=1)
    speaker_id: str = Field(..., min_length=1)
    utterance: str
    interruption_type: InterruptionType
    current_node_id: str = Field(..., min_length=1)
    is_ai_speaking: bool
    timestamp: float = Field(default_factory=time.time)

    def __getitem__(self, key: str) -> Any:
        val = getattr(self, key)
        return val.value if isinstance(val, Enum) else val

    def get(self, key: str, default: Any = None) -> Any:
        if not hasattr(self, key):
            return default
        return self[key]


class VoiceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    tier: VoiceTier
    embedding_vector: list[float] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class ContextNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    topic_summary: str = Field(..., min_length=1)
    active_speaker_id: Optional[str] = None
    status: NodeStatus = NodeStatus.OPEN
    child_nodes: list[str] = Field(default_factory=list)
    utterances: list[Utterance] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class RoomState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str = Field(..., min_length=1)
    root_node_id: str = Field(..., min_length=1)
    current_node_id: str = Field(..., min_length=1)
    active_speakers: list[str] = Field(default_factory=list)
    is_ai_speaking: bool = False
    lexicon_pack_ids: list[str] = Field(default_factory=list)
