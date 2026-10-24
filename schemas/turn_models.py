from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.session_models import BargeInDirective, VoiceTier


class AudioTurnEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str = Field(..., min_length=1)
    raw_utterance: str
    speaker_id_hint: Optional[str] = None
    embedding_vector: list[float] = Field(default_factory=list)
    vad_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rms_db: float = Field(default=-20.0)
    new_tangent_topic: Optional[str] = None


class TurnProcessingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str = Field(..., min_length=1)
    resolved_speaker_id: str = Field(..., min_length=1)
    speaker_tier: VoiceTier
    normalized_utterance: str
    active_node_id: str = Field(..., min_length=1)
    barge_in_directive: Optional[BargeInDirective] = None
    branched_tangent: bool = False
