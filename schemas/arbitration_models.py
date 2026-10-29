from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.session_models import VoiceTier


class CandidateSpeechFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_id: str = Field(..., min_length=1)
    tier: VoiceTier
    utterance: str
    vad_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rms_db: float = Field(default=-20.0)
    timestamp: float = Field(default_factory=time.time)


class ArbitrationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winning_speaker_id: Optional[str] = None
    winning_frame: Optional[CandidateSpeechFrame] = None
    queued_frames: list[CandidateSpeechFrame] = Field(default_factory=list)
    dropped_speaker_ids: list[str] = Field(default_factory=list)
