from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from schemas.arbitration_models import ArbitrationOutcome
from schemas.navigation_models import NavigationDecision
from schemas.session_models import BargeInDirective, VoiceTier


class RoomDiagnosticsReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str = Field(..., min_length=1)
    active_node_id: str = Field(..., min_length=1)
    active_topic: str = Field(..., min_length=1)
    total_node_count: int
    unresolved_tangent_ids: list[str] = Field(default_factory=list)
    speaker_tiers: dict[str, VoiceTier] = Field(default_factory=dict)
    active_lexicon_packs: list[str] = Field(default_factory=list)
    recent_navigation_decisions: list[NavigationDecision] = Field(
        default_factory=list
    )
    recent_arbitration_outcomes: list[ArbitrationOutcome] = Field(
        default_factory=list
    )
    recent_barge_in_directives: list[BargeInDirective] = Field(
        default_factory=list
    )
