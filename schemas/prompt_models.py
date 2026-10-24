from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CompiledContextPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str = Field(..., min_length=1)
    active_node_id: str = Field(..., min_length=1)
    active_topic: str = Field(..., min_length=1)
    lineage_breadcrumbs: list[str] = Field(default_factory=list)
    recent_utterances: list[str] = Field(default_factory=list)
    unresolved_tangent_topics: list[str] = Field(default_factory=list)
    domain_glossary: dict[str, str] = Field(default_factory=dict)
    system_prompt_markdown: str = Field(..., min_length=1)
