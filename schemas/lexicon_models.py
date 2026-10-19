from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LexiconTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical: str = Field(..., min_length=1)
    aliases: list[str] = Field(default_factory=list)
    boost_weight: float = Field(default=1.5, ge=0.0, le=10.0)
    definition: Optional[str] = None


class LexiconPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pack_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    domain: str = Field(..., min_length=1)
    terms: list[LexiconTerm] = Field(default_factory=list)
