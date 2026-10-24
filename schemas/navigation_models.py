from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class NavigationAction(str, Enum):
    NONE = "NONE"
    BRANCH_TANGENT = "BRANCH_TANGENT"
    POP_TO_ANCHOR = "POP_TO_ANCHOR"
    PARK_AND_POP = "PARK_AND_POP"
    JUMP_TO_NODE = "JUMP_TO_NODE"


class NavigationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: NavigationAction = NavigationAction.NONE
    target_topic: Optional[str] = None
    target_node_id: Optional[str] = None
    mark_resolved: bool = False
