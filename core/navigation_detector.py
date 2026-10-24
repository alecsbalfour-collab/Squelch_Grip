from __future__ import annotations

import re
from typing import Optional

from schemas.navigation_models import NavigationAction, NavigationDecision
from schemas.session_models import ContextNode

_PARK_PATTERNS: tuple[str, ...] = (
    r"\bpark\s+(?:this|that|it)\b",
    r"\btable\s+(?:this|that|it)\b",
    r"\bput\s+a\s+pin\s+in\s+(?:this|that|it)\b",
    r"\bput\s+(?:this|that|it)\s+on\s+hold\b",
)

_POP_PATTERNS: tuple[str, ...] = (
    r"\bback\s+to\s+the\s+(?:main|original|previous|parent)\s+(?:topic|point|thread|anchor)\b",
    r"\bback\s+to\s+what\s+we\s+were\s+(?:saying|discussing|talking\s+about)\b",
    r"\banyway[,\s]+back\s+to\s+(?:that|it|this|where\s+we\s+were)\b",
    r"\banyway[,\s]+back\s+to\s*$",
    r"\breturn\s+to\s+the\s+anchor\b",
    r"\bthat\s+resolves\s+that\b",
)

_JUMP_PATTERNS: tuple[str, ...] = (
    r"\b(?:going|go)\s+back\s+to\s+(.+?)\s*$",
    r"\brevisit(?:ing)?\s+(.+?)\s*$",
    r"\bback\s+to\s+(.+?)\s*$",
)

_BRANCH_PATTERNS: tuple[str, ...] = (
    r"\btangent\s+(?:on|about)\s+(.+?)\s*$",
    r"\bsidebar\s+(?:on|about)\s+(.+?)\s*$",
    r"\bside\s*note\s+(?:on|about)\s+(.+?)\s*$",
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "to", "of", "on", "in", "for", "and", "or",
        "we", "were", "was", "is", "it", "this", "that", "what", "our",
        "your", "back", "going", "go", "let", "lets", "us", "about",
    }
)

_TRAILING_PUNCT = ".,!?;:'\""


class NavigationIntentDetector:
    """Detects spoken conversational navigation cues (branch, park, pop, jump)."""

    def detect(
        self,
        utterance: str,
        nodes: Optional[dict[str, ContextNode]] = None,
    ) -> NavigationDecision:
        text = utterance.strip()
        if not text:
            return NavigationDecision(action=NavigationAction.NONE)

        for pattern in _PARK_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return NavigationDecision(action=NavigationAction.PARK_AND_POP)

        for pattern in _POP_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return NavigationDecision(
                    action=NavigationAction.POP_TO_ANCHOR,
                    mark_resolved=True,
                )

        if nodes:
            for pattern in _JUMP_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE)
                if match is None:
                    continue
                matched_node = self._match_node(
                    self._clean_topic(match.group(1)), nodes
                )
                if matched_node is not None:
                    return NavigationDecision(
                        action=NavigationAction.JUMP_TO_NODE,
                        target_topic=matched_node.topic_summary,
                        target_node_id=matched_node.node_id,
                    )

        for pattern in _BRANCH_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match is not None:
                topic = self._clean_topic(match.group(1))
                if topic:
                    return NavigationDecision(
                        action=NavigationAction.BRANCH_TANGENT,
                        target_topic=topic,
                    )

        return NavigationDecision(action=NavigationAction.NONE)

    def _match_node(
        self, query: str, nodes: dict[str, ContextNode]
    ) -> Optional[ContextNode]:
        """Substring or >=1 non-stopword token overlap against topic_summary."""
        query_lower = query.lower()
        query_tokens = self._content_tokens(query_lower)
        if not query_lower:
            return None

        best_node: Optional[ContextNode] = None
        best_score = 0
        for node in nodes.values():
            topic_lower = node.topic_summary.lower()
            score = len(query_tokens & self._content_tokens(topic_lower))
            if query_lower in topic_lower:
                score += 100
            if score > best_score:
                best_score = score
                best_node = node
        return best_node

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS

    @staticmethod
    def _clean_topic(raw: str) -> str:
        return raw.strip().rstrip(_TRAILING_PUNCT).strip()
