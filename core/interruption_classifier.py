from __future__ import annotations

import re

from schemas.session_models import InterruptionType

BACKCHANNEL_PHRASES: frozenset[str] = frozenset(
    {
        "uh-huh",
        "uh huh",
        "yeah",
        "yep",
        "got it",
        "right",
        "mm-hmm",
        "mm hmm",
        "okay",
        "sure",
        "i see",
    }
)

HARD_INTERRUPT_TRIGGERS: frozenset[str] = frozenset(
    {
        "wait",
        "stop",
        "hold on",
        "hang on",
        "actually",
        "pause",
        "no",
        "listen",
    }
)

_BACKCHANNEL_TOKENS: frozenset[str] = frozenset(
    token for phrase in BACKCHANNEL_PHRASES for token in phrase.split()
)

_BRACKET_TAG_RE = re.compile(r"\[[^\]]*\]")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


class InterruptionClassifier:
    """Low-latency heuristic classifier for overlapping incoming speech."""

    def classify(
        self,
        utterance: str,
        vad_confidence: float = 1.0,
        rms_db: float = -20.0,
    ) -> InterruptionType:
        raw = utterance.strip()

        if (
            vad_confidence < 0.45
            or rms_db < -42.0
            or not raw
            or (raw.startswith("[") and raw.endswith("]"))
        ):
            return InterruptionType.BACKGROUND_EVENT

        normalized = self._normalize(raw)
        if not normalized:
            return InterruptionType.BACKGROUND_EVENT

        for trigger in HARD_INTERRUPT_TRIGGERS:
            if normalized == trigger or normalized.startswith(trigger + " "):
                return InterruptionType.HARD_INTERRUPT

        if normalized in BACKCHANNEL_PHRASES:
            return InterruptionType.BACKCHANNEL

        tokens = normalized.split()
        if len(tokens) <= 3 and all(
            token in _BACKCHANNEL_TOKENS for token in tokens
        ):
            return InterruptionType.BACKCHANNEL

        return InterruptionType.HARD_INTERRUPT

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercases and strips bracketed tags, hyphens, and punctuation."""
        without_tags = _BRACKET_TAG_RE.sub(" ", text.lower())
        without_hyphens = without_tags.replace("-", " ")
        without_punct = _NON_WORD_RE.sub("", without_hyphens)
        return _WHITESPACE_RE.sub(" ", without_punct).strip()
