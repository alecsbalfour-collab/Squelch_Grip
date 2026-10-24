from __future__ import annotations

import math
from typing import Optional

from schemas.session_models import VoiceProfile, VoiceTier


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Computes the cosine similarity between two equal-length vectors."""
    if not vec_a or not vec_b:
        raise ValueError("cosine_similarity requires non-empty vectors.")
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}."
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    score = dot_product / (magnitude_a * magnitude_b)
    return max(-1.0, min(1.0, score))


class VoiceBiometricMatcher:
    """Cosine-similarity speaker matcher over enrolled VoiceProfile embeddings."""

    def __init__(self, similarity_threshold: float = 0.82) -> None:
        self.similarity_threshold = similarity_threshold

    def match_speaker(
        self,
        candidate_embedding: list[float],
        profiles: list[VoiceProfile],
    ) -> Optional[tuple[VoiceProfile, float]]:
        """Returns the best-matching (profile, score) at or above the threshold."""
        best_profile: Optional[VoiceProfile] = None
        best_score = float("-inf")

        for profile in profiles:
            embedding = profile.embedding_vector
            if not embedding or len(embedding) != len(candidate_embedding):
                continue
            score = cosine_similarity(candidate_embedding, embedding)
            if score > best_score:
                best_score = score
                best_profile = profile

        if best_profile is None or best_score < self.similarity_threshold:
            return None
        return best_profile, best_score

    def promote_tier(
        self, profile: VoiceProfile, target_tier: VoiceTier
    ) -> VoiceProfile:
        profile.tier = target_tier
        return profile
