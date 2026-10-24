from __future__ import annotations

import pytest

from core.interruption_classifier import InterruptionClassifier
from core.voice_biometrics import VoiceBiometricMatcher, cosine_similarity
from schemas.session_models import InterruptionType, VoiceProfile, VoiceTier


def _profile(
    user_id: str,
    embedding: list[float],
    tier: VoiceTier = VoiceTier.CONVERSATIONAL,
) -> VoiceProfile:
    return VoiceProfile(
        user_id=user_id,
        display_name=user_id,
        tier=tier,
        embedding_vector=embedding,
    )


def test_cosine_similarity_identical_orthogonal_opposite() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_magnitude_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0


def test_cosine_similarity_rejects_bad_vectors() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        cosine_similarity([], [1.0])
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [])


def test_match_speaker_selects_best_profile_above_threshold() -> None:
    matcher = VoiceBiometricMatcher(similarity_threshold=0.8)
    profiles = [
        _profile("usr-far", [0.0, 1.0]),
        _profile("usr-close", [0.9, 0.1]),
        _profile("usr-empty", []),
        _profile("usr-wrong-dim", [1.0, 1.0, 1.0]),
    ]

    result = matcher.match_speaker([1.0, 0.0], profiles)

    assert result is not None
    profile, score = result
    assert profile.user_id == "usr-close"
    assert score >= 0.8
    assert score == pytest.approx(cosine_similarity([1.0, 0.0], [0.9, 0.1]))


def test_match_speaker_below_threshold_returns_none() -> None:
    matcher = VoiceBiometricMatcher(similarity_threshold=0.95)
    profiles = [_profile("usr-partial", [0.6, 0.8])]

    assert matcher.match_speaker([1.0, 0.0], profiles) is None
    assert matcher.match_speaker([1.0], []) is None


def test_promote_tier_updates_profile() -> None:
    matcher = VoiceBiometricMatcher()
    profile = _profile("usr-temp", [0.1, 0.2], tier=VoiceTier.TEMPORARY)

    updated = matcher.promote_tier(profile, VoiceTier.PERMANENT)

    assert updated is profile
    assert updated.tier == VoiceTier.PERMANENT


def test_classifier_backchannel_phrases() -> None:
    classifier = InterruptionClassifier()

    for phrase in ("yeah", "Uh-huh!", "got it.", "mm-hmm", "I see", "okay, right"):
        assert classifier.classify(phrase) == InterruptionType.BACKCHANNEL, phrase


def test_classifier_hard_interrupt_triggers() -> None:
    classifier = InterruptionClassifier()

    for phrase in (
        "wait, what did you say?",
        "Stop.",
        "hold on a second",
        "actually, that's not right",
        "no, go back",
    ):
        assert classifier.classify(phrase) == InterruptionType.HARD_INTERRUPT, phrase


def test_classifier_background_event_squelch() -> None:
    classifier = InterruptionClassifier()

    assert (
        classifier.classify("yeah", vad_confidence=0.3)
        == InterruptionType.BACKGROUND_EVENT
    )
    assert (
        classifier.classify("yeah", rms_db=-50.0)
        == InterruptionType.BACKGROUND_EVENT
    )
    assert classifier.classify("[door slam]") == InterruptionType.BACKGROUND_EVENT
    assert classifier.classify("[cough]") == InterruptionType.BACKGROUND_EVENT
    assert classifier.classify("   ") == InterruptionType.BACKGROUND_EVENT
    assert (
        classifier.classify("[mic click] wait, stop deployment [noise]")
        == InterruptionType.HARD_INTERRUPT
    )


def test_classifier_multiword_questions_are_hard_interrupts() -> None:
    classifier = InterruptionClassifier()

    assert (
        classifier.classify("what do you think about the deployment schedule?")
        == InterruptionType.HARD_INTERRUPT
    )
    assert (
        classifier.classify("can we revisit the latency budget")
        == InterruptionType.HARD_INTERRUPT
    )
