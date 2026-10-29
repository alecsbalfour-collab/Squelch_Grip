from __future__ import annotations

from core.turn_arbitrator import TurnArbitrator
from schemas.arbitration_models import CandidateSpeechFrame
from schemas.session_models import VoiceTier


def _frame(
    speaker_id: str,
    tier: VoiceTier = VoiceTier.CONVERSATIONAL,
    utterance: str = "let me add something here",
    vad: float = 1.0,
    rms: float = -15.0,
    ts: float = 1000.0,
) -> CandidateSpeechFrame:
    return CandidateSpeechFrame(
        speaker_id=speaker_id,
        tier=tier,
        utterance=utterance,
        vad_confidence=vad,
        rms_db=rms,
        timestamp=ts,
    )


def test_tier_precedence_overrides_signal_strength() -> None:
    arb = TurnArbitrator()
    outcome = arb.arbitrate(
        [
            _frame("usr-conv", VoiceTier.CONVERSATIONAL, rms=-10.0),
            _frame("usr-perm", VoiceTier.PERMANENT, rms=-35.0),
        ]
    )
    assert outcome.winning_speaker_id == "usr-perm"
    assert outcome.winning_frame is not None
    assert len(outcome.queued_frames) == 1
    assert outcome.queued_frames[0].speaker_id == "usr-conv"


def test_floor_holder_bonus_breaks_ties() -> None:
    arb = TurnArbitrator(floor_hold_window_sec=2.5)
    outcome = arb.arbitrate(
        [
            _frame("usr-challenger", rms=-15.0, ts=100.5),
            _frame("usr-floor", rms=-15.0, ts=101.0),
        ],
        current_floor_speaker_id="usr-floor",
        last_turn_timestamp=100.0,
    )
    assert outcome.winning_speaker_id == "usr-floor"


def test_expired_floor_window_falls_back_to_timestamp() -> None:
    arb = TurnArbitrator(floor_hold_window_sec=2.5)
    outcome = arb.arbitrate(
        [
            _frame("usr-challenger", rms=-15.0, ts=100.5),
            _frame("usr-floor", rms=-15.0, ts=101.0),
        ],
        current_floor_speaker_id="usr-floor",
        last_turn_timestamp=90.0,
    )
    assert outcome.winning_speaker_id == "usr-challenger"


def test_rms_db_tiebreak_within_same_tier() -> None:
    arb = TurnArbitrator()
    outcome = arb.arbitrate(
        [
            _frame("usr-quiet", rms=-25.0),
            _frame("usr-loud", rms=-12.0),
        ]
    )
    assert outcome.winning_speaker_id == "usr-loud"


def test_background_noise_frames_are_dropped() -> None:
    arb = TurnArbitrator()
    outcome = arb.arbitrate(
        [
            _frame("usr-speaker", utterance="wait, i disagree"),
            _frame("usr-hvac", utterance="", vad=0.1),
            _frame("usr-door", utterance="[door slam]"),
        ]
    )
    assert outcome.winning_speaker_id == "usr-speaker"
    assert sorted(outcome.dropped_speaker_ids) == ["usr-door", "usr-hvac"]
    assert outcome.queued_frames == []


def test_backchannel_only_first_wins_rest_queued() -> None:
    arb = TurnArbitrator()
    outcome = arb.arbitrate(
        [
            _frame("usr-a", utterance="yeah"),
            _frame("usr-b", utterance="mm-hmm"),
        ]
    )
    assert outcome.winning_speaker_id == "usr-a"
    assert [f.speaker_id for f in outcome.queued_frames] == ["usr-b"]


def test_backchannels_queue_behind_substantive_winner() -> None:
    arb = TurnArbitrator()
    outcome = arb.arbitrate(
        [
            _frame("usr-back", utterance="got it"),
            _frame("usr-main", utterance="actually, let's change course"),
        ]
    )
    assert outcome.winning_speaker_id == "usr-main"
    assert [f.speaker_id for f in outcome.queued_frames] == ["usr-back"]


def test_no_frames_returns_empty_outcome() -> None:
    arb = TurnArbitrator()
    outcome = arb.arbitrate([])
    assert outcome.winning_speaker_id is None
    assert outcome.winning_frame is None
    assert outcome.queued_frames == []
    assert outcome.dropped_speaker_ids == []

    noise_only = arb.arbitrate([_frame("usr-noise", vad=0.1)])
    assert noise_only.winning_speaker_id is None
    assert noise_only.dropped_speaker_ids == ["usr-noise"]
