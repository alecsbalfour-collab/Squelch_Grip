from __future__ import annotations

from typing import Optional

from core.interruption_classifier import InterruptionClassifier
from schemas.arbitration_models import ArbitrationOutcome, CandidateSpeechFrame
from schemas.session_models import InterruptionType, VoiceTier


class TurnArbitrator:
    """Resolves simultaneous overlapping speech frames to a single floor winner."""

    TIER_WEIGHTS: dict[VoiceTier, int] = {
        VoiceTier.PERMANENT: 3,
        VoiceTier.TEMPORARY: 2,
        VoiceTier.CONVERSATIONAL: 1,
    }

    def __init__(
        self,
        classifier: Optional[InterruptionClassifier] = None,
        floor_hold_window_sec: float = 2.5,
    ) -> None:
        self.classifier = classifier or InterruptionClassifier()
        self.floor_hold_window_sec = floor_hold_window_sec

    def arbitrate(
        self,
        frames: list[CandidateSpeechFrame],
        current_floor_speaker_id: Optional[str] = None,
        last_turn_timestamp: Optional[float] = None,
    ) -> ArbitrationOutcome:
        substantive: list[CandidateSpeechFrame] = []
        backchannels: list[CandidateSpeechFrame] = []
        dropped: list[str] = []

        for frame in frames:
            interruption_type = self.classifier.classify(
                frame.utterance,
                vad_confidence=frame.vad_confidence,
                rms_db=frame.rms_db,
            )
            if interruption_type == InterruptionType.BACKGROUND_EVENT:
                dropped.append(frame.speaker_id)
            elif interruption_type == InterruptionType.BACKCHANNEL:
                backchannels.append(frame)
            else:
                substantive.append(frame)

        if not substantive:
            if backchannels:
                winner = backchannels[0]
                return ArbitrationOutcome(
                    winning_speaker_id=winner.speaker_id,
                    winning_frame=winner,
                    queued_frames=backchannels[1:],
                    dropped_speaker_ids=dropped,
                )
            return ArbitrationOutcome(dropped_speaker_ids=dropped)

        def _score(frame: CandidateSpeechFrame) -> tuple[int, int, float, float]:
            floor_bonus = (
                1
                if frame.speaker_id == current_floor_speaker_id
                and last_turn_timestamp is not None
                and 0.0
                <= (frame.timestamp - last_turn_timestamp)
                <= self.floor_hold_window_sec
                else 0
            )
            return (
                self.TIER_WEIGHTS[frame.tier],
                floor_bonus,
                frame.rms_db,
                -frame.timestamp,
            )

        ranked = sorted(substantive, key=_score, reverse=True)
        winner = ranked[0]
        return ArbitrationOutcome(
            winning_speaker_id=winner.speaker_id,
            winning_frame=winner,
            queued_frames=ranked[1:] + backchannels,
            dropped_speaker_ids=dropped,
        )
