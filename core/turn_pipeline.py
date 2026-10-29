from __future__ import annotations

import uuid
from collections import deque
from typing import Optional

from core.context_graph import ContextGraphOrchestrator
from core.interruption_classifier import InterruptionClassifier
from core.lexicon_manager import LexiconManager
from core.navigation_detector import NavigationIntentDetector
from core.state_manager import SessionStateManager
from core.turn_arbitrator import TurnArbitrator
from core.voice_biometrics import VoiceBiometricMatcher
from schemas.arbitration_models import (
    ArbitrationOutcome,
    CrosstalkBatchEvent,
    CrosstalkBatchResult,
)
from schemas.diagnostics_models import RoomDiagnosticsReport
from schemas.navigation_models import NavigationAction, NavigationDecision
from schemas.session_models import (
    BargeInDirective,
    InterruptionType,
    NodeStatus,
    VoiceProfile,
    VoiceTier,
)
from schemas.turn_models import AudioTurnEvent, TurnProcessingResult


class TurnPipeline:
    """End-to-end async pipeline unifying speaker resolution, lexicon
    normalization, and barge-in vs. normal turn routing."""

    def __init__(
        self,
        state_manager: SessionStateManager,
        lexicon_manager: LexiconManager,
        biometric_matcher: Optional[VoiceBiometricMatcher] = None,
        interruption_classifier: Optional[InterruptionClassifier] = None,
        navigation_detector: Optional[NavigationIntentDetector] = None,
        turn_arbitrator: Optional[TurnArbitrator] = None,
    ) -> None:
        self.state_manager = state_manager
        self.lexicon_manager = lexicon_manager
        self.biometric_matcher = biometric_matcher or VoiceBiometricMatcher()
        self.interruption_classifier = (
            interruption_classifier or InterruptionClassifier()
        )
        self.navigation_detector = navigation_detector or NavigationIntentDetector()
        self.turn_arbitrator = turn_arbitrator or TurnArbitrator(
            classifier=self.interruption_classifier
        )
        self._nav_decisions: dict[str, deque[NavigationDecision]] = {}
        self._arbitrations: dict[str, deque[ArbitrationOutcome]] = {}
        self._barge_ins: dict[str, deque[BargeInDirective]] = {}

    async def process_crosstalk_batch(
        self, batch: CrosstalkBatchEvent
    ) -> CrosstalkBatchResult:
        if self.state_manager.get_room(batch.room_id) is None:
            raise KeyError(f"Room '{batch.room_id}' does not exist.")

        outcome = self.turn_arbitrator.arbitrate(
            batch.frames,
            current_floor_speaker_id=batch.current_floor_speaker_id,
            last_turn_timestamp=batch.last_turn_timestamp,
        )
        self._buffer(self._arbitrations, batch.room_id).append(outcome)
        win = outcome.winning_frame
        if win is None:
            return CrosstalkBatchResult(arbitration=outcome, turn_result=None)

        existing = self.state_manager.get_voice_profile(win.speaker_id)
        if existing is None:
            await self.state_manager.register_voice_profile(
                VoiceProfile(
                    user_id=win.speaker_id,
                    display_name=win.speaker_id,
                    tier=win.tier,
                )
            )

        turn_result = await self.process_turn(
            AudioTurnEvent(
                room_id=batch.room_id,
                raw_utterance=win.utterance,
                speaker_id_hint=win.speaker_id,
                vad_confidence=win.vad_confidence,
                rms_db=win.rms_db,
            )
        )
        return CrosstalkBatchResult(
            arbitration=outcome, turn_result=turn_result
        )

    async def process_turn(self, event: AudioTurnEvent) -> TurnProcessingResult:
        orchestrator = self.state_manager.get_room(event.room_id)
        if orchestrator is None:
            raise KeyError(f"Room '{event.room_id}' does not exist.")

        normalized_utterance = self.lexicon_manager.normalize_utterance(
            event.raw_utterance, orchestrator.state.lexicon_pack_ids
        )
        interruption_type = self.interruption_classifier.classify(
            normalized_utterance,
            vad_confidence=event.vad_confidence,
            rms_db=event.rms_db,
        )

        profile = await self._resolve_speaker(
            event,
            allow_auto_register=(
                interruption_type != InterruptionType.BACKGROUND_EVENT
            ),
        )

        effective_tangent_topic = event.new_tangent_topic
        nav_decision: Optional[NavigationDecision] = None
        if (
            event.new_tangent_topic is None
            and interruption_type != InterruptionType.BACKGROUND_EVENT
        ):
            nav_decision = self.navigation_detector.detect(
                normalized_utterance, orchestrator.nodes
            )
            if (
                nav_decision.action == NavigationAction.BRANCH_TANGENT
                and nav_decision.target_topic
            ):
                effective_tangent_topic = nav_decision.target_topic
            if nav_decision.action != NavigationAction.NONE:
                self._buffer(self._nav_decisions, event.room_id).append(
                    nav_decision
                )

        directive: Optional[BargeInDirective] = None
        branched = False

        if orchestrator.state.is_ai_speaking:
            if interruption_type == InterruptionType.HARD_INTERRUPT:
                if effective_tangent_topic:
                    await orchestrator.branch_tangent(
                        new_topic=effective_tangent_topic,
                        speaker_id=profile.user_id,
                    )
                    branched = True
                elif (
                    nav_decision is not None
                    and nav_decision.action == NavigationAction.JUMP_TO_NODE
                    and nav_decision.target_node_id
                ):
                    await orchestrator.jump_to_node(nav_decision.target_node_id)

            directive = await orchestrator.handle_barge_in(
                speaker_id=profile.user_id,
                utterance=normalized_utterance,
                interruption_type=interruption_type,
            )
            self._buffer(self._barge_ins, event.room_id).append(directive)

            if (
                interruption_type == InterruptionType.HARD_INTERRUPT
                and nav_decision is not None
                and nav_decision.action
                in (NavigationAction.PARK_AND_POP, NavigationAction.POP_TO_ANCHOR)
            ):
                await self._apply_navigation(nav_decision, orchestrator)
                directive.current_node_id = orchestrator.state.current_node_id
        else:
            if interruption_type == InterruptionType.BACKGROUND_EVENT and not effective_tangent_topic:
                return TurnProcessingResult(
                    room_id=event.room_id,
                    resolved_speaker_id=profile.user_id,
                    speaker_tier=profile.tier,
                    normalized_utterance=normalized_utterance,
                    active_node_id=orchestrator.state.current_node_id,
                    barge_in_directive=None,
                    branched_tangent=False,
                )
            if effective_tangent_topic:
                await orchestrator.branch_tangent(
                    new_topic=effective_tangent_topic,
                    speaker_id=profile.user_id,
                )
                branched = True
            elif (
                nav_decision is not None
                and nav_decision.action == NavigationAction.JUMP_TO_NODE
                and nav_decision.target_node_id
            ):
                await orchestrator.jump_to_node(nav_decision.target_node_id)

            if normalized_utterance.strip():
                await orchestrator.record_utterance(
                    speaker_id=profile.user_id,
                    text=normalized_utterance,
                )

            if nav_decision is not None and nav_decision.action in (
                NavigationAction.PARK_AND_POP,
                NavigationAction.POP_TO_ANCHOR,
            ):
                await self._apply_navigation(nav_decision, orchestrator)

        return TurnProcessingResult(
            room_id=event.room_id,
            resolved_speaker_id=profile.user_id,
            speaker_tier=profile.tier,
            normalized_utterance=normalized_utterance,
            active_node_id=orchestrator.state.current_node_id,
            barge_in_directive=directive,
            branched_tangent=branched,
        )

    def get_room_diagnostics(self, room_id: str) -> RoomDiagnosticsReport:
        orchestrator = self.state_manager.get_room(room_id)
        if orchestrator is None:
            raise KeyError(f"Room '{room_id}' does not exist.")

        speaker_tiers = {}
        for speaker_id in orchestrator.state.active_speakers:
            profile = self.state_manager.get_voice_profile(speaker_id)
            if profile is not None:
                speaker_tiers[speaker_id] = profile.tier

        return RoomDiagnosticsReport(
            room_id=room_id,
            active_node_id=orchestrator.current_node.node_id,
            active_topic=orchestrator.current_node.topic_summary,
            total_node_count=len(orchestrator.nodes),
            unresolved_tangent_ids=[
                n.node_id for n in orchestrator.get_unresolved_tangents()
            ],
            speaker_tiers=speaker_tiers,
            active_lexicon_packs=list(orchestrator.state.lexicon_pack_ids),
            recent_navigation_decisions=list(
                self._nav_decisions.get(room_id, [])
            ),
            recent_arbitration_outcomes=list(
                self._arbitrations.get(room_id, [])
            ),
            recent_barge_in_directives=list(
                self._barge_ins.get(room_id, [])
            ),
        )

    @staticmethod
    def _buffer(store: dict, room_id: str) -> deque:
        if room_id not in store:
            store[room_id] = deque(maxlen=10)
        return store[room_id]

    async def _apply_navigation(
        self,
        decision: Optional[NavigationDecision],
        orchestrator: "ContextGraphOrchestrator",
    ) -> None:
        if decision is None:
            return
        if decision.action == NavigationAction.PARK_AND_POP:
            await orchestrator.set_node_status(NodeStatus.PARKED)
            await orchestrator.pop_to_anchor(mark_resolved=False)
        elif decision.action == NavigationAction.POP_TO_ANCHOR:
            await orchestrator.pop_to_anchor(mark_resolved=decision.mark_resolved)
        elif (
            decision.action == NavigationAction.JUMP_TO_NODE
            and decision.target_node_id
        ):
            await orchestrator.jump_to_node(decision.target_node_id)

    async def _resolve_speaker(
        self, event: AudioTurnEvent, allow_auto_register: bool = True
    ) -> VoiceProfile:
        if event.embedding_vector:
            match = self.biometric_matcher.match_speaker(
                event.embedding_vector,
                self.state_manager.list_voice_profiles(),
            )
            if match is not None:
                return match[0]

        if event.speaker_id_hint:
            existing = self.state_manager.get_voice_profile(event.speaker_id_hint)
            if existing is not None:
                return existing

        user_id = event.speaker_id_hint or f"guest-{uuid.uuid4().hex[:8]}"
        ephemeral_profile = VoiceProfile(
            user_id=user_id,
            display_name=user_id,
            tier=VoiceTier.CONVERSATIONAL,
            embedding_vector=event.embedding_vector,
        )
        if not allow_auto_register:
            return ephemeral_profile
        return await self.state_manager.register_voice_profile(ephemeral_profile)
