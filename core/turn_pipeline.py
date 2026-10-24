from __future__ import annotations

import uuid
from typing import Optional

from core.context_graph import ContextGraphOrchestrator
from core.interruption_classifier import InterruptionClassifier
from core.lexicon_manager import LexiconManager
from core.navigation_detector import NavigationIntentDetector
from core.state_manager import SessionStateManager
from core.voice_biometrics import VoiceBiometricMatcher
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
    ) -> None:
        self.state_manager = state_manager
        self.lexicon_manager = lexicon_manager
        self.biometric_matcher = biometric_matcher or VoiceBiometricMatcher()
        self.interruption_classifier = (
            interruption_classifier or InterruptionClassifier()
        )
        self.navigation_detector = navigation_detector or NavigationIntentDetector()

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
