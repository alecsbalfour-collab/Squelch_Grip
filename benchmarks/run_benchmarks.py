from __future__ import annotations

import asyncio
import statistics
import time
from typing import Callable

from core.context_graph import ContextGraphOrchestrator
from core.default_lexicons import register_default_lexicons
from core.interruption_classifier import InterruptionClassifier
from core.lexicon_manager import LexiconManager
from core.navigation_detector import NavigationIntentDetector
from core.prompt_compiler import ContextPromptCompiler
from core.state_manager import SessionStateManager
from core.turn_arbitrator import TurnArbitrator
from core.turn_pipeline import TurnPipeline
from core.voice_biometrics import VoiceBiometricMatcher
from schemas.arbitration_models import CandidateSpeechFrame
from schemas.session_models import VoiceProfile, VoiceTier
from schemas.turn_models import AudioTurnEvent


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round((pct / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _bench_sync(name: str, fn: Callable[[], object], iterations: int = 2000) -> dict[str, float | str]:
    latencies_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    latencies_ms.sort()
    total_s = sum(latencies_ms) / 1000.0
    return {
        "name": name,
        "iterations": iterations,
        "p50_ms": _percentile(latencies_ms, 50),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "mean_ms": statistics.mean(latencies_ms),
        "ops_per_sec": iterations / total_s if total_s > 0 else 0.0,
    }


async def _bench_async(name: str, fn: Callable[[], object], iterations: int = 1000) -> dict[str, float | str]:
    latencies_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        await fn()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    latencies_ms.sort()
    total_s = sum(latencies_ms) / 1000.0
    return {
        "name": name,
        "iterations": iterations,
        "p50_ms": _percentile(latencies_ms, 50),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "mean_ms": statistics.mean(latencies_ms),
        "ops_per_sec": iterations / total_s if total_s > 0 else 0.0,
    }


async def run_all_benchmarks() -> list[dict[str, float | str]]:
    classifier = InterruptionClassifier()
    matcher = VoiceBiometricMatcher()
    arbitrator = TurnArbitrator(classifier=classifier)
    nav_detector = NavigationIntentDetector()
    lexicon = LexiconManager()
    register_default_lexicons(lexicon)
    compiler = ContextPromptCompiler()

    profiles = [
        VoiceProfile(
            user_id=f"usr-{i}",
            display_name=f"Speaker {i}",
            tier=VoiceTier.PERMANENT if i == 0 else VoiceTier.TEMPORARY,
            embedding_vector=[float((i + j) % 7) + 0.1 for j in range(128)],
        )
        for i in range(10)
    ]
    candidate_vec = [float((0 + j) % 7) + 0.12 for j in range(128)]

    frames = [
        CandidateSpeechFrame(
            speaker_id="usr-guest",
            tier=VoiceTier.CONVERSATIONAL,
            utterance="can I jump in here",
            rms_db=-12.0,
        ),
        CandidateSpeechFrame(
            speaker_id="usr-host",
            tier=VoiceTier.PERMANENT,
            utterance="wait, hold on a second",
            rms_db=-20.0,
        ),
        CandidateSpeechFrame(
            speaker_id="usr-noise",
            tier=VoiceTier.CONVERSATIONAL,
            utterance="[cough]",
            vad_confidence=0.2,
            rms_db=-50.0,
        ),
    ]

    state_manager = SessionStateManager()
    for p in profiles:
        await state_manager.register_voice_profile(p)
    room = await state_manager.create_room("room-bench", "Root Architecture")
    lexicon.attach_pack_to_room(room.state, "lex-f1ndr")
    lexicon.attach_pack_to_room(room.state, "lex-dealr")
    await room.branch_tangent("Database Sharding", "usr-0")

    pipeline = TurnPipeline(state_manager=state_manager, lexicon_manager=lexicon)
    turn_event = AudioTurnEvent(
        room_id="room-bench",
        raw_utterance="finder and dealer counter offer are ready",
        embedding_vector=candidate_vec,
    )

    results = [
        _bench_sync(
            "InterruptionClassifier.classify",
            lambda: classifier.classify("wait, hold on a second", vad_confidence=0.95, rms_db=-18.0),
        ),
        _bench_sync(
            "VoiceBiometricMatcher (10x 128-dim)",
            lambda: matcher.match_speaker(candidate_vec, profiles),
        ),
        _bench_sync(
            "LexiconManager.normalize_utterance",
            lambda: lexicon.normalize_utterance(
                "finder and dealer counter offer on squelch grip",
                ["lex-f1ndr", "lex-dealr"],
            ),
        ),
        _bench_sync(
            "TurnArbitrator.arbitrate (3 frames)",
            lambda: arbitrator.arbitrate(frames, current_floor_speaker_id="usr-host"),
        ),
        _bench_sync(
            "NavigationIntentDetector.detect",
            lambda: nav_detector.detect("going back to Database Sharding", room.nodes),
        ),
        _bench_sync(
            "ContextPromptCompiler.compile (full)",
            lambda: compiler.compile(room, lexicon),
        ),
        _bench_sync(
            "ContextPromptCompiler.compile_delta",
            lambda: compiler.compile_delta(
                room,
                previous_node_id=room.state.root_node_id,
                since_timestamp=time.time() - 5.0,
            ),
        ),
        await _bench_async(
            "TurnPipeline.process_turn (end-to-end)",
            lambda: pipeline.process_turn(turn_event),
            iterations=1000,
        ),
    ]

    print("\n" + "=" * 92)
    print(" SQUELCHGRIP v1.0-alpha — REAL-TIME LATENCY & THROUGHPUT BENCHMARKS")
    print("=" * 92)
    print(
        f"{'Subsystem / Operation':<38} | {'p50 (ms)':>9} | {'p95 (ms)':>9} | {'p99 (ms)':>9} | {'Throughput':>14}"
    )
    print("-" * 92)
    for r in results:
        print(
            f"{str(r['name']):<38} | "
            f"{float(r['p50_ms']):>9.4f} | "
            f"{float(r['p95_ms']):>9.4f} | "
            f"{float(r['p99_ms']):>9.4f} | "
            f"{float(r['ops_per_sec']):>10,.0f} /s"
        )
    print("=" * 92 + "\n")
    return results


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
