from __future__ import annotations

from typing import Optional

from core.context_graph import ContextGraphOrchestrator
from core.lexicon_manager import LexiconManager
from schemas.prompt_models import CompiledContextPacket


class ContextPromptCompiler:
    """Compiles a deterministic Markdown snapshot of the conversation tree
    for downstream LLM consumption."""

    def __init__(self, max_recent_utterances: int = 10) -> None:
        self.max_recent_utterances = max_recent_utterances

    def compile(
        self,
        orchestrator: ContextGraphOrchestrator,
        lexicon_manager: Optional[LexiconManager] = None,
    ) -> CompiledContextPacket:
        lineage = orchestrator.get_active_lineage()
        lineage_breadcrumbs = [node.topic_summary for node in lineage]

        utterances = orchestrator.current_node.utterances
        window = (
            utterances[-self.max_recent_utterances :]
            if self.max_recent_utterances > 0
            else []
        )
        recent_utterances = [f"{u.speaker_id}: {u.text}" for u in window]

        unresolved_tangent_topics = [
            f"[{n.status.value}] {n.topic_summary} (node={n.node_id})"
            for n in orchestrator.get_unresolved_tangents()
            if n.node_id != orchestrator.current_node.node_id
        ]

        domain_glossary: dict[str, str] = {}
        if lexicon_manager is not None:
            for pack_id in orchestrator.state.lexicon_pack_ids:
                pack = lexicon_manager.get_pack(pack_id)
                if pack is None:
                    continue
                for term in pack.terms:
                    if term.definition:
                        domain_glossary[term.canonical] = term.definition

        return CompiledContextPacket(
            room_id=orchestrator.state.room_id,
            active_node_id=orchestrator.current_node.node_id,
            active_topic=orchestrator.current_node.topic_summary,
            lineage_breadcrumbs=lineage_breadcrumbs,
            recent_utterances=recent_utterances,
            unresolved_tangent_topics=unresolved_tangent_topics,
            domain_glossary=domain_glossary,
            system_prompt_markdown=self._render_markdown(
                orchestrator.current_node.topic_summary,
                lineage_breadcrumbs,
                recent_utterances,
                unresolved_tangent_topics,
                domain_glossary,
            ),
        )

    @staticmethod
    def _render_markdown(
        active_topic: str,
        lineage_breadcrumbs: list[str],
        recent_utterances: list[str],
        unresolved_tangent_topics: list[str],
        domain_glossary: dict[str, str],
    ) -> str:
        def _section(items: list[str]) -> list[str]:
            return items if items else ["- (none)"]

        lines = [
            "# Active Topic",
            "",
            active_topic,
            "",
            "## Lineage Breadcrumbs",
            "",
        ]
        lines += _section(
            [f"{i}. {t}" for i, t in enumerate(lineage_breadcrumbs, start=1)]
        )
        lines += ["", "## Recent Transcript", ""]
        lines += _section([f"- {u}" for u in recent_utterances])
        lines += ["", "## Unresolved / Parked Tangents", ""]
        lines += _section([f"- {t}" for t in unresolved_tangent_topics])
        lines += ["", "## Domain Glossary", ""]
        lines += _section(
            [f"- **{k}**: {v}" for k, v in sorted(domain_glossary.items())]
        )
        return "\n".join(lines)
