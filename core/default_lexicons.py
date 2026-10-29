from __future__ import annotations

from core.lexicon_manager import LexiconManager
from schemas.lexicon_models import LexiconPack, LexiconTerm


def register_default_lexicons(manager: LexiconManager) -> list[str]:
    """Registers the built-in f1ndr/dealr starter packs. Returns pack ids."""
    manager.register_pack(
        LexiconPack(
            pack_id="lex-f1ndr",
            name="F1ndr Domain Pack",
            domain="f1ndr",
            terms=[
                LexiconTerm(
                    canonical="F1ndr",
                    aliases=["finder", "findr", "f1nder"],
                    definition="F1ndr discovery and matching platform.",
                ),
                LexiconTerm(
                    canonical="SquelchGrip",
                    aliases=["squelch grip", "squelch"],
                    definition="Full-duplex conversational context graph engine.",
                ),
            ],
        )
    )
    manager.register_pack(
        LexiconPack(
            pack_id="lex-dealr",
            name="Dealr Domain Pack",
            domain="dealr",
            terms=[
                LexiconTerm(
                    canonical="Dealr",
                    aliases=["dealer", "deal er"],
                    definition="Dealr negotiation and deal-flow engine.",
                ),
                LexiconTerm(
                    canonical="counteroffer",
                    aliases=["counter offer"],
                    definition=(
                        "Revised commercial proposal in a negotiation thread."
                    ),
                ),
            ],
        )
    )
    return ["lex-f1ndr", "lex-dealr"]
