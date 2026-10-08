"""Turn outcome estimates into one legal ranking. Stub until the Strategist implements it.

The ranker does not invent play names and does not treat the highest raw
success probability as the call. It only scores candidates that
:func:`cfb_coach.madden.model.schema.legal_candidates` would keep.
"""

from __future__ import annotations

from typing import Sequence

from cfb_coach.madden.model.schema import (
    CandidatePlay,
    CoachingDecision,
    GameState,
    OpponentContext,
    PlayRanking,
    RosterSnapshot,
    SnapOutcome,
)


def rank_plays(
    game_state: GameState,
    candidate_plays: Sequence[CandidatePlay],
    roster_context: RosterSnapshot | None,
    opponent_context: OpponentContext | None,
    recent_history: Sequence[tuple[CoachingDecision, SnapOutcome | None]],
) -> PlayRanking:
    """Rank the legal pool for this snap. Not implemented.

    ``recent_history`` is its own argument. It is not a field on
    ``game_state``. Each pair is a decision that already ended and the outcome
    of that snap, or ``None`` when the outcome is still unknown. The outcome
    of the snap being ranked must not be included. Pass an empty sequence when
    no prior snap was supplied. Each kept candidate later gets a ``sampling_p``
    under ``policy_source``. Equal scores need a stable tie break on
    formation, play, then adjustment slot. This stub returns nothing and must
    not be used on the live path.
    """
    raise NotImplementedError
