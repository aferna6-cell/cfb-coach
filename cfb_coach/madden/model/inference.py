"""Live scoring with a time budget and a heuristic fallback. Stub.

The Live Engineer implements this. A model error or a budget overrun must
leave the deterministic caller in charge. This module must not be required
for ``playcaller.make_call`` until hybrid mode is wired, and heuristic mode
must not call it.
"""

from __future__ import annotations

from typing import Sequence

from cfb_coach.madden.model import adapters as adapters
from cfb_coach.madden.model.schema import (
    ML_LATENCY_BUDGET_MS,
    CandidatePlay,
    CoachingDecision,
    CoachingMode,
    GameState,
    OpponentContext,
    PreSnapObservation,
    RecentSnap,
    RosterSnapshot,
)

# Live scoring must take GameState and PreSnapObservation from this function.
# This stub does not call it.
_SITUATION_ADAPTER = adapters.situation_to_state


def rank_live(
    game_state: GameState,
    observation: PreSnapObservation,
    candidate_plays: Sequence[CandidatePlay],
    roster_context: RosterSnapshot | None,
    opponent_context: OpponentContext | None,
    recent_history: Sequence[RecentSnap] | None,
    *,
    mode: CoachingMode = CoachingMode.HEURISTIC,
    model_version: str | None = None,
    budget_ms: float | None = None,
) -> CoachingDecision:
    """Score ``candidate_plays`` inside the model budget. Not implemented.

    ``game_state`` and ``observation`` come from ``_SITUATION_ADAPTER``.
    The model budget is :data:`ML_LATENCY_BUDGET_MS` (150). It covers only
    the model call. Do not rewrite it from measured latency. ``budget_ms``
    ``None`` means the caller did not pass one; the implementation uses the
    fixed constant.     On failure the caller keeps the heuristic decision.
    ``fell_back`` is true only in hybrid, and ``fallback_reason`` is an
    ``MLStatus`` other than ``OK``. ``shadow_status`` is required in shadow
    and hybrid. The shadow-game counter counts only ``shadow_status == OK``.
    """
    del game_state, observation, candidate_plays, roster_context, opponent_context
    del recent_history, mode, model_version, budget_ms
    raise NotImplementedError
