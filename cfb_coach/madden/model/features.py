"""Decision-time feature construction. Stub until ML Training implements it.

Features may be built only from :class:`~cfb_coach.madden.model.schema.GameState`,
:class:`~cfb_coach.madden.model.schema.PreSnapObservation`, the candidate,
roster, opponent context, and snaps that already ended. Do not read
:class:`~cfb_coach.madden.model.schema.SnapOutcome` of the snap being decided.
"""

from __future__ import annotations

from typing import Sequence

from cfb_coach.madden.model.schema import (
    CandidatePlay,
    FeatureVector,
    GameState,
    OpponentContext,
    PreSnapObservation,
    RecentSnap,
    RosterSnapshot,
)


def pre_snap_feature_names() -> tuple[str, ...]:
    """Stable names of the decision-time columns. Not implemented."""
    raise NotImplementedError


def vector_for(
    game_state: GameState,
    observation: PreSnapObservation,
    candidate: CandidatePlay,
    roster: RosterSnapshot | None,
    opponent: OpponentContext | None,
    recent_history: Sequence[RecentSnap],
) -> FeatureVector:
    """One row for ``candidate``. Missing inputs stay missing. Not implemented."""
    raise NotImplementedError
