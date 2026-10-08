"""Convert a live situation into the ML contracts. Owner: Data Engineer.

:func:`situation_to_state` is the only function that builds a
:class:`~cfb_coach.madden.model.schema.GameState` and a
:class:`~cfb_coach.madden.model.schema.PreSnapObservation` from the live
parser. The logger and ``inference.py`` call it. They do not construct those
two types themselves.

The observation uses the situation's coverage and concept hints and their
sources. It does not read a logged snap row, ``coverage_seen``, or any outcome.
"""

from __future__ import annotations

from cfb_coach.game_score import GameScore
from cfb_coach.madden.model.schema import GameState, OpponentKind, PreSnapObservation
from cfb_coach.situation import Situation


def situation_to_state(
    situation: Situation,
    *,
    game_id: str | None = None,
    snap_id: str | None = None,
    session_id: str | None = None,
    madden_version: str | None = None,
    profile: str | None = None,
    our_team: str | None = None,
    opponent_id: str | None = None,
    opponent_team: str | None = None,
    opponent_type: OpponentKind | None = None,
    quarter: int | None = None,
    clock_seconds: int | None = None,
    timeouts_us: int | None = None,
    timeouts_them: int | None = None,
    score: GameScore | None = None,
) -> tuple[GameState, PreSnapObservation]:
    """Map one pre-snap :class:`~cfb_coach.situation.Situation`. Not implemented.

    Clock, timeouts, score, patch, and roster stay unknown unless the caller
    passes them. Do not default a missing hint to a coverage name, and do not
    copy ``Situation`` parser defaults into fields that this schema treats as
    unknown.
    """
    raise NotImplementedError
