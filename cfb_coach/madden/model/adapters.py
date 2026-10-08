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
from cfb_coach.madden.model.schema import (
    GameState,
    HintSource,
    OpponentKind,
    Possession,
    PreSnapObservation,
    Tri,
)
from cfb_coach.situation import Situation

_HINT_SOURCES = {
    "live": HintSource.LIVE,
    "last": HintSource.LAST,
    "scouted": HintSource.SCOUTED,
    "none": HintSource.NONE,
}


def _hint_source(raw: str | None) -> HintSource:
    if raw is None:
        return HintSource.UNKNOWN
    return _HINT_SOURCES.get(str(raw).strip().lower(), HintSource.UNKNOWN)


def _possession(situation: Situation) -> Possession:
    side = (situation.side or "").strip().lower()
    if side.startswith("d"):
        return Possession.DEFENSE
    if side.startswith("o"):
        return Possession.OFFENSE
    return Possession.UNKNOWN


def _goal_to_go(situation: Situation) -> Tri:
    # Situation.goal_line defaults to False when the phrase is absent — that is
    # not confirmation for the ML schema. Only treat an explicit goal cue as TRUE.
    extras = situation.extras or {}
    if extras.get("goal_to_go") is True or extras.get("goal_line_explicit") is True:
        return Tri.TRUE
    raw = (situation.raw or "").lower()
    if "goal" in raw and (
        "goal line" in raw
        or "&goal" in raw
        or " inches" in raw
        or raw.endswith("goal")
        or " & g" in raw
    ):
        return Tri.TRUE
    if situation.goal_line and ("goal" in raw or "gl" in raw.split()):
        return Tri.TRUE
    return Tri.UNKNOWN


def _score(
    situation: Situation,
    score: GameScore | None,
) -> GameScore | None:
    if score is not None:
        return score
    if situation.score_us is None and situation.score_them is None:
        return None
    if situation.score_us is None or situation.score_them is None:
        return None
    return GameScore(us=int(situation.score_us), them=int(situation.score_them))


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
    """Map one pre-snap :class:`~cfb_coach.situation.Situation`.

    Clock, timeouts, score, patch, and roster stay unknown unless the caller
    passes them. Do not default a missing hint to a coverage name, and do not
    copy ``Situation`` parser defaults into fields that this schema treats as
    unknown.
    """
    del session_id  # carried on CoachingDecision, not GameState
    cov = (situation.coverage_hint or "").strip() or None
    concept = (situation.concept_hint or "").strip() or None
    cov_source = _hint_source(situation.coverage_source)
    concept_source = _hint_source(situation.concept_source)
    if cov is None and cov_source is HintSource.NONE:
        pass
    elif cov is None:
        cov_source = HintSource.NONE if situation.coverage_source == "none" else cov_source

    q = quarter
    if q is None:
        raw_q = (situation.extras or {}).get("quarter")
        if isinstance(raw_q, int) and not isinstance(raw_q, bool):
            q = raw_q

    clock = clock_seconds
    if clock is None:
        raw_clock = (situation.extras or {}).get("clock_seconds")
        if isinstance(raw_clock, int) and not isinstance(raw_clock, bool):
            clock = raw_clock

    state = GameState(
        game_id=game_id,
        snap_id=snap_id,
        madden_version=madden_version,
        profile=profile,
        our_team=our_team,
        opponent_id=opponent_id,
        opponent_team=opponent_team,
        opponent_type=opponent_type if opponent_type is not None else OpponentKind.UNKNOWN,
        possession=_possession(situation),
        situation=situation,
        score=_score(situation, score),
        quarter=q,
        clock_seconds=clock,
        timeouts_us=timeouts_us,
        timeouts_them=timeouts_them,
        goal_to_go=_goal_to_go(situation),
    )
    observation = PreSnapObservation(
        source="live_situation",
        coverage_prediction=cov,
        coverage_prediction_source=cov_source if cov is not None else HintSource.NONE,
        coverage_prediction_confidence=None,
        concept_prediction=concept,
        concept_prediction_source=concept_source if concept is not None else HintSource.NONE,
    )
    return state, observation
