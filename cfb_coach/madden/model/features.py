"""Decision-time feature construction. Owner: ML Training.

Features may be built only from :class:`~cfb_coach.madden.model.schema.GameState`,
:class:`~cfb_coach.madden.model.schema.PreSnapObservation`, the candidate,
roster, opponent context, and snaps that already ended. Do not read
:class:`~cfb_coach.madden.model.schema.SnapOutcome` of the snap being decided.
"""

from __future__ import annotations

import re
from typing import Sequence

from cfb_coach.madden.model.schema import (
    FEATURE_SCHEMA_VERSION,
    CandidatePlay,
    FeatureValue,
    FeatureVector,
    GameState,
    OpponentContext,
    OpponentKind,
    Possession,
    PreSnapObservation,
    RecentSnap,
    RosterSnapshot,
    Tri,
)

# Stable column order for madden-ml.features.2.
# Categoricals use football family one-hots / unknown flags — not numeric hashes.
_FEATURE_NAMES: tuple[str, ...] = (
    "is_offense",
    "is_defense",
    "is_cpu",
    "is_human",
    "down",
    "distance",
    "yardline",
    "field_own",
    "field_mid",
    "field_red_zone",
    "field_goal_line",
    "score_diff",
    "quarter",
    "clock_seconds",
    "timeouts_us",
    "timeouts_them",
    "goal_to_go",
    "coverage_prediction_confidence",
    "has_coverage_prediction",
    "has_concept_prediction",
    "cov_fam_cover0",
    "cov_fam_cover1",
    "cov_fam_cover2",
    "cov_fam_cover3",
    "cov_fam_cover4",
    "cov_fam_cover6",
    "cov_fam_other",
    "cov_fam_unknown",
    "cand_play_family_pass",
    "cand_play_family_run",
    "cand_play_family_rpo",
    "cand_play_family_screen",
    "cand_play_family_unknown",
    "cand_form_gun",
    "cand_form_under",
    "cand_form_singleback",
    "cand_form_i",
    "cand_form_other",
    "cand_form_unknown",
    "cand_cov_family_zone",
    "cand_cov_family_man",
    "cand_cov_family_blitz",
    "cand_adjustment_slot",
    "opp_n_snaps",
    "opp_sample_size",
    "opp_shrinkage_weight",
    "opp_tendency_rate",
    "recent_n",
    "recent_success_rate",
    "recent_same_formation",
    "recent_same_play",
    "roster_mean_rating",
    "roster_n_features",
    "roster_complete",
    "source_live",
)


_PASS_RE = re.compile(
    r"\b(pass|slant|flood|mesh|spot|seam|vert|cross|wheel|out|in|curl|flat|"
    r"smash|levels|dagger|sail|shot|go|post|corner|fade|hitch)\b",
    re.I,
)
_RUN_RE = re.compile(
    r"\b(run|zone|power|counter|draw|dive|iso|toss|sweep|inside|outside|lead)\b",
    re.I,
)
_RPO_RE = re.compile(r"\brpo\b", re.I)
_SCREEN_RE = re.compile(r"\b(screen|bubble|tunnel)\b", re.I)
_ZONE_RE = re.compile(r"\b(cover\s*[234]|zone|match|tampa|quarters|palms|cloud)\b", re.I)
_MAN_RE = re.compile(r"\b(cover\s*1|man|zero|robber)\b", re.I)
_BLITZ_RE = re.compile(r"\b(blitz|pressure|fire|dog|sim)\b", re.I)


def pre_snap_feature_names() -> tuple[str, ...]:
    """Stable names of the decision-time columns."""
    return _FEATURE_NAMES


def _fv(name: str, value: float | None) -> FeatureValue:
    if value is None:
        return FeatureValue(name=name, value=None, missing=True)
    return FeatureValue(name=name, value=float(value), missing=False)


def _coverage_family_flags(text: str | None) -> dict[str, float]:
    raw = (text or "").strip().lower()
    flags = {
        "cov_fam_cover0": 0.0,
        "cov_fam_cover1": 0.0,
        "cov_fam_cover2": 0.0,
        "cov_fam_cover3": 0.0,
        "cov_fam_cover4": 0.0,
        "cov_fam_cover6": 0.0,
        "cov_fam_other": 0.0,
        "cov_fam_unknown": 0.0,
    }
    if not raw:
        flags["cov_fam_unknown"] = 1.0
        return flags
    if "cover 0" in raw or "cover0" in raw or re.search(r"\bzero\b", raw):
        flags["cov_fam_cover0"] = 1.0
    elif "cover 1" in raw or "cover1" in raw or re.search(r"\bman\b", raw):
        flags["cov_fam_cover1"] = 1.0
    elif "cover 2" in raw or "cover2" in raw:
        flags["cov_fam_cover2"] = 1.0
    elif "cover 3" in raw or "cover3" in raw:
        flags["cov_fam_cover3"] = 1.0
    elif "cover 4" in raw or "cover4" in raw or "quarters" in raw:
        flags["cov_fam_cover4"] = 1.0
    elif "cover 6" in raw or "cover6" in raw:
        flags["cov_fam_cover6"] = 1.0
    else:
        flags["cov_fam_other"] = 1.0
    return flags


def _formation_family_flags(text: str | None) -> dict[str, float]:
    raw = (text or "").strip().lower()
    flags = {
        "cand_form_gun": 0.0,
        "cand_form_under": 0.0,
        "cand_form_singleback": 0.0,
        "cand_form_i": 0.0,
        "cand_form_other": 0.0,
        "cand_form_unknown": 0.0,
    }
    if not raw:
        flags["cand_form_unknown"] = 1.0
        return flags
    if "gun" in raw:
        flags["cand_form_gun"] = 1.0
    elif "under" in raw or "pistol" in raw:
        flags["cand_form_under"] = 1.0
    elif "singleback" in raw or "single back" in raw or "strong" in raw:
        flags["cand_form_singleback"] = 1.0
    elif re.search(r"\bi[\s-]?form\b", raw) or raw.startswith("i "):
        flags["cand_form_i"] = 1.0
    else:
        flags["cand_form_other"] = 1.0
    return flags


def _field_zone(yardline: int | None) -> tuple[float | None, float | None, float | None, float | None]:
    if yardline is None:
        return None, None, None, None
    yl = int(yardline)
    own = 1.0 if yl < 40 else 0.0
    mid = 1.0 if 40 <= yl < 80 else 0.0
    rz = 1.0 if 80 <= yl < 97 else 0.0
    gl = 1.0 if yl >= 97 else 0.0
    return own, mid, rz, gl


def _tri01(value: Tri) -> float | None:
    if value is Tri.TRUE:
        return 1.0
    if value is Tri.FALSE:
        return 0.0
    return None


def _recent_stats(
    recent_history: Sequence[RecentSnap],
    candidate: CandidatePlay,
) -> tuple[float, float | None, float, float]:
    n = float(len(recent_history))
    if not recent_history:
        return 0.0, None, 0.0, 0.0
    successes = 0
    known = 0
    same_form = 0
    same_play = 0
    for snap in recent_history:
        if snap.outcome is not None and snap.outcome.success is Tri.TRUE:
            successes += 1
            known += 1
        elif snap.outcome is not None and snap.outcome.success is Tri.FALSE:
            known += 1
        if candidate.formation and snap.formation == candidate.formation:
            same_form += 1
        if candidate.play and snap.play == candidate.play:
            same_play += 1
    rate = (successes / known) if known else None
    return n, rate, float(same_form), float(same_play)


def _roster_stats(roster: RosterSnapshot | None) -> tuple[float | None, float, float | None]:
    if roster is None:
        return None, 0.0, None
    nums = [
        f.value_number
        for f in roster.features
        if not f.missing and f.value_number is not None
    ]
    mean = (sum(nums) / len(nums)) if nums else None
    complete = _tri01(roster.completeness)
    return mean, float(len(roster.features)), complete


def _tendency_rate(opponent: OpponentContext | None) -> float | None:
    if opponent is None or not opponent.tendencies:
        return None
    total_n = 0
    total_s = 0
    for t in opponent.tendencies:
        if t.sample_size is None and t.count is None:
            continue
        n = int(t.sample_size if t.sample_size is not None else t.count or 0)
        s = int(t.success or 0)
        total_n += n
        total_s += s
    if total_n <= 0:
        return None
    # Sample-size-aware shrinkage toward 0.5.
    prior = 0.5
    prior_n = 10.0
    return (total_s + prior * prior_n) / (total_n + prior_n)


def vector_for(
    game_state: GameState,
    observation: PreSnapObservation,
    candidate: CandidatePlay,
    roster: RosterSnapshot | None,
    opponent: OpponentContext | None,
    recent_history: Sequence[RecentSnap],
) -> FeatureVector:
    """One row for ``candidate``. Missing inputs stay missing."""
    sit = game_state.situation
    down = sit.down if sit is not None else None
    distance = sit.distance if sit is not None else None
    yardline = sit.yardline if sit is not None else None
    own, mid, rz, gl = _field_zone(yardline)

    score_diff = None
    if game_state.score is not None:
        score_diff = float(game_state.score.us - game_state.score.them)

    play_name = candidate.play or ""
    form_name = candidate.formation or ""
    blob = f"{form_name} {play_name}"

    recent_n, recent_rate, same_form, same_play = _recent_stats(recent_history, candidate)
    roster_mean, roster_n, roster_complete = _roster_stats(roster)

    opp_n = opponent.n_snaps if opponent is not None else None
    opp_sample = opponent.sample_size if opponent is not None else None
    opp_shrink = opponent.shrinkage_weight if opponent is not None else None
    if opp_shrink is None and opponent is not None and opp_n is not None:
        opp_shrink = min(1.0, float(opp_n) / (float(opp_n) + 20.0))

    values = {
        "is_offense": 1.0 if game_state.possession is Possession.OFFENSE else (
            0.0 if game_state.possession is Possession.DEFENSE else None
        ),
        "is_defense": 1.0 if game_state.possession is Possession.DEFENSE else (
            0.0 if game_state.possession is Possession.OFFENSE else None
        ),
        "is_cpu": 1.0 if game_state.opponent_type is OpponentKind.CPU else (
            0.0 if game_state.opponent_type is OpponentKind.HUMAN else None
        ),
        "is_human": 1.0 if game_state.opponent_type is OpponentKind.HUMAN else (
            0.0 if game_state.opponent_type is OpponentKind.CPU else None
        ),
        "down": float(down) if down is not None else None,
        "distance": float(distance) if distance is not None else None,
        "yardline": float(yardline) if yardline is not None else None,
        "field_own": own,
        "field_mid": mid,
        "field_red_zone": rz,
        "field_goal_line": gl,
        "score_diff": score_diff,
        "quarter": float(game_state.quarter) if game_state.quarter is not None else None,
        "clock_seconds": float(game_state.clock_seconds)
        if game_state.clock_seconds is not None
        else None,
        "timeouts_us": float(game_state.timeouts_us)
        if game_state.timeouts_us is not None
        else None,
        "timeouts_them": float(game_state.timeouts_them)
        if game_state.timeouts_them is not None
        else None,
        "goal_to_go": _tri01(game_state.goal_to_go),
        "coverage_prediction_confidence": observation.coverage_prediction_confidence,
        "has_coverage_prediction": 1.0 if observation.coverage_prediction else 0.0,
        "has_concept_prediction": 1.0 if observation.concept_prediction else 0.0,
        **_coverage_family_flags(observation.coverage_prediction),
        **_formation_family_flags(candidate.formation),
        "cand_play_family_pass": 1.0 if _PASS_RE.search(blob) else 0.0,
        "cand_play_family_run": 1.0 if _RUN_RE.search(blob) else 0.0,
        "cand_play_family_rpo": 1.0 if _RPO_RE.search(blob) else 0.0,
        "cand_play_family_screen": 1.0 if _SCREEN_RE.search(blob) else 0.0,
        "cand_play_family_unknown": 0.0
        if (_PASS_RE.search(blob) or _RUN_RE.search(blob) or _RPO_RE.search(blob) or _SCREEN_RE.search(blob))
        else (0.0 if play_name else 1.0),
        "cand_cov_family_zone": 1.0 if _ZONE_RE.search(blob) else 0.0,
        "cand_cov_family_man": 1.0 if _MAN_RE.search(blob) else 0.0,
        "cand_cov_family_blitz": 1.0 if _BLITZ_RE.search(blob) else 0.0,
        "cand_adjustment_slot": float(candidate.adjustment_slot)
        if candidate.adjustment_slot is not None
        else None,
        "opp_n_snaps": float(opp_n) if opp_n is not None else None,
        "opp_sample_size": float(opp_sample) if opp_sample is not None else None,
        "opp_shrinkage_weight": opp_shrink,
        "opp_tendency_rate": _tendency_rate(opponent),
        "recent_n": recent_n,
        "recent_success_rate": recent_rate,
        "recent_same_formation": same_form,
        "recent_same_play": same_play,
        "roster_mean_rating": roster_mean,
        "roster_n_features": roster_n,
        "roster_complete": roster_complete,
        "source_live": 1.0 if (observation.source or "").startswith("live") else (
            0.0 if observation.source else None
        ),
    }

    items = tuple(_fv(name, values[name]) for name in _FEATURE_NAMES)
    assert FEATURE_SCHEMA_VERSION  # keep version import live for callers
    return FeatureVector(feature_schema_version=FEATURE_SCHEMA_VERSION, items=items)


def intelligence_features(
    play: str,
    diagnosis: dict | None = None,
    strategy: dict | None = None,
) -> dict:
    """Coordinator context that is not part of the training vector.

    ``madden-ml.features.2`` stays unchanged. Missing route details stay null.
    """
    from cfb_coach.madden.model.football_knowledge import KNOWLEDGE_VERSION, profile_for_play

    profile = profile_for_play(play)
    observed = (diagnosis or {}).get("observed") or {}
    return {
        "feature_schema": "madden.football_intelligence.v1",
        "training_vector": FEATURE_SCHEMA_VERSION,
        "knowledge_version": KNOWLEDGE_VERSION,
        "concept_id": profile.get("concept_id"),
        "concept_confidence": profile.get("confidence"),
        "route_diagram": None,
        "player_assignments": None,
        "controller_inputs": None,
        "observed_shell": observed.get("shell"),
        "diagnosis_state": (diagnosis or {}).get("state"),
        "strategy_objective": (strategy or {}).get("objective"),
        "strategy_hypothesis": (strategy or {}).get("hypothesis_id"),
        "opponent_learning_hypothesis": (
            (strategy or {}).get("hypothesis_id")
            if str((strategy or {}).get("hypothesis_id") or "") in (
                "learned_passing_down_pressure", "revised_away_from_quick_pressure",
            )
            else None
        ),
    }


def dense_pair(vector: FeatureVector) -> list[float]:
    """Expand each feature into ``(value_or_0, missing_bit)`` for trainers."""
    out: list[float] = []
    by_name = {item.name: item for item in vector.items}
    for name in _FEATURE_NAMES:
        item = by_name.get(name)
        if item is None or item.missing or item.value is None:
            out.extend([0.0, 1.0])
        else:
            out.extend([float(item.value), 0.0])
    return out
