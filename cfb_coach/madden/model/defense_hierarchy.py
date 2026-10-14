"""Hierarchical defensive objective for the existing defensive coordinator.

This is not a second play caller. ``train_defense`` stores the fit on the
defense artifact, and ``_score`` uses it when verified snaps exist. Opponent
counts never update the general tables. A post-snap concept or coverage is
not a feature of the snap being predicted.

The objective is the cost of the outcome the offense achieved on a verified
defensive snap: conversion, yards, explosive gain, and touchdown. A 40-yard
touchdown costs more than a short first down. Uncalled plays have no
observed yards, so evaluation scores the play that was executed. It does not
claim a counterfactual stop.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from cfb_coach.madden.defense_select import call_family
from cfb_coach.madden.model.schema import LEAKAGE_FIELDS

HIERARCHY_VERSION = "madden.defense.hierarchy.v1"
PRIOR_CONVERSION = 0.45
PRIOR_YARDS = 5.5
PRIOR_EXPLOSIVE = 0.08
PRIOR_TOUCHDOWN = 0.03
EXPLOSIVE_YARDS = 15

# Shrinkage strengths. Opponent residuals use the largest strength so one
# player's sample cannot replace the general rate.
SITUATION_STRENGTH = 8.0
FAMILY_STRENGTH = 8.0
PLAY_STRENGTH = 10.0
OPPONENT_STRENGTH = 24.0
RESIDUAL_STRENGTH = 16.0

COST_WEIGHTS = {
    "conversion": 1.0,
    "yards": 0.03,
    "explosive": 1.25,
    "touchdown": 3.5,
}

MIN_SUPERVISED = 40
MIN_OPPONENTS = 2
MIN_GAMES = 4
MIN_HOLDOUT_SNAPS = 12
LATENCY_P95_MS = 150.0

# Names the fit is allowed to read. None of them are post-snap results.
PRE_SNAP_FEATURES = (
    "down",
    "distance",
    "red_zone",
    "goal_line",
    "two_minute",
    "score_for",
    "score_against",
    "quarter",
    "opp_personnel",
    "opp_formation",
    "defensive_family",
    "defensive_play",
    "recent_calls",
)

_HEADS = ("conv", "yards", "expl", "td")
_PRIORS = {
    "conv": PRIOR_CONVERSION,
    "yards": PRIOR_YARDS,
    "expl": PRIOR_EXPLOSIVE,
    "td": PRIOR_TOUCHDOWN,
}


def _empty_node() -> dict[str, list[float]]:
    return {head: [0.0, 0] for head in _HEADS}


def _add(node: dict[str, list[float]], head: str, value: float | None) -> None:
    if value is None:
        return
    node[head][0] += float(value)
    node[head][1] += 1


def _shrink(total: float, n: int, parent: float, strength: float) -> float:
    if n <= 0:
        return parent
    return (total + strength * parent) / (n + strength)


def _mean(values: Sequence[float], fallback: float) -> float:
    if not values:
        return fallback
    return sum(values) / len(values)


def situation_bucket(row: Mapping[str, Any]) -> str:
    from cfb_coach.madden.model.defense_coordinator import _bucket

    return _bucket(
        row.get("down"),
        row.get("distance"),
        red_zone=bool(row.get("red_zone")),
        goal_line=bool(row.get("goal_line")),
        two_minute=bool(row.get("two_minute")),
    )


def _score_phase(score_for: Any, score_against: Any) -> str | None:
    try:
        us = int(score_for)
        them = int(score_against)
    except (TypeError, ValueError):
        return None
    if us > them:
        return "leading"
    if us < them:
        return "trailing"
    return "tied"


def _clock_bucket(row: Mapping[str, Any]) -> str | None:
    if bool(row.get("two_minute")):
        return "two_minute"
    try:
        quarter = int(row.get("quarter"))
    except (TypeError, ValueError):
        return None
    if quarter >= 4:
        return "late"
    if quarter > 0:
        return "early_clock"
    return None


def _token(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def outcomes_from_row(row: Mapping[str, Any]) -> dict[str, float | None]:
    """Defense-perspective labels. Unknown stays None and is not trained as zero.

    On a defensive row, ``success`` true is a stop: the offense did not convert.
    That matches ``train_defense`` and ``dataset._success_from_yards``.
    """
    success = str(row.get("success") or "").lower()
    if success == "true":
        converted: float | None = 0.0
    elif success == "false":
        converted = 1.0
    else:
        converted = None
    yards: float | None
    try:
        raw_yards = row.get("yards")
        yards = float(raw_yards) if raw_yards is not None and raw_yards != "" else None
    except (TypeError, ValueError):
        yards = None
    result = str(row.get("result") or "").lower()
    kind = str(row.get("result_kind") or row.get("kind") or "").lower()
    touchdown: float | None = None
    flag = row.get("touchdown")
    if flag in (True, 1, "1", "true", "True"):
        touchdown = 1.0
    elif "touchdown" in result or kind in {"td", "touchdown"}:
        touchdown = 1.0
    elif yards is not None and (result or kind):
        touchdown = 0.0
    explosive: float | None = None
    if yards is not None:
        explosive = 1.0 if yards >= EXPLOSIVE_YARDS else 0.0
    return {
        "conv": converted,
        "yards": yards,
        "expl": explosive,
        "td": touchdown,
    }


def _observe(node: dict[str, list[float]], labels: Mapping[str, float | None]) -> None:
    for head in _HEADS:
        _add(node, head, labels.get(head))


def _new_opponent() -> dict[str, Any]:
    return {
        "global": _empty_node(),
        "situations": {},
        "families": {},
        "plays": {},
        "personnel": {},
        "formations": {},
        "score_phase": {},
        "clock": {},
        "games": [],
        "snaps": 0,
    }


def _node(tree: dict[str, Any], key: str) -> dict[str, list[float]]:
    return tree.setdefault(key, _empty_node())


def classify_defensive_candidate(row: Mapping[str, Any]) -> tuple[str, str]:
    """``supervised`` or ``excluded`` plus one reason. Recommendations are not executions."""
    if str(row.get("side") or "").lower() not in {"defense", "d"}:
        return "excluded", "not_defense"
    if row.get("recommendation_only"):
        return "excluded", "recommendation_only"
    if str(row.get("opponent_type") or "").lower() != "human":
        return "excluded", "not_human"
    if not str(row.get("opponent_id") or "").strip() or not str(row.get("game_id") or "").strip():
        return "excluded", "missing_identity"
    if str(row.get("executed_status") or "").lower() != "identified":
        return "excluded", "execution_not_identified"
    if str(row.get("executed_verification") or "").lower() != "verified":
        return "excluded", "execution_not_verified"
    if not row.get("executed_play"):
        return "excluded", "executed_play_missing"
    # Same identity gate as verified_human_rows. A missing flag is not eligible.
    if row.get("supervised_eligible") is not True:
        return "excluded", "not_supervised_eligible"
    if not row.get("label_available"):
        return "excluded", "label_unavailable"
    from cfb_coach.madden.defense_select import FAMILIES, call_family

    family = call_family(str(row.get("executed_play") or ""))
    if family not in FAMILIES:
        return "excluded", "family_not_in_defensive_menu"
    if str(row.get("success") or "").lower() not in {"true", "false"}:
        return "excluded", "success_label_missing"
    labels = outcomes_from_row(row)
    if labels["conv"] is None:
        return "excluded", "outcome_untrustworthy"
    return "supervised", "verified_executed_outcome"


def _logging_gaps(row: Mapping[str, Any]) -> list[str]:
    gaps = []
    if not str(row.get("drive_id") or "").strip():
        gaps.append("drive_id_missing")
    if not str(row.get("game_clock") or "").strip():
        gaps.append("game_clock_missing")
    return gaps


def audit_defensive_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reported_candidates: int | None = None,
) -> dict[str, Any]:
    """Count defense-side candidates. Supervised is the verified subset, by game."""
    candidates = 0
    supervised = 0
    reasons: dict[str, int] = defaultdict(int)
    gaps: dict[str, int] = defaultdict(int)
    games: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("side") or "").lower() not in {"defense", "d"}:
            continue
        candidates += 1
        status, reason = classify_defensive_candidate(row)
        game_id = str(row.get("game_id") or "").strip() or "(missing)"
        bucket = games.setdefault(game_id, {
            "candidates": 0,
            "supervised": 0,
            "excluded": {},
        })
        bucket["candidates"] += 1
        if status == "supervised":
            supervised += 1
            bucket["supervised"] += 1
        else:
            reasons[reason] += 1
            excluded = bucket["excluded"]
            excluded[reason] = int(excluded.get(reason, 0)) + 1
        for gap in _logging_gaps(row):
            gaps[gap] += 1
    report: dict[str, Any] = {
        "version": HIERARCHY_VERSION,
        "candidate_snaps": candidates,
        "supervised_snaps": supervised,
        "excluded_snaps": candidates - supervised,
        "excluded_reasons": dict(reasons),
        "logging_gaps": dict(gaps),
        "by_game": games,
        "games": len([gid for gid, row in games.items() if row["supervised"] > 0]),
        "reported_count_is_supervised": False,
        "post_snap_features_used": False,
        "note": (
            "Candidate snaps are not supervised examples. Play-specific training "
            "uses only verified executed human defensive snaps with a trustworthy "
            "outcome. A missing drive id or clock is reported and does not, by "
            "itself, delete a verified outcome."
        ),
    }
    if reported_candidates is not None:
        report["reported_candidate_count"] = int(reported_candidates)
        report["reported_count_matches_this_input"] = candidates == int(reported_candidates)
        report["explanation"] = (
            f"A reported {int(reported_candidates)} defensive snaps are candidates. "
            f"This input has {candidates} defense-side rows and {supervised} verified "
            "supervised examples. The reported figure is not a training count."
        )
    return report


def fit_defense_hierarchy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit general rates and separate per-opponent residuals.

    ``rows`` must already be the verified supervised subset. Opponent tables
    are the only place an ``opponent_id`` is stored.
    """
    leaked = sorted(LEAKAGE_FIELDS.intersection(PRE_SNAP_FEATURES))
    if leaked:
        raise RuntimeError(f"pre-snap feature list overlaps leakage fields: {leaked}")
    opponents: dict[str, dict[str, Any]] = {}
    for row in rows:
        status, _reason = classify_defensive_candidate(row)
        if status != "supervised":
            continue
        play = str(row.get("executed_play") or "")
        family = call_family(play) or "unknown"
        opponent_id = str(row["opponent_id"])
        game_id = str(row["game_id"])
        tree = opponents.setdefault(opponent_id, _new_opponent())
        if game_id not in tree["games"]:
            tree["games"].append(game_id)
        tree["snaps"] += 1
        labels = outcomes_from_row(row)
        bucket = situation_bucket(row)
        _observe(tree["global"], labels)
        _observe(_node(tree["situations"], bucket), labels)
        _observe(_node(tree["families"], f"{bucket}|{family}"), labels)
        _observe(_node(tree["plays"], f"{bucket}|{family}|{play}"), labels)
        personnel = _token(row.get("opp_personnel"))
        if personnel:
            _observe(_node(tree["personnel"], personnel), labels)
        formation = _token(row.get("opp_formation") or row.get("observed_offensive_formation"))
        if formation:
            _observe(_node(tree["formations"], formation), labels)
        phase = _score_phase(row.get("score_for", row.get("score_us")), row.get("score_against", row.get("score_them")))
        if phase:
            _observe(_node(tree["score_phase"], phase), labels)
        clock = _clock_bucket(row)
        if clock:
            _observe(_node(tree["clock"], clock), labels)
    general = _general_rates(opponents)
    supervised = sum(int(tree["snaps"]) for tree in opponents.values())
    return {
        "version": HIERARCHY_VERSION,
        "pre_snap_features": list(PRE_SNAP_FEATURES),
        "leakage_fields_used": [],
        "cost_weights": dict(COST_WEIGHTS),
        "opponent_strength": OPPONENT_STRENGTH,
        "supervised_snaps": supervised,
        "opponents_fitted": len(opponents),
        "games_fitted": len({game for tree in opponents.values() for game in tree["games"]}),
        "general_separated_from_opponents": len(opponents) >= 2,
        "single_opponent_overfit_risk": len(opponents) < 2,
        "opponents": opponents,
        "general": general,
        "promotion": {
            "allowed": False,
            "reason": "eval_not_run",
            "auto_promote": False,
        },
        "observational_only": True,
        "counterfactual_effects": False,
    }


def _level_rate(
    opponents: Mapping[str, Mapping[str, Any]],
    table: str,
    key: str | None,
    head: str,
    parent: float,
    strength: float,
) -> float:
    rates: list[float] = []
    for tree in opponents.values():
        if table == "global":
            node = tree["global"]
        else:
            node = (tree.get(table) or {}).get(key or "")
        if not node:
            continue
        total, n = node[head]
        if int(n) <= 0:
            continue
        rates.append(_shrink(float(total), int(n), parent, strength))
    return _mean(rates, parent)


def _general_rates(opponents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Unweighted mean of per-opponent shrunk rates. One opponent is one vote."""
    general: dict[str, Any] = {"global": {}, "situations": {}, "families": {}, "plays": {},
                               "personnel": {}, "formations": {}, "score_phase": {}, "clock": {}}
    for head, prior in _PRIORS.items():
        general["global"][head] = _level_rate(opponents, "global", None, head, prior, SITUATION_STRENGTH)
    situation_keys = {key for tree in opponents.values() for key in (tree.get("situations") or {})}
    for key in situation_keys:
        general["situations"][key] = {}
        for head in _HEADS:
            general["situations"][key][head] = _level_rate(
                opponents, "situations", key, head, general["global"][head], SITUATION_STRENGTH,
            )
    family_keys = {key for tree in opponents.values() for key in (tree.get("families") or {})}
    for key in family_keys:
        parent_key = key.split("|", 1)[0]
        parent = general["situations"].get(parent_key) or general["global"]
        general["families"][key] = {}
        for head in _HEADS:
            general["families"][key][head] = _level_rate(
                opponents, "families", key, head, parent[head], FAMILY_STRENGTH,
            )
    play_keys = {key for tree in opponents.values() for key in (tree.get("plays") or {})}
    for key in play_keys:
        parts = key.split("|")
        family_key = "|".join(parts[:2]) if len(parts) >= 2 else parts[0]
        parent = general["families"].get(family_key) or general["global"]
        general["plays"][key] = {}
        for head in _HEADS:
            general["plays"][key][head] = _level_rate(
                opponents, "plays", key, head, parent[head], PLAY_STRENGTH,
            )
    for table in ("personnel", "formations", "score_phase", "clock"):
        keys = {key for tree in opponents.values() for key in (tree.get(table) or {})}
        for key in keys:
            general[table][key] = {}
            for head in _HEADS:
                general[table][key][head] = _level_rate(
                    opponents, table, key, head, general["global"][head], RESIDUAL_STRENGTH,
                )
    return general


def _chain_rate(hierarchy: Mapping[str, Any], head: str, play_key: str, family_key: str, bucket: str) -> float:
    general = hierarchy.get("general") or {}
    play_rate = ((general.get("plays") or {}).get(play_key) or {}).get(head)
    if play_rate is not None:
        return float(play_rate)
    family_rate = ((general.get("families") or {}).get(family_key) or {}).get(head)
    if family_rate is not None:
        return float(family_rate)
    situation_rate = ((general.get("situations") or {}).get(bucket) or {}).get(head)
    if situation_rate is not None:
        return float(situation_rate)
    global_rate = (general.get("global") or {}).get(head)
    if global_rate is not None:
        return float(global_rate)
    return _PRIORS[head]


def _opponent_rate(
    hierarchy: Mapping[str, Any],
    opponent_id: str,
    head: str,
    play_key: str,
    general_rate: float,
) -> tuple[float, int]:
    tree = ((hierarchy.get("opponents") or {}).get(opponent_id) or {})
    node = (tree.get("plays") or {}).get(play_key)
    if not node or int(node[head][1]) <= 0:
        family_key = "|".join(play_key.split("|")[:2])
        node = (tree.get("families") or {}).get(family_key)
    if not node or int(node[head][1]) <= 0:
        node = (tree.get("situations") or {}).get(play_key.split("|", 1)[0])
    if not node or int(node[head][1]) <= 0:
        node = tree.get("global")
    if not node or int(node[head][1]) <= 0:
        return general_rate, 0
    total, n = float(node[head][0]), int(node[head][1])
    return _shrink(total, n, general_rate, OPPONENT_STRENGTH), n


def _residual(
    hierarchy: Mapping[str, Any],
    opponent_id: str,
    table: str,
    key: str | None,
    head: str,
    parent: float,
) -> float:
    if not key:
        return 0.0
    general = ((hierarchy.get("general") or {}).get(table) or {}).get(key) or {}
    base = float(general.get(head, parent))
    tree = ((hierarchy.get("opponents") or {}).get(opponent_id) or {})
    node = (tree.get(table) or {}).get(key)
    if node and int(node[head][1]) > 0:
        total, n = float(node[head][0]), int(node[head][1])
        rate = _shrink(total, n, base, RESIDUAL_STRENGTH)
        weight = n / (n + RESIDUAL_STRENGTH)
        return weight * (rate - parent)
    if head in general:
        return 0.15 * (base - parent)
    return 0.0


def context_from_situation(sit: Any) -> dict[str, Any]:
    extras = getattr(sit, "extras", None) or {}
    if not isinstance(extras, Mapping):
        extras = {}
    return {
        "down": getattr(sit, "down", None),
        "distance": getattr(sit, "distance", None),
        "red_zone": bool(getattr(sit, "red_zone", False)),
        "goal_line": bool(getattr(sit, "goal_line", False)),
        "two_minute": bool(getattr(sit, "two_minute", False)),
        "score_for": getattr(sit, "score_us", None),
        "score_against": getattr(sit, "score_them", None),
        "quarter": extras.get("quarter"),
        "opp_personnel": getattr(sit, "opp_personnel", None) or extras.get("opp_personnel"),
        "opp_formation": getattr(sit, "opp_formation", None) or extras.get("opp_formation"),
    }


def predict_cost(
    hierarchy: Mapping[str, Any] | None,
    *,
    play: str,
    context: Mapping[str, Any],
    opponent_id: str = "",
) -> dict[str, Any]:
    """Predict outcome cost from general knowledge plus a shrunk opponent residual."""
    family = call_family(play) or "unknown"
    bucket = situation_bucket(context)
    family_key = f"{bucket}|{family}"
    play_key = f"{bucket}|{family}|{play}"
    rates: dict[str, float] = {}
    general_rates: dict[str, float] = {}
    opponent_n = 0
    for head in _HEADS:
        general = _chain_rate(hierarchy or {}, head, play_key, family_key, bucket)
        general_rates[head] = general
        rate, n = _opponent_rate(hierarchy or {}, opponent_id, head, play_key, general)
        opponent_n = max(opponent_n, n)
        rate += _residual(hierarchy or {}, opponent_id, "personnel", _token(context.get("opp_personnel")), head, general)
        rate += _residual(hierarchy or {}, opponent_id, "formations", _token(context.get("opp_formation")), head, general)
        rate += _residual(
            hierarchy or {}, opponent_id, "score_phase",
            _score_phase(context.get("score_for"), context.get("score_against")), head, general,
        )
        rate += _residual(hierarchy or {}, opponent_id, "clock", _clock_bucket(context), head, general)
        if head != "yards":
            rate = min(0.98, max(0.01, rate))
        else:
            rate = max(0.0, rate)
        rates[head] = rate
    cost = (
        COST_WEIGHTS["conversion"] * rates["conv"]
        + COST_WEIGHTS["yards"] * rates["yards"]
        + COST_WEIGHTS["explosive"] * rates["expl"]
        + COST_WEIGHTS["touchdown"] * rates["td"]
    )
    return {
        "cost": cost,
        "conversion": rates["conv"],
        "expected_yards": rates["yards"],
        "explosive": rates["expl"],
        "touchdown": rates["td"],
        "general_conversion": general_rates["conv"],
        "opponent_snaps_used": opponent_n,
        "opponent_separated": True,
        "post_snap_features_used": False,
        "play_key": play_key,
        "family": family,
    }


def objective_cost(*, conversion: float, yards: float, explosive: float, touchdown: float) -> float:
    """Public cost used by tests. A long touchdown outranks a routine conversion."""
    return (
        COST_WEIGHTS["conversion"] * conversion
        + COST_WEIGHTS["yards"] * yards
        + COST_WEIGHTS["explosive"] * explosive
        + COST_WEIGHTS["touchdown"] * touchdown
    )


def _bootstrap_mean_diff(
    model_losses: Sequence[float],
    baseline_losses: Sequence[float],
    *,
    seed: int,
    draws: int = 400,
) -> dict[str, float | None]:
    """Mean of baseline loss minus model loss. Positive means the model fits better."""
    n = len(model_losses)
    if n == 0 or n != len(baseline_losses):
        return {"diff": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        total = 0.0
        for _i in range(n):
            index = rng.randrange(n)
            total += baseline_losses[index] - model_losses[index]
        diffs.append(total / n)
    diffs.sort()
    point = sum(baseline_losses[i] - model_losses[i] for i in range(n)) / n
    low = diffs[int(0.025 * (draws - 1))]
    high = diffs[int(0.975 * (draws - 1))]
    return {"diff": point, "ci_low": low, "ci_high": high}


def evaluate_defense_hierarchy(
    rows: Sequence[Mapping[str, Any]],
    *,
    book: Mapping[str, Sequence[str]] | None = None,
    seed: int = 7,
    latency_ms: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Hold out by game. Score the executed play, not an uncalled alternative.

    The baseline is the situation-only general rate, which is the hierarchical
    parent with the play and family removed. Beating it is a better prediction
    of what happened, not evidence that a different call would have saved a score.
    """
    audit = audit_defensive_rows(rows)
    supervised = [row for row in rows if classify_defensive_candidate(row)[0] == "supervised"]
    games = sorted({str(row["game_id"]) for row in supervised})
    model_brier: list[float] = []
    base_brier: list[float] = []
    model_yards: list[float] = []
    base_yards: list[float] = []
    by_situation: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0})
    if len(games) >= 2:
        for held in games:
            train = [row for row in supervised if str(row["game_id"]) != held]
            test = [row for row in supervised if str(row["game_id"]) == held]
            if not train or not test:
                continue
            fitted = fit_defense_hierarchy(train)
            for row in test:
                labels = outcomes_from_row(row)
                if labels["conv"] is None:
                    continue
                context = dict(row)
                pred = predict_cost(
                    fitted, play=str(row.get("executed_play") or ""),
                    context=context, opponent_id=str(row.get("opponent_id") or ""),
                )
                # Situation-only: wipe play/family tables so the chain stops at the bucket.
                situation_only = {
                    "general": {
                        "global": fitted["general"]["global"],
                        "situations": fitted["general"]["situations"],
                        "families": {},
                        "plays": {},
                        "personnel": {},
                        "formations": {},
                        "score_phase": {},
                        "clock": {},
                    },
                    "opponents": {},
                }
                base = predict_cost(
                    situation_only, play=str(row.get("executed_play") or ""),
                    context=context, opponent_id="",
                )
                actual = float(labels["conv"])
                model_brier.append((pred["conversion"] - actual) ** 2)
                base_brier.append((base["conversion"] - actual) ** 2)
                if labels["yards"] is not None:
                    model_yards.append(abs(pred["expected_yards"] - float(labels["yards"])))
                    base_yards.append(abs(base["expected_yards"] - float(labels["yards"])))
                bucket = situation_bucket(row)
                by_situation[bucket]["n"] += 1
    heuristic = _heuristic_selection(supervised, book=book)
    ci = _bootstrap_mean_diff(model_brier, base_brier, seed=seed)
    p95 = _percentile(list(latency_ms or []), 0.95)
    opponents = len({str(row.get("opponent_id") or "") for row in supervised})
    reasons = []
    if audit["supervised_snaps"] < MIN_SUPERVISED:
        reasons.append(f"supervised {audit['supervised_snaps']} < {MIN_SUPERVISED}")
    if opponents < MIN_OPPONENTS:
        reasons.append(f"opponents {opponents} < {MIN_OPPONENTS}")
    if len(games) < MIN_GAMES:
        reasons.append(f"games {len(games)} < {MIN_GAMES}")
    if len(model_brier) < MIN_HOLDOUT_SNAPS:
        reasons.append(f"holdout {len(model_brier)} < {MIN_HOLDOUT_SNAPS}")
    if ci["ci_low"] is None or float(ci["ci_low"]) <= 0:
        reasons.append("conversion Brier improvement CI does not exclude 0")
    if p95 is None or p95 >= LATENCY_P95_MS:
        reasons.append("latency p95 missing or not under 150 ms")
    allowed = not reasons
    return {
        "version": HIERARCHY_VERSION,
        "audit": audit,
        "holdout": "leave_one_game_out",
        "holdout_snaps": len(model_brier),
        "games": games,
        "opponents": opponents,
        "baseline_brier": (sum(base_brier) / len(base_brier)) if base_brier else None,
        "model_conversion_brier": (sum(model_brier) / len(model_brier)) if model_brier else None,
        "model_yards_mae": (sum(model_yards) / len(model_yards)) if model_yards else None,
        "baseline_yards_mae": (sum(base_yards) / len(base_yards)) if base_yards else None,
        "brier_improvement": ci,
        "by_situation": dict(by_situation),
        "heuristic_comparison": heuristic,
        "latency_p95_ms": p95,
        "promotion": {
            "allowed": allowed,
            "reason": "passed" if allowed else "; ".join(reasons),
            "auto_promote": False,
            "gates": {
                "min_supervised": MIN_SUPERVISED,
                "min_opponents": MIN_OPPONENTS,
                "min_games": MIN_GAMES,
                "min_holdout": MIN_HOLDOUT_SNAPS,
                "latency_p95_ms": LATENCY_P95_MS,
                "ci_excludes_zero": True,
            },
        },
        "limitations": [
            "Held-out scores predict the defensive play that was executed.",
            "They are not the yards an uncalled play would have allowed.",
            "A lower predicted cost is not a causal claim about a different call.",
            "Top-play disagreement with the heuristic is a ranking difference, not a result.",
            "Opponent residuals stay out of the general tables. One opponent cannot be promoted.",
        ],
    }


def _situation_for_row(row: Mapping[str, Any]) -> Any:
    from cfb_coach.madden.situation import Situation

    def _num(value: Any) -> int | None:
        try:
            return int(value) if value is not None and value != "" else None
        except (TypeError, ValueError):
            return None

    return Situation(
        raw="defense holdout",
        side="defense",
        down=_num(row.get("down")),
        distance=_num(row.get("distance")),
        red_zone=bool(row.get("red_zone")),
        goal_line=bool(row.get("goal_line")),
        two_minute=bool(row.get("two_minute")),
    )


def _heuristic_selection(
    supervised: Sequence[Mapping[str, Any]],
    *,
    book: Mapping[str, Sequence[str]] | None,
) -> dict[str, Any]:
    """Rank the confirmed book with the fit and with no artifact.

    Leave-one-game-out refits inside the caller. This comparison uses the
    plays in ``book`` and does not invent an outcome for the play that lost.
    """
    if not book:
        return {
            "available": False,
            "reason": "no confirmed defensive playbook was supplied",
            "top_play_agreement": None,
            "model_top_was_executed": None,
            "heuristic_top_was_executed": None,
        }
    games = sorted({str(row["game_id"]) for row in supervised})
    if len(games) < 2:
        return {
            "available": False,
            "reason": "need at least two games to compare rankings out of sample",
            "top_play_agreement": None,
            "model_top_was_executed": None,
            "heuristic_top_was_executed": None,
        }
    from cfb_coach.madden.model.defense_coordinator import rank_defense

    compared = 0
    agreed = 0
    model_hit = 0
    heuristic_hit = 0
    for held in games:
        train = [row for row in supervised if str(row["game_id"]) != held]
        test = [row for row in supervised if str(row["game_id"]) == held]
        if not train or not test:
            continue
        fitted = fit_defense_hierarchy(train)
        wrapper = {
            "hierarchy": fitted,
            "supervised_defensive_snaps": int(fitted.get("supervised_snaps") or 0),
        }
        for row in test:
            sit = _situation_for_row(row)
            opponent_id = str(row.get("opponent_id") or "")
            model_rank = rank_defense(sit, book, wrapper, opponent_id=opponent_id)
            heur_rank = rank_defense(sit, book, None, opponent_id=opponent_id)
            if not model_rank or not heur_rank:
                continue
            compared += 1
            same = (
                model_rank[0]["formation"], model_rank[0]["play"],
            ) == (
                heur_rank[0]["formation"], heur_rank[0]["play"],
            )
            agreed += int(same)
            executed = str(row.get("executed_play") or "")
            if model_rank[0]["play"] == executed:
                model_hit += 1
            if heur_rank[0]["play"] == executed:
                heuristic_hit += 1
    return {
        "available": compared > 0,
        "holdout_snaps": compared,
        "top_play_agreement": (agreed / compared) if compared else None,
        "model_top_was_executed": model_hit if compared else None,
        "heuristic_top_was_executed": heuristic_hit if compared else None,
        "claim_limit": (
            "Matching the executed play is a ranking comparison on the confirmed "
            "book. It is not the result an uncalled play would have produced."
        ),
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def promotion_status(model: Mapping[str, Any] | None) -> dict[str, Any]:
    """True only after a recorded eval passed. Enable is still a separate write."""
    hierarchy = (model or {}).get("hierarchy") or {}
    promo = hierarchy.get("promotion") or {}
    reason = str(promo.get("reason") or "eval_not_run")
    allowed = bool(promo.get("allowed")) and reason not in {"eval_not_run", "None"}
    return {
        "allowed": allowed,
        "reason": reason,
        "auto_promote": False,
        "supervised_snaps": int(
            hierarchy.get("supervised_snaps")
            or (model or {}).get("supervised_defensive_snaps")
            or 0
        ),
    }


def record_promotion(model: dict[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Attach an eval verdict. Does not set the live mode."""
    hierarchy = dict(model.get("hierarchy") or {})
    promo = dict((report.get("promotion") or {}))
    promo["auto_promote"] = False
    hierarchy["promotion"] = promo
    model["hierarchy"] = hierarchy
    model["auto_promote"] = False
    return model


def go_nogo_report(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    report: Mapping[str, Any] | None = None,
    reported_candidates: int = 178,
) -> dict[str, Any]:
    """Experimental defensive playtesting verdict. Default on an empty tree is NO-GO."""
    audit = audit_defensive_rows(list(rows or []), reported_candidates=reported_candidates)
    promo = (report or {}).get("promotion") or {}
    allowed = bool(promo.get("allowed"))
    verdict = "GO" if allowed else "NO-GO"
    reasons = []
    if not allowed:
        reasons.append(promo.get("reason") or "no passing defensive eval is stored")
    if audit["supervised_snaps"] < MIN_SUPERVISED:
        reasons.append(
            f"verified supervised snaps are {audit['supervised_snaps']}, "
            f"below {MIN_SUPERVISED}"
        )
    if audit["candidate_snaps"] == 0:
        reasons.append(
            "this checkout has no franchise database. Issue #75 records 45 "
            "established defensive snaps as aggregates, not rows. A reported "
            f"{int(reported_candidates)} was not found as rows here and was not "
            "treated as supervised examples"
        )
    launch = (
        'python3 -m cfb_coach ml defense-experimental --enable --db "$CFB_COACH_MADDEN_DB"'
    )
    rollback = (
        'python3 -m cfb_coach ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"'
    )
    return {
        "verdict": verdict,
        "experimental_defense_live": False if verdict == "NO-GO" else True,
        "heuristic_remains_default": True,
        "auto_promote": False,
        "audit": audit,
        "eval": report,
        "reasons": reasons,
        "launch_command": launch if verdict == "GO" else None,
        "enable_command_refuses_until_go": launch,
        "rollback_command": rollback,
        "train_command": (
            'python3 -m cfb_coach ml defense-experimental --retrain '
            '--db "$CFB_COACH_MADDEN_DB" --out /ABSOLUTE/PATH/defense_hierarchy.json'
        ),
        "audit_command": (
            'python3 -m cfb_coach ml defense-audit --db "$CFB_COACH_MADDEN_DB" '
            "--reported-candidates 178"
        ),
        "eval_command": (
            'python3 -m cfb_coach ml defense-hierarchy-eval --db "$CFB_COACH_MADDEN_DB" '
            "--model /ABSOLUTE/PATH/defense_hierarchy.json --seed 7"
        ),
        "offense_artifact": "not modified",
        "limitations": [
            "Do not enable experimental defense from this report unless verdict is GO.",
            "A missing artifact, or an artifact whose promotion.allowed is false, leaves the heuristic caller active.",
            "Predictions on held-out snaps are not counterfactual yards for plays that were not called.",
        ],
    }


# Issue #75 records aggregates. These constants are that public summary, not snaps.
ISSUE_75_PUBLIC_AGGREGATE = {
    "issue": 75,
    "build": "8ddbc29",
    "result": "loss 45-42",
    "last_logged_snap_score": "42-42",
    "final_points_absent_from_snap_log": True,
    "logged_offensive_interceptions": 3,
    "logged_defensive_interceptions_forced": 1,
    "user_reported_interceptions_thrown": 4,
    "drive_id": "empty on all snaps in the issue summary",
    "game_clock": "mostly empty in the issue summary",
    "executed_status": "unknown on all snaps in the issue summary",
}


def turnover_attribution(row: Mapping[str, Any]) -> str:
    """Unknown unless the row itself carries a cause and supporting evidence."""
    cause = str(row.get("turnover_cause") or "").strip().lower()
    evidence = str(row.get("turnover_evidence") or "").strip()
    if cause in {"read", "execution", "playcall"} and evidence:
        return cause
    return "unknown"


def _is_interception(row: Mapping[str, Any]) -> bool:
    kind = str(row.get("result_kind") or row.get("kind") or "").lower()
    result = str(row.get("result") or "").lower()
    if kind in {"int", "interception"}:
        return True
    if "interception" in result or result.strip() in {"int", "picked"}:
        return True
    return False


def reconcile_interceptions(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare issue #75's three logged throws with the user's report of four.

    This function does not add a snap. The defensive interception forced on
    John is a different event and is not counted as a throw by the user.
    """
    offense = []
    forced = []
    for row in rows or []:
        if not _is_interception(row):
            continue
        item = {
            "game_id": row.get("game_id"),
            "side": row.get("side"),
            "play": row.get("executed_play") or row.get("recommended_play"),
            "attribution": turnover_attribution(row),
        }
        side = str(row.get("side") or "").lower()
        if side in {"defense", "d"}:
            forced.append(item)
        else:
            offense.append(item)
    return {
        "issue_75": dict(ISSUE_75_PUBLIC_AGGREGATE),
        "rows_examined": len(list(rows or [])),
        "offensive_interceptions_in_rows": len(offense),
        "defensive_interceptions_forced_in_rows": len(forced),
        "offensive_attributions": offense,
        "snap_created": False,
        "fourth_interception": {
            "status": "unresolved",
            "snap_created": False,
            "attribution": "unknown",
            "why": (
                "Issue #75 records 3 offensive interceptions in the snap log. "
                "The user reports 4 thrown against John. The 1 interception the "
                "defense forced is John's throw, not a fourth throw by the user. "
                "The final score moved from a last logged 42-42 to 45-42, a "
                "3-point gap, which is not a pick-six. No available row identifies "
                "the fourth throw, so none was written."
            ),
        },
    }
