"""Offline observational eval: trained defense model versus the heuristic caller.

Held-out human games only. Rankings of unexecuted alternatives are not treated
as causal proof that a different call would have changed the snap.
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from cfb_coach.madden.defense_select import call_family
from cfb_coach.madden.model.defense_coordinator import (
    LATENCY_MS,
    load_model,
    rank_defense,
    train_defense,
)
from cfb_coach.madden.model.defense_intelligence import is_counted_defensive_snap
from cfb_coach.madden.situation import Situation

DEFAULT_BOOK = {
    "4-3 Over": ["Cover 3 Sky", "Cover 1 Contain", "OLB Fire Man", "LB Sting I", "Cover 4 Quarters"],
    "Nickel 3-3 Over": ["Bracket Switch Willie", "Cover 3 Match", "Cover 2 Sink", "Mid Blitz 0"],
    "Dime 2-3": ["Cover 3 Sky", "Cover 4 Drop", "Cover 2 Man"],
    "Dollar 3-2 Mug": ["Cover 3 Sky", "Cover 1 Robber", "Zero Blitz"],
}


def _sit(row: Mapping[str, Any]) -> Situation:
    down = row.get("down") or 1
    distance = row.get("distance") or 10
    try:
        down_i, dist_i = int(down), int(distance)
    except (TypeError, ValueError):
        down_i, dist_i = 1, 10
    extras = dict(row.get("extras") or {})
    return Situation(
        raw=str(row.get("situation_raw") or f"{down_i}&{dist_i}"),
        down=down_i,
        distance=dist_i,
        yardline=int(row["yardline"]) if str(row.get("yardline") or "").isdigit() else 40,
        side="defense",
        red_zone=bool(row.get("red_zone")),
        goal_line=bool(row.get("goal_line")),
        two_minute=bool(row.get("two_minute")),
        extras=extras,
    )


def _book(row: Mapping[str, Any], fallback: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    raw = row.get("installed_book") or row.get("book")
    if isinstance(raw, dict) and raw:
        return {str(k): [str(p) for p in v] for k, v in raw.items()}
    return {str(k): list(v) for k, v in fallback.items()}


def _outcome_flags(row: Mapping[str, Any]) -> dict[str, Any]:
    yards = None
    try:
        yards = int(row["yards"])
    except (KeyError, TypeError, ValueError):
        yards = None
    success = str(row.get("success") or "").lower() == "true"
    try:
        distance = int(row["distance"])
    except (KeyError, TypeError, ValueError):
        distance = None
    conversion = yards is not None and distance is not None and yards >= distance
    explosive = yards is not None and yards >= 15
    house = yards is not None and yards >= 40
    td = str(row.get("result") or "").lower() in ("td", "touchdown") or bool(row.get("touchdown"))
    return {
        "stop": success,
        "conversion_allowed": conversion,
        "explosive": explosive,
        "house": house,
        "td": td,
        "yards": yards,
    }


def evaluate_defense(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Mapping[str, Any] | None = None,
    book: Mapping[str, Sequence[str]] | None = None,
    holdout_games: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare the trained ranker with the established heuristic on held-out snaps.

    Metrics are observational: they score the executed play and the model's
    ranking of that play. They do not claim a counterfactual win.
    """
    verified = [r for r in rows if is_counted_defensive_snap(r)]
    games = sorted({str(r.get("game_id") or "") for r in verified if r.get("game_id")})
    if holdout_games:
        hold = {str(g) for g in holdout_games}
    elif len(games) >= 2:
        hold = {games[-1]}
    else:
        hold = set(games)
    train_rows = [r for r in verified if str(r.get("game_id") or "") not in hold]
    test_rows = [r for r in verified if str(r.get("game_id") or "") in hold]
    artifact = dict(model) if model is not None else train_defense(train_rows)
    fallback_book = book or DEFAULT_BOOK

    latencies: list[float] = []
    model_top_stop = 0
    heur_family_match = 0
    model_family_match = 0
    converted = 0
    explosives = 0
    tds = 0
    ranked_n = 0
    families_called: dict[str, int] = {}
    calibrated = []
    independent = 0

    for row in test_rows:
        sit = _sit(row)
        menu = _book(row, fallback_book)
        started = time.perf_counter()
        ranked = rank_defense(
            sit, menu, artifact,
            opponent_id=str(row.get("opponent_id") or ""),
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed)
        if not ranked:
            continue
        ranked_n += 1
        top = ranked[0]
        families_called[str(top["family"])] = families_called.get(str(top["family"]), 0) + 1
        executed = str(row.get("executed_play") or "")
        executed_family = call_family(executed)
        flags = _outcome_flags(row)
        converted += int(flags["conversion_allowed"])
        explosives += int(flags["explosive"])
        tds += int(flags["td"])
        if flags["stop"]:
            model_top_stop += int(bool(top.get("verified_play_n") or top.get("verified_family_n")))
        if executed_family and top["family"] == executed_family:
            model_family_match += 1
        match = next((item for item in ranked if item["play"] == executed), None)
        if match is not None:
            calibrated.append({
                "play": executed,
                "model_score": match["score"],
                "rank": next(i for i, item in enumerate(ranked, 1) if item["play"] == executed),
                "stop": flags["stop"],
                "verified_play_n": match.get("verified_play_n") or 0,
            })
            if int(match.get("verified_play_n") or 0) > 0:
                independent += 1
        from cfb_coach.madden.defense_select import mix_key, MIX
        ctx = mix_key(sit)
        heur_family = max(MIX.get(ctx) or MIX["early"], key=(MIX.get(ctx) or MIX["early"]).get)
        if executed_family == heur_family:
            heur_family_match += 1

    p95 = 0.0
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    n = max(1, ranked_n)
    return {
        "schema": "madden.defense.eval.v1",
        "model_version": artifact.get("model_version"),
        "evidence_quality": artifact.get("evidence_quality"),
        "supervised_defensive_snaps_train": artifact.get("supervised_defensive_snaps"),
        "holdout_games": sorted(hold),
        "holdout_snaps": len(test_rows),
        "ranked_snaps": ranked_n,
        "observational_only": True,
        "counterfactual_claim": False,
        "latency_ms": {
            "p50": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else 0.0,
            "p95": round(p95, 3),
            "max": round(max(latencies) if latencies else 0.0, 3),
            "budget": LATENCY_MS,
            "over_budget": sum(1 for x in latencies if x > LATENCY_MS),
        },
        "model": {
            "family_match_executed": round(model_family_match / n, 4),
            "uses_trained_play_evidence": independent,
            "coverage_diversity": families_called,
            "top_has_trained_stop_signal": model_top_stop,
        },
        "heuristic": {
            "family_match_executed": round(heur_family_match / n, 4),
        },
        "holdout_outcomes": {
            "conversions_allowed": converted,
            "explosives_ge_15": explosives,
            "touchdowns": tds,
        },
        "calibration_rows": calibrated,
        "limitations": [
            "Unexecuted alternatives have no observed outcome.",
            "Family match is not proof the model caused a stop.",
            "Small held-out n is not a causal Wild Card claim.",
            "John-specific rates are shrunk toward the general prior and held out by game.",
        ],
        "independent_of_offense": True,
    }


def load_and_evaluate(path: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return evaluate_defense(rows, model=load_model(path))
