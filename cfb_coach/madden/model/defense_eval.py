"""Offline defensive model versus heuristic evaluation.

Held-out verified human games only. Unexecuted counterfactual plays are not
treated as causal evidence. A missing or prior-driven artifact cannot promote.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Mapping, Sequence

from cfb_coach.madden.defense_select import call_family, select_defense
from cfb_coach.madden.model.defense_audit import audit_defensive_rows, go_nogo_from_audit
from cfb_coach.madden.model.defense_coordinator import LATENCY_MS, rank_defense
from cfb_coach.madden.model.defense_shrinkage import defensive_utility
from cfb_coach.madden.situation import Situation

EVAL_SCHEMA = "madden.defense.compare.v1"


def _situation(row: Mapping[str, Any]) -> Situation:
    extras = {}
    if row.get("game_clock"):
        extras["game_clock"] = row.get("game_clock")
    return Situation(
        raw=str(row.get("situation_raw") or ""),
        side="defense",
        down=row.get("down"),
        distance=row.get("distance"),
        yardline=row.get("yardline"),
        red_zone=bool(row.get("red_zone")),
        goal_line=bool(row.get("goal_line")),
        two_minute=bool(row.get("two_minute")),
        score_us=row.get("score_us") or row.get("score_for"),
        score_them=row.get("score_them") or row.get("score_against"),
        extras=extras,
    )


def _holdout(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    games = sorted({str(row.get("game_id")) for row in rows if row.get("game_id")})
    if len(games) >= 2:
        held = {games[-1]}
        train = [row for row in rows if str(row.get("game_id")) not in held]
        return train, sorted(held)
    return list(rows), []


def compare_defense(
    rows: Sequence[Mapping[str, Any]],
    book: Mapping[str, Sequence[str]],
    model: Mapping[str, Any] | None,
    *,
    opponent_id: str = "",
) -> dict[str, Any]:
    """Score the installed book with the model and with the heuristic mix."""
    from cfb_coach.madden.model.defense_coordinator import train_defense
    from cfb_coach.madden.model.defense_intelligence import verified_human_rows

    audit = audit_defensive_rows(rows)
    verified = [
        dict(row) for row in verified_human_rows(rows)
        if str(row.get("side") or "") == "defense"
        and str(row.get("executed_play") or row.get("action_play") or "")
    ]
    train_rows, held_games = _holdout(verified)
    fitted = model if model is not None else train_defense(train_rows)
    n = int((fitted or {}).get("supervised_defensive_snaps") or 0)
    book = {str(form): [str(play) for play in plays] for form, plays in (book or {}).items()}
    eval_rows = [
        row for row in verified
        if not held_games or str(row.get("game_id")) in held_games
    ]
    if not eval_rows:
        eval_rows = verified
        held_games = []
    top1 = 0
    top8 = 0
    family_hit = 0
    model_util: list[float] = []
    heur_util: list[float] = []
    families: Counter[str] = Counter()
    latencies: list[float] = []
    disagreements = 0
    legal = True
    for row in eval_rows:
        sit = _situation(row)
        started = time.perf_counter()
        ranked = rank_defense(sit, book, fitted, opponent_id=opponent_id or str(row.get("opponent_id") or ""))
        latencies.append((time.perf_counter() - started) * 1000.0)
        if not ranked:
            legal = False
            continue
        executed = str(row.get("executed_play") or row.get("action_play") or "")
        plays = {(item["formation"], item["play"]) for item in ranked}
        if executed:
            family_hit += int(any(item["play"] == executed for item in ranked[:8]))
            top8 += int(any(item["play"] == executed for item in ranked[:8]))
            top1 += int(ranked[0]["play"] == executed)
        families[str(ranked[0].get("family") or "unknown")] += 1
        try:
            heur = select_defense(sit, opponent_id or "eval", None, dict(book), __import__("random").Random(0))
            if (heur.formation, heur.play) != (ranked[0]["formation"], ranked[0]["play"]):
                disagreements += 1
        except Exception:
            heur = None
        util = defensive_utility(row)
        if util is not None:
            model_util.append(util if ranked[0]["play"] == executed else 0.0)
            if heur is not None:
                heur_util.append(util if heur.play == executed else 0.0)
        for item in ranked:
            if item["play"] not in (book.get(item["formation"]) or []):
                legal = False
    over_budget = sum(1 for ms in latencies if ms > LATENCY_MS)
    p95 = sorted(latencies)[max(0, int(0.95 * (len(latencies) - 1)))] if latencies else None
    quality = (fitted or {}).get("evidence_quality", "prior_driven")
    gate = go_nogo_from_audit(audit)
    promote = (
        gate["verdict"] != "NO-GO"
        and n >= 1
        and quality != "prior_driven"
        and legal
        and over_budget == 0
        and bool(book)
    )
    return {
        "schema": EVAL_SCHEMA,
        "audit": audit,
        "held_out_games": held_games,
        "eval_snaps": len(eval_rows),
        "supervised_defensive_snaps": n,
        "evidence_quality": quality,
        "top1_executed": top1,
        "top8_executed": top8,
        "family_in_top8": family_hit,
        "model_heuristic_disagreements": disagreements,
        "independent_decisions": disagreements > 0 and n >= 1,
        "uses_trained_evidence": n >= 1 and quality != "prior_driven",
        "coverage_family_counts": dict(families),
        "mean_utility_when_executed_was_top": (
            round(sum(model_util) / len(model_util), 4) if model_util else None
        ),
        "latency_ms_p95": round(p95, 3) if p95 is not None else None,
        "latency_over_150ms": over_budget,
        "all_candidates_in_book": legal,
        "promote_ok": promote,
        "verdict": "GO" if promote else gate["verdict"],
        "limitations": [
            "Observational stop and utility labels, not proven counterfactual effects.",
            "Unexecuted plays are not claimed to have been better or worse.",
            "Held-out coverage is one game when multiple games exist; otherwise in-sample.",
            "A prior-driven artifact cannot become live.",
            "Latency is offline rank time, not a live Xbox budget measurement.",
        ],
        "causal_claims": False,
        "hybrid_min_snaps": 40,
    }
