"""Joint offensive decision engine: (formation, play, adjustment_plan).

Replaces the sequential play-then-adjustment path for experimental mode while
preserving research eligibility, macro safety, NO_ADJUSTMENT competition and
the 150 ms latency budget via bounded candidate search.
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.football_situation import (
    evaluate_situation, play_situation_fit,
)
from cfb_coach.madden.model.offense_action_policy import choose_offense_action
from cfb_coach.madden.model.schema import ML_LATENCY_BUDGET_MS

POLICY = "joint_offensive_action.v1"
# Bound joint search: evaluate adjustment plans for the top-K play candidates
# plus a thin exploration slice from the remainder, never a fixed 5–6 repertoire.
TOP_PLAY_JOINT = 18
EXPLORE_PLAY_JOINT = 6
EXECUTION_COST_MACRO = 0.018
EXECUTION_COST_ADJUSTMENT = 0.012
EXECUTION_COST_MULTI = 0.022
COMPLEXITY_UNCERTAINTY = 0.01


def _adjustment_plan_from_choice(choice: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(choice.get("kind") or "none")
    if kind == "none":
        return {
            "kind": "none",
            "id": "NO_ADJUSTMENT",
            "label": "NO ADJUSTMENT",
            "components": [],
            "macro": None,
            "adjustment": None,
            "buttons": None,
            "armed": False,
            "verified": True,
            "score": float(choice.get("no_action_score") or choice.get("score") or 0.0),
            "reason": choice.get("reason") or "run selected play unchanged",
        }
    payload = choice.get("macro") if kind == "macro" else choice.get("adjustment")
    components = []
    if payload:
        components.append({
            "kind": kind,
            "id": payload.get("id") or choice.get("id"),
            "type": payload.get("kind") or kind,
            "buttons": payload.get("buttons"),
            "settings": payload.get("settings"),
            "sources": payload.get("source_ids") or payload.get("sources"),
        })
    return {
        "kind": kind,
        "id": choice.get("id"),
        "label": (
            (payload or {}).get("name")
            or (payload or {}).get("label")
            or str(choice.get("id") or kind)
        ),
        "components": components,
        "macro": choice.get("macro"),
        "adjustment": choice.get("adjustment"),
        "buttons": (payload or {}).get("buttons"),
        "armed": bool((payload or {}).get("verified_armed") or kind == "macro"),
        "verified": True,
        "score": float(choice.get("top_action_score") or choice.get("score") or 0.0),
        "reason": choice.get("reason") or "",
        "candidates": choice.get("candidates"),
        "observational_evidence_mode": choice.get("observational_evidence_mode"),
        "observation_not_causal": choice.get("observation_not_causal", True),
    }


def _execution_cost(plan: Mapping[str, Any]) -> float:
    kind = str(plan.get("kind") or "none")
    n = len(plan.get("components") or [])
    if kind == "none":
        return 0.0
    if n > 1:
        return EXECUTION_COST_MULTI
    if kind == "macro":
        return EXECUTION_COST_MACRO
    return EXECUTION_COST_ADJUSTMENT


def _legal_multi_components(components: Sequence[Mapping[str, Any]]) -> bool:
    """Reject conflicting hot-route targets or stacked incompatible protections."""
    targets: set[str] = set()
    protections = 0
    for item in components:
        typ = str(item.get("type") or item.get("kind") or "")
        if typ in ("hot_route", "adjustment") and "protect" not in typ:
            settings = item.get("settings") or []
            for row in settings:
                setting = str((row or {}).get("setting") or "").strip().upper()
                if setting:
                    if setting in targets:
                        return False
                    targets.add(setting)
            # Single researched adjustment payloads may encode target in id/label.
            target = str(item.get("target") or "").strip().upper()
            if target:
                if target in targets:
                    return False
                targets.add(target)
        if "protect" in typ or typ == "pass_protection":
            protections += 1
            if protections > 1:
                return False
    return True


def score_joint_candidate(
    *,
    play_row: Mapping[str, Any],
    plan: Mapping[str, Any],
    situation: Any,
    football: Any | None = None,
) -> dict[str, Any]:
    """Joint score = play model probability + situation fit + action value − costs."""
    fb = football or evaluate_situation(situation)
    play = str(play_row["play"])
    formation = str(play_row["formation"])
    base = float(play_row.get("selection_score", play_row.get("probability", 0.5)) or 0.5)
    fit, fit_reasons = play_situation_fit(play, fb)
    action_raw = float(plan.get("score") or 0.0)
    # NO_ADJUSTMENT competes with its threshold score; other actions use their
    # research/observational score as a modest additive term, never as a claim
    # of causal uplift. Cap the action contribution tightly.
    if plan.get("kind") == "none":
        action_term = 0.0
        no_adj_bonus = 0.008  # slight preference for simplicity when close
    else:
        action_term = max(-0.05, min(0.06, (action_raw - 0.30) * 0.35))
        no_adj_bonus = 0.0
    cost = _execution_cost(plan)
    uncertainty = float(play_row.get("uncertainty", 1.0) or 1.0)
    if plan.get("kind") != "none":
        cost += COMPLEXITY_UNCERTAINTY * uncertainty
    joint = base + fit + action_term + no_adj_bonus - cost
    return {
        "formation": formation,
        "play": play,
        "probability": float(play_row.get("probability", 0.5) or 0.5),
        "uncertainty": uncertainty,
        "play_selection_score": base,
        "situation_fit": fit,
        "situation_reasons": fit_reasons,
        "adjustment_plan": plan,
        "action_term": round(action_term, 5),
        "execution_cost": round(cost, 5),
        "joint_score": round(joint, 7),
        "policy": POLICY,
        "football_situation": fb.as_dict() if hasattr(fb, "as_dict") else fb,
    }


def choose_joint_offense_action(
    *,
    ranked_plays: Sequence[Mapping[str, Any]],
    sit: Any,
    book: dict[str, list[str]],
    active: Sequence[str],
    db: Any = None,
    opponent_id: str = "",
    weights: Mapping[str, float] | None = None,
    cooled: set[str] | None = None,
    score_phase: str | None = None,
    audibles: dict[str, list[str]] | None = None,
    allow_macros: bool = True,
    repeated: bool = False,
    game_memory: Mapping[str, Any] | None = None,
    started: float | None = None,
    budget_ms: float = ML_LATENCY_BUDGET_MS,
) -> dict[str, Any]:
    """Search legal (play × adjustment_plan) pairs under the live latency budget."""
    t0 = started if started is not None else time.perf_counter()
    empty = {
        "policy": POLICY,
        "kind": "none",
        "formation": None,
        "play": None,
        "adjustment_plan": _adjustment_plan_from_choice({"kind": "none", "no_action_score": 0.3}),
        "reason": "no eligible joint candidates",
        "candidates": [],
        "fell_back_play_only": False,
    }
    if not ranked_plays:
        return empty

    memory = dict(game_memory or {})
    fb = evaluate_situation(
        sit,
        repeated_live_look=repeated,
        opponent_tendencies=memory.get("opponent_tendencies"),
        roster_verified=bool(memory.get("roster_verified")),
    )
    # Memory may nudge exploration but must not invent current coverage.
    pressure_bias = float((memory.get("opponent_tendencies") or {}).get("pressure_rate") or 0.0)

    head = list(ranked_plays[:TOP_PLAY_JOINT])
    rest = list(ranked_plays[TOP_PLAY_JOINT:])
    explore = rest[:: max(1, len(rest) // max(1, EXPLORE_PLAY_JOINT))][:EXPLORE_PLAY_JOINT] if rest else []
    play_pool = head + [r for r in explore if r not in head]

    joint_rows: list[dict[str, Any]] = []
    for play_row in play_pool:
        elapsed = (time.perf_counter() - t0) * 1000.0
        if elapsed > budget_ms * 0.82:
            break
        formation = str(play_row["formation"])
        play = str(play_row["play"])
        if play not in (book or {}).get(formation, []):
            continue
        try:
            action = choose_offense_action(
                formation=formation, play=play, sit=sit, book=book,
                active=active, weights=weights, cooled=cooled,
                score_phase=score_phase, audibles=audibles,
                prediction=play_row, allow_macros=allow_macros,
                repeated=repeated, db=db, opponent_id=opponent_id,
            )
        except Exception:  # noqa: BLE001 — never break live path
            continue

        # Always include NO_ADJUSTMENT as a genuine competing option.
        none_plan = _adjustment_plan_from_choice({
            "kind": "none",
            "no_action_score": action.get("no_action_score", 0.3),
            "reason": "NO_ADJUSTMENT competing option",
            "candidates": action.get("candidates"),
        })
        plans = [none_plan]

        if action.get("kind") in ("macro", "adjustment"):
            chosen_plan = _adjustment_plan_from_choice(action)
            if _legal_multi_components(chosen_plan.get("components") or []):
                # Unverified / unarmed macros are already filtered by choose_offense_action.
                plans.append(chosen_plan)

        # Rank alternate eligible action candidates without inventing settings.
        for alt in (action.get("candidates") or [])[:5]:
            if alt.get("kind") == "none":
                continue
            if alt.get("kind") == action.get("kind") and alt.get("id") == action.get("id"):
                continue
            # Candidates list is a summary; only the chosen payload is fully
            # executable. Skip incomplete alts rather than invent buttons.
            continue

        for plan in plans:
            row = score_joint_candidate(
                play_row=play_row, plan=plan, situation=sit, football=fb,
            )
            if fb.pressure_credible or pressure_bias >= 0.55:
                # Soft bias toward protection / quick game when pressure is
                # credible or a high-sample tendency exists — still never
                # claims the current look is known from tendency alone.
                components = plan.get("components") or []
                if any("protect" in str(c.get("type") or "").lower() for c in components):
                    row["joint_score"] = round(row["joint_score"] + 0.01, 7)
                    row["situation_reasons"] = list(row["situation_reasons"]) + [
                        "pressure-context protection consideration"
                    ]
            joint_rows.append(row)

    if not joint_rows:
        # Fallback: best play with NO_ADJUSTMENT (preserves model-primary play).
        top = ranked_plays[0]
        plan = _adjustment_plan_from_choice({"kind": "none", "no_action_score": 0.3})
        row = score_joint_candidate(play_row=top, plan=plan, situation=sit, football=fb)
        return {
            "policy": POLICY,
            "kind": "none",
            "formation": row["formation"],
            "play": row["play"],
            "adjustment_plan": plan,
            "reason": "joint search empty; play-only with NO_ADJUSTMENT",
            "joint_score": row["joint_score"],
            "candidates": [row],
            "football_situation": fb.as_dict(),
            "fell_back_play_only": True,
            "n_plays_considered": 0,
            "n_joint_candidates": 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    joint_rows.sort(
        key=lambda r: (-r["joint_score"], -r["probability"], r["formation"], r["play"],
                       str((r["adjustment_plan"] or {}).get("id")))
    )
    best = joint_rows[0]
    plan = best["adjustment_plan"]
    return {
        "policy": POLICY,
        "kind": plan.get("kind"),
        "id": plan.get("id"),
        "formation": best["formation"],
        "play": best["play"],
        "adjustment_plan": plan,
        "macro": plan.get("macro"),
        "adjustment": plan.get("adjustment"),
        "reason": plan.get("reason") or "; ".join(best.get("situation_reasons") or []),
        "joint_score": best["joint_score"],
        "play_probability": best["probability"],
        "situation_fit": best["situation_fit"],
        "situation_reasons": best["situation_reasons"],
        "candidates": [
            {
                "formation": r["formation"], "play": r["play"],
                "joint_score": r["joint_score"],
                "adjustment": (r["adjustment_plan"] or {}).get("id"),
                "kind": (r["adjustment_plan"] or {}).get("kind"),
            }
            for r in joint_rows[:12]
        ],
        "football_situation": fb.as_dict(),
        "fell_back_play_only": False,
        "n_plays_considered": len(play_pool),
        "n_joint_candidates": len(joint_rows),
        "no_adjustment_can_win": True,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "scores_are": "joint_play_plus_research_action_not_causal",
    }
