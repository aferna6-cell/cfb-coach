"""Joint offensive decision: (formation, play, adjustment plan).

The unmodified play still comes from the model-primary policy, including its
anti-repeat sampling. Legal adjustments on every eligible play then compete
with that unmodified call. NO ADJUSTMENT is always in the race.

Research priors are not learned causal effects. A promoted action model may
nudge a score by at most its published bound, and only inside a matching
pre-snap context.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.football_situation import evaluate_situation, explain_play
from cfb_coach.madden.model.offense_action_policy import (
    _risk_threshold,
    _situation_trigger,
    _verified_button_sequence,
)

JOINT_POLICY = "joint_offense_action.v1"
# A research prior may move the joint score by at most this many points.
# Play probability remains the baseline.
MAX_RESEARCH_INFLUENCE = 0.06


def _clock_blocks(evaluation: Mapping[str, Any], n_actions: int) -> str | None:
    clock = evaluation.get("clock_seconds")
    if clock is not None and int(clock) < 10 and n_actions:
        return "play clock is too short for a pre-snap adjustment"
    if evaluation.get("two_minute") and n_actions > 2:
        return "two-minute offense cannot complete more than two sourced adjustments"
    return None


def _research_row(action_id: str) -> dict[str, Any]:
    from cfb_coach.madden import research_db

    return next(
        (dict(row) for row in research_db.offense_adjustments() if row.get("id") == action_id),
        {},
    )


def _part_view(candidate: Mapping[str, Any]) -> dict[str, Any]:
    meta = _research_row(str(candidate.get("id") or ""))
    return {
        "id": candidate.get("id"),
        "kind": candidate.get("kind") or meta.get("type"),
        "target": meta.get("target") or candidate.get("target"),
        "route": meta.get("route") or candidate.get("route"),
        "label": candidate.get("label"),
        "buttons": candidate.get("buttons"),
        "why": candidate.get("why"),
        "sources": list(candidate.get("sources") or meta.get("sources") or []),
        "vs": list(meta.get("vs") or candidate.get("vs") or []),
        "payload": dict(candidate),
    }


def _conflicts(parts: Sequence[Mapping[str, Any]]) -> str | None:
    from cfb_coach.madden.model.offense_macro_lab import conflicts

    rows = [
        {
            "type": p.get("kind"), "target": p.get("target"), "route": p.get("route"),
            "label": p.get("label"), "vs": p.get("vs") or [],
        }
        for p in parts
    ]
    return conflicts(rows)


def _single_score(
    *,
    kind: str,
    confidence: float,
    probability: float,
    coverage_class: str | None,
    goal_line: bool,
) -> float:
    """Same research prior scale as the single-action policy."""
    score = 0.20 + 0.14 * confidence + 0.03 * (probability - 0.5)
    if kind == "pass_protection" and coverage_class == "pressure":
        score += 0.105
    if kind == "hot_route" and coverage_class in ("man", "cover2", "two_high"):
        score += 0.05
    if kind == "macro":
        score = 0.24 + 0.12 * confidence + 0.05 * (probability - 0.5)
        score += 0.07 if goal_line else (0.055 if coverage_class == "pressure" else 0.04)
    return score


def legal_plans_for_play(
    *,
    formation: str,
    play: str,
    sit: Any,
    book: dict[str, list[str]],
    active: Sequence[str],
    prediction: Mapping[str, Any] | None,
    repeated: bool,
    audibles: dict[str, list[str]] | None,
    db: Any,
    opponent_id: str,
    evaluation: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    cooled: set[str] | None = None,
    score_phase: str | None = None,
    allow_macros: bool = True,
    learned: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Every legal plan for one play, including NO ADJUSTMENT.

    Audibles stay out until a separate post-audible execution record exists.
    Draft and unarmed macros are not plans.
    """
    from cfb_coach.madden.adjustments import offense_adjustment_candidates
    from cfb_coach.madden.model.offense_action_learning import score_shift
    from cfb_coach.madden.offense_macros import (
        clean_ids, offense_detail, pairs_in_book, suggest_for_snap,
    )
    from cfb_coach.madden.playcaller import coverage_class

    look = evaluation.get("coverage") or {}
    credible = look.get("state") == "observed" and bool(look.get("value"))
    cls = look.get("value") if credible else None
    source = "live" if credible else "none"
    pred = prediction or {}
    uncertainty = max(0.0, min(1.0, float(pred.get("uncertainty", 1.0) or 1.0)))
    confidence = 1.0 - uncertainty
    probability = max(0.0, min(1.0, float(pred.get("probability", 0.5) or 0.5)))
    threshold = _risk_threshold(sit, credible_look=credible)
    zone = "gl" if getattr(sit, "goal_line", False) else "rz" if getattr(sit, "red_zone", False) else "open"
    plans: list[dict[str, Any]] = [{
        "plan_id": "NO_ADJUSTMENT",
        "kind": "none",
        "parts": [],
        "research_score": threshold,
        "learned_shift": 0.0,
        "execution_cost": 0.0,
        "influence": 0.0,
        "reason": "run the play unchanged",
        "executable": True,
        "scores_are": "baseline_no_adjustment",
    }]
    if play not in (book or {}).get(formation, []):
        plans[0]["reason"] = "play is not in the confirmed book"
        plans[0]["executable"] = False
        return plans

    singles: list[dict[str, Any]] = []
    if credible:
        for candidate in offense_adjustment_candidates(
            play=play, formation=formation, coverage_class=cls,
            coverage_source=source, repeated=repeated, audibles=audibles,
        ):
            if candidate.get("kind") == "audible":
                continue
            if not candidate.get("id") or not candidate.get("sources"):
                continue
            if not _verified_button_sequence(candidate.get("buttons")):
                continue
            view = _part_view(candidate)
            score = _single_score(
                kind=str(view["kind"]), confidence=confidence, probability=probability,
                coverage_class=cls, goal_line=bool(evaluation.get("goal_line")),
            )
            if score <= threshold:
                continue
            blocked = _clock_blocks(evaluation, 1)
            if blocked:
                continue
            shift, _evidence = score_shift(
                learned, play=play, down=getattr(sit, "down", None),
                distance=getattr(sit, "distance", None), kind="adjustment",
                action_id=str(view["id"]),
                coverage_class=cls if source == "live" else None,
                coverage_source=source,
                red_zone=bool(getattr(sit, "red_zone", False)),
                goal_line=bool(getattr(sit, "goal_line", False)),
            )
            singles.append({
                "plan_id": str(view["id"]),
                "kind": "adjustment",
                "parts": [view],
                "research_score": score,
                "learned_shift": shift,
                "execution_cost": 0.012,
                "reason": view.get("why") or "",
                "executable": True,
                "scores_are": "research_policy_not_learned_action_effect",
            })

    # Pairs of compatible singles. A conflict is dropped, not repaired.
    # Snapshot first so newly appended packages are not paired again.
    base_singles = list(singles)
    for i, left in enumerate(base_singles):
        for right in base_singles[i + 1:]:
            parts = [left["parts"][0], right["parts"][0]]
            why = _conflicts(parts)
            if why:
                continue
            blocked = _clock_blocks(evaluation, 2)
            if blocked:
                continue
            package = max(left["research_score"], right["research_score"])
            if package <= threshold:
                continue
            singles_ids = "+".join(sorted(str(p["id"]) for p in parts))
            shift, _evidence = score_shift(
                learned, play=play, down=getattr(sit, "down", None),
                distance=getattr(sit, "distance", None), kind="adjustment",
                action_id=singles_ids,
                coverage_class=cls if source == "live" else None,
                coverage_source=source,
                red_zone=bool(getattr(sit, "red_zone", False)),
                goal_line=bool(getattr(sit, "goal_line", False)),
            )
            singles.append({
                "plan_id": singles_ids,
                "kind": "adjustment",
                "parts": parts,
                "research_score": package,
                "learned_shift": shift,
                "execution_cost": 0.028,
                "reason": "compatible sourced adjustments; combination is not a saved macro slot",
                "executable": True,
                "scores_are": "research_policy_not_learned_action_effect",
                "composition": True,
            })

    if allow_macros:
        full_cap = 1 + sum(len(ps) for ps in book.values())
        for mid in clean_ids(list(active)):
            try:
                suggestion = suggest_for_snap(
                    zone=zone, play=play, coverage=getattr(sit, "coverage_hint", None),
                    coverage_source=getattr(sit, "coverage_source", None) or "none",
                    active=[mid], down=getattr(sit, "down", None), repeated=repeated,
                    book=book, weights=dict(weights or {}), score_phase=score_phase,
                    cooled=cooled,
                )
            except (TypeError, ValueError, KeyError):
                continue
            if not suggestion or suggestion.get("id") != mid:
                continue
            pair = f"{play} ({formation})"
            if pair not in pairs_in_book(mid, book, cap=full_cap):
                continue
            detail = offense_detail(mid, book)
            if detail.get("needs_settings") or detail.get("gaps") or not detail.get("settings"):
                continue
            if not _verified_button_sequence(suggestion.get("buttons")):
                continue
            if suggestion.get("kind") == "look" and not credible:
                continue
            if suggestion.get("kind") == "situation" and not _situation_trigger(sit, "situation"):
                continue
            if _clock_blocks(evaluation, 1):
                continue
            score = _single_score(
                kind="macro", confidence=confidence, probability=probability,
                coverage_class=coverage_class(getattr(sit, "coverage_hint", None)) if credible else None,
                goal_line=bool(evaluation.get("goal_line")),
            )
            if score <= threshold:
                continue
            shift, _evidence = score_shift(
                learned, play=play, down=getattr(sit, "down", None),
                distance=getattr(sit, "distance", None), kind="macro", action_id=str(mid),
                coverage_class=cls if source == "live" else None, coverage_source=source,
                red_zone=bool(getattr(sit, "red_zone", False)),
                goal_line=bool(getattr(sit, "goal_line", False)),
            )
            singles.append({
                "plan_id": str(mid),
                "kind": "macro",
                "parts": [{
                    "id": mid, "kind": "macro", "buttons": suggestion.get("buttons"),
                    "why": suggestion.get("why"), "label": suggestion.get("name") or mid,
                    "payload": suggestion,
                }],
                "research_score": score,
                "learned_shift": shift,
                "execution_cost": 0.012,
                "reason": suggestion.get("why") or "",
                "executable": True,
                "scores_are": "research_policy_not_learned_action_effect",
            })
        if db is not None and opponent_id and credible:
            from cfb_coach.madden.model.offense_designer import verified_created_macros
            from cfb_coach.madden import research_db
            for created in verified_created_macros(db, opponent_id):
                if created.get("coverage") != cls:
                    continue
                if not any(
                    item.get("formation") == formation and item.get("play") == play
                    for item in created.get("base_pairs") or []
                ):
                    continue
                if not created.get("settings") or not created.get("source_ids"):
                    continue
                if created.get("state") == "draft" or created.get("verified_armed") is not True:
                    continue
                name = created["name"]
                buttons = research_db.buttons("offense", "custom_adjustments").replace(
                    "pick the adjustment", name
                )
                if not _verified_button_sequence(buttons):
                    continue
                score = _single_score(
                    kind="macro", confidence=confidence, probability=probability,
                    coverage_class=cls, goal_line=bool(evaluation.get("goal_line")),
                )
                if score <= threshold:
                    continue
                singles.append({
                    "plan_id": name,
                    "kind": "macro",
                    "parts": [{
                        "id": name, "kind": "macro", "buttons": buttons,
                        "why": created.get("fire_when") or "", "label": name,
                        "payload": {
                            "id": name, "name": name, "side": "offense", "kind": "look",
                            "buttons": buttons, "why": created.get("fire_when") or "",
                            "settings": created["settings"], "source_ids": created["source_ids"],
                            "verified_armed": True,
                        },
                    }],
                    "research_score": score,
                    "learned_shift": 0.0,
                    "execution_cost": 0.012,
                    "reason": created.get("fire_when") or "",
                    "executable": True,
                    "scores_are": "research_policy_not_learned_action_effect",
                })

    for plan in singles:
        margin = float(plan["research_score"]) - threshold
        influence = max(-MAX_RESEARCH_INFLUENCE, min(MAX_RESEARCH_INFLUENCE, margin * 0.25))
        influence += float(plan.get("learned_shift") or 0.0)
        influence -= float(plan.get("execution_cost") or 0.0)
        # Multi-action packages carry extra uncertainty until verified together.
        if plan.get("composition"):
            influence -= 0.01
            plan["scores_are"] = "research_policy_not_learned_action_effect"
        plan["influence"] = round(influence, 5)
        plans.append(plan)
    return plans


def _decision_from_plan(
    plan: Mapping[str, Any],
    *,
    formation: str,
    play: str,
    anchor: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    evaluation: Mapping[str, Any],
    learned_mode: str,
) -> dict[str, Any]:
    kind = str(plan.get("kind") or "none")
    parts = list(plan.get("parts") or [])
    payload = parts[0]["payload"] if len(parts) == 1 else None
    if len(parts) > 1:
        buttons = " ; then ".join(str(p.get("buttons") or "") for p in parts)
        payload = {
            "id": plan.get("plan_id"),
            "kind": "multi_adjustment",
            "label": " + ".join(str(p.get("label") or p.get("id")) for p in parts),
            "buttons": buttons,
            "why": plan.get("reason"),
            "sources": [s for p in parts for s in (p.get("sources") or [])],
            "parts": [p.get("id") for p in parts],
            "composition": True,
            "not_a_saved_macro_slot": True,
        }
        kind = "adjustment"
    macro = payload if kind == "macro" else None
    adjustment = payload if kind == "adjustment" else None
    return {
        "policy_version": JOINT_POLICY,
        "kind": "none" if kind == "none" else kind,
        "id": None if kind == "none" else plan.get("plan_id"),
        "macro": macro,
        "adjustment": adjustment,
        "reason": plan.get("reason") or "",
        "scores_are": plan.get("scores_are") or "research_policy_not_learned_action_effect",
        "candidates": candidates,
        "no_action_compared": True,
        "no_action_score": next(
            (c["score"] for c in candidates if c["id"] == "NO_ADJUSTMENT"), None
        ),
        "top_action_score": plan.get("joint_score"),
        "observational_evidence_mode": learned_mode,
        "observation_not_causal": True,
        "why_now": (
            "observed pre-snap look"
            if (evaluation.get("coverage") or {}).get("state") == "observed"
            else "verified situational condition or unchanged play"
        ),
        "joint": {
            "formation": formation,
            "play": play,
            "plan_id": plan.get("plan_id"),
            "baseline_formation": anchor.get("formation"),
            "baseline_play": anchor.get("play"),
            "displaced_baseline": (
                formation != anchor.get("formation") or play != anchor.get("play")
                or plan.get("plan_id") != "NO_ADJUSTMENT"
            ),
            "coverage_state": (evaluation.get("coverage") or {}).get("state"),
            "football": {
                "situations": evaluation.get("situations"),
                "score_phase": evaluation.get("score_phase"),
            },
        },
    }


def choose_joint_action(
    *,
    ranked: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
    sit: Any,
    book: dict[str, list[str]],
    active: Sequence[str] | None = None,
    db: Any = None,
    opponent_id: str = "",
    audibles: dict[str, list[str]] | None = None,
    memory: Mapping[str, Any] | None = None,
    repeated: bool = False,
    weights: Mapping[str, float] | None = None,
    cooled: set[str] | None = None,
    score_phase: str | None = None,
    allow_macros: bool = True,
) -> dict[str, Any]:
    """Pick one legal (formation, play, plan). Unmodified sampling is the anchor."""
    from cfb_coach.madden.model.offense_action_learning import load_action_evidence

    evaluation = evaluate_situation(sit, memory=memory)
    if weights is None or cooled is None or score_phase is None:
        try:
            from cfb_coach.game_score import classify
            from cfb_coach.madden.playcaller import _cooled_macros, _learned_macros
            from cfb_coach.tendency import is_repeated_coverage

            weights = _learned_macros(db, opponent_id) if weights is None else weights
            cooled = _cooled_macros(db, sit) if cooled is None else cooled
            ctx = classify(sit)
            if score_phase is None:
                score_phase = ctx.phase if ctx else None
            cov = getattr(sit, "coverage_hint", None)
            src = getattr(sit, "coverage_source", None) or "none"
            if not repeated and cov and src == "live" and db is not None:
                repeated = is_repeated_coverage(db, opponent_id, cov, sit, threshold=2)
        except Exception:  # noqa: BLE001 — missing helpers must not block the call
            weights = weights or {}
            cooled = cooled or set()
    try:
        learned = load_action_evidence(db) if db is not None else None
    except Exception:  # noqa: BLE001 — a missing meta table must not block the call
        learned = None
    learned_mode = (learned or {}).get("mode", "none")
    anchor_form = str(anchor.get("formation"))
    anchor_play = str(anchor.get("play"))
    anchor_score = float(anchor.get("selection_score", anchor.get("probability", 0.0)) or 0.0)
    best_plan: dict[str, Any] | None = None
    best_key: tuple[str, str] = (anchor_form, anchor_play)
    best_joint = anchor_score
    summaries: list[dict[str, Any]] = []
    plays_seen = 0
    for row in ranked:
        form, play = str(row.get("formation")), str(row.get("play"))
        if play not in (book or {}).get(form, []):
            continue
        plays_seen += 1
        fit = explain_play(play, sit, memory=memory)
        base = float(row.get("selection_score", row.get("probability", 0.0)) or 0.0)
        base += float(fit["coordinator_delta"])
        plans = legal_plans_for_play(
            formation=form, play=play, sit=sit, book=book,
            active=list(active or []), prediction=row, repeated=repeated,
            audibles=audibles, db=db, opponent_id=opponent_id,
            evaluation=evaluation, weights=weights, cooled=cooled,
            score_phase=score_phase, allow_macros=allow_macros, learned=learned,
        )
        for plan in plans:
            # Other plays' unchanged versions do not override anti-repeat sampling.
            if plan["plan_id"] == "NO_ADJUSTMENT" and (form, play) != (anchor_form, anchor_play):
                continue
            joint = base + float(plan.get("influence") or 0.0)
            plan["joint_score"] = round(joint, 5)
            summaries.append({
                "formation": form, "play": play, "kind": plan["kind"],
                "id": plan["plan_id"], "score": plan["joint_score"],
            })
            if joint > best_joint + 1e-9:
                best_joint = joint
                best_plan = plan
                best_key = (form, play)
    if best_plan is None or best_plan["plan_id"] == "NO_ADJUSTMENT":
        best_plan = {
            "plan_id": "NO_ADJUSTMENT", "kind": "none", "parts": [],
            "reason": "no verified adjustment beat the unchanged play",
            "scores_are": "baseline_no_adjustment", "joint_score": round(anchor_score, 5),
            "influence": 0.0,
        }
        best_key = (anchor_form, anchor_play)
    summaries.append({
        "formation": anchor_form, "play": anchor_play, "kind": "none",
        "id": "NO_ADJUSTMENT", "score": round(anchor_score, 5),
    })
    summaries.sort(key=lambda r: (-float(r["score"]), str(r["id"]), r["play"]))
    # Keep the audit bounded, but never hide the unchanged option.
    shown = summaries[:12]
    if not any(row["id"] == "NO_ADJUSTMENT" for row in shown):
        unchanged = next(row for row in summaries if row["id"] == "NO_ADJUSTMENT")
        shown.append(unchanged)
    decision = _decision_from_plan(
        best_plan, formation=best_key[0], play=best_key[1], anchor=anchor,
        candidates=shown, evaluation=evaluation, learned_mode=learned_mode,
    )
    decision["plays_considered"] = plays_seen
    decision["eligible_play_count"] = plays_seen
    return decision
