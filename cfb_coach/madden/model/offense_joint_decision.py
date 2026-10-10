"""Joint offensive decision: (formation, play, adjustment plan).

Every legal unmodified play, every legal single adjustment, every compatible
multi-adjustment, and every applicable verified-and-armed macro is scored.
The model-primary selection score already includes anti-repeat. This layer
does not replace that penalty with a fixed rotation.

A research prior can move a plan by at most ``MAX_RESEARCH_INFLUENCE``.
Football-knowledge and drive-strategy priors are smaller still, and they
are applied here rather than inside the model-primary selection score.
Expert/personal learning enters only through the capped ``expert_signal``
path (shadow by default). The older ``vod_model`` late call swap is not
invoked here, so VOD beaters cannot double-count or override joint scoring.
Neither cap can overturn a play whose learned selection score is clearly
higher. Near-ties inside ``INDIFFERENCE_BAND`` are explored with a stable
hash seed. A promoted action model may add its own bounded shift only inside
a matching pre-snap context.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.football_situation import (
    audit_situation_inputs,
    evaluate_situation,
    explain_play,
)
from cfb_coach.madden.model.offense_action_policy import (
    _risk_threshold,
    _situation_trigger,
    _verified_button_sequence,
)

JOINT_POLICY = "joint_offense_action.v3"
BASELINE_JOINT_POLICY = "joint_offense_action.v2"
# A research prior may move the joint score by at most this many points.
# Play probability remains the baseline.
MAX_RESEARCH_INFLUENCE = 0.06
# Complete scores inside this band are treated as tied. A wider gap is a
# real football difference, not a coin flip. This is not a play rotation.
INDIFFERENCE_BAND = 0.012


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


def _explore_near_ties(
    options: Sequence[Mapping[str, Any]],
    *,
    session_id: str | None,
    snap_seq: int | None,
    sit: Any,
    opponent_type: str,
    concentration: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Pick the best complete score. Sample only inside a near-tie band.

    The seed is a hash of the snap, not an index into a rotating menu.
    Anti-repeat stays in ``selection_score`` and is not applied again here.
    Concentrated recent calls can widen that band. A play whose learned
    probability is clearly higher is not pulled into the sample.
    """
    from cfb_coach.madden.model.experimental_model import _play_concept
    from cfb_coach.madden.model.football_knowledge import profile_for_play
    from cfb_coach.madden.model.offense_diversity import (
        CLEAR_SUPERIORITY,
        extra_band_width,
        sampling_weight,
    )

    ordered = sorted(
        options,
        key=lambda c: (
            -float(c["joint_score"]),
            str(c["plan"].get("plan_id")),
            str(c["play"]),
            str(c["formation"]),
        ),
    )
    leader = ordered[0]
    second = float(ordered[1]["joint_score"]) if len(ordered) > 1 else None
    margin = None if second is None else float(leader["joint_score"]) - second
    extra = extra_band_width(concentration, leader.get("row") or {})
    band = INDIFFERENCE_BAND + extra
    meta = {
        "method": "complete_action_value",
        "indifference_band": INDIFFERENCE_BAND,
        "extra_band": extra,
        "sampling_band": round(band, 5),
        "clear_superiority": CLEAR_SUPERIORITY,
        "research_cap": MAX_RESEARCH_INFLUENCE,
        "anti_repeat": "applied_inside_selection_score",
        "fixed_rotation": False,
        "minimum_quota": False,
        "sampled": False,
        "margin": None if margin is None else round(margin, 5),
        "reason": "highest complete football value",
    }
    if second is None or (margin is not None and margin > band):
        if extra > 0 and margin is not None and margin > band:
            meta["reason"] = (
                "highest complete football value; concentration did not "
                "override a clearer play"
            )
        return leader, meta
    near = [
        c for c in ordered
        if float(leader["joint_score"]) - float(c["joint_score"]) <= band + 1e-12
    ]
    leader_probability = float(
        (leader.get("row") or {}).get("probability")
        or (leader.get("row") or {}).get("selection_score")
        or 0.0
    )
    kept: list[Mapping[str, Any]] = []
    for item in near:
        row = item.get("row") or {}
        probability = float(row.get("probability") or row.get("selection_score") or 0.0)
        if item is not leader and leader_probability - probability > CLEAR_SUPERIORITY:
            continue
        kept.append(item)
    near = kept or [leader]
    if len(near) == 1:
        meta["reason"] = (
            "highest complete football value; close alternatives were "
            "clearly weaker on learned probability"
        )
        return near[0], meta
    counts: dict[str, int] = {}
    for item in near:
        concept = str(item["row"].get("play_concept") or _play_concept(str(item["play"])))
        counts[concept] = counts.get(concept, 0) + 1
    ceiling = float(leader["joint_score"])
    temperature = 0.04
    weights: list[float] = []
    for item in near:
        concept = str(item["row"].get("play_concept") or _play_concept(str(item["play"])))
        concept_id = profile_for_play(str(item["play"])).get("concept_id") or concept
        exponent = max(-16.0, (float(item["joint_score"]) - ceiling) / temperature)
        base_weight = math.exp(exponent) / math.sqrt(counts[concept])
        weights.append(sampling_weight(
            base_weight=base_weight,
            formation=str(item["formation"]),
            concept_id=str(concept_id) if concept_id else None,
            stats=concentration,
            apply_concentration=extra > 0,
        ))
    seed = (
        f"joint:{session_id or 'unscoped'}:"
        f"{snap_seq if snap_seq is not None else 'na'}:"
        f"{getattr(sit, 'down', None)}:{getattr(sit, 'distance', None)}:{opponent_type}"
    )
    raw = hashlib.sha256(seed.encode("utf-8")).digest()
    chosen = random.Random(int.from_bytes(raw[:8], "big")).choices(list(near), weights=weights, k=1)[0]
    meta.update({
        "sampled": True,
        "reason": (
            "concentrated recent calls; sampled inside an evidence-aware band, "
            "not a rotation or a quota"
            if extra > 0 else
            "near-tie exploration inside the indifference band; not a fixed rotation"
        ),
        "band_size": len(near),
        "seed": seed,
    })
    return chosen, meta


def _football_intelligence(
    *,
    chosen: Mapping[str, Any] | None,
    representatives: Sequence[Mapping[str, Any]],
    diagnosis: Mapping[str, Any] | None,
    strategy_state: Mapping[str, Any] | None,
    use_knowledge: bool,
    use_strategy: bool,
    use_opponent_learning: bool = False,
    learned_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Why the winning complete action beat the closest alternatives."""
    from cfb_coach.madden.model.concept_matchup import KNOWLEDGE_CAP
    from cfb_coach.madden.model.offense_strategy import STRATEGY_CAP

    def _why(row: Mapping[str, Any]) -> str:
        knowledge = row.get("knowledge") or {}
        strategy = row.get("strategy") or {}
        knowledge_text = "; ".join(knowledge.get("reasons") or []) or "no knowledge prior"
        strategy_text = "; ".join(strategy.get("reasons") or []) or "no strategy prior"
        learning = row.get("learning") or {}
        learning_text = "; ".join(learning.get("reasons") or []) or "no opponent-learning prior"
        expert = row.get("expert_learning") or {}
        expert_text = "; ".join(expert.get("reasons") or []) or "no expert-learning prior"
        withheld = "; ".join(knowledge.get("withheld") or [])
        text = (
            f"selection {float(row.get('selection_score') or 0):.3f}, "
            f"knowledge {float(knowledge.get('delta') or 0):+.3f} ({knowledge_text}), "
            f"strategy {float(strategy.get('delta') or 0):+.3f} ({strategy_text}), "
            f"learning {float(learning.get('delta') or 0):+.3f} ({learning_text}), "
            f"expert {float(expert.get('delta') or 0):+.3f} "
            f"(shadow {float(expert.get('shadow_delta') or 0):+.3f}; {expert_text})"
        )
        if withheld:
            text += f"; withheld: {withheld}"
        return text

    winner = None
    if chosen is not None:
        winner = {
            "formation": chosen.get("formation"),
            "play": chosen.get("play"),
            "joint_score": chosen.get("joint_score"),
            "concept_id": (chosen.get("knowledge") or {}).get("concept_id"),
            "knowledge_delta": (chosen.get("knowledge") or {}).get("delta"),
            "strategy_delta": (chosen.get("strategy") or {}).get("delta"),
            "learning_delta": (chosen.get("learning") or {}).get("delta"),
            "expert_delta": (chosen.get("expert_learning") or {}).get("delta"),
            "expert_shadow_delta": (chosen.get("expert_learning") or {}).get("shadow_delta"),
            "why": _why(chosen),
        }
    others = [
        row for row in representatives
        if chosen is None or (row.get("formation"), row.get("play")) != (
            chosen.get("formation"), chosen.get("play"),
        )
    ]
    others = sorted(others, key=lambda row: -float(row.get("joint_score") or 0))
    alternatives = []
    leader = float(chosen.get("joint_score") or 0) if chosen is not None else 0.0
    for row in others[:4]:
        alternatives.append({
            "formation": row.get("formation"),
            "play": row.get("play"),
            "joint_score": row.get("joint_score"),
            "knowledge_delta": (row.get("knowledge") or {}).get("delta"),
            "strategy_delta": (row.get("strategy") or {}).get("delta"),
            "learning_delta": (row.get("learning") or {}).get("delta"),
            "expert_delta": (row.get("expert_learning") or {}).get("delta"),
            "expert_shadow_delta": (row.get("expert_learning") or {}).get("shadow_delta"),
            "why_not": (
                f"complete score {float(row.get('joint_score') or 0):.3f} trails "
                f"{leader:.3f}. {_why(row)}"
            ),
        })
    summary = "No legal play was available."
    if winner is not None:
        summary = (
            f"{winner['formation']} — {winner['play']}: {winner['why']}. "
            "Learned selection remains the base score."
        )
        if alternatives:
            summary += " " + alternatives[0]["why_not"]
    expert_modes = {
        str((row.get("expert_learning") or {}).get("mode") or "off")
        for row in ([chosen] if chosen is not None else [])
    }
    return {
        "use_knowledge": use_knowledge,
        "use_strategy": use_strategy,
        "use_opponent_learning": use_opponent_learning,
        "knowledge_cap": KNOWLEDGE_CAP,
        "strategy_cap": STRATEGY_CAP,
        "opponent_learning": {
            "enabled": use_opponent_learning,
            "usable_verified_snaps": (learned_model or {}).get("usable_verified_snaps", 0),
            "hypothesis": (learned_model or {}).get("strategy_hypothesis"),
            "changes": (learned_model or {}).get("changes") or [],
        },
        "expert_learning": {
            "enabled": any(
                (row.get("expert_learning") or {}).get("mode") not in (None, "off")
                for row in (representatives or [])
            ),
            "modes": sorted(expert_modes),
            "cap": (
                (chosen or {}).get("expert_learning") or {}
            ).get("cap"),
            "live_influence": bool(
                ((chosen or {}).get("expert_learning") or {}).get("live_influence")
            ),
            "vod_model_not_on_joint_path": True,
        },
        "learned_evidence_remains_primary": True,
        "winner": winner,
        "alternatives": alternatives,
        "diagnosis": diagnosis,
        "strategy": strategy_state,
        "summary": summary,
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
    session_id: str | None = None,
    snap_seq: int | None = None,
    opponent_type: str = "cpu",
    use_knowledge: bool = True,
    use_strategy: bool = True,
    strategy_previous: Mapping[str, Any] | None = None,
    use_diversity: bool = True,
    recent_calls: Sequence[Any] | None = None,
    use_opponent_learning: bool = True,
    use_expert_learning: bool = True,
) -> dict[str, Any]:
    """Pick one legal (formation, play, plan) from the complete action space.

    Each eligible play contributes its best legal plan: unchanged, one
    adjustment, a compatible package, or an armed macro. A different
    unmodified play wins when its complete score is better. The sampled
    anchor is the baseline recorded on the decision, not a lock.
    """
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
    diagnosis = None
    strategy_state = None
    learned_model = None
    if use_knowledge or use_strategy or use_opponent_learning:
        from cfb_coach.madden.model.defensive_diagnosis import diagnose_defense

        diagnosis = diagnose_defense(sit, memory)
    if use_opponent_learning:
        from cfb_coach.madden.model.opponent_learning import fit_opponent_model

        learned_model = fit_opponent_model(memory)
    if use_strategy:
        from cfb_coach.madden.model.offense_strategy import current_strategy

        strategy_state = current_strategy(
            sit, memory, diagnosis, previous=strategy_previous,
            learned=learned_model if use_opponent_learning else None,
        )
    current_pressure = bool(
        diagnosis
        and diagnosis.get("state") == "observed"
        and (diagnosis.get("observed") or {}).get("pressure")
    )
    anchor_form = str(anchor.get("formation"))
    anchor_play = str(anchor.get("play"))
    anchor_score = float(anchor.get("selection_score", anchor.get("probability", 0.0)) or 0.0)
    summaries: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    plays_seen = 0
    for row in ranked:
        form, play = str(row.get("formation")), str(row.get("play"))
        if play not in (book or {}).get(form, []):
            continue
        plays_seen += 1
        fit = explain_play(play, sit, memory=memory)
        base = float(row.get("selection_score", row.get("probability", 0.0)) or 0.0)
        base += float(fit["coordinator_delta"])
        knowledge = {"delta": 0.0, "reasons": ["knowledge off"], "withheld": []}
        strategy_adj = {"delta": 0.0, "reasons": ["strategy off"], "withheld": []}
        learning_adj = {"delta": 0.0, "reasons": ["opponent learning off"], "withheld": []}
        expert_adj = {
            "delta": 0.0,
            "shadow_delta": 0.0,
            "reasons": ["expert learning off"],
            "withheld": [],
            "mode": "off",
            "live_influence": False,
        }
        if use_knowledge:
            from cfb_coach.madden.model.concept_matchup import evaluate_concept_matchup

            knowledge = evaluate_concept_matchup(play, sit, diagnosis)
        if use_strategy:
            from cfb_coach.madden.model.offense_strategy import strategy_adjustment

            strategy_adj = strategy_adjustment(
                play, strategy_state, knowledge_delta=float(knowledge.get("delta") or 0.0),
            )
        if use_opponent_learning:
            from cfb_coach.madden.model.opponent_learning import learning_adjustment

            learning_adj = learning_adjustment(
                play, sit, learned_model,
                knowledge_delta=float(knowledge.get("delta") or 0.0),
                strategy_delta=float(strategy_adj.get("delta") or 0.0),
                current_pressure_observed=current_pressure,
            )
        if use_expert_learning:
            from cfb_coach.madden.model.expert_signal import expert_learning_adjustment

            expert_adj = expert_learning_adjustment(
                play, sit, db=db, opponent_type=opponent_type,
                book=book, formation=form,
            )
        base += float(knowledge.get("delta") or 0.0)
        base += float(strategy_adj.get("delta") or 0.0)
        base += float(learning_adj.get("delta") or 0.0)
        base += float(expert_adj.get("delta") or 0.0)
        plans = legal_plans_for_play(
            formation=form, play=play, sit=sit, book=book,
            active=list(active or []), prediction=row, repeated=repeated,
            audibles=audibles, db=db, opponent_id=opponent_id,
            evaluation=evaluation, weights=weights, cooled=cooled,
            score_phase=score_phase, allow_macros=allow_macros, learned=learned,
        )
        unchanged = next(plan for plan in plans if plan["plan_id"] == "NO_ADJUSTMENT")
        unchanged_joint = base + float(unchanged.get("influence") or 0.0)
        eligible: list[dict[str, Any]] = []
        for plan in plans:
            joint = base + float(plan.get("influence") or 0.0)
            plan["joint_score"] = round(joint, 5)
            summaries.append({
                "formation": form, "play": play, "kind": plan["kind"],
                "id": plan["plan_id"], "score": plan["joint_score"],
            })
            if plan.get("executable") is False:
                continue
            # A sourced plan must beat running this same play unchanged.
            # Every unmodified play stays in the race either way.
            if plan["plan_id"] != "NO_ADJUSTMENT" and joint <= unchanged_joint + 1e-9:
                continue
            eligible.append(plan)
        if not eligible:
            continue
        best = max(
            eligible,
            key=lambda plan: (
                float(plan["joint_score"]),
                0 if plan["plan_id"] == "NO_ADJUSTMENT" else 1,
                str(plan["plan_id"]),
            ),
        )
        representatives.append({
            "joint_score": float(best["joint_score"]),
            "formation": form,
            "play": play,
            "plan": best,
            "row": row,
            "knowledge": knowledge,
            "strategy": strategy_adj,
            "learning": learning_adj,
            "expert_learning": expert_adj,
            "selection_score": float(row.get("selection_score", row.get("probability", 0.0)) or 0.0),
        })
    from cfb_coach.madden.model.offense_diversity import concentration_stats

    recent = list(recent_calls) if recent_calls is not None else list(
        (memory or {}).get("recent_recommendations") or []
    )
    concentration = concentration_stats(recent) if use_diversity else None
    exploration: dict[str, Any]
    if representatives:
        chosen, exploration = _explore_near_ties(
            representatives, session_id=session_id, snap_seq=snap_seq,
            sit=sit, opponent_type=opponent_type, concentration=concentration,
        )
        best_plan = dict(chosen["plan"])
        best_key = (str(chosen["formation"]), str(chosen["play"]))
    else:
        best_plan = {
            "plan_id": "NO_ADJUSTMENT", "kind": "none", "parts": [],
            "reason": "no legal play was available",
            "scores_are": "baseline_no_adjustment",
            "joint_score": round(anchor_score, 5),
            "influence": 0.0,
        }
        best_key = (anchor_form, anchor_play)
        exploration = {
            "method": "complete_action_value",
            "sampled": False,
            "reason": "no legal play was available",
            "anti_repeat": "applied_inside_selection_score",
            "fixed_rotation": False,
            "minimum_quota": False,
            "extra_band": 0.0,
            "research_cap": MAX_RESEARCH_INFLUENCE,
        }
    summaries.sort(key=lambda r: (-float(r["score"]), str(r["id"]), r["play"]))
    shown = summaries[:12]
    if summaries and not any(row["id"] == "NO_ADJUSTMENT" for row in shown):
        unchanged_row = next(row for row in summaries if row["id"] == "NO_ADJUSTMENT")
        shown.append(unchanged_row)
    decision = _decision_from_plan(
        best_plan, formation=best_key[0], play=best_key[1], anchor=anchor,
        candidates=shown, evaluation=evaluation, learned_mode=learned_mode,
    )
    decision["plays_considered"] = plays_seen
    decision["eligible_play_count"] = plays_seen
    decision["plans_compared"] = len(summaries)
    decision["plays_in_final_comparison"] = len(representatives)
    decision["exploration"] = exploration
    decision["diversity"] = {
        "eligible_installed_plays": plays_seen,
        "plays_considered": plays_seen,
        "consideration_rate": 1.0 if plays_seen else 0.0,
        "fixed_rotation": False,
        "minimum_quota": False,
        "extra_band": exploration.get("extra_band", 0.0),
        "clear_superiority": exploration.get("clear_superiority"),
        "recent_calls": (concentration or {}).get("calls", 0),
        "top_formation_share": (concentration or {}).get("top_formation_share"),
        "top_concept_share": (concentration or {}).get("top_concept_share"),
        "top_play_share": (concentration or {}).get("top_play_share"),
        "note": (
            "Every eligible installed play was scored. Variety is sampled only "
            "inside a close band, and a clearly stronger learned play is kept."
        ),
    }
    decision["input_audit"] = audit_situation_inputs(
        sit, evaluation=evaluation, memory=memory,
    )
    if not use_knowledge and not use_strategy:
        decision["policy_version"] = BASELINE_JOINT_POLICY
    intelligence = _football_intelligence(
        chosen=next(
            (row for row in representatives if row["formation"] == best_key[0] and row["play"] == best_key[1]),
            None,
        ),
        representatives=representatives,
        diagnosis=diagnosis,
        strategy_state=strategy_state,
        use_knowledge=use_knowledge,
        use_strategy=use_strategy,
        use_opponent_learning=use_opponent_learning,
        learned_model=learned_model,
    )
    decision["football_intelligence"] = intelligence
    chosen_rep = next(
        (row for row in representatives if row["formation"] == best_key[0] and row["play"] == best_key[1]),
        None,
    )
    decision["expert_learning"] = {
        "mode": ((chosen_rep or {}).get("expert_learning") or {}).get("mode", "off"),
        "live_influence": bool(((chosen_rep or {}).get("expert_learning") or {}).get("live_influence")),
        "applied_delta": ((chosen_rep or {}).get("expert_learning") or {}).get("delta", 0.0),
        "shadow_delta": ((chosen_rep or {}).get("expert_learning") or {}).get("shadow_delta", 0.0),
        "cap": ((chosen_rep or {}).get("expert_learning") or {}).get("cap"),
        "vod_prior_on_joint_path": False,
        "model_primary_retained": True,
    }
    joint = decision.get("joint") or {}
    football = dict(joint.get("football") or {})
    football["intelligence"] = {
        "summary": intelligence.get("summary"),
        "diagnosis_state": (intelligence.get("diagnosis") or {}).get("state"),
        "objective": (intelligence.get("strategy") or {}).get("objective"),
    }
    joint["football"] = football
    decision["joint"] = joint
    return decision
