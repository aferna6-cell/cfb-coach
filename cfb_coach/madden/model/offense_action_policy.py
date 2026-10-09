"""Research-grounded first stage of model-controlled offensive actions.

The *play* is selected by the trained experimental model. This module ranks
legally applicable hot routes/protections and armed Custom Adjustments using
pre-snap evidence and the selected play's model uncertainty. These action
scores are policy priors, NOT learned causal benefits. Actual action learning
requires separately verified executions and outcomes.

Never invent an editor setting or infer the next coverage from a prior snap.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


POLICY_VERSION = "offense_action_policy.situational_verified.v2"


def _risk_threshold(sit: Any, *, credible_look: bool) -> float:
    """Score to beat doing nothing; unknown looks require more evidence."""
    zone = bool(getattr(sit, "red_zone", False) or getattr(sit, "goal_line", False))
    try:
        short = (
            int(getattr(sit, "down", 0) or 0) in (3, 4)
            and 0 < float(getattr(sit, "distance", 0) or 0) <= 2
        )
    except (ValueError, TypeError):
        short = False
    if credible_look:
        return 0.275
    return 0.30 if zone or short else 0.36


def _situation_trigger(sit: Any, kind: str) -> bool:
    """A situation-only macro needs an actual situational need."""
    if kind != "situation":
        return False
    if getattr(sit, "goal_line", False) or getattr(sit, "red_zone", False):
        return True
    try:
        down = int(getattr(sit, "down", 0) or 0)
        distance = int(getattr(sit, "distance", 0) or 0)
        return down in (3, 4) and 0 < distance <= 2
    except (ValueError, TypeError):
        return False


def _verified_button_sequence(value: Any) -> bool:
    """Never display guessed or editor-unverified controller actions."""
    text = str(value or "").strip()
    blocked = ("VERIFY ON SCREEN", "NO SOURCE", "UNKNOWN", "NOT CONFIRMED")
    return bool(text) and not any(term in text.upper() for term in blocked)



def choose_offense_action(
    *,
    formation: str,
    play: str,
    sit: Any,
    book: dict[str, list[str]],
    active: Sequence[str],
    weights: Mapping[str, float] | None = None,
    cooled: set[str] | None = None,
    score_phase: str | None = None,
    audibles: dict[str, list[str]] | None = None,
    prediction: Mapping[str, Any] | None = None,
    allow_macros: bool = True,
    repeated: bool = False,
    db: Any = None,
    opponent_id: str = "",
) -> dict[str, Any]:
    """Choose exactly one eligible researched action, or none.

    Every action must fit the *model-selected* in-book (formation, play).
    Candidate ranking uses uncertainty and evidence quality, not fabricated
    learned route or macro payoff statistics.
    """
    from cfb_coach.madden.adjustments import offense_adjustment_candidates
    from cfb_coach.madden.offense_macros import (
        clean_ids, offense_detail, pairs_in_book, suggest_for_snap,
    )
    from cfb_coach.madden.playcaller import coverage_class

    empty = {
        "policy_version": POLICY_VERSION,
        "kind": "none", "macro": None, "adjustment": None,
        "reason": "no eligible supported adjustment",
        "scores_are": "research_policy_not_learned_action_effect",
        "candidates": [],
        "no_action_compared": True,
    }
    if play not in (book or {}).get(formation, []):
        return {**empty, "reason": "selected play is not in applied book"}
    pred = prediction or {}
    uncertainty = max(0.0, min(1.0, float(pred.get("uncertainty", 1.0) or 1.0)))
    confidence = 1.0 - uncertainty
    probability = max(0.0, min(1.0, float(pred.get("probability", 0.5) or 0.5)))
    cov = getattr(sit, "coverage_hint", None)
    source = getattr(sit, "coverage_source", None) or "none"
    zone = "gl" if getattr(sit, "goal_line", False) else "rz" if getattr(sit, "red_zone", False) else "open"
    cls = coverage_class(cov) if cov else None
    # A last-snap coverage is not a sufficient trigger by itself.
    credible_look = bool(cls) and (source == "live" or repeated)
    rows: list[dict[str, Any]] = []
    no_action_score = _risk_threshold(sit, credible_look=credible_look)
    macros_armed = clean_ids(list(active)) if allow_macros else []

    for mid in macros_armed:
        try:
            # Independent eligibility check for each armed macro; the existing
            # first-match heuristic must not decide which macro wins.
            suggestion = suggest_for_snap(
                zone=zone, play=play, coverage=cov,
                coverage_source=source, active=[mid],
                down=getattr(sit, "down", None), repeated=repeated,
                book=book, weights=dict(weights or {}), score_phase=score_phase,
                cooled=cooled,
            )
            if not suggestion or suggestion.get("id") != mid:
                continue
            # Existing helper matches the play name only. Enforce the precise
            # model-selected formation as well.
            pair = f"{play} ({formation})"
            full_cap = 1 + sum(len(ps) for ps in book.values())
            if pair not in pairs_in_book(mid, book, cap=full_cap):
                continue
            detail = offense_detail(mid, book)
            if detail.get("needs_settings") or detail.get("gaps") or not detail.get("settings"):
                continue
            if not _verified_button_sequence(suggestion.get("buttons")):
                continue
            if suggestion.get("kind") == "look" and not credible_look:
                continue
            if suggestion.get("kind") == "situation" and not _situation_trigger(sit, "situation"):
                continue
            w = max(-0.2, min(0.2, float((weights or {}).get(mid, 0.0) or 0.0)))
            score = 0.24 + 0.12 * confidence + 0.05 * (probability - 0.5) + 0.2 * w
            if suggestion.get("kind") == "look":
                score += 0.055 if cls == "pressure" else 0.025
            else:
                score += 0.07 if getattr(sit, "goal_line", False) else 0.04
            rows.append({
                "kind": "macro", "id": mid, "score": round(score, 5),
                "why": suggestion.get("why"), "payload": suggestion,
            })
        except (ValueError, TypeError, KeyError):
            continue

    # User-confirmed, model-created custom macros are kept in a separate
    # auditable registry. A *draft blueprint* is never eligible, even if the
    # selected play fits and the research looks promising.
    if allow_macros and db is not None and opponent_id and credible_look:
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
            name = created["name"]
            buttons = research_db.buttons("offense", "custom_adjustments").replace(
                "pick the adjustment", name
            )
            if not _verified_button_sequence(buttons):
                continue
            payload = {
                "id": name, "name": name, "side": "offense", "kind": "look",
                "buttons": buttons, "why": created.get("fire_when") or "",
                "settings": created["settings"],
                "source_ids": created["source_ids"],
                "verified_armed": True,
            }
            rows.append({
                "kind": "macro", "id": name,
                "score": round(0.255 + 0.12 * confidence +
                               0.05 * (probability - 0.5) +
                               (0.055 if cls == "pressure" else 0.025), 5),
                "why": payload["why"], "payload": payload,
            })

    if credible_look:
        for a in offense_adjustment_candidates(
            play=play, formation=formation, coverage_class=cls,
            coverage_source=source, repeated=repeated, audibles=audibles,
        ):
            # An audible switches away from the model-selected play and needs
            # its own post-audible execution provenance. Keep disabled until
            # that is tracked as a separate confirmed action.
            if a.get("kind") == "audible":
                continue
            if not a.get("id") or not a.get("sources"):
                continue
            if not _verified_button_sequence(a.get("buttons")):
                continue
            score = 0.20 + 0.14 * confidence + 0.03 * (probability - 0.5)
            if a.get("kind") == "pass_protection" and cls == "pressure":
                score += 0.105
            if a.get("kind") == "hot_route" and cls in ("man", "cover2", "two_high"):
                score += 0.05
            rows.append({
                "kind": "adjustment", "id": a["id"], "score": round(score, 5),
                "why": a.get("why"), "payload": a,
            })

    rows.sort(key=lambda r: (-r["score"], r["kind"], str(r["id"])))
    summary = [{"kind": x["kind"], "id": x["id"], "score": x["score"]}
               for x in rows]
    summary.append({"kind": "none", "id": "NO_ADJUSTMENT",
                    "score": round(no_action_score, 5)})
    if not rows or rows[0]["score"] <= no_action_score:
        return {
            **empty,
            "reason": (
                "no verified action improves on running the selected play unchanged"
                if rows else "no eligible supported adjustment"
            ),
            "candidates": summary,
            "no_action_score": no_action_score,
            "top_action_score": rows[0]["score"] if rows else None,
        }

    best = rows[0]
    return {
        "policy_version": POLICY_VERSION,
        "kind": best["kind"], "id": best["id"],
        "macro": best["payload"] if best["kind"] == "macro" else None,
        "adjustment": best["payload"] if best["kind"] == "adjustment" else None,
        "reason": best["why"] or "",
        "scores_are": "research_policy_not_learned_action_effect",
        "candidates": summary[:8],
        "no_action_score": no_action_score,
        "top_action_score": best["score"],
        "why_now": (
            "observed or independently confirmed defensive look"
            if credible_look else "verified situational condition"
        ),
    }
