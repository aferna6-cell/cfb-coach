"""Situation-aware, model-primary offensive selection without one-play collapse.

The trained model still supplies the probability for *every* legal candidate.
Only the model's own action-exposure history supplies a repetition cost:
the heuristic coach never chooses/ranks a play. When enough alternatives are
available, consecutive identical calls and screen-family spam are discouraged.
Probabilities stay unchanged; selection score and penalties are auditable.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


POLICY = "model_primary_repetition_aware.v1"


def _recent_calls(db: Any, opponent_id: str, limit: int = 6) -> list[tuple[str, str]]:
    if db is None or not opponent_id:
        return []
    try:
        rows = db.get_recent_snaps(opponent_id, side="offense", limit=limit)
    except Exception:  # noqa: BLE001 — live ML degrades to stateless model
        return []
    recent = []
    for row in rows:
        # The installed coach recommendation is observable even when actual
        # execution is unverified; repetition of its *call* is the issue here.
        form, play = row["formation"], row["play"]
        if form and play:
            recent.append((str(form), str(play)))
    return recent


def choose_model_play(
    ranked: Sequence[Mapping[str, Any]], *,
    recent_calls: Sequence[tuple[str, str]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank in-book model candidates with a documented repeat-exposure cost.

    The model remains the sole substantive success-score source. The policy
    is deterministic and does not invent feedback for unexecuted alternatives.
    Do not force variety when every legal alternative is materially inferior
    (probability gap > .18) or when only one legal call is available.
    """
    if not ranked:
        return [], {"policy": POLICY, "reason": "empty candidate pool"}
    recent = list(recent_calls)[:6]  # newest first
    total = len(ranked)
    choices: list[dict[str, Any]] = []
    from cfb_coach.madden.model.experimental_model import _play_concept, _play_family

    for item in ranked:
        row = dict(item)
        key = (str(row["formation"]), str(row["play"]))
        is_screen = _play_family(key[1]) == "screen"
        last4 = recent[:4]
        repeats = sum(p == key for p in last4)
        # Exact play repeating back-to-back is often a *model* collapse.
        repeated_last = bool(recent and recent[0] == key)
        consecutive = len(recent) > 1 and recent[0] == key and recent[1] == key
        screen_exposure = sum(
            _play_family(p[1]) == "screen" for p in last4
        )
        concept_exposure = sum(
            _play_concept(p[1]) == row.get("play_concept", _play_concept(key[1]))
            for p in last4
        )
        penalty = 0.0
        if total > 1:
            penalty += min(0.24, 0.06 * repeats)
            if repeated_last:
                penalty += 0.10
            if consecutive:
                penalty += 0.14
            if is_screen and screen_exposure >= 2:
                penalty += 0.09 + 0.03 * (screen_exposure - 2)
            if concept_exposure >= 3:
                penalty += 0.03
        # Do not rewrite calibrated model probabilities as penalized forecasts.
        row["selection_penalty"] = round(penalty, 5)
        row["selection_score"] = round(float(row["probability"]) - penalty, 7)
        row["selection_policy"] = POLICY
        choices.append(row)

    choices.sort(
        key=lambda r: (-r["selection_score"], -float(r["probability"]),
                       r.get("uncertainty", 1.0), r["formation"], r["play"])
    )
    # If exact call was recommended on two consecutive snaps, and a
    # reasonably close different concept exists, force a model-ranked pivot.
    last = recent[0] if recent else None
    has_twice = len(recent) > 1 and recent[0] == recent[1]
    if last and has_twice and total > 1 and (
        choices[0]["formation"], choices[0]["play"]
    ) == last:
        different = [
            r for r in choices if (r["formation"], r["play"]) != last
            and r.get("play_concept") != _play_concept(last[1])
            and float(ranked[0]["probability"]) - float(r["probability"]) <= 0.18
        ]
        if different:
            pivot = different[0]
            choices.remove(pivot)
            choices.insert(0, pivot)
            pivot["selection_reason"] = "two identical calls; credible model-ranked alternative"

    for i, r in enumerate(choices, start=1):
        r["rank"] = i
    return choices, {
        "policy": POLICY, "recent_calls_used": len(recent),
        "top_original": [ranked[0]["formation"], ranked[0]["play"]],
        "top_selected": [choices[0]["formation"], choices[0]["play"]],
        "selection_penalty": choices[0]["selection_penalty"],
        "reason": choices[0].get("selection_reason") or "model score adjusted for repeated calls",
    }


def select_from_database(
    ranked: Sequence[Mapping[str, Any]], *, db: Any, opponent_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return choose_model_play(ranked, recent_calls=_recent_calls(db, opponent_id))
