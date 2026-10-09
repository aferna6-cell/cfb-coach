"""Situationally sane, auditable and varied *model-primary* Madden offense.

A verified-data-trained success model scores each eligible play. This policy
adds football safety constraints, recent-calls exposure costs and reproducible
top-set sampling. The heuristic never picks a normal offensive call.

Sampling is exploration among plausible plays, NOT evidence that an unchosen
play would have performed better. The original model probabilities remain
unmodified and are logged separately from selection scores.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from cfb_coach.madden.catalog import is_deep, is_run
from cfb_coach.madden.model.experimental_model import _play_concept, _play_family

POLICY = "model_primary_contextual_variety.v2"


def _recent_calls(
    db: Any, opponent_id: str, limit: int = 20,
    session_id: str | None = None,
) -> list[tuple[str, str]]:
    """Observe *recommended* calls, never infer their actual execution."""
    if db is None or not opponent_id:
        return []
    try:
        if session_id:
            rows = db.conn.execute(
                "SELECT formation, play FROM snaps "
                "WHERE opponent_id=? AND side='offense' AND session_id=? "
                "ORDER BY id DESC LIMIT ?",
                (opponent_id, session_id, limit),
            ).fetchall()
        else:
            rows = db.get_recent_snaps(opponent_id, side="offense", limit=limit)
        return [(str(r["formation"]), str(r["play"])) for r in rows
                if r["formation"] and r["play"]]
    except Exception:  # noqa: BLE001 — safe stateless ranking
        return []


def _situational_adjustment(play: str, sit: Any) -> tuple[float, str]:
    """Small play-specific suitability signal; not a heuristic play choice."""
    if sit is None:
        return 0.0, "situation unavailable"
    down, distance = getattr(sit, "down", None), getattr(sit, "distance", None)
    two_minute = bool(getattr(sit, "two_minute", False))
    if down is None or distance is None:
        return 0.0, "down/distance unknown"
    try:
        d, yards = int(down), int(distance)
    except (TypeError, ValueError):
        return 0.0, "down/distance unparseable"
    run, screen = is_run(play), _play_family(play) == "screen"
    delta = 0.0
    reasons: list[str] = []
    if d in (3, 4) and yards >= 7:
        if run:
            delta -= 0.45  # eligible only if no legal pass survives
            reasons.append("third/fourth-and-long ground gain risk")
        if screen:
            delta -= 0.09
            reasons.append("screen behind conversion distance")
        if not run and not screen and is_deep(play):
            delta += 0.025
            reasons.append("route potentially reaches sticks")
    elif d in (3, 4) and yards <= 2:
        if run:
            delta += 0.025
            reasons.append("short-yardage run option")
        if is_deep(play):
            delta -= 0.035
            reasons.append("long-developing route on short yardage")
    if two_minute and yards >= 4 and run:
        delta -= 0.07
        reasons.append("two-minute tempo")
    return delta, "; ".join(reasons) or "normal situation"


def _stable_sample(
    options: list[dict[str, Any]], *,
    seed: str, temperature: float,
) -> dict[str, Any]:
    if len(options) == 1:
        return options[0]
    ceiling = max(float(o["selection_score"]) for o in options)
    weights = [
        math.exp(max(-10.0, (float(row["selection_score"]) - ceiling) / temperature))
        for row in options
    ]
    raw = hashlib.sha256(seed.encode("utf-8")).digest()
    return random.Random(int.from_bytes(raw[:8], "big")).choices(options, weights=weights, k=1)[0]


def choose_model_play(
    ranked: Sequence[Mapping[str, Any]], *,
    recent_calls: Sequence[tuple[str, str]] = (),
    sit: Any = None,
    opponent_type: str = "cpu",
    session_id: str | None = None,
    snap_seq: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pick from top model-backed choices, with long-horizon exposure costs.

    For CPU, explore more broadly among sufficiently comparable candidates.
    For human users, use a narrower candidate window. Session/snap-based seeds
    make replay/auditing deterministic, never round-robin through a fixed menu.
    """
    if not ranked:
        return [], {"policy": POLICY, "reason": "no eligible model candidates"}
    recent = list(recent_calls)[:20]
    cpu = opponent_type == "cpu"
    total = len(ranked)
    last4 = recent[:4]
    choices: list[dict[str, Any]] = []
    pair_count = {k: recent.count(k) for k in set(recent)}
    family_count = {
        family: sum(_play_family(p) == family for _f, p in recent)
        for family in ("run", "screen", "pass", "rpo")
    }
    concept_count = {
        concept: sum(_play_concept(p) == concept for _f, p in recent)
        for concept in {_play_concept(p) for _f, p in recent}
    }
    form_count = {f: sum(prev == f for prev, _p in recent) for f, _p in recent}

    for item in ranked:
        row = dict(item)
        key = (str(row["formation"]), str(row["play"]))
        fam = _play_family(key[1])
        concept = str(row.get("play_concept") or _play_concept(key[1]))
        count = pair_count.get(key, 0)
        screen_exposure = family_count.get("screen", 0)
        penalty = 0.0
        if total > 1:
            penalty += min(0.17, 0.028 * count)
            penalty += min(0.11, 0.016 * concept_count.get(concept, 0))
            penalty += min(0.045, 0.0045 * form_count.get(key[0], 0))
            if recent and recent[0] == key:
                penalty += 0.08
            if len(recent) > 1 and recent[0] == recent[1] == key:
                penalty += 0.12
            if fam == "screen" and screen_exposure >= 2:
                penalty += min(0.14, 0.05 + .015 * (screen_exposure - 2))
            if sum(p == key for p in last4) >= 3:
                penalty += 0.055
        situation_delta, situation_reason = _situational_adjustment(key[1], sit)
        # Do not discard a high-confidence/high-margin finding for cosmetic
        # variety. In contrast, low-data inflated scores are not sacrosanct.
        strongest_other = max(
            (float(p["probability"]) for p in ranked
             if (str(p["formation"]), str(p["play"])) != key),
            default=float(row["probability"]),
        )
        confident = (
            float(row.get("uncertainty", 1.0)) < 0.35
            and row.get("evidence_quality") in ("empirical", "verified")
        )
        if confident and float(row["probability"]) - strongest_other > 0.18:
            penalty = min(penalty, 0.065)
        row["selection_penalty"] = round(penalty, 5)
        row["situation_adjustment"] = round(situation_delta, 5)
        row["situation_reason"] = situation_reason
        row["selection_score"] = round(
            float(row["probability"]) + situation_delta - penalty, 7
        )
        row["selection_policy"] = POLICY
        choices.append(row)

    choices.sort(
        key=lambda r: (-r["selection_score"], -float(r["probability"]),
                       r.get("uncertainty", 1.0), r["formation"], r["play"])
    )
    best_score = float(choices[0]["selection_score"])
    # Controlled exploration among similarly valued *and situationally valid*
    # plays, not arbitrary random play calls.
    spread = 0.11 if cpu else 0.065
    shortlist = [
        row for row in choices
        if float(row["selection_score"]) >= best_score - spread
    ][:24 if cpu else 12]

    # Avoid filling a candidate set with twelve near-identical plays. Retain
    # the best two plays from any concept, increasing tactical unpredictability.
    diverse: list[dict[str, Any]] = []
    per_concept: dict[str, int] = {}
    for row in shortlist:
        concept = str(row.get("play_concept") or _play_concept(row["play"]))
        if per_concept.get(concept, 0) >= 2:
            continue
        per_concept[concept] = per_concept.get(concept, 0) + 1
        diverse.append(row)
    if not diverse:
        diverse = [choices[0]]

    # After two repeated play recommendations, prefer a different concept
    # unless strongly verified performance makes it clearly better.
    last = recent[0] if recent else None
    if (
        last and len(recent) >= 2 and recent[0] == recent[1]
        and (choices[0]["formation"], choices[0]["play"]) == last
    ):
        alternatives = [
            row for row in diverse
            if _play_concept(row["play"]) != _play_concept(last[1])
        ]
        if alternatives:
            diverse = alternatives

    seed = (
        f"{session_id or 'unscoped'}:{snap_seq if snap_seq is not None else len(recent)}:"
        f"{getattr(sit, 'down', None)}:{getattr(sit, 'distance', None)}:"
        f"{opponent_type}:{recent[0] if recent else '-'}"
    )
    chosen = _stable_sample(
        diverse, seed=seed,
        temperature=0.072 if cpu else 0.040,
    )
    choices.remove(chosen)
    choices.insert(0, chosen)
    chosen["selection_reason"] = (
        "model-controlled exploration among credible situational candidates"
        if len(diverse) > 1 else "best model-supported eligible choice"
    )
    for i, row in enumerate(choices, start=1):
        row["rank"] = i

    audit = {
        "policy": POLICY,
        "recent_calls_used": len(recent),
        "top_original": [ranked[0]["formation"], ranked[0]["play"]],
        "top_selected": [chosen["formation"], chosen["play"]],
        "original_probability": float(chosen["probability"]),
        "selection_score": chosen["selection_score"],
        "selection_penalty": chosen["selection_penalty"],
        "situation_adjustment": chosen["situation_adjustment"],
        "situation_reason": chosen["situation_reason"],
        "candidate_count": len(ranked),
        "shortlist_count": len(shortlist),
        "diversified_shortlist_count": len(diverse),
        "opponent_type": opponent_type,
        "seed_source": "stable_session_snap",  # do not log a secret/random seed
        "reason": chosen["selection_reason"],
    }
    return choices, audit


def select_from_database(
    ranked: Sequence[Mapping[str, Any]], *,
    db: Any, opponent_id: str,
    sit: Any = None, opponent_type: str = "cpu",
    session_id: str | None = None, snap_seq: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return choose_model_play(
        ranked,
        recent_calls=_recent_calls(db, opponent_id, session_id=session_id),
        sit=sit, opponent_type=opponent_type,
        session_id=session_id, snap_seq=snap_seq,
    )
