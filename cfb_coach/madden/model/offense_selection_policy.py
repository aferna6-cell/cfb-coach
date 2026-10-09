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
        score_us, score_them = (
            getattr(sit, "score_us", None), getattr(sit, "score_them", None)
        )
        if score_us is not None and score_them is not None:
            if score_us < score_them:
                delta -= 0.07
                reasons.append("trailing in two-minute drill")
            elif score_us > score_them:
                delta += 0.04
                reasons.append("protecting lead / keeping clock running")
        # With an unknown score, don't assume we are trailing.

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
    # O(N log N), not the earlier O(N²) strongest-alternative scan. The
    # formation designer may expose hundreds of legal plays, and the live
    # call must remain under its 150-ms clock budget.
    prob_order = sorted(
        ranked, key=lambda r: -float(r["probability"])
    )
    best_key = (str(prob_order[0]["formation"]), str(prob_order[0]["play"]))
    best_prob = float(prob_order[0]["probability"])
    runner_prob = next(
        (float(r["probability"]) for r in prob_order
         if (str(r["formation"]), str(r["play"])) != best_key),
        best_prob,
    )

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
        strongest_other = runner_prob if key == best_key else best_prob
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
    spread = (0.145 if float(choices[0].get("uncertainty", 1.0)) >= 0.55
              else 0.11) if cpu else 0.065
    # Never cap the model to 24/12 plays or two plays per concept. Every
    # situationally eligible play in the *entire applied playbook* stays in
    # the evaluated pool. Exploitation favors competitive options, while a
    # small separate exploration budget can sample the full eligible set.
    competitive = [
        row for row in choices
        if float(row["selection_score"]) >= best_score - spread
    ]
    if not competitive:
        competitive = [choices[0]]

    last = recent[0] if recent else None
    identical_twice = bool(last and len(recent) > 1 and recent[0] == recent[1])
    leader = choices[0]
    confident_dominance = (
        float(leader.get("uncertainty", 1.0)) < 0.35
        and leader.get("evidence_quality") in ("empirical", "verified")
        and float(leader["probability"]) -
        max((float(x["probability"]) for x in choices
             if (x["formation"], x["play"]) !=
             (leader["formation"], leader["play"])), default=0.0) > 0.18
    )
    seed = (
        f"{session_id or 'unscoped'}:{snap_seq if snap_seq is not None else len(recent)}:"
        f"{getattr(sit, 'down', None)}:{getattr(sit, 'distance', None)}:"
        f"{opponent_type}:{recent[0] if recent else '-'}"
    )
    random_bits = hashlib.sha256(seed.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(random_bits[:8], "big"))

    # Test games spend a modest fraction of choices learning the broader
    # installed playbook. Human-user games explore much more cautiously.
    # Even very low-ranked but football-eligible plays remain represented,
    # without claiming their unknown counterfactual outcomes are favorable.
    exploration_rate = 0.12 if cpu else 0.025
    broad_exploration = not confident_dominance and rng.random() < exploration_rate
    population = list(choices if broad_exploration else competitive)

    # Never call a third consecutive identical play or a fourth screen out
    # of five just because the current learned model has sparse data.
    if identical_twice and not confident_dominance:
        alternatives = [
            r for r in population
            if (r["formation"], r["play"]) != last
            and _play_concept(r["play"]) != _play_concept(last[1])
        ]
        if not alternatives:
            alternatives = [
                r for r in choices
                if (r["formation"], r["play"]) != last
                and _play_concept(r["play"]) != _play_concept(last[1])
            ]
        if alternatives:
            population = alternatives

    screen_streak = sum(_play_family(p) == "screen" for _f, p in last4)
    if screen_streak >= 3 and not confident_dominance:
        non_screen = [r for r in population if _play_family(r["play"]) != "screen"]
        if not non_screen:
            non_screen = [
                r for r in choices if _play_family(r["play"]) != "screen"
            ]
        if non_screen:
            population = non_screen
    if not population:
        population = [leader]

    # Concept-normalized sampling: one concept with 30 named variations
    # should not crowd out another with only three. No cap per concept.
    counts: dict[str, int] = {}
    for row in population:
        concept = str(row.get("play_concept") or _play_concept(row["play"]))
        counts[concept] = counts.get(concept, 0) + 1
    temperature = (0.19 if cpu else 0.13) if broad_exploration else (
        0.072 if cpu else 0.040
    )
    ceiling = max(float(r["selection_score"]) for r in population)
    weights = []
    for row in population:
        concept = str(row.get("play_concept") or _play_concept(row["play"]))
        # No zero-weight plays. Small novelty/uncertainty-aware exploration
        # is allowed only after situational filtering of the legal inventory.
        exponent = max(-16.0, (float(row["selection_score"]) - ceiling) / temperature)
        weights.append(math.exp(exponent) / math.sqrt(counts[concept]))
    chosen = rng.choices(population, weights=weights, k=1)[0]

    choices.remove(chosen)
    choices.insert(0, chosen)
    chosen["selection_reason"] = (
        "model-scored full-inventory exploration"
        if broad_exploration else (
            "model-scored eligible alternatives"
            if len(population) > 1 else "best model-supported eligible choice"
        )
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
        "shortlist_count": len(competitive),
        "diversified_shortlist_count": len(population),
        "full_eligible_population_count": len(choices),
        "sampling_population_count": len(population),
        "broad_exploration": broad_exploration,
        "exploration_rate": exploration_rate,
        "concepts_in_sampling_population": len(counts),
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


def summarize_call_variety(
    decisions: Sequence[Mapping[str, Any]],
    *,
    by_snap_situation: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Postgame *recommendation* metrics; never execution or causal claims."""
    situations = by_snap_situation or {}
    picks = []
    third_long_runs = 0
    third_long_known = 0
    for d in decisions:
        form, play = d.get("final_formation"), d.get("final_play")
        if not form or not play:
            continue
        picks.append((str(form), str(play)))
        snap = situations.get(str(d.get("snap_id") or "")) or {}
        try:
            down, distance = int(snap.get("down")), int(snap.get("distance"))
        except (ValueError, TypeError):
            continue
        if down in (3, 4) and distance >= 7:
            third_long_known += 1
            if is_run(str(play)):
                third_long_runs += 1

    longest = 0
    streak = 0
    prev = None
    for key in picks:
        streak = streak + 1 if key == prev else 1
        longest = max(longest, streak)
        prev = key
    n = len(picks)
    return {
        "recommended_calls": n,
        "distinct_plays": len(set(picks)),
        "distinct_formations": len({form for form, _ in picks}),
        "distinct_concepts": len({_play_concept(play) for _, play in picks}),
        "screen_calls": sum(_play_family(play) == "screen" for _, play in picks),
        "screen_share": round(
            sum(_play_family(play) == "screen" for _, play in picks) / n, 3
        ) if n else 0.0,
        "max_consecutive_same_play": longest,
        "known_third_or_fourth_long": third_long_known,
        "third_or_fourth_long_run_recommendations": third_long_runs,
        "note": (
            "These are displayed recommended calls. Without verified execution "
            "and outcome labels, do not attribute yards or performance to a call."
        ),
    }
