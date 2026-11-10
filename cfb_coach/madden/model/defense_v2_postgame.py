"""Defensive V2 postgame challenger evaluation; never promotes a model.

Champion remains frozen as of its recorded cutoff. For every subsequent game,
challenger is re-fit exclusively on earlier games, and both predict identical
chronological holdout snaps with only prior same-game opponent observations.
"""
from __future__ import annotations

from collections import defaultdict
import random
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import defense_v2_pilot as v2

MIN_NEW_GAMES = 8
MIN_NEW_SNAPS = 200


def _predict(model: Mapping[str, Any], r: v2.Snap,
             same_game: Sequence[v2.Snap]) -> float:
    belief = v2.concept_belief(model, same_game, session=r.session,
                               opponent=r.opponent, next_seq=int(r.seq or 0))
    return sum(
        belief[c] * v2.offense_risk(
            model, r.context, r.formation, r.play, r.family, c)[0]
        for c in v2.CONCEPTS
    )


def compare_champion(
    champion: Mapping[str, Any], rows: Sequence[v2.Snap],
    *, bootstrap: int = 600, seed: int = 19,
) -> dict[str, Any]:
    if champion.get("schema") != v2.SCHEMA or champion.get("policy") != "shadow_only":
        raise ValueError("not a V2 shadow champion")
    cutoff = champion.get("trained_through_started_ts")
    sessions = champion.get("trained_sessions")
    if not isinstance(cutoff, str) or not cutoff or not isinstance(sessions, list):
        raise ValueError("champion has no explicit game and time provenance")
    group: dict[str, list[v2.Snap]] = defaultdict(list)
    for r in rows:
        group[r.session].append(r)
    heldout = sorted(
        ((game, min(r.started_ts for r in snaps), snaps)
         for game, snaps in group.items()
         if min(r.started_ts for r in snaps) > cutoff and game not in sessions),
        key=lambda item: (item[1], item[0]),
    )
    scores: list[dict[str, Any]] = []
    for game, start, snaps in heldout:
        prior = [r for r in rows if r.started_ts < start]
        if not prior:
            continue
        challenger = v2.fit(prior)
        closs = vloss = 0.0
        count = 0
        for r in sorted(snaps, key=lambda r: (r.seq is None, r.seq or 0)):
            if r.seq is None or r.seq <= 0:
                continue
            closs += v2.logloss(_predict(champion, r, snaps), r.offense_success)
            vloss += v2.logloss(_predict(challenger, r, snaps), r.offense_success)
            count += 1
        if count:
            scores.append({"game": game, "n": count,
                           "champion_logloss": closs / count,
                           "challenger_logloss": vloss / count})
    n = sum(r["n"] for r in scores)
    gain = (sum((r["champion_logloss"] - r["challenger_logloss"]) * r["n"]
                for r in scores) / n if n else 0.0)
    ci = None
    if len(scores) >= 2:
        rng = random.Random(seed)
        draws = []
        for _ in range(bootstrap):
            sample = [rng.choice(scores) for __ in scores]
            count = sum(row["n"] for row in sample)
            draws.append(sum(
                (row["champion_logloss"] - row["challenger_logloss"]) * row["n"]
                for row in sample
            ) / count)
        draws.sort()
        ci = [draws[int(0.025 * (bootstrap - 1))],
              draws[int(0.975 * (bootstrap - 1))]]
    positive = bool(len(scores) >= MIN_NEW_GAMES and n >= MIN_NEW_SNAPS and
                    ci is not None and ci[0] > 0.0 and gain >= 0.02)
    return {
        "schema": "madden.defense.v2.postgame.report.v1",
        "champion_cutoff": cutoff, "new_games": len(scores),
        "new_snaps": n, "per_game": scores,
        "challenger_minus_champion_logloss_improvement": round(gain, 6),
        "game_cluster_95ci": ci,
        "predictive_promotion_gate": "PASS" if positive else "NO_GO",
        "actual_policy_value_proven": False,
        "model_promotion": "MANUAL_REVIEW_REQUIRED",
        "warning": (
            "Temporal predictive improvement is not causal defensive policy value. "
            "Keep champion unchanged without explicit owner review."
        ),
    }
