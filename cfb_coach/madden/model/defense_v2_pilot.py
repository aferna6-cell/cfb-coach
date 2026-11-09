"""Experimental defensive V2: verified CPU/human executions, time-correct beliefs, learned risk.

Shadow-only. No live caller changes, DB mutations, or model promotion. Observational
outcomes cannot establish the causal value of a defensive call.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import random
import sqlite3
from typing import Any, Mapping, Sequence

SCHEMA = "madden.defense.v2.shadow.v1"
CONCEPTS = ("run", "rpo", "vert", "flood", "cross", "stack", "scram")
MIN_GAMES = 8
MIN_EVAL_SNAPS = 200


@dataclass(frozen=True)
class Snap:
    session: str
    opponent: str
    started_ts: str
    seq: int | None
    formation: str
    play: str
    family: str
    context: str
    concept: str | None
    offense_success: bool
    yards: int | None = None
    touchdown: bool | None = None
    explosive: bool | None = None
    turnover: bool | None = None
    converted: bool | None = None


@dataclass(frozen=True)
class Observation:
    session: str
    opponent: str
    seq: int
    concept: str


def context_key(down: int | None, distance: int | None) -> str:
    if down is None or distance is None:
        return "unknown"
    if distance <= 3:
        return "short"
    if down >= 3 and distance >= 7:
        return "long"
    if down >= 3:
        return "medium"
    return "early"


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def available_franchises(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if "dynasty" not in _columns(conn, "game_sessions"):
        raise ValueError("game_sessions.dynasty unavailable: cannot isolate franchise")
    return [{"franchise": row[0], "games": row[1]} for row in
            conn.execute("SELECT dynasty, COUNT(*) FROM game_sessions GROUP BY dynasty ORDER BY dynasty")]


def verified_rows(conn: sqlite3.Connection, *, franchise: str) -> tuple[list[Snap], dict[str, int]]:
    """Strict execution labels only, including CPU games; do not train on recommendations."""
    if not franchise or not franchise.strip():
        raise ValueError("explicit franchise identifier required")
    required = {"session_id", "side", "result", "down", "distance", "executed_status",
                "executed_verification", "executed_formation", "executed_play", "snap_seq"}
    columns = _columns(conn, "snaps")
    if not required <= columns:
        raise ValueError("missing defensive verification columns: " + ", ".join(sorted(required - columns)))
    if not {"session_id", "opponent_id", "dynasty", "started_ts"} <= _columns(conn, "game_sessions"):
        raise ValueError("game session franchise/opponent identity unavailable")
    concept_col = "COALESCE(NULLIF(s.concept_typed,''),s.concept_seen)" if "concept_typed" in columns else (
        "s.concept_seen" if "concept_seen" in columns else "NULL")
    sql = ("SELECT s.session_id,g.opponent_id,g.started_ts,s.snap_seq,s.executed_status,s.executed_verification,"
           "s.executed_formation,s.executed_play,s.result,s.down,s.distance," + concept_col +
           " FROM snaps s JOIN game_sessions g ON g.session_id=s.session_id"
           " WHERE s.side='defense' AND g.dynasty=? ORDER BY g.started_ts,s.id")
    from cfb_coach.madden.model.defense_v1 import offense_won
    from cfb_coach.outcome import parse_outcome
    from cfb_coach.madden.defense_select import call_family
    from cfb_coach.madden.situation import concept_family

    stats: Counter[str] = Counter()
    rows: list[Snap] = []
    seen_sequences: set[tuple[str, int]] = set()
    for (session, opponent, started, seq, status, verification, formation, play, result, down, dist, concept) in conn.execute(sql, (franchise,)):
        stats["defensive_rows"] += 1
        if status != "identified" or verification != "verified" or not (formation and play):
            stats["unverified_execution"] += 1
            continue
        won = offense_won(result, down, dist)
        if won is None:
            stats["unlabeled_outcome"] += 1
            continue
        fam = call_family(str(play))
        if not fam:
            stats["unrecognized_defensive_family"] += 1
            continue
        if seq is not None:
            identity = (str(session), int(seq))
            if identity in seen_sequences:
                stats["duplicate_verified_sequence"] += 1
                continue
            seen_sequences.add(identity)
        normalized = concept_family(concept)
        if normalized not in CONCEPTS:
            normalized = None
        parsed = parse_outcome(str(result))
        known_result = parsed.kind != "unknown"
        conversion = (True if parsed.kind == "convert" else
                      (bool(parsed.yards >= int(dist)) if
                       parsed.yards is not None and dist is not None and
                       down is not None and int(down) >= 3 else None))
        rows.append(Snap(str(session), str(opponent or ""), str(started or ""),
                         int(seq) if seq is not None else None, str(formation), str(play),
                         fam, context_key(down, dist), normalized, bool(won),
                         yards=parsed.yards, touchdown=(parsed.kind == "td") if known_result else None,
                         explosive=(parsed.yards >= 20) if parsed.yards is not None else None,
                         turnover=parsed.is_turnover if known_result else None,
                         converted=conversion))
        stats["admitted_verified"] += 1
    stats["games"] = len({r.session for r in rows})
    stats["seq_known"] = sum(r.seq is not None for r in rows)
    stats["concept_known"] = sum(r.concept is not None for r in rows)
    return rows, dict(stats)


def earlier_opponent_observations(conn: sqlite3.Connection, *, franchise: str,
                                   session: str, opponent: str, next_seq: int) -> list[Observation]:
    """Read opponent observations independently of whether OUR execution was verified.

    Exclude unknown sequence numbers, open snaps, current/future snaps, other
    opponents, and other sessions/franchises.
    """
    if not all((franchise, session, opponent)) or next_seq <= 0:
        raise ValueError("explicit franchise, session, opponent and positive next_seq required")
    cols = _columns(conn, "snaps")
    if not {"session_id", "side", "snap_seq", "result"} <= cols:
        raise ValueError("snaps lacks chronological observation columns")
    concept = "COALESCE(NULLIF(s.concept_typed,''),s.concept_seen)" if "concept_typed" in cols else (
        "s.concept_seen" if "concept_seen" in cols else "NULL")
    from cfb_coach.madden.situation import concept_family
    sql = ("SELECT s.snap_seq," + concept + " FROM snaps s JOIN game_sessions g ON g.session_id=s.session_id"
           " WHERE g.dynasty=? AND s.session_id=? AND g.opponent_id=? AND s.side='defense'"
           " AND s.snap_seq > 0 AND s.snap_seq < ? AND NULLIF(TRIM(s.result), '') IS NOT NULL"
           " ORDER BY s.snap_seq,s.id")
    observations = []
    seen: set[int] = set()
    for seq, raw in conn.execute(sql, (franchise, session, opponent, next_seq)):
        if int(seq) in seen:
            continue
        family = concept_family(raw)
        if family in CONCEPTS:
            observations.append(Observation(session, opponent, int(seq), family))
            seen.add(int(seq))
    return observations


def _add(table: dict[str, list[int]], key: str, success: bool) -> None:
    val = table.setdefault(key, [0, 0])
    val[0] += int(success)
    val[1] += 1


def _key(*parts: str) -> str:
    return json.dumps(parts, separators=(",", ":"))


def fit(rows: Sequence[Snap]) -> dict[str, Any]:
    """Learn hierarchical success probabilities, not prewritten football counters."""
    glob: dict[str, list[int]] = {}
    ctx: dict[str, list[int]] = {}
    fam: dict[str, list[int]] = {}
    matchup: dict[str, list[int]] = {}
    play: dict[str, list[int]] = {}
    concepts: Counter[str] = Counter()
    outcomes: dict[str, dict[str, float | int]] = {}
    for r in rows:
        _add(glob, "all", r.offense_success)
        _add(ctx, r.context, r.offense_success)
        _add(fam, _key(r.context, r.family), r.offense_success)
        _add(play, _key(r.context, r.formation, r.play), r.offense_success)
        aux = outcomes.setdefault(_key(r.context, r.family), {})
        for attr, metric in (("touchdown", "td"), ("explosive", "explosive"),
                             ("turnover", "turnover"), ("converted", "conversion")):
            val = getattr(r, attr)
            if val is not None:
                aux[metric + "_n"] = int(aux.get(metric + "_n", 0)) + 1
                aux[metric + "_yes"] = int(aux.get(metric + "_yes", 0)) + int(bool(val))
        if r.yards is not None:
            aux["yards_n"] = int(aux.get("yards_n", 0)) + 1
            aux["yards_sum"] = float(aux.get("yards_sum", 0.0)) + r.yards
        if r.concept is not None:
            concepts[r.concept] += 1
            _add(matchup, _key(r.context, r.family, r.concept), r.offense_success)
    model = {"schema": SCHEMA, "target": "offense_success", "policy": "shadow_only",
             "n_verified": len(rows), "n_games": len({r.session for r in rows}),
             "global": glob, "contexts": ctx, "families": fam, "matchups": matchup,
             "plays": play, "concept_counts": dict(concepts),
             "outcome_targets": outcomes,
             "evidence": "verified execution only; observational, not causal"}
    model["structural"] = _fit_structural(rows, model)
    return model


def _fit_structural(rows: Sequence[Snap], model: Mapping[str, Any]) -> dict[str, Any]:
    """Ridge residual learner over documented defensive candidate feature axes.

    A football label alone never supplies empirical effect. Only fit this term
    with >=8 distinct verified games and >=200 verified snaps. No synthetic
    prior coefficients are installed for cold-start games.
    """
    if len(rows) < 200 or len({r.session for r in rows}) < 8:
        return {"active": False, "reason": "requires 200 verified snaps in 8 distinct games"}
    import numpy as np
    from cfb_coach.madden.model import defense_semantics as ds

    names = list(ds.CANDIDATE_NAMES)
    eligible = []
    for row in rows:
        feats = ds.candidate_features(row.formation, row.play)
        if feats.get("dk_cand_in_catalog", 0.0) <= 0:
            continue
        eligible.append((row, [float(feats[n]) for n in names]))
    if len(eligible) < 160:
        return {"active": False, "reason": "insufficient catalog-backed verified assignments"}
    X = np.asarray([vec for _, vec in eligible], dtype=float)
    centers = X.mean(axis=0)
    centered = X - centers
    outcomes = np.asarray([float(row.offense_success) - baseline(model, row.context)
                           for row, _ in eligible], dtype=float)
    # Zero-baseline, strong shrinkage. Does not claim an intervention effect.
    alpha = 60.0
    matrix = centered.T @ centered + np.eye(X.shape[1]) * alpha
    coefficients = np.linalg.solve(matrix, centered.T @ outcomes)
    return {"active": True, "method": "ridge_centered_context_residual",
            "n_train": len(eligible), "n_games": len({r.session for r, _ in eligible}),
            "names": names, "centers": centers.tolist(),
            "coefficients": coefficients.tolist(),
            "max_delta": 0.07,
            "note": "structural hypothesis trained from observational verified executions"}


def structural_delta(model: Mapping[str, Any], formation: str, play: str) -> float:
    art = model.get("structural") or {}
    if not art.get("active"):
        return 0.0
    from cfb_coach.madden.model import defense_semantics as ds
    feats = ds.candidate_features(formation, play)
    if feats.get("dk_cand_in_catalog", 0.0) <= 0:
        return 0.0
    names, centers, coefs = art["names"], art["centers"], art["coefficients"]
    if len(names) != len(centers) or len(names) != len(coefs):
        raise ValueError("invalid structural model dimensions")
    delta = sum((float(feats[name]) - float(center)) * float(coef)
                for name, center, coef in zip(names, centers, coefs))
    if not math.isfinite(delta):
        raise ValueError("nonfinite structural score")
    cap = min(0.07, max(0.0, float(art.get("max_delta", 0.0))))
    return max(-cap, min(cap, delta))


def _posterior(count: Sequence[int] | None, prior: float, strength: float) -> tuple[float, int]:
    success, n = (int(count[0]), int(count[1])) if count else (0, 0)
    return (success + strength * prior) / (n + strength), n


def baseline(model: Mapping[str, Any], context: str) -> float:
    p, _ = _posterior(model["global"].get("all"), 0.5, 4.0)
    return _posterior(model["contexts"].get(context), p, 16.0)[0]


def offense_risk(model: Mapping[str, Any], context: str, formation: str,
                 play: str, family: str, concept: str | None) -> tuple[float, dict[str, int]]:
    base = baseline(model, context)
    family_p, family_n = _posterior(model["families"].get(_key(context, family)), base, 16.0)
    play_p, play_n = _posterior(model["plays"].get(_key(context, formation, play)), family_p, 24.0)
    structural = structural_delta(model, formation, play)
    if concept is None:
        return max(0.001, min(0.999, play_p + structural)), {
            "family": family_n, "play": play_n, "matchup": 0}
    matchup_p, matchup_n = _posterior(model["matchups"].get(_key(context, family, concept)), family_p, 16.0)
    predicted = 0.7 * matchup_p + 0.3 * play_p + structural
    return max(0.001, min(0.999, predicted)), {
        "family": family_n, "play": play_n, "matchup": matchup_n}


def auxiliary_risk(model: Mapping[str, Any], context: str, family: str) -> dict[str, float | int]:
    """Smoothed observational secondary targets. UNKNOWN means abstain, not zero."""
    raw = (model.get("outcome_targets") or {}).get(_key(context, family)) or {}
    metrics: dict[str, float | int] = {}
    if not raw:
        return metrics
    # These are weak generic shrinkage priors, not learned play effects.
    defaults = {"td": (0.05, 20.0), "explosive": (0.10, 16.0),
                "turnover": (0.04, 20.0), "conversion": (0.30, 12.0)}
    for metric, (prior, strength) in defaults.items():
        n = int(raw.get(metric + "_n", 0))
        if n >= 10:
            yes = int(raw.get(metric + "_yes", 0))
            metrics[metric] = (yes + strength * prior) / (n + strength)
            metrics[metric + "_n"] = n
    n_yards = int(raw.get("yards_n", 0))
    if n_yards >= 10:
        metrics["expected_yards"] = (float(raw.get("yards_sum", 0.0)) + 12 * 5.0) / (n_yards + 12)
        metrics["yards_n"] = n_yards
    return metrics


def defensive_utility_risk(predicted_success: float, metrics: Mapping[str, float | int]) -> float:
    """Higher is worse for defense. Weights are transparent utility preferences,
    not pre-programmed coverage counters; data supplies each available outcome.
    """
    risk = predicted_success
    if "expected_yards" in metrics:
        risk += 0.002 * (float(metrics["expected_yards"]) - 5.0)
    if "td" in metrics:
        risk += 0.15 * (float(metrics["td"]) - 0.05)
    if "explosive" in metrics:
        risk += 0.10 * (float(metrics["explosive"]) - 0.10)
    if "turnover" in metrics:
        risk -= 0.10 * (float(metrics["turnover"]) - 0.04)
    if "conversion" in metrics:
        risk += 0.06 * (float(metrics["conversion"]) - 0.30)
    return risk


def concept_belief(model: Mapping[str, Any], prior_snaps: Sequence[Snap | Observation], *,
                   session: str, opponent: str, next_seq: int) -> dict[str, float]:
    """Update concept distribution using strictly earlier same-game evidence."""
    if next_seq <= 0:
        raise ValueError("next_seq must be a positive pre-snap sequence")
    counts = Counter({c: 1.0 + float(model["concept_counts"].get(c, 0)) for c in CONCEPTS})
    prior_n = sum(counts.values())
    counts = Counter({c: (n / prior_n) * 12.0 for c, n in counts.items()})
    seen: set[int] = set()
    for r in prior_snaps:
        if (r.session == session and r.opponent == opponent and
                r.seq is not None and 0 < r.seq < next_seq and r.seq not in seen and r.concept in CONCEPTS):
            counts[r.concept] += 1.0
            seen.add(r.seq)
    total = sum(counts.values())
    return {c: float(counts[c] / total) for c in CONCEPTS}


def rank(model: Mapping[str, Any], *, context: str, calls: Sequence[Mapping[str, Any]],
         belief: Mapping[str, float]) -> dict[str, Any]:
    """Rank only explicitly installed/legal calls. Does not apply any call."""
    if model.get("schema") != SCHEMA or model.get("policy") != "shadow_only":
        raise ValueError("unsupported or non-shadow defensive model")
    from cfb_coach.madden.defense_select import call_family
    if any(float(v) < 0 or not math.isfinite(float(v)) for v in belief.values()):
        raise ValueError("invalid offense concept probability")
    total = sum(float(belief.get(c, 0)) for c in CONCEPTS)
    if total <= 0 or abs(total - 1.0) > 1e-6:
        raise ValueError("offense concept probabilities must sum to one")
    scores = []
    for cand in calls:
        if cand.get("installed") is not True or cand.get("legal") is not True:
            continue
        form, play = str(cand.get("formation") or ""), str(cand.get("play") or "")
        fam = call_family(play)
        if not (form and play and fam):
            continue
        weighted = 0.0
        for concept in CONCEPTS:
            p, _n = offense_risk(model, context, form, play, fam, concept)
            weighted += float(belief.get(concept, 0)) * p
        _, support = offense_risk(model, context, form, play, fam, None)
        secondary = auxiliary_risk(model, context, fam)
        risk = defensive_utility_risk(weighted, secondary)
        scores.append({"formation": form, "play": play, "family": fam,
                       "predicted_offense_success": round(weighted, 5),
                       "defensive_utility_risk": round(risk, 5), "secondary_outcomes": secondary,
                       "support": support,
                       "structural_model_active": bool((model.get("structural") or {}).get("active")),
                       "learned_structural_delta": round(structural_delta(model, form, play), 5)})
    scores.sort(key=lambda x: (x["defensive_utility_risk"], x["formation"], x["play"]))
    return {"schema": SCHEMA, "applied": False, "ready_for_live": False,
            "model_games": model["n_games"], "model_snaps": model["n_verified"],
            "ranked": scores, "reason": "observational shadow predictions; no verified live gate"}


def logloss(p: float, y: bool) -> float:
    q = max(1e-5, min(1 - 1e-5, p))
    return -math.log(q if y else 1 - q)


def evaluate(rows: Sequence[Snap], *, bootstraps: int = 400, seed: int = 27) -> dict[str, Any]:
    """Forward-only whole-game replay; no future game/current-snap leakage."""
    grouped: dict[str, list[Snap]] = defaultdict(list)
    for r in rows:
        grouped[r.session].append(r)
    per_game: list[tuple[str, float, float, int]] = []
    for game, holdout in sorted(grouped.items(),
                                key=lambda item: (min(r.started_ts for r in item[1]), item[0])):
        started = min(r.started_ts for r in holdout)
        other = [r for r in rows if r.started_ts < started]
        if not other:
            continue
        model = fit(other)
        baseline_loss = challenger_loss = 0.0
        tested = 0
        for r in sorted(holdout, key=lambda item: (item.seq is None, item.seq or 0)):
            if r.seq is None or r.seq <= 0:
                continue
            distribution = concept_belief(model, holdout, session=r.session,
                                          opponent=r.opponent, next_seq=r.seq)
            prediction = sum(distribution[c] * offense_risk(model, r.context, r.formation,
                              r.play, r.family, c)[0] for c in CONCEPTS)
            challenger_loss += logloss(prediction, r.offense_success)
            baseline_loss += logloss(baseline(model, r.context), r.offense_success)
            tested += 1
        if tested:
            per_game.append((game, baseline_loss, challenger_loss, tested))
    total = sum(item[3] for item in per_game)
    gain = sum(item[1] - item[2] for item in per_game) / total if total else 0.0
    interval = None
    if len(per_game) >= 2:
        rng = random.Random(seed)
        values = []
        for _ in range(bootstraps):
            sample = [rng.choice(per_game) for __ in per_game]
            n = sum(x[3] for x in sample)
            values.append(sum(x[1] - x[2] for x in sample) / n)
        values.sort()
        interval = [values[int(0.025 * (bootstraps - 1))],
                    values[int(0.975 * (bootstraps - 1))]]
    passes = bool(total >= MIN_EVAL_SNAPS and len(per_game) >= MIN_GAMES and
                  interval is not None and interval[0] > 0 and gain >= 0.02)
    return {"heldout_games": len(per_game), "heldout_snaps": total,
            "logloss_improvement_vs_context_baseline": round(gain, 6),
            "game_cluster_bootstrap_95ci": interval,
            "prediction_gate": "PASS" if passes else "NO_GO",
            "live_activation": "NOT_ALLOWED", "observational_only": True}



def compare_champion_challenger(
    rows: Sequence[Snap], champion: Mapping[str, Any], *,
    bootstraps: int = 400, seed: int = 27,
) -> dict[str, Any]:
    """Prospective, whole-game evaluation relative to a *saved* V2 champion.

    Only games strictly after the saved champion's training cutoff qualify.
    At each test game the challenger is refit exclusively on earlier games.
    Games with unknown or duplicate sequence numbers are never replayed.
    A predictive comparison is not an online policy-value experiment.
    """
    if champion.get("schema") != SCHEMA or champion.get("policy") != "shadow_only":
        raise ValueError("saved champion must be a V2 shadow artifact")
    cutoff = champion.get("trained_through_started_ts")
    trained = champion.get("trained_sessions")
    if not isinstance(cutoff, str) or not cutoff or not isinstance(trained, list):
        raise ValueError("saved champion lacks source-traceable chronological cutoff")
    games: dict[str, list[Snap]] = defaultdict(list)
    for row in rows:
        games[row.session].append(row)
    results: list[tuple[str, float, float, int]] = []
    for session, holdout in sorted(
        games.items(), key=lambda item: (min(r.started_ts for r in item[1]), item[0])
    ):
        start = min(r.started_ts for r in holdout)
        if session in trained or start <= cutoff:
            continue
        earlier = [r for r in rows if r.started_ts < start and r.session != session]
        if not earlier:
            continue
        challenger = fit(earlier)
        champion_loss = challenger_loss = 0.0
        seen_seq: set[int] = set()
        scored = 0
        for row in sorted(holdout, key=lambda r: (r.seq is None, r.seq or 0)):
            if row.seq is None or row.seq <= 0 or row.seq in seen_seq:
                continue
            seen_seq.add(row.seq)
            champ_concepts = concept_belief(
                champion, holdout, session=session, opponent=row.opponent,
                next_seq=row.seq)
            challeng_concepts = concept_belief(
                challenger, holdout, session=session, opponent=row.opponent,
                next_seq=row.seq)
            pred_champ = sum(
                champ_concepts[c] * offense_risk(
                    champion, row.context, row.formation, row.play, row.family, c)[0]
                for c in CONCEPTS
            )
            pred_chall = sum(
                challeng_concepts[c] * offense_risk(
                    challenger, row.context, row.formation, row.play, row.family, c)[0]
                for c in CONCEPTS
            )
            champion_loss += logloss(pred_champ, row.offense_success)
            challenger_loss += logloss(pred_chall, row.offense_success)
            scored += 1
        if scored:
            results.append((session, champion_loss, challenger_loss, scored))
    n = sum(x[3] for x in results)
    gain = sum(x[1] - x[2] for x in results) / n if n else 0.0
    interval = None
    if len(results) >= 2:
        rng = random.Random(seed)
        samples = []
        for _ in range(bootstraps):
            sample = [rng.choice(results) for __ in results]
            m = sum(x[3] for x in sample)
            samples.append(sum(x[1] - x[2] for x in sample) / m)
        samples.sort()
        interval = [
            samples[int(0.025 * (bootstraps - 1))],
            samples[int(0.975 * (bootstraps - 1))],
        ]
    passed = (
        n >= MIN_EVAL_SNAPS and len(results) >= MIN_GAMES and
        interval is not None and interval[0] > 0 and gain >= 0.02
    )
    return {
        "saved_champion_cutoff": cutoff,
        "prospective_test_games": len(results),
        "prospective_test_snaps": n,
        "challenger_minus_champion_logloss_gain": round(gain, 6),
        "game_cluster_bootstrap_95ci": interval,
        "candidate_predictive_gate": "PASS" if passed else "NO_GO",
        "automatic_promotion": False,
        "live_policy_improvement_proven": False,
        "per_game": [
            {"session": g, "champion_logloss": round(c/n_g, 5),
             "challenger_logloss": round(v/n_g, 5), "n": n_g}
            for g, c, v, n_g in results
        ],
    }
