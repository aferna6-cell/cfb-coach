"""Train and evaluate the ``madden.defense.logistic.v2`` artifact from logged defensive snaps.

Each row is one human-game defensive snap with a parseable result. The label is
``defense_v1.offense_won``. The executed call is used only when verified;
otherwise the row preserves the displayed recommendation for diagnostics.
Only verified executions are admitted for play-specific training by default:


``executed_verified``
    The executed formation and play were verified.
``user_confirmed_logged_result``
    The user attested this session's logged results and opponent plays. The
    displayed call stands in for the executed call; execution itself was not
    verified.
``logged_unverified``
    Logged result, execution unknown. This is the ``logged_unverified_v1``
    policy the live artifact was built under.

``unattested_after`` keeps sessions that started at or after that timestamp out
of legacy analysis unless they are attested or verified. Strict training
requires verified execution regardless of an attested game score. Nothing
here writes the database.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cfb_coach.madden.model import defense_v1
from cfb_coach.madden.situation import Situation, parse_madden_situation
from cfb_coach.opponents import is_cpu_opponent

PROVENANCE_VERIFIED = "executed_verified"
PROVENANCE_ATTESTED = "user_confirmed_logged_result"
PROVENANCE_LOGGED = "logged_unverified"
LAMBDAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
# Same thresholds as ``evaluate.evaluate_model``'s held-out gate.
MIN_HOLDOUT = 30
MIN_GAMES = 4
MIN_IMPROVEMENT = 0.02


@dataclass(frozen=True)
class SnapRow:
    snap_id: int
    session_id: str
    opponent: str
    started_ts: str
    situation: Situation
    formation: str
    play: str
    offense_success: bool
    provenance: str
    admitted: bool
    excluded_reason: str = ""


def collect_rows(
    conn: sqlite3.Connection,
    *,
    attested: Mapping[str, str] | None = None,
    unattested_after: str | None = None,
    verified_only: bool = True,
) -> list[SnapRow]:
    """Human defensive snaps, with only verified executions admitted by default.

    Set ``verified_only=False`` solely to reproduce older exploratory analyses;
    never promote an artifact learned from displayed but unverified calls.
    """
    attested = dict(attested or {})
    cur = conn.execute(
        "SELECT s.id, s.session_id, g.opponent_id, g.started_ts, s.situation_raw, s.formation, s.play, "
        "s.result, s.down, s.distance, s.executed_verification, s.executed_formation, s.executed_play "
        "FROM snaps s JOIN game_sessions g ON g.session_id = s.session_id "
        "WHERE s.side = 'defense' ORDER BY g.started_ts, COALESCE(s.snap_seq, s.id), s.id"
    )
    rows: list[SnapRow] = []
    for (snap_id, session_id, opponent, started, raw, formation, play, result, down, distance,
         verification, ex_formation, ex_play) in cur:
        if is_cpu_opponent(str(opponent or "")):
            continue
        won = defense_v1.offense_won(result, down, distance)
        if won is None:
            continue
        if str(verification or "") == "verified" and ex_formation and ex_play:
            formation, play, provenance = ex_formation, ex_play, PROVENANCE_VERIFIED
        elif session_id in attested:
            provenance = PROVENANCE_ATTESTED
        else:
            provenance = PROVENANCE_LOGGED
        if not formation or not play:
            continue
        reason = ""
        if verified_only and provenance != PROVENANCE_VERIFIED:
            reason = "execution_unverified"
        elif (provenance == PROVENANCE_LOGGED and unattested_after
                and str(started or "") >= unattested_after):
            reason = "unattested_after_cutoff"
        rows.append(SnapRow(
            snap_id=int(snap_id),
            session_id=str(session_id),
            opponent=str(opponent or ""),
            started_ts=str(started or ""),
            situation=parse_madden_situation(str(raw or ""), default_side="defense"),
            formation=str(formation),
            play=str(play),
            offense_success=bool(won),
            provenance=provenance,
            admitted=not reason,
            excluded_reason=reason,
        ))
    return rows


def label_calls(rows: Iterable[SnapRow], label_source: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    """Family and package per call: the source artifact's label, else its coverage mapping."""
    book: dict[str, list[str]] = {}
    for row in rows:
        plays = book.setdefault(row.formation, [])
        if row.play not in plays:
            plays.append(row.play)
    return {
        (c["formation"], c["play"]): {"family": c["family"], "package": c["package"]}
        for c in defense_v1.installed_candidates(label_source, book)
    }


def _columns(features: Sequence[str], families: Sequence[str], packages: Sequence[str]) -> list[tuple]:
    cols: list[tuple] = [("base", name) for name in features]
    cols += [("fam", f) for f in families] + [("pkg", p) for p in packages]
    cols += [("fx", f, i) for f in families for i in defense_v1.INTERACTION_FEATURES]
    cols += [("px", p, i) for p in packages for i in defense_v1.INTERACTION_FEATURES]
    return cols


def _design(rows: Sequence[SnapRow], labels: Mapping[tuple[str, str], Mapping[str, str]],
            cols: Sequence[tuple], features: Sequence[str]) -> np.ndarray:
    index = {col: j for j, col in enumerate(cols)}
    out = np.zeros((len(rows), len(cols)))
    for r, row in enumerate(rows):
        label = labels[(row.formation, row.play)]
        for name, value in defense_v1.situation_feature_vector(row.situation, features).items():
            out[r, index[("base", name)]] = value
        out[r, index[("fam", label["family"])]] = 1.0
        out[r, index[("pkg", label["package"])]] = 1.0
        for name, value in defense_v1.interaction_values(row.situation).items():
            out[r, index[("fx", label["family"], name)]] = value
            out[r, index[("px", label["package"], name)]] = value
    return out


def _fit_logistic(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """L2 logistic regression on standardized columns; returns raw-scale intercept and weights."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    live = sd > 1e-12
    z = np.zeros_like(x)
    z[:, live] = (x[:, live] - mu[live]) / sd[live]
    design = np.hstack([np.ones((len(x), 1)), z])
    penalty = np.full(design.shape[1], lam)
    penalty[0] = 0.0
    w = np.zeros(design.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ w, -30, 30)))
        grad = design.T @ (p - y) + penalty * w
        hess = (design * (p * (1 - p))[:, None]).T @ design + np.diag(penalty) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    raw = np.zeros(x.shape[1])
    raw[live] = w[1:][live] / sd[live]
    intercept = float(w[0] - np.sum(raw[live] * mu[live]))
    return intercept, raw


def fit_artifact(
    rows: Sequence[SnapRow],
    labels: Mapping[tuple[str, str], Mapping[str, str]],
    template: Mapping[str, Any],
    lam: float,
) -> dict[str, Any]:
    """A v2 artifact fit on ``rows``. Vocabulary, sampling knobs, and candidates come from ``template``."""
    features = list(template["features"])
    families = list(template["call_families"])
    packages = list(template["packages"])
    cols = _columns(features, families, packages)
    x = _design(rows, labels, cols, features)
    y = np.array([1.0 if row.offense_success else 0.0 for row in rows])
    intercept, weights = _fit_logistic(x, y, lam)
    coef = {col: float(w) for col, w in zip(cols, weights)}
    candidates = [dict(c) for c in template["candidates"]]
    known = {(c["formation"], c["play"]) for c in candidates}
    for row in rows:
        key = (row.formation, row.play)
        if key not in known:
            known.add(key)
            candidates.append({"formation": row.formation, "play": row.play, **labels[key]})
    return {
        "schema": defense_v1.SCHEMA_V2,
        "policy": defense_v1.POLICY,
        "target": defense_v1.TARGET,
        "intercept": intercept,
        "features": features,
        "coefficients": {name: coef[("base", name)] for name in features},
        "call_families": families,
        "packages": packages,
        "family_coefficients": {f: coef[("fam", f)] for f in families},
        "package_coefficients": {p: coef[("pkg", p)] for p in packages},
        "family_situation_coefficients": {
            f: {i: coef[("fx", f, i)] for i in defense_v1.INTERACTION_FEATURES} for f in families
        },
        "package_situation_coefficients": {
            p: {i: coef[("px", p, i)] for i in defense_v1.INTERACTION_FEATURES} for p in packages
        },
        "top_k": template.get("top_k", defense_v1.DEFAULT_TOP_K),
        "temperature": template.get("temperature", defense_v1.DEFAULT_TEMPERATURE),
        "max_repeat": template.get("max_repeat", defense_v1.DEFAULT_MAX_REPEAT),
        "family_spread": template.get("family_spread", defense_v1.DEFAULT_FAMILY_SPREAD),
        "candidates": candidates,
    }


def predict(artifact: Mapping[str, Any], row: SnapRow,
            labels: Mapping[tuple[str, str], Mapping[str, str]]) -> float:
    label = labels[(row.formation, row.play)]
    return defense_v1.offense_success_probability(
        artifact, row.situation, {"formation": row.formation, "play": row.play, **label},
    )


def log_loss(probs: Sequence[float], labels: Sequence[bool]) -> float:
    total = 0.0
    for prob, label in zip(probs, labels):
        p = min(1.0 - 1e-6, max(1e-6, float(prob)))
        total += -math.log(p) if label else -math.log(1.0 - p)
    return total / len(probs) if probs else 0.0


def brier(probs: Sequence[float], labels: Sequence[bool]) -> float:
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(probs, labels)) / len(probs) if probs else 0.0


def auc(probs: Sequence[float], labels: Sequence[bool]) -> float | None:
    pos = [p for p, y in zip(probs, labels) if y]
    neg = [p for p, y in zip(probs, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def _games(rows: Iterable[SnapRow]) -> list[str]:
    return list(dict.fromkeys(row.session_id for row in rows))


def choose_lambda(rows: Sequence[SnapRow], labels: Mapping[tuple[str, str], Mapping[str, str]],
                  template: Mapping[str, Any], lambdas: Sequence[float] = LAMBDAS) -> float:
    """Leave-one-game-out inside ``rows`` only; ties go to the larger penalty."""
    games = _games(rows)
    if len(games) < 2:
        return max(lambdas)
    best: tuple[float, float] | None = None
    for lam in sorted(lambdas, reverse=True):
        probs: list[float] = []
        ys: list[bool] = []
        for game in games:
            train = [r for r in rows if r.session_id != game]
            art = defense_v1.validate_artifact(fit_artifact(train, labels, template, lam))
            for row in rows:
                if row.session_id == game:
                    probs.append(predict(art, row, labels))
                    ys.append(row.offense_success)
        score = log_loss(probs, ys)
        if best is None or score < best[0] - 1e-12:
            best = (score, lam)
    return best[1]


def _metrics(probs: Sequence[float], ys: Sequence[bool]) -> dict[str, Any]:
    return {"n": len(ys), "log_loss": log_loss(probs, ys), "brier": brier(probs, ys), "auc": auc(probs, ys)}


def leave_one_game_out(
    rows: Sequence[SnapRow],
    labels: Mapping[tuple[str, str], Mapping[str, str]],
    current: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    current_in_sample: Iterable[str] = (),
    lambdas: Sequence[float] = LAMBDAS,
) -> dict[str, Any]:
    """Score every game with a candidate trained on the other admitted games, and with ``current``.

    ``current`` is fixed, so games in ``current_in_sample`` are scored in-sample
    for it; that bias favors the current model.
    """
    in_sample = set(current_in_sample)
    per_game: list[dict[str, Any]] = []
    pooled: dict[str, list] = {"cand": [], "cur": [], "base": [], "y": [], "game": []}
    for game in _games(rows):
        held = [r for r in rows if r.session_id == game]
        train = [r for r in rows if r.admitted and r.session_id != game]
        lam = choose_lambda(train, labels, template, lambdas)
        cand = defense_v1.validate_artifact(fit_artifact(train, labels, template, lam))
        rate = sum(r.offense_success for r in train) / len(train)
        ys = [r.offense_success for r in held]
        cand_p = [predict(cand, r, labels) for r in held]
        cur_p = [predict(current, r, labels) for r in held]
        pooled["cand"] += cand_p
        pooled["cur"] += cur_p
        pooled["base"] += [rate] * len(held)
        pooled["y"] += ys
        pooled["game"] += [game] * len(held)
        per_game.append({
            "session_id": game, "opponent": held[0].opponent, "started_ts": held[0].started_ts,
            "n": len(held), "offense_success_rate": sum(ys) / len(ys), "lambda": lam,
            "current_in_sample": game in in_sample,
            "candidate": _metrics(cand_p, ys), "current": _metrics(cur_p, ys),
        })

    def subset(keep) -> dict[str, Any]:
        idx = [i for i, g in enumerate(pooled["game"]) if keep(g)]
        ys = [pooled["y"][i] for i in idx]
        return {
            "games": len({pooled["game"][i] for i in idx}),
            "candidate": _metrics([pooled["cand"][i] for i in idx], ys),
            "current": _metrics([pooled["cur"][i] for i in idx], ys),
            "train_base_rate": _metrics([pooled["base"][i] for i in idx], ys),
        }

    return {
        "per_game": per_game,
        "all_games": subset(lambda g: True),
        "current_out_of_sample": subset(lambda g: g not in in_sample),
    }


def gate(report: Mapping[str, Any]) -> dict[str, Any]:
    """``evaluate``'s held-out thresholds, candidate against current.

    PASSED needs at least ``MIN_HOLDOUT`` snaps from ``MIN_GAMES`` games, a
    log-loss gain of ``MIN_IMPROVEMENT`` over every game, Brier no worse, and no
    log-loss loss on the games the current model never saw.
    """
    pooled = report["all_games"]
    oos = report["current_out_of_sample"]
    n = pooled["candidate"]["n"]
    gain = pooled["current"]["log_loss"] - pooled["candidate"]["log_loss"]
    checks = {
        "min_holdout": n >= MIN_HOLDOUT,
        "min_games": pooled["games"] >= MIN_GAMES,
        "log_loss_gain": gain >= MIN_IMPROVEMENT,
        "brier_not_worse": pooled["candidate"]["brier"] <= pooled["current"]["brier"] + 1e-9,
        "unseen_games_not_worse": (
            oos["games"] == 0
            or oos["candidate"]["log_loss"] <= oos["current"]["log_loss"] + 1e-9
        ),
    }
    if not (checks["min_holdout"] and checks["min_games"]):
        result = "INSUFFICIENT"
    elif all(checks.values()):
        result = "PASSED"
    else:
        result = "FAILED"
    return {"result": result, "log_loss_gain": gain, "checks": checks,
            "thresholds": {"min_holdout": MIN_HOLDOUT, "min_games": MIN_GAMES,
                           "min_log_loss_gain": MIN_IMPROVEMENT}}


def training_summary(rows: Sequence[SnapRow], attested: Mapping[str, str],
                     unattested_after: str | None, lam: float) -> dict[str, Any]:
    sessions: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = sessions.setdefault(row.session_id, {
            "opponent": row.opponent, "started_ts": row.started_ts,
            "admitted": 0, "excluded": 0, "provenance": {},
        })
        if row.admitted:
            entry["admitted"] += 1
            entry["provenance"][row.provenance] = entry["provenance"].get(row.provenance, 0) + 1
        else:
            entry["excluded"] += 1
            entry["excluded_reason"] = row.excluded_reason
        if row.session_id in attested:
            entry["attestation"] = attested[row.session_id]
    admitted = [r for r in rows if r.admitted]
    provenance: dict[str, int] = {}
    for row in admitted:
        provenance[row.provenance] = provenance.get(row.provenance, 0) + 1
    return {
        "admitted_snaps": len(admitted),
        "admitted_games": len(_games(admitted)),
        "provenance": provenance,
        "label": "defense_v1.offense_won(result, down, distance)",
        "unattested_after": unattested_after,
        "l2_lambda": lam,
        "sessions": sessions,
    }
