"""OFFLINE learned candidate scoring for our own defensive calls, with catalog features added.

The existing ``defense_logistic_train`` learns P(offense succeeds | situation, call family, package). This module fits the SAME model
with the structural description of the candidate call appended (deep shell, coverage type, variant tokens, pressure concept, digit
hints, formation personnel; see ``defense_semantics.candidate_features``) and compares the two with the existing leave-one-game-out
protocol and gate.

Safety by construction
    * It reuses the trainer's private fitting helpers but never edits them.
    * Its artifact uses a different ``schema`` string, which ``defense_v1.validate_artifact`` rejects, so the live defensive caller can
      never load it. Promotion would need an explicit, separate change.
    * Nothing here picks a call. Features go into a learned score; coefficients come from data.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from cfb_coach.madden.model import defense_logistic_train as train
from cfb_coach.madden.model import defense_semantics as ds
from cfb_coach.madden.model import defense_v1
from cfb_coach.madden.model import heldout_eval as he

SCHEMA = "madden.defense.logistic.v2+knowledge_offline"
KNOWLEDGE_COLS = tuple(("dk", n) for n in ds.CANDIDATE_NAMES)
# Stricter than the live trainer's 4 games: this model adds one column per candidate-structure feature on top of the baseline columns.
MIN_GAMES_KNOWLEDGE = 8


def _cols(template: Mapping[str, Any]) -> list[tuple]:
    return train._columns(list(template["features"]), list(template["call_families"]), list(template["packages"])) + list(KNOWLEDGE_COLS)


def _design(rows: Sequence[train.SnapRow], labels: Mapping[tuple[str, str], Mapping[str, str]], template: Mapping[str, Any],
            cols: Sequence[tuple]) -> np.ndarray:
    base_cols = [c for c in cols if c[0] != "dk"]
    x_base = train._design(rows, labels, base_cols, list(template["features"]))
    x_dk = np.array([[ds.candidate_features(r.formation, r.play)[n] for _kind, n in KNOWLEDGE_COLS] for r in rows], dtype=float)
    return np.hstack([x_base, x_dk]) if len(rows) else np.zeros((0, len(cols)))


def fit_knowledge_artifact(rows: Sequence[train.SnapRow], labels: Mapping[tuple[str, str], Mapping[str, str]],
                           template: Mapping[str, Any], lam: float) -> dict[str, Any]:
    cols = _cols(template)
    x = _design(rows, labels, template, cols)
    y = np.array([1.0 if r.offense_success else 0.0 for r in rows])
    intercept, weights = train._fit_logistic(x, y, lam)
    return {"schema": SCHEMA, "policy": "offline_research_only", "target": defense_v1.TARGET, "lambda": lam, "intercept": float(intercept),
            "cols": [list(c) for c in cols], "weights": [float(w) for w in weights], "features": list(template["features"]),
            "call_families": list(template["call_families"]), "packages": list(template["packages"]),
            "catalog_digest": ds.data_digest(), "n_train": len(rows)}


def predict_knowledge(art: Mapping[str, Any], row: train.SnapRow, labels: Mapping[tuple[str, str], Mapping[str, str]]) -> float:
    template = {"features": art["features"], "call_families": art["call_families"], "packages": art["packages"]}
    cols = [tuple(c) for c in art["cols"]]
    x = _design([row], labels, template, cols)[0]
    z = art["intercept"] + float(np.dot(x, np.array(art["weights"])))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _choose_lambda(rows: Sequence[train.SnapRow], labels, template, lambdas: Sequence[float]) -> float:
    games = train._games(rows)
    if len(games) < 2:
        return max(lambdas)
    best: tuple[float, float] | None = None
    for lam in sorted(lambdas, reverse=True):
        probs: list[float] = []
        ys: list[bool] = []
        for game in games:
            art = fit_knowledge_artifact([r for r in rows if r.session_id != game], labels, template, lam)
            for row in rows:
                if row.session_id == game:
                    probs.append(predict_knowledge(art, row, labels))
                    ys.append(row.offense_success)
        score = train.log_loss(probs, ys)
        if best is None or score < best[0] - 1e-12:
            best = (score, lam)
    return best[1]


def compare_leave_one_game_out(rows: Sequence[train.SnapRow], labels: Mapping[tuple[str, str], Mapping[str, str]],
                               template: Mapping[str, Any], *, lambdas: Sequence[float] = train.LAMBDAS,
                               n_boot: int = 400, seed: int = 11) -> dict[str, Any]:
    """Each game is scored by the baseline and by the knowledge model, both trained on the OTHER admitted games with their own inner
    leave-one-game-out penalty choice. Same data, same protocol, same gate as the existing trainer."""
    pooled: dict[str, list] = {"knowledge": [], "baseline": [], "base_rate": [], "y": [], "game": []}
    per_game = []
    for game in train._games(rows):
        held = [r for r in rows if r.session_id == game]
        tr = [r for r in rows if r.admitted and r.session_id != game]
        if not tr:
            continue
        lam_b = train.choose_lambda(tr, labels, template, lambdas)
        base = defense_v1.validate_artifact(train.fit_artifact(tr, labels, template, lam_b))
        lam_k = _choose_lambda(tr, labels, template, lambdas)
        know = fit_knowledge_artifact(tr, labels, template, lam_k)
        rate = sum(r.offense_success for r in tr) / len(tr)
        ys = [r.offense_success for r in held]
        pb = [train.predict(base, r, labels) for r in held]
        pk = [predict_knowledge(know, r, labels) for r in held]
        pooled["knowledge"] += pk
        pooled["baseline"] += pb
        pooled["base_rate"] += [rate] * len(held)
        pooled["y"] += ys
        pooled["game"] += [game] * len(held)
        per_game.append({"session_id": game, "n": len(held), "lambda_baseline": lam_b, "lambda_knowledge": lam_k,
                         "baseline": train._metrics(pb, ys), "knowledge": train._metrics(pk, ys)})
    ys = pooled["y"]
    out = {"per_game": per_game, "games": len(per_game),
           "pooled": {"baseline": train._metrics(pooled["baseline"], ys), "knowledge": train._metrics(pooled["knowledge"], ys),
                      "train_base_rate": train._metrics(pooled["base_rate"], ys)}}
    preds = [{"game": g, "y": 1 if y else 0, "p": {"knowledge": k, "baseline": b}}
             for g, y, k, b in zip(pooled["game"], ys, pooled["knowledge"], pooled["baseline"])]
    out["paired_bootstrap"] = he.paired_bootstrap(preds, "knowledge", "baseline", n_boot=n_boot, seed=seed)
    out["gate"] = gate(out)
    return out


def gate(report: Mapping[str, Any]) -> dict[str, Any]:
    """The existing trainer's thresholds PLUS a game-clustered paired bootstrap whose 95% interval must exclude zero.

    The existing thresholds alone are too weak for this comparison: on pure coin-flip data they can report PASSED by chance. The
    interval requirement is what stops a noise win. Default is NOT to promote either way."""
    pooled = report["pooled"]
    n = pooled["knowledge"].get("n", 0)
    gain = pooled["baseline"]["log_loss"] - pooled["knowledge"]["log_loss"] if n else 0.0
    ci = report.get("paired_bootstrap") or {}
    checks = {"min_holdout": n >= train.MIN_HOLDOUT, "min_games": report["games"] >= MIN_GAMES_KNOWLEDGE,
              "log_loss_gain": gain >= train.MIN_IMPROVEMENT,
              "brier_not_worse": n > 0 and pooled["knowledge"]["brier"] <= pooled["baseline"]["brier"] + 1e-9,
              "interval_excludes_zero": ci.get("hi") is not None and ci["hi"] < 0.0}
    result = "INSUFFICIENT" if not (checks["min_holdout"] and checks["min_games"]) else "PASSED" if all(checks.values()) else "FAILED"
    return {"result": result, "log_loss_gain": gain, "checks": checks, "interval": {k: ci.get(k) for k in ("lo", "hi", "mean_diff")},
            "thresholds": {"min_holdout": train.MIN_HOLDOUT, "min_games": MIN_GAMES_KNOWLEDGE, "min_log_loss_gain": train.MIN_IMPROVEMENT,
                           "interval": "95% game-clustered paired bootstrap upper bound < 0"},
            "promotion": "none: offline research artifact only; the live defensive caller rejects this schema"}
