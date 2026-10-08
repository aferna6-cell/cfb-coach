"""L2-logistic regression with a separate penalty per feature group.

The intercept is not penalized. Situation, entity, and interaction columns
take ``lambda_situation``, ``lambda_entity``, and ``lambda_interaction``.
"""

from __future__ import annotations

import math

import numpy as np

from cfb_coach.vod_success.features import SITUATION_GROUPS, feature_map, penalty_for
from cfb_coach.vod_success.features import Snap

_MAX_ITER = 12
_CLIP = 15.0


class LogisticModel:
    def __init__(
        self,
        intercept: float,
        coef: dict[str, dict[str, float]],
        hp: tuple[float, float, float],
    ) -> None:
        self.intercept = float(intercept)
        self.coef = coef
        self.hp = hp

    def predict_row(self, feats: dict[str, str]) -> float:
        logit = self.intercept
        for group, level in feats.items():
            logit += self.coef.get(group, {}).get(level, 0.0)
        return _sigmoid(logit)

    def as_params(self, groups: tuple[str, ...]) -> dict:
        ordered: dict[str, dict[str, float]] = {}
        for group in sorted(groups):
            levels = self.coef.get(group, {})
            ordered[group] = {level: _round6(weight) for level, weight in sorted(levels.items())}
        situation, entity, interaction = self.hp
        return {
            "intercept": _round6(self.intercept),
            "coef": ordered,
            "hyperparams": {
                "lambda_situation": float(situation),
                "lambda_entity": float(entity),
                "lambda_interaction": float(interaction),
            },
        }


def fit_model(
    snaps: list[Snap],
    hp: tuple[float, float, float],
    *,
    situation_only: bool = False,
) -> LogisticModel:
    rows = [feature_map(snap, situation_only=situation_only) for snap in snaps]
    if situation_only:
        rows = [{k: v for k, v in row.items() if k in SITUATION_GROUPS} for row in rows]
    y = np.asarray([1.0 if snap.success else 0.0 for snap in snaps], dtype=float)
    intercept, coef = _fit(rows, y, hp)
    return LogisticModel(intercept, coef, hp)


def _fit(
    rows: list[dict[str, str]],
    y: np.ndarray,
    hp: tuple[float, float, float],
) -> tuple[float, dict[str, dict[str, float]]]:
    n = int(y.shape[0])
    if n == 0:
        return 0.0, {}
    columns: list[tuple[str, str, float]] = []
    index: dict[tuple[str, str], int] = {}
    for row in rows:
        for group, level in row.items():
            key = (group, level)
            if key not in index:
                index[key] = len(columns)
                columns.append((group, level, penalty_for(group, hp)))
    p = len(columns) + 1
    x = np.zeros((n, p), dtype=float)
    x[:, 0] = 1.0
    for i, row in enumerate(rows):
        for group, level in row.items():
            x[i, index[(group, level)] + 1] = 1.0
    penalty = np.zeros(p, dtype=float)
    for j, (_group, _level, lam) in enumerate(columns):
        penalty[j + 1] = lam
    weights = np.zeros(p, dtype=float)
    for _ in range(_MAX_ITER):
        eta = np.clip(x @ weights, -_CLIP, _CLIP)
        mu = 1.0 / (1.0 + np.exp(-eta))
        var = np.clip(mu * (1.0 - mu), 1e-4, None)
        z = eta + (y - mu) / var
        xtw = x.T * var
        a = xtw @ x
        a.flat[:: p + 1] += penalty
        b = xtw @ z
        try:
            updated = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            updated = np.linalg.lstsq(a, b, rcond=None)[0]
        if float(np.max(np.abs(updated - weights))) < 1e-6:
            weights = updated
            break
        weights = updated
    coef: dict[str, dict[str, float]] = {}
    for (group, level, _lam), weight in zip(columns, weights[1:]):
        coef.setdefault(group, {})[level] = float(weight)
    return float(weights[0]), coef


def log_loss(probs: list[float], labels: list[bool]) -> float:
    if not probs:
        return 0.0
    total = 0.0
    for prob, label in zip(probs, labels):
        p = min(1.0 - 1e-6, max(1e-6, prob))
        y = 1.0 if label else 0.0
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(probs)


def brier(probs: list[float], labels: list[bool]) -> float:
    if not probs:
        return 0.0
    total = 0.0
    for prob, label in zip(probs, labels):
        y = 1.0 if label else 0.0
        total += (prob - y) ** 2
    return total / len(probs)


def reliability(probs: list[float], labels: list[bool], *, n_bins: int = 10) -> tuple[float, list[dict]]:
    """ECE and the non-empty probability bins (width ``1/n_bins``)."""
    if not probs:
        return 0.0, []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for prob, label in zip(probs, labels):
        idx = min(n_bins - 1, max(0, int(prob * n_bins)))
        bins[idx].append((prob, label))
    rows = []
    ece = 0.0
    n = len(probs)
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        obs = sum(1.0 for _, lab in bucket if lab) / len(bucket)
        ece += (len(bucket) / n) * abs(obs - mean_pred)
        lo = i / n_bins
        hi = (i + 1) / n_bins
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(bucket),
                "mean_pred": round(mean_pred, 4),
                "obs_rate": round(obs, 4),
            }
        )
    return ece, rows


def _sigmoid(logit: float) -> float:
    logit = max(-_CLIP, min(_CLIP, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def _round6(value: float) -> float:
    return round(float(value), 6)
