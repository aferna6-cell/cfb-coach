"""Small, dependency-free learners (the repo's runtime is stdlib only).

* ``OffsetLogistic``  L2-regularized logistic regression on top of a fixed offset logit
  (Newton steps, Laplace covariance for uncertainty). Same family as the existing
  ``offense_structural_model`` residual, so the existing coordinator stays the base.
* ``OffsetGBT``       gradient-boosted shallow regression trees for log-loss, boosted FROM
  the same offset (so trees learn only what the existing model misses), with shrinkage,
  row subsampling and min-leaf regularization; ``GBTEnsemble`` adds game-bootstrap members
  whose spread is the uncertainty.
* ``ridge``           linear regression for expected yards.
* ``neural_challenger_justified``  explicit gate: a neural challenger is only trained when
  the verified data could support one.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

NN_MIN_VERIFIED_SNAPS = 5000
NN_MIN_GAMES = 60


def neural_challenger_justified(n_rows: int, n_games: int) -> tuple[bool, str]:
    if n_rows >= NN_MIN_VERIFIED_SNAPS and n_games >= NN_MIN_GAMES:
        return True, f"{n_rows} verified snaps from {n_games} games"
    return False, (f"not trained: {n_rows} verified snaps from {n_games} games is below the gate "
                   f"({NN_MIN_VERIFIED_SNAPS} snaps and {NN_MIN_GAMES} games); a neural network would memorize this sample")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _invert(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            raise ValueError("singular matrix")
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        m[c] = [x / d for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0.0:
                f = m[r][c]
                rc = m[c]
                m[r] = [x - f * y for x, y in zip(m[r], rc)]
    return [row[n:] for row in m]


@dataclass
class Standardizer:
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, X: Sequence[Sequence[float]]) -> "Standardizer":
        n = len(X)
        cols = list(zip(*X))
        mean = [sum(c) / n for c in cols]
        scale = [max(0.05, math.sqrt(max(0.0, sum(v * v for v in c) / n - m * m))) for c, m in zip(cols, mean)]
        return cls(mean, scale)

    def __call__(self, x: Sequence[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(x, self.mean, self.scale)]


@dataclass
class OffsetLogistic:
    l2: float = 10.0
    max_iter: int = 25
    weights: list[float] = field(default_factory=list)      # [bias] + d
    cov: list[list[float]] = field(default_factory=list)
    std: Standardizer = field(default_factory=Standardizer)
    kind: str = "offset_logistic"

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int], offset: Sequence[float]) -> "OffsetLogistic":
        self.std = Standardizer.fit(X)
        Z = [[1.0] + self.std(x) for x in X]
        d = len(Z[0])
        w = [0.0] * d
        lam = [1e-4] + [self.l2] * (d - 1)
        H_inv = None
        for _ in range(self.max_iter):
            g = [lam[i] * w[i] for i in range(d)]
            H = [[0.0] * d for _ in range(d)]
            for z, yi, oi in zip(Z, y, offset):
                p = sigmoid(oi + sum(wi * zi for wi, zi in zip(w, z)))
                wt = p * (1 - p)
                r = p - yi
                for i in range(d):
                    zi = z[i]
                    if zi == 0.0:
                        continue
                    g[i] += r * zi
                    Hi = H[i]
                    for j in range(d):
                        Hi[j] += wt * zi * z[j]
            for i in range(d):
                H[i][i] += lam[i]
            H_inv = _invert(H)
            step = [sum(H_inv[i][j] * g[j] for j in range(d)) for i in range(d)]
            w = [wi - si for wi, si in zip(w, step)]
            if max(abs(s) for s in step) < 1e-6:
                break
        self.weights, self.cov = w, H_inv or []
        return self

    def margin(self, x: Sequence[float]) -> tuple[float, float]:
        z = [1.0] + self.std(x)
        m = sum(wi * zi for wi, zi in zip(self.weights, z))
        var = sum(z[i] * self.cov[i][j] * z[j] for i in range(len(z)) for j in range(len(z))) if self.cov else 0.0
        return m, math.sqrt(max(0.0, var))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "l2": self.l2, "weights": [round(v, 8) for v in self.weights],
                "cov": [[round(v, 8) for v in r] for r in self.cov], "mean": self.std.mean, "scale": self.std.scale}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OffsetLogistic":
        return cls(l2=d["l2"], weights=list(d["weights"]), cov=[list(r) for r in d["cov"]],
                   std=Standardizer(list(d["mean"]), list(d["scale"])))


# ------------------------------------------------------------------ gradient-boosted trees

def _best_split(X, idx, g, h, min_leaf, lam, features):
    G, Hs = sum(g[i] for i in idx), sum(h[i] for i in idx)
    base = G * G / (Hs + lam)
    best = (0.0, None, None)
    for f in features:
        vals = sorted(idx, key=lambda i: X[i][f])
        gl = hl = 0.0
        for k in range(len(vals) - 1):
            i = vals[k]
            gl += g[i]
            hl += h[i]
            if k + 1 < min_leaf or len(vals) - k - 1 < min_leaf:
                continue
            if X[vals[k + 1]][f] == X[i][f]:
                continue
            gr, hr = G - gl, Hs - hl
            gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - base
            if gain > best[0]:
                best = (gain, f, (X[i][f] + X[vals[k + 1]][f]) / 2.0)
    return best


def _build(X, idx, g, h, depth, max_depth, min_leaf, lam, features, min_gain):
    G, Hs = sum(g[i] for i in idx), sum(h[i] for i in idx)
    leaf = {"v": -G / (Hs + lam)}
    if depth >= max_depth or len(idx) < 2 * min_leaf:
        return leaf
    gain, f, thr = _best_split(X, idx, g, h, min_leaf, lam, features)
    if f is None or gain < min_gain:
        return leaf
    left = [i for i in idx if X[i][f] <= thr]
    right = [i for i in idx if X[i][f] > thr]
    return {"f": f, "t": thr, "l": _build(X, left, g, h, depth + 1, max_depth, min_leaf, lam, features, min_gain),
            "r": _build(X, right, g, h, depth + 1, max_depth, min_leaf, lam, features, min_gain)}


def _tree_value(t, x):
    while "f" in t:
        t = t["l"] if x[t["f"]] <= t["t"] else t["r"]
    return t["v"]


@dataclass
class OffsetGBT:
    n_trees: int = 120
    learning_rate: float = 0.05
    max_depth: int = 2
    min_leaf: int = 25
    l2: float = 5.0
    subsample: float = 0.8
    colsample: float = 0.7
    min_gain: float = 0.05
    seed: int = 7
    trees: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "offset_gbt"

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int], offset: Sequence[float]) -> "OffsetGBT":
        rng = random.Random(self.seed)
        n, d = len(X), len(X[0])
        F = list(offset)
        self.trees = []
        for _ in range(self.n_trees):
            p = [sigmoid(v) for v in F]
            g = [pi - yi for pi, yi in zip(p, y)]
            h = [max(1e-6, pi * (1 - pi)) for pi in p]
            idx = [i for i in range(n) if rng.random() < self.subsample]
            feats = [f for f in range(d) if rng.random() < self.colsample] or list(range(d))
            t = _build(X, idx, g, h, 0, self.max_depth, self.min_leaf, self.l2, feats, self.min_gain)
            if "f" not in t and abs(t["v"]) < 1e-9:
                continue
            self.trees.append(t)
            for i in range(n):
                F[i] += self.learning_rate * _tree_value(t, X[i])
        return self

    def margin(self, x: Sequence[float]) -> float:
        return self.learning_rate * sum(_tree_value(t, x) for t in self.trees)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "n_trees": self.n_trees, "learning_rate": self.learning_rate,
                "max_depth": self.max_depth, "min_leaf": self.min_leaf, "l2": self.l2, "subsample": self.subsample,
                "colsample": self.colsample, "seed": self.seed, "trees": self.trees}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OffsetGBT":
        m = cls(**{k: d[k] for k in ("n_trees", "learning_rate", "max_depth", "min_leaf", "l2", "subsample",
                                     "colsample", "seed")})
        m.trees = d["trees"]
        return m


@dataclass
class GBTEnsemble:
    members: list[OffsetGBT] = field(default_factory=list)
    kind: str = "offset_gbt_ensemble"

    @classmethod
    def fit(cls, X, y, offset, groups: Sequence[str], *, n_members: int = 5, seed: int = 11, **params) -> "GBTEnsemble":
        rng = random.Random(seed)
        uniq = sorted(set(groups))
        members = [OffsetGBT(seed=seed, **params).fit(X, y, offset)]
        for m in range(1, n_members):
            pick = [rng.choice(uniq) for _ in uniq]          # game-level bootstrap
            idx = [i for g in pick for i, gg in enumerate(groups) if gg == g]
            members.append(OffsetGBT(seed=seed + m, **params).fit([X[i] for i in idx], [y[i] for i in idx],
                                                                  [offset[i] for i in idx]))
        return cls(members)

    def margin(self, x) -> tuple[float, float]:
        ms = [m.margin(x) for m in self.members]
        mean = ms[0]                                   # the full-data member is the point estimate
        sd = math.sqrt(sum((v - sum(ms) / len(ms)) ** 2 for v in ms) / max(1, len(ms) - 1)) if len(ms) > 1 else 0.0
        return mean, sd

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "members": [m.to_dict() for m in self.members]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GBTEnsemble":
        return cls([OffsetGBT.from_dict(m) for m in d["members"]])


def ridge(X: Sequence[Sequence[float]], y: Sequence[float], l2: float = 20.0) -> dict[str, Any]:
    std = Standardizer.fit(X)
    Z = [[1.0] + std(x) for x in X]
    d = len(Z[0])
    A = [[sum(z[i] * z[j] for z in Z) + (l2 if (i == j and i > 0) else (1e-6 if i == j else 0.0)) for j in range(d)]
         for i in range(d)]
    b = [sum(z[i] * yi for z, yi in zip(Z, y)) for i in range(d)]
    Ai = _invert(A)
    w = [sum(Ai[i][j] * b[j] for j in range(d)) for i in range(d)]
    resid = [yi - sum(wi * zi for wi, zi in zip(w, z)) for z, yi in zip(Z, y)]
    sigma = math.sqrt(sum(r * r for r in resid) / max(1, len(resid) - 1))
    return {"kind": "ridge", "weights": w, "mean": std.mean, "scale": std.scale, "sigma": sigma, "l2": l2}


def ridge_predict(m: dict[str, Any], x: Sequence[float]) -> float:
    z = [1.0] + [(v - mu) / s for v, mu, s in zip(x, m["mean"], m["scale"])]
    return sum(wi * zi for wi, zi in zip(m["weights"], z))
