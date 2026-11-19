"""Opponent offensive-concept beliefs: what will the offense run on THIS snap?

Trained from observed opponent plays (they do not need our defensive execution to be verified).
Every feature for snap t uses only: earlier games (opponent history) and strictly earlier, closed
snaps of the same game. The current snap's result / post-play concept is never an input.

Models (all forward-only evaluated with whole-game holdouts):
  * ``v2_dirichlet``  Defensive V2's concept_belief (pooled prior x12 + same-game counts), on these classes
  * ``hier_dirichlet`` pooled -> opponent history -> recency-weighted same game
  * ``contextual``    L2 multinomial logistic on down/distance/field, opponent history, same-game mix,
                       previous concept (the supervised challenger)
  * ``v2_plus_residual`` the same features as an L2 residual on top of the V2 belief (offset logits)
The live belief is the challenger only when it beats V2 on later held-out games; otherwise V2.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from cfb_coach.madden.ravens_defense.concepts import CLASSES, PASS_CLASSES

K = len(CLASSES)
SAME_GAME_DECAY = 0.85
V2_PRIOR_STRENGTH = 12.0


def context_tokens(down: int | None, distance: int | None, ytg: int | None) -> list[str]:
    t = []
    t.append({1: "d1", 2: "d2"}.get(down or 0, "d34" if down else "dunk"))
    if distance is None:
        t.append("dist_unk")
    else:
        t.append("short" if distance <= 3 else "medium" if distance <= 6 else "long")
    if ytg is None:
        t.append("field_unk")
    else:
        t.append("rz10" if ytg <= 10 else "rz20" if ytg <= 20 else "mid" if ytg <= 50 else "own")
    return t


CTX_TOKENS = ("d1", "d2", "d34", "short", "medium", "long", "rz10", "rz20", "mid", "own")


def _counts(rows: Iterable[Any]) -> Counter:
    return Counter(r.concept for r in rows if r.concept in CLASSES and r.closed)


class History:
    """Opponent profiles from earlier games only (built once per training cutoff / live session)."""

    def __init__(self, rows: Sequence[Any]):
        self.pooled = _counts(rows)
        self.by_opp: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            if r.concept in CLASSES and r.closed:
                self.by_opp[r.opponent][r.concept] += 1

    def to_json(self) -> dict[str, Any]:
        return {"pooled": dict(self.pooled), "by_opponent": {k: dict(v) for k, v in self.by_opp.items()}}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "History":
        h = cls([])
        h.pooled = Counter(d.get("pooled") or {})
        h.by_opp = defaultdict(Counter, {k: Counter(v) for k, v in (d.get("by_opponent") or {}).items()})
        return h

    def extend(self, rows: Sequence[Any]) -> "History":
        h = History.from_json(self.to_json())
        for r in rows:
            if r.concept in CLASSES and r.closed:
                h.pooled[r.concept] += 1
                h.by_opp[r.opponent][r.concept] += 1
        return h


def _earlier(same_game: Sequence[Any], order: int) -> list[Any]:
    """Strictly earlier, CLOSED snaps of this game; the current snap and later ones are excluded."""
    return sorted((r for r in same_game if r.order < order and r.closed), key=lambda r: r.order)


def v2_dirichlet(hist: History, same_game: Sequence[Any], order: int) -> list[float]:
    counts = {c: 1.0 + hist.pooled.get(c, 0) for c in CLASSES}
    tot = sum(counts.values())
    counts = {c: n / tot * V2_PRIOR_STRENGTH for c, n in counts.items()}
    for r in _earlier(same_game, order):
        if r.concept in CLASSES:
            counts[r.concept] += 1.0
    tot = sum(counts.values())
    return [counts[c] / tot for c in CLASSES]


def hier_dirichlet(hist: History, same_game: Sequence[Any], order: int, opponent: str) -> list[float]:
    pooled_tot = sum(hist.pooled.values())
    pooled = [(hist.pooled.get(c, 0) + 1.0) / (pooled_tot + K) for c in CLASSES]
    opp = hist.by_opp.get(opponent) or Counter()
    on = sum(opp.values())
    a = 10.0
    opp_p = [(opp.get(c, 0) + a * pooled[i]) / (on + a) for i, c in enumerate(CLASSES)]
    b = 8.0
    w = Counter()
    for j, r in enumerate(reversed(_earlier(same_game, order))):
        if r.concept in CLASSES:
            w[r.concept] += SAME_GAME_DECAY ** j
    wn = sum(w.values())
    return [(w.get(c, 0.0) + b * opp_p[i]) / (wn + b) for i, c in enumerate(CLASSES)]


def features(hist: History, same_game: Sequence[Any], snap: Any) -> list[float]:
    """Pre-snap feature vector. ``snap`` contributes ONLY down/distance/field/opponent/order."""
    x = [1.0]
    toks = set(context_tokens(snap.down, snap.distance, snap.yards_to_goal))
    x += [1.0 if t in toks else 0.0 for t in CTX_TOKENS]
    pooled = hier_dirichlet(hist, (), 1, "")  # pooled + opponent prior only, no same-game
    opp = hier_dirichlet(hist, (), 1, snap.opponent)
    x += [math.log(opp[i] / pooled[i]) for i in range(K)]
    earlier = _earlier(same_game, snap.order)
    w = Counter()
    for j, r in enumerate(reversed(earlier)):
        if r.concept in CLASSES:
            w[r.concept] += SAME_GAME_DECAY ** j
    wn = sum(w.values())
    conf = wn / (wn + 3.0)
    x += [conf * (w.get(c, 0.0) / wn - pooled[i]) if wn else 0.0 for i, c in enumerate(CLASSES)]
    prev = next((r.concept for r in reversed(earlier) if r.concept in CLASSES), None)
    x += [1.0 if prev == c else 0.0 for c in CLASSES]
    return x


FEATURE_NAMES = (["bias"] + [f"ctx:{t}" for t in CTX_TOKENS] + [f"opp_hist_logratio:{c}" for c in CLASSES]
                 + [f"same_game_mix:{c}" for c in CLASSES] + [f"prev_concept:{c}" for c in CLASSES])


def _softmax(z: Sequence[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def fit_contextual(X: Sequence[Sequence[float]], y: Sequence[int], *, l2: float = 3.0,
                   iters: int = 400, offsets: Sequence[Sequence[float]] | None = None) -> list[list[float]]:
    """L2 multinomial logistic (MAP). numpy is required for training (dev extra); prediction is pure Python."""
    import numpy as np
    Xa = np.asarray(X, dtype=float)
    n, d = Xa.shape
    Y = np.zeros((n, K))
    Y[np.arange(n), np.asarray(y)] = 1.0
    W = np.zeros((d, K))
    reg = np.full((d, 1), l2)
    reg[0, 0] = 0.01  # weak penalty on the class intercepts
    # warm start intercepts at log class frequencies
    freq = (Y.sum(0) + 1.0) / (n + K)
    O = np.zeros((n, K)) if offsets is None else np.asarray(offsets, dtype=float)
    if offsets is None:
        W[0] = np.log(freq) - np.log(freq).mean()
    else:
        reg[0, 0] = l2
    lr = 0.5
    def loss_grad(W):
        Z = Xa @ W + O
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        loss = -np.sum(Y * np.log(P + 1e-12)) / n + 0.5 * np.sum(reg * W * W) / n
        g = Xa.T @ (P - Y) / n + reg * W / n
        return loss, g
    loss, g = loss_grad(W)
    for _ in range(iters):
        step = lr
        while True:
            W2 = W - step * g
            l2_, g2 = loss_grad(W2)
            if l2_ <= loss - 1e-4 * step * float(np.sum(g * g)) or step < 1e-6:
                break
            step *= 0.5
        if abs(loss - l2_) < 1e-9:
            W, loss, g = W2, l2_, g2
            break
        W, loss, g = W2, l2_, g2
        lr = min(step * 2.0, 8.0)
    return W.tolist()


def predict_contextual(W: Sequence[Sequence[float]], x: Sequence[float],
                       offset: Sequence[float] | None = None) -> list[float]:
    z = [sum(x[i] * W[i][k] for i in range(len(x))) + (offset[k] if offset else 0.0) for k in range(K)]
    return _softmax(z)


def training_matrix(rows: Sequence[Any], hist_rows: Sequence[Any] | None = None) -> tuple[list, list, list]:
    """Rows of a set of games, each predicted with history = strictly earlier games."""
    games: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        games[r.session].append(r)
    ordered = sorted(games.items(), key=lambda kv: (min(r.started_ts for r in kv[1]), kv[0]))
    X, y, O = [], [], []
    seen: list[Any] = list(hist_rows or [])
    for sess, g in ordered:
        hist = History(seen)
        for r in sorted(g, key=lambda r: r.order):
            if r.concept in CLASSES and r.closed:
                X.append(features(hist, g, r))
                y.append(CLASSES.index(r.concept))
                O.append([math.log(max(1e-9, v)) for v in v2_dirichlet(hist, g, r.order)])
        seen += g
    return X, y, O


L2_GRID = (1.0, 3.0, 10.0, 30.0)


def select_l2(rows: Sequence[Any], residual: bool = False) -> tuple[float, dict[str, float]]:
    """Inner forward validation on TRAINING games only: hold out the last training game."""
    games = sorted({r.session: r.started_ts for r in rows}.items(), key=lambda kv: (kv[1], kv[0]))
    if len(games) < 3:
        return 10.0, {}
    last = games[-1][0]
    tr = [r for r in rows if r.session != last]
    te = [r for r in rows if r.session == last]
    Xtr, ytr, Otr = training_matrix(tr)
    hist = History(tr)
    scores = {}
    for l2 in L2_GRID:
        W = fit_contextual(Xtr, ytr, l2=l2, offsets=Otr if residual else None)
        ll, n = 0.0, 0
        for r in sorted(te, key=lambda r: r.order):
            if r.concept in CLASSES and r.closed:
                off = [math.log(max(1e-9, v)) for v in v2_dirichlet(hist, te, r.order)] if residual else None
                p = predict_contextual(W, features(hist, te, r), off)
                ll -= math.log(max(1e-9, p[CLASSES.index(r.concept)]))
                n += 1
        scores[str(l2)] = ll / n if n else float("inf")
    best = min(L2_GRID, key=lambda v: scores[str(v)])
    return best, scores


def fit(rows: Sequence[Any]) -> dict[str, Any]:
    X, y, O = training_matrix(rows)
    ok = len(set(y)) >= 2 and len(y) >= 20
    l2, inner = select_l2(rows)
    l2r, inner_r = select_l2(rows, residual=True)
    W = fit_contextual(X, y, l2=l2) if ok else None
    Wr = fit_contextual(X, y, l2=l2r, offsets=O) if ok else None
    return {"history": History(rows).to_json(), "contextual_weights": W, "l2": l2, "inner_validation": inner,
            "residual_weights": Wr, "residual_l2": l2r, "residual_inner_validation": inner_r,
            "n_train": len(y), "feature_names": FEATURE_NAMES, "classes": list(CLASSES)}


def belief(model: Mapping[str, Any], method: str, hist: History, same_game: Sequence[Any], snap: Any) -> list[float]:
    if method == "contextual" and model.get("contextual_weights"):
        return predict_contextual(model["contextual_weights"], features(hist, same_game, snap))
    if method == "v2_plus_residual" and model.get("residual_weights"):
        off = [math.log(max(1e-9, v)) for v in v2_dirichlet(hist, same_game, snap.order)]
        return predict_contextual(model["residual_weights"], features(hist, same_game, snap), off)
    if method == "hier_dirichlet":
        return hier_dirichlet(hist, same_game, snap.order, snap.opponent)
    return v2_dirichlet(hist, same_game, snap.order)


METHODS = ("marginal", "v2_dirichlet", "hier_dirichlet", "contextual", "v2_plus_residual")


def evaluate_forward(rows: Sequence[Any], *, bootstraps: int = 2000, seed: int = 27) -> dict[str, Any]:
    """Whole-game forward replay: fit on earlier games, predict each later game's snaps in order."""
    games: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        games[r.session].append(r)
    ordered = sorted(games.items(), key=lambda kv: (min(r.started_ts for r in kv[1]), kv[0]))
    per_game = []
    for i, (sess, g) in enumerate(ordered):
        prior = [r for s, gg in ordered[:i] for r in gg]
        if not any(r.concept in CLASSES for r in prior):
            continue
        model = fit(prior)
        hist = History(prior)
        pooled_tot = sum(hist.pooled.values())
        marg = [(hist.pooled.get(c, 0) + 1.0) / (pooled_tot + K) for c in CLASSES]
        stats = {m: {"ll": 0.0, "acc": 0, "brier": 0.0, "rp_ll": 0.0} for m in METHODS}
        n = 0
        for r in sorted(g, key=lambda r: r.order):
            if r.concept not in CLASSES or not r.closed:
                continue
            yi = CLASSES.index(r.concept)
            is_pass = r.concept in PASS_CLASSES
            n += 1
            for m in METHODS:
                p = marg if m == "marginal" else belief(model, m, hist, g, r)
                stats[m]["ll"] -= math.log(max(1e-9, p[yi]))
                stats[m]["acc"] += int(max(range(K), key=lambda k: p[k]) == yi)
                stats[m]["brier"] += sum((p[k] - (1.0 if k == yi else 0.0)) ** 2 for k in range(K))
                pp = min(1 - 1e-6, max(1e-6, sum(p[k] for k, c in enumerate(CLASSES) if c in PASS_CLASSES)))
                stats[m]["rp_ll"] -= math.log(pp if is_pass else 1 - pp)
        if n:
            per_game.append({"session": sess, "opponent": g[0].opponent, "started_ts": g[0].started_ts,
                             "n": n, "train_games": i, "train_snaps": model["n_train"], "l2": model["l2"],
                             **{m: {k: v / n for k, v in s.items()} for m, s in stats.items()}})
    total = sum(p["n"] for p in per_game)
    summary: dict[str, Any] = {"heldout_games": len(per_game), "heldout_snaps": total, "per_game": per_game}
    for m in METHODS:
        summary[m] = {k: (sum(p[m][k] * p["n"] for p in per_game) / total if total else None)
                      for k in ("ll", "acc", "brier", "rp_ll")}
    rng = random.Random(seed)
    for m in ("contextual", "v2_plus_residual", "hier_dirichlet", "marginal"):
        diffs = []
        for _ in range(bootstraps):
            sample = [rng.choice(per_game) for _ in per_game] if per_game else []
            nn = sum(p["n"] for p in sample)
            if nn:
                diffs.append(sum((p["v2_dirichlet"]["ll"] - p[m]["ll"]) * p["n"] for p in sample) / nn)
        diffs.sort()
        point = summary["v2_dirichlet"]["ll"] - summary[m]["ll"] if total else None
        summary[f"{m}_vs_v2_logloss_gain"] = {
            "point": point,
            "game_cluster_bootstrap_95ci": [diffs[int(0.025 * (len(diffs) - 1))], diffs[int(0.975 * (len(diffs) - 1))]]
            if len(per_game) >= 2 else None}
    best = "v2_dirichlet"
    for m in ("contextual", "v2_plus_residual", "hier_dirichlet"):
        ci = summary[f"{m}_vs_v2_logloss_gain"]["game_cluster_bootstrap_95ci"]
        if ci and ci[0] > 0 and summary[f"{m}_vs_v2_logloss_gain"]["point"] > (
                summary.get(f"{best}_vs_v2_logloss_gain", {}).get("point") or 0):
            best = m
    summary["selected_live_belief"] = best
    summary["selection_rule"] = ("challenger replaces V2 belief only if its game-clustered 95% CI of log-loss gain "
                                 "on later held-out games is entirely above 0")
    return summary
