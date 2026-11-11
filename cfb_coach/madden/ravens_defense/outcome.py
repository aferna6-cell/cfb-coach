"""Defensive outcome model: P(offense success | complete candidate, situation, offensive concept).

Architecture ``ravens_def_outcome.bayes_structural_logit.v1``
  logit = base(situation, concept)                       # candidate-agnostic, trained on OBSERVED opponent plays
        + beta * structural_lift(candidate, concept)    # sourced football_priors table read from the defense side
        + sum w * [axis(candidate) x concept]            # learned only from VERIFIED Ravens executions
        + sum u * [situation x axis(candidate)]          # (red zone, short, long, 3rd/4th) learned from verified only
        + v[candidate]                                   # per-candidate residual, strongly shrunk (rare combos)
Prior: beta ~ N(1, 0.5^2); every w/u/v ~ N(0, s^2). The posterior is a Laplace approximation (MAP + Hessian),
so predictions carry uncertainty. With ZERO verified executions the candidate part IS the prior: the artifact is
marked ``outcome_model_status = UNTRAINED_PRIOR_ONLY`` and no learned coverage effect is claimed.
Historical associations are observational; they do not prove a different call would have changed the outcome.
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from typing import Any, Mapping, Sequence

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense.concepts import CLASSES, FP_CONCEPTS
from cfb_coach.madden.ravens_defense.opponent import CTX_TOKENS, context_tokens

ARCH = "ravens_def_outcome.bayes_structural_logit.v1"
AXES = ("shell_4", "shell_3", "shell_2", "shell_split", "shell_unk", "type_match", "type_spot_zone", "type_unk",
        "press", "zone_it", "pkg_nickel_stock", "var_tampa", "var_cloud", "var_quarters", "var_palms")
CTX_AXES = ("deep_shell", "press", "type_match", "var_tampa", "zone_it")
CTX_FOR_INTERACTION = ("rz10", "rz20", "short", "long", "d34")
PRIOR = {"beta_mean": 1.0, "beta_sd": 0.5, "axis_concept_sd": 0.30, "ctx_axis_sd": 0.30, "candidate_sd": 0.20}
MIN_VERIFIED_FOR_TRAINED = 200   # same evidence bar as Defensive V2's prediction gate
MIN_GAMES_FOR_TRAINED = 8
UTILITY = {"success": 1.0, "touchdown": 0.15, "explosive": 0.10, "turnover": -0.05}  # Defensive V2 rank() weights
OPTIONAL_MACRO_COST = 0.004  # one extra pre-snap input (execution cost), not an outcome claim
AUX = ("explosive", "touchdown", "turnover", "first_down")


def axes(cand: rc.Candidate) -> dict[str, float]:
    s = cand.structure
    a = {f"shell_{s['shell']}": 1.0, f"type_{s['type']}": 1.0}
    if s.get("press") not in (None, "", "standard"):
        a["press"] = 1.0
    if s.get("zone_it"):
        a["zone_it"] = 1.0
    if cand.package == "nickel_stock":
        a["pkg_nickel_stock"] = 1.0
    for v in ("tampa", "cloud", "quarters", "palms"):
        if v in s.get("variants", ()):
            a[f"var_{v}"] = 1.0
    return {k: v for k, v in a.items() if k in AXES}


@lru_cache(maxsize=256)
def structural_lift(candidate_id: str, concept: str) -> tuple[float, tuple[str, ...]]:
    """Offense logit lift ALLOWED by this candidate vs this concept class, from the sourced prior table."""
    from cfb_coach.madden.model import football_priors as fp
    cand = rc.load().by_id()[candidate_id]
    desc = {k: cand.structure[k] for k in ("shell", "type", "variants", "press")}
    lift, ids = 0.0, []
    for fc, w in FP_CONCEPTS[concept].items():
        l, _conf, pid = fp.lift_for(desc, fc)
        lift += w * l
        ids += pid
    return lift, tuple(ids)


def phi(cand: rc.Candidate, concept: str, ctx: Sequence[str]) -> dict[str, float]:
    """Sparse candidate-specific feature vector (keys are parameter names)."""
    f: dict[str, float] = {}
    lift, _ = structural_lift(cand.candidate_id, concept)
    if lift:
        f["beta"] = lift
    ax = axes(cand)
    for a in ax:
        f[f"ax:{a}|{concept}"] = 1.0
    small = {"deep_shell": cand.structure["shell"] in ("4", "split"), "press": "press" in ax,
             "type_match": "type_match" in ax, "var_tampa": "var_tampa" in ax, "zone_it": "zone_it" in ax}
    for t in ctx:
        if t in CTX_FOR_INTERACTION:
            for a, on in small.items():
                if on:
                    f[f"ctx:{t}|{a}"] = 1.0
    f[f"cand:{cand.candidate_id}"] = 1.0
    return f


def prior_of(name: str) -> tuple[float, float]:
    if name == "beta":
        return PRIOR["beta_mean"], PRIOR["beta_sd"]
    if name.startswith("ax:"):
        return 0.0, PRIOR["axis_concept_sd"]
    if name.startswith("ctx:"):
        return 0.0, PRIOR["ctx_axis_sd"]
    return 0.0, PRIOR["candidate_sd"]


def param_names(cat: rc.RavensCatalog) -> list[str]:
    names = ["beta"] + [f"ax:{a}|{c}" for a in AXES for c in CLASSES]
    names += [f"ctx:{t}|{a}" for t in CTX_FOR_INTERACTION for a in CTX_AXES]
    names += [f"cand:{c.candidate_id}" for c in cat.candidates]
    return names


# ------------------------------------------------------------------ candidate-agnostic base heads
def base_x(concept: str | None, ctx: Sequence[str]) -> list[float]:
    x = [1.0] + [1.0 if concept == c else 0.0 for c in CLASSES] + [1.0 if t in ctx else 0.0 for t in CTX_TOKENS]
    return x


def _fit_logit(X: Sequence[Sequence[float]], y: Sequence[float], *, l2: float = 2.0) -> list[float] | None:
    if len(y) < 20 or len(set(y)) < 2:
        return None
    import numpy as np
    Xa, ya = np.asarray(X, float), np.asarray(y, float)
    w = np.zeros(Xa.shape[1])
    p0 = min(0.99, max(0.01, ya.mean()))
    w[0] = math.log(p0 / (1 - p0))
    reg = np.full(Xa.shape[1], l2)
    reg[0] = 1e-3
    for _ in range(50):
        p = 1 / (1 + np.exp(-(Xa @ w)))
        g = Xa.T @ (p - ya) + reg * w
        H = (Xa.T * (p * (1 - p))) @ Xa + np.diag(reg)
        step = np.linalg.solve(H, g)
        w -= step
        if float(np.abs(step).max()) < 1e-8:
            break
    return w.tolist()


def fit_base(rows: Sequence[Any]) -> dict[str, Any]:
    """Opponent success / aux rates by situation and concept class, from observed opponent plays.

    These rows were NOT verified executions of our call, so they never touch candidate terms."""
    heads: dict[str, Any] = {}
    for head in ("success",) + AUX:
        X, y = [], []
        for r in rows:
            val = r.offense_success if head == "success" else getattr(r, head)
            if val is None or not r.closed:
                continue
            X.append(base_x(r.concept, context_tokens(r.down, r.distance, r.yards_to_goal)))
            y.append(1.0 if val else 0.0)
        heads[head] = {"weights": _fit_logit(X, y), "n": len(y)}
    # Defensive V2's context-only success rate (its baseline() shrinkage), kept as the do-no-harm base
    from cfb_coach.madden.model import defense_v2_pilot as v2
    labeled = [r for r in rows if r.closed and r.offense_success is not None]
    glob = (sum(r.offense_success for r in labeled) + 2.0) / (len(labeled) + 4.0) if labeled else 0.5
    ctab: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in labeled:
        k = v2.context_key(r.down, r.distance)
        ctab[k][0] += int(r.offense_success)
        ctab[k][1] += 1
    heads["v2_context"] = {"global": glob, "contexts": {k: list(v) for k, v in ctab.items()}, "n": len(labeled)}
    heads["success_base_mode"] = "concept_logit"
    ys = defaultdict(list)
    for r in rows:
        if r.yards is not None and r.concept:
            ys[r.concept].append(r.yards)
    allv = [v for vs in ys.values() for v in vs]
    gm = sum(allv) / len(allv) if allv else None
    heads["yards"] = {"by_concept": {c: ((sum(v) + 10 * gm) / (len(v) + 10) if gm is not None else None)
                                     for c, v in ys.items()}, "global": gm, "n": len(allv)}
    return heads


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z > -30 else 0.0


def _v2_key(ctx: Sequence[str]) -> str:
    t = set(ctx)
    if "short" in t:
        return "short"
    if "d34" in t and "long" in t:
        return "long"
    if "d34" in t:
        return "medium"
    if "dist_unk" in t or "dunk" in t:
        return "unknown"
    return "early"


def base_logit(base: Mapping[str, Any], head: str, concept: str, ctx: Sequence[str]) -> float | None:
    if head == "success" and base.get("success_base_mode") == "v2_context" and base.get("v2_context"):
        v = base["v2_context"]
        s, n = (v["contexts"].get(_v2_key(ctx)) or [0, 0])
        p = (s + 16.0 * v["global"]) / (n + 16.0)
        return math.log(p / (1 - p))
    w = (base.get(head) or {}).get("weights")
    if not w:
        return None
    x = base_x(concept, ctx)
    return sum(a * b for a, b in zip(x, w))


# ------------------------------------------------------------------ candidate posterior (Laplace)
def fit_candidate_terms(cat: rc.RavensCatalog, verified: Sequence[Any], base: Mapping[str, Any]) -> dict[str, Any]:
    names = param_names(cat)
    idx = {n: i for i, n in enumerate(names)}
    mean0 = [prior_of(n)[0] for n in names]
    sd0 = [prior_of(n)[1] for n in names]
    rows = [r for r in verified if r.offense_success is not None and r.concept is not None and r.candidate_id]
    status = ("TRAINED" if len(rows) >= MIN_VERIFIED_FOR_TRAINED and len({r.session for r in rows}) >= MIN_GAMES_FOR_TRAINED
              else "PARTIAL_UNDER_EVIDENCE_BAR" if rows else "UNTRAINED_PRIOR_ONLY")
    if not rows:
        return {"names": names, "mean": mean0, "cov": None, "prior_sd": sd0, "n": 0, "games": 0, "status": status}
    import numpy as np
    by = cat.by_id()
    X = np.zeros((len(rows), len(names)))
    off = np.zeros(len(rows))
    y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        ctx = context_tokens(r.down, r.distance, r.yards_to_goal)
        for k, v in phi(by[r.candidate_id], r.concept, ctx).items():
            X[i, idx[k]] = v
        off[i] = base_logit(base, "success", r.concept, ctx) or 0.0
        y[i] = 1.0 if r.offense_success else 0.0
    m0, P0 = np.asarray(mean0), np.diag(1.0 / np.asarray(sd0) ** 2)
    w = m0.copy()
    for _ in range(60):
        p = 1 / (1 + np.exp(-(X @ w + off)))
        g = X.T @ (p - y) + P0 @ (w - m0)
        H = (X.T * (p * (1 - p))) @ X + P0
        step = np.linalg.solve(H, g)
        w -= step
        if float(np.abs(step).max()) < 1e-8:
            break
    p = 1 / (1 + np.exp(-(X @ w + off)))
    H = (X.T * (p * (1 - p))) @ X + P0
    cov = np.linalg.inv(H)
    return {"names": names, "mean": w.tolist(), "cov": cov.tolist(), "prior_sd": sd0, "n": len(rows),
            "games": len({r.session for r in rows}), "status": status}


class OutcomeModel:
    def __init__(self, cat: rc.RavensCatalog, base: Mapping[str, Any], cand_terms: Mapping[str, Any]):
        self.cat, self.base, self.terms = cat, base, cand_terms
        self.idx = {n: i for i, n in enumerate(cand_terms["names"])}
        self.mean = cand_terms["mean"]
        self.cov = cand_terms.get("cov")
        self.sd0 = cand_terms["prior_sd"]

    @property
    def status(self) -> str:
        return str(self.terms["status"])

    def _cand_logit(self, f: Mapping[str, float]) -> tuple[float, float]:
        mu = sum(v * self.mean[self.idx[k]] for k, v in f.items())
        keys = [self.idx[k] for k in f]
        vals = [f[k] for k in f]
        if self.cov is None:
            var = sum((v * self.sd0[i]) ** 2 for i, v in zip(keys, vals))
        else:
            var = sum(vals[a] * vals[b] * self.cov[keys[a]][keys[b]] for a in range(len(keys)) for b in range(len(keys)))
        return mu, max(0.0, var)

    def score(self, cand: rc.Candidate, belief: Sequence[float], ctx: Sequence[str]) -> dict[str, Any]:
        p_succ = 0.0
        m2 = 0.0
        per_concept = {}
        for k, c in enumerate(CLASSES):
            b = belief[k]
            base = base_logit(self.base, "success", c, ctx)
            base = 0.0 if base is None else base
            mu, var = self._cand_logit(phi(cand, c, ctx))
            z = base + mu
            p = _sig(z / math.sqrt(1.0 + math.pi * var / 8.0))
            lo, hi = _sig(z - 1.96 * math.sqrt(var)), _sig(z + 1.96 * math.sqrt(var))
            per_concept[c] = {"p": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                              "structural_lift": round(structural_lift(cand.candidate_id, c)[0], 4)}
            p_succ += b * p
            m2 += b * (p * p + ((hi - lo) / 3.92) ** 2)
        sd = math.sqrt(max(0.0, m2 - p_succ * p_succ))
        aux: dict[str, float | None] = {}
        for head in AUX:
            vals = [base_logit(self.base, head, c, ctx) for c in CLASSES]
            aux[head] = None if any(v is None for v in vals) else round(
                sum(belief[k] * _sig(v) for k, v in enumerate(vals)), 4)
        yd = self.base.get("yards") or {}
        if yd.get("global") is not None:
            aux["yards"] = round(sum(belief[k] * (yd["by_concept"].get(c) if yd["by_concept"].get(c) is not None
                                                  else yd["global"]) for k, c in enumerate(CLASSES)), 2)
        loss = UTILITY["success"] * p_succ
        for head in ("touchdown", "explosive", "turnover"):
            if aux.get(head) is not None:
                loss += UTILITY[head] * float(aux[head])
        if cand.optional_macro:
            loss += OPTIONAL_MACRO_COST
        return {"candidate_id": cand.candidate_id, "predicted_offense_success": round(p_succ, 4),
                "success_sd": round(sd, 4), "aux": aux, "defensive_loss": round(loss, 5),
                "per_concept": per_concept,
                "aux_candidate_specific": False,  # aux heads have no verified candidate terms yet
                "outcome_model_status": self.status}

    def to_json(self) -> dict[str, Any]:
        return {"architecture": ARCH, "prior": PRIOR, "base_heads": self.base, "candidate_terms": self.terms,
                "utility": UTILITY, "optional_macro_cost": OPTIONAL_MACRO_COST}

    @classmethod
    def from_json(cls, cat: rc.RavensCatalog, d: Mapping[str, Any]) -> "OutcomeModel":
        if d.get("architecture") != ARCH:
            raise ValueError("unsupported Ravens defensive outcome architecture")
        return cls(cat, d["base_heads"], d["candidate_terms"])
