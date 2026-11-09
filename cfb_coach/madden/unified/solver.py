"""Strategic decision solver, one pipeline for both coordinators:

    GameState -> LegalCandidates -> OpponentBeliefs -> MatchupPredictions -> StrategicOptimization -> FinalAction

Stage A screens concept families with an optimistic/pessimistic interval (a family is dropped only
when its upper bound cannot beat the best lower bound, and families with exact-level evidence are
never dropped). Stage B scores every exact action (stock play x compatible adjustments, including
NO ADJUSTMENT) against the belief distribution.

Objective per action, from OUR perspective (utility ``u``; defense minimises the opponent's value):

    score = robust_EV(u) - lambda * model_uncertainty - execution_cost - predictability_penalty

``robust_EV`` is the worst case over a total-variation ball around the belief (radius shrinks with
evidence). This is distributional robustness, NOT an equilibrium or minimax claim. There are no
fixed overrides, no random change-ups, and no post-selection substitutions: the displayed action is
the argmax (or the logged stochastic draw with its recorded policy probability).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from cfb_coach.madden.unified import knowledge as K
from cfb_coach.madden.unified.adjustments import AdjustmentPlanner
from cfb_coach.madden.unified.opponent import Belief
from cfb_coach.madden.unified.types import DEF, OFF, FootballAction, GameState, stable_hash
from cfb_coach.madden.unified.world_model import DefSig, OutcomePrediction, Profile, WorldModel, cold_start_prior

SOLVER_VERSION = "unified_solver.v1"


@dataclass(frozen=True)
class SolverConfig:
    lam_uncertainty: float = 0.8
    max_signatures: int = 12
    keep_families: int = 6
    z_weak: float = 1.64
    z_learned: float = 1.0
    min_adjust_gain: float = 0.01
    exec_cost_per_second: float = 0.012
    predictability_weight: float = 0.05
    softmax_temperature: float | None = None  # None => deterministic argmax
    seed: int | None = None
    report_top: int = 6


@dataclass
class ScoredAction:
    action: FootballAction
    ev: float
    robust: float
    uncertainty: float
    exec_cost: float
    predictability: float
    score: float
    expected: dict[str, float]  # belief-weighted outcome summary (from the offense point of view)
    support_n: float
    source: str  # cold_start_prior | mixed | learned
    adjust_support: str | None = None


@dataclass
class Decision:
    chosen: ScoredAction
    ranked: list[ScoredAction]
    candidate_set_id: str
    policy_prob: float
    seed: int | None
    model_version: str
    n_candidates: int
    n_stage_b: int
    families_kept: int
    families_total: int
    rejected_adjustments: list[tuple[str, str]]
    belief_sources: str
    fallback: str | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)


def _top_signatures(dist: Mapping[tuple, float], n: int, mass: float = 0.92, min_n: int = 4) -> list[tuple[tuple, float]]:
    """Most likely signatures covering ``mass`` of the belief (at least ``min_n``, at most ``n``); renormalised."""
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    items, acc = [], 0.0
    for kv in ranked:
        if len(items) >= n or (acc >= mass and len(items) >= min_n):
            break
        items.append(kv)
        acc += kv[1]
    tot = sum(p for _, p in items) or 1.0
    return [(s, p / tot) for s, p in items]


def robust_value(us: Sequence[float], ps: Sequence[float], eps: float) -> float:
    """Worst-case expected utility over a TV ball of radius ``eps`` around ``ps`` (exact)."""
    if not us:
        return 0.0
    order = sorted(range(len(us)), key=lambda i: us[i])
    p = list(ps)
    lo = order[0]
    take = min(eps, 1.0 - p[lo])
    removed = 0.0
    for i in reversed(order):
        if i == lo or removed >= take:
            continue
        d = min(p[i], take - removed)
        p[i] -= d
        removed += d
    p[lo] += removed
    return sum(pi * ui for pi, ui in zip(p, us))


class Solver:
    def __init__(self, model: WorldModel, planner: AdjustmentPlanner | None = None, personal: Any = None,
                 config: SolverConfig | None = None) -> None:
        self.model, self.planner, self.personal = model, planner or AdjustmentPlanner(), personal
        self.cfg = config or SolverConfig()
        self._info: dict[str, tuple] = {}
        self._oprof: dict[tuple, Profile] = {}
        self._dsigs: dict[tuple, DefSig] = {}

    # ---- helpers ---------------------------------------------------------------------
    @staticmethod
    def _fkey(features: Mapping[str, Any], *extra: Any) -> str:
        import json
        return json.dumps([sorted(features.items(), key=lambda kv: kv[0]), *extra], default=str)

    def _action_info(self, a: FootballAction, side: str) -> tuple[Profile | None, DefSig | None, str]:
        """Per-action profile and feature key, computed once for the life of the coordinator (candidates are cached)."""
        info = self._info.get(a.key)
        if info is None:
            prof = Profile.off(a) if side == OFF else None
            own = DefSig.from_signature(a.signature, a.play.play_id, {**a.play.features, **a.modified_features}) if side == DEF else None
            fk = self._fkey(prof.features, prof.play_type, prof.concept) if prof else self._fkey(own.features, own.ctype, own.family, own.vkey)
            info = self._info[a.key] = (prof, own, fk)
        return info

    def _dsig(self, sig: tuple) -> DefSig:
        d = self._dsigs.get(sig)
        if d is None:
            d = self._dsigs[sig] = DefSig.from_signature(sig)
        return d

    def _opp_profile(self, sig: tuple) -> Profile:
        pr = self._oprof.get(sig)
        if pr is None:
            pr = self._oprof[sig] = Profile(sig[0] if sig else "unknown", sig[1] if len(sig) > 1 else "unknown", None, K.offense_features_from_signature(sig))
        return pr

    def _pred(self, st: GameState, side: str, a: FootballAction | Profile, own_def: DefSig | None,
              sig: tuple, opp_ctx: str, memo: dict | None = None, ckey: str | None = None) -> OutcomePrediction:
        """Predictions are shared by every candidate with identical matchup features. The exact-play levels (L4/L5) are the only
        play-identity dependence, so a play WITH exact evidence is never memoised."""
        exact = side == OFF and isinstance(a, Profile) and a.play_id in self.model.seen_off_play_ids
        if memo is not None and ckey is not None and not exact:
            k = (ckey, sig)
            hit = memo.get(k)
            if hit is not None:
                return hit
        if side == OFF:  # our offense vs their defense signature
            d = self._dsig(sig)
            prior = None
            if memo is not None and ckey is not None:  # the cold-start prior only depends on features, never play identity
                prior = memo.get(("P", ckey, sig))
                if prior is None:
                    prior = memo[("P", ckey, sig)] = cold_start_prior(st, a.features, d.features)
            pr = self.model.predict(st, a, d, personal=self.personal, our_side=OFF, opponent_class=opp_ctx, prior=prior)
        else:
            pr = self.model.predict(st, self._opp_profile(sig), own_def, personal=None, our_side=DEF, opponent_class=opp_ctx)
        if memo is not None and ckey is not None and not exact:
            memo[(ckey, sig)] = pr
        return pr

    @staticmethod
    def _own_def(a: FootballAction) -> DefSig:
        return DefSig.from_signature(a.signature, a.play.play_id, {**a.play.features, **a.modified_features})

    def _eff_weight(self) -> float:
        o = self.model.objective
        return o.w_yards + o.w_first_down * 0.025

    # ---- core ------------------------------------------------------------------------
    def decide(self, st: GameState, side: str, candidates: Sequence[FootballAction], belief: Belief, *,
               recent_own_families: Sequence[str] = (), deadline_ms: float | None = None) -> Decision:
        import time
        t0 = time.perf_counter()
        cfg = self.cfg
        sign = 1.0 if side == OFF else -1.0
        legal = [a for a in candidates if a.legal and a.executable]
        if not any(a.no_adjustment for a in legal):
            raise ValueError("NO ADJUSTMENT must remain a legal candidate")
        sigs = _top_signatures(belief.dist, cfg.max_signatures)
        ps = [p for _, p in sigs]
        eps = belief.tv_radius
        opp_ctx = st.opponent_class
        w_eff = self._eff_weight()
        sign_adj = 1.0  # adjustment dy is "in our favour" -> utility +

        memo: dict = {}

        def stat_for(a: FootballAction, own_def: DefSig | None, prof: Profile | None, ckey: str | None = None):
            us, vs, vsds, ey, fd, td, to, ex, sk = ([] for _ in range(9))
            sup, srcs = 0.0, []
            if ckey is None:
                ckey = self._action_info(a, side)[2]
            for sig, _p in sigs:
                pr = self._pred(st, side, prof if side == OFF else a, own_def, sig, opp_ctx, memo, ckey)
                us.append(sign * pr.value)
                vsds.append(pr.value_sd)
                ey.append(pr.e_yards); fd.append(pr.p_first_down); td.append(pr.p_td); to.append(pr.p_turnover)
                ex.append(pr.p_explosive); sk.append(pr.p_sack)
                sup = max(sup, pr.support_n)
                srcs.append(pr.source)
            return us, vsds, (ey, fd, td, to, ex, sk), sup, srcs

        # ---- Stage A: concept-family screening on representative (stock) plays -------
        t1 = time.perf_counter()
        fams: dict[str, list[FootballAction]] = {}
        for a in legal:
            fams.setdefault(a.play.concept_path[1] if len(a.play.concept_path) > 1 else a.play.concept, []).append(a)
        fam_bounds: dict[str, tuple[float, float, float]] = {}
        exact_evidence: set[str] = set()
        for fam, acts in fams.items():
            rep = acts[0]
            prof = Profile.off(rep) if side == OFF else None
            prof_coarse = Profile(prof.play_type, prof.concept, None, prof.features) if prof else None
            own_coarse = DefSig.from_signature(rep.signature, None, {**rep.play.features, **rep.modified_features}) if side == DEF else None
            us, vsds, _, _, _ = stat_for(rep, own_coarse, prof_coarse, "A:" + self._action_info(rep, side)[2])
            ev = sum(p * u for p, u in zip(ps, us))
            unc = sum(p * s for p, s in zip(ps, vsds))
            fam_bounds[fam] = (ev, ev - 1.5 * unc, ev + 1.5 * unc)
            if any((x.play.play_id in self.model.seen_off_play_ids) if side == OFF else (x.signature[-1] in self.model.seen_vkeys)
                   for x in acts):
                exact_evidence.add(fam)
        best_lcb = max(b[1] for b in fam_bounds.values())
        ranked_f = sorted(fam_bounds, key=lambda f: -fam_bounds[f][0])
        keep = set(ranked_f[:cfg.keep_families]) | {f for f, b in fam_bounds.items() if b[2] >= best_lcb} | exact_evidence
        t2 = time.perf_counter()

        # ---- Stage B: exact actions --------------------------------------------------------
        noadj_scores: dict[str, ScoredAction] = {}
        scored: list[ScoredAction] = []
        rejected: list[tuple[str, str]] = []
        pending_adj: list[FootballAction] = []
        partial = False

        coarse_cache: dict[str, float] = {}
        full_cache: dict[str, tuple] = {}

        def core_for(a: FootballAction, prof: Profile | None, own: DefSig | None, fk: str) -> tuple:
            us, vsds, mats, sup, srcs = stat_for(a, own, prof)
            ev = sum(p * u for p, u in zip(ps, us))
            rob = robust_value(us, ps, eps)
            unc = sum(p * s for p, s in zip(ps, vsds))
            expd = {k: sum(p * v for p, v in zip(ps, m)) for k, m in zip(("e_yards", "p_first_down", "p_td", "p_turnover", "p_explosive", "p_sack"), mats)}
            src = "learned" if all(s == "learned" for s in srcs) else "cold_start_prior" if all(s == "cold_start_prior" for s in srcs) else "mixed"
            return (ev, rob, unc, expd, sup, src)

        def coarse_ev(a: FootballAction) -> float:
            prof, own, fk = self._action_info(a, side)
            if fk in coarse_cache:
                return coarse_cache[fk]
            if not (side == OFF and a.play.play_id in self.model.seen_off_play_ids):
                # no exact-level evidence: the coarse and exact predictions are identical, so compute once and reuse in stage B
                full_cache[fk] = core_for(a, prof, own, fk)
                coarse_cache[fk] = full_cache[fk][0]
                return coarse_cache[fk]
            pc = Profile(prof.play_type, prof.concept, None, prof.features)
            us, _v, _m, _s, _x = stat_for(a, None, pc, "A:" + fk)
            coarse_cache[fk] = sum(p * u for p, u in zip(ps, us))
            return coarse_cache[fk]

        # Branch-and-bound for plays with exact-level evidence (the only expensive path): their exact-level correction is bounded
        # by 3 sd of the model's own effect-size prior (tau at L4+L5). score <= expected utility <= coarse + swing.
        o = self.model.objective
        tau_y = self.model.tau_y[4] + self.model.tau_y[5]
        tau_r = self.model.tau_r[4] + self.model.tau_r[5]
        swing = 3.0 * (o.w_yards * tau_y + (o.w_first_down * 1.3 + o.w_td + o.w_turnover + o.turnover_field_factor * 20 + o.w_sack) * tau_r)
        pruned = 0

        order = [a for fam in keep for a in fams[fam]]
        order.sort(key=lambda a: (a.adjustments != (), -coarse_ev(a) if not a.adjustments else 0.0))  # most promising first
        for a in order:
            if deadline_ms is not None and (time.perf_counter() - t0) * 1e3 > deadline_ms * 0.85 and noadj_scores:
                partial = True
                break
            if a.adjustments:
                pending_adj.append(a)
                continue
            if side == OFF and scored and a.play.play_id in self.model.seen_off_play_ids \
                    and coarse_ev(a) + swing < max(x.score for x in scored):
                pruned += 1  # provably cannot beat the best exact score already found
                continue
            prof, own, fk = self._action_info(a, side)
            exact = side == OFF and a.play.play_id in self.model.seen_off_play_ids
            core = None if exact else full_cache.get(fk)
            if core is None:
                core = core_for(a, prof, own, fk)
                if not exact:
                    full_cache[fk] = core
            ev, rob, unc, expd, sup, src = core
            pred_pen = self._predictability(a, st, recent_own_families)
            sa = ScoredAction(a, ev, rob, unc, 0.0, pred_pen, rob - cfg.lam_uncertainty * unc - pred_pen, dict(expd), sup, src)
            noadj_scores[a.play.play_id] = sa
            scored.append(sa)
        for a in pending_adj:  # adjusted variants judged against their own stock play's NO ADJUSTMENT
            base = noadj_scores.get(a.play.play_id)
            if base is None:
                rejected.append((a.key, "stock play not evaluated"))
                continue
            dys, sds, sup_kinds = [], [], []
            for sig, _p in sigs:
                dy_t, sd_t = 0.0, 0.0
                for adj in a.adjustments:
                    dy, sd, kind = self.planner.effect(adj, side, sig)
                    dy_t += dy
                    sd_t = math.hypot(sd_t, sd)
                    sup_kinds.append(kind)
                dys.append(dy_t * w_eff)  # sign_adj: dy is always in OUR favour
                sds.append(sd_t * w_eff)
            d_mean = sum(p * d for p, d in zip(ps, dys))
            d_sd = sum(p * s for p, s in zip(ps, sds))
            cost = a.exec_cost_s * cfg.exec_cost_per_second
            learned = all(k == "learned" for k in sup_kinds)
            z = cfg.z_learned if learned else cfg.z_weak
            lcb_gain = d_mean - z * d_sd - cost
            if lcb_gain <= cfg.min_adjust_gain:
                rejected.append((a.key, f"benefit {d_mean:+.3f} does not survive uncertainty {z:.2f}x{d_sd:.3f} + cost {cost:.3f}"))
                continue
            sa = ScoredAction(a, base.ev + d_mean, base.robust + d_mean, math.hypot(base.uncertainty, d_sd), cost,
                              base.predictability,
                              base.robust + d_mean - cfg.lam_uncertainty * math.hypot(base.uncertainty, d_sd) - cost - base.predictability,
                              dict(base.expected), base.support_n, base.source, "learned" if learned else "weak_prior")
            scored.append(sa)
        scored.sort(key=lambda s: (-round(s.score, 9), -s.action.play.confidence, s.action.key))
        t3 = time.perf_counter()

        chosen, prob = scored[0], 1.0
        seed = cfg.seed
        if cfg.softmax_temperature:
            pool = scored[:8]
            m = max(s.score for s in pool)
            wts = [math.exp((s.score - m) / cfg.softmax_temperature) for s in pool]
            tot = sum(wts)
            rng = random.Random(seed)
            r, acc = rng.random() * tot, 0.0
            for s, w in zip(pool, wts):
                acc += w
                if r <= acc:
                    chosen, prob = s, w / tot
                    break
        cset = stable_hash(sorted(a.key for a in legal))
        return Decision(chosen, scored[:cfg.report_top], cset, prob, seed, SOLVER_VERSION + "/" + self.model.objective.name,
                        len(legal), len(scored) + len(rejected) + pruned, len(keep), len(fams), rejected, belief.prediction_source,
                        "deadline_partial_search" if partial else None,
                        {"stageA_ms": (t2 - t1) * 1e3, "stageB_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3})

    def _predictability(self, a: FootballAction, st: GameState, recent: Sequence[str]) -> float:
        """Sequencing penalty — only where supported (human opponent who can adapt); CPU: none."""
        if st.opponent_class != "human" or not recent:
            return 0.0
        fam = a.play.concept_path[1] if len(a.play.concept_path) > 1 else a.play.concept
        return self.cfg.predictability_weight * sum(1 for f in recent[-3:] if f == fam) / 3.0

