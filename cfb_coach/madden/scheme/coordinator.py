"""civil_scheme_only offensive coordinator: picks ONLY from the scheme's listed plays, every time
(26 catalog pass variants + 12 owner-confirmed designed runs (t349u) + the 7 red-zone plays, by field band).

Order per snap (never calls the heuristic ``_pick_offense`` or the experimental offense path):
  1. legal filter    ``legality.legal_set``: catalog + Bucs stock + applied book + owner-verified
  2. belief          opponent coverage / pressure belief (earlier games + this game's closed snaps, recency weighted)
  3. score           existing-model offset + learned head + regularized adjustment transfer + expert/beater terms
                     + per-variant learned residual from verified variant outcomes - repetition penalty
  4. pick            ``pick_mixed``: outside the opponent 10, a situation-gated run-pass draw (owner t348u/t349u:
                     "a good balance of running"), then the seeded near-best draw inside the drawn family;
                     inside the 10 the plain near-best draw (red-zone selection unchanged). Then the sibling mix
                     (owner t370u): a near-best legal variation of the drawn play. Trailing late (t370u): pass-first
                     outside the 10, red-zone passes mixed in inside the 10. Re-validated by
                     ``legality.revalidate_call`` before display
  5. nothing legal   NO_LEGAL_SCHEME_PLAY with the reasons; no fallback to any other play
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.scheme import beliefs as bl
from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import design as dz
from cfb_coach.madden.scheme import evidence as ev
from cfb_coach.madden.scheme import features as ft
from cfb_coach.madden.scheme import learners as ln
from cfb_coach.madden.scheme import legality as lg
from cfb_coach.madden.scheme import mode as md
from cfb_coach.madden.scheme import model as sm
from cfb_coach.madden.scheme import priors as pr

# Variety (owner t338u): penalties are per variant_id ONLY. Calling one variant never changes another variant's
# score, including other variants of the same base play (e.g. Texas Y Stutter Wheel 1 vs 2).
REPEAT_W = 0.45            # logit penalty for the most recent call of the same variant in this game
REPEAT_DECAY = 0.7         # x0.7 per older call, over the last REPEAT_WINDOW calls
REPEAT_WINDOW = 8
GAME_W = 0.10              # extra logit penalty per earlier call of the same variant this game
GAME_CAP = 1.0
# Mixing: sample among the near-best candidates instead of always taking the argmax.
MIX_MARGIN = 0.08          # candidates within this utility of the best are eligible
MIX_TAU = 0.025            # Boltzmann temperature on utility (smaller = greedier)
# 10-20 band: red_zone-tagged plays get a lift so they actually appear in the mix (owner t338u).
RZ_MIX_BONUS = 0.35
# Run-pass mix (owner t348u/t349u). Outside the opponent 10, when both families are legal:
#   logit P(run) = RUN_PRIOR_LOGIT[down][distance bucket]                (soft situation prior, hand-set)
#                + RUN_LEARN_W * (best run utility - best pass utility)  (learned: model, residuals, beater, repeats)
#                + RUN_TEND_W * (passes - runs) / RUN_TEND_WINDOW        (this game's recent calls: stay unpredictable)
# clipped to [RUN_P_MIN, RUN_P_MAX]; then the family is drawn (seeded) and ``pick`` runs inside that family.
# Without the gate the untrained stock runs never enter the 0.08 near-best pool next to 26 pass variants that carry
# expert/beater terms (that already happened to the Close Flex Stretch in the 10-20 band).
RUN_DIST_BUCKETS = ((2, "short"), (4, "short_med"), (7, "medium"), (10, "long"), (999, "very_long"))
RUN_PRIOR_LOGIT = {
    1: {"short": 0.6, "short_med": 0.4, "medium": 0.1, "long": -0.2, "very_long": -0.8},
    2: {"short": 0.6, "short_med": 0.3, "medium": -0.2, "long": -0.6, "very_long": -1.3},
    3: {"short": 0.3, "short_med": -0.4, "medium": -1.4, "long": -2.0, "very_long": -2.6},
    4: {"short": 0.3, "short_med": -0.4, "medium": -1.4, "long": -2.0, "very_long": -2.6},
}
RUN_LEARN_W = 10.0          # logit per unit of utility gap (0.05 better -> +0.5 logit)
RUN_TEND_W = 0.6
RUN_TEND_WINDOW = 6
RUN_P_MIN, RUN_P_MAX = 0.03, 0.85
RESIDUAL_K = 10.0
RESIDUAL_CAP = 0.6          # floor: bad verified outcomes of a variant can still pull it down by up to 0.6 logit
# Sibling variety (owner t370u, Michael 59-42 audit). Variants stay independent for repetition (t338u): there is NO
# family-level repeat penalty. Instead:
#  1. the verified-outcome residual may lift a variant by at most RESIDUAL_POS_CAP (was +0.6), so history alone
#     (+0.06 utility at p~0.5, under MIX_MARGIN) cannot freeze high-history variants into the top 8;
#  2. a per-variant novelty lift for catalog pass variants that have siblings (same formation + base play), from
#     THAT variant's own verified-outcome count only: NOVELTY_W * NOVELTY_K / (NOVELTY_K + n). Unused -> +0.25 logit,
#     4 outcomes -> +0.125, 20 -> +0.04. Another variant's calls/outcomes never change it;
#  3. ``pick`` draws the play as before, then mixes among that play's legal siblings within SIBLING_MARGIN utility
#     of the best sibling (flatter SIBLING_TAU), so near-best variations of the same play are actually rotated.
RESIDUAL_POS_CAP = 0.20
NOVELTY_W = 0.25
NOVELTY_K = 4.0
SIBLING_MARGIN = 0.15
SIBLING_TAU = 0.05
# Trailing late (owner t370u: "when we are down ... in the 4th we have to get aggressive"). Uses the live score +
# quarter / two-minute flag (``game_score.classify``: phase "trail" and late = Q4+/OT or two-minute).
#  outside the 10: run logit -= TRAIL_RUN_LOGIT * strength (strength 0.55 down 1-2, 0.75 down 3-7, 1.0 down 8+),
#                  P(run) capped at TRAIL_RUN_P_MAX, and P(run) = 0 when a pass is ranked above the best designed run
#                  or the distance is >= TRAIL_LONG (2nd-and-20, 2nd-and-10, 3rd-and-7 ...): pass family only;
#  inside the 10:  a family draw is turned on: P(pass) = TRAIL_RZ_PASS_BASE + TRAIL_RZ_PASS_STRENGTH * strength
#                  (+TRAIL_RZ_LONG_BONUS at distance >= 4), at least TRAIL_RZ_PASS_TOP when a red-zone pass is
#                  ranked above the best run, capped at TRAIL_RZ_PASS_MAX. Runs (incl. RPO Alert QB Draw) stay
#                  callable. Not trailing late: everything above is off (normal run-pass mix, red-zone unchanged).
TRAIL_RUN_LOGIT = 2.5
TRAIL_RUN_P_MAX = 0.20
TRAIL_LONG = 7
TRAIL_RZ_PASS_BASE = 0.40
TRAIL_RZ_PASS_STRENGTH = 0.30
TRAIL_RZ_LONG_BONUS = 0.15
TRAIL_RZ_PASS_TOP = 0.75
TRAIL_RZ_PASS_MAX = 0.85
LATENCY_TARGET_MS = 150.0


@dataclass
class Scored:
    variant: sc.SchemeVariant
    p_success: float
    sd: float
    p_first_down: float
    p_explosive: float
    p_negative: float
    exp_yards: float | None
    utility: float
    parts: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"variant_id": self.variant.variant_id, "p_success": round(self.p_success, 4),
                "sd_logit": round(self.sd, 4), "p_first_down": round(self.p_first_down, 4),
                "p_explosive": round(self.p_explosive, 4), "p_negative": round(self.p_negative, 4),
                "exp_yards": None if self.exp_yards is None else round(self.exp_yards, 2),
                "utility": round(self.utility, 4), "parts": self.parts}


class Scorer:
    """Loaded once per artifact; per-snap scoring is pure arithmetic."""

    def __init__(self, model: sm.SchemeModel, cat: sc.SchemeCatalog, geo: dz.GeometrySource):
        self.model, self.cat, self.geo = model, cat, geo
        self.adjusted = {v.variant_id: dz.adjusted_design(v, geo) for v in cat.variants}
        self.stock = {v.pair: dz.stock_design(v.formation, v.base_play, geo) for v in cat.variants}
        self.rf_adj = {k: dz.route_features(d) for k, d in self.adjusted.items()}
        self.rf_stock = {k: dz.route_features(d) for k, d in self.stock.items()}
        pairs: dict[tuple[str, str], int] = {}
        for v in cat.variants:
            pairs[v.pair] = pairs.get(v.pair, 0) + 1
        # pass variants that share formation + base play with another catalog variant (t370u novelty lift)
        self.has_siblings = {v.variant_id: (not v.is_run and pairs[v.pair] > 1) for v in cat.variants}

    def offset_p(self, v: sc.SchemeVariant, *, down, distance, yardline, opponent_id, opponent_type) -> float:
        if self.model.existing is None:
            raise RuntimeError("artifact has no existing-model offset (existing_offense.json): retrain")
        return float(em.predict_success(self.model.existing, formation=v.formation, play=v.base_play, down=down,
                                        distance=distance, yardline=yardline, opponent_id=opponent_id,
                                        opponent_type=opponent_type)["probability"])

    def score(self, variants: Sequence[sc.SchemeVariant], *, down, distance, yardline, opponent_id: str,
              opponent_type: str, belief: bl.Belief, recent_calls: Sequence[str] = (),
              game_calls: Sequence[str] | None = None,
              variant_stats: Mapping[str, tuple[float, float]] | None = None,
              offset_override: Mapping[str, float] | None = None, stock_only: bool = False) -> list[Scored]:
        """``stock_only`` scores every variant as its unadjusted base play (evaluation diagnostics only)."""
        variant_stats = variant_stats or {}
        game_count: dict[str, int] = {}
        for rid in (game_calls if game_calls is not None else recent_calls):
            game_count[rid] = game_count.get(rid, 0) + 1
        band = sc.field_band(yardline)
        counts = {k: n for k, (_s, n) in variant_stats.items()}
        expert = pr.bonuses(variants, down=down, distance=distance, yardline=yardline, mix=belief.mix,
                            families=belief.families, info=belief.gate_confidence, verified_counts=counts)
        out: list[Scored] = []
        base_cache: dict[tuple[str, str], float] = {}
        for v in variants:
            if offset_override and v.variant_id in offset_override:
                op = offset_override[v.variant_id]
            else:
                if v.pair not in base_cache:
                    base_cache[v.pair] = self.offset_p(v, down=down, distance=distance, yardline=yardline,
                                                       opponent_id=opponent_id, opponent_type=opponent_type)
                op = base_cache[v.pair]
            xs, _ = ft.vector(down=down, distance=distance, yardline=yardline, belief=belief,
                              design=self.stock[v.pair], opponent_type=opponent_type, route_f=self.rf_stock[v.pair])
            xa, _ = ft.vector(down=down, distance=distance, yardline=yardline, belief=belief,
                              design=self.adjusted[v.variant_id], opponent_type=opponent_type,
                              route_f=self.rf_adj[v.variant_id])
            if stock_only:
                xa = xs
            ps, pa = self.model.predict(xs, op), self.model.predict(xa, op)
            d = self.adjusted[v.variant_id]
            extra_var = (len(d.unrepresented) * sm.UNREPRESENTED_SD) ** 2
            extra_var += sum(1 for _i, _sh, m in d.stems if m == "directional") * sm.DIRECTIONAL_STEM_SD ** 2
            if d.geometry_status in ("name_only", "unknown"):
                extra_var += sm.NAME_ONLY_SD ** 2
            oos = self.model.out_of_support(xa)
            zs: dict[str, tuple[float, float]] = {}
            for t in sm.TARGETS:
                z_s, sd_s, _ = ps[t]
                z_a, sd_a, _ = pa[t]
                delta = z_a - z_s
                z = z_s + sm.VARIANT_TRANSFER * delta
                sd = math.sqrt(sd_s ** 2 + (sm.VARIANT_TRANSFER * delta) ** 2 + extra_var + 0.02 * len(oos))
                zs[t] = (z, sd)
            e = expert[v.variant_id]
            pen = repeat_penalty(v.variant_id, recent_calls, game_count)
            rz_mix = RZ_MIX_BONUS if (band == "ten_to_twenty" and v.field_zone == sc.RED_ZONE) else 0.0
            z_succ = zs["success"][0] + e["bonus"] + e["beater"] - pen + rz_mix
            # per-play learning: verified outcomes of THIS variant pull its estimate (shrunk toward the model)
            k, n = variant_stats.get(v.variant_id, (0.0, 0.0))
            resid = 0.0
            resid_raw = 0.0
            if n > 0:
                p0 = ln.sigmoid(z_succ)
                resid_raw = (k - n * p0) / (n + RESIDUAL_K) / max(0.1, p0 * (1 - p0))
                resid = max(-RESIDUAL_CAP, min(RESIDUAL_POS_CAP, resid_raw))
            # t370u: underused siblings get a lift from their OWN verified count only (independent per variant_id)
            novelty = NOVELTY_W * NOVELTY_K / (NOVELTY_K + n) if self.has_siblings.get(v.variant_id) else 0.0
            z_succ += resid + novelty
            p_s = ln.sigmoid(z_succ)
            p_fd = ln.sigmoid(zs["first_down"][0])
            p_ex = ln.sigmoid(zs["explosive"][0])
            p_neg = ln.sigmoid(zs["negative"][0])
            util = sm.UTILITY["success"] * p_s + sm.UTILITY["explosive"] * p_ex + sm.UTILITY["negative"] * p_neg
            yards = self.model.exp_yards(xa, op)
            out.append(Scored(v, p_s, zs["success"][1], p_fd, p_ex, p_neg, yards, util, {
                "existing_model_p": round(op, 4), "model_success_stock": round(ps["success"][2], 4),
                "adjustment_logit_delta": round(sm.VARIANT_TRANSFER * (pa["success"][0] - ps["success"][0]), 4),
                "expert_prior": round(e["bonus"], 4), "beater": round(e["beater"], 4),
                "beats": e["beats"], "gate_top": e["gate_top"], "tagged_beater_for_top": e["tagged_beater_for_top"],
                "coverage_tags_known": e["coverage_tags_known"], "repeat_penalty": round(pen, 4),
                "red_zone_mix_bonus": rz_mix,
                "verified_variant_outcomes": n, "variant_residual": round(resid, 4),
                "variant_residual_uncapped": round(resid_raw, 4), "novelty_bonus": round(novelty, 4),
                "geometry": d.geometry_status, "unrepresented_adjustments": [i + 1 for i in d.unrepresented],
                "stems": [[i + 1, sh, m] for i, sh, m in d.stems],
                "out_of_support": oos[:6]}))
        out.sort(key=lambda s: (-s.utility, s.variant.variant_id))
        return out


def repeat_penalty(variant_id: str, recent: Sequence[str], game_count: Mapping[str, int]) -> float:
    """Logit penalty from THIS variant_id's own earlier calls only (newest first in ``recent``)."""
    reps = sum(REPEAT_DECAY ** i for i, rid in enumerate(list(recent)[:REPEAT_WINDOW]) if rid == variant_id)
    return REPEAT_W * reps + min(GAME_CAP, GAME_W * game_count.get(variant_id, 0))


def _draw(items: Sequence[Scored], best: float, tau: float, rng: Any) -> tuple[Scored, list[float]]:
    w = [math.exp((s.utility - best) / tau) for s in items]
    r, acc, chosen = rng.random() * sum(w), 0.0, items[-1]
    for s, wi in zip(items, w):
        acc += wi
        if r <= acc:
            chosen = s
            break
    return chosen, w


def pick(ranked: Sequence[Scored], seed: str) -> tuple[Scored, dict[str, Any]]:
    """Seeded Boltzmann draw among candidates within MIX_MARGIN utility of the best (reproducible per seed), then the
    sibling mix (owner t370u): among the drawn play's legal variants (same formation + base play) within
    SIBLING_MARGIN of the best of them, a flatter seeded draw picks the variation. The play's own probability is
    unchanged; only which of its near-best variations is called."""
    import random
    ranked = list(ranked)
    best = ranked[0].utility
    pool = [s for s in ranked if best - s.utility <= MIX_MARGIN]
    play, w = _draw(pool, best, MIX_TAU, random.Random(seed))
    tot = sum(w)
    mix: dict[str, Any] = {"rule": f"seeded sample among {len(pool)} within {MIX_MARGIN} utility (tau {MIX_TAU})",
                           "pool": [[s.variant.variant_id, round(wi / tot, 3)] for s, wi in zip(pool, w)]}
    group = [s for s in ranked if s.variant.pair == play.variant.pair]
    g_best = max(s.utility for s in group)
    sibs = [s for s in group if g_best - s.utility <= SIBLING_MARGIN]
    chosen = play
    if len(sibs) > 1:
        chosen, sw = _draw(sibs, g_best, SIBLING_TAU, random.Random(seed + "|sibling"))
        st = sum(sw)
        mix["sibling_mix"] = {
            "rule": f"seeded draw among {len(sibs)} legal sibling variants of {play.variant.formation} "
                    f"{play.variant.base_play} within {SIBLING_MARGIN} utility of the best sibling (tau {SIBLING_TAU})",
            "play_drawn": play.variant.variant_id, "chosen": chosen.variant.variant_id,
            "pool": [[s.variant.variant_id, round(wi / st, 3)] for s, wi in zip(sibs, sw)]}
    else:
        mix["sibling_mix"] = {"rule": "no legal near-best sibling variant", "play_drawn": play.variant.variant_id,
                              "chosen": play.variant.variant_id, "pool": [[play.variant.variant_id, 1.0]]}
    mix["rank"] = ranked.index(chosen) + 1
    return chosen, mix


def _dist_bucket(distance: Any) -> str | None:
    try:
        d = int(distance)
    except (TypeError, ValueError):
        return None
    return next(b for hi, b in RUN_DIST_BUCKETS if d <= hi)


def run_prior_logit(down: Any, distance: Any) -> float:
    """Soft situation prior for calling a run (0 when down/distance is unknown)."""
    try:
        dn = int(down)
    except (TypeError, ValueError):
        return 0.0
    b = _dist_bucket(distance)
    return RUN_PRIOR_LOGIT.get(dn, {}).get(b, 0.0) if b else 0.0


def trailing_late(sit: Any) -> dict[str, Any] | None:
    """Trailing late (owner t370u) from the live score + quarter/two-minute flag, or None (also when no score is set).
    Late = 4th quarter / OT or two-minute (``game_score.classify``)."""
    from cfb_coach.game_score import classify
    try:
        ctx = classify(sit)
    except Exception:  # noqa: BLE001 — a bad score field means "unknown", never a forced mode
        return None
    if ctx is None or ctx.phase != "trail" or not ctx.late or ctx.margin >= 0:
        return None
    return {"margin": ctx.margin, "quarter": ctx.quarter, "two_minute": bool(getattr(sit, "two_minute", False)),
            "strength": ctx.strength, "score": f"{ctx.us}-{ctx.them}"}


def _is_long(distance: Any) -> bool:
    try:
        return int(distance) >= TRAIL_LONG
    except (TypeError, ValueError):
        return False


def run_probability(ranked: Sequence[Scored], *, down: Any, distance: Any, recent_families: Sequence[str] = (),
                    trail: Mapping[str, Any] | None = None) -> tuple[float | None, dict[str, Any]]:
    """P(call a run) for this snap, or None when only one family is legal. Pure function of the scored
    candidates (learned) + the situation prior + this game's recent family calls + trailing late (t370u)."""
    runs = [s for s in ranked if s.variant.is_run]
    passes = [s for s in ranked if not s.variant.is_run]
    if not runs or not passes:
        return None, {}
    gap = max(s.utility for s in runs) - max(s.utility for s in passes)
    prior = run_prior_logit(down, distance)
    rec = list(recent_families)[:RUN_TEND_WINDOW]
    tend = RUN_TEND_W * (sum(1 for f in rec if f != sc.RUN) - sum(1 for f in rec if f == sc.RUN)) / RUN_TEND_WINDOW
    trail_z = -TRAIL_RUN_LOGIT * float(trail["strength"]) if trail else 0.0
    z = prior + RUN_LEARN_W * gap + tend + trail_z
    p = min(RUN_P_MAX, max(RUN_P_MIN, ln.sigmoid(z)))
    info: dict[str, Any] = {"p_run": round(p, 4), "prior_logit": prior, "learned_gap": round(gap, 4),
                            "learned_logit": round(RUN_LEARN_W * gap, 4), "tendency_logit": round(tend, 4),
                            "recent_families": rec}
    if trail:
        rule = f"run logit {trail_z:+.2f}, P(run) <= {TRAIL_RUN_P_MAX}"
        p = min(p, TRAIL_RUN_P_MAX)
        if gap <= 0:
            p, rule = 0.0, "a pass is ranked above the best designed run: pass only"
        elif _is_long(distance):
            p, rule = 0.0, f"distance >= {TRAIL_LONG}: pass only"
        info.update(p_run=round(p, 4), trailing_late={**trail, "logit": round(trail_z, 4), "rule": rule})
    return p, info


def _trailing_red_zone_pass_probability(ranked: Sequence[Scored], *, distance: Any,
                                        trail: Mapping[str, Any]) -> tuple[float | None, dict[str, Any]]:
    runs = [s for s in ranked if s.variant.is_run]
    passes = [s for s in ranked if not s.variant.is_run]
    if not runs or not passes:
        return None, {}
    p = TRAIL_RZ_PASS_BASE + TRAIL_RZ_PASS_STRENGTH * float(trail["strength"])
    try:
        long_ = int(distance) >= 4
    except (TypeError, ValueError):
        long_ = False
    if long_:
        p += TRAIL_RZ_LONG_BONUS
    if max(s.utility for s in passes) >= max(s.utility for s in runs):
        p = max(p, TRAIL_RZ_PASS_TOP)
    p = min(TRAIL_RZ_PASS_MAX, p)
    return p, {"p_pass": round(p, 4), "trailing_late": {**trail, "rule": "inside the 10: red-zone passes mixed in "
                                                         f"(P(pass) {p:.2f}); runs incl. RPO Alert QB Draw stay legal"}}


def pick_mixed(ranked: Sequence[Scored], seed: str, *, down: Any, distance: Any, yardline: Any,
               recent_families: Sequence[str] = (), trail: Mapping[str, Any] | None = None
               ) -> tuple[Scored, dict[str, Any]]:
    """Run-pass mix (owner t348u/t349u) then the near-best draw inside the drawn family.

    Inside the opponent 10 the gate is off (red-zone plays mixed as before) unless trailing late (t370u), when a
    family draw mixes the legal red-zone passes in."""
    import random
    if sc.field_band(yardline) == "inside_10":
        p_pass, info = (_trailing_red_zone_pass_probability(ranked, distance=distance, trail=trail)
                        if trail else (None, {}))
        if p_pass is None:
            top, mix = pick(ranked, seed)
            return top, mix | {"family_gate": "off (inside the opponent 10: red-zone selection unchanged)"}
        fam = sc.PASS if random.Random(seed + "|family").random() < p_pass else sc.RUN
        pool = [s for s in ranked if (s.variant.is_run) == (fam == sc.RUN)]
        top, mix = pick(pool, seed)
        mix["rank"] = list(ranked).index(top) + 1
        return top, mix | {"family_gate": "on (inside the 10, trailing late)", "family": fam, **info}
    p_run, info = run_probability(ranked, down=down, distance=distance, recent_families=recent_families, trail=trail)
    if p_run is None:
        top, mix = pick(ranked, seed)
        return top, mix | {"family_gate": "off (only one family legal)"}
    fam = sc.RUN if random.Random(seed + "|family").random() < p_run else sc.PASS
    pool = [s for s in ranked if (s.variant.is_run) == (fam == sc.RUN)]
    top, mix = pick(pool, seed)
    mix["rank"] = list(ranked).index(top) + 1
    return top, mix | {"family_gate": "on", "family": fam, **info}


_SCORER: dict[tuple[str, str], Scorer] = {}


def load_scorer(art_dir: str | Path, cat: sc.SchemeCatalog | None = None) -> Scorer:
    art = Path(art_dir)
    meta = art / "metadata.json"
    key = (str(art.resolve()), str(meta.stat().st_mtime_ns) if meta.is_file() else "")
    hit = _SCORER.get(key)
    if hit is not None:
        return hit
    model = sm.load(art)
    cat = cat or sc.load_cached()
    want = (model.meta or {}).get("catalog_version")
    if want and want != cat.version():
        raise ValueError(f"model was trained for catalog {want}, the repo catalog is {cat.version()}: retrain")
    s = Scorer(model, cat, dz.GeometrySource())
    _SCORER.clear()
    _SCORER[key] = s
    return s


def warm_up(db: Any) -> float:
    """Load the catalog, model and geometry before the first snap so the first live call meets the latency target.
    Read-only; returns the load time in ms (0 when no artifact is configured)."""
    t0 = time.perf_counter()
    art = md.artifact_dir(db)
    if art is None or not (Path(art) / "model.json").is_file():
        return 0.0
    scorer = load_scorer(art, sc.load_cached())
    for v in scorer.cat.variants:
        dz.adjusted_design(v, scorer.geo)
    return (time.perf_counter() - t0) * 1000.0


def variant_stats(conn: Any, cat: sc.SchemeCatalog) -> dict[str, tuple[float, float]]:
    """variant -> (successes, n) from verified, fully executed variant outcomes (all games incl. this one)."""
    out: dict[str, list[float]] = {}
    evs = [e for e in ev.variant_evidence(conn, cat) if e.usable and e.complete]
    if not evs:
        return {}
    from cfb_coach.outcome import parse_outcome
    for e in evs:
        r = conn.execute("SELECT down, distance, yards, result FROM snaps WHERE id = ?", (e.snap_id,)).fetchone()
        kind = parse_outcome(str(r[3])).kind if r[3] else None
        y = r[2]
        lab = dataset_mod._success_from_yards(side="offense", down=r[0], distance=r[1],
                                              yards=int(y) if y not in (None, "") else None, kind=kind)
        if lab in ("true", "false"):
            s = out.setdefault(e.variant_id, [0.0, 0.0])
            s[0] += 1.0 if lab == "true" else 0.0
            s[1] += 1.0
    return {k: (v[0], v[1]) for k, v in out.items()}


def recent_calls(conn: Any, session_id: str | None, limit: int | None = None) -> list[str]:
    """Variant ids called on this game's closed snaps, newest first (re-calls and previews of the same snap are not
    counted). Each variant is its own id: variants of one base play do not share history."""
    if not session_id or not ev._has_table(conn, "snaps"):
        return []
    rows = conn.execute(
        "SELECT recommended_adjustment_id FROM snaps WHERE session_id = ? AND recommended_adjustment_id LIKE ?"
        " ORDER BY id DESC" + (" LIMIT ?" if limit else ""),
        (session_id, ev.ADJ_PREFIX + "%", *((limit,) if limit else ()))).fetchall()
    return [str(r[0])[len(ev.ADJ_PREFIX):] for r in rows]


def families_of(calls: Sequence[str], cat: sc.SchemeCatalog) -> list[str]:
    """run/pass family of each earlier call (newest first); unknown ids are skipped."""
    out = []
    for vid in calls:
        v = cat.get(vid)
        if v is not None:
            out.append(v.play_family)
    return out


def _belief_for(conn, sit, opponent_id: str, opponent_type: str) -> bl.Belief:
    extras = sit.extras if isinstance(getattr(sit, "extras", None), dict) else {}
    look = sit.coverage_hint if getattr(sit, "coverage_source", "none") == "live" else None
    # A "last" hint is the previous snap's coverage, already a closed snap: not counted twice.
    return bl.live_belief(conn, session_id=extras.get("session_id"), opponent_id=opponent_id,
                          opponent_type=opponent_type, presnap_look=look)


def no_legal_call(reasons: Sequence[str], *, explanation: str, verdicts: Mapping[str, Any] | None = None,
                  shadow: bool = False):
    from cfb_coach.madden.playcaller import MaddenCall
    call = MaddenCall(side="offense", formation="", play=lg.NO_LEGAL, adj_or_macro="",
                      read_or_user="", rationale=f"{lg.NO_LEGAL}: {explanation}")
    call.scheme_variant = {"status": lg.NO_LEGAL, "explanation": explanation, "reasons": list(reasons),
                           "verdicts": {k: v.as_dict() if hasattr(v, "as_dict") else v for k, v in (verdicts or {}).items()},
                           "shadow": shadow}
    call.call_label = "CIVIL_SCHEME_ONLY"
    return call


def _summarize_block(verdicts: Mapping[str, lg.Verdict]) -> list[str]:
    from collections import Counter
    c = Counter(r.split(":")[0].split(" (")[0] for v in verdicts.values() for r in v.reasons[:1])
    return [f"{n} variant(s): {r}" for r, n in c.most_common()]


def explain(top: Scored, belief: bl.Belief, runner_up: Scored | None) -> str:
    p = top.parts
    bits = [f"P(success) {top.p_success:.0%} (±{top.sd:.2f} logit)", f"existing model {p['existing_model_p']:.0%} for the base play"]
    if abs(p["adjustment_logit_delta"]) >= 0.01:
        bits.append(f"adjustments {p['adjustment_logit_delta']:+.2f} logit (transferred from route shapes, unverified)")
    if p["beater"] >= 0.05:
        g = p["gate_top"]
        tag = "tagged beater" if p["tagged_beater_for_top"] else "untagged (neutral share)"
        bits.append(f"belief leans {g[0]} (gate {g[1]:.2f}): {tag} {p['beater']:+.2f}")
    if p["repeat_penalty"]:
        bits.append(f"repeat penalty -{p['repeat_penalty']:.2f}")
    if p["verified_variant_outcomes"]:
        bits.append(f"{int(p['verified_variant_outcomes'])} verified outcomes of this variant ({p['variant_residual']:+.2f}, "
                    f"lift capped at +{RESIDUAL_POS_CAP})")
    if p.get("novelty_bonus"):
        bits.append(f"underused-variant lift {p['novelty_bonus']:+.2f}")
    top_cov = ", ".join(f"{k} {v:.0%}" for k, v in sorted(belief.mix.items(), key=lambda kv: -kv[1])[:3]) or "no coverage history"
    bits.append(f"coverage belief: {top_cov}; pressure {belief.pressure:.0%}")
    if runner_up is not None:
        bits.append(f"next: {runner_up.variant.variant_name} ({runner_up.utility - top.utility:+.3f})")
    return "; ".join(bits)


def mix_explanation(mix: Mapping[str, Any]) -> str:
    """Audit text for the t370u terms: sibling mix and trailing-late aggression (also in the decision log)."""
    bits = []
    sm_ = mix.get("sibling_mix") or {}
    if len(sm_.get("pool") or []) > 1:
        pool = ", ".join(f"{vid.replace('civil_', '')} {w:.0%}" for vid, w in sm_["pool"])
        bits.append(f"sibling mix: {pool}")
    tl = mix.get("trailing_late")
    if tl:
        p = (f"P(run) {mix['p_run']:.0%}" if "p_run" in mix else
             f"P(pass) {mix['p_pass']:.0%}" if "p_pass" in mix else "")
        bits.append(f"trailing late {tl.get('score')} (margin {tl.get('margin')}, Q{tl.get('quarter')}, "
                    f"strength {tl.get('strength')}): {tl.get('rule')}" + (f"; {p}" if p else ""))
    else:
        bits.append("trailing late: off")
    return "; ".join(bits)


def variation_text(name: str, vid: str, adj_lines: Sequence[str]) -> str:
    """Owner instruction t336u: every displayed/logged call names the exact variation AND all its adjustments.
    This text is the call's first line, which is also what the snap log stores as ``our_call``."""
    adjs = "; ".join(adj_lines) if adj_lines else "none (catalog lists no adjustments: stock play)"
    return f"VARIATION: {name} [{vid}] — ADJUSTMENTS: {adjs}"


def build_call(top: Scored, ranked: Sequence[Scored], belief: bl.Belief, *, book: Mapping[str, Any] | None,
               shadow: bool, ledger: Mapping[str, Any] | None = None):
    from cfb_coach.madden.playcaller import MaddenCall
    v = top.variant
    alias = ((ledger or {}).get(v.variant_id) or {}).get("book_play")
    f, q = lg.book_names(book, v.formation, alias or v.base_play)
    formation, play = (f or v.formation), (q or alias or v.base_play)
    from cfb_coach.madden.scheme import adjustments as adjm
    specs = {sp.index: sp for sp in adjm.parse_variant(v)}
    adjs = [f"{a.index + 1}) {a.line()}"
            + (f" [{adjm.stem_label(specs[a.index].stem_shift)}]" if adjm.stem_label(specs[a.index].stem_shift) else "")
            for a in v.adjustments]
    warnings = []
    if not v.name_verified and v.field_zone == "any":
        warnings.append("variant NAME unverified in the source guide")
    if alias:
        warnings.append(f"guide base play '{v.base_play}' is called '{alias}' in your book (owner-confirmed name)")
    elif not lg.stock_has(v.formation, v.base_play)[1] and lg.owner_book_confirmation_applies(sc.load_cached()):
        warnings.append(f"'{v.base_play}' is called by the dataset name: the repo's stock data spells it differently, "
                        "but you confirmed every dataset play is in your book (t337u)")
    if v.group == sc.DESIGNED_RUN:
        warnings.append("DESIGNED RUN (owner 2026-11-11, t349u): stock play from your Buccaneers book, no adjustments; "
                        "not callable inside the opponent 10")
    elif lg.owner_confirmation_applies(sc.load_cached()):
        warnings.append("adjustments owner-confirmed executable (2026-11-10, t333u); no recorded outcome for this "
                        "variant yet")
    if v.field_zone == sc.INSIDE_10:
        warnings.append("RED-ZONE play (owner 2026-11-10, t335u/t338u): inside the opponent 10 only; "
                        "formation mapped from the sheet name + stock book")
    elif v.field_zone == sc.RED_ZONE:
        warnings.append("RED-ZONE play (owner 2026-11-10, t338u, 5WR Tight sheet): inside the 20 only "
                        "(Inside 10 and 10-20); no coverage tags on the sheet")
    if v.field_zone != "any":
        if not v.adjustments:
            warnings.append("the catalog lists NO adjustments for this package play: run it as the stock play")
        if v.verification_notes:
            warnings.append(v.verification_notes)
    if shadow:
        warnings.append("SHADOW: owner verification skipped; do not treat as a live call")
    if v.group == sc.DESIGNED_RUN:
        warnings.append("run: no coverage tags (unknown, not negative)")
    elif not v.coverage_tags_known:
        warnings.append("no coverage tags for this formation (unknown, not negative)")
    directional = [st for st in top.parts.get("stems") or [] if st[2] == "directional"]
    if directional:
        warnings.append("stem(s) " + ", ".join(f"#{st[0]}" for st in directional) + " applied directionally on "
                        "provisional geometry (exact depth unknown; included in the uncertainty)")
    if top.parts.get("unrepresented_adjustments"):
        warnings.append("adjustments not represented on provisional geometry: "
                        + ", ".join(f"#{i}" for i in top.parts["unrepresented_adjustments"]))
    if not v.is_run:
        warnings.append("route geometry is a provisional AceMadden-derived prior, not a verified assignment")
    why = explain(top, belief, next((s for s in ranked if s is not top), None))
    call = MaddenCall(side="offense", formation=formation, play=play,
                      adj_or_macro=variation_text(v.variant_name, v.variant_id, adjs),
                      read_or_user="Read: none provided (no verified read progression)",
                      rationale=f"CIVIL SCHEME ONLY — {why}")
    call.adjustment = {"id": f"{ev.ADJ_PREFIX}{v.variant_id}", "label": f"{v.variant_name} [{v.variant_id}]",
                       "buttons": "no button map — set by hand: " + "; ".join(adjs), "why": why}
    call.scheme_variant = {
        "status": "CALL", "variant_id": v.variant_id, "variant_name": v.variant_name, "formation": v.formation,
        "base_play": v.base_play, "book_play": alias or None, "adjustment_lines": adjs, "field_zone": v.field_zone,
        "play_family": v.play_family, "group": v.group,
        "adjustments": [[a.target, a.action] for a in v.adjustments],
        "p_success": round(top.p_success, 4), "uncertainty_logit_sd": round(top.sd, 4),
        "p_first_down": round(top.p_first_down, 4), "p_explosive": round(top.p_explosive, 4),
        "p_negative": round(top.p_negative, 4), "exp_yards": top.exp_yards, "utility": round(top.utility, 4),
        "warnings": warnings, "explanation": why, "shadow": shadow, "parts": top.parts,
        "ranking": [s.as_dict() for s in ranked[:8]], "belief": belief.as_log(),
    }
    call.call_label = "CIVIL_SCHEME_ONLY"
    return call


def select(scorer: Scorer, cat: sc.SchemeCatalog, conn: Any, *, db: Any, sit: Any, opponent_id: str,
           book: Mapping[str, Any] | None, shadow: bool = False, belief: bl.Belief | None = None):
    from cfb_coach.opponents import is_cpu_opponent
    legal, verdicts = lg.legal_set(cat, db=db, book=book, require_verified=not shadow, yardline=sit.yardline)
    if not legal:
        return no_legal_call(_summarize_block(verdicts), verdicts=verdicts, shadow=shadow,
                             explanation="no catalog variant passes the legal filter: " + " | ".join(_summarize_block(verdicts))), None
    opp_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
    belief = belief or _belief_for(conn, sit, opponent_id, opp_type)
    session_id = (sit.extras or {}).get("session_id") if isinstance(getattr(sit, "extras", None), dict) else None
    game = recent_calls(conn, session_id)
    ranked = scorer.score(legal, down=sit.down, distance=sit.distance, yardline=sit.yardline,
                          opponent_id=opponent_id, opponent_type=opp_type, belief=belief,
                          recent_calls=game[:REPEAT_WINDOW], game_calls=game, variant_stats=variant_stats(conn, cat))
    seed = f"{session_id}|{len(game)}|{sit.down}|{sit.distance}|{sit.yardline}"
    trail = trailing_late(sit)
    top, mix = pick_mixed(ranked, seed, down=sit.down, distance=sit.distance, yardline=sit.yardline,
                          recent_families=families_of(game, cat), trail=trail)
    mix["trailing_late"] = mix.get("trailing_late") or ({**trail, "rule": "only one family legal"} if trail else None)
    call = build_call(top, ranked, belief, book=book, shadow=shadow, ledger=lg.load_verified(db))
    call.scheme_variant["mix"] = mix
    extra = mix_explanation(mix)
    if extra:
        call.scheme_variant["explanation"] = f"{call.scheme_variant['explanation']}; {extra}"
        call.rationale = f"{call.rationale}; {extra}"
    call.scheme_variant["yardline"] = sit.yardline     # re-validation applies the same field-position gate
    call.scheme_variant["candidates"] = [v.variant_id for v in legal]   # the legal set that was scored
    return call, ranked


def call_civil(sit: Any, opponent_id: str, db: Any, book: Mapping[str, Any] | None, *, shadow: bool = False,
               log: bool = True):
    """Entry point from ``playcaller.make_call``. Never raises into the caller; never falls back to another play."""
    t0 = time.perf_counter()
    try:
        cat = sc.load_cached()
        art = md.artifact_dir(db)
        if art is None or not (Path(art) / "model.json").is_file():
            return no_legal_call(["no trained civil-scheme model"], explanation="civil_scheme_only is on but no trained "
                                 "model artifact was found (run `ml civil-scheme train`)", shadow=shadow)
        scorer = load_scorer(art, cat)
        want = md.load_state(db).get("model_sha256")
        if want and want != (scorer.model.meta or {}).get("model_sha256"):
            return no_legal_call(["model mismatch"], explanation="the loaded model is not the one `enable` recorded",
                                 shadow=shadow)
        applied = None
        if db is not None:
            from cfb_coach.madden import playbook
            applied = playbook.load_books(db).get("offense")
        call, ranked = select(scorer, cat, db.conn, db=db, sit=sit, opponent_id=opponent_id, book=applied, shadow=shadow)
        lg.revalidate_call(db, call, cat, yardline=sit.yardline)
    except Exception as exc:  # noqa: BLE001 — fail closed with the reason, never a hidden fallback
        call, ranked = no_legal_call([f"error: {exc.__class__.__name__}"], explanation=f"civil coordinator error: {exc}",
                                     shadow=shadow), None
    ms = (time.perf_counter() - t0) * 1000.0
    call.scheme_variant["latency_ms"] = round(ms, 2)
    call.scheme_variant["latency_target_ms"] = LATENCY_TARGET_MS
    if log and db is not None and not getattr(db, "read_only", False):
        try:
            sv = call.scheme_variant
            ev.log_decision(db.conn, {
                "session_id": (sit.extras or {}).get("session_id") if isinstance(getattr(sit, "extras", None), dict) else None,
                "opponent_id": opponent_id, "mode": "shadow" if shadow else md.CIVIL, "status": sv["status"],
                "variant_id": sv.get("variant_id"), "formation": sv.get("formation"), "base_play": sv.get("base_play"),
                "adjustments": sv.get("adjustments"), "down": sit.down, "distance": sit.distance, "yardline": sit.yardline,
                "situation_raw": getattr(sit, "raw", None), "belief": sv.get("belief"),
                "prediction": {k: sv.get(k) for k in ("p_success", "uncertainty_logit_sd", "p_first_down", "p_explosive",
                                                       "p_negative", "exp_yards", "utility", "latency_ms")}
                | ({"variation": variation_text(sv["variant_name"], sv["variant_id"], sv.get("adjustment_lines") or []),
                    "variant_name": sv["variant_name"], "field_zone": sv.get("field_zone"),
                    "mix": sv.get("mix")}          # t370u: sibling mix + trailing-late terms for later audits
                   if sv.get("status") == "CALL" else {}),
                "ranking": [[r["variant_id"], r["utility"]] for r in sv.get("ranking") or []],
                "model_version": (md.load_state(db) or {}).get("model_sha256"), "catalog_version": sc.load_cached().version(),
                "explanation": sv.get("explanation"),
            })
        except Exception:  # noqa: BLE001 — logging must not block the call
            pass
    return call


def format_lines(call: Any) -> list[str]:
    sv = getattr(call, "scheme_variant", None) or {}
    if sv.get("status") == lg.NO_LEGAL:
        return [f"{lg.NO_LEGAL}", f"  why: {sv.get('explanation')}"]
    lines = [f"FORMATION: {sv['formation']}", f"BASE PLAY: {sv['base_play']}",
             f"VARIATION: {sv['variant_name']}  [{sv['variant_id']}]", "ADJUSTMENTS (all, in order):"]
    lines += [f"  {line}" for line in sv.get("adjustment_lines") or
              [f"{i + 1}) {t}: {a}" for i, (t, a) in enumerate(sv["adjustments"])]
              or ["none: the catalog lists no adjustments for this play (stock play)"]]
    lines.append(f"P(success) {sv['p_success']:.0%}  uncertainty ±{sv['uncertainty_logit_sd']:.2f} logit  "
                 f"1st down {sv['p_first_down']:.0%}  explosive {sv['p_explosive']:.0%}  sack/TO/loss {sv['p_negative']:.0%}"
                 + (f"  exp yards {sv['exp_yards']:.1f}" if sv.get("exp_yards") is not None else ""))
    lines.append("WARNINGS: " + " | ".join(sv["warnings"]))
    lines.append("WHY: " + sv["explanation"])
    lines.append("READS: none shown (no verified read progression)")
    return lines
