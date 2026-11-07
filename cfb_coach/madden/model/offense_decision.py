"""Joint offensive play + adjustment decision (research / CPU-pilot engine).

Rules this module enforces:
* the learned model (structural artifact over the existing coordinator) scores
  and ranks every eligible installed play; no heuristic override;
* an adjustment is chosen only if (a) Madden's executable registry validates it
  (legality, conflicts, play clock, installed/confirmed status), (b) the modeled
  gain clears a margin AND its lower confidence bound is positive;
* the output is always definitive: a specific plan, or ``NO ADJUSTMENT``;
* only pre-snap inputs are accepted (``assert_pre_snap``).

Adjustment scores extrapolate the structural weights to plays the model never
saw hot-routed; the decision carries ``basis`` saying so.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from dataclasses import replace as _dc_replace

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import football_priors as fpri
from cfb_coach.madden.model import offense_diversity as div
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_taxonomy as tx
from cfb_coach.madden.model.strategic_context import assert_pre_snap

DECISION_VERSION = "offense_decision.v1"
NO_ADJUSTMENT = "NO ADJUSTMENT"
BASIS = "structural_extrapolation_unvalidated"
PRESNAP_FIELDS = frozenset({"down", "distance", "yardline", "opponent_id", "opponent_type", "mix",
                            "play_clock_remaining", "coach_mode", "score_diff", "quarter", "clock"})
INDIFFERENCE_BAND = 0.012     # same band the existing joint decision uses
DEFAULT_MARGIN = 0.02
DEFAULT_Z = 1.0
DEFAULT_BUDGET_MS = 100.0   # internal; the live budget is 150 ms (schema.ML_LATENCY_BUDGET_MS)


@dataclass(frozen=True)
class Situation:
    down: int | None = None
    distance: int | None = None
    yardline: int | None = None
    opponent_id: str | None = None
    opponent_type: str | None = None
    mix: Mapping[str, float] | None = None       # PRE-snap opponent coverage mix (earlier snaps only)
    play_clock_remaining: float | None = None
    coach_mode: bool = False

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "Situation":
        assert_pre_snap(m.keys())
        extra = set(m) - PRESNAP_FIELDS
        if extra:
            raise ValueError(f"unknown situation fields: {sorted(extra)}")
        return cls(**{k: m[k] for k in m if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class MacroEffect:
    """User-verified effect of an installed offensive Custom Adjustment: exact hot-route-equivalent changes."""
    name: str
    changes: Mapping[str, str]       # slot -> Madden route option
    verified: bool
    source: str


@dataclass
class PlanScore:
    actions: tuple[ex.PlannedAction, ...]
    label: str
    probability: float
    gain_logit: float
    gain_se: float
    gain_prob: float
    errors: list[str]
    executable_now: bool
    comparison: dict[str, Any]
    accepted: bool = False
    reject_reason: str = ""


@dataclass
class Decision:
    formation: str
    play: str
    adjustment: str
    probability: float
    stock_probability: float
    interval: tuple[float, float]
    actions: tuple[ex.PlannedAction, ...] = ()
    executable_now: bool = True
    final_structure: pf.PlayStructure | None = None
    ranking: list[dict[str, Any]] = field(default_factory=list)
    plans: list[PlanScore] = field(default_factory=list)
    mix_source: str = ""
    latency_ms: float = 0.0
    truncated: bool = False
    evidence: str = "catalog_only"
    basis: str = "stock_play"
    notes: list[str] = field(default_factory=list)
    matchup: dict[str, Any] | None = None
    football_priors: bool = False

    def as_log(self) -> dict[str, Any]:
        return {
            "version": DECISION_VERSION, "formation": self.formation, "play": self.play,
            "adjustment": self.adjustment, "probability": round(self.probability, 4),
            "stock_probability": round(self.stock_probability, 4),
            "interval90": [round(x, 4) for x in self.interval],
            "actions": [{"id": a.action_id, "params": dict(a.params)} for a in self.actions],
            "executable_now": self.executable_now, "mix_source": self.mix_source,
            "latency_ms": round(self.latency_ms, 2), "truncated": self.truncated,
            "evidence": self.evidence, "basis": self.basis, "notes": self.notes,
            "football_priors": self.football_priors, "matchup": (None if not self.matchup else {"edge": round(self.matchup["edge"], 4), "cells": self.matchup["cells"]}),
            "ranking": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in self.ranking[:8]],
            "considered_plans": [{"label": p.label, "p": round(p.probability, 4), "gain": round(p.gain_prob, 4),
                                  "accepted": p.accepted, "reject": p.reject_reason} for p in self.plans[:12]],
        }


BEAM_WIDTH = 6      # search breadth (runtime bound), NOT a limit on how many adjustments a play may carry
POOL_SIZE = 12      # single actions eligible to be combined


def _key(a: ex.PlannedAction) -> tuple:
    return (a.action_id, tuple(sorted(dict(a.params).items())))


def label_for(actions: Sequence[ex.PlannedAction]) -> str:
    if not actions:
        return NO_ADJUSTMENT
    parts = []
    for a in actions:
        if a.action_id == "off.hot_route":
            parts.append(f"HOT ROUTE {a.params['target']} -> {a.params['route']}")
        elif a.action_id.startswith("off.saved."):
            parts.append("MACRO " + a.action_id.removeprefix("off.saved."))
        else:
            parts.append(a.action_id + ("".join(f" {k}={v}" for k, v in sorted(a.params.items()))))
    return " + ".join(parts)


def _play_type(s: pf.PlayStructure) -> str:
    return "pass" if s.hint.get("play_type") in ("pass", "play_action") or s.n_known >= 2 else "run"


def _evidence(s: pf.PlayStructure) -> str:
    if s.n_known:
        return f"partial_play_jobs({s.n_known} sourced routes)"
    return "catalog_only"


NON_ROUTE_ACTIONS = ("off.pass_protection", "off.chip_block", "off.motion", "off.fake_snap")


def _apply_actions(s: pf.PlayStructure, actions: Sequence[ex.PlannedAction],
                   macro_effects: Mapping[str, MacroEffect]) -> pf.PlayStructure:
    """Ordered transformations of the VERIFIED stock structure. Two actions may never change the same receiver, and an
    original assignment that is unknown is never invented (``apply_hot_route`` refuses). Reads are recomputed from the result:
    ``apply_hot_route`` clears the stock read order, so a stale progression cannot leak through."""
    touched: dict[str, str] = {}
    for a in actions:
        if a.action_id == "off.hot_route":
            changes = {a.params["target"]: a.params["route"]}
        elif a.action_id in NON_ROUTE_ACTIONS:
            continue                                   # protection / blocking / motion: no receiver route changes
        else:
            eff = macro_effects.get(a.action_id.removeprefix("off.saved."))
            if eff is None or not eff.verified:
                raise pf.StructureError("macro effect is not user-verified; cannot represent it")
            changes = dict(eff.changes)
        for slot in changes:
            if slot in touched:
                raise pf.StructureError(f"conflicting assignments: {slot} is changed by both {touched[slot]} and {a.action_id}")
            touched[slot] = a.action_id
        s = pf.apply_route_changes(s, changes)
    return s


def decide(artifact: sm.StructuralArtifact, situation: Situation,
           candidates: Sequence[tuple[str, str]], *,
           loadout: ex.Loadout | None = None, confirmations: Mapping[str, Any] | None = None,
           macro_effects: Mapping[str, MacroEffect] | None = None, mode: str = "live",
           margin: float = DEFAULT_MARGIN, z: float = DEFAULT_Z, budget_ms: float = DEFAULT_BUDGET_MS,
           adjust_top: int = 5, max_pairs: int | None = None, allow_adjustments: bool = True,
           recent_calls: Sequence[Any] | None = None, beam_width: int = BEAM_WIDTH, pool_size: int = POOL_SIZE,
           extra_actions: Sequence[ex.PlannedAction] = (), non_route_gain: Mapping[str, tuple[float, float]] | None = None,
           priors: bool = False, learned_cells: Mapping[tuple[str, str], float] | None = None) -> Decision:
    """``mode='live'`` requires confirmed Madden executability; ``'research'`` scores but marks not executable."""
    if mode not in ("live", "research"):
        raise ValueError(mode)
    if not candidates:
        raise ValueError("no eligible candidates")
    t0 = time.perf_counter()
    macro_effects = macro_effects or {}
    kw = dict(down=situation.down, distance=situation.distance, yardline=situation.yardline,
              opponent_id=situation.opponent_id, opponent_type=situation.opponent_type)
    mix = situation.mix
    ranked: list[tuple[pf.PlayStructure, sm.Score]] = []
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for form, play in candidates:
        s = sm.structure_any(form, play)
        sc = sm.score(artifact, s, mix=mix, **kw)
        if priors:
            # football prior: expected logit lift of this play's concepts vs the opponent's coverage MIXTURE, added to the learned score
            e = fpri.matchup_edge(s, mix, learned=learned_cells)
            edges[(form, play)] = e
            lg = sm._logit(sc.probability) + e["edge"]
            sc = _dc_replace(sc, probability=sm._sigmoid(lg), logit_se=(sc.logit_se ** 2 + e["se"] ** 2) ** 0.5)
        ranked.append((s, sc))
    ranked.sort(key=lambda t: (-t[1].probability, t[1].logit_se, t[0].formation, t[0].play))
    if recent_calls:
        # Existing variety logic: only reorders plays inside the indifference band (+ extra band when the
        # recent calls are concentrated). A clearly better play is never moved.
        stats = div.concentration_stats(recent_calls)
        top = ranked[0][1]
        band = INDIFFERENCE_BAND + div.extra_band_width(stats, {"uncertainty": min(1.0, top.logit_se)})
        near = [t for t in ranked if top.probability - t[1].probability <= band]
        if len(near) > 1:
            def w(t: tuple[pf.PlayStructure, sm.Score]) -> float:
                cid = (t[0].hint.get("concepts") or [None])[0]
                return div.sampling_weight(base_weight=t[1].probability, formation=t[0].formation,
                                           concept_id=cid, stats=stats, apply_concentration=True)
            near.sort(key=lambda t: (-w(t), t[0].formation, t[0].play))
            ranked = near + [t for t in ranked if t not in near]
    best_s, best = ranked[0]
    ranking = [{"formation": s.formation, "play": s.play, "probability": sc.probability,
                "base_probability": sc.base_probability, "residual_logit": sc.residual_logit,
                "logit_se": sc.logit_se, "rank": i + 1} for i, (s, sc) in enumerate(ranked)]
    decision = Decision(formation=best_s.formation, play=best_s.play, adjustment=NO_ADJUSTMENT,
                        probability=best.probability, stock_probability=best.probability,
                        interval=best.interval, final_structure=best_s, ranking=ranking,
                        mix_source=best.mix_source, evidence=_evidence(best_s))
    decision.football_priors = bool(priors)
    if priors:
        decision.matchup = edges.get((decision.formation, decision.play))
    if not allow_adjustments:
        decision.latency_ms = (time.perf_counter() - t0) * 1000
        return decision

    plans: list[PlanScore] = []
    truncated = False
    for s, sc in ranked[:adjust_top]:
        if _play_type(s) != "pass" or not s.n_known:
            continue
        targets = pf.eligible_hot_route_targets(s)
        ctx = ex.PlayContext(side="offense", formation=s.formation, play=s.play, play_type="pass",
                             eligible_receivers=s.skill_slots, play_clock_remaining=situation.play_clock_remaining,
                             coach_mode=situation.coach_mode)

        scorer = sm.DeltaScorer(artifact, s, mix=mix, opponent_id=situation.opponent_id,
                                sit={"down": situation.down, "distance": situation.distance, "yardline": situation.yardline})
        base_logit = sm._logit(sc.probability)

        def consider(actions: tuple[ex.PlannedAction, ...]) -> PlanScore | None:
            try:
                mod = _apply_actions(s, actions, macro_effects)
            except pf.StructureError as exc:
                return None if "already runs" in str(exc) else PlanScore(
                    actions, label_for(actions), sc.probability, 0.0, 0.0, 0.0, [str(exc)], False, {},
                    reject_reason=str(exc))
            gain, se, outside = scorer.score(mod)
            if priors:                                  # the transformed play has different concepts: re-evaluate its matchup, don't reuse the stock one
                e_mod = fpri.matchup_edge(mod, mix, learned=learned_cells)
                gain += e_mod["edge"] - edges[(s.formation, s.play)]["edge"]
                se = (se * se + e_mod["se"] ** 2) ** 0.5
            for a_ in actions:                         # evidence-backed gains of non-route actions (protection etc.); none by default
                g_, se_ = (non_route_gain or {}).get(a_.action_id, fpri.protection_edge(mix) if priors and a_.action_id == "off.pass_protection" else (0.0, 0.0))
                if a_.action_id in NON_ROUTE_ACTIONS:
                    gain, se = gain + g_, (se * se + se_ * se_) ** 0.5
            p = sm._sigmoid(base_logit + gain)
            ps = PlanScore(actions, f"{s.play}: {label_for(actions)}", p, gain, se, p - sc.probability, [], False, {})
            # Cheap numeric screens first; Madden legality only for survivors.
            if outside:
                ps.reject_reason = "modified play outside training support: " + ", ".join(outside[:3])
            elif ps.gain_prob < margin:
                ps.reject_reason = f"modeled gain {ps.gain_prob:+.3f} below margin {margin}"
            elif gain - z * se <= 0:
                ps.reject_reason = "gain lower confidence bound not above zero"
            else:
                errs = ex.validate_plan(actions, ctx, loadout=loadout, confirmations=confirmations,
                                        require_live=(mode == "live"))
                ps.errors = errs
                ps.executable_now = not ex.validate_plan(actions, ctx, loadout=loadout,
                                                         confirmations=confirmations, require_live=True) \
                    if mode != "live" else not errs
                ps.comparison = pf.compare_structures(s, mod)
                if errs:
                    ps.reject_reason = errs[0]
                else:
                    ps.accepted = True
            plans.append(ps)
            return ps

        singles: list[PlanScore] = []
        for slot, option in itertools.product(targets, tx.HOT_ROUTE_OPTIONS):
            if (time.perf_counter() - t0) * 1000 > budget_ms:
                truncated = True
                break
            ps = consider((ex.PlannedAction("off.hot_route", {"target": slot, "route": option}),))
            if ps is not None and (ps.accepted or ps.errors):   # errors are only set after the numeric screens passed
                singles.append(ps)
        if loadout is not None:
            for entry in loadout.entries:
                eff = macro_effects.get(entry.name)
                if entry.side != "offense" or eff is None or not eff.verified or loadout.usable("offense", entry.name) is None:
                    continue
                ps = consider((ex.PlannedAction(f"off.saved.{entry.name}"),))
                if ps is not None and (ps.accepted or ps.errors):
                    singles.append(ps)
        for pa in extra_actions:
            ps = consider((pa,))
            if ps is not None and (ps.accepted or ps.errors):
                singles.append(ps)
        singles.sort(key=lambda p: -p.gain_logit)
        # Bounded beam search over COMBINATIONS. The only limits are legality (validate_plan: eligible players, exclusive/conflicting
        # controls, play clock), conflicting receiver assignments, evidence (a combination must beat its own parent on the lower
        # confidence bound), and the latency budget. There is no cap on the number of adjustments in a plan.
        # Dominance pruning: per receiver keep only its best two route options, so the pool spans DIFFERENT receivers
        # (otherwise the top singles are all alternatives for one receiver and can never be combined).
        per_target: dict[Any, int] = {}
        pool = []
        for p_ in singles:
            tgt = p_.actions[0].params.get("target", p_.actions[0].action_id)
            if per_target.get(tgt, 0) >= 2:
                continue
            per_target[tgt] = per_target.get(tgt, 0) + 1
            pool.append(p_)
        pool = pool[:pool_size]
        beam = [p for p in singles[:beam_width] if p.accepted or not p.errors]
        seen = {frozenset(_key(a) for a in p.actions) for p in singles}
        while beam and not truncated:
            nxt: list[PlanScore] = []
            for parent in beam:
                for single in pool:
                    if (time.perf_counter() - t0) * 1000 > budget_ms:
                        truncated = True
                        break
                    act = single.actions[0]
                    if any(_key(act) == _key(x) for x in parent.actions):
                        continue
                    if act.action_id == "off.hot_route" and any(x.action_id == "off.hot_route" and x.params.get("target") == act.params.get("target")
                                                                 for x in parent.actions):
                        continue                          # one command per receiver: cheap pre-filter, the full check is _apply_actions
                    combo = parent.actions + single.actions
                    sig = frozenset(_key(x) for x in combo)
                    if sig in seen:
                        continue                          # dedupe: same set in a different order
                    seen.add(sig)
                    child = consider(combo)
                    # dominance: keep only extensions that raise the lower confidence bound over the parent
                    if child is not None and not child.reject_reason.startswith("modified play outside") \
                            and (child.gain_logit - z * child.gain_se) > (parent.gain_logit - z * parent.gain_se) + 1e-9 \
                            and (child.accepted or not child.errors):
                        nxt.append(child)
                if truncated:
                    break
            nxt.sort(key=lambda p: -(p.gain_logit - z * p.gain_se))
            beam = nxt[:beam_width]

    # Best accepted plan must beat the best STOCK play by the margin, not just its own play.
    accepted = [p for p in plans if p.accepted and p.probability >= best.probability + margin]
    decision.plans = sorted(plans, key=lambda p: -p.probability)
    decision.truncated = truncated
    if accepted:
        win = max(accepted, key=lambda p: (p.probability, -p.gain_se, p.label))
        play_name = win.label.split(": ", 1)[0]
        s = next(st for st, _ in ranked if st.play == play_name)
        decision.formation, decision.play = s.formation, s.play
        decision.adjustment = label_for(win.actions)
        decision.actions = win.actions
        decision.probability = win.probability
        decision.stock_probability = next(sc.probability for st, sc in ranked if st.play == play_name)
        lg = sm._logit(win.probability)
        decision.interval = (sm._sigmoid(lg - 1.64 * win.gain_se), sm._sigmoid(lg + 1.64 * win.gain_se))
        decision.executable_now = win.executable_now
        decision.final_structure = _apply_actions(s, win.actions, macro_effects)
        if priors:
            decision.matchup = fpri.matchup_edge(decision.final_structure, mix, learned=learned_cells)
        decision.evidence = _evidence(s)
        decision.basis = BASIS
        decision.notes.append("adjustment scored by structural extrapolation; training never saw hot-routed plays")
    if truncated:
        decision.notes.append("adjustment search truncated by the latency budget")
    if priors and decision.matchup is None:
        decision.matchup = edges.get((decision.formation, decision.play))
    decision.football_priors = bool(priors)
    decision.latency_ms = (time.perf_counter() - t0) * 1000
    return decision
