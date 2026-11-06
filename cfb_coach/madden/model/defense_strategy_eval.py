"""Offline/shadow assessment of legal defensive calls in context. Not read by the live caller.

For each candidate it reports the expected threat, what the call is (coverage
family, front, pressure), how exposed it is to run, pass and explosive plays,
conversion/touchdown risk, opponent-specific evidence, uncertainty, and why it
ranks where it does. It reuses the existing pieces instead of re-deciding:

* legality: ``defense_v1.light_box_allowed`` (red zone, short yardage)
* coverage family: ``defense_select.call_family``
* concept counters: ``defense_select.COUNTERS`` (a broad football prior)
* opponent adjustment: ``opponent_tendencies.defense_success_delta``
* conversion estimate: ``defense_v1.offense_success_probability`` when an
  artifact is supplied (trained on logged, unverified calls)

Every component is tagged with its evidence class:
``measured`` (outcomes of confirmed executions), ``model_unverified``,
``observed_opponent`` (what the opponent was seen doing), ``prior`` (coded
football knowledge) or ``hypothesis``. Priors and hypotheses carry small
weights; nothing is labeled proven.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.strategic_context import PreSnapContext

VERSION = "defense_strategy_eval.v1"
THREAT_FAMILIES = ("run", "rpo", "vert", "flood", "cross", "stack", "scram")
PASS_FAMILIES = ("vert", "flood", "cross", "stack")
RECENCY_DECAY = 0.85
HANDOFF = 12.0
MIN_MEASURED = 20

EVIDENCE_WEIGHT = {"measured": 1.0, "model_unverified": 0.6, "observed_opponent": 0.6,
                   "prior": 0.35, "hypothesis": 0.15}

# Situation priors for the threat when nothing is known about the opponent (labeled ``prior``).
SITUATION_THREAT_PRIOR = {
    "short": {"run": 0.55, "rpo": 0.1, "cross": 0.15, "flood": 0.1, "vert": 0.05, "stack": 0.03, "scram": 0.02},
    "long": {"run": 0.15, "rpo": 0.05, "cross": 0.25, "flood": 0.25, "vert": 0.22, "stack": 0.05, "scram": 0.03},
    "goal_line": {"run": 0.6, "rpo": 0.1, "cross": 0.12, "flood": 0.1, "vert": 0.03, "stack": 0.03, "scram": 0.02},
    "red_zone": {"run": 0.4, "rpo": 0.1, "cross": 0.2, "flood": 0.17, "vert": 0.06, "stack": 0.05, "scram": 0.02},
    "two_minute": {"run": 0.1, "rpo": 0.05, "cross": 0.25, "flood": 0.28, "vert": 0.25, "stack": 0.05, "scram": 0.02},
    "early": {"run": 0.45, "rpo": 0.1, "cross": 0.15, "flood": 0.13, "vert": 0.1, "stack": 0.05, "scram": 0.02},
}

# How exposed a coverage family is to an explosive pass (prior/hypothesis, not measured).
EXPLOSIVE_EXPOSURE = {"two_high": 0.15, "cover2": 0.3, "single_high": 0.4, "man": 0.55, "pressure": 0.7}
FRONT_RUN_ADJ = {"heavy": 0.25, "light": -0.35, "base": 0.0}


@dataclass(frozen=True)
class Component:
    name: str
    value: float
    evidence: str
    note: str = ""


@dataclass
class CandidateAssessment:
    formation: str
    play: str
    coverage_family: str | None
    front: str
    pressure: bool
    legal: bool
    illegal_reason: str = ""
    assessed: bool = True
    run_answer: float = 0.0
    pass_answer: float = 0.0
    explosive_exposure: float = 0.0
    conversion_risk: float | None = None
    td_risk: float | None = None
    opponent_delta: float = 0.0
    score: float = 0.0
    components: list[Component] = field(default_factory=list)
    rank: int = 0
    versus_next: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DefensiveAssessment:
    version: str
    situation_key: str
    threat: dict[str, float]
    threat_sources: dict[str, Any]
    threat_entropy: float
    uncertainty: list[str]
    candidates: list[CandidateAssessment]
    measured_basis: dict[str, Any]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def situation_key(ctx: PreSnapContext) -> str:
    if ctx.goal_line.value:
        return "goal_line"
    if ctx.red_zone.value:
        return "red_zone"
    if ctx.two_minute.value:
        return "two_minute"
    if ctx.short_yardage.value:
        return "short"
    if ctx.long_yardage.value:
        return "long"
    return "early"


def threat_distribution(
    ctx: PreSnapContext,
    *,
    league_prior: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Expected opponent concept family: league/situation prior, opponent profile, then this game.

    This game's observed concepts weigh n/(n+12), recent snaps counting more.
    The opponent profile (if any) sits between prior and game evidence.
    """
    from cfb_coach.madden.situation import concept_family

    key = situation_key(ctx)
    base = dict(league_prior or SITUATION_THREAT_PRIOR[key])
    base_source = "league_observed" if league_prior else "prior"
    prof = {k: float(v) for k, v in (ctx.opponent.concept_shares or {}).items() if k in THREAT_FAMILIES}
    prof_n = ctx.opponent.own_plays
    if prof and prof_n > 0:
        w = prof_n / (prof_n + HANDOFF)
        base = {f: (1 - w) * base.get(f, 0.0) + w * prof.get(f, 0.0) for f in THREAT_FAMILIES}
    fams = [concept_family(c) for c in ctx.history.opponent_observed_concepts]
    fams = [f for f in fams if f in THREAT_FAMILIES]
    weights = [RECENCY_DECAY ** (len(fams) - 1 - i) for i in range(len(fams))]
    eff_n = sum(weights)
    game: dict[str, float] = {}
    for f, wt in zip(fams, weights):
        game[f] = game.get(f, 0.0) + wt
    w_game = eff_n / (eff_n + HANDOFF) if eff_n else 0.0
    threat = {}
    for f in THREAT_FAMILIES:
        g = game.get(f, 0.0) / eff_n if eff_n else 0.0
        threat[f] = (1 - w_game) * base.get(f, 0.0) + w_game * g
    total = sum(threat.values()) or 1.0
    threat = {f: v / total for f, v in threat.items()}
    sources = {"situation": key, "base": base_source, "profile_plays": prof_n,
               "game_plays": len(fams), "game_weight": round(w_game, 3)}
    return threat, sources


def _front(cand: Mapping[str, Any]) -> str:
    from cfb_coach.madden.model.defense_v1 import _is_heavy_front, _is_light_box

    if _is_light_box(cand):
        return "light"
    if _is_heavy_front(cand):
        return "heavy"
    return "base"


def _legal(ctx: PreSnapContext, sit: Any, front: str, run_share: float | None) -> tuple[bool, str]:
    from cfb_coach.madden.model.defense_v1 import light_box_allowed

    if front == "light" and sit is not None and not light_box_allowed(sit, run_share):
        return False, "light box not allowed here (red zone, short yardage, or run-leaning opponent)"
    return True, ""


def assess_candidates(
    ctx: PreSnapContext,
    candidates: Sequence[Mapping[str, Any]],
    *,
    sit: Any = None,
    artifact: Mapping[str, Any] | None = None,
    opponent_profile: Mapping[str, Any] | None = None,
    league_prior: Mapping[str, float] | None = None,
    measured: Mapping[str, Mapping[str, Any]] | None = None,
) -> DefensiveAssessment:
    """Rank legal defensive calls with a structured explanation. Offline only."""
    from cfb_coach.madden.defense_select import COUNTERS, call_family
    from cfb_coach.madden.model import opponent_tendencies as ot

    started = time.perf_counter()
    threat, sources = threat_distribution(ctx, league_prior=league_prior)
    p_run = threat["run"] + 0.5 * threat["rpo"]
    p_pass = 1.0 - p_run
    pass_total = sum(threat[f] for f in PASS_FAMILIES) or 1.0
    key = sources["situation"]
    leading_late = bool(ctx.two_minute.value) and (ctx.score_diff.value or 0) > 0
    explosive_weight = 1.5 if leading_late else 1.0
    measured = dict(measured or {})
    out: list[CandidateAssessment] = []
    for cand in candidates:
        form, play = str(cand.get("formation") or ""), str(cand.get("play") or "")
        fam = cand.get("coverage_family") or call_family(play)
        front = _front(cand)
        legal, why = _legal(ctx, sit, front, p_run)
        comps: list[Component] = []
        run_answer = COUNTERS["run"].get(fam or "", 0.0) + FRONT_RUN_ADJ[front]
        pass_answer = sum(threat[f] / pass_total * COUNTERS[f].get(fam or "", 0.0) for f in PASS_FAMILIES)
        rpo_scram = threat["rpo"] * COUNTERS["rpo"].get(fam or "", 0.0) + threat["scram"] * COUNTERS["scram"].get(fam or "", 0.0)
        exposure = EXPLOSIVE_EXPOSURE.get(fam or "", 0.45) * (threat["vert"] + 0.5 * threat["flood"]) / 0.35
        comps.append(Component("run_answer", p_run * run_answer, "prior",
                               f"P(run)={p_run:.2f} x counter {run_answer:+.2f}"))
        comps.append(Component("pass_answer", p_pass * pass_answer, "prior",
                               f"P(pass)={p_pass:.2f} x counter {pass_answer:+.2f}"))
        comps.append(Component("rpo_scramble_answer", rpo_scram, "prior"))
        comps.append(Component("explosive_exposure", -explosive_weight * 0.3 * exposure, "hypothesis",
                               "deep-shot exposure by coverage shell" + (" (leading late: weighted up)" if leading_late else "")))
        conv = None
        if artifact is not None and sit is not None and cand.get("family") and cand.get("package"):
            from cfb_coach.madden.model.defense_v1 import offense_success_probability

            try:
                conv = offense_success_probability(artifact, sit, cand)
                comps.append(Component("model_offense_success", -(conv - 0.5), "model_unverified",
                                       "defense model trained on logged calls whose execution is unconfirmed"))
            except (KeyError, TypeError, ValueError):
                conv = None
        td = None
        if ctx.red_zone.value and conv is not None:
            td = conv * (1.0 if ctx.goal_line.value else 0.6)
        delta = 0.0
        if opponent_profile:
            delta = ot.defense_success_delta({"formation": form, "play": play, "family": cand.get("family") or fam or "",
                                              "package": cand.get("package") or ""}, opponent_profile)
            if delta:
                comps.append(Component("opponent_profile", -delta, "observed_opponent",
                                       f"{int(opponent_profile.get('n') or 0)} observed plays ({opponent_profile.get('scope') or 'all downs'})"))
        m = measured.get(str(fam))
        if m and int(m.get("n") or 0) >= MIN_MEASURED and m.get("offense_success") is not None:
            comps.append(Component("measured_family_outcome", -(float(m["offense_success"]) - 0.5), "measured",
                                   f"{m['n']} confirmed executions"))
        score = sum(EVIDENCE_WEIGHT[c.evidence] * c.value for c in comps)
        out.append(CandidateAssessment(
            formation=form, play=play, coverage_family=fam, front=front, pressure=fam == "pressure",
            legal=legal, illegal_reason=why, assessed=fam is not None, run_answer=run_answer, pass_answer=pass_answer,
            explosive_exposure=exposure, conversion_risk=conv, td_risk=td, opponent_delta=delta,
            score=score, components=comps,
        ))
    ranked = sorted(out, key=lambda c: (not c.legal, not c.assessed, -c.score))
    for i, c in enumerate(ranked):
        c.rank = i + 1
        if i + 1 < len(ranked):
            c.versus_next = _compare(c, ranked[i + 1])
    uncertainty = list(ctx.missing) + [f"inconsistent:{p}" for p in ctx.inconsistencies]
    if sources["game_plays"] == 0 and sources["profile_plays"] == 0:
        uncertainty.append("no opponent history: threat is the situation prior only")
    if not any(int((measured.get(str(c.coverage_family)) or {}).get("n") or 0) >= MIN_MEASURED for c in out):
        uncertainty.append("no confirmed defensive executions: no call is measured better than another")
    entropy = -sum(p * math.log(p) for p in threat.values() if p > 0) / math.log(len(THREAT_FAMILIES))
    return DefensiveAssessment(
        version=VERSION, situation_key=key, threat=threat, threat_sources=sources,
        threat_entropy=entropy, uncertainty=uncertainty, candidates=ranked,
        measured_basis={k: dict(v) for k, v in measured.items()},
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def _compare(a: CandidateAssessment, b: CandidateAssessment) -> str:
    if a.legal and not b.legal:
        return f"above {b.play}: that call is not legal here ({b.illegal_reason})"
    if a.assessed and not b.assessed:
        return f"above {b.play}: its coverage is not recognized, so it is left to the established caller"
    av = {c.name: EVIDENCE_WEIGHT[c.evidence] * c.value for c in a.components}
    bv = {c.name: EVIDENCE_WEIGHT[c.evidence] * c.value for c in b.components}
    ev = {c.name: c.evidence for c in a.components + b.components}
    diffs = sorted(((av.get(k, 0.0) - bv.get(k, 0.0), k) for k in set(av) | set(bv)), reverse=True)
    top = [f"{k} ({ev[k]}) {d:+.3f}" for d, k in diffs if d > 0.005][:2]
    margin = a.score - b.score
    if not top or margin < 0.01:
        return f"about even with {b.play} (margin {margin:+.3f}); not a meaningful difference"
    return f"above {b.play} by {margin:+.3f}: " + ", ".join(top)
