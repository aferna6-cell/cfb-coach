"""Shared predictive world model: outcome | GameState + OffensiveAction + DefensiveAction.

Both coordinators query ``WorldModel.predict`` — offense maximises the value it
returns, defense minimises it. One hierarchical empirical-Bayes table is shared:

    L0 situation  ->  L1 (play_type x def_type)  ->  L2 (concept x def_family)
                  ->  L3 (concept x def_variant) ->  L4 (exact off play x def_variant)
                  ->  L5 (exact off play x exact def play)

A rare play borrows from its concept (normal-normal / beta shrinkage toward the
parent, ``w = n / (n + k)``) and its uncertainty widens accordingly. Before any
data exists every level collapses to ``cold_start_prior``: explicit football
principle shifts with a wide variance, labelled ``uncalibrated``. It is not a
trained model and is never reported as one.

Data from our offense and our defense feed the SAME matchup tables (transferable
football knowledge). Separate context offsets (``our_side`` x opponent class) and
personal-execution offsets are kept apart and shrunk hard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from cfb_coach.madden.unified.types import (DEF, EXEC_CONFIRMED, EXEC_MODIFIED, OFF, FootballAction, GameState,
                                            Outcome, SnapObservation)

MODEL_VERSION = "unified_world_model.v1"
METRICS = ("yards", "fd", "td", "to", "expl", "sack")
TAU_YARDS = (0.8, 1.0, 1.3, 1.3, 1.0, 1.0)  # prior sd of a level's true effect around its parent (yards)
TAU_RATE = 0.05  # same for rates (probability units)
MIN_NOISE = 4.0
PRIOR_N = 3.0
OBJECTIVE_NAME = "provisional_objective.v1"  # NOT expected points; see ProvisionalObjective


# ----------------------------------------------------------------------------- objective
@dataclass(frozen=True)
class ProvisionalObjective:
    """Documented provisional football value (offense perspective, ~points).

    Not EPA and not calibrated to Madden. Weights are explicit so a validated
    state-value model can replace ``value`` without touching callers.
    """

    w_yards: float = 0.07
    w_first_down: float = 0.55
    w_td: float = 5.2
    w_turnover: float = 3.6
    w_sack: float = 0.35
    turnover_field_factor: float = 0.02  # turnover is worse the closer we are to scoring
    name: str = OBJECTIVE_NAME
    calibrated: bool = False

    def value(self, st: GameState, ey: float, fd: float, td: float, to: float, sack: float) -> float:
        to_goal = st.to_goal if st.to_goal is not None else 50
        to_cost = self.w_turnover + self.turnover_field_factor * max(0, 80 - to_goal) / 4
        late_fd = 1.3 if st.down in (3, 4) else 1.0  # converting matters more on later downs
        return (self.w_yards * ey + self.w_first_down * late_fd * fd + self.w_td * td
                - to_cost * to - self.w_sack * sack)


class NotCalibrated(RuntimeError):
    pass


def fit_state_value(possessions: Sequence[Mapping[str, Any]], min_possessions: int = 200) -> dict[str, float]:
    """Tabular Madden state value from real possessions: mean points by (zone, down, dist bucket).

    Each possession: {"start": {"yardline","down","distance"}, "points": int}. Refuses (raises
    ``NotCalibrated``) below ``min_possessions``; the repo currently has no such dataset, so the
    provisional objective is used and no EPA is ever fabricated.
    """
    if len(possessions) < min_possessions:
        raise NotCalibrated(f"{len(possessions)} possessions < {min_possessions}: use ProvisionalObjective")
    table: dict[str, list[float]] = {}
    for p in possessions:
        s = p["start"]
        st = GameState("fit", 1, OFF, "x", s.get("down"), s.get("distance"), s.get("yardline"))
        table.setdefault(st.situation_key(), []).append(float(p["points"]))
    return {k: sum(v) / len(v) for k, v in table.items() if len(v) >= 10}


# ----------------------------------------------------------------------------- cold-start prior
BASE = {  # play_type -> (mean yards, sd yards, turnover, sack, explosive)  UNCALIBRATED football priors
    "run": (4.1, 4.3, 0.012, 0.0, 0.05), "pass": (6.0, 8.0, 0.030, 0.055, 0.10),
    "screen": (4.6, 5.5, 0.018, 0.02, 0.06), "play_action": (6.1, 8.5, 0.030, 0.06, 0.11),
    "rpo": (5.4, 6.5, 0.026, 0.045, 0.08), "option": (4.5, 5.5, 0.02, 0.0, 0.06), "unknown": (5.2, 7.0, 0.022, 0.035, 0.08),
}
COMPLETION = {"pass": 0.62, "play_action": 0.60, "screen": 0.76, "rpo": 0.62}  # uncalibrated football priors
SHIFT_CAP = 1.0  # max |prior matchup shift| in yards


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _tail(threshold: float, mean: float, sd: float) -> float:
    return max(0.001, min(0.999, 1.0 - _phi((threshold - 0.5 - mean) / max(sd, 0.5))))


def cold_start_prior(st: GameState, off: Mapping[str, Any], dfn: Mapping[str, Any]) -> dict[str, float]:
    """Explicit football-principle prior. ``off``: profile dict, ``dfn``: defense feature dict."""
    ptype = off.get("play_type", "unknown")
    mean, sd, to, sack, expl = BASE.get(ptype, BASE["unknown"])
    shift, why = 0.0, 0
    blitz, man = bool(dfn.get("blitz")), dfn.get("coverage_type") == "man"
    shell = dfn.get("deep_shell")
    if ptype in ("pass", "play_action", "screen", "rpo"):
        if man and "man" in off.get("may_exploit", ()):
            shift += 0.6
        if blitz:
            shift += 0.5 if off.get("quick") else -0.6
            sack += -0.02 if off.get("quick") else 0.03
            if ptype == "screen":
                shift += 0.6
        if shell in (3, 4) and off.get("deep"):
            shift -= 0.6
        if (shell in (0, 1, None) and man) and off.get("deep"):
            shift += 0.4
            to += 0.005
        if dfn.get("bucket") in ("cover_1_robber",) and "robber" in off.get("reduced_by", ()):
            shift -= 0.5
        if dfn.get("coverage_family") == "cover3" and "cover3_seam" in off.get("may_exploit", ()):
            shift += 0.3
    elif ptype == "run":
        if blitz:
            shift += 0.2
            sd += 0.8
        if man and dfn.get("deep_shell") in (0,):
            shift += 0.4
    shift = max(-SHIFT_CAP, min(SHIFT_CAP, shift))
    mean += shift
    if st.goal_to_go or st.red_zone:  # compressed field: fewer explosives, shorter gains
        sd *= 0.8
        expl *= 0.4 if st.red_zone else 1.0
    d = st.distance or 8
    tg = st.to_goal or 60
    if ptype in COMPLETION:  # passing: completion x conditional gain (bimodal, so short yardage favours the run)
        cr = COMPLETION[ptype] + (0.05 if off.get("quick") else 0.0) - (0.13 if off.get("deep") else 0.0) - (0.03 if blitz else 0.0)
        cr = max(0.35, min(0.85, cr))
        mu_c = max(1.0, mean) / cr
        sd_c = 7.5 if off.get("deep") else 5.5
        p_fd = cr * _tail(d, mu_c, sd_c)
        p_td = min(p_fd, cr * _tail(tg, mu_c, sd_c)) if tg <= 40 else 0.001
        sd = math.sqrt(max(1.0, cr * (sd_c ** 2 + mu_c ** 2) - mean ** 2))
    else:
        p_fd = _tail(d, mean, sd)
        p_td = min(p_fd, _tail(tg, mean, sd)) if tg <= 40 else 0.001
    return {"yards": mean, "fd": p_fd, "td": p_td, "to": to, "expl": min(0.3, expl), "sack": max(0.0, sack),
            "sd": sd, "prior_sd": 1.3 + 0.6 * abs(shift)}


# ----------------------------------------------------------------------------- accumulators
class Acc:
    __slots__ = ("n", "s", "y2")

    def __init__(self) -> None:
        self.n = 0.0
        self.s = [0.0] * 6
        self.y2 = 0.0

    def add(self, vec: Sequence[float], w: float) -> None:
        self.n += w
        for i, v in enumerate(vec):
            self.s[i] += w * v
        self.y2 += w * vec[0] * vec[0]


def outcome_vector(o: Outcome, st: GameState) -> list[float]:
    y = float(o.yards or 0.0)
    fd = o.first_down if o.first_down is not None else (st.distance is not None and y >= st.distance)
    td = bool(o.touchdown)
    return [y, float(bool(fd) or td), float(td), float(bool(o.turnover)),
            float(o.explosive if o.explosive is not None else y >= 15), float(bool(o.sack))]


@dataclass(frozen=True)
class OutcomePrediction:
    e_yards: float
    sd_yards: float
    p_first_down: float
    p_td: float
    p_turnover: float
    p_explosive: float
    p_sack: float
    value: float  # offense-perspective provisional value
    value_sd: float  # model (epistemic) uncertainty of ``value``, not outcome variance
    support_n: float  # effective observed samples at the deepest informative level
    level: int  # deepest level with data (-1 = pure cold-start prior)
    source: str  # cold_start_prior | learned | mixed
    objective: str = OBJECTIVE_NAME
    calibrated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Profile:
    """Matchup-relevant view of one side's call at every hierarchy level."""

    play_type: str
    concept: str
    play_id: str | None
    features: Mapping[str, Any]

    @classmethod
    def off(cls, a: FootballAction) -> "Profile":
        f = dict(a.play.features)
        f.update(a.modified_features)
        return cls(str(f.get("play_type", a.play.play_type)), str(f.get("concept", a.play.concept)), a.play.play_id, f)


@dataclass(frozen=True)
class DefSig:
    ctype: str
    family: str
    vkey: str
    play_id: str | None = None
    features: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_signature(cls, sig: Sequence[str], play_id: str | None = None, features: Mapping[str, Any] | None = None) -> "DefSig":
        s = tuple(sig) + ("unknown",) * (3 - len(sig))
        f = dict(features or {})
        f.setdefault("coverage_type", s[0])
        f.setdefault("coverage_family", s[1])
        f.setdefault("blitz", s[0] == "pressure")
        f.setdefault("bucket", s[2])
        f.setdefault("man", 1.0 if s[0] == "man" else 0.0)
        return cls(s[0], s[1], s[2], play_id, f)


class WorldModel:
    def __init__(self, objective: ProvisionalObjective | None = None) -> None:
        self.objective = objective or ProvisionalObjective()
        self.levels: list[dict[tuple, Acc]] = [dict() for _ in range(6)]
        self.ctx_resid: dict[tuple, Acc] = {}  # (our_side, opponent_class) yard residual vs shared prediction
        self.adj_resid: dict[tuple, Acc] = {}  # (action_id, def_type) yard residual vs stock prediction
        self.n_records = 0
        self.tau_y = list(TAU_YARDS)
        self.tau_r = [TAU_RATE] * 6
        self.tau_fitted = False
        self.provenance: dict[str, int] = {}  # rows learned per source: our_offense_executed / our_defense_executed
        self.seen_off_play_ids: set[str] = set()
        self.seen_vkeys: set[str] = set()

    # ---- keys -------------------------------------------------------------------
    @staticmethod
    def _keys(st: GameState, p: Profile, d: DefSig) -> list[tuple]:
        sit = (st.down, st.dist_bucket(), "rz" if st.red_zone else "o")
        return [sit, (p.play_type, d.ctype), (p.concept, d.family), (p.concept, d.vkey),
                (p.play_id, d.vkey), (p.play_id, d.play_id)]

    # ---- inference --------------------------------------------------------------
    def _posterior(self, st: GameState, p: Profile, d: DefSig, depth: int) -> tuple[list[float], float, float, int, float, float]:
        """Cold-start prior (state dependent) + hierarchically shrunk RESIDUAL learned at L0..L<depth>.

        Each level shrinks toward its parent with weight ``n / (n + k)``, ``k = noise / tau^2`` — the
        classic normal-normal rule — so noisy rare cells move little and well-observed cells move a lot.
        Returns (means[6], var_of_mean, outcome_sd, deepest_level, deepest_n, prior_sd).
        """
        pr = cold_start_prior(st, p.features, d.features)
        prior = [pr[k] for k in METRICS]
        sd = pr["sd"]
        var = pr["prior_sd"] ** 2
        resid = [0.0] * 6
        deepest, deepest_n = -1, 0.0
        keys = self._keys(st, p, d)
        for lvl in range(depth + 1):
            if lvl >= 4 and keys[lvl][0] is None:
                continue
            if lvl == 5 and keys[lvl][1] is None:
                continue
            a = self.levels[lvl].get(keys[lvl])
            if a is None or a.n <= 0:
                continue
            raw = [x / a.n for x in a.s]
            ydvar = max(MIN_NOISE, a.y2 / a.n - raw[0] ** 2)
            ws = []
            for i in range(6):
                if i == 0:
                    k = ydvar / self.tau_y[lvl] ** 2
                else:
                    pm = min(max(prior[i] + resid[i], 0.01), 0.99)
                    k = max(pm * (1 - pm), 0.02) / self.tau_r[lvl] ** 2
                w = a.n / (a.n + k)
                ws.append(w)
                resid[i] = w * raw[i] + (1 - w) * resid[i]
            w0 = ws[0]
            var = w0 * w0 * (ydvar / a.n) + (1 - w0) ** 2 * var
            sd = math.sqrt(max(ydvar, 1.0) + 0.0) if a.n >= 10 else sd
            deepest, deepest_n = lvl, a.n
        m = [prior[0] + resid[0]] + [min(0.999, max(0.001, prior[i] + resid[i])) for i in range(1, 6)]
        return m, var, sd, deepest, deepest_n, pr["prior_sd"]

    def predict(self, st: GameState, off: FootballAction | Profile, dfn: DefSig, *, personal: Any = None,
                our_side: str | None = None, opponent_class: str | None = None) -> OutcomePrediction:
        p = off if isinstance(off, Profile) else Profile.off(off)
        depth = 5 if dfn.play_id and p.play_id else 4 if p.play_id else 3
        m, var, sd, lvl, n, _ = self._posterior(st, p, dfn, depth)
        ey = m[0]
        extra_var = 0.0
        if our_side is not None:
            c = self.ctx_resid.get((our_side, opponent_class or "UNKNOWN"))
            if c and c.n > 0:
                w = c.n / (c.n + 30.0)
                ey += w * c.s[0] / c.n
        if personal is not None:
            ey, extra = personal.adjust(p, ey, our_side)
            extra_var += extra
        fd, td, to, ex, sk = (min(max(x, 0.0), 1.0) for x in m[1:])
        td = min(td, fd)
        to_eff = to + (0.0 if personal is None else personal.turnover_delta(p))
        v = self.objective.value(st, ey, fd, td, max(0.0, to_eff), sk)
        # propagate mean uncertainty (yards) + rate-uncertainty into value uncertainty
        rate_sd = math.sqrt(max(fd * (1 - fd), 0.01) / (n + 6.0))
        vsd = math.sqrt((self.objective.w_yards ** 2) * (var + extra_var)
                        + (self.objective.w_first_down * rate_sd) ** 2 + (0.5 * self.objective.w_td * rate_sd * 0.4) ** 2)
        src = "cold_start_prior" if lvl < 0 else "learned" if n >= 30 else "mixed"
        return OutcomePrediction(ey, sd, fd, td, to_eff, ex, sk, v, vsd, n, lvl, src, self.objective.name,
                                 self.objective.calibrated)

    # ---- learning ---------------------------------------------------------------
    def update(self, st: GameState, p: Profile, d: DefSig, outcome: Outcome, *, weight: float = 1.0,
               our_side: str | None = None, opponent_class: str | None = None):
        if not outcome.known or weight <= 0:
            return None
        vec = outcome_vector(outcome, st)
        if our_side is not None:  # context residual measured against the pre-update shared prediction
            base = self.predict(st, p, d).e_yards
            self.ctx_resid.setdefault((our_side, opponent_class or "UNKNOWN"), Acc()).add([vec[0] - base, 0, 0, 0, 0, 0], weight)
        pr = cold_start_prior(st, p.features, d.features)
        vec = [v - pr[k] for v, k in zip(vec, METRICS)]  # learn the residual against the state-dependent prior
        keys = self._keys(st, p, d)
        if p.play_id:
            self.seen_off_play_ids.add(p.play_id)
        self.seen_vkeys.add(d.vkey)
        for lvl, k in enumerate(keys):
            if lvl >= 4 and k[0] is None:
                continue
            if lvl == 5 and k[1] is None:
                continue
            self.levels[lvl].setdefault(k, Acc()).add(vec, weight)
        self.n_records += 1
        return keys, vec, weight

    def fit_hierarchy(self, records: Iterable[tuple]) -> dict[str, Any]:
        """Batch fit with empirical-Bayes estimation of each level's effect sd (tau).

        ``tau_l^2`` is the between-cell variance of level-l cell means around their parent cell, minus
        sampling noise (method of moments), floored. This stops deep cells from over-reacting when the
        data say they barely differ from their concept, and lets them move when they do.
        """
        logs = []
        for st, p, d, oc, w, side in records:
            r = self.update(st, p, d, oc, weight=w, our_side=side, opponent_class=st.opponent_class)
            if r:
                logs.append(r)
        if not logs:
            return {}
        noise = [max(self._pooled_var(logs, i), 1e-6) for i in range(6)]
        report: dict[str, Any] = {}
        for lvl in range(2, 6):
            num = [0.0] * 6
            den = 0.0
            for keys, vec, w in logs:
                pass
            cells: dict[tuple, Acc] = self.levels[lvl]
            parents: dict[tuple, Acc] = self.levels[lvl - 1]
            # parent of a cell is recovered from the logged records (child key -> parent key is a function)
            pmap: dict[tuple, tuple] = {}
            for keys, _v, _w in logs:
                if lvl >= 4 and (keys[lvl][0] is None or (lvl == 5 and keys[lvl][1] is None)):
                    continue
                pmap[keys[lvl]] = keys[lvl - 1]
            for key, a in cells.items():
                pk = pmap.get(key)
                pa = parents.get(pk) if pk is not None else None
                if pa is None or a.n < 5:
                    continue
                den += a.n
                for i in range(6):
                    diff = a.s[i] / a.n - pa.s[i] / pa.n
                    num[i] += a.n * max(0.0, diff * diff - noise[i] / a.n)
            if den > 0:
                self.tau_y[lvl] = min(2.5, max(0.2, math.sqrt(num[0] / den)))
                self.tau_r[lvl] = min(0.2, max(0.015, math.sqrt(num[1] / den)))
            report[f"L{lvl}"] = {"tau_yards": round(self.tau_y[lvl], 3), "tau_rate": round(self.tau_r[lvl], 4)}
        self.tau_fitted = True
        return report

    @staticmethod
    def _pooled_var(logs: Sequence[tuple], i: int) -> float:
        tot = sum(w for _, _, w in logs)
        mu = sum(w * v[i] for _, v, w in logs) / tot
        return sum(w * (v[i] - mu) ** 2 for _, v, w in logs) / tot

    def n_support(self) -> dict[str, float]:
        return {f"L{i}": sum(a.n for a in lv.values()) for i, lv in enumerate(self.levels)}

    def to_dict(self) -> dict[str, Any]:
        return {"version": MODEL_VERSION, "n_records": self.n_records, "objective": self.objective.name,
                "calibrated": self.objective.calibrated,
                "levels": [[[list(k), a.n, a.s, a.y2] for k, a in lv.items()] for lv in self.levels],
                "ctx": [[list(k), a.n, a.s, a.y2] for k, a in self.ctx_resid.items()],
                "provenance": self.provenance, "tau_y": self.tau_y, "tau_r": self.tau_r, "tau_fitted": self.tau_fitted}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WorldModel":
        if d.get("version") != MODEL_VERSION:
            raise ValueError("model version mismatch")
        m = cls()
        m.n_records = d["n_records"]
        m.provenance = dict(d.get("provenance", {}))
        m.tau_y, m.tau_r, m.tau_fitted = list(d.get("tau_y", m.tau_y)), list(d.get("tau_r", m.tau_r)), d.get("tau_fitted", False)
        for i, lv in enumerate(d["levels"]):
            for k, n, s, y2 in lv:
                a = Acc()
                a.n, a.s, a.y2 = n, list(s), y2
                m.levels[i][tuple(k)] = a
        for lv in m.levels[3:5]:
            for k in lv:
                if k[0] and str(k[0]).startswith("off:"):
                    m.seen_off_play_ids.add(k[0])
                m.seen_vkeys.add(k[1])
        for k, n, s, y2 in d.get("ctx", []):
            a = Acc()
            a.n, a.s, a.y2 = n, list(s), y2
            m.ctx_resid[tuple(k)] = a
        return m


def records_from_observation(obs: SnapObservation) -> list[tuple[GameState, Profile, DefSig, Outcome, float, str]]:
    """Turn one closed snap into world-model training rows. Strict label rules:

    * our own call counts only when ``executed`` is verified (recommendation != execution);
    * the opponent's call counts only through ``opponent_action`` (revealed), weighted by confidence;
    * an observed opposing call is never treated as our execution.
    """
    from cfb_coach.madden.unified import knowledge as K

    if not obs.closed or obs.outcome is None or not obs.outcome.known or obs.opponent_action is None:
        return []
    if obs.executed is None or obs.executed_status not in (EXEC_CONFIRMED, EXEC_MODIFIED):
        return []
    oa = obs.opponent_action
    w = {"owner_confirmed": 1.0, "typed_exact": 1.0, "visual_inferred": 0.5, "unverified": 0.0}[oa.verification]
    w *= max(0.0, min(1.0, oa.confidence or 0.0))
    if w <= 0:
        return []
    if obs.state.our_side == OFF:
        prof = Profile.off(obs.executed)
        play = K.defense_play(oa.formation, oa.play) if oa.formation and oa.play else None
        d = DefSig.from_signature(play.signature, play.play_id, play.features) if play else DefSig.from_signature(oa.signature)
    else:
        play = K.offense_play(oa.formation, oa.play) if oa.formation and oa.play else None
        if play is None and not oa.signature:
            return []
        f = dict(play.features) if play else {"play_type": oa.signature[0], "concept": oa.signature[-1]}
        prof = Profile(str(f.get("play_type", "unknown")), str(f.get("concept", "unknown")), play.play_id if play else None, f)
        ex = obs.executed
        d = DefSig.from_signature(ex.signature, ex.play.play_id, ex.play.features)
    return [(obs.state, prof, d, obs.outcome, w, obs.state.our_side)]
