"""Bayesian opponent intelligence (one system, separate offensive / defensive observations).

Belief over the opponent's next call signature is built by hierarchical Dirichlet
smoothing, parent -> child:

    general Madden prior (situation-tilted)
      -> opponent class counts (cpu / human)
      -> this opponent's finished games, overall, then in this situation
      -> THIS game, overall, in this situation, after their previous call,
         and after our previous call (reaction)

``p_child(x) = (c(x) + k * p_parent(x)) / (N + k)``. In-game nodes decay by recency.
Updated after every closed snap; no model weights are retrained mid-game.

Time fencing: only CLOSED snaps with ``seq < state.seq`` from the current game and
games already closed (and not the current one) are visible. A revealed call that is
unverified carries zero weight.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from cfb_coach.madden.unified.types import DEF, OFF, GameState, SnapObservation, other_side

OPP_VERSION = "unified_opponent.v1"
K_NODE = {"class": 25.0, "hist_all": 14.0, "hist_sit": 9.0, "game_all": 7.0, "game_sit": 6.0, "game_seq": 6.0,
          "game_react": 6.0}
RECENCY_DECAY = 0.93
Sig = tuple


def _norm(d: Mapping[Sig, float]) -> dict[Sig, float]:
    t = sum(d.values())
    return {k: v / t for k, v in d.items()} if t > 0 else {}


DEF_FAMILY_MASS = {"cover3": 0.27, "cover1": 0.18, "cover2": 0.15, "cover4": 0.13, "cover6": 0.05, "cover9": 0.01,
                   "cover0": 0.02, "pressure": 0.14, "unknown": 0.05}
OFF_TYPE_MASS = {"run": 0.42, "pass": 0.40, "play_action": 0.09, "screen": 0.05, "rpo": 0.03, "option": 0.01}


def general_prior(opp_side: str) -> dict[Sig, float]:
    """Unconditional, UNCALIBRATED prior: football-style mass per family, split evenly over the signatures
    that exist in the Madden catalog. Replaced by evidence as soon as observations arrive."""
    from cfb_coach.madden.unified import knowledge as K

    groups: dict[str, set[Sig]] = defaultdict(set)
    if opp_side == DEF:
        for id_ in K.defense_catalog():
            if id_.startswith("inst:"):
                f, p = id_[5:].split("|", 1)
                sig = K.defense_play(f, p).signature
                if sig == ("unknown", "unknown", "unknown"):
                    continue
                groups["pressure" if sig[0] == "pressure" else sig[1]].add(sig)
        mass = DEF_FAMILY_MASS
    else:
        import json
        pb = json.loads((K.DATA / "playbooks_full.json").read_text())["books"]["offense"]
        seen = {(fm, pl) for bk in pb.values() for fm, rec in bk["formations"].items() for pl in rec["plays"]}
        for fm, pl in seen:
            sig = K.offense_play(fm, pl).signature
            groups[sig[0] if sig[0] in OFF_TYPE_MASS else "pass"].add(sig)
        mass = OFF_TYPE_MASS
    out: dict[Sig, float] = {}
    for g, sigs in groups.items():
        for sig in sigs:
            out[sig] = mass.get(g, 0.02) / len(sigs)
    return _norm(out)


def situation_tilt(opp_side: str, p: Mapping[Sig, float], st: GameState) -> dict[Sig, float]:
    """Situation tilt on the general prior only (football-principle, bounded)."""
    out = dict(p)
    d, dist = st.down, st.distance
    for sig in out:
        if opp_side == OFF:  # opponent offense: run/pass tilt
            is_run = sig[0] == "run"
            if d in (2, 3, 4) and dist is not None and dist >= 8:
                out[sig] *= 0.55 if is_run else 1.25
            elif dist is not None and dist <= 2:
                out[sig] *= 1.5 if is_run else 0.9
            elif d == 1:
                out[sig] *= 1.2 if is_run else 0.95
        else:  # opponent defense: pressure tilt on obvious passing downs
            if sig[0] == "pressure" and d in (3, 4) and dist is not None and dist >= 7:
                out[sig] *= 1.5
            if st.goal_to_go and sig[1] in ("cover0", "cover1"):
                out[sig] *= 1.4
    return _norm(out)


@dataclass
class OppRecord:
    opponent_id: str
    game_id: str
    game_order: int
    seq: int
    sig: Sig
    situation: str
    prev_sig: Sig | None
    our_prev: str | None  # our previous call family, for reaction modelling
    weight: float
    opponent_class: str


@dataclass
class Belief:
    dist: dict[Sig, float]
    n_effective: float  # evidence mass behind the posterior (excludes the prior)
    alpha0: float  # total Dirichlet concentration = n_effective + prior strength
    sources: list[str]
    entropy: float
    top: list[tuple[Sig, float]] = field(default_factory=list)

    @property
    def prediction_source(self) -> str:
        return "+".join(self.sources)

    @property
    def tv_radius(self) -> float:
        """Total-variation radius of the plausible-opponent-distribution set (robust solver)."""
        return min(0.35, 1.6 / math.sqrt(self.alpha0 + 1.0))


class OpponentModel:
    """Beliefs over the opponent's calls on ONE side of the ball."""

    def __init__(self, opp_side: str, *, prior: Mapping[Sig, float] | None = None) -> None:
        self.opp_side = opp_side
        self._prior = dict(prior) if prior is not None else general_prior(opp_side)
        self.records: list[OppRecord] = []
        self.closed_games: dict[str, int] = {}  # game_id -> order index
        self.class_counts: dict[str, dict[Sig, float]] = defaultdict(lambda: defaultdict(float))
        self._order = 0

    # ---- ingestion ------------------------------------------------------------
    def set_game_order(self, game_id: str) -> int:
        if game_id not in self.closed_games:
            self._order += 1
            self.closed_games.setdefault("__open__" + game_id, self._order)
        return self.closed_games.get(game_id, self._order)

    def _order_of(self, game_id: str) -> int:
        return self.closed_games.get(game_id) or self.closed_games.get("__open__" + game_id) or 0

    def observe(self, obs: SnapObservation, *, prev_obs: SnapObservation | None = None) -> bool:
        """Add one CLOSED snap. Returns True if it carried usable (verified) evidence."""
        oa = obs.opponent_action
        if not obs.closed or oa is None or oa.side != self.opp_side or not oa.signature:
            return False
        w = {"owner_confirmed": 1.0, "typed_exact": 1.0, "visual_inferred": 0.5, "unverified": 0.0}[oa.verification]
        w *= max(0.0, min(1.0, oa.confidence or 0.0))
        if w <= 0:
            return False
        gid = obs.state.game_id
        if not self._order_of(gid):
            self.set_game_order(gid)
        prev_sig = prev_obs.opponent_action.signature if prev_obs and prev_obs.opponent_action else None
        our_prev = None
        if prev_obs is not None and prev_obs.recommended is not None and prev_obs.executed_status != "not_executed":
            ex = prev_obs.executed or prev_obs.recommended
            our_prev = ex.play.signature[1] if ex.play.signature[1:] else ex.play.signature[0]
        self.records.append(OppRecord(obs.state.opponent_id, gid, self._order_of(gid), obs.state.seq, tuple(oa.signature),
                                      obs.state.situation_key(), tuple(prev_sig) if prev_sig else None, our_prev, w,
                                      obs.state.opponent_class))
        return True

    def close_game(self, game_id: str) -> None:
        order = self._order_of(game_id) or self.set_game_order(game_id)
        self.closed_games[game_id] = order
        self.closed_games.pop("__open__" + game_id, None)
        for r in self.records:
            if r.game_id == game_id:
                self.class_counts[r.opponent_class][r.sig] += r.weight

    # ---- belief ---------------------------------------------------------------
    @staticmethod
    def _smooth(parent: Mapping[Sig, float], counts: Mapping[Sig, float], k: float) -> dict[Sig, float]:
        n = sum(counts.values())
        if n <= 0:
            return dict(parent)
        keys = set(parent) | set(counts)
        return {x: (counts.get(x, 0.0) + k * parent.get(x, 0.0)) / (n + k) for x in keys}

    def beliefs(self, st: GameState, *, our_prev_family: str | None = None) -> Belief:
        opp = st.opponent_id
        p = situation_tilt(self.opp_side, self._prior, st)
        sources = ["general_prior"]
        sit = st.situation_key()
        cur_order = self._order_of(st.game_id)
        # class node (finished games only)
        cc = self.class_counts.get(st.opponent_class)
        n_ev = 0.0
        if cc:
            p = self._smooth(p, cc, K_NODE["class"])
            sources.append(f"class[{st.opponent_class}]")
        hist = [r for r in self.records if r.opponent_id == opp and r.game_id != st.game_id
                and r.game_id in self.closed_games and (not cur_order or r.game_order < cur_order)]
        game = [r for r in self.records if r.opponent_id == opp and r.game_id == st.game_id and r.seq < st.seq]
        game.sort(key=lambda r: r.seq)
        if hist:
            c_all: dict[Sig, float] = defaultdict(float)
            c_sit: dict[Sig, float] = defaultdict(float)
            for r in hist:
                c_all[r.sig] += r.weight
                if r.situation == sit:
                    c_sit[r.sig] += r.weight
            p = self._smooth(p, c_all, K_NODE["hist_all"])
            p = self._smooth(p, c_sit, K_NODE["hist_sit"])
            n_ev += sum(c_all.values())
            sources.append(f"hist({len(hist)})")
        if game:
            last = game[-1].seq
            dec = lambda r: r.weight * (RECENCY_DECAY ** (last - r.seq))  # noqa: E731
            c_all, c_sit, c_seq, c_re = (defaultdict(float) for _ in range(4))
            prev = game[-1].sig
            for r in game:
                w = dec(r)
                c_all[r.sig] += w
                if r.situation == sit:
                    c_sit[r.sig] += w
                if r.prev_sig is not None and r.prev_sig == prev:
                    c_seq[r.sig] += w
                if our_prev_family is not None and r.our_prev == our_prev_family:
                    c_re[r.sig] += w
            p = self._smooth(p, c_all, K_NODE["game_all"])
            p = self._smooth(p, c_sit, K_NODE["game_sit"])
            p = self._smooth(p, c_seq, K_NODE["game_seq"])
            p = self._smooth(p, c_re, K_NODE["game_react"])
            n_ev += sum(c_all.values())
            sources.append(f"game({len(game)})")
        p = _norm(p)
        ent = -sum(v * math.log(v) for v in p.values() if v > 0)
        top = sorted(p.items(), key=lambda kv: -kv[1])[:6]
        return Belief(p, n_ev, n_ev + K_NODE["class"] / 2 + 4.0, sources, ent, top)
