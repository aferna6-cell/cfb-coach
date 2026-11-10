"""Opponent coverage / pressure beliefs before a snap (no leakage).

Reuses the existing opponent model's pieces: ``historical_defense_baseline.normalize_label``
for the post-play coverage the user reports, ``football_priors.descriptor`` for coverage
families, and the same prior strength as ``offense_live_coverage.blend_counts``. Adds what
the 26-variant coordinator needs on top:

* hierarchy  opponent -> opponent type (CPU / human) -> all opponents, from EARLIER games only;
* recency    same-game observations decay by ``RECENCY`` per snap (the previous play counts most);
* pressure   a separate Beta estimate from reported pressure / Cover 0 / Cover 1 pressure labels;
* families   man / spot zone / match, single-high / two-high / zero, as probabilities.

An unreported or unrecognized coverage adds NO mass (unknown stays unknown, never "bad").
A snap's own coverage and anything later never enter its belief.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from cfb_coach.madden.model import football_priors as fpri
from cfb_coach.madden.model import historical_defense_baseline as hb
from cfb_coach.madden.model import offense_live_coverage as lc

PRIOR_STRENGTH = lc.PRIOR_STRENGTH      # same as the live blend
TYPE_STRENGTH = 8.0
RECENCY = 0.85                           # same-game decay per earlier snap
GAME_RECENCY = 0.9                       # earlier-game decay per game
PRESSURE_LABELS = {"pressure_unspecified", "cover_0"}
PRESSURE_PRIOR = (1.0, 4.0)              # weak Beta prior on pressure rate when no history exists


def _label(raw: Any) -> str:
    return hb.normalize_label(raw)


def _is_named(lab: str) -> bool:
    return lab in hb.KNOWN_COVERAGES


def _is_pressure(lab: str, raw: Any) -> bool:
    return lab in PRESSURE_LABELS or "pressure" in str(raw or "").lower() or "blitz" in str(raw or "").lower()


@dataclass
class Belief:
    mix: dict[str, float]                 # fine coverage label -> probability (named labels only)
    pressure: float                       # P(pressure) on this snap
    families: dict[str, float]            # man / zone / match / one_high / two_high / zero / unknown_shell
    n_same_game: float                    # effective same-game named observations
    n_history: float                      # earlier-game named observations behind the prior
    prior_level: str                      # opponent | opponent_type | global | none
    recent: tuple[str, ...] = ()
    basis: str = "earlier_games_prior+same_game_recency"

    @property
    def info(self) -> float:
        n = self.n_same_game + self.n_history
        return n / (n + PRIOR_STRENGTH)

    @property
    def gate_confidence(self) -> float:
        """Evidence behind the belief for the coverage-beater gate: 0 at <= 1 effective observation, 1 at >= 4
        (same-game observations, recency weighted, plus recency-weighted earlier games)."""
        n = self.n_same_game + self.n_history
        return max(0.0, min(1.0, (n - 1.0) / 3.0))

    @property
    def entropy(self) -> float:
        ps = [p for p in self.mix.values() if p > 0]
        return -sum(p * math.log(p) for p in ps) / math.log(max(2, len(hb.KNOWN_COVERAGES))) if ps else 1.0

    def as_log(self) -> dict[str, Any]:
        top = sorted(self.mix.items(), key=lambda kv: -kv[1])[:5]
        return {"basis": self.basis, "prior_level": self.prior_level, "pressure": round(self.pressure, 3),
                "families": {k: round(v, 3) for k, v in self.families.items()},
                "top_coverages": [[k, round(v, 3)] for k, v in top], "n_same_game": round(self.n_same_game, 2),
                "n_history": round(self.n_history, 2), "recent": list(self.recent)}


def families(mix: Mapping[str, float]) -> dict[str, float]:
    out = {"man": 0.0, "zone": 0.0, "match": 0.0, "one_high": 0.0, "two_high": 0.0, "zero": 0.0, "unknown_shell": 0.0}
    for k, p in mix.items():
        d = fpri.descriptor(k)
        if not d:
            out["unknown_shell"] += p
            continue
        t = d.get("type")
        if t == "man":
            out["man"] += p
        elif t == "spot_zone":
            out["zone"] += p
        elif t == "match":
            out["match"] += p
        sh = d.get("shell")
        if sh == "0":
            out["zero"] += p
        elif sh in ("1", "3"):
            out["one_high"] += p
        elif sh in ("2", "4", "split"):
            out["two_high"] += p
        else:
            out["unknown_shell"] += p
    return out


class _Counts:
    def __init__(self) -> None:
        self.named: Counter = Counter()
        self.press = 0.0
        self.labeled = 0.0

    def add(self, lab: str, raw: Any, w: float = 1.0) -> None:
        if _is_pressure(lab, raw):
            self.press += w
            self.labeled += w
        elif _is_named(lab):
            self.labeled += w
        if _is_named(lab):
            self.named[lab] += w

    def scaled(self, f: float) -> "_Counts":
        c = _Counts()
        c.named = Counter({k: v * f for k, v in self.named.items()})
        c.press, c.labeled = self.press * f, self.labeled * f
        return c

    def merge(self, other: "_Counts") -> None:
        self.named.update(other.named)
        self.press += other.press
        self.labeled += other.labeled


def _dist(c: Counter) -> dict[str, float]:
    t = sum(c.values())
    return {k: v / t for k, v in c.items()} if t else {}


def _blend(counts: Mapping[str, float], prior: Mapping[str, float], strength: float) -> dict[str, float]:
    n = sum(counts.values())
    if not prior:
        return _dist(Counter(counts))
    keys = set(prior) | set(counts)
    p = {k: (counts.get(k, 0.0) + strength * prior.get(k, 0.0)) / (n + strength) for k in keys}
    t = sum(p.values())
    return {k: v / t for k, v in sorted(p.items()) if v > 0}


def prior_from_history(opp: _Counts | None, typ: _Counts | None, glob: _Counts | None) -> tuple[dict[str, float], float, float, str]:
    """(mix, pressure_rate, n_history, level) from earlier-game counts: opponent <- type <- global."""
    g = _dist(glob.named) if glob else {}
    t = _blend(typ.named, g, TYPE_STRENGTH) if typ and sum(typ.named.values()) else g
    o = _blend(opp.named, t, PRIOR_STRENGTH) if opp and sum(opp.named.values()) else t
    level = ("opponent" if opp and sum(opp.named.values()) else "opponent_type" if typ and sum(typ.named.values())
             else "global" if g else "none")
    a0, b0 = PRESSURE_PRIOR
    pg = (a0 + (glob.press if glob else 0)) / (a0 + b0 + (glob.labeled if glob else 0))
    pt = ((typ.press + TYPE_STRENGTH * pg) / (typ.labeled + TYPE_STRENGTH)) if typ and typ.labeled else pg
    po = ((opp.press + PRIOR_STRENGTH * pt) / (opp.labeled + PRIOR_STRENGTH)) if opp and opp.labeled else pt
    n_hist = sum((opp or _Counts()).named.values()) or sum((typ or _Counts()).named.values())
    return o, po, float(n_hist), level


def combine(prior_mix: Mapping[str, float], prior_press: float, n_hist: float, level: str,
            same_game: Sequence[tuple[str, Any]]) -> Belief:
    """``same_game``: earlier CLOSED offense snaps of this game, most recent FIRST, as (normalized label, raw)."""
    c = _Counts()
    for age, (lab, raw) in enumerate(same_game):
        c.add(lab, raw, RECENCY ** age)
    mix = _blend(c.named, prior_mix, PRIOR_STRENGTH) if prior_mix else _dist(c.named)
    press = (c.press + PRIOR_STRENGTH * prior_press) / (c.labeled + PRIOR_STRENGTH)
    recent = tuple(lab for lab, _ in same_game[:3])
    return Belief(mix=mix, pressure=press, families=families(mix), n_same_game=sum(c.named.values()),
                  n_history=n_hist, prior_level=level, recent=recent)


def _closed(r: Mapping[str, Any]) -> bool:
    return str(r.get("side") or "offense") == "offense" and str(r.get("result") or "").strip() != ""


def annotate_rows(rows: Sequence[Mapping[str, Any]], game_order: Mapping[str, int]) -> list[Belief]:
    """Belief for every row, using only earlier games (by ``game_order``) and earlier closed snaps of the same game.

    ``rows`` must be chronological within a game (``dataset.build_rows`` order)."""
    by_game: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_game.setdefault(str(r.get("game_id") or ""), []).append(i)
    games = sorted(by_game, key=lambda g: game_order.get(g, 10 ** 9))
    per_game_counts: dict[str, tuple[str, str, _Counts]] = {}
    for g in games:
        idx = by_game[g]
        c = _Counts()
        for i in idx:
            r = rows[i]
            if _closed(r):
                c.add(_label(r.get("coverage_seen")), r.get("coverage_seen"))
        r0 = rows[idx[0]]
        per_game_counts[g] = (str(r0.get("opponent_id") or ""), str(r0.get("opponent_type") or ""), c)
    out: list[Belief | None] = [None] * len(rows)
    for gi, g in enumerate(games):
        opp, typ, _ = per_game_counts[g]
        o, t, a = _Counts(), _Counts(), _Counts()
        for back, h in enumerate(reversed(games[:gi])):
            ho, ht, hc = per_game_counts[h]
            w = GAME_RECENCY ** back
            sc = hc.scaled(w)
            a.merge(sc)
            if ht == typ:
                t.merge(sc)
            if ho == opp:
                o.merge(sc)
        pm, pp, nh, lvl = prior_from_history(o, t, a)
        hist: list[tuple[str, Any]] = []
        for i in by_game[g]:
            r = rows[i]
            out[i] = combine(pm, pp, nh, lvl, hist)
            if _closed(r):
                hist.insert(0, (_label(r.get("coverage_seen")), r.get("coverage_seen")))
    return [b for b in out]  # type: ignore[misc]


def live_belief(conn: Any, *, session_id: str | None, opponent_id: str, opponent_type: str,
                presnap_look: str | None = None) -> Belief:
    """Read-only SELECTs. Earlier games = every other session; same game = closed snaps of ``session_id``.

    ``presnap_look``: this snap's own live pre-snap coverage read, counted once as the most recent observation."""
    rows = conn.execute(
        "SELECT s.session_id, s.opponent_id, s.coverage_seen, s.result, s.side, s.id FROM snaps s "
        "WHERE s.side = 'offense' AND s.result IS NOT NULL AND trim(s.result) != '' ORDER BY s.id").fetchall()
    sessions_order: list[str] = []
    per: dict[str, tuple[str, _Counts]] = {}
    same: list[tuple[str, Any]] = []
    for sid, opp, cov, _res, _side, _id in rows:
        sid = str(sid or "")
        lab = _label(cov)
        if session_id and sid == session_id:
            same.insert(0, (lab, cov))
            continue
        if sid not in per:
            per[sid] = (str(opp or ""), _Counts())
            sessions_order.append(sid)
        per[sid][1].add(lab, cov)
    from cfb_coach.opponents import is_cpu_opponent
    o, t, a = _Counts(), _Counts(), _Counts()
    for back, sid in enumerate(reversed(sessions_order)):
        so, c = per[sid]
        sc = c.scaled(GAME_RECENCY ** back)
        a.merge(sc)
        styp = "cpu" if is_cpu_opponent(so) else "human"
        if styp == opponent_type:
            t.merge(sc)
        if so == opponent_id:
            o.merge(sc)
    pm, pp, nh, lvl = prior_from_history(o, t, a)
    if presnap_look:
        same.insert(0, (_label(presnap_look), presnap_look))
    b = combine(pm, pp, nh, lvl, same)
    if presnap_look:
        b.basis += "+live_presnap_look"
    return b
