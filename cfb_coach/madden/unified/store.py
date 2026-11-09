"""Isolated SQLite log for the unified coordinator.

Four append-only tables keep the facts apart (spec §3):
``decisions`` (what we recommended), ``executions`` (what we actually ran),
``opponent_obs`` (what the other side was revealed to run), ``outcomes`` (the result).
Nothing is inferred across tables; a recommendation never populates ``executions``.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from cfb_coach.madden.unified.guard import assert_isolated
from cfb_coach.madden.unified.types import (EXEC_UNKNOWN, Adjustment, FootballAction, FootballPlay, GameState, OpponentAction,
                                            Outcome, SnapObservation)

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions(decision_id TEXT PRIMARY KEY, snap_id TEXT NOT NULL, game_id TEXT NOT NULL, seq INTEGER NOT NULL,
  side TEXT NOT NULL, ts TEXT, state_json TEXT NOT NULL, action_json TEXT NOT NULL, card_json TEXT NOT NULL, policy_prob REAL,
  candidate_set_id TEXT, model_version TEXT, seed INTEGER, latency_ms REAL, fell_back INTEGER, fallback_reason TEXT);
CREATE TABLE IF NOT EXISTS executions(snap_id TEXT PRIMARY KEY, executed_json TEXT, executed_status TEXT NOT NULL,
  actual_adjustments_json TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS opponent_obs(snap_id TEXT PRIMARY KEY, action_json TEXT NOT NULL, verification TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS outcomes(snap_id TEXT PRIMARY KEY, outcome_json TEXT NOT NULL, closed_ts TEXT);
CREATE TABLE IF NOT EXISTS games(game_id TEXT PRIMARY KEY, opponent_id TEXT, our_side_mix TEXT, started_ts TEXT, closed_ts TEXT, seq_order INTEGER);
"""


def _play_from(d: dict[str, Any]) -> FootballPlay:
    from cfb_coach.madden.unified.types import Assignment
    d = dict(d)
    d["assignments"] = tuple(Assignment(**a) for a in d.get("assignments", ()))
    for k in ("concept_path", "signature", "variant", "provenance", "unknown_fields"):
        d[k] = tuple(d.get(k, ()))
    return FootballPlay(**d)


def action_to_dict(a: FootballAction | None) -> dict[str, Any] | None:
    return None if a is None else asdict(a)


def action_from_dict(d: dict[str, Any] | None) -> FootballAction | None:
    if d is None:
        return None
    adjs = tuple(Adjustment(**x) for x in d.get("adjustments", ()))
    return FootballAction(_play_from(d["play"]), adjs, d.get("modified_features", {}), d.get("legal", True),
                          d.get("illegal_reason"), d.get("executable", True))


class UnifiedStore:
    def __init__(self, path: str | Path, *, protected: Iterable[str | Path] = ()) -> None:
        self.path = assert_isolated(path, protected=protected)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def log_decision(self, decision_id: str, st: GameState, action: FootballAction, card: dict[str, Any], *, policy_prob: float,
                     candidate_set_id: str, model_version: str, seed: int | None, latency_ms: float, fell_back: bool,
                     fallback_reason: str | None, ts: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (decision_id, st.snap_id, st.game_id, st.seq, st.our_side, ts, json.dumps(st.to_dict()),
                         json.dumps(action_to_dict(action)), json.dumps(card, default=str), policy_prob, candidate_set_id,
                         model_version, seed, latency_ms, int(fell_back), fallback_reason))
        self.db.commit()

    def log_execution(self, snap_id: str, executed: FootballAction | None, status: str,
                      actual: tuple[Adjustment, ...] | None, ts: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO executions VALUES(?,?,?,?,?)",
                        (snap_id, json.dumps(action_to_dict(executed)), status,
                         None if actual is None else json.dumps([asdict(a) for a in actual]), ts))
        self.db.commit()

    def log_opponent(self, snap_id: str, oa: OpponentAction, ts: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO opponent_obs VALUES(?,?,?,?)", (snap_id, json.dumps(asdict(oa)), oa.verification, ts))
        self.db.commit()

    def log_outcome(self, snap_id: str, o: Outcome, ts: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO outcomes VALUES(?,?,?)", (snap_id, json.dumps(asdict(o)), ts))
        self.db.commit()

    def load_observations(self) -> list[SnapObservation]:
        """Rebuild observations from the four tables; join is by snap_id only (no cross-table inference)."""
        rows = self.db.execute("SELECT decision_id,snap_id,state_json,action_json FROM decisions ORDER BY game_id, seq").fetchall()
        obs: list[SnapObservation] = []
        for did, sid, sj, aj in rows:
            st = GameState.from_dict(json.loads(sj))
            ex = self.db.execute("SELECT executed_json,executed_status,actual_adjustments_json FROM executions WHERE snap_id=?", (sid,)).fetchone()
            oa = self.db.execute("SELECT action_json FROM opponent_obs WHERE snap_id=?", (sid,)).fetchone()
            oc = self.db.execute("SELECT outcome_json,closed_ts FROM outcomes WHERE snap_id=?", (sid,)).fetchone()
            action = action_from_dict(json.loads(aj))
            o = SnapObservation(st, recommended=action, decision_id=did)
            if ex:
                actual = None if ex[2] is None else tuple(Adjustment(**x) for x in json.loads(ex[2]))
                o = o.with_execution(action_from_dict(json.loads(ex[0])) if ex[0] and ex[0] != "null" else None, ex[1], actual)
            if oa:
                d = json.loads(oa[0]); d["signature"] = tuple(d.get("signature", ()))
                o = o.with_opponent(OpponentAction(**d))
            if oc:
                from dataclasses import replace
                o = replace(o, outcome=Outcome(**json.loads(oc[0])), closed=True, closed_ts=oc[1], cutoff_seq=st.seq)
            obs.append(o)
        return obs

    def counts(self) -> dict[str, int]:
        return {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("decisions", "executions", "opponent_obs", "outcomes")}

    def load_games(self) -> list[list[SnapObservation]]:
        """Observations grouped by game, games ordered by their first decision timestamp (chronology)."""
        obs = self.load_observations()
        first_ts = {gid: ts for gid, ts in self.db.execute("SELECT game_id, MIN(ts) FROM decisions GROUP BY game_id")}
        by: dict[str, list[SnapObservation]] = {}
        for o in obs:
            by.setdefault(o.state.game_id, []).append(o)
        return [sorted(by[g], key=lambda o: o.state.seq) for g in sorted(by, key=lambda g: (first_ts.get(g) or "", g))]
