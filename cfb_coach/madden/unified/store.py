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
CREATE TABLE IF NOT EXISTS decisions(decision_id TEXT PRIMARY KEY, snap_id TEXT NOT NULL UNIQUE, game_id TEXT NOT NULL, seq INTEGER NOT NULL,
  side TEXT NOT NULL, ts TEXT, state_json TEXT NOT NULL, action_json TEXT NOT NULL, card_json TEXT NOT NULL, policy_prob REAL,
  candidate_set_id TEXT, model_version TEXT, seed INTEGER, latency_ms REAL, fell_back INTEGER, fallback_reason TEXT,
  UNIQUE(game_id, seq));
CREATE TABLE IF NOT EXISTS executions(snap_id TEXT NOT NULL, rev INTEGER NOT NULL, executed_json TEXT, executed_status TEXT NOT NULL,
  actual_adjustments_json TEXT, ts TEXT, PRIMARY KEY(snap_id, rev));
CREATE TABLE IF NOT EXISTS opponent_obs(snap_id TEXT NOT NULL, rev INTEGER NOT NULL, action_json TEXT NOT NULL, verification TEXT, ts TEXT,
  PRIMARY KEY(snap_id, rev));
CREATE TABLE IF NOT EXISTS outcomes(snap_id TEXT NOT NULL, rev INTEGER NOT NULL, outcome_json TEXT NOT NULL, closed_ts TEXT,
  PRIMARY KEY(snap_id, rev));
CREATE TABLE IF NOT EXISTS games(game_id TEXT PRIMARY KEY, opponent_id TEXT, started_ts TEXT, closed_ts TEXT);
CREATE TABLE IF NOT EXISTS voids(snap_id TEXT PRIMARY KEY, ts TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS http_idem(request_key TEXT PRIMARY KEY, path TEXT, response_json TEXT, ts TEXT);
"""


STORE_VERSION = 2


class DuplicateSnapError(RuntimeError):
    """A snap id (or game/seq) was decided twice."""


class AlreadyRecorded(RuntimeError):
    """A fact for this snap was recorded already; correct it explicitly with amend=True (appends a revision)."""


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
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)  # HTTP worker threads; callers serialise with a lock
        ver = self.db.execute("PRAGMA user_version").fetchone()[0]
        has_old = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='decisions'").fetchone()
        if ver not in (0, STORE_VERSION) or (ver == 0 and has_old):
            raise RuntimeError(f"{self.path} was written by an older unified store layout; start a new store file")
        self.db.executescript(SCHEMA)
        self.db.execute(f"PRAGMA user_version={STORE_VERSION}")

    def close(self) -> None:
        self.db.close()

    def log_decision(self, decision_id: str, st: GameState, action: FootballAction, card: dict[str, Any], *, policy_prob: float,
                     candidate_set_id: str, model_version: str, seed: int | None, latency_ms: float, fell_back: bool,
                     fallback_reason: str | None, ts: str) -> None:
        try:
            self.db.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (decision_id, st.snap_id, st.game_id, st.seq, st.our_side, ts, json.dumps(st.to_dict()),
                         json.dumps(action_to_dict(action)), json.dumps(card, default=str), policy_prob, candidate_set_id,
                         model_version, seed, latency_ms, int(fell_back), fallback_reason))
        except sqlite3.IntegrityError as e:
            raise DuplicateSnapError(f"snap {st.snap_id} already has a decision ({e})") from e
        self.db.commit()

    def _next_rev(self, table: str, snap_id: str, amend: bool) -> int:
        row = self.db.execute(f"SELECT MAX(rev) FROM {table} WHERE snap_id=?", (snap_id,)).fetchone()[0]
        if row is None:
            return 1
        if not amend:
            raise AlreadyRecorded(f"{table} already has a row for {snap_id}; pass amend=True to append a corrected revision")
        return row + 1

    def log_execution(self, snap_id: str, executed: FootballAction | None, status: str,
                      actual: tuple[Adjustment, ...] | None, ts: str, *, amend: bool = False) -> int:
        rev = self._next_rev("executions", snap_id, amend)
        self.db.execute("INSERT INTO executions VALUES(?,?,?,?,?,?)",
                        (snap_id, rev, json.dumps(action_to_dict(executed)), status,
                         None if actual is None else json.dumps([asdict(a) for a in actual]), ts))
        self.db.commit()
        return rev

    def log_opponent(self, snap_id: str, oa: OpponentAction, ts: str, *, amend: bool = False) -> int:
        rev = self._next_rev("opponent_obs", snap_id, amend)
        self.db.execute("INSERT INTO opponent_obs VALUES(?,?,?,?,?)", (snap_id, rev, json.dumps(asdict(oa)), oa.verification, ts))
        self.db.commit()
        return rev

    def log_outcome(self, snap_id: str, o: Outcome, ts: str, *, amend: bool = False) -> int:
        rev = self._next_rev("outcomes", snap_id, amend)
        self.db.execute("INSERT INTO outcomes VALUES(?,?,?,?)", (snap_id, rev, json.dumps(asdict(o)), ts))
        self.db.commit()
        return rev

    def max_seq(self, game_id: str) -> int:
        r = self.db.execute("SELECT MAX(seq) FROM decisions WHERE game_id=?", (game_id,)).fetchone()[0]
        return int(r or 0)

    def known_games(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT DISTINCT game_id FROM decisions")}

    def mark_game(self, game_id: str, opponent_id: str, ts: str, *, closed: bool = False) -> None:
        self.db.execute("INSERT OR IGNORE INTO games(game_id, opponent_id, started_ts) VALUES(?,?,?)", (game_id, opponent_id, ts))
        if closed:
            self.db.execute("UPDATE games SET closed_ts=? WHERE game_id=?", (ts, game_id))
        self.db.commit()

    def has_row(self, table: str, snap_id: str) -> bool:
        assert table in ("executions", "opponent_obs", "outcomes", "decisions")
        return self.db.execute(f"SELECT 1 FROM {table} WHERE snap_id=? LIMIT 1", (snap_id,)).fetchone() is not None

    def card_json(self, snap_id: str) -> dict[str, Any] | None:
        r = self.db.execute("SELECT card_json FROM decisions WHERE snap_id=?", (snap_id,)).fetchone()
        return json.loads(r[0]) if r else None

    def void(self, snap_id: str, ts: str, reason: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO voids VALUES(?,?,?)", (snap_id, ts, reason))
        self.db.commit()

    def voided(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT snap_id FROM voids")}

    def unfinished_games(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT game_id FROM games WHERE closed_ts IS NULL ORDER BY started_ts")]

    def idem_get(self, key: str) -> dict[str, Any] | None:
        r = self.db.execute("SELECT response_json FROM http_idem WHERE request_key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def idem_put(self, key: str, path: str, response: dict[str, Any], ts: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO http_idem VALUES(?,?,?,?)", (key, path, json.dumps(response, default=str), ts))
        self.db.commit()

    def closed_games(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT game_id FROM games WHERE closed_ts IS NOT NULL")}

    def load_observations(self) -> list[SnapObservation]:
        """Rebuild observations from the four tables; join is by snap_id only (no cross-table inference)."""
        rows = self.db.execute("SELECT decision_id,snap_id,state_json,action_json FROM decisions ORDER BY game_id, seq").fetchall()
        obs: list[SnapObservation] = []
        for did, sid, sj, aj in rows:
            st = GameState.from_dict(json.loads(sj))
            ex = self.db.execute("SELECT executed_json,executed_status,actual_adjustments_json FROM executions WHERE snap_id=? ORDER BY rev DESC LIMIT 1", (sid,)).fetchone()
            oa = self.db.execute("SELECT action_json FROM opponent_obs WHERE snap_id=? ORDER BY rev DESC LIMIT 1", (sid,)).fetchone()
            oc = self.db.execute("SELECT outcome_json,closed_ts FROM outcomes WHERE snap_id=? ORDER BY rev DESC LIMIT 1", (sid,)).fetchone()
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
