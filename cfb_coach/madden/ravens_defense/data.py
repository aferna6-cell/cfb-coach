"""Read-only extraction of defensive snaps and the training-data audit.

Two independent axes per defensive snap:
  OUR EXECUTION   verified_ravens_execution | verified_other_execution | recommended_unexecuted | unknown
  OPPONENT PLAY   observed (concept class + closed result) | not observed
Only ``verified_ravens_execution`` with a reliable outcome may train candidate-specific outcome
terms. Observed opponent plays may train opponent beliefs and candidate-agnostic base rates.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense.concepts import classify
from cfb_coach.madden.ravens_defense.users import parse_offense_shape

EXPLOSIVE_YARDS = 15


@dataclass(frozen=True)
class DefSnap:
    row_id: int
    session: str
    opponent: str
    opponent_type: str
    franchise: str
    started_ts: str
    ts: str
    order: int                 # chronological order within the game (snap_seq when known, else ts/id)
    seq: int | None
    down: int | None
    distance: int | None
    yards_to_goal: int | None
    concept: str | None
    concept_raw: str | None
    shape: str | None
    closed: bool
    offense_success: bool | None
    yards: float | None
    explosive: bool | None
    touchdown: bool | None
    turnover: bool | None
    first_down: bool | None
    execution: str             # verified_ravens_execution | verified_other_execution | recommended_unexecuted | unknown
    candidate_id: str | None
    candidate_reason: str


def open_ro(path: str | Path) -> sqlite3.Connection:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"database not found: {p}")
    conn = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _outcomes(result: Any, down: Any, distance: Any, recorded_yards: Any) -> dict[str, Any]:
    from cfb_coach.madden.model.defense_v1 import offense_won
    from cfb_coach.outcome import parse_outcome
    parsed = parse_outcome(None if result is None else str(result))
    won = offense_won(result, down, distance)
    yards = None
    try:
        yards = float(recorded_yards) if recorded_yards is not None else (
            float(parsed.yards) if parsed.yards is not None else None)
        if yards is not None and (not math.isfinite(yards) or abs(yards) > 100):
            yards = None
    except (TypeError, ValueError):
        yards = None
    known = parsed.kind != "unknown"
    td = parsed.kind == "td" if known else None
    first = None
    if known:
        if parsed.kind in ("td", "convert"):
            first = True
        elif yards is not None and distance:
            first = yards >= float(distance)
        elif parsed.kind in ("incomplete", "int", "fumble", "sack", "stop", "loss"):
            first = False
    explosive = None if yards is None and not td else bool((yards or 0) >= EXPLOSIVE_YARDS)
    return {"offense_success": won, "yards": yards, "touchdown": td,
            "turnover": parsed.is_turnover if known else None, "first_down": first, "explosive": explosive}


def _decision_candidates(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Recommended Ravens candidate per (session, ml_snap_id) from the live decision log."""
    if not _has_table(conn, "ravens_defense_decisions"):
        return {}
    return {(str(s), str(m)): str(c) for s, m, c in conn.execute(
        "SELECT session_id, ml_snap_id, candidate_id FROM ravens_defense_decisions "
        "WHERE status='CALL' AND ml_snap_id IS NOT NULL ORDER BY id")}


def read_snaps(conn: sqlite3.Connection, cat: rc.RavensCatalog | None = None, *,
               standing_macro_attestation: bool = False, session_filter: str | None = None) -> list[DefSnap]:
    cat = cat or rc.load()
    cols = _cols(conn, "snaps")
    gcols = _cols(conn, "game_sessions")
    need = {"session_id", "side", "result", "down", "distance"}
    if not need <= cols:
        raise ValueError("snaps table lacks " + ", ".join(sorted(need - cols)))
    def c(name: str, alias: str = "s") -> str:
        return f"{alias}.{name}" if name in (cols if alias == "s" else gcols) else "NULL"
    concept = (f"COALESCE(NULLIF(s.concept_typed,''), s.concept_seen)" if "concept_typed" in cols else c("concept_seen"))
    sql = ("SELECT s.id, s.session_id, COALESCE(" + c("opponent_id", "g") + ", " + c("opponent_id") + "), " + c("dynasty", "g") + ", "
           + c("started_ts", "g") + ", s.ts, " + c("snap_seq") + ", s.down, s.distance, " + c("yardline") + ", "
           + concept + ", " + c("situation_raw") + ", s.result, " + c("yards") + ", " + c("executed_status") + ", "
           + c("executed_verification") + ", " + c("executed_formation") + ", " + c("executed_play") + ", "
           + c("executed_macro") + ", " + c("formation") + ", " + c("play") + ", " + c("ml_snap_id") +
           " FROM snaps s LEFT JOIN game_sessions g ON g.session_id = s.session_id"
           " WHERE s.side = 'defense'" + (" AND s.session_id = ?" if session_filter else "") +
           " ORDER BY s.session_id, s.ts, s.id")
    from cfb_coach.opponents import is_cpu_opponent
    decided = _decision_candidates(conn)
    out: list[DefSnap] = []
    order: Counter[str] = Counter()
    for (rid, sess, opp, dyn, started, ts, seq, down, dist, yl, craw, sraw, result, yards, est, ever,
         eform, eplay, emacro, rform, rplay, msid) in conn.execute(sql, (session_filter,) if session_filter else ()):
        sess = str(sess or f"nosession:{rid}")
        order[sess] += 1
        closed = bool(str(result or "").strip())
        oc = _outcomes(result, down, dist, yards) if closed else {k: None for k in (
            "offense_success", "yards", "touchdown", "turnover", "first_down", "explosive")}
        verified = str(est or "") == "identified" and str(ever or "") == "verified" and eform and eplay
        cand_id, reason = None, "no_verified_execution"
        if verified:
            macros = [m.strip() for m in str(emacro).split("+")] if emacro else None
            cand, reason = rc.candidate_for_execution(cat, eform, eplay, macros)
            if cand is None and reason == "macro_unconfirmed" and msid:
                rec = decided.get((sess, str(msid)))
                by = cat.by_id()
                if rec in by and by[rec].formation == eform and by[rec].base_play == eplay and standing_macro_attestation:
                    cand, reason = by[rec], "ok_standing_macro_attestation"
            execution = ("verified_ravens_execution" if cand is not None else
                         "verified_other_execution" if reason in ("not_two_formation", "play_not_in_scheme") else
                         "verified_ravens_macro_unconfirmed")
            cand_id = cand.candidate_id if cand is not None else None
        else:
            execution = "recommended_unexecuted" if (rform or rplay) else "unknown"
        out.append(DefSnap(
            int(rid), sess, str(opp or ""), "cpu" if is_cpu_opponent(str(opp or "")) else "human",
            str(dyn or ""), str(started or ts or ""), str(ts or ""), int(seq) if seq is not None else order[sess],
            int(seq) if seq is not None else None,
            int(down) if down is not None else None, int(dist) if dist is not None else None,
            int(yl) if yl is not None else None, classify(craw), craw,
            parse_offense_shape(sraw), closed, oc["offense_success"], oc["yards"], oc["explosive"],
            oc["touchdown"], oc["turnover"], oc["first_down"], execution, cand_id, reason))
    return out


def lions_offense_only_games(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT g.session_id, g.opponent_id, g.dynasty, g.started_ts, "
        "SUM(s.side='offense'), SUM(s.side='defense') FROM game_sessions g "
        "LEFT JOIN snaps s ON s.session_id = g.session_id GROUP BY g.session_id ORDER BY g.started_ts").fetchall()
    return [{"session": r[0], "opponent": r[1], "franchise": r[2], "started_ts": r[3],
             "offense_snaps": int(r[4] or 0)} for r in rows if int(r[5] or 0) == 0 and int(r[4] or 0) > 0]


def audit(conn: sqlite3.Connection, snaps: Sequence[DefSnap]) -> dict[str, Any]:
    by_exec = Counter(s.execution for s in snaps)
    observed = [s for s in snaps if s.concept is not None and s.closed]
    games = defaultdict(list)
    for s in snaps:
        games[s.session].append(s)
    per_game = []
    for sess, rows in sorted(games.items(), key=lambda kv: kv[1][0].started_ts):
        r0 = rows[0]
        per_game.append({"session": sess, "opponent": r0.opponent, "opponent_type": r0.opponent_type,
                         "franchise": r0.franchise, "started_ts": r0.started_ts, "defensive_snaps": len(rows),
                         "verified_ravens_executions": sum(x.execution == "verified_ravens_execution" for x in rows),
                         "recommended_unexecuted": sum(x.execution == "recommended_unexecuted" for x in rows),
                         "observed_opponent_concepts": sum(x.concept is not None and x.closed for x in rows),
                         "snap_seq_known": sum(x.seq is not None for x in rows)})
    offense_only = lions_offense_only_games(conn)
    return {
        "defensive_snaps": len(snaps),
        "games_with_defensive_snaps": len(games),
        "our_execution_axis": dict(by_exec),
        "verified_ravens_executions_with_outcome": sum(
            s.execution == "verified_ravens_execution" and s.offense_success is not None for s in snaps),
        "observed_opponent_plays": len(observed),
        "observed_opponent_plays_by_class": dict(Counter(s.concept for s in observed)),
        "closed_with_outcome": sum(s.offense_success is not None for s in snaps),
        "unknown_unverified_records": sum(s.execution in ("unknown", "verified_ravens_macro_unconfirmed")
                                          or (s.execution == "recommended_unexecuted" and s.concept is None)
                                          for s in snaps),
        "by_franchise": dict(Counter(s.franchise for s in snaps)),
        "by_opponent_type": dict(Counter(s.opponent_type for s in snaps)),
        "offense_shape_logged": sum(s.shape is not None for s in snaps),
        "per_game": per_game,
        "lions_offense_only_games_excluded": {"count": len(offense_only),
                                              "offense_snaps": sum(g["offense_snaps"] for g in offense_only)},
        "rule": "Only verified_ravens_execution rows with an outcome train candidate-specific terms.",
    }
