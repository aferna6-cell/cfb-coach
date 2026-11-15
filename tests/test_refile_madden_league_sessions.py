"""Idempotent refile of the Chiefs-Colts cpu misfile and the Titans-Colts online misfile."""
from __future__ import annotations

import sqlite3
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from scripts.refile_madden_league_sessions import (  # noqa: E402
    CHIEFS,
    CHIEFS_COLTS_SESSION,
    CHIEFS_TARGET,
    COLTS,
    TITANS as TEN,
    TITANS_TARGET,
    main,
    run,
)

ET = ZoneInfo("America/New_York")


def _create(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE game_sessions (
            session_id TEXT PRIMARY KEY,
            opponent_id TEXT NOT NULL,
            dynasty TEXT,
            started_ts TEXT NOT NULL,
            ended_ts TEXT,
            notes TEXT,
            play_count INTEGER NOT NULL DEFAULT 0,
            league TEXT,
            user_team TEXT,
            opponent_team TEXT,
            opponent_type TEXT
        );
        CREATE TABLE snaps (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            opponent_id TEXT,
            side TEXT,
            league TEXT,
            user_team TEXT,
            opponent_team TEXT
        );
        CREATE TABLE ravens_defense_decisions (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            opponent_id TEXT,
            status TEXT,
            league TEXT,
            opponent_team TEXT
        );
        CREATE TABLE opponents (
            id TEXT PRIMARY KEY,
            display_name TEXT,
            team_now TEXT,
            skill TEXT,
            confidence TEXT,
            profile_json TEXT
        );
        INSERT INTO opponents (id, display_name, skill) VALUES ('cpu', 'CPU', 'cpu');
        """
    )
    conn.commit()
    return conn


def _insert_session(conn, session_id, started_ts, *, opponent_id="cpu", dynasty="online",
                    league=None, user_team=None, opponent_team=None, opponent_type=None):
    conn.execute(
        "INSERT INTO game_sessions (session_id, opponent_id, dynasty, started_ts, play_count, "
        "league, user_team, opponent_team, opponent_type) VALUES (?,?,?,?,1,?,?,?,?)",
        (session_id, opponent_id, dynasty, started_ts, league, user_team, opponent_team, opponent_type),
    )
    conn.execute(
        "INSERT INTO snaps (session_id, opponent_id, side, league, user_team, opponent_team) "
        "VALUES (?,?, 'offense', ?, ?, ?)",
        (session_id, opponent_id, league, user_team, opponent_team),
    )
    conn.execute(
        "INSERT INTO ravens_defense_decisions (session_id, opponent_id, status, league, opponent_team) "
        "VALUES (?,?, 'CALL', ?, ?)",
        (session_id, opponent_id, league, opponent_team),
    )
    conn.commit()


def _ts(year, month, day, hour, minute) -> str:
    return datetime(year, month, day, hour, minute, tzinfo=ET).astimezone(
        ZoneInfo("UTC")
    ).isoformat()


class RefileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self.id().replace(".", "_"))  # unused; we use TemporaryDirectory
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.db = Path(self._td.name) / "madden27.db"
        self.conn = _create(self.db)

    def tearDown(self) -> None:
        self.conn.close()
        self._td.cleanup()

    def _row(self, sid: str):
        self.conn.commit()
        # reopen to see writes from the script
        con = sqlite3.connect(self.db)
        con.row_factory = sqlite3.Row
        try:
            return con.execute("SELECT * FROM game_sessions WHERE session_id=?", (sid,)).fetchone()
        finally:
            con.close()

    def test_requires_db_and_preview_does_not_write(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            main([])
        self.assertEqual(cm.exception.code, 2)
        _insert_session(self.conn, CHIEFS_COLTS_SESSION, _ts(2026, 11, 10, 12, 0),
                        opponent_id="cpu", dynasty="online")
        self.assertEqual(run(self.db, apply=False), 0)
        row = self._row(CHIEFS_COLTS_SESSION)
        self.assertEqual(row["opponent_id"], "cpu")

    def test_chiefs_session_moves_to_colts_and_stays_in_chiefs_league(self) -> None:
        _insert_session(self.conn, CHIEFS_COLTS_SESSION, _ts(2026, 11, 10, 12, 0),
                        opponent_id="cpu", dynasty="online")
        # A later Chiefs-tagged Colts game — this is the Titans misfile.
        titans_sid = "abcd1234abcd1234"
        _insert_session(
            self.conn, titans_sid, _ts(2026, 11, 15, 16, 25),
            opponent_id="online_indianapolis_colts", dynasty="online",
            league="online", user_team=CHIEFS, opponent_team=COLTS, opponent_type="human",
        )
        self.assertEqual(run(self.db, apply=True), 0)
        chiefs = self._row(CHIEFS_COLTS_SESSION)
        for k, v in CHIEFS_TARGET.items():
            self.assertEqual(chiefs[k], v, k)
        titans = self._row(titans_sid)
        for k, v in TITANS_TARGET.items():
            self.assertEqual(titans[k], v, k)
        con = sqlite3.connect(self.db)
        try:
            snap_c = con.execute("SELECT opponent_id, league, user_team, opponent_team FROM snaps "
                                 "WHERE session_id=?", (CHIEFS_COLTS_SESSION,)).fetchone()
            self.assertEqual(tuple(snap_c), ("online_indianapolis_colts", "online", CHIEFS, COLTS))
            snap_t = con.execute("SELECT opponent_id, league, user_team, opponent_team FROM snaps "
                                 "WHERE session_id=?", (titans_sid,)).fetchone()
            self.assertEqual(tuple(snap_t), ("titans_indianapolis_colts", "titans", TEN, COLTS))
            dec_t = con.execute("SELECT opponent_id, league, opponent_team FROM ravens_defense_decisions "
                                "WHERE session_id=?", (titans_sid,)).fetchone()
            self.assertEqual(tuple(dec_t), ("titans_indianapolis_colts", "titans", COLTS))
            oids = {r[0] for r in con.execute("SELECT id FROM opponents")}
            self.assertIn("online_indianapolis_colts", oids)
            self.assertIn("titans_indianapolis_colts", oids)
        finally:
            con.close()
        # Idempotent second apply.
        self.assertEqual(run(self.db, apply=True), 0)
        chiefs2 = self._row(CHIEFS_COLTS_SESSION)
        self.assertEqual(chiefs2["league"], "online")
        self.assertEqual(chiefs2["user_team"], CHIEFS)
        self.assertEqual(self._row(titans_sid)["league"], "titans")

    def test_known_chiefs_session_is_never_moved_to_titans_even_after_cutover(self) -> None:
        # Even if its timestamp is after the cutover, identity wins.
        _insert_session(
            self.conn, CHIEFS_COLTS_SESSION, _ts(2026, 11, 15, 17, 0),
            opponent_id="cpu", dynasty="online",
        )
        self.assertEqual(run(self.db, apply=True), 0)
        row = self._row(CHIEFS_COLTS_SESSION)
        self.assertEqual(row["league"], "online")
        self.assertEqual(row["user_team"], CHIEFS)
        self.assertEqual(row["opponent_id"], "online_indianapolis_colts")

    def test_multiple_clear_candidates_are_refused(self) -> None:
        _insert_session(self.conn, CHIEFS_COLTS_SESSION, _ts(2026, 11, 10, 12, 0),
                        opponent_id="cpu")
        _insert_session(
            self.conn, "sess111111111111", _ts(2026, 11, 15, 16, 25),
            opponent_id="online_indianapolis_colts", dynasty="online",
            league="online", user_team=CHIEFS, opponent_team=COLTS, opponent_type="human",
        )
        _insert_session(
            self.conn, "sess222222222222", _ts(2026, 11, 15, 16, 40),
            opponent_id="online_indianapolis_colts", dynasty="online",
            league="online", user_team=CHIEFS, opponent_team=COLTS, opponent_type="human",
        )
        self.assertEqual(run(self.db, apply=True), 0)
        # Chiefs session is still applied; the ambiguous Titans pair is left alone.
        self.assertEqual(self._row(CHIEFS_COLTS_SESSION)["opponent_id"], "online_indianapolis_colts")
        self.assertEqual(self._row("sess111111111111")["league"], "online")
        self.assertEqual(self._row("sess222222222222")["league"], "online")

    def test_cpu_colts_after_cutover_moves_to_titans(self) -> None:
        _insert_session(self.conn, CHIEFS_COLTS_SESSION, _ts(2026, 11, 10, 12, 0),
                        opponent_id="cpu")
        sid = "cpuaftercutover01"
        _insert_session(self.conn, sid, _ts(2026, 11, 15, 16, 21), opponent_id="cpu", dynasty="online")
        self.assertEqual(run(self.db, apply=True), 0)
        row = self._row(sid)
        for k, v in TITANS_TARGET.items():
            self.assertEqual(row[k], v, k)
        self.assertEqual(self._row(CHIEFS_COLTS_SESSION)["user_team"], CHIEFS)
