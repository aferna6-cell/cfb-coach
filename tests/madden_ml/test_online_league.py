"""Madden online league (Chiefs): opponent-team tags, scout, inventory, Lions untouched."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden import data as mdata
from cfb_coach.madden import franchise
from cfb_coach.madden import online_league as ol
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.session import start_session

ROOT = Path(__file__).resolve().parents[2]
KC = "Kansas City Chiefs"


def _env(tmp: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "HOME": str(tmp),
        "CFB_COACH_DB": str(tmp / "coach.db"),
        "CFB_COACH_MADDEN_DB": str(tmp / "madden27.db"),
        "CFB_COACH_AI_RESEARCH": "off",
        "PYTHONPATH": str(ROOT),
    })
    env.pop("CFB_COACH_MADDEN_PRIMARY_TEAM", None)
    env.pop("CFB_COACH_OFFENSE_MODE", None)
    return env


def _cli(tmp: Path, *argv: str, stdin: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "cfb_coach", *argv],
        input=stdin, capture_output=True, text=True, env=_env(tmp), cwd=str(ROOT), timeout=180,
    )


def _apply_books(db_path: Path) -> None:
    from tests.madden_ml.live_book import apply_offense_book

    db = CoachDB(db_path, seed=mdata.load_seed())
    try:
        apply_offense_book(db, {"Gun Bunch": ["Inside Zone", "Mesh"]},
                           defense={"Nickel Over": ["Cover 3 Sky", "Cover 4 Quarters"]})
    finally:
        db.close()


class TeamResolution(unittest.TestCase):
    def test_chiefs_rejected_every_spelling(self) -> None:
        for raw in ("Kansas City Chiefs", "KC", "kc", "Chiefs", "kansas city"):
            with self.assertRaises(ValueError, msg=raw) as cm:
                ol.resolve_opponent_team(raw, KC)
            self.assertIn("your team", str(cm.exception))

    def test_31_other_teams_accepted_and_canonical(self) -> None:
        teams = ol.opponent_teams(KC)
        self.assertEqual(len(teams), 31)
        self.assertNotIn(KC, teams)
        for t in teams:
            self.assertEqual(ol.resolve_opponent_team(t, KC), t)
            self.assertEqual(ol.resolve_opponent_team(t.lower(), KC), t)
            self.assertEqual(ol.resolve_opponent_team(t.split()[-1], KC), t)

    def test_abbreviations(self) -> None:
        for raw, want in (("BUF", "Buffalo Bills"), ("sf", "San Francisco 49ers"), ("niners", "San Francisco 49ers"),
                          ("DET", "Detroit Lions"), ("wsh", "Washington Commanders"), ("LAR", "Los Angeles Rams")):
            self.assertEqual(ol.resolve_opponent_team(raw, KC), want)

    def test_missing_or_unknown_refused(self) -> None:
        for raw in (None, "", "Los Angeles", "Toronto Argonauts"):
            with self.assertRaises(ValueError):
                ol.resolve_opponent_team(raw, KC)


class ProfileConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self._old = os.environ.get("CFB_COACH_DB")
        os.environ["CFB_COACH_DB"] = str(Path(self._td.name) / "coach.db")
        os.environ.pop("CFB_COACH_MADDEN_PRIMARY_TEAM", None)

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("CFB_COACH_DB", None)
        else:
            os.environ["CFB_COACH_DB"] = self._old
        self._td.cleanup()

    def test_online_profile_is_chiefs_and_primary_stays_lions(self) -> None:
        self.assertEqual(franchise.normalize_profile("online"), franchise.ONLINE)
        self.assertEqual(franchise.team_for_profile("online"), KC)
        self.assertEqual(franchise.team_for_profile("primary"), "Detroit Lions")
        cfg = franchise.profile_config("online")
        self.assertEqual(cfg["team"], KC)
        self.assertFalse(cfg["allow_experimental_active"])
        franchise.save_config(online_team="KC")
        on_disk = json.loads(franchise.config_path().read_text())
        self.assertEqual(on_disk["primary_team"], "Detroit Lions")
        self.assertEqual(on_disk["online_team"], KC)
        self.assertEqual(franchise.load_config()["primary_team"], "Detroit Lions")

    def test_online_is_never_the_sticky_profile(self) -> None:
        db = CoachDB(Path(self._td.name) / "m.db", seed=mdata.load_seed())
        try:
            franchise.set_session_profile(db, "primary")
            self.assertEqual(franchise.set_session_profile(db, "online"), "online")
            self.assertEqual(franchise.get_session_profile(db), "primary")
            db.set_meta(franchise.META_KEY, "online")  # stale value from anywhere
            self.assertEqual(franchise.get_session_profile(db), "primary")
        finally:
            db.close()


class Scout(unittest.TestCase):
    def test_scout_uses_named_team_book(self) -> None:
        rep = ol.stock_scout("Buffalo Bills", KC)
        self.assertIn("NOT learned", rep["source"])
        self.assertEqual(rep["defense"]["book"], "Bills")
        self.assertEqual(rep["offense"]["book"], "Bills")
        self.assertGreater(rep["defense"]["plays"], 0)
        other = ol.stock_scout("San Francisco 49ers", KC)
        self.assertEqual(other["defense"]["book"], "49ers")
        self.assertNotEqual(rep["defense"]["plays"], 0)

    def test_chiefs_never_scouted(self) -> None:
        with self.assertRaises(ValueError):
            ol.stock_scout(KC, KC)

    def test_all_31_have_stock_books(self) -> None:
        for t in ol.opponent_teams(KC):
            rep = ol.stock_scout(t, KC)
            self.assertIsNotNone(rep["defense"], t)
            self.assertIsNotNone(rep["offense"], t)


class PersistAndInventory(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self._td.name) / "m.db", seed=mdata.load_seed())

    def tearDown(self) -> None:
        self.db.close()
        self._td.cleanup()

    def _online(self, oid: str, team: str) -> str:
        from cfb_coach.madden.human_opponents import register_cold_start

        register_cold_start(self.db, oid)
        ctx = ol.league_context(self.db, oid, team, cpu=False)
        sess = start_session(oid, dynasty="online", db=self.db, notes="t", league=ctx)
        return sess.session_id

    def _snap(self, oid: str, sid: str | None, side: str = "offense", **kw) -> None:
        self.db.log_snap(opponent_id=oid, side=side, situation_raw="1&10", our_call="x",
                         session_id=sid, down=1, distance=10, **kw)

    def test_session_and_snaps_tagged(self) -> None:
        sid = self._online("gt_one", "bills")
        self._snap("gt_one", sid, result="+6", coverage_seen="Cover 3")
        self._snap("gt_one", sid, side="defense", result="incomplete", concept_seen="Mesh")
        row = self.db.conn.execute("SELECT * FROM game_sessions WHERE session_id = ?", (sid,)).fetchone()
        self.assertEqual((row["league"], row["user_team"], row["opponent_team"], row["opponent_type"]),
                         ("online", KC, "Buffalo Bills", "human"))
        self.assertIn("NOT learned", json.loads(row["scout_json"])["stock"]["source"])
        snaps = self.db.conn.execute("SELECT league, user_team, opponent_team FROM snaps").fetchall()
        self.assertEqual([tuple(s) for s in snaps], [("online", KC, "Buffalo Bills")] * 2)

    def test_same_person_different_teams_keyed_by_session(self) -> None:
        a = self._online("gt_two", "BUF")
        self._snap("gt_two", a, result="+3")
        b = self._online("gt_two", "MIA")
        self._snap("gt_two", b, result="+3")
        inv = {r["team"]: r for r in ol.inventory(self.db, KC)}
        self.assertEqual((inv["Buffalo Bills"]["games"], inv["Buffalo Bills"]["snaps"]), (1, 1))
        self.assertEqual((inv["Miami Dolphins"]["games"], inv["Miami Dolphins"]["snaps"]), (1, 1))
        self.assertEqual(inv["Miami Dolphins"]["opponents"], ["gt_two"])

    def test_inventory_hides_chiefs_and_ignores_franchise_games(self) -> None:
        lions = start_session("cpu", dynasty="primary", db=self.db, notes="lions")
        self._snap("cpu", lions.session_id, result="+4")
        rows = ol.inventory(self.db, KC)
        self.assertEqual(len(rows), 31)
        self.assertNotIn(KC, [r["team"] for r in rows])
        self.assertTrue(all(r["games"] == 0 and r["status"] == "none yet" for r in rows))
        text = ol.format_inventory(self.db, KC)
        self.assertNotIn(f"\n{KC}", text)
        lions_row = self.db.conn.execute("SELECT league, opponent_team FROM game_sessions "
                                         "WHERE session_id = ?", (lions.session_id,)).fetchone()
        self.assertEqual(tuple(lions_row), (None, None))
        snap = self.db.conn.execute("SELECT league FROM snaps").fetchone()
        self.assertIsNone(snap[0])

    def test_logged_summary_feeds_next_scout(self) -> None:
        sid = self._online("gt_three", "Jets")
        self._snap("gt_three", sid, result="+12", coverage_seen="Cover 2")
        rep = ol.scout_report(self.db, "New York Jets", KC)
        self.assertEqual(rep["logged"]["games"], 1)
        self.assertEqual(rep["logged"]["coverage_seen"], {"Cover 2": 1})
        self.assertIn("coverage seen: Cover 2 1", ol.format_scout(rep))

    def test_html_live_controller_tags_session(self) -> None:
        from cfb_coach.madden.human_opponents import register_cold_start

        register_cold_start(self.db, "gt_html")
        ctx = ol.league_context(self.db, "gt_html", "Packers", cpu=False)
        ctrl = LivePlayController(
            db=self.db, opponent_id="gt_html", make_call=lambda *a, **k: None,
            parse_situation=parse_madden_situation, learn_summary=lambda: "",
            brand="Madden 27 Franchise", dynasty="online", league_context=ctx,
        )
        ctrl.start()
        row = self.db.conn.execute("SELECT league, user_team, opponent_team FROM game_sessions "
                                   "WHERE session_id = ?", (ctrl.session_id,)).fetchone()
        self.assertEqual(tuple(row), ("online", KC, "Green Bay Packers"))
        self.assertEqual(ctrl.state()["opponent_team"], "Green Bay Packers")


class BackwardCompatible(unittest.TestCase):
    def test_old_db_without_league_columns_migrates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "old.db"
            con = sqlite3.connect(path)
            con.executescript(
                """
                CREATE TABLE game_sessions (session_id TEXT PRIMARY KEY, opponent_id TEXT NOT NULL,
                    dynasty TEXT, started_ts TEXT NOT NULL, ended_ts TEXT, notes TEXT,
                    play_count INTEGER NOT NULL DEFAULT 0);
                INSERT INTO game_sessions VALUES ('old1', 'cpu', 'primary', '2026-11-01T00:00:00+00:00',
                    NULL, 'lions', 40);
                """
            )
            con.commit()
            con.close()
            db = CoachDB(path, seed=mdata.load_seed())
            try:
                row = db.conn.execute("SELECT * FROM game_sessions WHERE session_id='old1'").fetchone()
                self.assertEqual(row["dynasty"], "primary")
                self.assertEqual(row["play_count"], 40)
                self.assertIsNone(row["league"])
                self.assertIsNone(row["opponent_team"])
                self.assertEqual(sum(r["games"] for r in ol.inventory(db, KC)), 0)
            finally:
                db.close()


class CliEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        CoachDB(self.tmp / "madden27.db", seed=mdata.load_seed()).close()
        _apply_books(self.tmp / "madden27.db")

    def tearDown(self) -> None:
        self._td.cleanup()

    def _play(self, *extra: str, stdin: str = "q\n") -> subprocess.CompletedProcess:
        return _cli(self.tmp, "play", "--game", "madden27", "--terminal", "--no-overlay", *extra, stdin=stdin)

    def test_online_requires_team_and_refuses_chiefs(self) -> None:
        r = self._play("--franchise", "online", "-o", "gamer_a", "--heuristic-offense")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("--opponent-team is required", r.stderr)
        r = self._play("--franchise", "online", "-o", "gamer_a", "--opponent-team", "KC", "--heuristic-offense")
        self.assertEqual(r.returncode, 2)
        self.assertIn("your team", r.stderr)

    def test_opponent_team_outside_online_refused(self) -> None:
        r = self._play("-o", "cpu", "--opponent-team", "BUF")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--franchise online", r.stderr)

    def test_online_game_logs_tags_and_leaves_lions_alone(self) -> None:
        stdin = "1&10 my 25\n+6 cover 3\n2&4 my 31\n+5\nq\n"
        r = self._play("--franchise", "online", "-o", "gamer_b", "--opponent-team", "BUF",
                       "--heuristic-offense", stdin=stdin)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ONLINE LEAGUE: Kansas City Chiefs vs Buffalo Bills", r.stdout)
        self.assertIn("stock-book scout", r.stdout)
        con = sqlite3.connect(self.tmp / "madden27.db")
        try:
            sess = con.execute("SELECT dynasty, league, user_team, opponent_team, opponent_type "
                               "FROM game_sessions").fetchall()
            self.assertEqual(sess, [("online", "online", KC, "Buffalo Bills", "human")])
            snaps = con.execute("SELECT league, user_team, opponent_team, opponent_id FROM snaps").fetchall()
            self.assertGreaterEqual(len(snaps), 2)
            self.assertTrue(all(s == ("online", KC, "Buffalo Bills", "gamer_b") for s in snaps))
            sticky = con.execute("SELECT value FROM meta WHERE key = 'franchise_profile'").fetchall()
            self.assertNotIn(("online",), sticky)
        finally:
            con.close()
        cfg = _cli(self.tmp, "config", "--game", "madden27")
        self.assertIn("primary team (franchise_primary): Detroit Lions", cfg.stdout)
        self.assertIn("online team  (online league):     Kansas City Chiefs", cfg.stdout)
        inv = _cli(self.tmp, "league", "--game", "madden27")
        self.assertEqual(inv.returncode, 0, inv.stderr)
        self.assertIn("Opponent teams with data: 1/31", inv.stdout)
        self.assertNotIn("Kansas City Chiefs        ", inv.stdout)
        bills = [ln for ln in inv.stdout.splitlines() if ln.startswith("Buffalo Bills")]
        self.assertEqual(len(bills), 1)
        self.assertIn("gamer_b", bills[0])


if __name__ == "__main__":
    unittest.main()
