"""A play or result with no new down is the snap that just happened.

Terminal (`play --terminal`) and the live form both log it on the pending call,
move the ball, and let the next call — including a repeated-concept macro — see it.
`undo` removes that note. A line that still has a down stays a new situation.
"""

from __future__ import annotations

import io
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from cfb_coach.cli import main
from cfb_coach.db import CoachDB
from cfb_coach.games import madden_db_path
from cfb_coach.last_snap import BallSpot, advance_ball, classify_last_snap
from cfb_coach.situation import parse_situation


class _Isolated(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)
        env = {"CFB_COACH_DB": str(self.dir / "coach.db"), "HOME": str(self.dir)}
        self._env = mock.patch.dict(os.environ, env, clear=False)
        self._env.start()
        for key in ("CFB_COACH_MADDEN_DB", "CFB_COACH_MADDEN_PRIMARY_TEAM", "CFB_COACH_MADDEN_LAB_TEAM"):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        self._env.stop()
        self._td.cleanup()

    def play(self, argv: list[str], lines: list[str]) -> tuple[int, str]:
        feed = iter(lines)

        def _input(*_a, **_k):
            try:
                return next(feed)
            except StopIteration:
                raise EOFError

        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch("builtins.input", _input), mock.patch(
            "cfb_coach.prep_browser.open_prep_html"
        ):
            rc = main(argv)
        return rc, buf.getvalue()


class TestClassifyAndAdvance(unittest.TestCase):
    def test_play_or_result_is_the_last_snap_and_a_down_is_a_new_call(self) -> None:
        mesh = classify_last_snap("mesh", parse_situation)
        self.assertEqual(mesh["concept"], "Mesh")
        self.assertIsNone(mesh["result"])
        # "4 verts" is the concept, not a 4-yard gain.
        verts = classify_last_snap("4 verts", parse_situation, side="defense")
        self.assertIsNone(verts["result"])
        self.assertEqual(verts["concept"], "Four Verticals")
        self.assertEqual(classify_last_snap("+7", parse_situation)["result"], "+7")
        self.assertEqual(classify_last_snap("td", parse_situation)["result"], "td")
        self.assertEqual(classify_last_snap("int", parse_situation)["result"], "int")
        self.assertEqual(classify_last_snap("+7 mesh", parse_situation)["concept"], "Mesh")
        # A live look, and any line that still has a down, is the next call.
        self.assertIsNone(classify_last_snap("showing cover 2", parse_situation))
        self.assertIsNone(classify_last_snap("2&7 mesh", parse_situation))
        self.assertIsNone(classify_last_snap("1&10 my 35 mesh spot", parse_situation))

    def test_ball_advances_from_the_result(self) -> None:
        sit = parse_situation("1&10 my 25")
        spot, note = advance_ball(sit, "+7", "offense", BallSpot())
        self.assertEqual((spot.down, spot.distance, spot.yardline), (2, 3, 32))
        self.assertIn("2&3 yl32", note)
        sit_d = parse_situation("d 1&10 my 25")
        spot_d, _ = advance_ball(sit_d, "+7", "defense", BallSpot())
        self.assertEqual((spot_d.down, spot_d.distance, spot_d.yardline), (2, 3, 18))
        td, td_note = advance_ball(sit, "td", "offense", BallSpot(score_us=21, score_them=14))
        self.assertEqual((td.score_us, td.score_them), (27, 14))
        self.assertIn("27-14", td_note)
        turned, _ = advance_ball(sit, "int", "offense", BallSpot(down=1, distance=10, yardline=25))
        self.assertEqual(turned.yardline, 25)
        fourth = parse_situation("4&5 my 40")
        dead, dead_note = advance_ball(fourth, "+1", "offense", BallSpot())
        self.assertIsNone(dead.down)
        self.assertIn("turnover on downs", dead_note)


class TestCfbTerminal(_Isolated):
    def test_mesh_and_gain_log_the_last_snap_and_undo_removes_it(self) -> None:
        rc, out = self.play(
            ["play", "-o", "gavin", "--terminal", "--no-overlay"],
            ["1&10 my 25", "mesh", "+7", "showing cover 2", "undo", "undo", "q"],
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("logged last snap: Mesh", out)
        self.assertIn("logged last snap: +7 | Mesh | now 2&3 yl32", out)
        # The play and the gain are not new calls. The live look is, at the new spot.
        self.assertEqual(out.count("heard:"), 2)
        self.assertIn("heard: 2&3 yl32 [live:Cover 2]", out)
        self.assertIn("undo: removed the last snap note", out)
        db = CoachDB(self.dir / "coach.db")
        try:
            self.assertEqual(db.count_snaps("gavin"), 0)
        finally:
            db.close()

    def test_down_and_play_stays_a_new_call_until_result(self) -> None:
        rc, out = self.play(
            ["play", "-o", "gavin", "--terminal", "--no-overlay"],
            ["1&10 my 35 mesh spot", "result +4", "2&7 mesh", "q"],
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("heard: 1&10 yl35 [prev:Mesh Spot]", out)
        self.assertIn("logged last snap: +4 | Mesh Spot | now 2&6 yl39", out)
        self.assertIn("heard: 2&7 yl39 [prev:Mesh]", out)
        db = CoachDB(self.dir / "coach.db")
        try:
            rows = db.get_recent_snaps("gavin", limit=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["concept_seen"], "Mesh Spot")
            self.assertEqual(rows[0]["result"], "+4")
            self.assertIn("1&10", rows[0]["situation_raw"])
        finally:
            db.close()

    def test_repeated_verts_count_and_arm_vert(self) -> None:
        lines = [
            "d 1&10", "4 verts", "+15",
            "d 2&8", "4 verts", "+18",
            "d 2&6 showing 4 verts", "q",
        ]
        rc, out = self.play(["play", "-o", "gavin", "--terminal", "--no-overlay"], lines)
        self.assertEqual(rc, 0, out)
        self.assertEqual(out.count("logged last snap"), 4)
        call = out[out.rfind("heard: 2&6"):]
        self.assertIn("[live:Four Verticals]", call)
        self.assertIn("| VERT |", call, call)
        db = CoachDB(self.dir / "coach.db")
        try:
            rows = list(reversed(db.get_recent_snaps("gavin", side="defense", limit=5)))
            self.assertEqual([r["concept_seen"] for r in rows], ["Four Verticals", "Four Verticals"])
            tend = db.get_tendencies("gavin", "offense_concept")
            verts = [r for r in tend if r["key"] == "Four Verticals"]
            self.assertEqual(len(verts), 1)
            self.assertEqual(verts[0]["count"], 2)
        finally:
            db.close()

    def test_cover_2_bumps_their_coverage_once(self) -> None:
        rc, out = self.play(
            ["play", "-o", "gavin", "--terminal", "--no-overlay"],
            ["1&10 my 25", "cover 2", "undo", "q"],
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("logged last snap: Cover 2", out)
        db = CoachDB(self.dir / "coach.db")
        try:
            self.assertEqual(db.count_snaps("gavin"), 0)
            rows = db.get_tendencies("gavin", "their_coverage")
            cover = [r for r in rows if r["key"] == "Cover 2"]
            self.assertEqual(cover[0]["count"], 0)
        finally:
            db.close()


class TestMaddenTerminal(_Isolated):
    def test_repeated_verts_arm_safe_deep_and_undo_drops_the_snap(self) -> None:
        rc, _ = self.play(
            ["prep", "--game", "madden27", "-o", "james", "--offline", "--text"],
            [],
        )
        self.assertEqual(rc, 0)
        lines = [
            "d 1&10", "4 verts", "+15",
            "d 2&8", "4 verts", "+18",
            "d 2&6 showing 4 verts",
            "undo", "q",
        ]
        rc, out = self.play(
            ["play", "--game", "madden27", "-o", "james", "--terminal", "--no-overlay"],
            lines,
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("logged last snap: Four Verticals", out)
        self.assertIn("logged last snap: +15 | Four Verticals", out)
        call = out[out.rfind("heard:"):]
        self.assertIn("[live:Four Verticals]", call)
        self.assertIn("SAFE DEEP", call, call)
        self.assertIn("LB → SAFE DEEP", call)
        self.assertIn("undo: removed the last snap note", out)
        db = CoachDB(madden_db_path())
        try:
            rows = db.get_recent_snaps("james", side="defense", limit=5)
            # undo removed the +18 update; the second snap is still the concept-only note
            # until a second undo. One undo is the result. Both concepts stay.
            self.assertEqual(
                [r["concept_seen"] for r in rows],
                ["Four Verticals", "Four Verticals"],
            )
        finally:
            db.close()


class TestBrowserForm(_Isolated):
    def _ctrl(self):
        from cfb_coach.live_server import LivePlayController

        class Call:
            def __init__(self, sit):
                self.side = sit.side
                self.formation = "Gun"
                self.play = "Mesh Spot"
                self.macro = None

            def format(self):
                return f"{self.formation} — {self.play}"

        db = CoachDB(self.dir / "coach.db")
        self.addCleanup(db.close)
        ctrl = LivePlayController(
            db=db, opponent_id="gavin", make_call=lambda sit, **k: Call(sit),
            parse_situation=parse_situation, learn_summary=lambda: "", session_id="s1",
        )
        return ctrl

    def test_look_lands_on_the_snap_just_played_and_the_ball_moves(self) -> None:
        ctrl = self._ctrl()
        ctrl.call_only("1&10 my 25")
        res = ctrl.result_and_call(outcome="+7", sit_raw="1&10 my 25 mesh", their="")
        self.assertTrue(res["ok"], res)
        self.assertIn("2&3", res["heard"])
        self.assertIn("yl32", res["heard"])
        self.assertEqual(res["state"]["ball"]["down"], 2)
        self.assertEqual(res["state"]["ball"]["distance"], 3)
        self.assertEqual(res["state"]["ball"]["yardline"], 32)
        rows = ctrl.db.get_session_snaps("s1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["result"], "+7")
        self.assertEqual(rows[0]["concept_seen"], "Mesh")
        self.assertIn("1&10", rows[0]["situation_raw"])
        # They typed the next down themselves: that wins over the computed spot.
        moved = ctrl.result_and_call(outcome="+4", sit_raw="3&2 opp 40", their="cover 2")
        self.assertIn("3&2", moved["heard"])
        self.assertEqual(moved["state"]["ball"]["down"], 3)
        rows = ctrl.db.get_session_snaps("s1")
        self.assertEqual(rows[1]["coverage_seen"], "Cover 2")
        self.assertNotEqual(rows[1]["concept_seen"], "Mesh")
        undone = ctrl.undo_last()
        self.assertIn("undo", undone["message"])
        self.assertEqual(len(ctrl.db.get_session_snaps("s1")), 1)
        self.assertEqual(len(undone["state"]["log"]), 1)

    def test_td_adds_six_and_int_does_not_flip_the_field(self) -> None:
        ctrl = self._ctrl()
        ctrl.call_only("1&10 my 25", score_us=7, score_them=0, score_set=True)
        td = ctrl.result_and_call(outcome="td", sit_raw="1&10 my 25")
        self.assertEqual(td["state"]["ball"]["score_us"], 13)
        self.assertEqual(td["state"]["live_score"]["us"], 13)
        ctrl.call_only("1&10 my 40")
        kept = ctrl.result_and_call(outcome="int", sit_raw="1&10 my 40")
        self.assertEqual(kept["state"]["ball"]["yardline"], 40)
        self.assertEqual(kept["state"]["ball"]["down"], 1)
        self.assertEqual(ctrl.db.get_session_snaps("s1")[-1]["result"], "int")

    def test_html_has_undo_and_writes_the_ball_into_the_form(self) -> None:
        from cfb_coach.live_server import render_live_html

        html = render_live_html(self._ctrl())
        self.assertIn('id="btn-undo"', html)
        self.assertIn("/api/undo", html)
        self.assertIn("st.ball", html)


class TestMaddenBrowserMacro(_Isolated):
    def test_two_logged_verts_arm_safe_deep_on_the_next_live_look(self) -> None:
        from cfb_coach.live_server import LivePlayController
        from cfb_coach.madden.playcaller import make_call
        from cfb_coach.madden.prep import build_prep_plan
        from cfb_coach.madden.situation import parse_madden_situation

        db = CoachDB(madden_db_path(), seed=__import__(
            "cfb_coach.madden.data", fromlist=["load_seed"]
        ).load_seed())
        self.addCleanup(db.close)
        build_prep_plan("james", db=db, offline=True)

        def _make(sit, **_k):
            return make_call(sit, "james", db, rng=random.Random(1))

        ctrl = LivePlayController(
            db=db, opponent_id="james", make_call=_make,
            parse_situation=parse_madden_situation, learn_summary=lambda: "", session_id="m1",
        )
        ctrl.call_only("d 1&10", side="defense")
        ctrl.result_and_call(outcome="+15", sit_raw="d 2&8 4 verts", side="defense")
        ctrl.result_and_call(outcome="+18", sit_raw="d 2&6 4 verts", side="defense")
        rows = db.get_session_snaps("m1")
        self.assertEqual([r["concept_seen"] for r in rows], ["Four Verticals", "Four Verticals"])
        live = ctrl.call_only("d 2&6 showing 4 verts", side="defense")
        self.assertIn("SAFE DEEP", live["call_text"], live["call_text"])
        self.assertIn("LB → SAFE DEEP", live["call_text"])
        self.assertEqual(len(db.get_session_snaps("m1")), 2)
