"""Typed live play overlay + natural situation phrases (v1.9.6)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.cli import build_parser
from cfb_coach.copilot import default_overlay_path, write_overlay_html
from cfb_coach.situation import parse_situation


class TestNaturalSituationPhrases(unittest.TestCase):
    def test_first_and_ten_my_35_last_play_cover(self) -> None:
        sit = parse_situation(
            "1st and 10 my 35 cover 2 (this was the last play)"
        )
        self.assertEqual(sit.down, 1)
        self.assertEqual(sit.distance, 10)
        self.assertEqual(sit.yardline, 35)
        self.assertEqual(sit.coverage_hint, "Cover 2")
        self.assertEqual(sit.coverage_source, "last")
        self.assertEqual(sit.side, "offense")
        self.assertFalse(sit.red_zone)

    def test_first_ampersand_our_and_opp_yardlines(self) -> None:
        own = parse_situation("1st & 10 our 35")
        self.assertEqual(own.down, 1)
        self.assertEqual(own.distance, 10)
        self.assertEqual(own.yardline, 35)
        self.assertEqual(own.side, "offense")

        opp = parse_situation("1st & 10 opp 40")
        self.assertEqual(opp.yardline, 60)  # 100 - 40
        self.assertEqual(opp.side, "offense")  # opp+digits ≠ defense side

        their = parse_situation("2&5 their 25")
        self.assertEqual(their.yardline, 75)
        self.assertFalse(their.red_zone)

        rz = parse_situation("1&10 opp 15")
        self.assertEqual(rz.yardline, 85)
        self.assertTrue(rz.red_zone)

    def test_ball_on_and_live_vs_last_coverage(self) -> None:
        ball = parse_situation("ball on 35")
        self.assertEqual(ball.yardline, 35)

        live_show = parse_situation("2&7 showing c2")
        self.assertEqual(live_show.coverage_hint, "Cover 2")
        self.assertEqual(live_show.coverage_source, "live")

        # Bare coverage = previous-play context (Aidan UX — no need to say last)
        bare = parse_situation("1&10 cover 2")
        self.assertEqual(bare.coverage_source, "last")

        live_word = parse_situation("1&10 live cover 2")
        self.assertEqual(live_word.coverage_source, "live")

        last = parse_situation("1&10 last cover 2")
        self.assertEqual(last.coverage_source, "last")

        showing = parse_situation("1&10 showing cover 2")
        self.assertEqual(showing.coverage_source, "live")
        self.assertEqual(showing.coverage_hint, "Cover 2")



class TestAidanCpuOffenseTyping(unittest.TestCase):
    """CPU offense: D&D (+ yl) + previous play name only — no 'last' word."""

    def test_mesh_spot_prev_concept(self) -> None:
        sit = parse_situation("1&10 my 35 mesh spot")
        self.assertEqual(sit.down, 1)
        self.assertEqual(sit.distance, 10)
        self.assertEqual(sit.yardline, 35)
        self.assertIn("Mesh", sit.concept_hint or "")
        self.assertEqual(sit.concept_source, "last")
        self.assertEqual(sit.coverage_source, "none")
        from cfb_coach.situation import format_heard
        self.assertIn("[prev:Mesh Spot]", format_heard(sit))

    def test_showing_cover_2_live(self) -> None:
        sit = parse_situation("1&10 showing cover 2")
        self.assertEqual(sit.coverage_hint, "Cover 2")
        self.assertEqual(sit.coverage_source, "live")
        from cfb_coach.situation import format_heard
        self.assertIn("[live:Cover 2]", format_heard(sit))

    def test_deep_flood_prev_concept(self) -> None:
        sit = parse_situation("2&7 deep flood")
        self.assertEqual(sit.down, 2)
        self.assertEqual(sit.distance, 7)
        self.assertEqual(sit.concept_hint, "Deep Flood")
        self.assertEqual(sit.concept_source, "last")
        from cfb_coach.situation import format_heard
        self.assertIn("[prev:Deep Flood]", format_heard(sit))

    def test_bare_cover_2_prev(self) -> None:
        sit = parse_situation("1&10 cover 2")
        self.assertEqual(sit.coverage_hint, "Cover 2")
        self.assertEqual(sit.coverage_source, "last")
        from cfb_coach.situation import format_heard
        self.assertEqual(format_heard(sit), "heard: 1&10 [prev:Cover 2]")

    def test_book_play_aliases(self) -> None:
        cases = [
            ("1&10 hb base", "HB Base"),
            ("2&5 counter y", "Counter Y"),
            ("1&10 inside zone", "Inside Zone"),
            ("3&8 four verticals", "Four Verticals"),
            ("2&7 cross wheels", "Cross Wheels"),
            ("1&10 whip trail", "Return Whip Trail"),
            ("1&10 mtn rpo", "Mtn RPO Zone Alert"),
        ]
        for raw, want in cases:
            sit = parse_situation(raw)
            self.assertEqual(sit.concept_hint, want, msg=raw)
            self.assertEqual(sit.concept_source, "last", msg=raw)


class TestPlayOverlay(unittest.TestCase):
    def test_write_play_overlay_without_look(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "play.html"
            out = write_overlay_html(
                str(path),
                None,
                [],
                call_text="Gun Bunch X Nasty — Inside Zone | No adj | Front → Cutback",
                short_line="1&10 my 35",
                mode="play",
            )
            html = Path(out).read_text(encoding="utf-8")
            self.assertIn("PLAY", html)
            self.assertIn("Inside Zone", html)
            self.assertIn("1&amp;10 my 35", html)
            self.assertIn("v1.9.6", html)
            self.assertNotIn("front=", html)

    def test_default_overlay_path(self) -> None:
        p = default_overlay_path()
        self.assertTrue(str(p).endswith("copilot_overlay.html"))
        self.assertIn(".cfb-coach", str(p))

    def test_cli_play_overlay_flags(self) -> None:
        p = build_parser()
        args = p.parse_args(["play", "--opponent", "gavin", "--no-overlay"])
        self.assertTrue(args.no_overlay)
        args2 = p.parse_args(
            ["play", "--opponent", "gavin", "--overlay", "/tmp/x.html"]
        )
        self.assertEqual(args2.overlay, "/tmp/x.html")
        self.assertFalse(getattr(args2, "no_overlay", False))

    def test_play_once_writes_overlay(self) -> None:
        from cfb_coach import cli as cli_mod

        with tempfile.TemporaryDirectory() as td:
            overlay = Path(td) / "ov.html"
            p = build_parser()
            args = p.parse_args(
                [
                    "play",
                    "--opponent",
                    "cpu",
                    "--once",
                    "1st and 10 my 35 cover 2 (this was the last play)",
                    "--overlay",
                    str(overlay),
                ]
            )
            # Avoid touching real DB home if possible — CoachDB uses env/default
            with mock.patch.object(cli_mod, "open_prep_html", create=True):
                rc = cli_mod.cmd_play(args)
            self.assertEqual(rc, 0)
            self.assertTrue(overlay.is_file())
            html = overlay.read_text(encoding="utf-8")
            self.assertIn("PLAY", html)
            # Call line should be present (formation em dash play)
            self.assertIn("—", html)


if __name__ == "__main__":
    unittest.main()


class TestSituationAwareCalls(unittest.TestCase):
    def test_format_heard_phrases(self) -> None:
        from cfb_coach.situation import format_heard, parse_situation

        self.assertEqual(
            format_heard(parse_situation("1st and 10 my 35")),
            "heard: 1&10 yl35",
        )
        self.assertEqual(
            format_heard(parse_situation("1st and 10 my 35 cover 2 last")),
            "heard: 1&10 yl35 [prev:Cover 2]",
        )
        self.assertEqual(
            format_heard(
                parse_situation("1st and 10 my 35 showing cover 2")
            ),
            "heard: 1&10 yl35 [live:Cover 2]",
        )
        self.assertEqual(
            format_heard(parse_situation("1&10 my 35 mesh spot")),
            "heard: 1&10 yl35 [prev:Mesh Spot]",
        )
        self.assertEqual(
            format_heard(parse_situation("1&10 cover 2")),
            "heard: 1&10 [prev:Cover 2]",
        )
        self.assertEqual(
            format_heard(
                parse_situation(
                    "1st and 10 my 35 cover 2 (this was the last play)"
                )
            ),
            "heard: 1&10 yl35 [prev:Cover 2]",
        )

    def test_make_call_uses_first_and_ten_not_gl(self) -> None:
        import random

        from cfb_coach.db import CoachDB, resolve_db_path_from_env
        from cfb_coach.playcaller import make_call
        from cfb_coach.situation import parse_situation

        sit = parse_situation("1st and 10 my 35")
        self.assertEqual(sit.down, 1)
        self.assertEqual(sit.distance, 10)
        self.assertEqual(sit.yardline, 35)
        self.assertFalse(sit.red_zone)
        self.assertFalse(sit.goal_line)

        db = CoachDB(resolve_db_path_from_env())
        try:
            gl_markers = ("GL/", "RZ short", "goal line", "bully run", "HB Dive")
            for seed in range(25):
                call = make_call(
                    parse_situation("1st and 10 my 35"),
                    "gavin",
                    db,
                    rng=random.Random(seed),
                )
                self.assertIn("sit 1&10 yl35", call.rationale)
                low = (call.rationale or "").lower()
                self.assertNotIn("gl/rz short", low)
                self.assertFalse(
                    any(m.lower() in low for m in ("gl/rz short", "goal-line")),
                    msg=call.rationale,
                )
                # Play itself should not be a GL dive package name
                self.assertNotEqual(call.play, "HB Dive")
        finally:
            db.close()

    def test_last_coverage_not_hard_counter_stamp(self) -> None:
        import random

        from cfb_coach.db import CoachDB, resolve_db_path_from_env
        from cfb_coach.playcaller import make_call
        from cfb_coach.situation import parse_situation

        sit = parse_situation("1st and 10 my 35 cover 2 last")
        self.assertEqual(sit.coverage_source, "last")
        db = CoachDB(resolve_db_path_from_env())
        try:
            call = make_call(sit, "gavin", db, rng=random.Random(1))
            self.assertIn("sit 1&10 yl35 last:Cover 2", call.rationale)
            # Last must not unlock REPEATED beater language
            self.assertNotIn("REPEATED", call.rationale)
        finally:
            db.close()
