"""Live us-them score: optional input, session persistence, score-sensitive call mix."""

from __future__ import annotations

import random
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from cfb_coach.game_score import (
    GameScore,
    LiveContext,
    absorb_and_stamp,
    classify,
    context_from_args,
    interpret_live_command,
    offense_score_bonus,
    shift_defense_mix,
)
from cfb_coach.situation import format_heard, parse_situation


def _stamp(raw: str, us: int | None, them: int | None, quarter: int | None = None, *, side: str = "offense"):
    sit = parse_situation(raw, default_side=side)
    if side:
        sit.side = side
    ctx = LiveContext(
        score=None if us is None else GameScore(us, them or 0),
        quarter=quarter,
    )
    absorb_and_stamp(sit, ctx)
    return sit


class TestParseAndSession(unittest.TestCase):
    def test_us_them_formats_and_bad_input(self):
        self.assertEqual(GameScore(21, 14).label, "21-14")
        for text in ("21-14", "21 to 14", "us 21 them 14", "21 14"):
            got = context_from_args(Namespace(score=text, quarter=None)).score
            self.assertEqual((got.us, got.them), (21, 14), text)
        with self.assertRaises(SystemExit):
            context_from_args(Namespace(score="lots", quarter=None))
        self.assertIsNone(context_from_args(Namespace(score=None, quarter=4)).score)
        self.assertEqual(context_from_args(Namespace(score=None, quarter=4)).quarter, 4)

    def test_down_and_distance_still_parses_and_score_is_optional(self):
        sit = parse_situation("4th and 1 my 40")
        self.assertEqual((sit.down, sit.distance), (4, 1))
        self.assertIsNone(sit.score_us)
        self.assertNotIn("quarter", sit.extras)
        heard = format_heard(parse_situation("1&10 my 35 mesh spot"))
        self.assertEqual(heard, "heard: 1&10 yl35 [prev:Mesh Spot]")

    def test_inline_score_and_quarter_and_session_update(self):
        sit = parse_situation("2&6 my 40 q4 score 21-17")
        self.assertEqual((sit.score_us, sit.score_them, sit.extras.get("quarter")), (21, 17, 4))
        self.assertIn("Q4", format_heard(sit))
        self.assertIn("21-17", format_heard(sit))
        ctx = LiveContext()
        absorb_and_stamp(sit, ctx)
        nxt = parse_situation("1&10 my 30")
        absorb_and_stamp(nxt, ctx)
        self.assertEqual((nxt.score_us, nxt.score_them, nxt.extras["quarter"]), (21, 17, 4))
        # This snap's score replaces the session score
        upd = parse_situation("3&8 opp 40 score 17-24")
        absorb_and_stamp(upd, ctx)
        self.assertEqual(ctx.score.label, "17-24")

    def test_terminal_commands_persist_until_cleared(self):
        ctx = LiveContext()
        self.assertIsNone(interpret_live_command("1&10 my 25", ctx))
        self.assertIn("21-14", interpret_live_command("score 21-14", ctx))
        self.assertEqual(ctx.score.label, "21-14")
        self.assertIn("Q4", interpret_live_command("quarter 4", ctx))
        self.assertIn("cleared", interpret_live_command("score clear", ctx))
        self.assertIsNone(ctx.score)
        self.assertEqual(ctx.quarter, 4)
        self.assertIn("not understood", interpret_live_command("score nope", ctx))


class TestModulation(unittest.TestCase):
    def test_early_close_is_neutral_and_late_score_moves_offense(self):
        early = _stamp("1&10 my 25", 14, 10, 2)
        self.assertEqual(classify(early).phase, "neutral")
        self.assertEqual(offense_score_bonus(early, "Deep Flood", deep=True), 0.0)
        self.assertEqual(offense_score_bonus(early, "Inside Zone", run=True), 0.0)

        up = _stamp("1&10 my 25", 28, 7, 4)
        down = _stamp("1&10 my 25", 7, 28, 4)
        self.assertEqual(classify(up).phase, "prevent")
        self.assertEqual(classify(down).phase, "trail")
        self.assertGreater(
            offense_score_bonus(up, "Inside Zone", run=True),
            offense_score_bonus(up, "Deep Flood", deep=True),
        )
        self.assertLess(offense_score_bonus(up, "Deep Flood", deep=True), 0)
        self.assertGreater(
            offense_score_bonus(down, "Deep Flood", deep=True),
            offense_score_bonus(down, "Inside Zone", run=True),
        )
        self.assertGreater(offense_score_bonus(down, "Deep Flood", deep=True), 0.1)

    def test_third_and_long_keeps_the_pass_lean_when_up_big(self):
        from cfb_coach.playcaller import _book_menu

        book = {"formations": {"Gun Bunch X Nasty": ["Inside Zone", "Deep Flood", "Mesh Spot"]}}
        plain = parse_situation("3&12 my 20")
        up = _stamp("3&12 my 20", 28, 7, 4)
        _, b0 = _book_menu(plain, [], book)
        _, b1 = _book_menu(up, [], book)
        form = "Gun Bunch X Nasty"
        # D&D still prefers the pass; score only trims the shot, it does not flip to a run
        self.assertGreater(b1[(form, "Deep Flood")], b1[(form, "Inside Zone")])
        self.assertLess(b1[(form, "Deep Flood")], b0[(form, "Deep Flood")])

    def test_no_book_trailing_adds_shot_plays(self):
        from cfb_coach.playcaller import _is_deep, make_call

        def deep_rate(us: int, them: int, quarter: int) -> float:
            n = 0
            for i in range(24):
                call = make_call(
                    _stamp("1&10 my 35", us, them, quarter), "cpu", None, rng=random.Random(i),
                )
                if _is_deep(call.play):
                    n += 1
            return n / 24

        self.assertGreater(deep_rate(7, 31, 4), deep_rate(31, 7, 4) + 0.2)
        self.assertEqual(deep_rate(14, 10, 1), 0.0)

    def test_goal_line_score_bonus_stays_small(self):
        sit = _stamp("1&goal opp 4", 7, 28, 4)
        self.assertLess(abs(offense_score_bonus(sit, "HB Dive", run=True)), 0.08)
        self.assertLess(abs(offense_score_bonus(sit, "Verticals", deep=True)), 0.08)

    def test_cfb_defense_prevent_vs_aggression_and_packages_stay(self):
        from cfb_coach.playcaller import make_call

        up = make_call(_stamp("1&10 my 40", 28, 7, 4, side="defense"), "gavin", None, rng=random.Random(0))
        down = make_call(_stamp("1&10 my 40", 7, 28, 4, side="defense"), "gavin", None, rng=random.Random(0))
        early = make_call(_stamp("1&10 my 40", 14, 10, 1, side="defense"), "gavin", None, rng=random.Random(0))
        plain = make_call(parse_situation("1&10 my 40", default_side="defense"), "gavin", None, rng=random.Random(0))
        self.assertEqual(up.play, "Cover 4 Quarters")
        self.assertIn("prevent", up.rationale)
        self.assertEqual(down.play, "Mid Blitz 0")
        self.assertIn("HEAT", down.adj_or_macro)
        self.assertIn("trailing", down.rationale)
        self.assertEqual(early.play, plain.play)
        self.assertEqual(early.formation, plain.formation)
        self.assertIn("neutral", early.rationale)

        gl = make_call(_stamp("1&goal opp 3", 28, 7, 4, side="defense"), "gavin", None, rng=random.Random(1))
        self.assertIn("4-3", gl.formation)
        short = make_call(_stamp("3&1 my 40", 7, 28, 4, side="defense"), "gavin", None, rng=random.Random(1))
        self.assertIn("4-3", short.formation)
        self.assertNotIn("HEAT", short.adj_or_macro)

    def test_fourth_down_and_two_point_notes(self):
        from cfb_coach.playcaller import make_call

        go = make_call(_stamp("4&6 opp 45", 10, 24, 4), "cpu", None, rng=random.Random(0))
        self.assertIn("go for it", go.rationale)
        safe = make_call(_stamp("4&6 my 30", 24, 10, 4), "cpu", None, rng=random.Random(0))
        self.assertIn("don't force", safe.rationale)
        two = make_call(_stamp("1&10 opp 12 q4 score 14-22", None, None), "cpu", None, rng=random.Random(0))
        self.assertIn("2-pt live", two.rationale)

    def test_madden_mix_and_offense_bonus(self):
        from cfb_coach.madden.defense_select import MIX, select_defense
        from cfb_coach.madden.playcaller import _offense_pool

        base = dict(MIX["early"])
        up = shift_defense_mix(base, _stamp("1&10 my 40", 28, 7, 4, side="defense"))
        down = shift_defense_mix(base, _stamp("1&10 my 40", 7, 28, 4, side="defense"))
        early = shift_defense_mix(base, _stamp("1&10 my 40", 7, 7, 1, side="defense"))
        self.assertGreater(up["two_high"], down["two_high"])
        self.assertGreater(down["pressure"], up["pressure"])
        self.assertEqual(early, base)
        self.assertAlmostEqual(sum(up.values()), 1.0, places=5)

        book = {
            "Nickel Over": ["Cover 4 Quarters", "Cover 3 Sky", "Tampa 2", "Cover 1"],
            "Nickel Double Mug": ["Mid Blitz"],
        }

        def pressure_rate(us: int, them: int) -> float:
            n = 0
            for i in range(40):
                sit = _stamp("1&10 my 40", us, them, 4, side="defense")
                pick = select_defense(
                    sit, "gavin", None, book, random.Random(i), scouting=None, research_counters=[],
                )
                if pick.family == "pressure":
                    n += 1
            return n / 40

        self.assertGreater(pressure_rate(7, 28), pressure_rate(28, 7) + 0.15)

        obook = {"Gun Trips": ["Inside Zone", "Four Verticals"]}
        og = {"situations": {}}
        up_o = _stamp("1&10 my 25", 28, 7, 4)
        dn_o = _stamp("1&10 my 25", 7, 28, 4)
        _, bu = _offense_pool(up_o, obook, og, "early_down", None)
        _, bd = _offense_pool(dn_o, obook, og, "early_down", None)
        self.assertGreater(bu[("Gun Trips", "Inside Zone")], bu[("Gun Trips", "Four Verticals")])
        self.assertGreater(bd[("Gun Trips", "Four Verticals")], bd[("Gun Trips", "Inside Zone")])


class TestLiveSessionAndLearn(unittest.TestCase):
    def test_html_controller_keeps_score_until_updated(self):
        from cfb_coach.db import CoachDB
        from cfb_coach.live_server import LivePlayController, render_live_html
        from cfb_coach.playcaller import make_call

        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "coach.db")
            ctrl = LivePlayController(
                db=db,
                opponent_id="gavin",
                make_call=lambda sit, **kw: make_call(sit, "gavin", db, **kw),
                parse_situation=parse_situation,
                learn_summary=lambda: "",
                live_score=GameScore(14, 14),
                quarter=1,
            )
            html = render_live_html(ctrl)
            self.assertIn('id="score-us"', html)
            self.assertIn('value="14"', html)
            first = ctrl.call_only("1&10 my 30", side="defense", quarter=4, score_us=7, score_them=28, score_set=True)
            self.assertTrue(first["ok"])
            self.assertIn("trailing", first["call_text"] + ctrl.last_call.rationale)
            second = ctrl.call_only("2&8 my 35", side="defense", quarter=4)
            self.assertEqual(second["state"]["live_score"]["label"], "7-28")
            self.assertEqual(ctrl.last_sit.score_us, 7)
            self.assertIn("trailing", ctrl.last_call.rationale)
            ctrl.call_only("1&10 my 40", side="offense", quarter=4, score_us=28, score_them=7, score_set=True)
            self.assertIn("clock", ctrl.last_call.rationale)
            ctrl.call_only("1&10 my 40", quarter=1, score_us="", score_them="", score_set=True)
            self.assertIsNone(ctrl.live_score)
            self.assertNotIn("score ", ctrl.last_call.rationale)
            db.close()

    def test_logged_live_score_tags_retrain_and_final_score_still_parses(self):
        from cfb_coach import learning as L
        from cfb_coach.db import CoachDB

        class Sess:
            def __init__(self):
                self.session_id = "g1"
                self.opponent_id = "cpu"
                self.dynasty = "alabama"
                self.started_ts = datetime.now(timezone.utc).isoformat()
                self.ended_ts = None
                self.notes = ""
                self.play_count = 1

            def to_dict(self):
                return {
                    "session_id": self.session_id,
                    "opponent_id": self.opponent_id,
                    "dynasty": self.dynasty,
                    "started_ts": self.started_ts,
                    "ended_ts": self.ended_ts,
                    "notes": self.notes,
                    "play_count": self.play_count,
                }

        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "coach.db")
            db.start_game_session(Sess())
            db.log_snap(
                opponent_id="cpu", side="offense", situation_raw="1&10 q4 score 7-21",
                our_call="x", formation="Gun Bunch X Nasty", play="Deep Flood",
                down=1, distance=10, yardline=40, quarter=4, result="+12",
                session_id="g1", notes="score_us=7 score_them=21",
            )
            db.log_snap(
                opponent_id="cpu", side="offense", situation_raw="1&10",
                our_call="x", formation="Gun Bunch X Nasty", play="Mesh Spot",
                down=1, distance=10, yardline=50, result="+5", session_id="g1",
            )
            db.end_game_session("g1", result_wl="loss", score="21-7")
            comp = L.compute_all(db.conn)
            tagged = [e for e in comp["evals"] if "live_score" in e.tags]
            plain = [e for e in comp["evals"] if "live_score" not in e.tags]
            self.assertEqual(len(tagged), 1)
            self.assertIn("score_sensitive", tagged[0].tags)
            self.assertGreater(tagged[0].leverage, plain[0].leverage)
            parsed = comp["results"]["g1"]["parsed"]
            self.assertEqual(parsed["ours"], 7)  # winner-first final score + a loss
            self.assertEqual(parsed["theirs"], 21)
            db.close()


class TestCliFlags(unittest.TestCase):
    def test_play_and_call_accept_score_for_both_games(self):
        from cfb_coach.cli import build_parser

        p = build_parser()
        cfb = p.parse_args(["call", "-o", "gavin", "-s", "1&10", "--score", "21-14", "--quarter", "4"])
        self.assertEqual(cfb.score, "21-14")
        self.assertEqual(cfb.quarter, 4)
        self.assertEqual(cfb.game, "cfb27")
        mad = p.parse_args([
            "play", "--game", "madden27", "-o", "gavin", "--score", "3-10", "--quarter", "5", "--once", "1&10",
        ])
        self.assertEqual(mad.game, "madden27")
        self.assertEqual(mad.score, "3-10")
        self.assertEqual(mad.quarter, 5)


if __name__ == "__main__":
    unittest.main()
