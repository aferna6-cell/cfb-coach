"""v1.12 smarter retrain: zones, leverage, turnovers, caps, W/L, rebuild, live ranking."""

from __future__ import annotations

import random
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from cfb_coach import learning as L
from cfb_coach.db import CoachDB
from cfb_coach.outcome import parse_outcome
from cfb_coach.situation import parse_situation
from cfb_coach.zones import GOAL_LINE, OPEN, RED_ZONE, field_zone, zone_of_situation


def _snap(**kw):
    base = {"id": 1, "opponent_id": "cpu", "side": "offense", "down": 1, "distance": 10,
            "yardline": 50, "formation": "Gun Bunch X Nasty", "play": "Mesh Spot",
            "result": "+5", "coverage_seen": "Cover 3 Sky", "session_id": "s1"}
    base.update(kw)
    return base


class _Sess:
    def __init__(self, sid, opp="cpu"):
        self.sid, self.opp = sid, opp

    def to_dict(self):
        return {"session_id": self.sid, "opponent_id": self.opp,
                "started_ts": datetime.now(timezone.utc).isoformat()}


class TestZones(unittest.TestCase):
    def test_field_zone_buckets(self):
        self.assertEqual(field_zone(50, 10), OPEN)
        self.assertEqual(field_zone(79, 10), OPEN)  # opp 21
        self.assertEqual(field_zone(81, 10), RED_ZONE)  # opp 19
        self.assertEqual(field_zone(88, 2), RED_ZONE)  # 3rd & 2 at the 12
        self.assertEqual(field_zone(94, 6), GOAL_LINE)  # 1st & goal at the 6
        self.assertEqual(field_zone(96, 10), GOAL_LINE)  # inside the 5
        self.assertEqual(field_zone(85, 15), RED_ZONE)  # goal-to-go but at the 15

    def test_own_territory_is_not_red_zone(self):
        for raw in ("1&10 my 20", "2&5 my 3", "3&inches my 1"):
            sit = parse_situation(raw)
            self.assertFalse(sit.red_zone, raw)
            self.assertFalse(sit.goal_line, raw)
            self.assertEqual(zone_of_situation(sit), OPEN, raw)

    def test_goal_to_go_distance(self):
        sit = parse_situation("1&goal opp 6")
        self.assertEqual((sit.down, sit.distance, sit.yardline), (1, 6, 94))
        self.assertTrue(sit.goal_line)
        sit = parse_situation("3&2 opp 12")
        self.assertEqual(zone_of_situation(sit), RED_ZONE)
        self.assertFalse(sit.goal_line)

    def test_fumble_is_turnover(self):
        self.assertTrue(parse_outcome("fumble lost").is_turnover)
        self.assertTrue(parse_outcome("Fumble recovered by defense").is_turnover)
        self.assertTrue(parse_outcome("int").is_turnover)
        self.assertFalse(parse_outcome("sack").is_turnover)


class TestLeverageAndScores(unittest.TestCase):
    def test_leverage(self):
        self.assertEqual(L.leverage_for(1, False), 1.0)
        self.assertEqual(L.leverage_for(3, False), 2.0)
        self.assertEqual(L.leverage_for(4, False), 3.0)
        self.assertEqual(L.leverage_for(1, True), 2.0)
        self.assertLessEqual(L.leverage_for(4, True), L.LEVERAGE_MAX)

    def test_turnover_heavier_than_incompletion_and_sack(self):
        inc = L.evaluate_snap(_snap(result="incomplete"))
        sack = L.evaluate_snap(_snap(result="sack"))
        pick = L.evaluate_snap(_snap(result="int"))
        fum = L.evaluate_snap(_snap(result="fumble lost"))
        self.assertLess(pick.base_score, sack.base_score)
        self.assertLess(sack.base_score, inc.base_score)
        self.assertAlmostEqual(pick.base_score / inc.base_score, 3.5)
        self.assertEqual(fum.base_score, pick.base_score)

    def test_situational_success(self):
        self.assertTrue(L.evaluate_snap(_snap(down=1, distance=10, result="+4")).success)
        self.assertFalse(L.evaluate_snap(_snap(down=3, distance=5, result="+1")).success)
        e = L.evaluate_snap(_snap(down=4, distance=3, yardline=92, result="stop"))
        self.assertEqual(e.zone, RED_ZONE)
        self.assertEqual(e.leverage, 3.0)

    def test_parse_score_winner_first(self):
        p = L.parse_score("14-7", "loss")
        self.assertEqual((p["ours"], p["theirs"], p["margin"]), (7, 14, 7))
        p = L.parse_score("21-17", "win")
        self.assertEqual((p["ours"], p["theirs"]), (21, 17))
        self.assertAlmostEqual(L.margin_factor(14), 1.5)
        self.assertAlmostEqual(L.margin_factor(0), 1.0)


class TestCaps(unittest.TestCase):
    def test_squash_caps_and_order(self):
        vals = [L.squash(m, L.CAP_FAMILY) for m in (-50, -2, -0.5, 0.5, 2, 50)]
        self.assertTrue(all(abs(v) <= L.CAP_FAMILY for v in vals))
        self.assertEqual(vals, sorted(vals))
        self.assertLessEqual(abs(L.squash(99, L.cap_for_key("vs_look::a::b::c"))), L.CAP_VS_LOOK)

    def test_many_snaps_do_not_snowball(self):
        evals = [L.evaluate_snap(_snap(id=i, result="+8")) for i in range(1, 400)]
        acc = L.aggregate(evals)
        w = L.weights_from_acc(acc)
        for (_side, key), d in w.items():
            self.assertLessEqual(abs(d["weight"]), L.cap_for_key(key) + 1e-9, key)


class _DBCase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.td.name) / "coach.db")

    def tearDown(self):
        self.db.close()
        self.td.cleanup()

    def log(self, sid, down, dist, yl, form, play, result, cov="Cover 3 Sky"):
        return self.db.log_snap(opponent_id="cpu", side="offense", situation_raw=f"{down}&{dist}",
                                our_call=f"{form} — {play}", formation=form, play=play, down=down,
                                distance=dist, yardline=yl, result=result, coverage_seen=cov, session_id=sid)

    def weights(self):
        return {r["key"]: float(r["weight"]) for r in self.db.get_gameplan_weights("cpu")}


class TestZoneLearning(_DBCase):
    def test_rz_failure_hurts_rz_more_than_open(self):
        self.db.start_game_session(_Sess("g1"))
        for _ in range(4):
            self.log("g1", 1, 10, 50, "Gun Bunch X Nasty", "RZ PA X Whip", "+6")
        for yl in (95, 95, 94):
            self.log("g1", 1, 5, yl, "Gun Bunch X Nasty", "RZ PA X Whip", "int")
        L.rebuild_all(self.db)
        w = self.weights()
        pk = "play::Gun Bunch X Nasty::RZ PA X Whip"
        self.assertLess(w[f"zone::gl::{pk}"], -0.5)
        self.assertLess(w[f"zone::rz::{pk}"], 0)
        self.assertGreater(w[f"zone::open::{pk}"], 0)  # open field barely touched
        self.assertGreater(w[f"zone::open::{pk}"], w[f"zone::gl::{pk}"])

    def test_loss_adds_stall_penalty_win_adds_scoring_boost(self):
        def build(result_wl):
            db = CoachDB(Path(self.td.name) / f"{result_wl}.db")
            db.start_game_session(_Sess("g"))
            db.log_snap(opponent_id="cpu", side="offense", situation_raw="1&10", our_call="x",
                        formation="Singleback Deuce Close", play="Mtn Duo", down=1, distance=10,
                        yardline=85, result="+2", session_id="g")
            db.log_snap(opponent_id="cpu", side="offense", situation_raw="4&3", our_call="x",
                        formation="Singleback Deuce Close", play="Mtn Duo", down=4, distance=3,
                        yardline=92, result="stop", session_id="g")
            db.end_game_session("g", result_wl=result_wl, score="24-17")
            comp = L.compute_all(db.conn)
            db.close()
            return comp

        loss = build("loss")
        win = build("win")
        last_loss = [e for e in loss["evals"] if e.down == 4][0]
        last_win = [e for e in win["evals"] if e.down == 4][0]
        self.assertIn("drive_end_rz_fail", last_loss.tags)
        self.assertIn("loss_adjusted", last_loss.tags)
        self.assertLess(last_loss.score, last_win.score)
        # modest: the loss penalty is smaller than a turnover
        self.assertGreater(last_loss.score - last_win.score, L.SCORE_TURNOVER)

    def test_one_time_rebuild_with_backup_and_meta_key(self):
        self.db.start_game_session(_Sess("g1"))
        self.log("g1", 1, 10, 50, "Gun Bunch X Nasty", "Mesh Spot", "+9")
        self.db.bump_gameplan_weight("cpu", "offense", "run_first", 11.7)  # legacy snowball
        n_before = self.db.count_snaps("cpu")
        lines = []
        res = L.ensure_rules_current(self.db, printer=lines.append)
        self.assertIsNotNone(res)
        self.assertEqual(len(lines), 1)
        self.assertIn("backup", lines[0])
        bpath = Path(res["backup_path"])
        self.assertTrue(bpath.is_file())
        self.assertEqual(self.db.get_meta(L.META_RULES_KEY), L.RULES_VERSION)
        self.assertEqual(self.db.count_snaps("cpu"), n_before)  # history kept
        self.assertLessEqual(abs(self.weights().get("run_first", 0.0)), L.CAP_FAMILY)
        # second call is a no-op
        self.assertIsNone(L.ensure_rules_current(self.db, printer=lines.append))
        self.assertEqual(len(lines), 1)
        # backup still holds the old snowballed weight
        old = CoachDB(bpath)
        try:
            self.assertAlmostEqual({r["key"]: r["weight"] for r in old.get_gameplan_weights("cpu")}["run_first"], 11.7)
        finally:
            old.close()

    def test_postgame_summary_lists_rz_turnovers_risers(self):
        from cfb_coach.gameplan import postgame_summary

        self.db.start_game_session(_Sess("g1"))
        self.log("g1", 1, 10, 60, "Gun Bunch X Nasty", "Inside Zone", "+7")
        self.log("g1", 2, 10, 85, "Gun Bunch X Nasty", "RZ PA X Whip", "sack")
        self.log("g1", 3, 18, 80, "Gun Bunch X Nasty", "Z Spot GoalLine", "int")
        self.db.end_game_session("g1", result_wl="loss", score="14-7")
        txt = postgame_summary(self.db, "cpu", dynasty="ohio_state")
        for needle in ("Red zone / goal line", "TURNOVER", "SACK", "Loss adjustment", "Top risers", "Top fallers"):
            self.assertIn(needle, txt)


class TestLiveRanking(_DBCase):
    def test_caller_uses_zone_weights(self):
        from cfb_coach.playcaller import make_call, rank_offense_candidates

        self.db.start_game_session(_Sess("g1"))
        for _ in range(6):
            self.log("g1", 1, 5, 95, "Gun Bunch X Nasty", "RZ PA X Whip", "int")
            self.log("g1", 1, 5, 95, "Singleback Deuce Close", "Mtn Duo", "td")
        L.rebuild_all(self.db)
        sit = parse_situation("1&goal opp 5")
        rows = rank_offense_candidates(sit, self.db, "cpu")
        by = {r["play"]: r for r in rows}
        self.assertGreater(by["Mtn Duo"]["p"], by["RZ PA X Whip"]["p"])
        self.assertEqual(rows[0]["play"], "Mtn Duo")
        call = make_call(parse_situation("1&goal opp 5"), "cpu", self.db, rng=random.Random(0))
        self.assertIn("learned gl", call.rationale)

    def test_learned_weights_prefer_opponent_over_global(self):
        self.db.bump_gameplan_weight("cpu", "offense", "zone::rz::play::A::B", 1.0)
        self.db.bump_gameplan_weight("global", "offense", "zone::rz::play::A::B", 0.35)
        self.db.bump_gameplan_weight("global", "offense", "zone::rz::play::C::D", 0.2)
        lw = L.LearnedWeights.load(self.db, "cpu")
        self.assertAlmostEqual(lw.get("zone::rz::play::A::B"), 1.0)
        self.assertAlmostEqual(lw.get("zone::rz::play::C::D"), 0.2)


class TestCliRebuild(_DBCase):
    def test_cli_rebuild_command(self):
        import io
        import os
        from contextlib import redirect_stdout

        from cfb_coach.cli import build_parser

        self.db.start_game_session(_Sess("g1"))
        self.log("g1", 1, 10, 50, "Gun Bunch X Nasty", "Mesh Spot", "+9")
        path = str(self.db.path)
        self.db.close()
        old = os.environ.get("CFB_COACH_DB")
        os.environ["CFB_COACH_DB"] = path
        try:
            args = build_parser().parse_args(["rebuild"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = args.func(args)
        finally:
            if old is None:
                os.environ.pop("CFB_COACH_DB", None)
            else:
                os.environ["CFB_COACH_DB"] = old
            self.db = CoachDB(Path(path))
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("Rebuilt learned weights", out)
        self.assertIn("backup:", out)
        self.assertTrue(list(Path(self.td.name).glob("coach.db.bak-*")))


if __name__ == "__main__":
    unittest.main()
