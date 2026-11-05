"""Fixes from the Dillon game review. Synthetic inputs; no user database."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from cfb_coach.madden.model.concept_matchup import KNOWLEDGE_CAP, evaluate_concept_matchup
from cfb_coach.madden.model.defensive_diagnosis import diagnose_defense
from cfb_coach.madden.model.football_situation import SELECTION_PRIORS, selection_adjustment
from cfb_coach.madden.model.offense_outcome_adaptation import SESSION_INT_PENALTY, outcome_penalty
from cfb_coach.madden.model.offense_selection_policy import choose_model_play
from cfb_coach.madden.model.offense_strategy import STRATEGY_CAP, current_strategy, strategy_adjustment


def _sit(**kw):
    base = dict(down=1, distance=10, yardline=40, coverage_hint="cover 1", coverage_source="last",
                red_zone=False, goal_line=False, two_minute=False, extras={})
    base.update(kw)
    return SimpleNamespace(**base)


def _mem(labels):
    return {"observed_labels": list(labels)}


class RecentCoverageTests(unittest.TestCase):
    def test_last_snap_entries_still_build_an_in_game_read(self):
        picture = diagnose_defense(_sit(), _mem(["Cover 1"] * 4))
        self.assertEqual(picture["state"], "inferred")
        self.assertEqual(picture["inferred"]["shell"], "cover_1")
        self.assertEqual(picture["inferred"]["source"], "recent_snaps")
        self.assertIsNone(picture["observed"]["shell"])
        self.assertTrue(picture["previous_snap_not_copied"])

    def test_a_mid_game_switch_beats_the_early_looks(self):
        picture = diagnose_defense(_sit(), _mem(["Cover 6"] * 8 + ["Cover 1"] * 5))
        self.assertEqual(picture["inferred"]["shell"], "cover_1")

    def test_one_previous_snap_alone_is_still_unknown(self):
        picture = diagnose_defense(_sit(), _mem(["Cover 1"]))
        self.assertEqual(picture["state"], "unknown")


class ManAnswerTests(unittest.TestCase):
    def setUp(self):
        self.sit = _sit()
        self.memory = _mem(["Cover 3 Sky", "Cover 1", "Cover 1", "Cover 1", "Cover 1", "Cover 1"])
        self.diagnosis = diagnose_defense(self.sit, self.memory)
        self.plan = current_strategy(self.sit, self.memory, self.diagnosis)

    def _total(self, play):
        knowledge = evaluate_concept_matchup(play, self.sit, self.diagnosis)["delta"]
        strategy = strategy_adjustment(play, self.plan, knowledge_delta=knowledge)["delta"]
        return knowledge, strategy

    def test_man_beaters_gain_and_slow_shots_do_not(self):
        self.assertEqual(self.plan["hypothesis_id"], "man_answers")
        for play in ("Mesh", "Slants"):
            knowledge, strategy = self._total(play)
            self.assertEqual(knowledge, KNOWLEDGE_CAP)
            self.assertEqual(strategy, STRATEGY_CAP)
        for play in ("Switch HB Wheel", "PA Post Shot Choice", "Mtn PA Flood"):
            self.assertEqual(self._total(play), (0.0, 0.0))

    def test_third_and_long_still_withholds_the_quick_bonus(self):
        sit = _sit(down=3, distance=10)
        knowledge = evaluate_concept_matchup("Mesh", sit, diagnose_defense(sit, self.memory))
        self.assertEqual(knowledge["delta"], 0.0)


class SituationPriorTests(unittest.TestCase):
    def test_screen_on_fourth_and_one_is_penalized_but_rpo_bubble_is_not(self):
        sit = _sit(down=4, distance=1)
        screen, reason = selection_adjustment("HB Slip Screen", sit)
        self.assertEqual(screen, SELECTION_PRIORS["short_yardage_screen"]["delta"])
        self.assertIn("screen", reason)
        rpo, _ = selection_adjustment("RPO Alert Bubble", sit)
        self.assertGreater(rpo, 0)

    def test_play_action_on_fourth_and_long_is_penalized(self):
        sit = _sit(down=4, distance=10)
        pa, reason = selection_adjustment("Mtn PA Flood", sit)
        dropback, _ = selection_adjustment("Flood", sit)
        self.assertLess(pa, dropback)
        self.assertIn("play action", reason)


class RepeatAndTurnoverTests(unittest.TestCase):
    def test_a_second_screen_already_costs_something(self):
        ranked = [
            {"formation": "Gun", "play": "HB Slip Screen", "probability": 0.5},
            {"formation": "Gun", "play": "Stick", "probability": 0.5},
        ]
        rows, _audit = choose_model_play(
            ranked, recent_calls=[("Gun Trips", "WR Screen")], sit=_sit(), session_id="s", snap_seq=2,
        )
        by_play = {r["play"]: r for r in rows}
        self.assertGreater(by_play["HB Slip Screen"]["selection_penalty"], by_play["Stick"]["selection_penalty"])

    def test_a_play_intercepted_this_game_is_penalized(self):
        history = [{"executed_play": "Bench", "executed_formation": "Gun", "result": "int",
                    "session_id": "g1", "yardline": 40}]
        cost, explain = outcome_penalty("Gun", "Bench", _sit(), history, session_id="g1")
        self.assertGreaterEqual(cost, SESSION_INT_PENALTY)
        self.assertIn("intercepted_this_game", explain["penalties"])
        other, _ = outcome_penalty("Gun", "Bench", _sit(), history, session_id="g2")
        self.assertEqual(other, 0.0)


def _dsit(down, distance, yardline):
    return SimpleNamespace(down=down, distance=distance, yardline=yardline, red_zone=False,
                           goal_line=False, short_yardage=False, side="defense")


class LightBoxTests(unittest.TestCase):
    def test_dime_needs_a_passing_down_or_a_pass_first_opponent(self):
        from cfb_coach.madden.model.defense_v1 import light_box_allowed

        self.assertFalse(light_box_allowed(_dsit(1, 7, 7), 0.1))      # inside our 20
        self.assertFalse(light_box_allowed(_dsit(3, 4, 27), 0.1))     # short yardage
        self.assertFalse(light_box_allowed(_dsit(4, 4, 21), 0.1))
        self.assertTrue(light_box_allowed(_dsit(3, 8, 40), 0.9))      # obvious passing down
        self.assertFalse(light_box_allowed(_dsit(1, 10, 33), 0.54))   # balanced opponent
        self.assertFalse(light_box_allowed(_dsit(1, 10, 33), None))   # unknown run share
        self.assertTrue(light_box_allowed(_dsit(1, 10, 33), 0.3))

    def test_inside_the_five_wants_a_heavy_front(self):
        from cfb_coach.madden.model.defense_v1 import _wants_heavy_front

        self.assertTrue(_wants_heavy_front(_dsit(3, 4, 4)))
        self.assertFalse(_wants_heavy_front(_dsit(1, 10, 33)))

    def test_situation_guard_drops_light_boxes(self):
        from cfb_coach.madden.model.defense_v1 import _guarded_ranked

        rows = [{"formation": "Dime 3-2 Odd", "play": "Cover 3", "package": "dime"},
                {"formation": "Nickel Over", "play": "Cover 3", "package": "nickel"}]
        kept, guard = _guarded_ranked(rows, _dsit(1, 10, 40), False, light_ok=False)
        self.assertEqual([r["formation"] for r in kept], ["Nickel Over"])
        self.assertEqual(guard, "no_light_box_situation")


class LiveResultTests(unittest.TestCase):
    def test_offense_win_is_graded_against_the_sticks(self):
        from cfb_coach.madden.model.defense_v1 import offense_won

        self.assertTrue(offense_won("td", 1, 10))
        self.assertFalse(offense_won("+2", 1, 10))
        self.assertTrue(offense_won("+5", 1, 10))
        self.assertFalse(offense_won("+3", 3, 4))
        self.assertTrue(offense_won("+6", 2, 10))
        self.assertFalse(offense_won("incomplete", 2, 10))
        self.assertIsNone(offense_won(None, 1, 10))

    def test_a_beaten_coverage_family_is_pushed_down(self):
        from cfb_coach.madden.model.defense_v1 import _live_result_adjust

        ranked = [
            {"formation": "Nickel Over", "play": "Cover 1 Hole", "family": "one_high_man", "package": "nickel",
             "offense_success": 0.40},
            {"formation": "Nickel Over", "play": "Cover 3 Match", "family": "cover3", "package": "nickel",
             "offense_success": 0.42},
        ]
        results = [("4-3 Over", "Cover 1 Hole", True)] * 4 + [("4-3 Over", "Cover 3 Sky", False)] * 2
        summary = _live_result_adjust(ranked, results)
        self.assertEqual(ranked[0]["play"], "Cover 3 Match")
        self.assertIn("man 4/4", summary)


class DefensiveLoggingTests(unittest.TestCase):
    def test_leftover_text_on_a_situation_line_is_their_play(self):
        from cfb_coach.last_snap import leftover_play_text

        self.assertEqual(leftover_play_text("3&4 my 27 trail shake"), "trail shake")
        self.assertEqual(leftover_play_text("1&10 opp 30 q2 7-14"), "")
        self.assertEqual(leftover_play_text("2&7 my 43 clock 5:15"), "")

    def test_close_from_form_keeps_unnamed_calls_and_td_plays(self):
        import tempfile
        from pathlib import Path

        from cfb_coach.db import CoachDB
        from cfb_coach.last_snap import LastSnapBook
        from cfb_coach.madden.playcaller import MaddenCall
        from cfb_coach.madden.situation import parse_madden_situation

        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "t.db")
            try:
                book = LastSnapBook(db=db, opponent_id="dillon", parse_situation=parse_madden_situation,
                                    session_id="g1")
                call = MaddenCall("defense", "Nickel Over", "Cover 3 Match", "No adj", "User hook", "")
                book.remember_call(call, parse_madden_situation("1&10 my 40", default_side="defense"))
                book.close_from_form("+6", "", next_raw="2&4 my 34 trail shake")
                self.assertEqual(book.concept, "trail shake")
                book.remember_call(call, parse_madden_situation("2&4 my 34", default_side="defense"))
                book.close_from_form("td hb scissors", "")
                self.assertEqual(book.concept, "hb scissors")
            finally:
                db.close()


class CpuGameFixTests(unittest.TestCase):
    def test_cover_2_man_is_man_underneath_not_an_underneath_zone(self):
        from cfb_coach.madden.model.defensive_diagnosis import parse_look

        look = parse_look("cover 2 man")
        self.assertEqual(look["shell"], "cover_2_man")
        self.assertEqual(look["structure"], "two_high")
        self.assertIn("man", look["principles"])
        self.assertNotIn("zone", look["principles"])
        memory = _mem(["Cover 2 Man"] * 6)
        picture = diagnose_defense(_sit(), memory)
        self.assertEqual(picture["inferred"]["shell"], "cover_2_man")
        plan = current_strategy(_sit(), memory, picture)
        self.assertNotIn(plan["hypothesis_id"], ("underneath_space", "man_answers"))
        self.assertEqual(diagnose_defense(_sit(), _mem(["Cover 2"] * 6))["inferred"]["shell"], "cover_2")

    def test_installed_route_names_map_to_concepts(self):
        from cfb_coach.madden.model.football_knowledge import profile_for_play

        expected = {
            "Spot Choice": "spot", "L Spot": "spot", "Bench": "out_break", "Return Bench": "out_break",
            "Mills Double Out": "out_break", "Mtn FK Rub": "rub", "Mtn Salem Drag": "mesh",
            "Inside Cross": "deep_cross", "Slot Dig Wide Corner": "levels", "Quick Short Post": "dagger",
            "Mtn HB Swing": "texas_angle", "PA Post Shot Choice": "play_action", "PA X Cross Shot": "play_action",
            "Mesh Spot": "mesh", "Mtn Rub Smash": "smash",
        }
        for play, concept in expected.items():
            self.assertEqual(profile_for_play(play)["concept_id"], concept, play)
        self.assertIsNone(profile_for_play("Totally Unknown Play")["concept_id"])

    def test_goal_line_blitzes_drop_the_run_bonus_and_lift_quick_throws(self):
        from cfb_coach.madden.model.football_situation import coordinator_components, evaluate_situation

        sit = _sit(down=3, distance=3, yardline=97, coverage_hint="sam will blitz", goal_line=True)
        calm = evaluate_situation(sit, memory=_mem(["Cover 2 Man", "Cover 2 Man", "Cover 2 Man"]))
        hot = evaluate_situation(sit, memory=_mem(["Cover 2 Man", "pressure", "sam will blitz"]))
        self.assertFalse(calm["recent_pressure"])
        self.assertTrue(hot["recent_pressure"])

        def ids(play, picture):
            return {p["id"] for p in coordinator_components(play, picture)}

        self.assertIn("goal_line_run", ids("HB Stretch", calm))
        self.assertNotIn("goal_line_run", ids("HB Stretch", hot))
        self.assertIn("recent_pressure_quick", ids("Slants", hot))
        self.assertNotIn("recent_pressure_quick", ids("HB Slip Screen", hot))
        self.assertNotIn("recent_pressure_quick", ids("Slants", calm))


def _his(concept, won, play="Cover 3 Sky", down=1, distance=10, yardline=40, current=True):
    return {"concept": concept, "won": won, "play": play, "down": down, "distance": distance,
            "yardline": yardline, "current": current}


def _rows():
    return [
        {"formation": "Nickel Over", "play": "Cover 3 Match", "family": "cover3", "package": "nickel",
         "offense_success": 0.40},
        {"formation": "Nickel Over", "play": "Cover 2 Invert", "family": "two_high_zone", "package": "nickel",
         "offense_success": 0.41},
    ]


class RepeatBeaterTests(unittest.TestCase):
    def test_a_play_that_beat_us_twice_is_a_repeat_beater_and_one_win_is_not(self):
        from cfb_coach.madden.model.defense_v1 import repeat_beaters

        beaters = repeat_beaters([_his("Slants", True), _his("Slants", True), _his("Mesh", True),
                                  _his("Mid Zone", True), _his("Mid Zone", False), _his("Mid Zone", False)])
        self.assertEqual([b["concept"] for b in beaters], ["Slants"])

    def test_prior_games_against_him_count_half(self):
        from cfb_coach.madden.model.defense_v1 import repeat_beaters

        self.assertEqual(repeat_beaters([_his("Slants", True, current=False)] * 3), [])
        beaters = repeat_beaters([_his("Slants", True, current=False)] * 4)
        self.assertEqual(beaters[0]["wins"], 2.0)

    def test_counters_the_play_and_avoids_the_coverage_it_beat(self):
        from cfb_coach.madden.model.defense_v1 import _repeat_counter_adjust

        ranked = _rows()
        summary = _repeat_counter_adjust(ranked, _dsit(1, 10, 40), [_his("Slants", True)] * 3)
        self.assertEqual(ranked[0]["play"], "Cover 2 Invert")
        self.assertIn("Slants 3/3", summary)

    def test_stopped_plays_change_nothing(self):
        from cfb_coach.madden.model.defense_v1 import _repeat_counter_adjust

        ranked = _rows()
        self.assertEqual(_repeat_counter_adjust(ranked, _dsit(1, 10, 40), [_his("Slants", False)] * 4), "")
        self.assertEqual(ranked[0]["play"], "Cover 3 Match")

    def test_the_shift_is_capped(self):
        from cfb_coach.madden.model.defense_v1 import REPEAT_MAX_SHIFT, _repeat_counter_adjust

        ranked = _rows()
        plays = [_his(name, True) for name in ("Slants", "Crossers", "Mesh", "Return Bench") for _ in range(6)]
        _repeat_counter_adjust(ranked, _dsit(1, 10, 40), plays)
        for row in ranked:
            self.assertLessEqual(abs(row.get("repeat_adjust", 0.0)), REPEAT_MAX_SHIFT)


class RunFitTests(unittest.TestCase):
    def test_pressure_needs_three_runs_that_are_winning(self):
        from cfb_coach.madden.model.defense_v1 import run_fit_pressure

        self.assertEqual(run_fit_pressure([_his("HB Dive", True)] * 2)[0], 0.0)
        self.assertEqual(run_fit_pressure([_his("HB Dive", False)] * 4)[0], 0.0)
        self.assertGreater(run_fit_pressure([_his("HB Dive", True)] * 4)[0], 0.0)
        self.assertEqual(run_fit_pressure([_his("HB Dive", True, current=False)] * 4)[0], 0.0)

    def test_hot_runs_push_light_boxes_down_and_heavy_fronts_up_most_on_run_downs(self):
        from cfb_coach.madden.model.defense_v1 import _run_fit_adjust

        def ranked():
            return [
                {"formation": "Dime 3-2 Odd", "play": "Cover 3", "family": "cover3", "package": "dime",
                 "offense_success": 0.38},
                {"formation": "46 Normal", "play": "Cover 3", "family": "cover3", "package": "base",
                 "offense_success": 0.42},
            ]

        hot = [_his("HB Dive", True)] * 4 + [_his("Stretch", True)] * 2
        early = ranked()
        summary = _run_fit_adjust(early, _dsit(2, 3, 40), hot)
        self.assertEqual(early[0]["formation"], "46 Normal")
        self.assertIn("6/6", summary)
        passing = ranked()
        _run_fit_adjust(passing, _dsit(3, 9, 40), hot)
        by = {r["formation"]: r for r in early}
        by_pass = {r["formation"]: r for r in passing}
        self.assertGreater(abs(by["Dime 3-2 Odd"]["run_fit_adjust"]), abs(by_pass["Dime 3-2 Odd"]["run_fit_adjust"]))

    def test_coordinator_takes_an_explicit_opponent_play_list(self):
        from cfb_coach.madden.model.defense_coordinator import _defense_v1_opponent_plays

        plays = [_his("Slants", True)]
        self.assertEqual(_defense_v1_opponent_plays({"defense_v1_opponent_plays": plays}, None, "g1"), plays)
        self.assertEqual(_defense_v1_opponent_plays({}, None, "g1"), [])


if __name__ == "__main__":
    unittest.main()
