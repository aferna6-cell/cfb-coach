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


if __name__ == "__main__":
    unittest.main()
