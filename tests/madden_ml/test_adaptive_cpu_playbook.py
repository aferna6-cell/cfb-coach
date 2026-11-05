"""Offline regressions: goal-line risk, screen losses, unscheduled formation choice.

Unit tests use synthetic calls only, never user game data or inferred plays.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from cfb_coach.madden.model.offense_outcome_adaptation import (
    apply_outcome_adaptation, outcome_penalty,
)
from cfb_coach.madden.model.offense_formation_rotation import (
    apply_formation_rotation, choose_active_formations,
)


def row(play, *, form="Gun Test", result="+4", yards=4,
        yardline=60, session_id="game1"):
    return {
        "executed_formation": form, "executed_play": play,
        "result": result, "yards": yards, "yardline": yardline,
        "session_id": session_id, "executed_status": "identified",
        "executed_verification": "verified",
    }


def candidate(form, play, p):
    return {"formation": form, "play": play,
            "probability": p, "selection_score": p, "uncertainty": .25}


class AdaptationTests(unittest.TestCase):
    def test_goal_line_int_penalty_is_targeted_not_a_pass_ban(self):
        sit = SimpleNamespace(goal_line=True, yardline=98)
        history = [
            row("Mesh Post", result="int", yards=None, yardline=98),
            row("Mesh Post", result="int", yards=None, yardline=97),
            row("Mesh Post", result="td", yards=None, yardline=97),
            row("Slants", result="td", yards=None, yardline=99),
        ]
        penalty, diag = outcome_penalty("Gun Test", "Mesh Post", sit, history)
        self.assertGreater(penalty, 0)
        self.assertLessEqual(penalty, 0.075)
        self.assertIn("goal_line_interceptions", diag["penalties"])
        self.assertEqual(outcome_penalty("Gun Test", "Slants", sit, history)[0], 0.0)
        self.assertEqual(outcome_penalty(
            "Gun Test", "Mesh Post", SimpleNamespace(goal_line=False, yardline=70),
            history,
        )[0], 0.0)

    def test_recent_screen_outcome_penalty_after_real_losses(self):
        sit = SimpleNamespace(goal_line=False, yardline=55)
        history = [
            row("HB Slip Screen", result="-4", yards=-4),
            row("HB Slip Screen", result="-2", yards=-2),
            row("HB Slip Screen", result="-9", yards=-9),
            row("HB Slip Screen", result="+12", yards=12),
        ]
        bad, _ = outcome_penalty("Gun Test", "HB Slip Screen", sit, history,
                                 session_id="game1")
        self.assertGreater(bad, 0)
        self.assertLessEqual(bad, 0.105)
        self.assertEqual(outcome_penalty(
            "Gun Test", "HB Slip Screen", sit,
            [row("HB Slip Screen", result="+12", yards=12)]
        )[0], 0.0)

    def test_probability_is_not_overwritten(self):
        sit = SimpleNamespace(goal_line=True, yardline=98)
        original = [candidate("Gun Test", "Mesh Post", .7),
                    candidate("Pistol", "HB Dive", .66)]
        history = [
            row("Mesh Post", result="int", yardline=98, yards=None),
            row("Mesh Post", result="int", yardline=99, yards=None),
        ]
        changed, audit = apply_outcome_adaptation(original, sit=sit, history=history)
        risky = next(r for r in changed if r["play"] == "Mesh Post")
        self.assertEqual(risky["probability"], .7)
        self.assertLess(risky["selection_score"], .7)
        self.assertEqual(audit["penalized_candidates"], 1)
        self.assertEqual(len(changed), 2)


class RotationTests(unittest.TestCase):
    def test_no_period_or_forced_change_when_model_scores_stay_equal(self):
        rows = [candidate("A", "pA", .75),
                candidate("B", "pB", .74),
                candidate("C", "pC", .70),
                candidate("D", "pD", .62)]
        first, details = choose_active_formations(rows)
        second, details_2 = choose_active_formations(rows, previous=first)
        self.assertEqual(first, second)
        self.assertFalse(details["scheduled_rotation"])
        self.assertFalse(details_2["scheduled_rotation"])
        self.assertNotIn("D", second)

    def test_challenger_enters_on_changed_model_evidence(self):
        old = [candidate("A", "a", .75), candidate("B", "b", .73),
               candidate("C", "c", .65)]
        active, _ = choose_active_formations(old)
        self.assertNotIn("C", active)
        new = [candidate("A", "a", .73), candidate("B", "b", .72),
               candidate("C", "c", .78)]
        active2, info = choose_active_formations(new, previous=active)
        self.assertIn("C", active2)
        self.assertIn("C", info["entered"])
        self.assertNotIn("snap_interval", info)

    def test_all_installed_plays_stay_legal_even_if_inactive(self):
        rows = [candidate("A", "a", .72), candidate("B", "b", .60)]
        out, details = apply_formation_rotation(rows, session_id=None)
        self.assertEqual(len(out), 2)
        self.assertIn("B", details["challengers"])
        self.assertTrue(details["no_physical_book_change"])
        self.assertEqual(
            next(r for r in out if r["formation"] == "B")["formation_pool_state"],
            "challenger",
        )


if __name__ == "__main__":
    unittest.main()
