"""Sprint 9: situational offensive adjustment and grounded macro triggering.

Action values are research-based priors, NOT trained effect estimates. Only
the selected full-inventory formation/play can receive a sourced, executable
action, and doing nothing is a legitimate winning option.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from cfb_coach.madden.model.offense_action_policy import (
    _risk_threshold,
    choose_offense_action,
)


def snap(*, coverage=None, source="none", down=2, distance=8, rz=False, gl=False):
    return SimpleNamespace(
        coverage_hint=coverage, coverage_source=source,
        down=down, distance=distance, red_zone=rz, goal_line=gl,
    )


class OffensiveActionIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.book = {"Gun Bunch": ["Mesh", "Inside Zone"]}
        self.pred = {"probability": .65, "uncertainty": .20}
        self.kw = {
            "formation": "Gun Bunch", "play": "Mesh", "book": self.book,
            "active": [], "prediction": self.pred, "sit": snap(),
        }

    def choose(self, **kwargs):
        data = dict(self.kw)
        data.update(kwargs)
        return choose_offense_action(**data)

    def test_no_look_means_no_guessed_coverage_hot_route(self):
        d = self.choose()
        self.assertEqual(d["kind"], "none")
        self.assertTrue(d["no_action_compared"])
        self.assertIsNone(d["macro"])
        self.assertIsNone(d["adjustment"])

    def test_last_snap_look_is_not_enough(self):
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates"
        ) as candidates:
            d = self.choose(sit=snap(coverage="Cover 1", source="last"))
        self.assertEqual(d["kind"], "none")
        candidates.assert_not_called()

    def test_known_live_blitz_can_trigger_sourced_protection(self):
        action = {
            "id": "PROT-EDGE", "kind": "pass_protection",
            "label": "Slide protect", "sources": ["guide"],
            "buttons": "LB then select Slide protect",
            "why": "Live pressure at the edge",
        }
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=[action]
        ):
            result = self.choose(sit=snap(coverage="Blitz", source="live"))
        self.assertEqual(result["kind"], "adjustment")
        self.assertEqual(result["id"], "PROT-EDGE")
        self.assertGreater(result["top_action_score"], result["no_action_score"])
        self.assertIn("defensive look", result["why_now"])

    def test_no_action_wins_against_low_confidence_research_action(self):
        weak = {"probability": .51, "uncertainty": .98}
        action = {
            "id": "HOT-MAN", "kind": "hot_route", "sources": ["source"],
            "buttons": "Y then select receiver", "label": "Hot route",
        }
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=[action]
        ):
            result = self.choose(
                prediction=weak, sit=snap(coverage="Cover 1", source="live")
            )
        self.assertEqual(result["kind"], "none")
        self.assertIn("NO_ADJUSTMENT", [r["id"] for r in result["candidates"]])

    def test_unverified_controller_buttons_cannot_fire(self):
        action = {
            "id": "HOT-UNSOURCED", "kind": "hot_route",
            "sources": ["one"], "buttons": "VERIFY the button", "label": "Hot",
        }
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=[action]
        ):
            result = self.choose(sit=snap(coverage="Cover 1", source="live"))
        self.assertEqual(result["kind"], "none")
        self.assertNotIn("HOT-UNSOURCED", [r["id"] for r in result["candidates"]])

    def test_red_zone_situation_macro_without_coverage_read(self):
        # This simulates a macro actually present in the saved/armed game slot.
        suggestion = {
            "id": "O-RUN", "kind": "situation", "name": "O-RUN",
            "buttons": "LB then O-RUN", "why": "red zone run",
            "settings": [{"setting": "Block", "value": "Default"}],
        }
        detail = {
            "needs_settings": False, "gaps": [],
            "settings": [{"setting": "Block", "value": "Default"}],
        }
        with (
            mock.patch("cfb_coach.madden.offense_macros.clean_ids", return_value=["O-RUN"]),
            mock.patch("cfb_coach.madden.offense_macros.suggest_for_snap", return_value=suggestion),
            mock.patch("cfb_coach.madden.offense_macros.offense_detail", return_value=detail),
            mock.patch("cfb_coach.madden.offense_macros.pairs_in_book",
                       return_value=["Inside Zone (Gun Bunch)"]),
        ):
            result = self.choose(
                play="Inside Zone", active=["O-RUN"], sit=snap(rz=True),
            )
        self.assertEqual(result["kind"], "macro")
        self.assertEqual(result["id"], "O-RUN")
        self.assertEqual(result["why_now"], "verified situational condition")

    def test_situation_macro_does_not_auto_fire_every_open_field_play(self):
        suggestion = {
            "id": "O-RUN", "kind": "situation", "name": "O-RUN",
            "buttons": "LB then O-RUN", "why": "run", "settings": [1],
        }
        with (
            mock.patch("cfb_coach.madden.offense_macros.clean_ids", return_value=["O-RUN"]),
            mock.patch("cfb_coach.madden.offense_macros.suggest_for_snap", return_value=suggestion),
            mock.patch("cfb_coach.madden.offense_macros.offense_detail",
                       return_value={"needs_settings":False,"gaps":[],"settings":[1]}),
            mock.patch("cfb_coach.madden.offense_macros.pairs_in_book",
                       return_value=["Inside Zone (Gun Bunch)"]),
        ):
            result = self.choose(
                play="Inside Zone", active=["O-RUN"], sit=snap(),
            )
        self.assertEqual(result["kind"], "none")

    def test_unknown_goal_distance_does_not_crash(self):
        self.assertEqual(
            _risk_threshold(snap(down=3, distance="Goal"), credible_look=False),
            .36,
        )

    def test_uninstalled_play_cannot_trigger_macro(self):
        result = self.choose(
            play="Fabricated Play", active=["O-RUN"], sit=snap(rz=True)
        )
        self.assertEqual(result["kind"], "none")
        self.assertIn("not in applied book", result["reason"])

    def test_macro_lookup_no_longer_truncates_to_80(self):
        suggestion = {
            "id": "MAN", "kind": "look", "name": "MAN",
            "buttons": "LB then MAN", "why": "Cover 1",
        }
        books = {"Gun Bunch": ["Other " + str(i) for i in range(100)] + ["Mesh"]}
        def pairs(_mid, book, *, cap=6):
            self.assertGreater(cap, 80)
            return ["Mesh (Gun Bunch)"]
        with (
            mock.patch("cfb_coach.madden.offense_macros.clean_ids", return_value=["MAN"]),
            mock.patch("cfb_coach.madden.offense_macros.suggest_for_snap", return_value=suggestion),
            mock.patch("cfb_coach.madden.offense_macros.offense_detail",
                       return_value={"needs_settings":False, "gaps":[], "settings":[{"setting":"Route"}]}),
            mock.patch("cfb_coach.madden.offense_macros.pairs_in_book", side_effect=pairs),
        ):
            result = self.choose(
                book=books, active=["MAN"],
                sit=snap(coverage="Cover 1", source="live"),
            )
        self.assertIn("MAN", [c["id"] for c in result["candidates"]])


if __name__ == "__main__":
    unittest.main()
