"""Regressions for optional clockless IF-look advice.

These are hypothetical instructions only: no live-look prompt, no auto-execution,
no alteration to the primary play or to ML/offense action attribution.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from cfb_coach.madden.model.conditional_look_actions import suggest_conditional_action
from cfb_coach.madden.playcaller import MaddenCall


class FakeDB:
    def __init__(self, coverages=()):
        self.coverages = list(coverages)

    def get_recent_snaps(self, opponent_id, *, side, limit):
        assert side == "offense"
        return [{"coverage_seen": name} for name in self.coverages][-limit:]


class ConditionalAdviceTests(unittest.TestCase):
    def setUp(self):
        self.call = MaddenCall("offense", "Gun Test", "Mesh", "No adj", "Primary")
        self.sit = SimpleNamespace(
            down=3, distance=7, coverage_source="last",
            coverage_hint="Cover 3", red_zone=False, goal_line=False,
        )
        self.book = {"Gun Test": ["Mesh"]}

    def _macros(self, *, settings=True, paired=True, looks=("Cover 2", "Cover 6")):
        def found(**kwargs):
            return ({"id": "C2", "kind": "look"} if kwargs["coverage"] in looks
                    else None)
        return mock.patch.multiple(
            "cfb_coach.madden.offense_macros",
            clean_ids=mock.Mock(return_value=["C2"]),
            macro_book_index=mock.Mock(return_value={
                "active": ["C2"], "pair_sets": {"C2": {
                    "Mesh (Gun Test)" if paired else "Mesh (Wrong Formation)"}},
                "details": {},
            }),
            cached_offense_detail=mock.Mock(return_value={
                "xbox_name": "C2", "buttons": "LB → C2", "settings": (
                    [{"setting": "confirmed"}] if settings else []),
                "needs_settings": not settings, "gaps": []}),
            suggest_for_snap=mock.Mock(side_effect=found),
        )

    def test_without_showing_or_clock_hint_is_conditional_not_executed(self):
        with self._macros():
            hint = suggest_conditional_action(
                self.call, self.sit, book=self.book, active=["C2"],
                db=FakeDB(["Cover 2", "Cover 2"]), opponent_id="ryan",
            )
        self.assertEqual(hint["condition"], "Cover 2")
        self.assertTrue(hint["conditional_only"])
        self.assertFalse(hint["executed"])
        self.assertIsNone(self.call.macro)
        self.assertIsNone(self.call.adjustment)
        self.assertEqual(self.sit.coverage_source, "last")
        self.assertEqual(self.sit.coverage_hint, "Cover 3")

    def test_logged_opponent_tendency_prioritizes_if_clause_not_current_look(self):
        with self._macros():
            hint = suggest_conditional_action(
                self.call, self.sit, book=self.book, active=["C2"],
                db=FakeDB(["Cover 6", "Cover 6", "Cover 2"]),
                opponent_id="ryan",
            )
        self.assertEqual(hint["condition"], "Cover 6")
        self.assertEqual(self.sit.coverage_hint, "Cover 3")

    def test_does_not_offer_missing_or_unpaired_macro(self):
        with self._macros(settings=False):
            self.assertIsNone(suggest_conditional_action(
                self.call, self.sit, book=self.book, active=["C2"]))
        with self._macros(paired=False):
            self.assertIsNone(suggest_conditional_action(
                self.call, self.sit, book=self.book, active=["C2"]))

    def test_existing_action_live_look_and_defense_remain_unchanged(self):
        with self._macros():
            self.call.macro = "C2"
            self.assertIsNone(suggest_conditional_action(
                self.call, self.sit, book=self.book, active=["C2"]))
            self.call.macro = None
            self.sit.coverage_source = "live"
            self.assertIsNone(suggest_conditional_action(
                self.call, self.sit, book=self.book, active=["C2"]))
            self.sit.coverage_source = "none"
            self.call.side = "defense"
            self.assertIsNone(suggest_conditional_action(
                self.call, self.sit, book=self.book, active=["C2"]))

    def test_not_an_active_action_in_call_formatter(self):
        self.call.conditional_action = {
            "condition": "Cover 2", "name": "C2", "buttons": "LB → C2",
        }
        self.assertNotIn("ADJ:", self.call.headline())
        self.assertNotIn("MACRO:", self.call.headline())
        self.assertIn("IF Cover 2", self.call.format())
        self.assertIn("not logged as executed", self.call.format())


if __name__ == "__main__":
    unittest.main()
