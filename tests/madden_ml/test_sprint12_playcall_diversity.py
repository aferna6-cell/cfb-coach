"""Strategic variety across the installed book, without a play quota."""
from __future__ import annotations

import unittest

from cfb_coach.madden.model.offense_diversity import (
    RECORDED_BASELINE,
    evaluate_strategic_diversity,
)
from cfb_coach.madden.model.offense_strategy import strategy_report


class StrategicDiversityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluate_strategic_diversity()

    def test_full_book_is_considered_and_varied_without_a_quota(self):
        sheet = self.report["with_diversity"]
        self.assertEqual(sheet["consideration_rate"], 1.0)
        self.assertGreater(sheet["eligible_installed_plays"], 15)
        self.assertGreater(sheet["formations"]["distinct"], RECORDED_BASELINE["formations"])
        self.assertGreater(
            sheet["labeled_concept_families"]["distinct"],
            RECORDED_BASELINE["labeled_concept_families"],
        )
        self.assertGreater(sheet["packages"]["distinct"], 1)
        self.assertEqual(sheet["situations"]["distinct"], 8)
        self.assertIn("adjustments", sheet)
        self.assertFalse(sheet["fixed_rotation"])
        self.assertFalse(sheet["minimum_quota"])
        self.assertLess(sheet["recommended_play_coverage"], 1.0)
        self.assertLessEqual(sheet["mean_probability_drop"], 0.06)
        self.assertEqual(sheet["third_long_run_calls"], 0)

    def test_recorded_28_7_comparison_is_diversity_not_quality(self):
        comparison = self.report["baseline_comparison"]
        self.assertEqual(comparison["recorded_baseline"]["game_id"], "9f2ebdeb9d8f4e2d")
        self.assertEqual(comparison["recorded_baseline"]["top_formation_recommendations"], 46)
        self.assertEqual(comparison["recorded_baseline"]["recommendations"], 63)
        self.assertTrue(comparison["less_concentrated_than_recorded_top_formation"])
        self.assertTrue(comparison["decision_quality_not_inferred"])
        self.assertTrue(comparison["not_a_win_rate_claim"])
        self.assertTrue(self.report["not_a_win_rate_claim"])
        self.assertIn("not evidence", comparison["recorded_baseline"]["note"])

    def test_concentration_band_varies_close_calls_and_keeps_a_clear_leader(self):
        history = self.report["concentrated_history"]
        locked = history["without_diversity_band"]
        opened = history["with_diversity"]
        self.assertEqual(locked["formations"]["top_share"], 1.0)
        self.assertTrue(history["band_changes_the_call_sheet"])
        self.assertTrue(history["best_play_still_used"])
        self.assertGreater(opened["formations"]["distinct"], 1)
        self.assertEqual(opened["consideration_rate"], 1.0)
        self.assertLessEqual(opened["mean_probability_drop"], 0.06)
        probe = self.report["clear_superiority_probe"]
        self.assertTrue(probe["held_clear_leader"])
        self.assertEqual(probe["chosen_play"], "Mesh")
        self.assertEqual(probe["plays_considered"], probe["eligible"])

    def test_strategy_command_labels_the_recorded_baseline(self):
        report = strategy_report(None, "cpu")
        baseline = report["recorded_diversity_baseline"]
        self.assertEqual(baseline["formations"], 3)
        self.assertEqual(baseline["labeled_concept_families"], 3)
        self.assertEqual(baseline["top_formation_share"], round(46 / 63, 4))
        self.assertTrue(baseline["not_a_decision_quality_benchmark"])
        self.assertFalse(report["history_modified"])
