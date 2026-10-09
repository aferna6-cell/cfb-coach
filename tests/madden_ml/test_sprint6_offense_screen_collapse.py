"""Regression: prevent HB Slip Screen mono-play and keep full installed formations."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden.model.offense_selection_policy import choose_model_play, select_from_database


def candidates():
    return [
        {"formation": "Gun Doubles", "play": "HB Slip Screen",
         "probability": 0.61, "play_concept": "screen", "uncertainty": .8},
        {"formation": "Gun Bunch", "play": "Mesh",
         "probability": 0.55, "play_concept": "mesh", "uncertainty": .8},
        {"formation": "Gun Bunch", "play": "Inside Zone",
         "probability": 0.53, "play_concept": "run_concept", "uncertainty": .8},
        {"formation": "Gun Tight", "play": "HB Screen",
         "probability": 0.54, "play_concept": "screen", "uncertainty": .8},
    ]


class ModelAntiCollapseTests(unittest.TestCase):
    def test_screen_can_be_model_top_first_snap(self):
        ranked, audit = choose_model_play(candidates())
        self.assertEqual(audit["top_original"][1], "HB Slip Screen")
        self.assertEqual(audit["recent_calls_used"], 0)
        self.assertIn(ranked[0]["play"], {r["play"] for r in candidates()})
        self.assertEqual(ranked[0]["selection_policy"],
                         "model_primary_contextual_variety.v2")

    def test_same_screen_not_repeated_forever_when_credible_alternatives(self):
        calls = [("Gun Doubles", "HB Slip Screen")] * 4
        ranked, audit = choose_model_play(candidates(), recent_calls=calls)
        self.assertNotEqual(ranked[0]["play"], "HB Slip Screen")
        self.assertNotEqual(ranked[0]["play"], "HB Screen")
        self.assertIn(ranked[0]["play"], ("Mesh", "Inside Zone"))
        self.assertEqual(audit["top_original"][1], "HB Slip Screen")
        self.assertEqual(audit["top_selected"][1], ranked[0]["play"])
        self.assertTrue(all("selection_penalty" in r for r in ranked))

    def test_prior_driven_model_cannot_repeat_identical_screen_forever(self):
        rows = candidates()
        rows[0]["probability"] = .99
        rows[0]["evidence_quality"] = "prior_driven"
        rows[0]["uncertainty"] = .9
        ranked, _ = choose_model_play(
            rows, recent_calls=[("Gun Doubles", "HB Slip Screen")] * 3
        )
        self.assertNotEqual(ranked[0]["play"], "HB Slip Screen")
        self.assertNotEqual(ranked[0]["play_concept"], "screen")

    def test_screen_family_spam_penalty_includes_different_screen_names(self):
        ranked, _ = choose_model_play(
            candidates(), recent_calls=[
                ("Gun Tight", "HB Screen"), ("Gun Doubles", "HB Slip Screen"),
                ("Gun Tight", "HB Screen"), ("Gun Doubles", "HB Slip Screen"),
            ]
        )
        self.assertIn(ranked[0]["play"], ("Mesh", "Inside Zone"))

    def test_only_legal_play_remains_call_and_not_heuristic(self):
        row = candidates()[0]
        ranked, _ = choose_model_play([row], recent_calls=[
            ("Gun Doubles", "HB Slip Screen")] * 4)
        self.assertEqual(ranked[0]["play"], "HB Slip Screen")
        self.assertEqual(ranked[0]["selection_penalty"], 0)

    def test_strong_model_advantage_is_not_automatically_discarded(self):
        rows = candidates()
        rows[0]["probability"] = .97
        rows[0]["uncertainty"] = 0.1
        rows[0]["evidence_quality"] = "verified"
        rows[1]["probability"] = .40
        ranked, _ = choose_model_play(rows, recent_calls=[
            ("Gun Doubles", "HB Slip Screen")] * 3)
        self.assertEqual(ranked[0]["play"], "HB Slip Screen")

    def test_reads_recommended_calls_without_inventing_executions(self):
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "db.sqlite", seed={"opponents": {
                "cpu": {"display_name": "CPU", "team_now": "DET",
                        "skill": "cpu", "confidence": "low", "profile_json": "{}"}
            }})
            try:
                for x in range(3):
                    db.log_snap(
                        opponent_id="cpu", side="offense",
                        situation_raw=f"{x+1}&10",
                        our_call="Gun Doubles — HB Slip Screen",
                        formation="Gun Doubles", play="HB Slip Screen",
                        result="+2", session_id="g1",
                        executed_status="unknown",
                        executed_verification="unknown",
                    )
                ranked, audit = select_from_database(
                    candidates(), db=db, opponent_id="cpu"
                )
                self.assertNotEqual(ranked[0]["play"], "HB Slip Screen")
                self.assertEqual(audit["recent_calls_used"], 3)
                # Anti-repetition is exposure to *recommendations*, NOT a
                # claim those unknown plays were actually executed.
                self.assertEqual(
                    db.conn.execute(
                        "SELECT COUNT(*) FROM snaps WHERE executed_verification='verified'"
                    ).fetchone()[0], 0
                )
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
