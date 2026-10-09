"""Sprint 11 deterministic acceptance tests for the adaptive coordinator."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model.experimental_model import ExperimentalArtifact
from cfb_coach.madden.model.football_situation import (
    evaluate_situation, game_context_from_database, score_play_suitability,
)
from cfb_coach.madden.model.offense_action_policy import choose_offense_action
from cfb_coach.madden.model.offense_coordinator import (
    choose_joint_offensive_decision,
)


def situation(
    down: int = 2, distance: int = 7, *,
    coverage: str | None = None, source: str = "none",
    score_us: int | None = None, score_them: int | None = None,
    two_minute: bool = False, yardline: int = 50,
) -> SimpleNamespace:
    return SimpleNamespace(
        down=down, distance=distance, yardline=yardline,
        red_zone=yardline >= 80, goal_line=yardline >= 97,
        coverage_hint=coverage, coverage_source=source,
        score_us=score_us, score_them=score_them,
        two_minute=two_minute, extras={"quarter": 4, "clock_seconds": 90},
    )


class FootballSituationTests(unittest.TestCase):
    def test_third_long_values_conversion_path(self):
        state = evaluate_situation(situation(3, 10))
        run = score_play_suitability("HB Dive", state)
        sticks = score_play_suitability("Four Verticals", state)
        self.assertLess(run["proxy_score"], sticks["proxy_score"])
        self.assertIn("first-down path", " ".join(run["reasons"]))

    def test_late_lead_and_trailing_objectives_meaningfully_differ(self):
        lead = evaluate_situation(
            situation(score_us=24, score_them=17, two_minute=True)
        )
        trail = evaluate_situation(
            situation(score_us=17, score_them=24, two_minute=True)
        )
        self.assertEqual(lead.clock_objective, "consume_clock")
        self.assertEqual(trail.clock_objective, "preserve_clock")
        self.assertGreater(
            score_play_suitability("Inside Zone", lead)["proxy_score"],
            score_play_suitability("Inside Zone", trail)["proxy_score"],
        )

    def test_previous_coverage_never_becomes_current_certainty(self):
        state = evaluate_situation(
            situation(coverage="Cover 1", source="last"),
            opponent_evidence={
                "dominant_coverage": "Cover 1", "sample_size": 9, "confidence": .53,
            },
        )
        self.assertFalse(state.coverage_is_current)
        self.assertIn("current_coverage", state.uncertainty)
        self.assertEqual(state.tendency_coverage, "Cover 1")


class JointDecisionTests(unittest.TestCase):
    def setUp(self):
        self.sit = situation()

    def test_every_eligible_play_has_no_adjustment_candidate_beyond_old_limits(self):
        plays = [f"Pass Concept {n}" for n in range(40)]
        book = {"Gun Complete": plays}
        ranked = [
            {
                "formation": "Gun Complete", "play": play,
                "probability": .55, "selection_score": .55 - n / 10000,
                "uncertainty": .6, "evidence_quality": "prior_driven",
            }
            for n, play in enumerate(plays)
        ]
        chosen, action, audit = choose_joint_offensive_decision(
            ranked, sit=self.sit, book=book, active=[], db=None,
            opponent_id="cpu", session_id="game", snap_seq=1,
        )
        self.assertIn(chosen["play"], plays)
        self.assertEqual(action["kind"], "none")
        self.assertEqual(audit["eligible_play_count"], 40)
        self.assertEqual(audit["no_adjustment_candidates"], 40)
        self.assertEqual(audit["legal_joint_decision_count"], 40)

    def test_compatible_multi_adjustments_are_evaluated(self):
        candidates = [
            {
                "id": "HOT-A", "kind": "hot_route", "target": "X", "route": "Slant",
                "label": "X slant", "sources": ["guide"], "source_ids": ["s1"],
                "buttons": "Y then X then Slant", "why": "man answer",
            },
            {
                "id": "PROT-A", "kind": "pass_protection", "target": "Line",
                "route": "Slide Left", "label": "Slide left", "sources": ["guide"],
                "source_ids": ["s1"], "buttons": "LB then Slide Left",
                "why": "pressure answer",
            },
        ]
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=candidates,
        ):
            result = choose_offense_action(
                formation="Gun Bunch", play="Mesh",
                sit=situation(coverage="Blitz", source="live"),
                book={"Gun Bunch": ["Mesh"]}, active=[],
                prediction={"probability": .7, "uncertainty": .1},
            )
        kinds = [row["kind"] for row in result["legal_plans"]]
        self.assertIn("multi_adjustment", kinds)
        multi = next(row for row in result["legal_plans"]
                     if row["kind"] == "multi_adjustment")
        self.assertEqual(len(multi["payload"]["steps"]), 2)

    def test_conflicting_hot_route_targets_are_rejected(self):
        candidates = [
            {
                "id": name, "kind": "hot_route", "target": "X", "route": route,
                "label": route, "sources": ["guide"], "buttons": f"Y X {route}",
            }
            for name, route in (("A", "Slant"), ("B", "Out"))
        ]
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=candidates,
        ):
            result = choose_offense_action(
                formation="Gun Bunch", play="Mesh",
                sit=situation(coverage="Cover 1", source="live"),
                book={"Gun Bunch": ["Mesh"]}, active=[],
                prediction={"probability": .7, "uncertainty": .1},
            )
        self.assertNotIn(
            "multi_adjustment", [row["kind"] for row in result["legal_plans"]]
        )


class PlanningAndMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "s11.db", seed={
            "opponents": {"cpu": {
                "display_name": "CPU", "team_now": "DET", "skill": "cpu",
                "confidence": "low", "profile_json": "{}",
            }}
        })

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_planner_evaluates_catalog_and_retains_complete_formations(self):
        from cfb_coach.madden.model.offense_designer import (
            SITUATION_MATRIX, design_offense,
        )

        catalogue = {
            "A": {
                "Gun A": ["Mesh", "Inside Zone", "Four Verticals"],
                "Goal Line A": ["HB Dive", "PA Boot"],
            },
            "B": {"Pistol B": ["Stretch", "Flood", "Slants"]},
        }
        plan = design_offense(
            self.db, opponent_id="cpu", max_formations=2,
            artifact=ExperimentalArtifact(global_rate=.5),
            catalogue=catalogue,
        )
        self.assertEqual(plan["catalog_formations_evaluated"], 3)
        self.assertEqual(
            set(plan["situation_coverage"]),
            {scenario["id"] for scenario in SITUATION_MATRIX},
        )
        for formation, plays in plan["book"]["formations"].items():
            source = plan["book"]["formation_sources"][formation]
            self.assertEqual(plays, catalogue[source][formation])
        self.assertEqual(
            plan["inventory_id"], plan["complete_inventory_fingerprint"]
        )
        self.assertTrue(plan["chosen_formation_details"])

    def test_game_context_is_session_scoped_completed_history(self):
        for session, coverage in (
            ("current", "Blitz"), ("current", "Blitz"), ("other", "Cover 3")
        ):
            self.db.log_snap(
                opponent_id="cpu", side="offense", session_id=session,
                situation_raw="1&10", formation="Gun", play="Mesh",
                our_call="Gun — Mesh", result="+5", coverage_seen=coverage,
            )
        context = game_context_from_database(
            self.db, session_id="current", opponent_id="cpu"
        )
        self.assertEqual(context["sample_size"], 2)
        self.assertEqual(context["dominant_coverage"], "Blitz")
        self.assertNotIn("Cover 3", context["coverage_counts"])

    def test_generated_compositions_stay_draft_until_explicit_verification(self):
        from cfb_coach.madden.model import offense_designer, offense_macro_lab

        installed = {
            "side": "offense", "mode": "custom", "name": "Installed",
            "rev": 2, "locked_ts": "confirmed", "source_book": None,
            "formation_sources": {"Gun": "A"},
            "formations": {"Gun": ["Mesh"]}, "audibles": {}, "core": ["Gun"],
        }
        playbook._save_state(
            self.db, {"applied": {"offense": installed}, "pending": {}}
        )
        researched = [
            {
                "id": "H1", "type": "hot_route", "target": "X", "route": "Slant",
                "vs": ["man"], "sources": ["source-a"],
            },
            {
                "id": "P1", "type": "pass_protection", "target": "Line",
                "route": "Slide Left", "vs": ["man"], "sources": ["source-b"],
            },
        ]
        with mock.patch.object(
            offense_macro_lab.research_db, "offense_adjustments",
            return_value=researched,
        ):
            plan = offense_macro_lab.stage_variants(self.db, limit=16)
        self.assertGreater(plan["added"], 0)
        self.assertEqual(offense_designer.verified_created_macros(self.db, "cpu"), [])
        self.assertEqual(offense_designer.verified_macro_configurations(self.db), [])
        self.assertTrue(any(
            item["kind"] == "multi_action" and "DRAFT" in item["status"]
            for item in plan["proposals"]
        ))


if __name__ == "__main__":
    unittest.main()
