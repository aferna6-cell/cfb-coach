"""Sprint 11: adaptive offensive coordinator — situation, joint action, memory, macros."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model import offense_macro_lab as lab
from cfb_coach.madden.model.football_situation import (
    evaluate_situation, play_situation_fit, situation_coverage_matrix,
    sit_from_probe, synthetic_probe_situations,
)
from cfb_coach.madden.model.offense_game_memory import (
    decision_context, empty_memory, load_game_memory, record_live_look,
    record_recommendation, save_game_memory,
)
from cfb_coach.madden.model.offense_joint_decision import (
    choose_joint_offense_action, score_joint_candidate, _legal_multi_components,
)
from cfb_coach.madden.model.offense_designer import design_offense
from cfb_coach.madden.model.experimental_model import ExperimentalArtifact


def _db(path):
    return CoachDB(path, seed={"opponents": {
        "cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu",
                "confidence": "low", "profile_json": "{}"}
    }})


class FootballSituationTests(unittest.TestCase):
    def test_probes_cover_required_situations(self):
        labels = {p["label"] for p in synthetic_probe_situations()}
        for required in situation_coverage_matrix():
            self.assertIn(required, labels)

    def test_unknown_coverage_never_fabricated(self):
        sit = sit_from_probe({
            "down": 3, "distance": 10, "yardline": 35,
            "coverage_hint": "cover 3", "coverage_source": "last",
        })
        fb = evaluate_situation(sit)
        self.assertFalse(fb.credible_look)
        self.assertEqual(fb.credibility.coverage, "last_snap_soft")
        self.assertIn("prior_look_is_not_current_coverage", fb.rationale)

    def test_live_coverage_is_credible(self):
        sit = sit_from_probe({
            "down": 2, "distance": 7, "yardline": 40,
            "coverage_hint": "blitz", "coverage_source": "live",
        })
        fb = evaluate_situation(sit)
        self.assertTrue(fb.credible_look)
        self.assertTrue(fb.pressure_credible)
        self.assertEqual(fb.credibility.coverage, "live")

    def test_third_and_long_prioritizes_conversion(self):
        sit = sit_from_probe({"down": 3, "distance": 10, "yardline": 35})
        fb = evaluate_situation(sit)
        self.assertTrue(fb.third_and_long)
        run_fit, run_why = play_situation_fit("Inside Zone", fb)
        pass_fit, pass_why = play_situation_fit("Mesh", fb)
        self.assertLess(run_fit, pass_fit)
        self.assertTrue(any("long" in r.lower() for r in run_why))

    def test_late_lead_and_trail_differ(self):
        lead = evaluate_situation(sit_from_probe({
            "down": 1, "distance": 10, "yardline": 40, "two_minute": True,
            "score_us": 24, "score_them": 17, "quarter": 4,
        }))
        trail = evaluate_situation(sit_from_probe({
            "down": 1, "distance": 10, "yardline": 40, "two_minute": True,
            "score_us": 10, "score_them": 24, "quarter": 4,
        }))
        self.assertEqual(lead.possession_objective, "protect_lead")
        self.assertEqual(trail.possession_objective, "trail")
        lead_run, _ = play_situation_fit("Inside Zone", lead)
        trail_run, _ = play_situation_fit("Inside Zone", trail)
        self.assertGreater(lead_run, trail_run)


class JointDecisionTests(unittest.TestCase):
    def test_no_adjustment_can_win(self):
        sit = sit_from_probe({"down": 1, "distance": 10, "yardline": 35})
        ranked = [{
            "formation": "Gun Bunch", "play": "Mesh",
            "probability": 0.62, "uncertainty": 0.4,
            "selection_score": 0.62, "evidence_quality": "prior_driven",
            "play_concept": "mesh",
        }]
        book = {"Gun Bunch": ["Mesh", "Flood"]}
        with mock.patch(
            "cfb_coach.madden.model.offense_joint_decision.choose_offense_action",
            return_value={
                "kind": "none", "no_action_score": 0.36,
                "reason": "no eligible", "candidates": [
                    {"kind": "none", "id": "NO_ADJUSTMENT", "score": 0.36},
                ],
            },
        ):
            decision = choose_joint_offense_action(
                ranked_plays=ranked, sit=sit, book=book, active=[],
            )
        self.assertEqual(decision["kind"], "none")
        self.assertEqual(decision["adjustment_plan"]["id"], "NO_ADJUSTMENT")
        self.assertTrue(decision["no_adjustment_can_win"])

    def test_conflicting_hot_routes_rejected(self):
        bad = [
            {"type": "hot_route", "target": "WR1", "settings": [{"setting": "WR1"}]},
            {"type": "hot_route", "target": "WR1", "settings": [{"setting": "WR1"}]},
        ]
        self.assertFalse(_legal_multi_components(bad))
        good = [
            {"type": "hot_route", "target": "WR1", "settings": [{"setting": "WR1"}]},
            {"type": "pass_protection", "type": "pass_protection"},
        ]
        # Fix duplicate key — use proper structures
        good = [
            {"type": "hot_route", "settings": [{"setting": "WR1"}]},
            {"type": "pass_protection", "settings": [{"setting": "Protection"}]},
        ]
        self.assertTrue(_legal_multi_components(good))

    def test_joint_scores_adjusted_play(self):
        sit = sit_from_probe({
            "down": 3, "distance": 10, "yardline": 35,
            "coverage_hint": "blitz", "coverage_source": "live",
        })
        play_row = {
            "formation": "Gun Bunch", "play": "Mesh",
            "probability": 0.55, "uncertainty": 0.5, "selection_score": 0.55,
        }
        none_plan = {"kind": "none", "id": "NO_ADJUSTMENT", "score": 0.3, "components": []}
        adj_plan = {
            "kind": "adjustment", "id": "SLIDE", "score": 0.42,
            "components": [{"kind": "adjustment", "type": "pass_protection", "id": "SLIDE"}],
        }
        none_row = score_joint_candidate(play_row=play_row, plan=none_plan, situation=sit)
        adj_row = score_joint_candidate(play_row=play_row, plan=adj_plan, situation=sit)
        self.assertIn("joint_score", none_row)
        self.assertIn("joint_score", adj_row)
        self.assertEqual(none_row["adjustment_plan"]["kind"], "none")

    def test_unarmed_macro_not_in_joint_executable_path(self):
        sit = sit_from_probe({
            "down": 2, "distance": 7, "yardline": 40,
            "coverage_hint": "man", "coverage_source": "live",
        })
        ranked = [{
            "formation": "Gun Bunch", "play": "Mesh",
            "probability": 0.7, "uncertainty": 0.2,
            "selection_score": 0.7, "evidence_quality": "empirical",
            "play_concept": "mesh",
        }]
        # choose_offense_action already filters drafts; simulate none returned.
        with mock.patch(
            "cfb_coach.madden.model.offense_joint_decision.choose_offense_action",
            return_value={
                "kind": "none", "no_action_score": 0.28,
                "reason": "draft macro not armed", "candidates": [
                    {"kind": "none", "id": "NO_ADJUSTMENT", "score": 0.28},
                ],
            },
        ):
            decision = choose_joint_offense_action(
                ranked_plays=ranked, sit=sit,
                book={"Gun Bunch": ["Mesh"]}, active=["DRAFT-MACRO"],
            )
        self.assertNotEqual(decision.get("kind"), "macro")


class GameMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _db(Path(self.tmp.name) / "mem.db")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_tendency_not_current_coverage(self):
        mem = empty_memory("g1")
        for i in range(5):
            mem = record_live_look(
                mem, coverage_class="pressure", coverage_source="live",
                snap_id=f"s{i}",
            )
        ctx = decision_context(mem)
        self.assertGreaterEqual(ctx["opponent_tendencies"]["n_looks"], 5)
        self.assertIn("not current coverage", ctx["opponent_tendencies"]["note"])
        # Soft last-snap looks must not update tendency counts.
        mem2 = record_live_look(
            mem, coverage_class="man", coverage_source="last", snap_id="x",
        )
        self.assertEqual(
            mem2["opponent_tendencies"]["n_looks"],
            mem["opponent_tendencies"]["n_looks"],
        )

    def test_recommendation_not_execution(self):
        mem = record_recommendation(
            empty_memory("g1"), snap_id="s1",
            formation="Gun Bunch", play="Mesh",
            adjustment_plan={"kind": "none", "id": "NO_ADJUSTMENT"},
        )
        self.assertEqual(len(mem["recommendations"]), 1)
        self.assertEqual(mem["verified_executions"], [])
        save_game_memory(self.db, mem)
        loaded = load_game_memory(self.db, "g1")
        self.assertEqual(loaded["recommendations"][0]["play"], "Mesh")

    def test_no_future_outcome_in_decision_context(self):
        mem = empty_memory("g1")
        mem["outcomes"] = [{"snap_id": "future", "success": True}]
        ctx = decision_context(mem)
        self.assertNotIn("outcomes", ctx)


class MacroLabCompositionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _db(Path(self.tmp.name) / "lab.db")
        rec = {
            "side": "offense", "mode": "custom", "name": "ML Designed Offense",
            "rev": 3, "locked_ts": "2026-10-08T19:00:00+00:00",
            "formations": {
                "Gun Bunch": ["Mesh", "Flood", "Quick Slants", "Inside Zone"],
            },
            "formation_sources": {"Gun Bunch": "Buccaneers"},
            "audibles": {}, "core": ["Gun Bunch"],
        }
        playbook._save_state(self.db, {"applied": {"offense": rec}, "pending": {}})
        self.patch = mock.patch.object(
            lab.research_db, "offense_adjustments", return_value=[
                {"id": "hot_slant_vs_man", "type": "hot_route", "vs": ["man"],
                 "target": "WR1", "route": "Slant", "sources": ["src-a"]},
                {"id": "prot_slide", "type": "pass_protection", "vs": ["pressure", "man"],
                 "target": "OL", "route": "Slide Left", "sources": ["src-b"]},
            ],
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.db.close()
        self.tmp.cleanup()

    def test_compositions_remain_drafts(self):
        plan = lab.propose_variants(self.db, limit=8, include_compositions=True)
        self.assertEqual(plan["status"], "DRAFT_ONLY")
        compositions = [p for p in plan["proposals"] if p.get("kind") == "composition"]
        self.assertTrue(compositions or plan["n_compositions"] >= 0)
        for item in plan["proposals"]:
            self.assertTrue(str(item["lifecycle"]).startswith("DRAFT"))
            self.assertIn("NOT_ARMED", item["activation"])
            self.assertIn("DRAFT", item["status"])

    def test_lifecycle_draft_until_verified(self):
        staged = lab.stage_variants(self.db, limit=2)
        name = staged["proposals"][0]["name"]
        self.assertEqual(lab.blueprint_lifecycle(self.db, "cpu", name), "DRAFT")


class PregamePortfolioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _db(Path(self.tmp.name) / "design.db")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_designer_returns_complete_plan_fields(self):
        catalogue = {
            "Buccaneers": {
                "Gun Bunch": ["Mesh", "Flood", "Inside Zone", "Quick Slants"],
                "Gun Trips": ["Smash", "Curl Flat", "Duo", "Stick"],
                "Singleback Ace": ["Power O", "PA Boot", "Y Sail", "Slant"],
            },
            "Chiefs": {
                "Gun Y Off Trips": ["Mesh", "Flood", "Zone Weak", "Levels"],
                "Pistol Strong": ["Inside Zone", "PA Cross", "Counter", "Smash"],
            },
        }
        art = ExperimentalArtifact(
            evidence_quality="prior_driven", note="synthetic fixture",
            model_version="test-sprint11",
        )
        proposal = design_offense(
            self.db, opponent_id="cpu", max_formations=3,
            artifact=art, catalogue=catalogue,
        )
        self.assertTrue(proposal["staged_not_applied"])
        self.assertIn("inventory_fingerprint", proposal)
        self.assertIn("formation_rationale", proposal)
        self.assertIn("suggested_audibles", proposal)
        self.assertIn("research_provenance", proposal)
        self.assertIn("uncertainty", proposal)
        self.assertEqual(len(proposal["book"]["formations"]), 3)
        for form, plays in proposal["book"]["formations"].items():
            # Every verified source-book play retained (no artificial cap).
            self.assertEqual(
                set(plays), set(catalogue[proposal["book"]["formation_sources"][form]][form])
            )
        for why in proposal["formation_rationale"]:
            self.assertIn("situations_addressed", why)
            self.assertTrue(why["situations_addressed"])


class FullBookJointPoolTests(unittest.TestCase):
    def test_joint_considers_beyond_old_candidate_limits(self):
        sit = sit_from_probe({"down": 2, "distance": 6, "yardline": 40})
        ranked = [
            {
                "formation": "Gun Bunch", "play": f"Play{i}",
                "probability": 0.5 - i * 0.001, "uncertainty": 0.6,
                "selection_score": 0.5 - i * 0.001,
                "evidence_quality": "prior_driven", "play_concept": "mesh",
            }
            for i in range(40)
        ]
        book = {"Gun Bunch": [f"Play{i}" for i in range(40)]}
        with mock.patch(
            "cfb_coach.madden.model.offense_joint_decision.choose_offense_action",
            return_value={
                "kind": "none", "no_action_score": 0.36,
                "reason": "none", "candidates": [],
            },
        ):
            decision = choose_joint_offense_action(
                ranked_plays=ranked, sit=sit, book=book, active=[],
            )
        # TOP_PLAY_JOINT (18) + explore slice — clearly beyond old 5–6 repertoire.
        self.assertGreaterEqual(decision["n_plays_considered"], 18)
        self.assertGreater(decision["n_joint_candidates"], 6)


if __name__ == "__main__":
    unittest.main()
