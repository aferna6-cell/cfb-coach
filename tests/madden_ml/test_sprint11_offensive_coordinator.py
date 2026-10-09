"""Sprint 11: joint coordinator, portfolio, drafts and provenance.

Franchise logs from the user's laptop are not in this environment. Scenarios
built here are synthetic.
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model.football_situation import (
    SITUATION_NAMES, evaluate_situation, explain_play,
)
from cfb_coach.madden.model.offense_coordinator_eval import compare_policies
from cfb_coach.madden.model.offense_game_memory import (
    pre_snap_context, record_closed_snap, remember_recommendation,
)
from cfb_coach.madden.model.offense_joint_decision import (
    choose_joint_action, legal_plans_for_play,
)
from cfb_coach.madden.model.offense_macro_lab import (
    compose_drafts, conflicts, recommend_blueprint_fate,
)
from cfb_coach.madden.model.offense_selection_policy import choose_model_play


def _seed() -> dict:
    return {"opponents": {"cpu": {
        "display_name": "CPU", "team_now": "DET", "skill": "cpu",
        "confidence": "low", "profile_json": "{}",
    }}}


def sit(**kwargs):
    base = dict(
        down=1, distance=10, yardline=40, red_zone=False, goal_line=False,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={},
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


CATALOG = {
    "Buccaneers": {
        "Gun Bunch": ["Mesh", "Inside Zone", "Flood", "Four Verticals", "Quick Slants"],
        "Singleback Wing": ["Stretch", "PA Boot", "Quick Slants"],
        "Goal Line": ["HB Dive", "QB Sneak"],
    },
    "Lions": {
        "Gun Tight": ["Mesh", "HB Dive", "Smash", "Slants", "Flood"],
        "Gun Trips": ["Inside Zone", "Verticals", "PA Cross", "HB Draw"],
    },
}


class SituationTests(unittest.TestCase):
    def test_third_and_long_prefers_a_route_that_can_convert(self):
        mesh = explain_play("Mesh", sit(down=3, distance=10))
        dive = explain_play("HB Dive", sit(down=3, distance=10))
        screen = explain_play("HB Slip Screen", sit(down=3, distance=10))
        self.assertLess(dive["selection_delta"], 0)
        self.assertLess(screen["selection_delta"], 0)
        self.assertGreater(mesh["selection_delta"], dive["selection_delta"])

    def test_second_and_short_keeps_both_conversion_and_upside(self):
        run = explain_play("Inside Zone", sit(down=2, distance=2))
        shot = explain_play("Four Verticals", sit(down=2, distance=2))
        self.assertGreater(run["coordinator_delta"], 0)
        self.assertGreater(shot["coordinator_delta"], 0)

    def test_late_lead_and_trail_differ(self):
        lead = explain_play("Inside Zone", sit(
            down=1, distance=10, two_minute=False, score_us=21, score_them=14,
            extras={"quarter": 4, "clock_seconds": 300},
        ))
        trail = explain_play("Inside Zone", sit(
            down=2, distance=8, two_minute=True, score_us=14, score_them=21,
        ))
        self.assertGreater(lead["coordinator_delta"], 0)
        self.assertLess(trail["selection_delta"], 0)

    def test_two_minute_lead_values_the_run(self):
        lead = explain_play("Inside Zone", sit(
            down=1, distance=10, two_minute=True, score_us=24, score_them=17,
        ))
        self.assertGreater(lead["selection_delta"], 0)

    def test_last_snap_coverage_is_not_current(self):
        picture = evaluate_situation(sit(coverage_hint="Cover 1", coverage_source="last"))
        self.assertEqual(picture["coverage"]["state"], "inferred")
        self.assertIsNone(picture["coverage"]["value"])
        self.assertIn("not the current", picture["coverage"]["note"])

    def test_unknown_coverage_is_not_observed(self):
        picture = evaluate_situation(sit())
        self.assertEqual(picture["coverage"]["state"], "unknown")
        play = explain_play("Mesh", sit())
        self.assertTrue(any(p["id"] == "coverage_unknown" for p in play["coordinator_parts"]))
        self.assertEqual(play["coordinator_delta"], 0)

    def test_portfolio_names_every_required_situation(self):
        self.assertEqual(
            set(SITUATION_NAMES),
            {"normal", "short_yardage", "third_and_long", "red_zone", "goal_line",
             "backed_up", "two_minute", "clock_management"},
        )


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed=_seed())

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_plan_keeps_every_play_and_covers_situations(self):
        from cfb_coach.madden.model import experimental_model
        art = experimental_model.ExperimentalArtifact(
            model_version="test.model", n_supervised=0, evidence_quality="prior_driven",
            global_rate=0.5,
        )
        plan = designer.design_offense(
            self.db, opponent_id="cpu", max_formations=3, artifact=art, catalogue=CATALOG,
        )
        self.assertLessEqual(len(plan["book"]["formations"]), 3)
        for form, plays in plan["book"]["formations"].items():
            source = plan["book"]["formation_sources"][form]
            self.assertEqual(plays, CATALOG[source][form])
        coverage = plan["situation_coverage"]
        for name in SITUATION_NAMES:
            self.assertIn(name, coverage)
        self.assertTrue(plan["formation_rationales"])
        self.assertIn("uncertainty", plan["provenance"])
        self.assertEqual(plan["provenance"]["roster"]["state"], "unknown")
        self.assertTrue(plan["inventory_id"])
        for macro in plan["macro_blueprints"]:
            self.assertIn("DRAFT", macro["status"])
            self.assertIn("NOT_ARMED", macro["activation"])
        self.assertIsNone(designer.staged_design(self.db))


class JointDecisionTests(unittest.TestCase):
    def setUp(self):
        self.book = {
            "Gun Bunch": ["Mesh", "Inside Zone", "Four Verticals", "HB Slip Screen"],
            "Gun Trips": ["Flood", "HB Draw"],
        }
        self.ranked = []
        for form, plays in self.book.items():
            for play in plays:
                self.ranked.append({
                    "formation": form, "play": play, "probability": 0.55,
                    "uncertainty": 0.4, "evidence_quality": "prior_driven",
                    "play_concept": play.lower(),
                })

    def _anchor(self, snap):
        ranked, _audit = choose_model_play(
            self.ranked, sit=snap, session_id="joint", snap_seq=1, opponent_type="cpu",
        )
        return ranked

    def test_every_eligible_play_is_considered(self):
        ranked = self._anchor(sit())
        decision = choose_joint_action(
            ranked=ranked, anchor=ranked[0], sit=sit(), book=self.book, active=[],
        )
        self.assertEqual(decision["plays_considered"], sum(len(p) for p in self.book.values()))
        self.assertIn("NO_ADJUSTMENT", [row["id"] for row in decision["candidates"]])

    def test_no_adjustment_wins_without_a_live_look(self):
        ranked = self._anchor(sit())
        decision = choose_joint_action(
            ranked=ranked, anchor=ranked[0], sit=sit(), book=self.book, active=[],
        )
        self.assertEqual(decision["kind"], "none")
        legal = [play for plays in self.book.values() for play in plays]
        self.assertIn(decision["joint"]["play"], legal)

    def test_a_better_unmodified_play_beats_the_sampled_anchor(self):
        ranked = [
            {
                "formation": "Gun Bunch", "play": "Mesh", "probability": 0.40,
                "selection_score": 0.40, "uncertainty": 0.50,
                "evidence_quality": "prior_driven", "play_concept": "mesh",
            },
            {
                "formation": "Gun Trips", "play": "Flood", "probability": 0.72,
                "selection_score": 0.72, "uncertainty": 0.50,
                "evidence_quality": "prior_driven", "play_concept": "flood",
            },
        ]
        decision = choose_joint_action(
            ranked=ranked, anchor=ranked[0], sit=sit(), book=self.book, active=[],
        )
        self.assertEqual(decision["kind"], "none")
        self.assertEqual(decision["joint"]["play"], "Flood")
        self.assertEqual(decision["joint"]["baseline_play"], "Mesh")
        self.assertTrue(decision["joint"]["displaced_baseline"])
        self.assertFalse(decision["exploration"]["fixed_rotation"])

    def test_research_prior_does_not_overturn_a_stronger_play(self):
        ranked = [
            {
                "formation": "Gun Bunch", "play": "Mesh", "probability": 0.80,
                "selection_score": 0.80, "uncertainty": 0.20,
                "evidence_quality": "empirical", "play_concept": "mesh",
            },
            {
                "formation": "Gun Trips", "play": "Flood", "probability": 0.55,
                "selection_score": 0.55, "uncertainty": 0.40,
                "evidence_quality": "prior_driven", "play_concept": "flood",
            },
        ]
        action = {
            "id": "HOT-MAN", "kind": "hot_route", "sources": ["source"],
            "buttons": "Y then select receiver", "label": "Hot route", "why": "man",
        }
        snap = sit(coverage_hint="Cover 1", coverage_source="live")

        def only_on_the_weaker_play(**kwargs):
            return [action] if kwargs.get("play") == "Flood" else []

        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            side_effect=only_on_the_weaker_play,
        ):
            decision = choose_joint_action(
                ranked=ranked, anchor=ranked[1], sit=snap, book=self.book, active=[],
            )
        self.assertEqual(decision["kind"], "none")
        self.assertEqual(decision["joint"]["play"], "Mesh")

    def test_near_ties_are_not_a_fixed_rotation(self):
        ranked = []
        for form, plays in self.book.items():
            for play in plays:
                ranked.append({
                    "formation": form, "play": play, "probability": 0.55,
                    "selection_score": 0.55, "uncertainty": 0.50,
                    "evidence_quality": "prior_driven", "play_concept": play.lower(),
                })
        calls = []
        for seq in range(1, 13):
            decision = choose_joint_action(
                ranked=ranked, anchor=ranked[0], sit=sit(), book=self.book,
                active=[], session_id="rotate", snap_seq=seq, opponent_type="cpu",
            )
            calls.append(decision["joint"]["play"])
            self.assertEqual(decision["kind"], "none")
        order = [row["play"] for row in ranked]
        self.assertGreater(len(set(calls)), 1)
        self.assertNotEqual(calls, [order[i % len(order)] for i in range(len(calls))])

    def test_negative_learned_shift_blocks_a_marginal_hot_route(self):
        ranked = self._anchor(sit(coverage_hint="Cover 1", coverage_source="live"))
        action = {
            "id": "HOT-MAN", "kind": "hot_route", "sources": ["source"],
            "buttons": "Y then select receiver", "label": "Hot route", "why": "man",
        }
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=[action],
        ), mock.patch(
            "cfb_coach.madden.model.offense_action_learning.score_shift",
            return_value=(-0.04, {"mode": "bounded_active"}),
        ):
            decision = choose_joint_action(
                ranked=ranked, anchor=ranked[0],
                sit=sit(coverage_hint="Cover 1", coverage_source="live"),
                book=self.book, active=[],
            )
        self.assertEqual(decision["kind"], "none")

    def test_short_clock_blocks_adjustments(self):
        ranked = self._anchor(sit(
            coverage_hint="Cover 1", coverage_source="live",
            extras={"clock_seconds": 5, "quarter": 2},
        ))
        action = {
            "id": "HOT-MAN", "kind": "hot_route", "sources": ["source"],
            "buttons": "Y then select receiver", "label": "Hot route", "why": "man",
        }
        snap = sit(
            coverage_hint="Cover 1", coverage_source="live",
            extras={"clock_seconds": 5, "quarter": 2},
        )
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=[action],
        ):
            decision = choose_joint_action(
                ranked=ranked, anchor=ranked[0], sit=snap, book=self.book, active=[],
            )
        self.assertEqual(decision["kind"], "none")
        self.assertNotIn("HOT-MAN", [row["id"] for row in decision["candidates"]])

    def test_compatible_adjustments_can_be_scored_together_and_conflicts_drop(self):
        self.assertEqual(conflicts([
            {"type": "hot_route", "target": "WR1", "vs": ["man"]},
            {"type": "hot_route", "target": "WR1", "vs": ["man"]},
        ]), "two hot routes assign WR1")
        self.assertIn("max protect", conflicts([
            {"type": "hot_route", "target": "HB", "route": "Flat", "vs": ["pressure"]},
            {"type": "pass_protection", "route": "Max protect", "vs": ["pressure"]},
        ]))
        left = {
            "id": "HOT-WR", "kind": "hot_route", "target": "WR1", "route": "Slant",
            "label": "Hot WR", "buttons": "X then slant", "sources": ["guide"],
            "vs": ["man"], "why": "man",
        }
        right = {
            "id": "HOT-TE", "kind": "hot_route", "target": "TE", "route": "Corner",
            "label": "Hot TE", "buttons": "Y then corner", "sources": ["guide"],
            "vs": ["man"], "why": "man",
        }
        ranked = self._anchor(sit(coverage_hint="Cover 1", coverage_source="live"))
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=[left, right],
        ), mock.patch(
            "cfb_coach.madden.model.offense_joint_decision._research_row",
            side_effect=lambda action_id: {
                "HOT-WR": {"id": "HOT-WR", "type": "hot_route", "target": "WR1", "route": "Slant", "vs": ["man"], "sources": ["guide"]},
                "HOT-TE": {"id": "HOT-TE", "type": "hot_route", "target": "TE", "route": "Corner", "vs": ["man"], "sources": ["guide"]},
            }.get(action_id, {}),
        ):
            decision = choose_joint_action(
                ranked=ranked, anchor=ranked[0],
                sit=sit(coverage_hint="Cover 1", coverage_source="live"),
                book=self.book, active=[],
            )
            plans = legal_plans_for_play(
                formation=ranked[0]["formation"], play=ranked[0]["play"],
                sit=sit(coverage_hint="Cover 1", coverage_source="live"),
                book=self.book, active=[], prediction=ranked[0], repeated=False,
                audibles=None, db=None, opponent_id="",
                evaluation=evaluate_situation(
                    sit(coverage_hint="Cover 1", coverage_source="live")
                ),
            )
        ids = [row["id"] for row in decision["candidates"]]
        self.assertEqual(decision["kind"], "adjustment")
        self.assertIn("NO_ADJUSTMENT", ids)
        self.assertTrue(any(plan.get("composition") for plan in plans))

    def test_no_adjustment_can_still_win_against_a_weak_hot_route(self):
        weak = [{
            **row, "probability": 0.51, "uncertainty": 0.99,
        } for row in self.ranked]
        ranked, _audit = choose_model_play(
            weak, sit=sit(coverage_hint="Cover 1", coverage_source="live"),
            session_id="weak", snap_seq=2, opponent_type="cpu",
        )
        action = {
            "id": "HOT-MAN", "kind": "hot_route", "sources": ["source"],
            "buttons": "Y then select receiver", "label": "Hot route", "why": "man",
        }
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=[action],
        ):
            decision = choose_joint_action(
                ranked=ranked, anchor=ranked[0],
                sit=sit(coverage_hint="Cover 1", coverage_source="live"),
                book=self.book, active=[],
            )
        self.assertEqual(decision["kind"], "none")

    def test_unarmed_draft_macro_is_not_offered(self):
        ranked = self._anchor(sit(coverage_hint="Cover 1", coverage_source="live"))
        with mock.patch(
            "cfb_coach.madden.model.offense_designer.verified_created_macros",
            return_value=[{
                "name": "ML-DRAFT", "coverage": "man", "verified_armed": False,
                "state": "draft", "settings": [{"setting": "Route", "value": "Slant"}],
                "source_ids": ["guide"],
                "base_pairs": [{"formation": "Gun Bunch", "play": ranked[0]["play"]}],
            }],
        ):
            decision = choose_joint_action(
                ranked=ranked, anchor=ranked[0],
                sit=sit(coverage_hint="Cover 1", coverage_source="live"),
                book=self.book, active=[], db=object(), opponent_id="cpu",
            )
        self.assertNotIn("ML-DRAFT", [row["id"] for row in decision["candidates"]])

    def test_anti_repeat_is_not_a_fixed_rotation(self):
        recent = []
        calls = []
        for n in range(1, 16):
            ranked, _audit = choose_model_play(
                self.ranked, recent_calls=recent, sit=sit(),
                session_id="variety", snap_seq=n, opponent_type="cpu",
            )
            calls.append(ranked[0]["play"])
            recent.insert(0, (ranked[0]["formation"], ranked[0]["play"]))
        self.assertGreater(len(set(calls)), 2)
        cycle = calls[:3]
        self.assertNotEqual(calls, (cycle * 5)[:15])

    def test_joint_latency_stays_inside_the_budget_on_a_wider_book(self):
        wide = []
        book = {}
        for index in range(40):
            form = f"Gun {index // 8}"
            play = f"Concept {index}"
            book.setdefault(form, []).append(play)
            wide.append({
                "formation": form, "play": play, "probability": 0.5,
                "uncertainty": 0.5, "evidence_quality": "prior_driven",
                "selection_score": 0.5, "play_concept": "pass",
            })
        started = time.perf_counter()
        choose_joint_action(
            ranked=wide, anchor=wide[0], sit=sit(), book=book, active=[],
        )
        self.assertLess((time.perf_counter() - started) * 1000.0, 150.0)


class MacroAndMemoryTests(unittest.TestCase):
    def test_new_composition_stays_a_draft(self):
        primitives = [
            {"id": "a", "type": "hot_route", "target": "WR1", "route": "Slant",
             "vs": ["man"], "sources": ["s1"]},
            {"id": "b", "type": "hot_route", "target": "TE", "route": "Out",
             "vs": ["man"], "sources": ["s2"]},
        ]
        drafts = compose_drafts(
            {"Gun Bunch": ["Mesh"]}, {"Gun Bunch": "Buccaneers"}, primitives=primitives,
        )
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["state"], "draft")
        self.assertIn("DRAFT", drafts[0]["status"])
        self.assertIn("NOT_ARMED", drafts[0]["activation"])
        self.assertEqual(drafts[0]["source_action_ids"], ["a", "b"])

    def test_real_research_pair_that_conflicts_is_not_emitted(self):
        drafts = compose_drafts({"Gun Bunch": ["Mesh"]}, {"Gun Bunch": "Buccaneers"})
        self.assertEqual(drafts, [])

    def test_fate_does_not_retire_a_draft(self):
        fate = recommend_blueprint_fate({"name": "ML-X", "status": "DRAFT"}, None)
        self.assertEqual(fate["recommendation"], "retain")
        self.assertEqual(fate["state"], "draft")

    def test_memory_does_not_leak_the_current_snap(self):
        tmp = tempfile.TemporaryDirectory()
        db = CoachDB(Path(tmp.name) / "mem.db", seed=_seed())
        try:
            remember_recommendation(
                db, session_id="g", snap_id="s1", snap_seq=1,
                formation="Gun Bunch", play="Mesh",
            )
            before = pre_snap_context(db, session_id="g", snap_seq=1, snap_id="s1")
            self.assertEqual(before["sample_size"], 0)
            record_closed_snap(
                db, session_id="g", snap_id="s1", snap_seq=1,
                executed_formation="Gun Bunch", executed_play="Mesh",
                observed_defense="Blitz", outcome="+6", verified=True,
                adjustment_applied=False, no_adjustment_confirmed=True,
            )
            still = pre_snap_context(db, session_id="g", snap_seq=1, snap_id="s1")
            self.assertEqual(still["sample_size"], 0)
            nxt = pre_snap_context(db, session_id="g", snap_seq=2, snap_id="s2")
            self.assertEqual(nxt["sample_size"], 1)
            self.assertEqual(nxt["state"], "unknown")
            self.assertIsNone(nxt["inferred_look"])
        finally:
            db.close()
            tmp.cleanup()

    def test_one_failure_does_not_condemn_a_concept(self):
        tmp = tempfile.TemporaryDirectory()
        db = CoachDB(Path(tmp.name) / "mem.db", seed=_seed())
        try:
            remember_recommendation(
                db, session_id="g", snap_id="s1", snap_seq=1,
                formation="Gun Bunch", play="Mesh",
            )
            record_closed_snap(
                db, session_id="g", snap_id="s1", snap_seq=1,
                executed_play="Mesh", executed_formation="Gun Bunch",
                outcome="incomplete", verified=True, observed_defense="Cover 3",
            )
            nxt = pre_snap_context(db, session_id="g", snap_seq=2)
            self.assertEqual(nxt["discouraged_plays"], [])
        finally:
            db.close()
            tmp.cleanup()


class ComparisonTests(unittest.TestCase):
    def test_synthetic_comparison_is_labeled_and_legal(self):
        report = compare_policies(seed=11)
        self.assertEqual(report["evidence"], "synthetic_fixtures")
        self.assertEqual(report["user_franchise_logs"], "not_available_in_this_environment")
        self.assertEqual(report["third_and_long_run_calls"], 0)
        self.assertEqual(report["legal_action_rate"], 1.0)
        self.assertEqual(report["unknown_coverage_labeled_observed"], 0)
        self.assertLess(report["max_joint_latency_ms"], 150.0)
        again = compare_policies(seed=11)
        self.assertEqual(report["calls"], again["calls"])


if __name__ == "__main__":
    unittest.main()
