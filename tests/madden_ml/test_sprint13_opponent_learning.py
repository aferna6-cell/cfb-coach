"""Sprint 13: opponent learning stays conservative and chronological."""
from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest import mock

from cfb_coach.madden.model.defensive_observation import structure_observation
from cfb_coach.madden.model.offense_joint_decision import choose_joint_action
from cfb_coach.madden.model.opponent_learning import (
    LEARNING_CAP,
    fit_opponent_model,
    learning_adjustment,
    opponent_learning_report,
    readonly_madden_db_path,
)
from cfb_coach.madden.model.opponent_learning_eval import (
    evaluate_opponent_learning,
    replay_prior_only,
    _changing_game,
)
from cfb_coach.madden.model.offense_strategy import current_strategy
from cfb_coach.madden.model.schema import ML_LATENCY_BUDGET_MS
from cfb_coach.madden.model.video_observation import (
    dry_run_import,
    synthetic_video_fixtures,
    validate_video_observation,
)


def _sit(**kwargs):
    base = dict(
        down=3, distance=8, yardline=40, red_zone=False, goal_line=False,
        two_minute=False, score_us=14, score_them=17, coverage_hint=None,
        coverage_source="none", extras={}, quarter=2,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _pressure_game(n_early: int, early: bool, n_late: int, late: bool) -> list[dict]:
    rows = []
    seq = 1
    for flag, count in ((early, n_early), (late, n_late)):
        for _ in range(count):
            rows.append({
                "snap_seq": seq,
                "down": 3,
                "distance": 8,
                "yards": 2 if flag else 9,
                "formation": "Gun Bunch",
                "play": "Mesh",
                "concept_id": "mesh",
                "observation": {
                    "coverage_shell": None,
                    "pressure": flag,
                    "blitz": flag,
                    "legacy_ambiguous": False,
                },
                "verified_execution": True,
                "prelabeled": True,
                "success": not flag,
                "conversion": not flag,
                "turnover": False,
                "sack": False,
                "model_probability": 0.55,
            })
            seq += 1
    return rows


class ObservationTests(unittest.TestCase):
    def test_cover_1_and_blitz_are_both_stored(self):
        observed = structure_observation(
            "showing cover 1 blitz", timing="pre_snap", source="live",
        )
        self.assertEqual(observed["coverage_shell"], "cover_1")
        self.assertTrue(observed["pressure"])
        self.assertTrue(observed["blitz"])
        self.assertTrue(observed["man_evidence"])
        self.assertFalse(observed["legacy_ambiguous"])
        self.assertEqual(observed["timing"], "pre_snap")

    def test_ambiguous_man_is_not_upgraded(self):
        observed = structure_observation("man", timing="historical", source="legacy_label")
        self.assertIsNone(observed["coverage_shell"])
        self.assertTrue(observed["man_evidence"])
        self.assertTrue(observed["legacy_ambiguous"])
        self.assertFalse(observed["upgraded_from_ambiguous"])
        self.assertIsNone(observed["front"])
        self.assertLessEqual(observed["confidence"], 0.4)

    def test_two_high_is_not_a_shell(self):
        observed = structure_observation("two-high", timing="pre_snap", source="live")
        self.assertIsNone(observed["coverage_shell"])
        self.assertEqual(observed["safety_depth"], "two_high")


class LearnerTests(unittest.TestCase):
    def test_sparse_pressure_stays_uncertain(self):
        model = fit_opponent_model(records=_pressure_game(3, True, 0, False))
        self.assertFalse(model["changes"])
        self.assertIsNone(model["strategy_hypothesis"])
        passing = next(row for row in model["estimates"] if row["context"] == "passing_downs")
        self.assertEqual(passing["state"], "insufficient")
        self.assertEqual(passing["confidence"], 0.0)

    def test_passing_down_pressure_change_needs_both_halves(self):
        model = fit_opponent_model(records=_pressure_game(8, False, 8, True))
        change = model["changes"][0]
        self.assertEqual(change["metric"], "pressure_on_passing_downs")
        self.assertEqual(change["direction"], "increased")
        self.assertFalse(change["automatic_conclusion"])
        self.assertEqual(model["strategy_hypothesis"]["id"], "learned_passing_down_pressure")

    def test_later_evidence_can_remove_the_change(self):
        rows = _changing_quiet()
        model = fit_opponent_model(records=rows)
        self.assertFalse(any(
            row["metric"] == "pressure_on_passing_downs" for row in model["changes"]
        ))
        self.assertIsNone(model["strategy_hypothesis"])

    def test_five_yards_on_third_and_12_is_not_a_conversion(self):
        from cfb_coach.madden.model.opponent_learning import _labels

        labeled = _labels({
            "down": 3, "distance": 12, "outcome": "gain 5",
        })
        self.assertFalse(labeled["success"])
        self.assertFalse(labeled["conversion"])
        early = _labels({"down": 1, "distance": 10, "outcome": "gain 5"})
        self.assertTrue(early["success"])
        self.assertFalse(early["conversion"])

    def test_recommendations_are_not_training_rows(self):
        from cfb_coach.madden.model.opponent_learning import records_from_memory_events

        records = records_from_memory_events([
            {
                "verification": "recommendation_only",
                "recommended_play": "Mesh",
                "outcome": "gain 20",
                "observed_defense": "blitz",
                "snap_seq": 1,
            },
            {
                "verification": "verified_execution",
                "executed_play": "Mesh",
                "executed_formation": "Gun Bunch",
                "outcome": "gain 4",
                "observed_defense": "no pressure",
                "snap_seq": 2,
                "presnap": {"down": 3, "distance": 8},
            },
        ])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["play"], "Mesh")
        model = fit_opponent_model(records=records)
        self.assertEqual(model["usable_verified_snaps"], 1)
        self.assertFalse(model["trained_on_recommendations"])


def _changing_quiet() -> list[dict]:
    rows = []
    seq = 1
    for flag, count in ((False, 8), (True, 8), (False, 8)):
        for _ in range(count):
            rows.append({
                "snap_seq": seq,
                "down": 3,
                "distance": 8,
                "yards": 2 if flag else 9,
                "formation": "Gun Bunch",
                "play": "Mesh",
                "concept_id": "mesh",
                "observation": {
                    "pressure": flag, "blitz": flag, "legacy_ambiguous": False,
                    "coverage_shell": None,
                },
                "verified_execution": True,
                "prelabeled": True,
                "success": not flag,
                "conversion": not flag,
                "turnover": False,
                "sack": False,
            })
            seq += 1
    return rows


class JointTests(unittest.TestCase):
    def test_learning_can_tip_a_tie_and_cannot_overturn_a_better_play(self):
        records = _pressure_game(8, False, 8, True)
        memory = {"verified_snap_records": records}
        book = {"Gun Bunch": ["Mesh", "Power"]}
        ranked = [
            {"formation": "Gun Bunch", "play": "Mesh", "probability": 0.55, "selection_score": 0.55},
            {"formation": "Gun Bunch", "play": "Power", "probability": 0.55, "selection_score": 0.55},
        ]
        learned = choose_joint_action(
            ranked=ranked, anchor=ranked[0], sit=_sit(), book=book, memory=memory,
            active=[], db=None, session_id="s13", snap_seq=20,
            use_knowledge=False, use_strategy=True, use_opponent_learning=True,
        )
        baseline = choose_joint_action(
            ranked=ranked, anchor=ranked[0], sit=_sit(), book=book, memory=memory,
            active=[], db=None, session_id="s13", snap_seq=20,
            use_knowledge=False, use_strategy=True, use_opponent_learning=False,
        )
        self.assertEqual(learned["joint"]["play"], "Mesh")
        self.assertGreater(
            learned["football_intelligence"]["winner"]["learning_delta"], 0.0,
        )
        self.assertLessEqual(
            learned["football_intelligence"]["winner"]["learning_delta"], LEARNING_CAP,
        )
        self.assertEqual(
            learned["football_intelligence"]["strategy"]["hypothesis_id"],
            "learned_passing_down_pressure",
        )
        self.assertEqual(baseline["football_intelligence"]["winner"]["learning_delta"], 0.0)
        strong = [
            {"formation": "Gun Bunch", "play": "Power", "probability": 0.82, "selection_score": 0.82},
            {"formation": "Gun Bunch", "play": "Mesh", "probability": 0.55, "selection_score": 0.55},
        ]
        kept = choose_joint_action(
            ranked=strong, anchor=strong[0], sit=_sit(), book=book, memory=memory,
            active=[], db=None, session_id="s13", snap_seq=21,
            use_knowledge=False, use_strategy=False, use_opponent_learning=True,
        )
        self.assertEqual(kept["joint"]["play"], "Power")

    def test_current_pressure_does_not_stack_a_second_learning_bonus(self):
        model = fit_opponent_model(records=_pressure_game(8, False, 8, True))
        adj = learning_adjustment(
            "Mesh", _sit(coverage_hint="showing blitz", coverage_source="live"),
            model, current_pressure_observed=True,
        )
        self.assertEqual(adj["delta"], 0.0)
        self.assertTrue(adj["withheld"])

    def test_poor_conversion_revises_the_hypothesis(self):
        rows = []
        for seq in range(1, 17):
            rows.append({
                "snap_seq": seq,
                "down": 3,
                "distance": 8,
                "yards": 1,
                "formation": "Gun Bunch",
                "play": "Mesh",
                "concept_id": "mesh",
                "observation": {
                    "pressure": seq > 8, "blitz": seq > 8, "legacy_ambiguous": False,
                    "coverage_shell": None,
                },
                "verified_execution": True,
                "prelabeled": True,
                "success": False,
                "conversion": False,
                "turnover": False,
                "sack": False,
            })
        model = fit_opponent_model(records=rows)
        self.assertEqual(model["strategy_hypothesis"]["id"], "revised_away_from_quick_pressure")
        plan = current_strategy(_sit(), learned=model)
        self.assertTrue(plan["revised"])
        adj = learning_adjustment("Mesh", _sit(), model)
        self.assertEqual(adj["delta"], 0.0)

    def test_full_book_stays_inside_the_latency_budget(self):
        forms = [f"Formation {i}" for i in range(15)]
        book = {form: [f"{form} Play {n}" for n in range(12)] for form in forms}
        ranked = [
            {"formation": form, "play": play, "probability": 0.5, "selection_score": 0.5}
            for form, plays in book.items() for play in plays
        ]
        memory = {"verified_snap_records": _pressure_game(8, False, 8, True)}
        started = time.perf_counter()
        decision = choose_joint_action(
            ranked=ranked, anchor=ranked[0], sit=_sit(), book=book, memory=memory,
            active=[], db=None, session_id="latency", snap_seq=3,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.assertEqual(ML_LATENCY_BUDGET_MS, 150)
        self.assertLess(elapsed_ms, ML_LATENCY_BUDGET_MS)
        self.assertEqual(decision["diversity"]["consideration_rate"], 1.0)
        self.assertEqual(decision["plays_considered"], 180)


class ReplayTests(unittest.TestCase):
    def test_snap_n_cannot_see_itself(self):
        game = _changing_game()
        for index in range(len(game)):
            model = replay_prior_only(game, index)
            self.assertEqual(model["usable_verified_snaps"], index)
            self.assertTrue(all(row["snap_seq"] < game[index]["snap_seq"] for row in game[:index]))
        report = evaluate_opponent_learning()
        self.assertEqual(report["temporal_leakage_failures"], 0)
        self.assertFalse(report["sparse_three_snaps_published_change"])
        self.assertEqual(report["after_16_snaps"]["hypothesis"], "learned_passing_down_pressure")
        self.assertIsNone(report["after_reversal"]["hypothesis"])
        self.assertFalse(report["win_rate_claim"])
        self.assertEqual(report["held_out_evaluation"], "insufficient")
        self.assertFalse(report["learned_from_video"])
        learned = report["policy_calls_at_snap_16"]["sprint13_opponent_learning"]
        both = report["policy_calls_at_snap_16"]["sprint13_learning_and_strategy"]
        self.assertEqual(learned["play"], "Mesh")
        self.assertEqual(both["play"], "Mesh")
        self.assertGreater(learned["learning_delta"], 0)


class VideoTests(unittest.TestCase):
    def test_dry_run_keeps_rows_pending_and_rejects_bad_ones(self):
        fixtures = synthetic_video_fixtures()
        bad = [
            {"human_verification": "verified_manual", "confidence": 0.9},
            {
                "game_id": "g",
                "recording_id": "r",
                "snap_id": "s",
                "unresolved_snap_association": "also",
                "video_timestamp": 1,
                "observation_time_relative_to_snap": "pre_snap",
                "confidence": 0.4,
                "human_verification": "pending",
                "defensive_alignment": {"shells_mentioned": ["cover_1", "cover_2"]},
                "observed_game_state": {"down": 1},
            },
            {},
        ]
        result = dry_run_import(fixtures + bad)
        self.assertFalse(result["writes_database"])
        self.assertFalse(result["learned_from_video"])
        self.assertTrue(result["separate_from_verified_observations"])
        self.assertEqual(result["accepted_count"], 2)
        self.assertGreaterEqual(result["rejected_count"], 3)
        self.assertTrue(all(row["status"] == "video_pending_validation" for row in result["accepted"]))
        reasons = {reason for row in result["rejected"] for reason in row["reasons"]}
        self.assertIn("verified_manual_is_not_a_video_status", reasons)
        self.assertIn("contradictory_shells", reasons)
        self.assertIn("empty_observation", reasons)
        pending = validate_video_observation(fixtures[0])
        self.assertFalse(pending["verified_execution"])


class ReportTests(unittest.TestCase):
    def test_missing_database_is_read_only(self):
        from cfb_coach.madden.model.cli import cmd_ml_opponent_learning

        missing = readonly_madden_db_path().with_name("definitely-missing-madden.db")
        args = SimpleNamespace(opponent="cpu", game_id=None)
        with mock.patch(
            "cfb_coach.madden.model.opponent_learning.readonly_madden_db_path",
            return_value=missing,
        ):
            code = cmd_ml_opponent_learning(args)
        self.assertEqual(code, 2)
        report = opponent_learning_report(None, opponent_id="cpu")
        self.assertFalse(report["history_modified"])
        self.assertFalse(report["learned_from_video"])
        self.assertEqual(report["held_out_evaluation"], "insufficient")
        self.assertIn("verified_executions_with_outcomes", report["missing_data"])


if __name__ == "__main__":
    unittest.main()
