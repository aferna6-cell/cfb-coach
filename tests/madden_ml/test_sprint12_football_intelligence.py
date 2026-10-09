"""Sprint 12: football knowledge, diagnosis, strategy, and the 15-formation cap."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model.concept_matchup import KNOWLEDGE_CAP, evaluate_concept_matchup
from cfb_coach.madden.model.defensive_diagnosis import (
    diagnose_defense,
    shrink_shells,
)
from cfb_coach.madden.model.features import FEATURE_SCHEMA_VERSION, _FEATURE_NAMES, intelligence_features
from cfb_coach.madden.model.football_knowledge import (
    CONCEPTS,
    LIVE_PATH_CALLS_LLM,
    compiled_knowledge,
    knowledge_report,
    profile_for_play,
    validate_knowledge,
)
from cfb_coach.madden.model.offense_intelligence_eval import (
    USER_REPORTED_CONTEXT,
    evaluate_intelligence,
)
from cfb_coach.madden.model.offense_joint_decision import choose_joint_action
from cfb_coach.madden.model.offense_portfolio import select_formation_portfolio
from cfb_coach.madden.model.offense_strategy import (
    STRATEGY_CAP,
    current_strategy,
    save_strategy,
    strategy_adjustment,
    strategy_report,
)
from cfb_coach.madden.model.schema import ML_LATENCY_BUDGET_MS


def _sit(**kwargs):
    base = dict(
        down=1, distance=10, yardline=40, red_zone=False, goal_line=False,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={}, quarter=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _rows(book, score=0.55):
    out = []
    for form, plays in book.items():
        for play in plays:
            out.append({
                "formation": form, "play": play, "probability": score,
                "selection_score": score, "uncertainty": 0.6,
            })
    return out


class KnowledgeTests(unittest.TestCase):
    def test_compiled_store_has_no_invented_routes(self):
        self.assertEqual(validate_knowledge(), [])
        compiled = compiled_knowledge()
        self.assertFalse(compiled["live_path_calls_llm"])
        self.assertFalse(LIVE_PATH_CALLS_LLM)
        for row in CONCEPTS.values():
            self.assertIsNone(row["route_diagram"])
            self.assertIsNone(row["player_assignments"])
            self.assertIsNone(row["controller_inputs"])
            self.assertNotEqual(row["madden_27_applicability"], "verified")
        mesh = profile_for_play("Mesh")
        self.assertEqual(mesh["concept_id"], "mesh")
        self.assertEqual(mesh["layer"], "hypothesis")
        self.assertIsNone(mesh["route_diagram"])
        self.assertIn("name", mesh["note"].lower())
        unknown = profile_for_play("Totally Unknown Play")
        self.assertIsNone(unknown["concept_id"])
        self.assertIsNone(unknown["route_diagram"])

    def test_validation_rejects_an_unverified_route_diagram(self):
        forged = {"mesh": dict(CONCEPTS["mesh"])}
        forged["mesh"]["route_diagram"] = {"wr1": "shallow"}
        problems = validate_knowledge(forged)
        self.assertTrue(any("route diagram" in item for item in problems))

    def test_report_names_sources_and_does_not_leak_user_games(self):
        report = knowledge_report("mesh")
        self.assertTrue(report["found"])
        blob = str(report)
        self.assertNotIn("9f2ebdeb9d8f4e2d", blob)
        self.assertNotIn("d2e3214fbb944af9", blob)
        self.assertTrue(report["sources"])
        self.assertIn("url", report["sources"][0])

    def test_training_vector_is_unchanged(self):
        self.assertEqual(FEATURE_SCHEMA_VERSION, "madden-ml.features.2")
        self.assertGreater(len(_FEATURE_NAMES), 40)
        extra = intelligence_features("Mesh")
        self.assertEqual(extra["training_vector"], "madden-ml.features.2")
        self.assertIsNone(extra["route_diagram"])
        self.assertIsNone(extra["controller_inputs"])


class DiagnosisTests(unittest.TestCase):
    def test_two_high_is_not_cover_2_or_quarters(self):
        picture = diagnose_defense(
            _sit(coverage_hint="showing two-high", coverage_source="live"),
            formation_name="Quarters",
        )
        self.assertEqual(picture["state"], "observed")
        self.assertIsNone(picture["observed"]["shell"])
        self.assertEqual(picture["observed"]["structure"], "two_high")
        self.assertTrue(picture["formation_name_ignored"])

    def test_previous_snap_is_not_current_coverage(self):
        picture = diagnose_defense(
            _sit(coverage_hint="cover 1", coverage_source="last"),
            formation_name="Cover 3",
        )
        self.assertEqual(picture["state"], "unknown")
        self.assertIsNone(picture["observed"]["shell"])
        self.assertTrue(picture["previous_snap_not_copied"])

    def test_sparse_shells_stay_unknown_until_the_sample_moves(self):
        thin = shrink_shells({"cover_1": 1})
        self.assertEqual(thin["state"], "unknown")
        self.assertIsNone(thin["mode"])
        strong = shrink_shells({"cover_1": 5})
        self.assertEqual(strong["state"], "inferred")
        self.assertEqual(strong["mode"], "cover_1")
        picture = diagnose_defense(None, {"distribution": {"man": 9}})
        self.assertIsNone(picture["inferred"]["shell"])

    def test_explicit_cover_names_are_observed_only_on_the_live_snap(self):
        picture = diagnose_defense(
            _sit(coverage_hint="showing cover 2", coverage_source="live")
        )
        self.assertEqual(picture["observed"]["shell"], "cover_2")
        self.assertIn("zone", picture["observed"]["principles"])


class MatchupTests(unittest.TestCase):
    def test_third_and_8_does_not_bonus_mesh_for_man_coverage(self):
        sit = _sit(down=3, distance=8, coverage_hint="showing cover 1", coverage_source="live")
        mesh = evaluate_concept_matchup("Mesh", sit)
        self.assertEqual(mesh["delta"], 0.0)
        self.assertTrue(mesh["withheld"])
        self.assertLessEqual(mesh["delta"], KNOWLEDGE_CAP)
        book = {
            "Gun Bunch": ["Mesh", "Inside Zone", "Four Verticals", "Flood"],
        }
        decision = choose_joint_action(
            ranked=_rows(book), anchor=_rows(book)[0], sit=sit, book=book,
            active=[], db=None, session_id="s", snap_seq=1,
        )
        self.assertEqual(decision["joint"]["play"], "Four Verticals")
        self.assertNotEqual(decision["joint"]["play"], "Mesh")
        self.assertTrue(decision["football_intelligence"]["summary"])

    def test_third_and_4_man_can_prefer_mesh_without_overturning_a_better_play(self):
        sit = _sit(down=3, distance=4, coverage_hint="showing cover 1", coverage_source="live")
        mesh = evaluate_concept_matchup("Mesh", sit)
        self.assertGreater(mesh["delta"], 0.0)
        self.assertLessEqual(mesh["delta"], KNOWLEDGE_CAP)
        book = {"Gun Bunch": ["Mesh", "Power"]}
        tied = choose_joint_action(
            ranked=_rows(book), anchor=_rows(book)[0], sit=sit, book=book,
            active=[], db=None, session_id="s", snap_seq=2,
        )
        self.assertEqual(tied["joint"]["play"], "Mesh")
        ranked = _rows(book)
        for row in ranked:
            if row["play"] == "Power":
                row["selection_score"] = 0.80
                row["probability"] = 0.80
        stronger = choose_joint_action(
            ranked=ranked, anchor=ranked[0], sit=sit, book=book,
            active=[], db=None, session_id="s", snap_seq=3,
        )
        self.assertEqual(stronger["joint"]["play"], "Power")

    def test_knowledge_prior_is_not_inside_the_selection_score(self):
        from cfb_coach.madden.model.football_situation import selection_adjustment

        delta, _reason = selection_adjustment(
            "Mesh", _sit(down=3, distance=4, coverage_hint="showing cover 1", coverage_source="live"),
        )
        self.assertEqual(delta, 0.0)


class StrategyTests(unittest.TestCase):
    def test_underneath_hypothesis_can_break_a_tie_and_is_not_a_run_script(self):
        memory = {"observed_labels": ["cover 2"] * 5, "sample_size": 5}
        diagnosis = diagnose_defense(_sit(), memory)
        self.assertEqual(diagnosis["inferred"]["shell"], "cover_2")
        plan = current_strategy(_sit(down=1, distance=10), memory, diagnosis)
        self.assertEqual(plan["hypothesis_id"], "underneath_space")
        self.assertTrue(plan["not_a_script"])
        mesh = strategy_adjustment("Mesh", plan, knowledge_delta=0.0)
        power = strategy_adjustment("Power", plan, knowledge_delta=0.0)
        self.assertEqual(mesh["delta"], STRATEGY_CAP)
        self.assertEqual(power["delta"], 0.0)
        self.assertTrue(any("establish the run" in line for line in power["reasons"]))
        book = {"Gun Bunch": ["Mesh", "Power"]}
        full = choose_joint_action(
            ranked=_rows(book), anchor=_rows(book)[0], sit=_sit(down=1, distance=10),
            book=book, memory=memory, active=[], db=None, session_id="s", snap_seq=4,
            use_knowledge=True, use_strategy=True,
        )
        baseline = choose_joint_action(
            ranked=_rows(book), anchor=_rows(book)[0], sit=_sit(down=1, distance=10),
            book=book, memory=memory, active=[], db=None, session_id="s", snap_seq=4,
            use_knowledge=False, use_strategy=False,
        )
        self.assertEqual(full["joint"]["play"], "Mesh")
        self.assertEqual(full["policy_version"], "joint_offense_action.v3")
        self.assertEqual(baseline["policy_version"], "joint_offense_action.v2")
        self.assertEqual(full["football_intelligence"]["winner"]["strategy_delta"], STRATEGY_CAP)
        self.assertEqual(baseline["football_intelligence"]["winner"]["strategy_delta"], 0.0)

    def test_verified_failures_revise_the_hypothesis(self):
        memory = {
            "observed_labels": ["cover 2"] * 5,
            "verified_concepts": {"success": {}, "failure": {"mesh": 3}},
        }
        plan = current_strategy(_sit(down=2, distance=6), memory)
        self.assertEqual(plan["hypothesis_id"], "revised_away_from_underneath")
        adj = strategy_adjustment("Mesh", plan, knowledge_delta=0.0)
        self.assertEqual(adj["delta"], 0.0)

    def test_unknown_drive_boundary_keeps_the_previous_hypothesis(self):
        previous = {
            "hypothesis_id": "underneath_space",
            "hypothesis": "Opponent frequently exposes underneath space while protecting deep zones.",
            "objective": "exploit_recurring_alignment",
            "revised": False,
        }
        plan = current_strategy(None, None, previous=previous)
        self.assertEqual(plan["drive_boundary"], "unknown")
        self.assertTrue(plan["kept_previous_hypothesis"])
        self.assertEqual(plan["hypothesis_id"], "underneath_space")

    def test_strategy_save_does_not_rewrite_snaps(self):
        from cfb_coach.db import CoachDB

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = CoachDB(Path(tmp.name) / "madden.db", seed={"opponents": {}})
        self.addCleanup(db.close)
        before = db.conn.execute("SELECT COUNT(*) AS n FROM snaps").fetchone()["n"]
        save_strategy(db, "session", current_strategy(_sit(down=1, distance=10)))
        after = db.conn.execute("SELECT COUNT(*) AS n FROM snaps").fetchone()["n"]
        self.assertEqual(before, after)
        report = strategy_report(db, "cpu")
        self.assertFalse(report["history_modified"])
        self.assertIn("no_staged_playbook", report["information_gaps"])


class PortfolioTests(unittest.TestCase):
    def _portfolio(self, books, limit):
        def flat(_art, _form, _plays, _opp, _name, _probe):
            return 0.4

        with mock.patch(
            "cfb_coach.madden.model.offense_portfolio._situation_value",
            side_effect=flat,
        ):
            return select_formation_portfolio(
                art=object(), all_books=books, active={"formations": {}},
                opponent_id="cpu", max_formations=limit, db=None,
            )

    def test_identical_formations_do_not_fill_the_cap(self):
        books = {"Buccaneers": {
            f"Gun {i:02d}": ["Mesh", "Inside Zone"] for i in range(15)
        }}
        result = self._portfolio(books, 15)
        self.assertEqual(len(result["chosen"]), 1)
        self.assertEqual(result["chosen"][0]["plays"], ["Mesh", "Inside Zone"])
        self.assertTrue(result["football_plan"]["cap_is_not_a_quota"])
        self.assertTrue(result["football_plan"]["plays_not_trimmed"])

    def test_new_concept_families_are_kept_and_the_cap_is_15(self):
        plays = {
            0: ["Mesh", "Inside Zone"],
            1: ["Flood", "Inside Zone"],
            2: ["Dagger", "Inside Zone"],
            3: ["Smash", "Inside Zone"],
            4: ["Stick", "Inside Zone"],
            5: ["Screen", "Inside Zone"],
        }
        books = {"Buccaneers": {f"Gun {i:02d}": plays[i] for i in plays}}
        result = self._portfolio(books, 15)
        self.assertEqual(len(result["chosen"]), 6)
        wide = {"Buccaneers": {
            f"Gun {i:02d}": [f"Family{i} Mesh", "Inside Zone"] for i in range(20)
        }}
        # Same mesh concept: diversity comes from the distinct play name only
        # when the name introduces a new concept. Twenty copies of mesh stop early.
        capped = self._portfolio(wide, 15)
        self.assertLess(len(capped["chosen"]), 15)
        distinct = {"Buccaneers": {
            f"Set {i:02d}": [name, "Inside Zone"]
            for i, name in enumerate([
                "Mesh", "Flood", "Dagger", "Drive", "Levels", "Smash", "Stick",
                "Spacing", "Curl Flat", "Four Verticals", "Slant Flat", "Wheel",
                "Texas", "Screen", "Play Action", "Counter",
            ])
        }}
        filled = self._portfolio(distinct, 15)
        self.assertEqual(len(filled["chosen"]), 15)
        self.assertEqual(designer.MAX_FORMATIONS, 15)
        with self.assertRaises(ValueError):
            designer.design_offense(
                object(), opponent_id="cpu", max_formations=16, catalogue=distinct,
            )


class JointLatencyTests(unittest.TestCase):
    def test_fifteen_formation_book_stays_inside_the_budget(self):
        book = {
            f"Gun {i:02d}": ["Mesh", "Inside Zone", "Flood", "Four Verticals"] + [
                f"Package {i}-{n}" for n in range(8)
            ]
            for i in range(15)
        }
        rows = _rows(book, 0.5)
        started = time.perf_counter()
        decision = choose_joint_action(
            ranked=rows, anchor=rows[0], sit=_sit(), book=book,
            active=[], db=None, session_id="latency", snap_seq=1,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self.assertEqual(decision["plays_considered"], 15 * 12)
        self.assertLess(elapsed, ML_LATENCY_BUDGET_MS)
        self.assertEqual(ML_LATENCY_BUDGET_MS, 150)
        live = _sit(coverage_hint="showing cover 1", coverage_source="live")
        started = time.perf_counter()
        live_decision = choose_joint_action(
            ranked=rows, anchor=rows[0], sit=live, book=book,
            active=[], db=None, session_id="latency", snap_seq=2,
        )
        live_elapsed = (time.perf_counter() - started) * 1000.0
        self.assertEqual(live_decision["plays_considered"], 180)
        self.assertLess(live_elapsed, ML_LATENCY_BUDGET_MS)


class EvaluationTests(unittest.TestCase):
    def test_ablation_explains_differences_without_a_win_rate_claim(self):
        report = evaluate_intelligence()
        self.assertTrue(report["not_a_win_rate_claim"])
        self.assertFalse(report["live_path_calls_llm"])
        self.assertEqual(report["validation_problems"], [])
        self.assertEqual(report["mesh_third_and_8_man_delta"], 0.0)
        self.assertIsNone(report["mesh_route_diagram"])
        self.assertIsNone(report["two_high_shell"])
        self.assertTrue(report["within_150_ms"])
        self.assertEqual(report["plays_considered_15"], 180)
        self.assertFalse(report["local_database"]["opened"])
        self.assertTrue(all(row["why"] for row in report["comparisons"]))
        self.assertTrue(all(row["not_a_win_rate_claim"] for row in report["comparisons"]))
        reported = {row["game_id"]: row for row in USER_REPORTED_CONTEXT}
        self.assertTrue(reported["9f2ebdeb9d8f4e2d"]["not_evidence_for_sprint_12"])
        self.assertTrue(reported["d2e3214fbb944af9"]["not_a_completed_win"])
        self.assertIsNone(reported["d2e3214fbb944af9"]["final_score"])

    def test_commands_are_read_only(self):
        from cfb_coach.cli import build_parser

        parser = build_parser()
        knowledge = parser.parse_args(["ml", "football-knowledge", "--concept", "mesh"])
        strategy = parser.parse_args(["ml", "offense-strategy", "-o", "cpu"])
        self.assertEqual(knowledge.concept, "mesh")
        self.assertEqual(strategy.opponent, "cpu")
        self.assertEqual(knowledge.func.__name__, "cmd_ml_football_knowledge")
        self.assertEqual(strategy.func.__name__, "cmd_ml_offense_strategy")
