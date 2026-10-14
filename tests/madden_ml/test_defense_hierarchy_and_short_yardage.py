"""Hierarchical defense objective and short-yardage offense competition."""
from __future__ import annotations

import json
import time
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import defense_coordinator as defense
from cfb_coach.madden.model.call_outcome_log import (
    clock_from_situation_text,
    record_final_score_gap,
)
from cfb_coach.madden.model.defense_hierarchy import (
    HIERARCHY_VERSION,
    audit_defensive_rows,
    evaluate_defense_hierarchy,
    fit_defense_hierarchy,
    go_nogo_report,
    objective_cost,
    predict_cost,
    promotion_status,
    reconcile_interceptions,
    record_promotion,
    turnover_attribution,
)
from cfb_coach.madden.model.football_situation import (
    compare_short_yardage_policies,
    selection_adjustment,
    short_yardage_policy_report,
)
from cfb_coach.madden.model.offense_selection_policy import choose_model_play
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.situation import Situation


def _verified(play: str, *, success: str, opponent: str, game: str, yards: int,
              touchdown: bool = False, goal_line: bool = False, **extra) -> dict:
    row = {
        "side": "defense",
        "opponent_type": "human",
        "opponent_id": opponent,
        "game_id": game,
        "supervised_eligible": True,
        "executed_status": "identified",
        "executed_verification": "verified",
        "executed_play": play,
        "label_available": True,
        "success": success,
        "yards": yards,
        "result": "touchdown" if touchdown else f"+{yards}",
        "down": 3,
        "distance": 1,
        "goal_line": goal_line,
        "red_zone": False,
        "two_minute": False,
    }
    row.update(extra)
    return row


def _sit(**kwargs) -> Situation:
    raw = kwargs.pop("raw", "d 3&1 goal line")
    sit = Situation(
        raw=raw, side="defense", down=kwargs.pop("down", 3),
        distance=kwargs.pop("distance", 1), goal_line=kwargs.pop("goal_line", True),
    )
    for key, value in kwargs.items():
        setattr(sit, key, value)
    return sit


class AuditTests(unittest.TestCase):
    def test_reported_178_is_not_a_supervised_count(self) -> None:
        rows = []
        for game_index, game in enumerate(("g1", "g2", "g3")):
            for i in range(10):
                rows.append(_verified(
                    "Cover 4 Drop", success="true", opponent="gavin", game=game, yards=2,
                ))
            for i in range(20):
                rows.append({
                    "side": "defense", "opponent_type": "human", "opponent_id": "john",
                    "game_id": game, "recommended_play": "Cover 3 Sky",
                    "executed_status": "unknown", "executed_verification": "unknown",
                    "recommendation_only": True,
                })
            for i in range(8):
                rows.append({
                    "side": "defense", "opponent_type": "human", "opponent_id": "john",
                    "game_id": game, "executed_play": "Cover 3 Sky",
                    "executed_status": "identified", "executed_verification": "unknown",
                    "success": "true", "yards": 4, "label_available": True,
                    "supervised_eligible": True,
                })
        # 3 games * (10 + 20 + 8) = 114. Pad with unlabeled defense rows to 178.
        while len(rows) < 178:
            rows.append({
                "side": "defense", "opponent_type": "cpu", "opponent_id": "cpu",
                "game_id": "cpu-pad", "executed_play": "Cover 3", "success": "true",
                "executed_status": "identified", "executed_verification": "verified",
                "label_available": True, "supervised_eligible": True, "yards": 1,
            })
        self.assertEqual(len(rows), 178)
        report = audit_defensive_rows(rows, reported_candidates=178)
        self.assertEqual(report["candidate_snaps"], 178)
        self.assertLess(report["supervised_snaps"], 178)
        self.assertEqual(report["reported_count_is_supervised"], False)
        self.assertIn("execution_not_verified", report["excluded_reasons"])
        self.assertIn("recommendation_only", report["excluded_reasons"])
        self.assertIn("not_human", report["excluded_reasons"])
        supervised_by_game = sum(bucket["supervised"] for bucket in report["by_game"].values())
        self.assertEqual(supervised_by_game, report["supervised_snaps"])
        self.assertEqual(report["by_game"]["g1"]["supervised"], 10)
        self.assertGreater(report["logging_gaps"].get("drive_id_missing", 0), 0)
        self.assertGreater(report["logging_gaps"].get("game_clock_missing", 0), 0)
        unlabeled = dict(_verified(
            "Cover 4 Drop", success="true", opponent="gavin", game="g1", yards=2,
        ))
        unlabeled["label_available"] = False
        self.assertEqual(
            audit_defensive_rows([unlabeled])["excluded_reasons"].get("label_unavailable"),
            1,
        )
        unknown_family = _verified(
            "Not A Coverage", success="true", opponent="gavin", game="g1", yards=2,
        )
        self.assertEqual(
            audit_defensive_rows([unknown_family])["excluded_reasons"].get(
                "family_not_in_defensive_menu",
            ),
            1,
        )

    def test_empty_checkout_does_not_invent_the_178(self) -> None:
        report = audit_defensive_rows([], reported_candidates=178)
        self.assertEqual(report["candidate_snaps"], 0)
        self.assertEqual(report["supervised_snaps"], 0)
        self.assertIn("not a training count", report["explanation"])
        verdict = go_nogo_report([])
        self.assertEqual(verdict["verdict"], "NO-GO")
        self.assertIsNone(verdict["launch_command"])
        self.assertIn("defense-experimental --disable", verdict["rollback_command"])
        self.assertFalse(verdict["auto_promote"])
        self.assertTrue(any("Issue #75 records 45" in line for line in verdict["reasons"]))
        self.assertTrue(any("178" in line for line in verdict["reasons"]))


class HierarchyEvidenceTests(unittest.TestCase):
    def test_touchdown_costs_more_than_a_routine_first_down(self) -> None:
        routine = objective_cost(conversion=1, yards=4, explosive=0, touchdown=0)
        bomb = objective_cost(conversion=1, yards=40, explosive=1, touchdown=1)
        self.assertGreater(bomb, routine * 3)

    def test_trained_evidence_changes_the_call(self) -> None:
        book = {"Nickel": ["Cover 4 Drop", "Cover 0"]}
        rows = []
        for i in range(16):
            rows.append(_verified(
                "Cover 4 Drop", success="true", opponent="gavin", game="g1",
                yards=2, goal_line=True,
            ))
            rows.append(_verified(
                "Cover 0", success="false", opponent="gavin", game="g1",
                yards=40, touchdown=True, goal_line=True,
            ))
        model = defense.train_defense(rows)
        self.assertEqual(model["supervised_defensive_snaps"], 32)
        self.assertEqual(model["hierarchy"]["version"], HIERARCHY_VERSION)
        self.assertEqual(model["offense_artifact"], "not_modified")
        self.assertFalse(model["auto_promote"])
        self.assertTrue(model["hierarchy"]["single_opponent_overfit_risk"])
        plain = defense.rank_defense(_sit(), book, None)
        trained = defense.rank_defense(_sit(), book, model, opponent_id="gavin")
        self.assertEqual(len(trained), 2)
        self.assertEqual(trained[0]["play"], "Cover 4 Drop")
        self.assertEqual(trained[0]["score_source"], "hierarchical_objective")
        self.assertNotEqual(plain[0]["play"], trained[0]["play"])
        self.assertLess(trained[0]["defensive_prediction"]["cost"], trained[1]["defensive_prediction"]["cost"])

    def test_opponent_residuals_do_not_replace_general_knowledge(self) -> None:
        rows = []
        for game in ("jg1", "jg2"):
            for _ in range(10):
                rows.append(_verified(
                    "Cover 3 Sky", success="false", opponent="john", game=game,
                    yards=30, touchdown=True,
                ))
        for game in ("gg1", "gg2"):
            for _ in range(10):
                rows.append(_verified(
                    "Cover 3 Sky", success="true", opponent="gavin", game=game, yards=1,
                ))
        fitted = fit_defense_hierarchy(rows)
        self.assertTrue(fitted["general_separated_from_opponents"])
        context = _verified("Cover 3 Sky", success="true", opponent="gavin", game="gg1", yards=1)
        john = predict_cost(fitted, play="Cover 3 Sky", context=context, opponent_id="john")
        gavin = predict_cost(fitted, play="Cover 3 Sky", context=context, opponent_id="gavin")
        other = predict_cost(fitted, play="Cover 3 Sky", context=context, opponent_id="tiano")
        self.assertGreater(john["conversion"], other["conversion"])
        self.assertGreater(other["conversion"], gavin["conversion"])
        self.assertLess(john["conversion"], 0.9)
        self.assertEqual(other["opponent_snaps_used"], 0)
        self.assertFalse(other["post_snap_features_used"])

    def test_post_snap_concept_does_not_change_the_fit(self) -> None:
        base = _verified("Cover 4 Drop", success="true", opponent="gavin", game="g1", yards=3)
        other = dict(base, concept_seen="Four Verticals", coverage_seen="Cover 3")
        left = fit_defense_hierarchy([base])
        right = fit_defense_hierarchy([other])
        self.assertEqual(left["opponents"], right["opponents"])
        self.assertEqual(left["leakage_fields_used"], [])

    def test_single_opponent_and_empty_eval_refuse_promotion(self) -> None:
        rows = [
            _verified("Cover 4 Drop", success="true", opponent="john", game=f"g{i // 8}", yards=2)
            for i in range(40)
        ]
        model = defense.train_defense(rows)
        self.assertFalse(promotion_status(model)["allowed"])
        report = evaluate_defense_hierarchy(rows, seed=7, latency_ms=[1.0, 2.0, 3.0])
        self.assertFalse(report["promotion"]["allowed"])
        self.assertIn("opponents", report["promotion"]["reason"])
        self.assertTrue(any("not the yards an uncalled play" in line for line in report["limitations"]))
        self.assertFalse(report["heuristic_comparison"]["available"])

    def test_held_out_ranking_disagrees_with_the_heuristic(self) -> None:
        rows = []
        for game in ("g1", "g2", "g3", "g4"):
            for _ in range(8):
                rows.append(_verified(
                    "Cover 4 Drop", success="true", opponent="gavin", game=game,
                    yards=2, goal_line=True,
                ))
                rows.append(_verified(
                    "Cover 0", success="false", opponent="john", game=game,
                    yards=40, touchdown=True, goal_line=True,
                ))
        book = {"Nickel": ["Cover 4 Drop", "Cover 0"]}
        report = evaluate_defense_hierarchy(
            rows, book=book, seed=7, latency_ms=[1.0, 1.5, 2.0],
        )
        comparison = report["heuristic_comparison"]
        self.assertTrue(comparison["available"])
        self.assertEqual(comparison["top_play_agreement"], 0.0)
        self.assertIn("not the result an uncalled play", comparison["claim_limit"])
        self.assertIn("ranking difference", " ".join(report["limitations"]))
        fitted = fit_defense_hierarchy(rows)
        sit = _sit()
        trained = defense.rank_defense(
            sit, book, {"hierarchy": fitted, "supervised_defensive_snaps": fitted["supervised_snaps"]},
            opponent_id="gavin",
        )
        plain = defense.rank_defense(sit, book, None)
        self.assertEqual(trained[0]["play"], "Cover 4 Drop")
        self.assertEqual(trained[0]["score_source"], "hierarchical_objective")
        self.assertNotEqual(plain[0]["play"], trained[0]["play"])
        self.assertTrue(report["promotion"]["allowed"])


class LiveDefenseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed={"opponents": {}})
        self.book = {"Nickel": ["Cover 4 Drop", "Cover 0"]}
        from cfb_coach.madden import playbook
        playbook._save_state(self.db, {"applied": {"defense": {
            "side": "defense", "mode": "custom", "name": "test",
            "source_book": None, "rev": 1, "locked_ts": "test",
            "formations": self.book,
        }}, "pending": {}})

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _model(self) -> dict:
        rows = []
        for game in ("g1", "g2", "g3", "g4"):
            for opponent, play, success, yards, td in (
                ("gavin", "Cover 4 Drop", "true", 2, False),
                ("john", "Cover 0", "false", 40, True),
            ):
                for _ in range(6):
                    rows.append(_verified(
                        play, success=success, opponent=opponent, game=game,
                        yards=yards, touchdown=td, goal_line=True,
                    ))
        return defense.train_defense(rows)

    def test_a_passing_eval_can_be_enabled_without_auto_promotion(self) -> None:
        from cfb_coach.madden.model.cli import cmd_ml_defense_experimental

        rows = []
        for game in ("g1", "g2", "g3", "g4"):
            for opponent, play, success, yards, td in (
                ("gavin", "Cover 4 Drop", "true", 2, False),
                ("john", "Cover 0", "false", 40, True),
            ):
                for _ in range(8):
                    rows.append(_verified(
                        play, success=success, opponent=opponent, game=game,
                        yards=yards, touchdown=td, goal_line=True,
                    ))
        art = defense.train_defense(rows)
        report = evaluate_defense_hierarchy(rows, seed=7, latency_ms=[1.0, 1.2, 1.4, 1.6])
        self.assertTrue(report["promotion"]["allowed"])
        self.assertFalse(report["promotion"]["auto_promote"])
        record_promotion(art, report)
        self.assertFalse(art["auto_promote"])
        dest = Path(self.tmp.name) / "defense.json"
        defense.save_model(art, dest)
        self.db.set_meta(defense.MODEL_META, str(dest))
        self.assertEqual(defense.mode(self.db), "off")
        code = cmd_ml_defense_experimental(Namespace(
            enable=True, disable=False, retrain=False,
            db=str(Path(self.tmp.name) / "madden.db"), out=None, cli_parser=None,
        ))
        self.assertEqual(code, 0)
        self.assertEqual(defense.mode(self.db), "experimental")

    def test_enable_does_not_train_or_go_live(self) -> None:
        from cfb_coach.madden.model.cli import cmd_ml_defense_experimental
        with mock.patch.object(defense, "train_defense", side_effect=AssertionError("trained")):
            code = cmd_ml_defense_experimental(Namespace(
                enable=True, disable=False, retrain=False,
                db=str(Path(self.tmp.name) / "madden.db"), out=None,
                cli_parser=None,
            ))
        self.assertEqual(code, 2)
        self.assertEqual(defense.mode(self.db), "off")

    def test_live_call_uses_the_model_and_falls_back(self) -> None:
        art = self._model()
        dest = Path(self.tmp.name) / "defense.json"
        defense.save_model(art, dest)
        self.db.set_meta(defense.MODEL_META, str(dest))
        defense.set_mode(self.db, "experimental")
        heuristic = MaddenCall("defense", "Nickel", "Cover 0", "No adj", "User hook", "h")
        sit = _sit()
        sit.extras = {
            "session_id": "live-1", "snap_seq": 3, "snap_id": "snap-1",
            "game_id": "g9", "defense_armed_ids": [],
            "require_trained_defense_artifact": True,
        }
        shown = defense.maybe_apply_defense(heuristic, sit, "gavin", self.db, self.book)
        self.assertIn(shown.play, self.book[shown.formation])
        self.assertEqual(shown.play, "Cover 4 Drop")
        self.assertIsNone(shown.macro)
        decision = getattr(shown, defense.DECISION_ATTR)
        self.assertFalse(decision.fell_back)
        self.assertLess(decision.latency_ms, 150)
        started = time.perf_counter()
        defense.rank_defense(sit, self.book, art, opponent_id="gavin")
        self.assertLess((time.perf_counter() - started) * 1000.0, 150)

        with mock.patch.object(defense, "live_pick", side_effect=TimeoutError("budget")):
            timed = defense.maybe_apply_defense(heuristic, sit, "gavin", self.db, self.book)
        self.assertEqual(timed.play, heuristic.play)
        self.assertTrue(timed._pending_ml_decision.fell_back)

        fake = {
            "selected": {"formation": "Nickel", "play": "NOT A CALL", "family": "man", "score": 1},
            "top": [{"formation": "Nickel", "play": "NOT A CALL", "family": "man", "score": 1,
                     "score_source": "hierarchical_objective", "football_intelligence": {}}],
            "all": [{"formation": "Nickel", "play": "NOT A CALL", "family": "man", "score": 1,
                     "score_source": "hierarchical_objective", "football_intelligence": {}}],
            "tendencies": {}, "model_quality": "limited", "n_candidates": 1,
        }
        with mock.patch.object(defense, "live_pick", return_value=fake):
            illegal = defense.maybe_apply_defense(heuristic, sit, "gavin", self.db, self.book)
        self.assertEqual(illegal.play, heuristic.play)
        self.assertIn("illegal play", illegal._pending_ml_decision.ml_status_detail)

    def test_unarmed_macro_is_dropped_and_the_play_stays(self) -> None:
        art = self._model()
        dest = Path(self.tmp.name) / "defense.json"
        defense.save_model(art, dest)
        self.db.set_meta(defense.MODEL_META, str(dest))
        defense.set_mode(self.db, "experimental")
        heuristic = MaddenCall("defense", "Nickel", "Cover 0", "No adj", "User hook", "h")
        sit = _sit()
        sit.extras = {
            "session_id": "live-2", "snap_seq": 4, "snap_id": "snap-2",
            "defense_armed_ids": ["SAFE DEEP"],
        }

        def _macro(*_args, **_kwargs):
            return {
                "kind": "macro", "id": "QTRS OVERTOP", "influence": 0.01,
                "why": "not armed", "executable": True,
                "macro": {"name": "QTRS OVERTOP", "settings": []},
            }

        with mock.patch(
            "cfb_coach.madden.model.defense_action_policy.select_defensive_action",
            side_effect=_macro,
        ):
            shown = defense.maybe_apply_defense(heuristic, sit, "gavin", self.db, self.book)
        self.assertEqual(shown.play, "Cover 4 Drop")
        self.assertIsNone(shown.macro)
        self.assertIn("QTRS OVERTOP", shown.adjustment_dropped)


class ShortYardageTests(unittest.TestCase):
    def test_passes_compete_and_a_clear_run_still_wins(self) -> None:
        sit = Situation(raw="3rd and 1", side="offense", down=3, distance=1)
        self.assertAlmostEqual(selection_adjustment("HB Dive", sit)[0], 0.025)
        self.assertAlmostEqual(selection_adjustment("Mesh", sit)[0], 0.02)
        self.assertAlmostEqual(selection_adjustment("PA Boot", sit)[0], 0.018)
        self.assertAlmostEqual(selection_adjustment("RPO Read", sit)[0], 0.02)
        self.assertAlmostEqual(selection_adjustment("Four Verticals", sit)[0], -0.035)
        report = short_yardage_policy_report()
        self.assertFalse(report["quota"])
        self.assertEqual(report["offense_artifact"], "not modified")
        self.assertEqual(report["baseline_deltas"]["quick"], 0.0)
        self.assertGreater(report["challenger_deltas"]["quick"], 0.0)
        close = compare_short_yardage_policies([
            {"play": "HB Dive", "probability": 0.55},
            {"play": "Mesh", "probability": 0.56},
        ])
        self.assertGreater(
            close["by_play"]["Mesh"]["challenger_score"],
            close["by_play"]["HB Dive"]["challenger_score"],
        )
        self.assertGreater(
            close["by_play"]["HB Dive"]["baseline_score"],
            close["by_play"]["Mesh"]["baseline_score"],
        )
        clear = [
            {"formation": "Gun", "play": "HB Dive", "probability": 0.70,
             "play_concept": "run_concept", "uncertainty": 0.4},
            {"formation": "Gun", "play": "Mesh", "probability": 0.55,
             "play_concept": "mesh", "uncertainty": 0.4},
        ]
        ranked, _audit = choose_model_play(clear, sit=sit, opponent_type="human", session_id="s", snap_seq=1)
        scores = {row["play"]: row["selection_score"] for row in ranked}
        self.assertGreater(scores["HB Dive"], scores["Mesh"])

    def test_repeated_runs_yield_when_a_pass_is_close(self) -> None:
        sit = Situation(raw="4th and 1", side="offense", down=4, distance=1)
        recent = [
            ("Gun", "HB Dive"), ("Gun", "Inside Zone"),
            ("Gun", "Power"), ("Gun", "Stretch"),
        ]
        options = [
            {"formation": "Gun", "play": "HB Dive", "probability": 0.55,
             "play_concept": "run_concept", "uncertainty": 0.5},
            {"formation": "Gun", "play": "Mesh", "probability": 0.55,
             "play_concept": "mesh", "uncertainty": 0.5},
            {"formation": "Gun", "play": "PA Boot", "probability": 0.54,
             "play_concept": "pa", "uncertainty": 0.5},
            {"formation": "Gun", "play": "RPO Read", "probability": 0.54,
             "play_concept": "rpo", "uncertainty": 0.5},
        ]
        ranked, audit = choose_model_play(
            options, recent_calls=recent, sit=sit, opponent_type="human",
            session_id="short", snap_seq=9,
        )
        scores = {row["play"]: row["selection_score"] for row in ranked}
        self.assertGreater(scores["Mesh"], scores["HB Dive"])
        self.assertFalse(audit.get("quota", False))
        self.assertEqual(audit["policy"], "model_primary_contextual_variety.v2")


class WildCardLoggingTests(unittest.TestCase):
    def test_interception_reconciliation_does_not_write_a_snap(self) -> None:
        before = []
        report = reconcile_interceptions(before)
        self.assertEqual(before, [])
        self.assertFalse(report["snap_created"])
        self.assertEqual(report["fourth_interception"]["status"], "unresolved")
        self.assertEqual(report["fourth_interception"]["attribution"], "unknown")
        self.assertIn("not a fourth throw", report["fourth_interception"]["why"])
        evidenced = {
            "side": "offense", "result": "interception",
            "turnover_cause": "read", "turnover_evidence": "threw the flat late",
        }
        self.assertEqual(turnover_attribution(evidenced), "read")
        self.assertEqual(turnover_attribution({"side": "offense", "result": "int"}), "unknown")
        counted = reconcile_interceptions([evidenced, {"side": "defense", "result_kind": "int"}])
        self.assertEqual(counted["offensive_interceptions_in_rows"], 1)
        self.assertEqual(counted["defensive_interceptions_forced_in_rows"], 1)
        self.assertEqual(counted["offensive_attributions"][0]["attribution"], "read")

    def test_final_score_gap_does_not_invent_a_snap(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = CoachDB(Path(tmp.name) / "madden.db", seed={"opponents": {}})
        self.addCleanup(db.close)
        before = db.conn.execute("SELECT COUNT(*) AS n FROM snaps").fetchone()["n"]
        with mock.patch("cfb_coach.session.last_logged_score", return_value=("42-42", "tie")):
            payload = record_final_score_gap(db, "session-1", "45-42")
        after = db.conn.execute("SELECT COUNT(*) AS n FROM snaps").fetchone()["n"]
        self.assertEqual(before, after)
        self.assertTrue(payload["gap"])
        self.assertFalse(payload["snap_created"])
        stored = json.loads(db.get_meta("final_score_gap.v1:session-1"))
        self.assertEqual(stored["last_snap_score"], "42-42")
        self.assertEqual(stored["session_score"], "45-42")

    def test_labeled_clock_is_read_and_a_down_is_not(self) -> None:
        self.assertEqual(clock_from_situation_text("3rd and 1 clock 1:22"), "1:22")
        self.assertIsNone(clock_from_situation_text("3rd and 1"))
        self.assertIsNone(clock_from_situation_text("3rd and 2 at the 40"))


if __name__ == "__main__":
    unittest.main()
