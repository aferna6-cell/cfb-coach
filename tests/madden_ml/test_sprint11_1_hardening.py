"""Sprint 11.1: shared pregame limit, situation audit, read-only postgame."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model.football_situation import audit_situation_inputs
from cfb_coach.madden.model.offense_postgame import (
    evaluate_saved_games,
    missing_database_report,
)
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.situation import parse_situation


def _seed() -> dict:
    return {"opponents": {"cpu": {
        "display_name": "CPU", "team_now": "DET", "skill": "cpu",
        "confidence": "low", "profile_json": "{}",
    }}}


def _wide_catalog() -> dict:
    formations = {
        f"Gun {index:02d}": ["Inside Zone", "Mesh", "Four Verticals"]
        for index in range(10)
    }
    formations["Goal Line"] = ["HB Dive", "QB Sneak"]
    return {"Buccaneers": formations}


class PregameWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed=_seed())

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_commands_share_the_formation_cap(self):
        from cfb_coach.cli import build_parser

        parser = build_parser()
        coordinator = parser.parse_args(["ml", "offense-coordinator", "-o", "cpu"])
        design = parser.parse_args(["ml", "offense-design", "-o", "cpu", "--stage", "--text"])
        self.assertEqual(designer.MAX_FORMATIONS, 15)
        self.assertEqual(designer.DEFAULT_MAX_FORMATIONS, 15)
        self.assertEqual(coordinator.max_formations, 15)
        self.assertEqual(design.max_formations, coordinator.max_formations)

    def test_preview_and_stage_share_one_fingerprint(self):
        from cfb_coach.madden.model import experimental_model

        art = experimental_model.ExperimentalArtifact(
            model_version="test.model", n_supervised=0, evidence_quality="prior_driven",
            global_rate=0.5,
        )
        catalog = _wide_catalog()
        from cfb_coach.madden.model.football_situation import SITUATION_NAMES

        def distinct_situation_value(_art, formation, _plays, _opponent, situation, _probe):
            situations = list(SITUATION_NAMES)
            if not formation.startswith("Gun"):
                return 0.2
            index = int(formation.split()[-1])
            return 0.9 if situation == situations[index % len(situations)] else 0.1

        with mock.patch(
            "cfb_coach.madden.model.offense_portfolio._situation_value",
            side_effect=distinct_situation_value,
        ):
            preview = designer.design_offense(
                self.db, opponent_id="cpu", max_formations=8, artifact=art, catalogue=catalog,
            )
            again = designer.design_offense(
                self.db, opponent_id="cpu", max_formations=8, artifact=art, catalogue=catalog,
            )
            smaller = designer.design_offense(
                self.db, opponent_id="cpu", max_formations=4, artifact=art, catalogue=catalog,
            )
        self.assertEqual(preview["proposal_id"], again["proposal_id"])
        self.assertEqual(preview["inventory_id"], again["inventory_id"])
        self.assertEqual(preview["max_formations"], 8)
        self.assertLessEqual(len(preview["book"]["formations"]), 8)
        self.assertNotEqual(preview["proposal_id"], smaller["proposal_id"])
        staged = designer.stage_design(self.db, preview)
        self.assertEqual(staged["proposal_id"], preview["proposal_id"])
        self.assertEqual(staged["inventory_id"], preview["inventory_id"])
        view = designer.plan_inspection(preview)
        self.assertEqual(view["inventory_fingerprint"], preview["inventory_id"])
        self.assertEqual(
            sum(len(row["plays"]) for row in view["formations"]),
            preview["n_plays"],
        )
        self.assertIn("current_coverage", view["pregame_diagnostic"]["missing"])
        self.assertNotIn("current_coverage", view["pregame_diagnostic"]["known"])
        self.assertTrue(view["confirm_after_physical_install"]["physical_installation_required"])
        self.assertEqual(designer.staged_design(self.db)["proposal_id"], preview["proposal_id"])


class SituationAuditTests(unittest.TestCase):
    def test_labeled_clock_and_timeouts_reach_the_audit(self):
        sit = parse_madden_situation(
            "3rd and 7 my 21 clock 1:24 timeouts 2 q4 score 21-14 showing cover 1"
        )
        audit = audit_situation_inputs(sit)
        self.assertEqual(sit.extras["clock_seconds"], 84)
        self.assertEqual(sit.extras["timeouts_us"], 2)
        self.assertEqual(sit.extras["quarter"], 4)
        self.assertIn("quarter", audit["known"])
        self.assertIn("clock_seconds", audit["known"])
        self.assertIn("timeouts_us", audit["known"])
        self.assertIn("score", audit["known"])
        self.assertIn("down", audit["known"])
        self.assertIn("distance", audit["known"])
        self.assertIn("yardline", audit["known"])
        self.assertIn("coverage", audit["known"])
        self.assertEqual(audit["fields"] and next(
            row["value"] for row in audit["fields"] if row["name"] == "coverage"
        ), "man")
        self.assertNotIn("coverage", audit["missing"])

    def test_previous_coverage_is_not_current(self):
        sit = parse_situation("2nd and 5 opp 40 cover 1")
        audit = audit_situation_inputs(sit)
        self.assertEqual(sit.coverage_source, "last")
        self.assertIn("coverage", audit["inferred"])
        self.assertNotIn("coverage", audit["known"])
        coverage = next(row for row in audit["fields"] if row["name"] == "coverage")
        self.assertIsNone(coverage["value"])

    def test_absent_clock_is_missing_not_carried(self):
        earlier = parse_situation("1st and 10 clock 2:00 timeouts 3")
        later = parse_situation("2nd and 6")
        self.assertEqual(earlier.extras["clock_seconds"], 120)
        self.assertNotIn("clock_seconds", later.extras)
        audit = audit_situation_inputs(later)
        self.assertIn("clock_seconds", audit["missing"])
        self.assertIn("timeouts_us", audit["missing"])
        self.assertIn("quarter", audit["missing"])
        self.assertIn("red_zone", audit["missing"])

    def test_down_and_distance_is_not_a_clock(self):
        sit = parse_situation("3rd and 8")
        self.assertIsNone(sit.extras.get("clock_seconds"))
        self.assertIsNone(sit.extras.get("timeouts_us"))


class PostgameReportTests(unittest.TestCase):
    def _decision(self, **extra) -> str:
        payload = {
            "experimental_offense": {
                "inventory_id": "inv-1",
                "inventory_confirmed": True,
                "probability": 0.62,
                "offense_action": {
                    "kind": "adjustment",
                    "id": "HOT-MAN",
                    "plays_considered": 40,
                    "plans_compared": 80,
                    "joint": {"football": {"score_phase": "protect_lead"}},
                },
                "pre_snap_action_context": {
                    "down": 3, "distance": 8, "quarter": 4,
                    "clock_seconds": 90, "two_minute": True,
                    "red_zone": True, "yardline": 85,
                },
                "input_audit": {
                    "fields": [
                        {"name": "red_zone", "status": "known", "value": True, "note": "yard line"},
                    ],
                },
            },
        }
        payload["experimental_offense"].update(extra)
        return json.dumps(payload)

    def test_recommendation_is_not_an_executed_adjustment(self):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "madden.db"
        db = CoachDB(path, seed=_seed())
        try:
            db.conn.execute(
                "INSERT INTO game_sessions (session_id, opponent_id, started_ts, play_count, score, result_wl) "
                "VALUES ('g-blowout', 'cpu', '2026-01-01T00:00:00Z', 2, '45-7', 'W')"
            )
            db.conn.execute(
                "INSERT INTO game_sessions (session_id, opponent_id, started_ts, play_count, score, result_wl) "
                "VALUES ('g-close', 'cpu', '2026-01-02T00:00:00Z', 1, '17-14', 'W')"
            )
            db.conn.execute(
                "INSERT INTO ml_decisions (decision_ts, game_id, snap_id, session_id, mode, "
                "final_formation, final_play, shadow_status, decision_json, latency_ms_model, created_ts) "
                "VALUES ('t', 'g-blowout', 'snap-1', 'g-blowout', 'experimental', "
                "'Gun Bunch', 'Mesh', 'ok', ?, 42.0, 't')",
                (self._decision(),),
            )
            db.conn.execute(
                "INSERT INTO ml_decisions (decision_ts, game_id, snap_id, session_id, mode, "
                "final_formation, final_play, shadow_status, decision_json, latency_ms_model, created_ts) "
                "VALUES ('t', 'g-close', 'snap-2', 'g-close', 'experimental', "
                "'Gun Trips', 'Flood', 'timeout', ?, NULL, 't')",
                (json.dumps({"experimental_offense": {"offense_action": {"kind": "none"}}}),),
            )
            db.conn.execute(
                "INSERT INTO ml_outcomes (snap_id, game_id, executed_status, executed_formation, "
                "executed_play, executed_verification, outcome_json, created_ts) "
                "VALUES ('snap-1', 'g-blowout', 'identified', 'Gun Bunch', 'Mesh', 'verified', ?, 't')",
                (json.dumps({
                    "result": "+12",
                    "offense_action_explicitly_confirmed": False,
                    "executed_adjustment_id": None,
                }),),
            )
            db.conn.execute(
                "INSERT INTO snaps (ts, opponent_id, side, session_id, ml_snap_id, down, distance, "
                "yardline, quarter, formation, play, result, coverage_seen) "
                "VALUES ('t', 'cpu', 'offense', 'g-blowout', 'snap-1', 3, 8, 85, 4, "
                "'Gun Bunch', 'Mesh', '+12', 'Cover 1')"
            )
            db.conn.commit()
        finally:
            db.close()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        report_db = CoachDB.open_read_only(path)
        try:
            report = evaluate_saved_games(report_db, limit=2)
        finally:
            report_db.close()
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertFalse(report["history_modified"])
        self.assertEqual(report["cpu_blowout_wins"], ["g-blowout"])
        self.assertNotIn("g-close", report["cpu_blowout_wins"])
        blowout = next(game for game in report["games"] if game["game_id"] == "g-blowout")
        self.assertEqual(blowout["final_score"]["us"], 45)
        self.assertEqual(blowout["final_score"]["them"], 7)
        self.assertEqual(blowout["recommended_offensive_plays"], 1)
        self.assertEqual(blowout["verified_executed_offensive_plays"], 1)
        self.assertEqual(blowout["adjustments"]["recommended"], 1)
        self.assertEqual(blowout["adjustments"]["verified_applied"], 0)
        self.assertEqual(blowout["third_down"]["verified_conversions"], 1)
        self.assertEqual(blowout["red_zone"]["decisions"], 1)
        self.assertEqual(blowout["clock_management"]["decisions"], 1)
        self.assertGreaterEqual(blowout["formation_diversity"]["verified_executed"]["distinct"], 1)
        self.assertEqual(blowout["fallback_count"], 0)
        self.assertTrue(blowout["inventory"]["consistent"])
        self.assertEqual(blowout["prediction_vs_outcome"]["comparable_snaps"], 1)
        close = next(game for game in report["games"] if game["game_id"] == "g-close")
        self.assertEqual(close["fallback_count"], 1)
        self.assertEqual(close["recommended_offensive_plays"], 0)
        self.assertEqual(close["final_score"]["us"], 17)
        self.assertIsNone(close["stored_result_label"])
        tmp.cleanup()

    def test_missing_database_is_reported_without_a_guess(self):
        report = missing_database_report("/tmp/does-not-exist-madden27.db")
        self.assertFalse(report["database"]["exists"])
        self.assertEqual(report["user_cpu_blowouts"], "not_found_in_this_environment")
        self.assertEqual(report["games"], [])
        self.assertFalse(report["history_modified"])


if __name__ == "__main__":
    unittest.main()
