"""Defensive ML parity: audit, shrinkage, live gates, logging, eval."""
from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden.model import defense_coordinator as defense
from cfb_coach.madden.model.cli import cmd_ml_defense_experimental
from cfb_coach.madden.model.defense_audit import (
    audit_defensive_rows,
    reconcile_interceptions,
)
from cfb_coach.madden.model.defense_eval import compare_defense
from cfb_coach.madden.model.defense_shrinkage import defensive_utility
from cfb_coach.madden.model.schema import MLStatus, PolicySource
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.situation import parse_madden_situation

BOOK = {
    "Nickel Over": ["Cover 3 Sky", "Tampa 2", "Cover 1 Robber", "Mid Blitz 0"],
    "4-3 Over": ["Cover 3 Sky", "Cover 1 Hole", "Mike Blitz"],
    "Dime Normal": ["Cover 4 Drop", "Cover 2 Sink"],
}


def _row(**kwargs):
    base = {
        "supervised_eligible": True,
        "side": "defense",
        "executed_play": "Cover 3 Sky",
        "success": "true",
        "executed_verification": "verified",
        "executed_status": "identified",
        "label_available": True,
        "opponent_type": "human",
        "opponent_id": "gavin",
        "game_id": "g1",
        "down": 1,
        "distance": 10,
        "yards": 2,
        "result": "+2",
    }
    base.update(kwargs)
    return base


def _sit(**kwargs):
    fields = {
        "down": 1, "distance": 10, "yardline": 40,
        "red_zone": False, "goal_line": False, "two_minute": False,
        "concept_hint": None, "concept_source": "none", "extras": {},
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


class AuditTests(unittest.TestCase):
    def test_candidates_are_not_automatically_supervised(self):
        rows = [
            _row(),
            _row(supervised_eligible=False, game_id="g1", executed_play="Tampa 2"),
            _row(opponent_type="cpu", game_id="g2", executed_play="Mike Blitz"),
            _row(executed_verification="unknown", game_id="g1"),
            _row(side="offense", executed_play="Inside Zone", game_id="g3"),
            _row(game_id="g4", opponent_id="john", success="true", yards=1),
        ]
        audit = audit_defensive_rows(rows)
        self.assertEqual(audit["candidate_defensive_snaps"], 5)
        self.assertEqual(audit["supervised_play_specific"], 2)
        self.assertIn("cpu_or_unknown_opponent", audit["excluded"])
        self.assertIn("unverified_execution", audit["excluded"])
        self.assertEqual(audit["by_game"]["g1"]["supervised"], 1)
        self.assertEqual(audit["john_supervised"], 1)
        self.assertEqual(audit["non_john_supervised"], 1)
        self.assertTrue(audit["reported_178_is_not_a_training_count"])

    def test_wildcard_int_gap_is_not_invented(self):
        report = reconcile_interceptions()
        self.assertEqual(report["logged_offensive_int"], 3)
        self.assertEqual(report["user_reported_offensive_int"], 4)
        self.assertEqual(report["gap"], 1)
        self.assertFalse(report["invented_snap"])
        self.assertEqual(report["attribution"], "unknown")


class ShrinkageTests(unittest.TestCase):
    def test_forty_yard_td_hurts_more_than_a_first_down(self):
        td = defensive_utility(_row(success="false", result="td +40", yards=40, kind="td"))
        conv = defensive_utility(_row(success="false", result="+4", yards=4, distance=4))
        self.assertIsNotNone(td)
        self.assertIsNotNone(conv)
        self.assertLess(td, conv)
        self.assertLess(td, -4.0)

    def test_train_writes_independent_shrinkage_artifact(self):
        model = defense.train_defense([
            _row(),
            _row(game_id="g2", executed_play="Mid Blitz 0", success="false",
                 result="td +42", yards=42, opponent_id="john"),
        ])
        self.assertEqual(model["schema"], defense.SCHEMA)
        self.assertEqual(model["supervised_defensive_snaps"], 2)
        self.assertEqual(model["supervised_by_game"]["g1"], 1)
        self.assertEqual(model["supervised_by_game"]["g2"], 1)
        self.assertEqual(model["model_kind"], "hierarchical_shrinkage.v1")
        self.assertTrue(model["independent_of_offense_artifact"])
        self.assertTrue(model["post_snap_not_used_as_live"])
        self.assertIn("Cover 3 Sky", model["plays"])
        self.assertTrue(model["family_utility"])
        self.assertEqual(model["training_audit"]["john_supervised"], 1)

    def test_trained_evidence_changes_rank_versus_prior(self):
        prior = defense.rank_defense(_sit(down=3, distance=8), BOOK, None)
        trained = defense.train_defense([
            _row(executed_play="Mid Blitz 0", success="true", yards=-6, result="sack",
                 down=3, distance=8, game_id="g-press"),
            _row(executed_play="Mid Blitz 0", success="true", yards=-4, result="sack",
                 down=3, distance=8, game_id="g-press-2", opponent_id="pat"),
        ])
        learned = defense.rank_defense(_sit(down=3, distance=8), BOOK, trained)
        self.assertEqual(len(learned), sum(len(v) for v in BOOK.values()))
        self.assertTrue(any(row.get("utility", {}).get("n_play", 0) > 0 for row in learned))
        prior_blitz = next(row for row in prior if row["play"] == "Mid Blitz 0")
        learned_blitz = next(row for row in learned if row["play"] == "Mid Blitz 0")
        self.assertGreater(learned_blitz["score"], prior_blitz["score"])
        self.assertGreater(learned_blitz["verified_play_n"], 0)


class EnableAndLiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed={"opponents": {}})
        from cfb_coach.madden import playbook
        playbook._save_state(self.db, {"applied": {
            "defense": {"side": "defense", "mode": "stock", "name": "49ers",
                        "source_book": "49ers", "rev": 1, "locked_ts": "test",
                        "formations": BOOK}
        }, "pending": {}})

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_enable_refuses_without_verified_training_and_does_not_auto_train(self):
        args = Namespace(
            enable=True, disable=False, retrain=False,
            db=str(Path(self.tmp.name) / "madden.db"), out=None,
            cli_parser=None,
        )
        with mock.patch.object(defense, "train_defense", side_effect=AssertionError("auto-train")):
            from cfb_coach.madden.model import cli as cli_mod
            with mock.patch.object(cli_mod, "open_explicit_madden_db", return_value=self.db):
                with mock.patch.object(cli_mod, "_require_db", return_value=Path(args.db)):
                    with mock.patch.object(self.db, "close"):
                        code = cmd_ml_defense_experimental(args)
        self.assertEqual(code, 2)
        self.assertEqual(defense.mode(self.db), "off")

    def test_missing_artifact_timeout_illegal_and_unarmed_macro_fall_back(self):
        defense.set_mode(self.db, "experimental")
        heuristic = MaddenCall(
            "defense", "Nickel Over", "Tampa 2", "No adj", "User hook", "h",
        )
        sit = _sit(extras={"session_id": "s", "snap_seq": 1})
        missing = defense.maybe_apply_defense(heuristic, sit, "gavin", self.db, BOOK)
        self.assertIs(missing, heuristic)
        self.assertIs(missing._pending_ml_decision.fallback_reason, MLStatus.MODEL_MISSING)
        self.assertIn("fallback_source=select_defense", missing._pending_ml_decision.ml_status_detail)

        art = defense.train_defense([_row()])
        dest = Path(self.tmp.name) / "defense.json"
        defense.save_model(art, dest)
        self.db.set_meta(defense.MODEL_META, str(dest))

        with mock.patch.object(defense, "live_pick", side_effect=TimeoutError("budget")):
            timed = defense.maybe_apply_defense(heuristic, sit, "gavin", self.db, BOOK)
        self.assertIs(timed._pending_ml_decision.fallback_reason, MLStatus.TIMEOUT)
        self.assertEqual((timed.formation, timed.play), (heuristic.formation, heuristic.play))

        fake = {
            "selected": {"formation": "Nickel Over", "play": "NOT A CALL", "family": "man", "score": 1},
            "top": [{"formation": "Nickel Over", "play": "NOT A CALL", "family": "man",
                     "score": 1, "football_intelligence": {}}],
            "tendencies": {}, "model_quality": "limited", "n_candidates": 1,
        }
        with mock.patch.object(defense, "live_pick", return_value=fake):
            illegal = defense.maybe_apply_defense(heuristic, sit, "gavin", self.db, BOOK)
        self.assertIs(illegal._pending_ml_decision.fallback_reason, MLStatus.INVALID_OUTPUT)

        armed = MaddenCall(
            "defense", "Nickel Over", "Cover 3 Sky", "DRAFT-SHEET", "User hook", "h",
            macro="DRAFT-SHEET",
        )
        kept = defense.maybe_apply_defense(armed, sit, "gavin", self.db, BOOK)
        self.assertIs(kept, armed)
        self.assertEqual(kept.macro, "DRAFT-SHEET")
        self.assertIs(kept._pending_ml_decision.fallback_reason, MLStatus.CIRCUIT_OPEN)

        live = defense.maybe_apply_defense(heuristic, sit, "gavin", self.db, BOOK)
        self.assertIn(live.play, BOOK[live.formation])
        self.assertIs(live._pending_ml_decision.policy_source, PolicySource.MODEL)
        self.assertLessEqual(float(live._pending_ml_decision.latency_ms or 0), defense.LATENCY_MS)
        self.assertNotIn("offense.json", str(self.db.get_meta(defense.MODEL_META)))

    def test_compare_refuses_promotion_on_tiny_sample(self):
        art = defense.train_defense([_row()])
        report = compare_defense([_row()], BOOK, art, opponent_id="gavin")
        self.assertFalse(report["promote_ok"])
        self.assertFalse(report["causal_claims"])
        self.assertIn("Observational", report["limitations"][0])
        self.assertEqual(report["supervised_defensive_snaps"], 1)


class ClockAndOutcomeTests(unittest.TestCase):
    def test_clock_carries_forward_and_int_stays_unattributed(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = CoachDB(Path(tmp.name) / "madden.db", seed={"opponents": {
            "cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu", "confidence": "low"}
        }})
        self.addCleanup(db.close)

        class _Call:
            side = "offense"
            formation = "Gun Bunch"
            play = "Mesh"
            macro = None
            adjustment = None
            def format(self):
                return "Gun Bunch — Mesh"

        ctrl = LivePlayController(
            db=db, opponent_id="cpu", make_call=lambda sit, **_k: _Call(),
            parse_situation=parse_madden_situation, learn_summary=lambda: "",
            brand="Madden 27 Franchise", enable_execution_verify=True,
        )
        ctrl.start()
        first = ctrl.call_only("1&10 my 25", game_clock="8:00", quarter=2)
        self.assertTrue(first["ok"])
        second = ctrl.result_and_call(
            outcome="INT", sit_raw="d 1&10", quarter=2, verify_touched=True,
            executed_status="identified",
        )
        self.assertTrue(second["ok"], second)
        outcome = db.conn.execute(
            "SELECT outcome_json, game_clock FROM ml_outcomes ORDER BY id"
        ).fetchall()
        self.assertGreaterEqual(len(outcome), 1)
        payload = json.loads(outcome[0]["outcome_json"] or "{}")
        self.assertEqual(payload.get("kind"), "int")
        self.assertEqual(payload.get("attribution"), "unknown")
        clocks = [row["game_clock"] for row in db.conn.execute("SELECT game_clock FROM snaps")]
        self.assertTrue(any(clock == "8:00" for clock in clocks))
        ended = ctrl.end_game(result_wl="loss", score="45-42")
        self.assertTrue(ended["ok"])
        session = db.conn.execute(
            "SELECT score FROM game_sessions WHERE session_id=?", (ctrl.session_id,)
        ).fetchone()
        self.assertEqual(session["score"], "45-42")
