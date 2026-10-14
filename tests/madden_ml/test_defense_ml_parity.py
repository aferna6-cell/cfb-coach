"""Defensive ML parity: audit, explosive objective, live gates, short yardage.

Synthetic fixtures only. No franchise database and no real game rows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import defense_coordinator as defense
from cfb_coach.madden.model.defense_audit import audit_defensive_training
from cfb_coach.madden.model.football_situation import selection_adjustment
from cfb_coach.madden.model.offense_selection_policy import choose_model_play
from cfb_coach.madden.model.wildcard_postgame import audit_wildcard_session
from cfb_coach.madden.model.dual_control import install_primary_lions_books
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.situation import parse_madden_situation

_REPO = Path(__file__).resolve().parents[2]


def _env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["PYTHONPATH"] = str(_REPO)
    env.pop("CFB_COACH_DB", None)
    env.pop("CFB_COACH_MADDEN_DB", None)
    return env


def _run(argv: list[str], cwd: Path, home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cfb_coach", *argv],
        cwd=cwd,
        env=_env(home),
        capture_output=True,
        text=True,
        check=False,
    )


def _verified(game: str, play: str, success: bool, **extra: object) -> dict:
    row = {
        "side": "defense",
        "game_id": game,
        "match_id": game,
        "down": 1,
        "distance": 10,
        "executed_play": play,
        "success": success,
        "executed_verification": "verified",
        "executed_status": "identified",
        "accepted": "accepted",
        "opponent_id": "synth",
        "opponent_type": "human",
        "label_available": True,
    }
    row.update(extra)
    return row


class DefenseAuditTests(unittest.TestCase):
    def test_audit_counts_by_game_and_excludes_uncertain(self) -> None:
        rows = [
            _verified("g1", "Cover 3 Sky", True, yards=3),
            _verified("g1", "Cover 1 Contain", False, yards=22),
            _verified("g2", "Cover 3 Sky", True, yards=0),
            {
                "side": "defense",
                "game_id": "g3",
                "executed_play": "Cover 3 Sky",
                "success": True,
                "accepted": "unknown",
            },
            {
                "side": "defense",
                "game_id": "g3",
                "recommendation_only": True,
                "executed_play": "Cover 2 Man",
            },
        ]
        report = audit_defensive_training(rows)
        self.assertEqual(report["candidate_defensive_rows"], 5)
        self.assertEqual(report["verified_by_source"]["total"], 3)
        self.assertEqual(report["play_specific_supervised"], 3)
        by_game = {row["game_id"]: row["verified_total"] for row in report["verified_by_game"]}
        self.assertEqual(by_game, {"g1": 2, "g2": 1})
        self.assertTrue(
            "execution_unrecorded" in report["exclusion_counts"]
            or "accepted=unknown" in report["exclusion_counts"]
        )
        self.assertIn("recommendation_only", report["exclusion_counts"])
        self.assertFalse(report["auto_promote"])


class DefenseObjectiveTests(unittest.TestCase):
    def test_train_stores_explosive_and_td_tables_by_game(self) -> None:
        rows = [
            _verified("g1", "Cover 3 Sky", True, yards=2),
            _verified("g1", "Cover 3 Sky", False, yards=41, result="td +41"),
            _verified("g2", "Cover 1 Contain", True, yards=1),
            _verified("g2", "Cover 1 Contain", False, yards=18),
        ]
        art = defense.train_defense(rows, preverified=True)
        self.assertEqual(art["supervised_defensive_snaps"], 4)
        self.assertEqual(art["supervised_by_game"], {"g1": 2, "g2": 2})
        self.assertEqual(art["objective"], "situational_defense.v1")
        self.assertIn("Cover 3 Sky", art["touchdown_risk"])
        self.assertTrue(art["independent_of_offense_artifact"])
        self.assertFalse(art["auto_promote"])

    def test_score_penalizes_explosive_and_td_risk_more_than_routine_gain(self) -> None:
        from cfb_coach.madden.situation import Situation

        rows = []
        for i in range(8):
            rows.append(_verified("g1", "Cover 3 Sky", False, yards=45, result="td +45"))
            rows.append(_verified("g2", "Cover 1 Contain", True, yards=2 + (i % 3)))
        art = defense.train_defense(rows, preverified=True)
        sit = Situation(raw="d 1&10", side="defense", down=1, distance=10, yardline=40)
        book = {"4-3 Over": ["Cover 3 Sky", "Cover 1 Contain"]}
        ranked = defense.rank_defense(sit, book, art)
        by_play = {row["play"]: row for row in ranked}
        self.assertGreater(
            by_play["Cover 1 Contain"]["score"],
            by_play["Cover 3 Sky"]["score"],
        )
        self.assertGreater(
            by_play["Cover 3 Sky"]["touchdown_risk"],
            by_play["Cover 1 Contain"]["touchdown_risk"],
        )
        self.assertGreater(
            by_play["Cover 3 Sky"]["explosive_risk"],
            by_play["Cover 1 Contain"]["explosive_risk"],
        )


class ShortYardageOffenseTests(unittest.TestCase):
    def test_short_yardage_pass_pa_rpo_compete_with_run(self) -> None:
        run_delta, _ = selection_adjustment("Inside Zone", type("S", (), {"down": 3, "distance": 1})())
        pass_delta, _ = selection_adjustment("Stick", type("S", (), {"down": 3, "distance": 1})())
        pa_delta, _ = selection_adjustment("PA Drag Wheel", type("S", (), {"down": 3, "distance": 1})())
        rpo_delta, _ = selection_adjustment("RPO Peek Glance", type("S", (), {"down": 3, "distance": 1})())
        deep_delta, _ = selection_adjustment("Four Verticals", type("S", (), {"down": 3, "distance": 1})())
        self.assertGreater(run_delta, 0)
        self.assertGreater(pass_delta, 0)
        self.assertGreater(pa_delta, 0)
        self.assertGreater(rpo_delta, 0)
        self.assertLess(deep_delta, 0)
        # Pass/PA stay within a conversion band of the run; no artificial quota.
        self.assertLess(abs(run_delta - pass_delta), 0.01)
        self.assertLess(abs(run_delta - pa_delta), 0.01)

    def test_short_yardage_run_exposure_lets_pass_win_near_tie(self) -> None:
        sit = type("S", (), {"down": 3, "distance": 1, "short_yardage": True})()
        opts = [
            {
                "formation": "Gun Doubles",
                "play": "Inside Zone",
                "probability": 0.51,
                "play_concept": "run_concept",
                "uncertainty": 0.7,
                "evidence_quality": "prior_driven",
            },
            {
                "formation": "Gun Doubles",
                "play": "Stick",
                "probability": 0.505,
                "play_concept": "stick",
                "uncertainty": 0.7,
                "evidence_quality": "prior_driven",
            },
        ]
        recent = [
            ("Gun Doubles", "Inside Zone"),
            ("Gun Doubles", "HB Dive"),
            ("I Form", "Counter Y"),
            ("Gun Doubles", "HB Mid Zone"),
        ]
        ranked, _audit = choose_model_play(
            opts, sit=sit, recent_calls=recent, session_id="short", snap_seq=9,
            opponent_type="human",
        )
        run = next(row for row in ranked if row["play"] == "Inside Zone")
        throw = next(row for row in ranked if row["play"] == "Stick")
        self.assertGreater(run["selection_penalty"], 0)
        self.assertGreaterEqual(throw["selection_score"], run["selection_score"])


class WildcardPostgameTests(unittest.TestCase):
    def test_reconciles_fourth_int_from_orphan_without_inventing_snap(self) -> None:
        snaps = [
            {"side": "offense", "snap_id": 1, "result": "INT", "score_for": 14, "score_against": 21},
            {"side": "offense", "snap_id": 2, "result": "pick", "score_for": 21, "score_against": 28},
            {"side": "offense", "snap_id": 3, "result": "interception", "score_for": 28, "score_against": 35},
            {"side": "offense", "snap_id": 4, "result": "+7", "score_for": 42, "score_against": 42,
             "drive_id": None, "game_clock": None, "accepted": "unknown"},
        ]
        decisions = [
            {
                "side": "offense",
                "id": 99,
                "result": "INT",
                "orphan_reason": "superseded",
                "snap_link": "no_snap_recorded",
            },
        ]
        report = audit_wildcard_session(
            snaps=snaps,
            decisions=decisions,
            reported_final={"for": 42, "against": 45},
            reported_offense_ints=3,
            user_reported_ints=4,
            armed_defense_macros=["SAFE DEEP"],
        )
        self.assertEqual(report["interceptions"]["offense_snap_interceptions"], 3)
        self.assertEqual(report["interceptions"]["orphan_int_candidates"], 1)
        self.assertTrue(
            report["interceptions"]["discrepancy"]["reconcilable_without_invention"]
        )
        self.assertFalse(report["invented_snaps"])
        self.assertEqual(report["completeness"]["missing"]["drive_id"], 4)
        for event in report["interceptions"]["events"]:
            self.assertEqual(event["attribution"], "unknown")

    def test_flags_unarmed_macro_shown_on_defense(self) -> None:
        report = audit_wildcard_session(
            snaps=[{
                "side": "defense",
                "snap_id": 1,
                "recommended_adjustment_id": "QTRS OVERTOP",
                "score_for": 0,
                "score_against": 0,
            }],
            armed_defense_macros=["SAFE DEEP"],
        )
        self.assertEqual(len(report["macros"]["unarmed_shown"]), 1)
        self.assertEqual(report["macros"]["unarmed_shown"][0]["macro"], "QTRS OVERTOP")


class DefenseLiveIntegrationTests(unittest.TestCase):
    def test_enable_refuses_without_passing_eval_and_live_stays_heuristic(self) -> None:
        import random

        from cfb_coach.madden import data as mdata

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cwd = home / "cwd"
            cwd.mkdir()
            db_path = home / "madden27.db"
            art_path = home / "defense.json"
            db = CoachDB(db_path, seed=mdata.load_seed())
            try:
                books = install_primary_lions_books(db)
            finally:
                db.close()
            rows_path = home / "rows.jsonl"
            rows = [
                _verified("g1", "Cover 3 Sky", True, yards=2),
                _verified("g1", "Cover 1 Contain", False, yards=9),
                _verified("g2", "Cover 3 Sky", True, yards=1),
                _verified("g2", "LB Sting I", False, yards=12),
            ]
            rows_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
            )
            trained = _run(
                ["ml", "train-defense", "--path", str(rows_path), "--out", str(art_path), "--seed", "7"],
                cwd, home,
            )
            self.assertEqual(trained.returncode, 0, trained.stderr)
            self.assertIn("auto_promote: false", trained.stdout)
            art = json.loads(art_path.read_text(encoding="utf-8"))
            self.assertGreater(art["supervised_defensive_snaps"], 0)
            self.assertIn("supervised_by_game", art)
            self.assertTrue(art.get("independent_of_offense_artifact"))
            # Small synthetic N will not clear the enable gate.
            db = CoachDB(db_path, seed=mdata.load_seed())
            try:
                db.set_meta(defense.MODEL_META, str(art_path))
            finally:
                db.close()
            enable = _run(
                ["ml", "defense-experimental", "--enable", "--db", str(db_path)],
                cwd, home,
            )
            self.assertNotEqual(enable.returncode, 0)
            self.assertIn("Cannot enable", enable.stderr)
            # Live path with mode off / missing gate stays the established caller.
            db = CoachDB(db_path, seed=mdata.load_seed())
            try:
                from cfb_coach.madden.playbook import load_books

                defense.set_mode(db, "off")
                loaded = load_books(db)
                playbook = {
                    "offense": loaded["offense"]["formations"],
                    "defense": loaded["defense"]["formations"],
                }
                sit = parse_madden_situation("d 1&10 my 25")
                call = make_call(
                    sit, "john", db,
                    rng=random.Random(3),
                    playbook=playbook,
                    active_macros={"offense": [], "defense": []},
                    allow_human_defense_ml=True,
                    live_macros=True,
                )
                text = call.format() if hasattr(call, "format") else str(call)
                self.assertNotIn("LABEL: EXPERIMENTAL", text)
                self.assertIn("LABEL: HEURISTIC", text)
            finally:
                db.close()

    def test_audit_defense_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cwd = home / "cwd"
            cwd.mkdir()
            path = home / "rows.jsonl"
            path.write_text(
                json.dumps(_verified("g1", "Cover 3 Sky", True, yards=1)) + "\n"
                + json.dumps({
                    "side": "defense", "game_id": "g9", "accepted": "unknown",
                    "executed_play": "Cover 2",
                }) + "\n",
                encoding="utf-8",
            )
            proc = _run(["ml", "audit-defense", "--path", str(path)], cwd, home)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("verified_live/vod/total: 1/0/1", proc.stdout)
            self.assertIn("auto_promote: false", proc.stdout)


if __name__ == "__main__":
    unittest.main()
