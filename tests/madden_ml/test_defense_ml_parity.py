"""Defensive ML parity: verified training, independent ranking, live gates."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import data as mdata
from cfb_coach.madden import playbook
from cfb_coach.madden.human_opponents import ensure_madden_opponent
from cfb_coach.madden.macros import store_selection
from cfb_coach.madden.model import defense_coordinator as defense
from cfb_coach.madden.model.defense_audit import audit_defensive_rows
from cfb_coach.madden.model.defense_eval import evaluate_defense
from cfb_coach.madden.model.schema import MLStatus
from cfb_coach.madden.playcaller import MaddenCall, make_call
from cfb_coach.madden.situation import parse_madden_situation

BOOK = {
    "4-3 Over": ["Cover 3 Sky", "Cover 1 Contain", "OLB Fire Man", "Zero Blitz"],
    "Nickel 3-3 Over": ["Cover 3 Match", "Bracket Switch Willie", "Mid Blitz 0"],
    "Dime 2-3": ["Cover 4 Drop", "Cover 3 Sky", "Cover 2 Man"],
}


def _row(*, game: str, play: str, success: bool, yards: int, down: int = 3,
         distance: int = 8, opponent: str = "gavin", **extra) -> dict:
    payload = {
        "side": "defense",
        "opponent_type": "human",
        "opponent_id": opponent,
        "game_id": game,
        "supervised_eligible": True,
        "executed_verification": "verified",
        "executed_status": "identified",
        "executed_play": play,
        "executed_formation": "4-3 Over" if play != "Cover 4 Drop" else "Dime 2-3",
        "label_available": True,
        "success": "true" if success else "false",
        "down": down,
        "distance": distance,
        "yards": yards,
        "red_zone": False,
        "goal_line": False,
        "two_minute": False,
        "result": "td" if (not success and yards >= 40) else ("stop" if success else f"gain {yards}"),
    }
    payload.update(extra)
    return payload


def _verified_corpus() -> list[dict]:
    rows = []
    for i in range(8):
        rows.append(_row(game="game-a", play="Cover 3 Sky", success=True, yards=2, down=3, distance=8))
        rows.append(_row(game="game-a", play="Zero Blitz", success=False, yards=42, down=3, distance=8,
                         result="td", touchdown=True))
    for i in range(6):
        rows.append(_row(game="game-b", play="Cover 3 Sky", success=True, yards=1, down=3, distance=7))
        rows.append(_row(game="game-b", play="Zero Blitz", success=False, yards=18, down=1, distance=10))
    rows.append(_row(
        game="game-a", play="Cover 3 Sky", success=True, yards=0, down=3, distance=8,
        executed_verification="unknown", supervised_eligible=False, label_available=True,
    ))
    rows.append({
        "side": "defense", "opponent_type": "cpu", "opponent_id": "cpu", "game_id": "cpu-1",
        "supervised_eligible": True, "executed_verification": "verified",
        "executed_status": "identified", "executed_play": "Cover 3 Sky",
        "label_available": True, "success": "false", "down": 3, "distance": 8, "yards": 40,
    })
    return rows


class DefenseAuditTests(unittest.TestCase):
    def test_candidates_are_not_automatically_supervised(self):
        report = audit_defensive_rows(_verified_corpus())
        self.assertEqual(report["supervised"]["play_specific_training"], 28)
        self.assertGreater(report["candidates"]["in_this_source"], 28)
        self.assertIn("execution_unverified", report["exclusions"])
        self.assertIn("not_human", report["exclusions"])
        by_game = {g["game_id"]: g["verified_supervised"] for g in report["by_game"]}
        self.assertEqual(by_game["game-a"], 16)
        self.assertEqual(by_game["game-b"], 12)
        self.assertEqual(report["candidates"]["reported_elsewhere"], 178)
        self.assertIsNotNone(report["candidates"]["reported_gap_explanation"])

    def test_int_reconciliation_does_not_invent_a_snap(self):
        rows = [
            {"side": "offense", "result": "INT", "game_id": "wc", "play": "Stick"},
            {"side": "offense", "result": "INT return 15", "game_id": "wc", "play": "Mesh"},
            {"side": "offense", "result": "interception", "game_id": "wc", "play": "Flood Seam"},
        ]
        report = audit_defensive_rows(rows)
        self.assertEqual(report["interceptions"]["logged_offensive_ints"], 3)
        self.assertEqual(report["interceptions"]["user_reported_ints"], 4)
        self.assertEqual(report["interceptions"]["unlogged_user_gap"], 1)
        self.assertTrue(all(item["attribution"] == "unknown" for item in report["interceptions"]["attribution"]))
        self.assertNotIn("invented", report["interceptions"]["reconciliation"].lower())


class DefenseModelTests(unittest.TestCase):
    def test_train_ignores_unverified_and_versions_independently(self):
        art = defense.train_defense(_verified_corpus())
        self.assertEqual(art["supervised_defensive_snaps"], 28)
        self.assertEqual(art["side"], "defense")
        self.assertTrue(art["independent_of_offense"])
        self.assertTrue(str(art["model_version"]).startswith("madden-ml.defense.coordinator."))
        self.assertNotIn("offense", art["model_version"])
        self.assertGreater(art["plays"]["Cover 3 Sky"][0], art["plays"]["Zero Blitz"][0])
        self.assertEqual(art["training_audit"]["verified_defensive_snaps"], 28)

    def test_model_uses_trained_evidence_and_differs_from_mix_prior(self):
        art = defense.train_defense(_verified_corpus())
        sit = SimpleNamespace(
            down=3, distance=8, yardline=40, red_zone=False, goal_line=False,
            two_minute=False, concept_hint=None, concept_source="none", extras={},
        )
        ranked = defense.rank_defense(sit, BOOK, art, opponent_id="gavin")
        self.assertEqual(len(ranked), sum(len(v) for v in BOOK.values()))
        sky = next(r for r in ranked if r["play"] == "Cover 3 Sky")
        blitz = next(r for r in ranked if r["play"] == "Zero Blitz")
        self.assertGreater(sky["verified_play_n"], 0)
        self.assertGreater(blitz["verified_play_n"], 0)
        self.assertGreater(sky["score"], blitz["score"])
        self.assertEqual(ranked[0]["play"], "Cover 3 Sky")
        # Mix prior on long downs leans two-high, not Cover 3. The pick is evidence.
        self.assertEqual(sky["family"], "single_high")

    def test_house_touchdown_weighs_more_than_a_first_down(self):
        rows = []
        for _ in range(8):
            rows.append(_row(
                game="g1", play="Cover 3 Sky", success=False, yards=4,
                down=3, distance=4, result="gain 4",
            ))
            rows.append(_row(
                game="g1", play="Zero Blitz", success=False, yards=42,
                down=3, distance=4, result="td", touchdown=True,
            ))
        art = defense.train_defense(rows)
        sit = SimpleNamespace(
            down=3, distance=4, yardline=40, red_zone=False, goal_line=False,
            two_minute=False, concept_hint=None, concept_source="none", extras={},
        )
        ranked = defense.rank_defense(sit, BOOK, art, opponent_id="gavin")
        sky = next(r for r in ranked if r["play"] == "Cover 3 Sky")
        blitz = next(r for r in ranked if r["play"] == "Zero Blitz")
        self.assertGreater(sky["score"], blitz["score"])
        self.assertGreater(blitz["verified_play_n"], 0)

    def test_john_specific_rates_do_not_replace_general_knowledge(self):
        rows = _verified_corpus()
        for i in range(3):
            rows.append(_row(
                game="john-1", play="Zero Blitz", success=True, yards=0,
                opponent="john", down=3, distance=8,
            ))
        art = defense.train_defense(rows)
        general = art["plays"]["Zero Blitz"]
        john = art["opponent_specific"]["plays"]["john"]["Zero Blitz"]
        self.assertGreater(general[1], john[1])
        sit = SimpleNamespace(
            down=3, distance=8, yardline=40, red_zone=False, goal_line=False,
            two_minute=False, concept_hint=None, concept_source="none", extras={},
        )
        vs_gavin = defense.rank_defense(sit, BOOK, art, opponent_id="gavin")
        vs_john = defense.rank_defense(sit, BOOK, art, opponent_id="john")
        self.assertEqual(vs_gavin[0]["play"], "Cover 3 Sky")
        gavin_blitz = next(r for r in vs_gavin if r["play"] == "Zero Blitz")
        john_blitz = next(r for r in vs_john if r["play"] == "Zero Blitz")
        self.assertGreaterEqual(john_blitz["score"], gavin_blitz["score"] - 1e-9)

    def test_post_snap_concept_is_not_a_live_feature(self):
        art = defense.train_defense(_verified_corpus())
        last = defense.rank_defense(
            SimpleNamespace(
                down=3, distance=8, yardline=40, red_zone=False, goal_line=False,
                two_minute=False, concept_hint="Four Verticals", concept_source="last", extras={},
            ),
            BOOK, art,
        )
        none = defense.rank_defense(
            SimpleNamespace(
                down=3, distance=8, yardline=40, red_zone=False, goal_line=False,
                two_minute=False, concept_hint=None, concept_source="none", extras={},
            ),
            BOOK, art,
        )
        self.assertEqual(
            [(r["formation"], r["play"], r["score"]) for r in last],
            [(r["formation"], r["play"], r["score"]) for r in none],
        )

    def test_promotion_refuses_prior_driven_and_missing_artifacts(self):
        ok, reason = defense.experimental_promotion_allowed(None)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_defense_artifact")
        thin = defense.train_defense([_row(game="g1", play="Cover 3 Sky", success=True, yards=1)])
        ok, reason = defense.experimental_promotion_allowed(thin)
        self.assertFalse(ok)
        self.assertIn("insufficient_verified_training", reason)
        fat = defense.train_defense(_verified_corpus())
        ok, reason = defense.experimental_promotion_allowed(fat)
        self.assertTrue(ok)

    def test_eval_is_observational_and_under_latency_budget(self):
        report = evaluate_defense(_verified_corpus(), book=BOOK)
        self.assertTrue(report["observational_only"])
        self.assertFalse(report["counterfactual_claim"])
        self.assertEqual(report["latency_ms"]["over_budget"], 0)
        self.assertLessEqual(report["latency_ms"]["max"], 150)
        self.assertGreater(report["model"]["uses_trained_play_evidence"], 0)
        self.assertIn("holdout_games", report)


class DefenseLiveGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed=mdata.load_seed())
        ensure_madden_opponent(self.db, "john")
        playbook._save_state(self.db, {"applied": {
            "defense": {
                "side": "defense", "mode": "custom", "name": "Lions",
                "source_book": "Lions", "rev": 1, "locked_ts": "test",
                "formations": BOOK,
            }
        }, "pending": {}})
        store_selection(self.db, "john", {
            "offense": ["O-RUN"],
            "defense": ["SAFE DEEP", "TAMPA MABLE"],
        })
        art = defense.train_defense(_verified_corpus())
        dest = Path(self.tmp.name) / "defense.json"
        defense.save_model(art, dest)
        self.db.set_meta(defense.MODEL_META, str(dest))
        defense.set_mode(self.db, "experimental")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_enable_cli_refuses_without_verified_training(self):
        from cfb_coach.cli import main

        db_path = Path(self.tmp.name) / "empty.db"
        empty = CoachDB(db_path, seed=mdata.load_seed())
        playbook._save_state(empty, {"applied": {
            "defense": {
                "side": "defense", "mode": "custom", "name": "Lions",
                "source_book": "Lions", "rev": 1, "locked_ts": "test",
                "formations": BOOK,
            }
        }, "pending": {}})
        empty.close()
        code = main(["ml", "defense-experimental", "--enable", "--db", str(db_path)])
        self.assertEqual(code, 2)
        check = CoachDB.open_read_only(db_path)
        try:
            self.assertNotEqual(check.get_meta(defense.MODE_META), "experimental")
        finally:
            check.close()

    def test_enable_refuses_when_combined_with_retrain(self):
        from cfb_coach.cli import main

        code = main([
            "ml", "defense-experimental", "--enable", "--retrain",
            "--db", str(self.db.path),
            "--out", str(Path(self.tmp.name) / "combo.json"),
        ])
        self.assertEqual(code, 2)

    def test_live_model_pick_is_experimental_in_book_and_fast(self):
        sit = parse_madden_situation("d 3&8 my 40", default_side="defense")
        sit.extras["require_trained_defense_artifact"] = True
        sit.extras["defense_armed_ids"] = ["SAFE DEEP", "TAMPA MABLE"]
        started = time.perf_counter()
        call = make_call(
            sit, "john", self.db, playbook={"offense": {}, "defense": BOOK},
            allow_human_defense_ml=True, live_macros=False,
            active_macros={"offense": [], "defense": []},
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self.assertLessEqual(elapsed, 150)
        self.assertIn(call.play, BOOK[call.formation])
        from cfb_coach.madden.model.dual_control import displayed_label

        self.assertEqual(displayed_label(call), "EXPERIMENTAL")
        self.assertIn("DEFENSE ML", call.rationale)
        self.assertNotEqual(call.macro, "QTRS OVERTOP")

    def test_unarmed_macro_is_dropped_not_shown(self):
        heuristic = MaddenCall(
            "defense", "4-3 Over", "Cover 1 Contain", "No adj", "User hook", "h",
        )
        sit = parse_madden_situation("d 3&8 showing mesh", default_side="defense")
        sit.extras["require_trained_defense_artifact"] = True
        sit.extras["defense_armed_ids"] = ["SAFE DEEP"]
        fake = {
            "selected": {"formation": "4-3 Over", "play": "Cover 3 Sky", "family": "single_high", "score": 1},
            "top": [{"formation": "4-3 Over", "play": "Cover 3 Sky", "family": "single_high",
                     "score": 1, "football_intelligence": {}}],
            "tendencies": {}, "model_quality": "limited", "n_candidates": 4,
        }
        plan = {"id": "QTRS OVERTOP", "kind": "macro", "influence": 0.2, "why": "unarmed",
                "macro": {"name": "QTRS OVERTOP", "settings": []}}
        with mock.patch.object(defense, "live_pick", return_value=fake):
            with mock.patch(
                "cfb_coach.madden.model.defense_action_policy.select_defensive_action",
                return_value=plan,
            ):
                call = defense.maybe_apply_defense(
                    heuristic, sit, "john", self.db, BOOK,
                )
        self.assertEqual(call.play, "Cover 3 Sky")
        self.assertIsNone(call.macro)
        self.assertIn("dropped unarmed", call.rationale)

    def test_timeout_and_illegal_play_fall_back_labeled_heuristic(self):
        heuristic = MaddenCall("defense", "4-3 Over", "Cover 1 Contain", "No adj", "User hook", "h")
        sit = parse_madden_situation("d 2&7", default_side="defense")
        sit.extras["require_trained_defense_artifact"] = True
        sit.extras["defense_armed_ids"] = ["SAFE DEEP"]
        with mock.patch.object(defense, "live_pick", side_effect=TimeoutError("budget")):
            timed = defense.maybe_apply_defense(heuristic, sit, "john", self.db, BOOK)
        self.assertEqual((timed.formation, timed.play), (heuristic.formation, heuristic.play))
        from cfb_coach.madden.model.dual_control import displayed_label as _label

        self.assertEqual(_label(timed), "HEURISTIC")
        self.assertIs(timed._pending_ml_decision.fallback_reason, MLStatus.TIMEOUT)
        fake = {
            "selected": {"formation": "4-3 Over", "play": "NOT A CALL", "family": "man", "score": 1},
            "top": [{"formation": "4-3 Over", "play": "NOT A CALL", "family": "man",
                     "score": 1, "football_intelligence": {}}],
            "tendencies": {}, "model_quality": "limited", "n_candidates": 1,
        }
        with mock.patch.object(defense, "live_pick", return_value=fake):
            illegal = defense.maybe_apply_defense(heuristic, sit, "john", self.db, BOOK)
        self.assertEqual((illegal.formation, illegal.play), (heuristic.formation, heuristic.play))
        self.assertIs(illegal._pending_ml_decision.fallback_reason, MLStatus.INVALID_OUTPUT)

    def test_insufficient_training_stays_heuristic_even_if_mode_is_experimental(self):
        original = self.db.get_meta(defense.MODEL_META)
        thin = defense.train_defense([_row(game="g1", play="Cover 3 Sky", success=True, yards=1)])
        dest = Path(self.tmp.name) / "thin.json"
        defense.save_model(thin, dest)
        self.db.set_meta(defense.MODEL_META, str(dest))
        try:
            sit = parse_madden_situation("d 3&8 my 40", default_side="defense")
            sit.extras["require_trained_defense_artifact"] = True
            sit.extras["defense_armed_ids"] = ["SAFE DEEP", "TAMPA MABLE"]
            call = make_call(
                sit, "john", self.db, playbook={"offense": {}, "defense": BOOK},
                allow_human_defense_ml=True, live_macros=False,
                active_macros={"offense": [], "defense": []},
            )
            from cfb_coach.madden.model.dual_control import displayed_label

            self.assertEqual(displayed_label(call), "HEURISTIC")
            self.assertNotIn("DEFENSE ML", call.rationale)
        finally:
            self.db.set_meta(defense.MODEL_META, original)

    def test_train_defense_cli_does_not_enable(self):
        from cfb_coach.cli import main

        original_meta = self.db.get_meta(defense.MODEL_META)
        original_mode = defense.mode(self.db)
        defense.set_mode(self.db, "off")
        out = Path(self.tmp.name) / "fresh.json"
        src = Path(self.tmp.name) / "rows.jsonl"
        lines = []
        for i in range(12):
            row = _row(game="game-a", play="Cover 3 Sky", success=True, yards=2)
            row["snap_id"] = f"game-a-snap-{i:02d}"
            lines.append(json.dumps(row))
        src.write_text("\n".join(lines), encoding="utf-8")
        try:
            code = main([
                "ml", "train-defense", "--db", str(self.db.path), "--out", str(out), "--path", str(src),
            ])
            self.assertTrue(out.is_file())
            self.assertEqual(defense.mode(self.db), "off")
            self.assertEqual(self.db.get_meta(defense.MODEL_META), str(out.resolve()))
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["side"], "defense")
            self.assertTrue(payload["independent_of_offense"])
            self.assertIn(code, (0, 2))
            if code == 0:
                self.assertGreaterEqual(int(payload["supervised_defensive_snaps"]), 10)
        finally:
            self.db.set_meta(defense.MODEL_META, original_meta)
            defense.set_mode(self.db, original_mode)
