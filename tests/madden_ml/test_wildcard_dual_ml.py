"""Dual experimental coordinators for a human Wild Card game against John.

Synthetic database only. Defense is not trained on zero verified snaps.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.cli import main
from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden import data as mdata
from cfb_coach.madden.human_opponents import ensure_madden_opponent
from cfb_coach.madden.macros import store_selection
from cfb_coach.madden.model import defense_coordinator as defense
from cfb_coach.madden.model import experimental_live, experimental_model, inference
from cfb_coach.madden.model.dual_control import (
    adjustment_block_reason,
    defense_artifact_report,
    install_primary_lions_books,
    latency_summary,
    macro_is_live,
)
from cfb_coach.madden.model.schema import CoachingMode, ML_LATENCY_BUDGET_MS, MLStatus
from cfb_coach.madden.playbook import load_books
from cfb_coach.madden.playcaller import MaddenCall, make_call
from cfb_coach.madden.situation import parse_madden_situation

_OFFENSE_SITS = [
    "1&10 my 25",
    "2&6 my 31 showing cover 3",
    "3&8 my 40 showing cover 2",
    "1&10 opp 40 showing cover 4",
    "2&9 opp 33 showing cover 1",
    "3&2 opp 28",
    "4&1 opp 42",
    "4&3 my 48",
    "4&12 opp 45 showing cover 6",
    "1&10 opp 18",
    "2&7 opp 14 showing cover 3",
    "3&goal opp 4",
    "1&goal opp 2 showing cover 0",
    "2&8 opp 22 two minute clock 1:20",
    "3&6 my 35 hurry clock 0:35",
    "1&10 opp 30 showing quarters",
    "2&5 opp 16 showing tampa 2",
    "1&10 my 40 showing bear front",
    "2&11 opp 37 showing invert cover 2",
    "3&1 opp 9",
    "1&15 my 12 clock 9:40",
    "4&2 opp 11 two minute",
    "2&2 opp 6 showing cover 9 palms",
    "3&14 opp 44 showing some unfamiliar shell",
]
_DEFENSE_SITS = [f"d {raw}" if not raw.startswith("d ") else raw for raw in _OFFENSE_SITS]
_ACTIVE = {
    "offense": ["MATCH", "C2", "O-RUN", "C3", "MAN", "RZ", "ZERO", "PROT"],
    "defense": ["SAFE DEEP", "TAMPA MABLE", "QTRS OVERTOP", "SKY CONTAIN"],
}


def _percentile_ok(summary: dict[str, float]) -> bool:
    return summary["max"] <= ML_LATENCY_BUDGET_MS or summary["n"] == 0


class DualControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.home = Path(self._td.name)
        self.db = CoachDB(self.home / "madden27.db", seed=mdata.load_seed())
        ensure_madden_opponent(self.db, "john")
        store_selection(self.db, "john", _ACTIVE)
        self.books = install_primary_lions_books(self.db)
        self.offense_book = load_books(self.db)["offense"]["formations"]
        self.defense_book = load_books(self.db)["defense"]["formations"]
        self.artifact = self.home / "offense.json"
        art = experimental_model.train_experimental([], side="offense")
        experimental_model.save_artifact(art, self.artifact)
        experimental_live.set_artifact_path(self.db, self.artifact)
        self.assertEqual(ML_LATENCY_BUDGET_MS, 150)
        self.assertGreaterEqual(len(self.books["offense"]), 2)
        self.assertGreaterEqual(len(self.books["defense"]), 2)
        self.assertEqual(load_books(self.db)["offense"]["source_book"], "Lions")
        self.assertEqual(load_books(self.db)["defense"]["source_book"], "Lions")

    def tearDown(self) -> None:
        self.db.close()
        self._td.cleanup()

    def _call(self, raw: str, **kwargs):
        sit = parse_madden_situation(raw, default_side=kwargs.pop("default_side", "offense"))
        return make_call(
            sit, "john", self.db, playbook=None, active_macros=_ACTIVE,
            rng=random.Random(0), allow_human_ml=kwargs.get("allow_human_ml", False),
            allow_human_defense_ml=kwargs.get("allow_human_defense_ml", False),
            live_macros=True,
        )

    def test_modes_and_failures_are_independent(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        defense.set_mode(self.db, "off")
        self.assertEqual(defense.mode(self.db), "off")
        defense.set_mode(self.db, "experimental")
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.EXPERIMENTAL)
        inference.set_mode(self.db, CoachingMode.HEURISTIC)
        self.assertEqual(defense.mode(self.db), "experimental")
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)

        with mock.patch(
            "cfb_coach.madden.model.experimental_live.apply_experimental_offense",
            side_effect=RuntimeError("offense exploded"),
        ):
            offense = self._call("1&10 my 25", allow_human_ml=True)
        self.assertEqual(offense.call_label, "HEURISTIC")
        self.assertIn("LABEL: HEURISTIC", offense.format())
        self.assertNotIn("LABEL: EXPERIMENTAL", offense.format())
        self.assertEqual(offense.ml_experimental["fallback_source"], "heuristic_offense")
        self.assertEqual(defense.mode(self.db), "experimental")
        self.assertIn(offense.play, self.offense_book[offense.formation])

        with mock.patch(
            "cfb_coach.madden.model.defense_coordinator.maybe_apply_defense",
            side_effect=RuntimeError("defense exploded"),
        ):
            defensive = self._call("d 2&7", allow_human_defense_ml=True)
        self.assertEqual(defensive.call_label, "HEURISTIC")
        self.assertEqual(defensive.ml_defense_fallback["fallback_source"], "select_defense")
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.EXPERIMENTAL)
        self.assertIn(defensive.play, self.defense_book[defensive.formation])
        kept = self._call("1&10 opp 40 showing cover 3", allow_human_ml=True)
        self.assertEqual(kept.call_label, "EXPERIMENTAL")
        self.assertIn("LABEL: EXPERIMENTAL", kept.format())
        self.assertNotIn("LABEL: HEURISTIC", kept.format())

    def test_offense_fallbacks_log_reason_and_source(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        with mock.patch.object(experimental_live, "resolve_artifact_path", return_value=None):
            with mock.patch.object(experimental_model, "train_experimental", side_effect=AssertionError("trained")):
                missing = self._call("1&10 my 25", allow_human_ml=True)
        decision = getattr(missing, "_pending_ml_decision")
        self.assertTrue(decision.fell_back)
        self.assertIs(decision.fallback_reason, MLStatus.MODEL_MISSING)
        self.assertIn("fallback_source=heuristic_offense", decision.ml_status_detail or "")
        self.assertEqual(missing.call_label, "HEURISTIC")
        experimental_live.set_artifact_path(self.db, self.artifact)

        heuristic = MaddenCall("offense", "Gun Doubles Clamp Stack", "Inside Zone", "No adj", "r", "h")
        sit = parse_madden_situation("1&10 my 25")
        book = {"Gun Doubles Clamp Stack": ["Inside Zone"]}

        def _boom(*_args, **_kwargs):
            raise RuntimeError("rank failed")

        with mock.patch.object(experimental_model, "rank_candidates", side_effect=_boom):
            _call, dec = experimental_live.apply_experimental_offense(
                heuristic_call=heuristic, sit=sit, opponent_id="john", db=self.db,
                book=book, active=_ACTIVE["offense"],
            )
        self.assertTrue(dec.fell_back)
        self.assertIs(dec.fallback_reason, MLStatus.EXCEPTION)
        self.assertIn("fallback_source=heuristic_offense", dec.ml_status_detail or "")
        self.assertTrue(_call.ml_experimental["fell_back"])
        self.assertEqual(_call.ml_experimental["fallback_source"], "heuristic_offense")

        def _slow(*_args, **_kwargs):
            time.sleep(0.05)
            return [{"formation": "Gun Doubles Clamp Stack", "play": "Inside Zone",
                     "probability": 0.5, "uncertainty": 0.2, "play_concept": "run",
                     "evidence_quality": "prior_driven"}]

        with mock.patch.object(experimental_model, "rank_candidates", side_effect=_slow):
            _call, dec = experimental_live.apply_experimental_offense(
                heuristic_call=heuristic, sit=sit, opponent_id="john", db=self.db,
                book=book, active=_ACTIVE["offense"], budget_ms=1,
            )
        self.assertIs(dec.fallback_reason, MLStatus.TIMEOUT)
        self.assertEqual(_call.ml_experimental["fallback_source"], "heuristic_offense")
        self.assertLessEqual(float(dec.latency_ms or 0), 150)

        illegal = MaddenCall("offense", "Gun Doubles Clamp Stack", "Inside Zone", "No adj", "r", "h")
        with mock.patch.object(
            experimental_model, "rank_candidates",
            return_value=[{"formation": "Gun Doubles Clamp Stack", "play": "NOT A PLAY",
                           "probability": 0.9, "uncertainty": 0.1, "play_concept": "pass",
                           "evidence_quality": "prior_driven"}],
        ):
            _call, dec = experimental_live.apply_experimental_offense(
                heuristic_call=illegal, sit=sit, opponent_id="john", db=self.db,
                book=book, active=_ACTIVE["offense"],
            )
        self.assertIs(dec.fallback_reason, MLStatus.INVALID_OUTPUT)
        self.assertIs(_call, illegal)

        bad = MaddenCall(
            "offense", "Gun Doubles Clamp Stack", "Inside Zone", "No adj", "r", "h",
            macro="NOT-A-REAL-MACRO",
        )
        with mock.patch.object(experimental_live, "rebuild_offense_attachments", return_value=bad):
            _call, dec = experimental_live.apply_experimental_offense(
                heuristic_call=heuristic, sit=sit, opponent_id="john", db=self.db,
                book=book, active=_ACTIVE["offense"],
            )
        self.assertIs(dec.fallback_reason, MLStatus.INVALID_OUTPUT)
        self.assertIn("invalid_adjustment", dec.ml_status_detail or "")
        self.assertEqual(_call.ml_experimental["fallback_source"], "heuristic_offense")
        self.assertIsNone(getattr(_call, "macro", None))

    def test_defense_opt_in_falls_back_without_a_trained_artifact(self) -> None:
        defense.set_mode(self.db, "experimental")
        report = defense_artifact_report(self.db)
        self.assertFalse(report["valid_real_training"])
        self.assertEqual(report["training_rows"], 0)
        with mock.patch.object(defense, "train_defense", side_effect=AssertionError("trained")):
            call = self._call("d 1&10 opp 25 showing cover 3", allow_human_defense_ml=True)
        self.assertEqual(call.call_label, "HEURISTIC")
        self.assertNotIn("LABEL: EXPERIMENTAL", call.format())
        decision = getattr(call, "_pending_ml_decision")
        self.assertIs(decision.fallback_reason, MLStatus.MODEL_MISSING)
        self.assertIn("fallback_source=select_defense", decision.ml_status_detail or "")
        self.assertIn(call.play, self.defense_book[call.formation])
        plain = self._call("d 2&7")
        self.assertFalse(hasattr(plain, "_pending_ml_decision"))
        self.assertEqual(plain.call_label, "HEURISTIC")

    def test_defense_timeout_exception_and_illegal_play_fall_back(self) -> None:
        row = {
            "supervised_eligible": True, "side": "defense", "executed_play": "Cover 3",
            "success": "true", "executed_verification": "verified",
            "executed_status": "identified", "label_available": True,
            "opponent_type": "human", "opponent_id": "john", "game_id": "synthetic-unit",
        }
        art = defense.train_defense([row])
        dest = self.home / "defense.json"
        defense.save_model(art, dest)
        self.db.set_meta(defense.MODEL_META, str(dest))
        defense.set_mode(self.db, "experimental")
        self.assertGreater(int(art["supervised_defensive_snaps"]), 0)
        heuristic = MaddenCall("defense", "Nickel 3-3 Over", "Cover 3", "No adj", "User hook", "h")
        sit = parse_madden_situation("d 2&7 showing mesh", default_side="defense")
        sit.extras["require_trained_defense_artifact"] = True
        sit.extras["defense_armed_ids"] = list(_ACTIVE["defense"])

        with mock.patch.object(defense, "live_pick", side_effect=TimeoutError("budget")):
            timed = defense.maybe_apply_defense(
                heuristic, sit, "john", self.db, self.defense_book,
            )
        self.assertEqual((timed.formation, timed.play), (heuristic.formation, heuristic.play))
        self.assertIs(timed._pending_ml_decision.fallback_reason, MLStatus.TIMEOUT)
        self.assertIn("fallback_source=select_defense", timed._pending_ml_decision.ml_status_detail)

        with mock.patch.object(defense, "live_pick", side_effect=RuntimeError("boom")):
            exploded = defense.maybe_apply_defense(
                heuristic, sit, "john", self.db, self.defense_book,
            )
        self.assertIs(exploded._pending_ml_decision.fallback_reason, MLStatus.EXCEPTION)

        fake = {
            "selected": {"formation": "Nickel 3-3 Over", "play": "NOT A CALL", "family": "man", "score": 1},
            "top": [{"formation": "Nickel 3-3 Over", "play": "NOT A CALL", "family": "man",
                     "score": 1, "football_intelligence": {}}],
            "tendencies": {}, "model_quality": "limited", "n_candidates": 1,
        }
        with mock.patch.object(defense, "live_pick", return_value=fake):
            illegal = defense.maybe_apply_defense(
                heuristic, sit, "john", self.db, self.defense_book,
            )
        self.assertEqual((illegal.formation, illegal.play), (heuristic.formation, heuristic.play))
        self.assertIs(illegal._pending_ml_decision.fallback_reason, MLStatus.INVALID_OUTPUT)
        self.assertIn("illegal play", illegal._pending_ml_decision.ml_status_detail)

        blocked = adjustment_block_reason(
            MaddenCall("defense", "Nickel 3-3 Over", "Cover 3", "No adj", "User hook", "h", macro="DRAFT-SHEET"),
            side="defense", armed=_ACTIVE["defense"], db=self.db, opponent_id="john",
        )
        self.assertIsNotNone(blocked)
        self.assertTrue(macro_is_live("SAFE DEEP", side="defense", armed_ids=_ACTIVE["defense"]))
        self.assertTrue(macro_is_live("O-RUN", side="offense", armed_ids=_ACTIVE["offense"]))
        self.assertFalse(macro_is_live("O-RUN", side="offense", armed_ids=[]))

    def test_mid_session_rollback_keeps_history(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        defense.set_mode(self.db, "experimental")

        def _make(sit, **kwargs):
            return make_call(
                sit, "john", self.db, active_macros=_ACTIVE, allow_human_ml=True,
                allow_human_defense_ml=True, live_macros=True, rng=random.Random(1),
            )

        ctrl = LivePlayController(
            db=self.db, opponent_id="john", make_call=_make,
            parse_situation=parse_madden_situation, learn_summary=lambda: "synthetic",
            brand="Madden 27 Franchise", enable_execution_verify=True,
        )
        ctrl.start()
        first = ctrl.call_only("1&10 my 25 showing cover 4")
        self.assertTrue(first["ok"])
        self.assertIn("LABEL: EXPERIMENTAL", first["call_text"])
        logged = ctrl.result_and_call(
            outcome="+4", sit_raw="2&6 my 29", their="cover 3", verify_touched=False,
        )
        self.assertTrue(logged["ok"], logged)
        before = self.db.conn.execute("SELECT COUNT(*) AS n FROM ml_decisions").fetchone()["n"]
        snaps_before = self.db.conn.execute("SELECT COUNT(*) AS n FROM snaps").fetchone()["n"]
        self.assertGreater(before, 0)
        self.assertGreater(snaps_before, 0)
        inference.set_mode(self.db, CoachingMode.HEURISTIC)
        self.assertEqual(defense.mode(self.db), "experimental")
        second = ctrl.call_only("3&3 my 33")
        self.assertTrue(second["ok"])
        self.assertIn("LABEL: HEURISTIC", second["call_text"])
        self.assertNotIn("LABEL: EXPERIMENTAL", second["call_text"])
        logged_again = ctrl.result_and_call(
            outcome="+2", sit_raw="d 1&10", their="inside zone", verify_touched=True,
            executed_status="used_recommended",
        )
        self.assertTrue(logged_again["ok"], logged_again)
        after = self.db.conn.execute("SELECT COUNT(*) AS n FROM ml_decisions").fetchone()["n"]
        snaps_mid = self.db.conn.execute("SELECT COUNT(*) AS n FROM snaps").fetchone()["n"]
        self.assertGreaterEqual(after, before)
        self.assertGreater(snaps_mid, snaps_before)
        defense.set_mode(self.db, "off")
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.HEURISTIC)
        third = ctrl.call_only("d 2&8", side="defense")
        self.assertTrue(third["ok"])
        self.assertIn("LABEL: HEURISTIC", third["call_text"])
        self.assertNotIn("DEFENSE ML", third["call_text"])
        ended = ctrl.end_game(result_wl="win", score="17-14")
        self.assertTrue(ended["ok"], ended)
        session = self.db.conn.execute(
            "SELECT ended_ts, score, result_wl FROM game_sessions WHERE session_id=?",
            (ctrl.session_id,),
        ).fetchone()
        self.assertTrue(session["ended_ts"])
        self.assertEqual(session["score"], "17-14")
        self.assertEqual(session["result_wl"], "win")
        snaps_after = self.db.conn.execute("SELECT COUNT(*) AS n FROM snaps").fetchone()["n"]
        self.assertGreaterEqual(snaps_after, snaps_mid)
        self.assertEqual(
            self.db.conn.execute("SELECT COUNT(*) AS n FROM ml_decisions").fetchone()["n"],
            after,
        )

    def test_lions_validation_and_report(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        defense.set_mode(self.db, "experimental")
        offense_rows = []
        defense_rows = []
        with mock.patch.object(defense, "train_defense", side_effect=AssertionError("no defense train")):
            with mock.patch.object(
                experimental_model, "train_experimental", side_effect=AssertionError("no live retrain"),
            ):
                for raw in _OFFENSE_SITS:
                    started = time.perf_counter()
                    call = self._call(raw, allow_human_ml=True)
                    wall = (time.perf_counter() - started) * 1000.0
                    self.assertIn(call.formation, self.offense_book)
                    self.assertIn(call.play, self.offense_book[call.formation])
                    self.assertIn(call.call_label, ("EXPERIMENTAL", "HEURISTIC"))
                    self.assertIn(f"LABEL: {call.call_label}", call.format())
                    if call.call_label == "HEURISTIC":
                        self.assertNotIn("LABEL: EXPERIMENTAL", call.format())
                    if call.macro:
                        self.assertTrue(macro_is_live(
                            call.macro, side="offense", armed_ids=_ACTIVE["offense"],
                        ))
                    decision = getattr(call, "_pending_ml_decision", None)
                    model_ms = None if decision is None else decision.latency_ms_model
                    if model_ms is not None and float(model_ms) > 150:
                        self.assertEqual(call.call_label, "HEURISTIC")
                    offense_rows.append({
                        "situation": raw,
                        "formation": call.formation,
                        "play": call.play,
                        "label": call.call_label,
                        "macro": call.macro,
                        "wall_ms": round(wall, 3),
                        "model_ms": None if model_ms is None else round(float(model_ms), 3),
                        "fell_back": bool(getattr(decision, "fell_back", call.call_label != "EXPERIMENTAL")),
                        "fallback_reason": None if decision is None or not decision.fell_back else decision.fallback_reason.value,
                        "fallback_source": None if not isinstance(getattr(call, "ml_experimental", None), dict) else call.ml_experimental.get("fallback_source"),
                    })
                for raw in _DEFENSE_SITS:
                    started = time.perf_counter()
                    call = self._call(raw, allow_human_defense_ml=True, default_side="defense")
                    wall = (time.perf_counter() - started) * 1000.0
                    self.assertEqual(call.side, "defense")
                    self.assertIn(call.formation, self.defense_book)
                    self.assertIn(call.play, self.defense_book[call.formation])
                    self.assertEqual(call.call_label, "HEURISTIC")
                    self.assertNotIn("LABEL: EXPERIMENTAL", call.format())
                    self.assertNotIn("DEFENSE ML", call.rationale)
                    if call.macro:
                        self.assertTrue(macro_is_live(
                            call.macro, side="defense", armed_ids=_ACTIVE["defense"], db=self.db, opponent_id="john",
                        ))
                    decision = getattr(call, "_pending_ml_decision", None)
                    self.assertIsNotNone(decision)
                    self.assertTrue(decision.fell_back)
                    self.assertIs(decision.fallback_reason, MLStatus.MODEL_MISSING)
                    defense_rows.append({
                        "situation": raw,
                        "formation": call.formation,
                        "play": call.play,
                        "label": call.call_label,
                        "wall_ms": round(wall, 3),
                        "model_ms": None if decision.latency_ms_model is None else round(float(decision.latency_ms_model), 3),
                        "fell_back": True,
                        "fallback_reason": decision.fallback_reason.value,
                        "fallback_source": "select_defense",
                    })
        self.assertGreaterEqual(len(offense_rows), 20)
        self.assertGreaterEqual(len(defense_rows), 20)
        categories = {
            "fourth": any(row["situation"].startswith("4&") for row in offense_rows),
            "red_zone": any("opp 1" in row["situation"] or "goal" in row["situation"] for row in offense_rows),
            "clock": any("clock" in row["situation"] or "two minute" in row["situation"] for row in offense_rows),
            "coverage": any("showing" in row["situation"] for row in offense_rows),
            "unfamiliar": any("unfamiliar" in row["situation"] or "bear front" in row["situation"] for row in offense_rows),
        }
        self.assertTrue(all(categories.values()), categories)
        offense_formations = {row["formation"] for row in offense_rows}
        defense_formations = {row["formation"] for row in defense_rows}
        self.assertGreaterEqual(len(offense_formations), 1)
        self.assertGreaterEqual(len(defense_formations), 1)
        offense_fallback = sum(1 for row in offense_rows if row["fell_back"]) / len(offense_rows)
        defense_fallback = sum(1 for row in defense_rows if row["fell_back"]) / len(defense_rows)
        self.assertEqual(defense_fallback, 1.0)
        wall_o = latency_summary(row["wall_ms"] for row in offense_rows)
        wall_d = latency_summary(row["wall_ms"] for row in defense_rows)
        model_o = latency_summary(row["model_ms"] for row in offense_rows if row["model_ms"] is not None)
        model_d = latency_summary(row["model_ms"] for row in defense_rows if row["model_ms"] is not None)
        self.assertTrue(_percentile_ok(wall_o))
        self.assertTrue(_percentile_ok(wall_d))
        self.assertTrue(_percentile_ok(model_o))
        self.assertTrue(_percentile_ok(model_d))
        from cfb_coach.madden.model.dual_control import artifact_fingerprint

        finger = artifact_fingerprint(str(self.artifact))
        defense_report = defense_artifact_report(self.db)
        profile = self.db.get_opponent("john") or {}
        self.assertEqual(profile["archetype"], "unknown")
        self.assertEqual(profile["confidence"], "none")
        report = {
            "offense_artifact": finger,
            "defense_artifact": defense_report,
            "offense_n": len(offense_rows),
            "defense_n": len(defense_rows),
            "offense_fallback_rate": offense_fallback,
            "defense_fallback_rate": defense_fallback,
            "offense_wall": wall_o,
            "defense_wall": wall_d,
            "offense_model": model_o,
            "defense_model": model_d,
            "offense_formations": sorted(offense_formations),
            "defense_formations": sorted(defense_formations),
            "offense_decisions": offense_rows,
            "defense_decisions": defense_rows,
            "offense_experimental": sum(1 for row in offense_rows if row["label"] == "EXPERIMENTAL"),
            "defense_experimental": sum(1 for row in defense_rows if row["label"] == "EXPERIMENTAL"),
        }
        Path("/tmp/wildcard_dual_ml_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.assertGreater(report["offense_experimental"], 0)
        self.assertEqual(report["defense_experimental"], 0)
        self.assertGreater(finger["sha256"], "")
        self.assertEqual(finger["training_rows"], 0)


class DualCliTests(unittest.TestCase):
    def test_commands_exist_in_one_shell_without_retrain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            db_path = home / "madden27.db"
            db = CoachDB(db_path, seed=mdata.load_seed())
            try:
                ensure_madden_opponent(db, "john")
                store_selection(db, "john", _ACTIVE)
                install_primary_lions_books(db)
                art = experimental_model.train_experimental([], side="offense")
                dest = home / "offense.json"
                experimental_model.save_artifact(art, dest)
                experimental_live.set_artifact_path(db, dest)
                inference.set_mode(db, CoachingMode.HEURISTIC)
                defense.set_mode(db, "off")
            finally:
                db.close()
            out = home / "backups"
            script = f"""
set -euo pipefail
export CFB_COACH_MADDEN_DB={db_path}
export HOME={home}
python3 -m cfb_coach ml backup-db --db "$CFB_COACH_MADDEN_DB" --out {out}
python3 -m cfb_coach ml defense-experimental
python3 -m cfb_coach ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"
python3 -m cfb_coach preflight --game madden27 --mode human -o john --no-vod-prior --db "$CFB_COACH_MADDEN_DB"
python3 -m cfb_coach ml experimental
python3 -m cfb_coach play --game madden27 -o john --franchise primary --once "1&10 my 25 showing cover 3" --no-overlay --terminal --no-vod-prior --ml-human-control --db "$CFB_COACH_MADDEN_DB"
python3 -m cfb_coach ml heuristic
if python3 -m cfb_coach play --game madden27 -o john --franchise primary --once "2&6 my 28" --no-overlay --terminal --no-vod-prior --ml-human-control --db "$CFB_COACH_MADDEN_DB"; then
  echo "offense flag should refuse after ml heuristic"
  exit 1
fi
python3 -m cfb_coach play --game madden27 -o john --franchise primary --once "2&6 my 28" --no-overlay --terminal --no-vod-prior --db "$CFB_COACH_MADDEN_DB"
python3 -m cfb_coach ml experimental
python3 -c 'from pathlib import Path; from cfb_coach.madden.model import defense_coordinator as defense; from cfb_coach.db import CoachDB; from cfb_coach.madden.data import load_seed; db=CoachDB(Path("{db_path}"), seed=load_seed()); defense.set_mode(db, "experimental"); db.close()'
python3 -m cfb_coach play --game madden27 -o john --once "d 2&7 showing mesh" --no-overlay --terminal --no-vod-prior --ml-defense-control --db "$CFB_COACH_MADDEN_DB"
python3 -m cfb_coach ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"
if python3 -m cfb_coach play --game madden27 -o john --once "d 1&10" --no-overlay --terminal --no-vod-prior --ml-defense-control --db "$CFB_COACH_MADDEN_DB"; then
  echo "defense flag should refuse when the defense mode is off"
  exit 1
fi
python3 -m cfb_coach play --game madden27 -o john --once "d 1&10" --no-overlay --terminal --no-vod-prior --db "$CFB_COACH_MADDEN_DB"
"""
            self.assertNotIn("--retrain", script)
            env = os.environ.copy()
            env["HOME"] = str(home)
            env["CFB_COACH_MADDEN_DB"] = str(db_path)
            proc = subprocess.run(
                ["bash", "-lc", script], cwd="/workspace", env=env,
                text=True, capture_output=True, check=False,
            )
            combined = proc.stdout + proc.stderr
            self.assertEqual(proc.returncode, 0, combined)
            self.assertIn("method: sqlite3.Connection.backup", combined)
            self.assertIn("verdict: PASS", combined)
            self.assertIn("HUMAN OFFENSE ML PILOT: ON", combined)
            self.assertIn("LABEL: EXPERIMENTAL", combined)
            self.assertIn("mode: heuristic", combined)
            self.assertIn("requires experimental mode", combined)
            self.assertIn("HUMAN DEFENSE PILOT: ON", combined)
            self.assertIn("LABEL: HEURISTIC", combined)
            self.assertNotIn("--retrain", combined)
            self.assertNotIn("trained:", combined)
            backups = list(out.iterdir())
            self.assertTrue(backups)
            checked = CoachDB(db_path, seed=mdata.load_seed())
            try:
                self.assertEqual(defense.mode(checked), "off")
                self.assertIs(inference.resolve_mode(checked), CoachingMode.EXPERIMENTAL)
                decisions = checked.conn.execute("SELECT COUNT(*) AS n FROM ml_decisions").fetchone()["n"]
                self.assertGreater(decisions, 0)
                profile = checked.get_opponent("john") or {}
                self.assertEqual(profile["archetype"], "unknown")
                self.assertEqual(profile.get("offense") or {}, {})
            finally:
                checked.close()


if __name__ == "__main__":
    unittest.main()
