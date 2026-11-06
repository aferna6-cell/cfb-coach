"""Opt-in NFL prior shadow: off by default, never changes the live call."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden.data import reads_for
from cfb_coach.madden.model import experimental_live, experimental_model, inference, nfl_prior_shadow as shadow
from cfb_coach.madden.model.schema import CoachingMode
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.situation import Situation


def _table(**overrides) -> dict:
    cells = {f"{d}|{i}|{j}": {"run": 0.40, "pass": 0.55, "p_pass": 0.6}
             for d in (1, 2, 3, 4) for i in range(7) for j in range(7)}
    table = {
        "schema": shadow.TABLE_SCHEMA, "model_version": "nfl-prior-shadow.test",
        "distance_edges": [1, 2, 3, 6, 10, 15], "yardline_edges": [5, 10, 20, 40, 60, 80],
        "cells": cells, "platt": [0.0, 1.0], "base_rate": 0.5, "prior_strength": 1.0,
    }
    table.update(overrides)
    return table


def _sit() -> Situation:
    return Situation(raw="1st & 10 at mid", side="offense", down=1, distance=10, yardline=50,
                     coverage_hint=None, coverage_source="none")


RANKED = [
    {"formation": "Gun Bunch", "play": "Inside Zone", "probability": 0.52},
    {"formation": "Gun Doubles", "play": "Four Verticals", "probability": 0.50},
]


class ShadowUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.table_path = self.dir / "table.json"
        self.table_path.write_text(json.dumps(_table()), encoding="utf-8")
        self.log = self.dir / "shadow.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _observe(self, **kw):
        args = dict(ranked=RANKED, chosen=("Gun Bunch", "Inside Zone"), sit=_sit(), opponent_type="cpu",
                    session_id="g1", snap_id="g1-0001", snap_seq=1, remaining_ms=100.0)
        args.update(kw)
        return shadow.observe(**args)

    def test_off_by_default_writes_nothing(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(self._observe())
        self.assertFalse(self.log.exists())

    def test_enabled_logs_version_prediction_situation_confidence_latency(self) -> None:
        env = {shadow.ENV_TABLE: str(self.table_path), shadow.ENV_LOG: str(self.log)}
        with mock.patch.dict("os.environ", env, clear=True):
            out = self._observe()
        self.assertEqual(out["status"], "ok")
        rec = json.loads(self.log.read_text().splitlines()[0])
        for key in ("model_version", "situation", "live_pick", "shadow_top", "confidence", "latency_ms",
                    "candidates", "in_validated_scope"):
            self.assertIn(key, rec)
        self.assertEqual(rec["live_pick"], {"formation": "Gun Bunch", "play": "Inside Zone"})
        self.assertEqual(rec["situation"]["yards_to_goal"], 50.0)
        # The pass cell beats the run cell, so the shadow would have preferred the pass.
        self.assertEqual(rec["shadow_top"]["play"], "Four Verticals")
        self.assertTrue(rec["shadow_would_differ"])
        self.assertLess(rec["latency_ms"], shadow.SHADOW_BUDGET_MS)

    def test_broken_table_is_an_error_record_not_an_exception(self) -> None:
        self.table_path.write_text(json.dumps(_table(schema="wrong")), encoding="utf-8")
        env = {shadow.ENV_TABLE: str(self.table_path), shadow.ENV_LOG: str(self.log)}
        with mock.patch.dict("os.environ", env, clear=True):
            out = self._observe()
        self.assertEqual(out["status"], "error")

    def test_skips_when_the_decision_budget_is_nearly_spent(self) -> None:
        env = {shadow.ENV_TABLE: str(self.table_path), shadow.ENV_LOG: str(self.log)}
        with mock.patch.dict("os.environ", env, clear=True):
            out = self._observe(remaining_ms=1.0)
        self.assertEqual(out["status"], "skipped_budget")
        self.assertFalse(self.log.exists())

    def test_zero_strength_or_prior_at_base_rate_leaves_the_model_probability(self) -> None:
        self.assertAlmostEqual(shadow.blend(0.6, 0.3, 0.5, 0.0), 0.6)
        self.assertAlmostEqual(shadow.blend(0.6, 0.5, 0.5, 1.0), 0.6)

    def test_lookup_uses_offense_yards_to_goal_bins(self) -> None:
        table = _table()
        table["cells"]["1|4|0"] = {"run": 0.9, "pass": 0.9, "p_pass": 0.5}
        p, kind, cell = shadow.nfl_probability(table, down=1, distance=10, yardline=97, play="Inside Zone")
        self.assertEqual((kind, cell), ("run", "1|4|0"))
        self.assertAlmostEqual(p, 0.9)
        self.assertIsNone(shadow.nfl_probability(table, down=None, distance=10, yardline=50, play="Mesh"))


class ShadowLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = CoachDB(self.dir / "t.db", seed={"opponents": {"cpu_lions": {
            "display_name": "Lions CPU", "team_now": "DET", "skill": "cpu", "confidence": "low",
            "profile_json": "{}"}}})
        self.book = {"Gun Bunch": ["Mesh", "Stick", "Smash"], "Gun Doubles": ["Y Cross", "Four Verticals", "HB Dive"],
                     "Singleback Ace": ["PA Boot", "Inside Zone", "Slant Flat"]}
        art = experimental_model.train_experimental([], side="offense")
        dest = self.dir / "exp.json"
        experimental_model.save_artifact(art, dest)
        experimental_live.set_artifact_path(self.db, dest)
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        self.table_path = self.dir / "table.json"
        self.table_path.write_text(json.dumps(_table()), encoding="utf-8")

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _call(self, seq: int):
        heur = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", reads_for("Mesh"), "heuristic")
        return experimental_live.apply_experimental_offense(
            heuristic_call=heur, sit=_sit(), opponent_id="cpu_lions", db=self.db, book=self.book,
            active=[], game_id="g-shadow", snap_id=None, snap_seq=seq, commit=False,
        )

    def test_live_pick_is_identical_with_shadow_on_off_or_crashing(self) -> None:
        log = self.dir / "shadow.jsonl"
        with mock.patch.dict("os.environ", {}, clear=True):
            off_call, off_dec = self._call(1)
        env = {shadow.ENV_TABLE: str(self.table_path), shadow.ENV_LOG: str(log)}
        with mock.patch.dict("os.environ", env, clear=True):
            on_call, on_dec = self._call(1)
            with mock.patch.object(shadow, "observe", side_effect=RuntimeError("boom")):
                crash_call, crash_dec = self._call(1)
        picks = {(c.formation, c.play) for c in (off_call, on_call, crash_call)}
        self.assertEqual(len(picks), 1)
        self.assertFalse(off_dec.fell_back or on_dec.fell_back or crash_dec.fell_back)
        self.assertNotIn("nfl_prior_shadow", off_call.ml_experimental)
        self.assertEqual(on_call.ml_experimental["nfl_prior_shadow"]["status"], "ok")
        self.assertNotIn("nfl_prior_shadow", crash_call.ml_experimental)
        self.assertEqual(len(log.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
