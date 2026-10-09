"""Sprint 10 / 11: macro lab and concept-specific Custom Adjustment proposals."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model import offense_macro_lab as lab
from cfb_coach.madden.model.offense_designer import (
    META_BLUEPRINTS, META_APPROVED, verified_created_macros,
)


def _db(path):
    return CoachDB(path, seed={"opponents": {
        "cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu",
                "confidence": "low", "profile_json": "{}"}
    }})


class MacroLabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _db(Path(self.tmp.name) / "coach.db")
        rec = {
            "side": "offense", "mode": "custom", "name": "ML Designed Offense (custom)",
            "rev": 6, "locked_ts": "2026-10-08T19:00:00+00:00",
            "formations": {
                "Gun Bunch": ["Mesh", "Flood", "Quick Slants", "Inside Zone"],
            },
            "formation_sources": {"Gun Bunch": "Buccaneers"},
            "audibles": {}, "core": ["Gun Bunch"],
        }
        playbook._save_state(self.db, {"applied": {"offense": rec}, "pending": {}})
        self.patch = mock.patch.object(
            lab.research_db, "offense_adjustments", return_value=[
                {"id": "hot_slant_vs_man", "type": "hot_route", "vs": ["man"],
                 "target": "WR1", "route": "Slant", "sources": ["prodigy-beat-man"]},
            ],
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.db.close()
        self.tmp.cleanup()

    def test_model_creates_new_named_play_targeted_source_backed_drafts(self):
        plan = lab.propose_variants(self.db)
        self.assertEqual(plan["status"], "DRAFT_ONLY")
        self.assertTrue(plan["proposals"])
        self.assertEqual(plan["installed_revision"], 6)
        for item in plan["proposals"]:
            self.assertTrue(item["name"].startswith("ML-"))
            self.assertEqual(item["coverage"], "man")
            self.assertEqual(item["source_action_id"], "hot_slant_vs_man")
            self.assertTrue(item["source_ids"])
            self.assertTrue(item["base_pairs"])
            self.assertIn("NOT_ARMED", item["activation"])
            self.assertEqual(len(item["settings"]), 1)
            self.assertTrue(item["settings"][0]["sources"])
        self.assertIsNone(self.db.get_meta(META_BLUEPRINTS))

    def test_stage_drafts_does_not_arm_and_is_idempotent(self):
        first = lab.stage_variants(self.db, limit=4)
        self.assertGreater(first["added"], 0)
        second = lab.stage_variants(self.db, limit=4)
        self.assertEqual(second["added"], 0)
        saved = json.loads(self.db.get_meta(META_BLUEPRINTS))
        self.assertEqual(len(saved["macro_blueprints"]), first["added"])
        self.assertEqual(verified_created_macros(self.db, "cpu"), [])
        self.assertIsNone(self.db.get_meta(META_APPROVED.format(opponent="cpu")))

    def test_cannot_design_without_confirmed_installed_book(self):
        state = playbook._load_state(self.db)
        state["applied"]["offense"]["locked_ts"] = None
        playbook._save_state(self.db, state)
        with self.assertRaises(ValueError):
            lab.propose_variants(self.db)


if __name__ == "__main__":
    unittest.main()
