"""Read-only action readiness reports distinguish prep selection from verification."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model import offense_action_inventory as inventory
from cfb_coach.madden.model.offense_designer import (
    META_APPROVED, META_BLUEPRINTS,
)


class ActionInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "db.sqlite", seed={"opponents": {
            "cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu",
                    "confidence": "low", "profile_json": "{}"}
        }})
        self.book = {"Gun Bunch": ["Mesh", "Inside Zone"]}
        self.rec = {
            "side": "offense", "name": "ML Designed Offense (custom)", "rev": 2,
            "mode": "custom", "formations": self.book, "formation_sources": {},
        }
        playbook._save_state(self.db, {"applied": {"offense": self.rec}, "pending": {}})

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_no_saved_macros_not_misrepresented_as_armed(self):
        out = inventory.offense_actions_report(self.db, "cpu")
        self.assertTrue(out["playbook_installed"])
        self.assertEqual(out["saved_loadout_macros"], 0)
        self.assertEqual(out["verified_model_created_macros"], 0)
        self.assertTrue(out["no_action_available"])

    def test_selected_macro_reports_readiness_but_not_editor_confirmation(self):
        self.db.set_meta("active_macros:cpu", json.dumps({
            "schema": 2, "offense": ["O-RUN"], "defense": []
        }))
        with (
            mock.patch.object(inventory, "clean_ids", return_value=["O-RUN"]),
            mock.patch.object(inventory, "offense_detail", return_value={
                "settings": [{"section": "Blocking", "value": "Default"}],
                "needs_settings": False, "gaps": [],
                "fire_when": "In short-yardage runs", "settings_source": "User notes",
            }),
            mock.patch.object(inventory, "pairs_in_book",
                              return_value=["Inside Zone (Gun Bunch)"]),
        ):
            report = inventory.offense_actions_report(self.db, "cpu")
        self.assertEqual(report["saved_loadout_macros"], 1)
        self.assertEqual(report["ready_in_saved_loadout"], 1)
        row = report["existing_macros"][0]
        self.assertFalse(row["editor_confirmed_armed"])
        self.assertEqual(row["supported_pairs"], 1)
        self.assertTrue(row["eligible_if_editor_armed_and_triggered"])

    def test_generated_macro_is_not_live_until_separately_verified(self):
        b = {
            "name": "ML-MAN-HR", "settings": [{"setting": "Receiver", "value": "Slant"}],
            "source_ids": ["guide"], "base_pairs": [{"formation": "Gun Bunch", "play": "Mesh"}],
            "fire_when": "Observed live man look",
        }
        self.db.set_meta(META_BLUEPRINTS, json.dumps({"macro_blueprints": [b]}))
        report = inventory.offense_actions_report(self.db, "cpu")
        self.assertEqual(report["verified_model_created_macros"], 0)
        self.assertEqual(report["unverified_generated_drafts"], ["ML-MAN-HR"])
        self.db.set_meta(
            META_APPROVED.format(opponent="cpu"),
            json.dumps({"schema":1, "macros": [{**b, "verified_armed": True}]}),
        )
        report = inventory.offense_actions_report(self.db, "cpu")
        self.assertEqual(report["verified_model_created_macros"], 1)
        self.assertEqual(report["unverified_generated_drafts"], [])
        self.assertTrue(report["verified_generated_macros"][0]["eligible_if_triggered"])
        other = inventory.offense_actions_report(self.db, "gavin")
        self.assertEqual(other["verified_model_created_macros"], 0)


if __name__ == "__main__":
    unittest.main()
