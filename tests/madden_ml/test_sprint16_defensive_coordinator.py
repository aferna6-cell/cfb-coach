"""Sprint 16 defensive coordinator: catalog, decisions, verification, opt-in safety."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model import defense_coordinator as defense
from cfb_coach.madden.model import defense_macro_lab as macros

CATALOG = {
    "49ers": {
        "Nickel Over": ["Cover 3 Sky", "Tampa 2", "Cover 1 Robber", "Mid Blitz 0"],
        "Dime Normal": ["Cover 4 Drop", "Cover 2 Sink", "Cover 3 Match"],
        "4-3 Over": ["Cover 3 Sky", "Cover 1 Hole", "Mike Blitz"],
    },
    "Lions": {
        "Nickel Double Mug": ["Mid Blitz 0", "Cover 1 Robber", "Cover 3 Match"],
        "Dollar": ["Cover 4 Drop", "Cover 2 Man"],
    },
}


def sit(**kwargs):
    fields = {
        "down": 1, "distance": 10, "yardline": 40,
        "red_zone": False, "goal_line": False, "two_minute": False,
        "concept_hint": None, "concept_source": "none", "extras": {},
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


class DefensivePolicyTests(unittest.TestCase):
    def test_defense_design_is_formations_with_every_play(self):
        p = defense.design_defense(max_formations=3, catalogue=CATALOG)
        self.assertGreater(len(p["formations"]), 0)
        self.assertLessEqual(len(p["formations"]), 3)
        self.assertEqual(p["mode"], "proposal_only")
        for form, calls in p["formations"].items():
            self.assertEqual(calls, CATALOG[p["formation_sources"][form]][form])

    def test_formation_cap_and_empty_rejection(self):
        with self.assertRaises(ValueError):
            defense.design_defense(max_formations=16, catalogue=CATALOG)
        with self.assertRaises(ValueError):
            defense.design_defense(catalogue={})

    def test_all_installed_calls_compete_not_fixed_six(self):
        book = CATALOG["49ers"]
        rows = defense.rank_defense(sit(down=3, distance=8), book)
        n = sum(bool(defense.call_family(p)) for ps in book.values() for p in ps)
        self.assertEqual(len(rows), n)
        self.assertEqual({(r["formation"], r["play"]) for r in rows},
                         {(f, p) for f, ps in book.items() for p in ps
                          if defense.call_family(p)})

    def test_current_live_concept_not_last_snap_tell(self):
        book = CATALOG["49ers"]
        last = defense.rank_defense(
            sit(down=3, distance=8, concept_hint="Four Verticals", concept_source="last"),
            book
        )
        none = defense.rank_defense(sit(down=3, distance=8), book)
        self.assertEqual([(r["formation"], r["play"], r["score"]) for r in last],
                         [(r["formation"], r["play"], r["score"]) for r in none])
        live = defense.rank_defense(
            sit(down=3, distance=8, concept_hint="Four Verticals", concept_source="live"),
            book
        )
        self.assertTrue(any(x["current_offensive_concept_used"] == "vert" for x in live))

    def test_ignores_unverified_outcomes(self):
        good = {
            "supervised_eligible": True, "side": "defense",
            "executed_play": "Cover 3 Sky", "success": "true",
        }
        fake = dict(good, supervised_eligible=False, eligibility="unverified", success="false")
        m = defense.train_defense([good, fake])
        self.assertEqual(m["supervised_defensive_snaps"], 1)
        self.assertEqual(m["evidence_quality"], "prior_driven")
        self.assertEqual(m["families"]["single_high"], [1, 1])

    def test_stable_near_tie_and_no_unknown_calls(self):
        book = CATALOG["49ers"]
        rows = defense.rank_defense(sit(), book)
        result = defense._stable_choice(rows, "gameA", 4)
        self.assertEqual(result, defense._stable_choice(rows, "gameA", 4))
        self.assertIn(result["play"], book[result["formation"]])


class DefensiveMacroTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed={"opponents": {}})
        self.book = CATALOG["49ers"]
        playbook._save_state(self.db, {"applied": {
            "defense": {"side": "defense", "mode": "stock", "name": "49ers",
                        "source_book": "49ers", "rev": 1, "locked_ts": "test",
                        "formations": self.book}
        }, "pending": {}})

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _recipes(self):
        return [
            {"id": "ONE", "base": {"formation": "Nickel Over", "play": "Cover 3 Sky"},
             "answers": ["cross"],
             "settings": [{"section": "General", "setting": "Coverage Shading",
                           "value": "Inside", "source": "civil-def-macros"}]},
            {"id": "TWO", "base": {"formation": "Nickel Over", "play": "Cover 3 Sky"},
             "answers": ["cross"],
             "settings": [{"section": "General", "setting": "QB Contain",
                           "value": "Both", "source": "civil-rollouts"}]},
        ]

    def test_original_macro_composed_with_full_editor_defaults(self):
        rows = macros.compose_defensive_macros(self.book, recipes=self._recipes())
        self.assertTrue(rows)
        first = rows[0]
        self.assertTrue(first["name"].startswith("ML-D-"))
        self.assertEqual(first["state"], "draft")
        self.assertFalse(first["verified_armed"])
        self.assertEqual(len(first["settings"]), len(macros._fields()))
        self.assertGreater(sum(x["value"] == "Default" for x in first["settings"]), 0)

    def test_conflicting_macro_settings_are_rejected(self):
        rows = self._recipes()
        rows[1]["settings"][0] = {
            "section": "General", "setting": "Coverage Shading",
            "value": "Outside", "source": "civil-def-macros",
        }
        self.assertEqual(macros.compose_defensive_macros(self.book, recipes=rows), [])

    def test_macro_cannot_learn_unknown_editor_keys(self):
        rows = self._recipes()
        rows[1]["settings"][0]["setting"] = "Invented Custom Hot Coverage"
        self.assertEqual(macros.compose_defensive_macros(self.book, recipes=rows), [])

    def test_stage_does_not_activate_without_attestation(self):
        rows = macros.compose_defensive_macros(self.book, recipes=self._recipes())
        with mock.patch.object(macros, "compose_defensive_macros", return_value=rows):
            staged = macros.stage_drafts(self.db)
        self.assertFalse(staged["armed"])
        self.assertEqual(macros.approved_macros(self.db, "gavin"), [])
        with self.assertRaises(ValueError):
            macros.verify_macro(self.db, "gavin", rows[0]["name"], "yes")

    def test_verified_macro_only_fires_for_correct_play_and_live_context(self):
        rows = macros.compose_defensive_macros(self.book, recipes=self._recipes())
        with mock.patch.object(macros, "compose_defensive_macros", return_value=rows):
            macros.stage_drafts(self.db)
        with mock.patch.object(macros, "load_selection", create=True):
            with mock.patch("cfb_coach.madden.macros.load_selection", return_value={"offense": [], "defense": []}):
                verified = macros.verify_macro(
                    self.db, "gavin", rows[0]["name"],
                    "I created all editor rows, tested the base play, and armed a defense slot."
                )
        self.assertTrue(verified["verified_armed"])
        self.assertTrue(macros.compatible_verified_macros(
            self.db, "gavin", "Nickel Over", "Cover 3 Sky", "cross"
        ))
        self.assertEqual(macros.compatible_verified_macros(
            self.db, "gavin", "Nickel Over", "Tampa 2", "cross"
        ), [])
        self.assertEqual(macros.compatible_verified_macros(
            self.db, "gavin", "Nickel Over", "Cover 3 Sky", "vert"
        ), [])
        macros.unverify_macro(self.db, "gavin", rows[0]["name"])
        self.assertEqual(macros.approved_macros(self.db, "gavin"), [])

    def test_defense_mode_opt_in_preserves_offense(self):
        self.assertEqual(defense.mode(self.db), "off")
        defense.set_mode(self.db, "experimental")
        self.assertEqual(defense.mode(self.db), "experimental")
        defense.set_mode(self.db, "off")
        self.assertEqual(defense.mode(self.db), "off")
        self.assertIsNone(self.db.get_meta("ml_experimental_artifact"))

    def test_staged_defense_install_needs_physical_attestation(self):
        with mock.patch.object(defense.catalog, "book_names", return_value=list(CATALOG)):
            with mock.patch.object(defense.catalog, "book_formations",
                                   side_effect=lambda side, name: CATALOG[name]):
                plan = defense.stage_design(self.db, max_formations=2)
                self.assertEqual(plan["status"], "staged_only")
                self.assertEqual(playbook.load_books(self.db)["defense"]["name"], "49ers")
                with self.assertRaises(ValueError):
                    defense.confirm_design(self.db, plan["proposal_id"], "yes")
                confirmed = defense.confirm_design(
                    self.db, plan["proposal_id"],
                    "All defensive formations and every source play have been installed and tested in Madden."
                )
                self.assertTrue(confirmed["confirmed"])
                self.assertEqual(confirmed["generated_macros_armed"], 0)


if __name__ == "__main__":
    unittest.main()
