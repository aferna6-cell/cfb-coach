"""Sprint 6C: model-driven book editing and self-authored macro blueprints.

Unit tests intentionally use a tiny deterministic catalog so the real public
Madden book catalog can change without changing test expectations.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model import offense_designer as designer


CATALOG = {
    "Buccaneers": {
        "Gun Bunch": ["Mesh", "Inside Zone", "Flood", "Four Verticals", "Quick Slants"],
        "Singleback Wing": ["Stretch", "PA Boot", "Quick Slants"],
    },
    "Lions": {
        "Gun Tight": ["Mesh", "HB Dive", "Smash", "Slants", "Flood"],
        "Gun Trips": ["Inside Zone", "Verticals", "PA Cross", "HB Draw"],
    },
}


def _seed() -> dict:
    return {"opponents": {"cpu": {
        "display_name": "CPU", "team_now": "DET", "skill": "cpu",
        "confidence": "low", "profile_json": "{}"
    }}}


class ModelDesignedOffenseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed=_seed())
        self.old = {
            "side": "offense", "mode": "custom", "name": "Buccaneers",
            "source_book": "Buccaneers", "rev": 3, "trimmed": True,
            "formations": {"Gun Bunch": ["Mesh", "Inside Zone"]},
            "audibles": {}, "core": ["Gun Bunch"], "locked_ts": "test",
        }
        playbook._save_state(self.db, {"applied": {"offense": copy.deepcopy(self.old)},
                                        "pending": {}})
        self.artifact = experimental_model.ExperimentalArtifact(
            model_version="test.model", n_supervised=3,
            evidence_quality="mixed", global_rate=0.5
        )
        self.catalog_patch = mock.patch.object(
            designer.catalog, "book_formations",
            side_effect=lambda side, source: dict(CATALOG.get(source, {}))
        )
        self.catalog_patch.start()

    def tearDown(self) -> None:
        self.catalog_patch.stop()
        self.db.close()
        self.tmp.cleanup()

    def plan(self, **kwargs):
        return designer.design_offense(
            self.db, opponent_id="cpu", max_formations=3, max_plays=3,
            artifact=self.artifact, catalogue=CATALOG, **kwargs
        )

    def test_model_can_change_formations_and_individual_plays(self) -> None:
        plan = self.plan()
        self.assertLessEqual(len(plan["book"]["formations"]), 3)
        self.assertGreater(sum(map(len, plan["book"]["formations"].values())), 0)
        changes = plan["changes"]
        self.assertTrue(
            any(r["action"] in ("ADD_PLAY", "REMOVE_PLAY", "ADD_FORMATION", "REMOVE_FORMATION")
                for r in changes)
        )
        self.assertEqual(plan["book"]["rev"], 4)
        for f, plays in plan["book"]["formations"].items():
            source = plan["book"]["formation_sources"][f]
            self.assertTrue(set(plays).issubset(CATALOG[source][f]))
            self.assertLessEqual(len(plays), 3)
        self.assertEqual(playbook.load_books(self.db)["offense"], self.old)
        self.assertIsNone(designer.staged_design(self.db))

    def test_generated_macro_blueprints_are_drafts_and_sourced(self) -> None:
        p = self.plan()
        self.assertTrue(p["macro_blueprints"])
        for m in p["macro_blueprints"]:
            self.assertIn("DRAFT", m["status"])
            self.assertTrue(m["settings"])
            self.assertTrue(m["source_ids"])
            self.assertIn("NOT_ARMED", m["activation"])
            self.assertTrue(all(
                pair["play"] in p["book"]["formations"][pair["formation"]]
                for pair in m["base_pairs"]
            ))
        self.assertEqual(playbook.load_books(self.db)["offense"], self.old)

    def test_staging_does_not_change_applied_book_or_macro_selection(self) -> None:
        before_meta = self.db.get_meta(playbook.META_KEY)
        pending = designer.stage_design(self.db, self.plan())
        self.assertEqual(designer.staged_design(self.db)["proposal_id"], pending["proposal_id"])
        self.assertEqual(self.db.get_meta(playbook.META_KEY), before_meta)
        self.assertEqual(designer.stage_design(self.db, self.plan()), pending)

    def test_confirmation_requires_exact_id_and_explicit_installed_attestation(self) -> None:
        p = designer.stage_design(self.db, self.plan())
        with self.assertRaises(ValueError):
            designer.confirm_installed(self.db, proposal_id="bad", attestation="I installed and checked every play")
        with self.assertRaises(ValueError):
            designer.confirm_installed(self.db, proposal_id=p["proposal_id"], attestation="yes")
        self.assertEqual(playbook.load_books(self.db)["offense"], self.old)
        result = designer.confirm_installed(
            self.db, proposal_id=p["proposal_id"],
            attestation="I installed all the listed plays and formations in the Madden custom playbook editor.",
        )
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["macro_blueprints_activated"], 0)
        self.assertEqual(playbook.load_books(self.db)["offense"]["formations"], p["book"]["formations"])
        self.assertIsNone(designer.staged_design(self.db))
        history = self.db.get_meta(designer.META_HISTORY)
        self.assertIn(p["proposal_id"], history)

    def test_refuse_stale_playbook_revision(self) -> None:
        pending = designer.stage_design(self.db, self.plan())
        state = playbook._load_state(self.db)
        state["applied"]["offense"]["rev"] = 11
        playbook._save_state(self.db, state)
        with self.assertRaises(ValueError):
            designer.confirm_installed(
                self.db, proposal_id=pending["proposal_id"],
                attestation="I installed and checked every play in Madden's custom editor",
            )
        self.assertEqual(playbook.load_books(self.db)["offense"]["rev"], 11)

    def test_refuse_uncatalogued_play_even_when_marked_installed(self) -> None:
        p = self.plan()
        p["book"]["formations"][next(iter(p["book"]["formations"]))].append("Fabricated Hot Route Play")
        designer.stage_design(self.db, p)
        with self.assertRaises(ValueError):
            designer.confirm_installed(
                self.db, proposal_id=p["proposal_id"],
                attestation="I installed and checked every play in Madden's custom editor",
            )
        self.assertEqual(playbook.load_books(self.db)["offense"], self.old)

    def test_reject_oversized_offense_book(self) -> None:
        with self.assertRaises(ValueError):
            self.plan(max_plays=100)


if __name__ == "__main__":
    unittest.main()
