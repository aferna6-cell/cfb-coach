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
        opts = {"opponent_id": "cpu", "max_formations": 3,
                "artifact": self.artifact, "catalogue": CATALOG}
        opts.update(kwargs)
        return designer.design_offense(self.db, **opts)

    def test_model_can_change_formations_and_includes_every_source_play(self) -> None:
        plan = self.plan()
        self.assertLessEqual(len(plan["book"]["formations"]), 3)
        self.assertGreater(sum(map(len, plan["book"]["formations"].values())), 0)
        changes = plan["changes"]
        self.assertTrue(
            any(r["action"] in ("ADD_FORMATION", "REMOVE_FORMATION", "REINSTALL_FULL_FORMATION")
                for r in changes)
        )
        self.assertEqual(plan["book"]["rev"], 4)
        for f, plays in plan["book"]["formations"].items():
            source = plan["book"]["formation_sources"][f]
            self.assertEqual(plays, CATALOG[source][f])
            self.assertGreaterEqual(len(plays), 3)
        self.assertFalse(any("PLAY" in row["action"] and "FORMATION" not in row["action"] for row in changes))
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

    def test_rollback_requires_game_confirmation_and_restores_old_book(self) -> None:
        p = designer.stage_design(self.db, self.plan())
        designer.confirm_installed(
            self.db, proposal_id=p["proposal_id"],
            attestation="I installed every listed formation and play inside Madden.",
        )
        with self.assertRaises(ValueError):
            designer.rollback_design(self.db, proposal_id=p["proposal_id"], attestation="yes")
        result = designer.rollback_design(
            self.db, proposal_id=p["proposal_id"],
            attestation="I reinstalled the old Buccaneers custom book and checked every play in Madden.",
        )
        self.assertTrue(result["rolled_back"])
        restored = playbook.load_books(self.db)["offense"]
        self.assertEqual(restored["formations"], self.old["formations"])
        self.assertEqual(restored["rev"], p["book"]["rev"] + 1)

    def test_normal_prep_cannot_silently_replace_approved_ml_book(self) -> None:
        pending = designer.stage_design(self.db, self.plan())
        designer.confirm_installed(
            self.db, proposal_id=pending["proposal_id"],
            attestation="I installed every listed play and formation in Madden.",
        )
        before = playbook.load_books(self.db)["offense"]
        plan = playbook.plan_side(
            "offense", before, opp={"_id": "cpu", "archetype": ""},
            choice=None, team="Detroit Lions",
        )
        self.assertEqual(plan["record"]["formations"], before["formations"])
        self.assertEqual(plan["change"], "none")
        self.assertEqual(plan["deltas"], [])

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
        with self.assertRaises(ValueError):
            designer.stage_design(self.db, p)
        self.assertEqual(playbook.load_books(self.db)["offense"], self.old)

    def test_model_created_macro_requires_separate_editor_verification(self) -> None:
        from types import SimpleNamespace

        from cfb_coach.madden.model.offense_action_policy import choose_offense_action
        staged = designer.stage_design(self.db, self.plan())
        designer.confirm_installed(
            self.db, proposal_id=staged["proposal_id"],
            attestation="I installed all the proposed plays and formations inside Madden.",
        )
        macro = staged["macro_blueprints"][0]
        pair = macro["base_pairs"][0]
        context = SimpleNamespace(
            coverage_hint="Cover 1" if macro["coverage"] == "man" else "Blitz",
            coverage_source="live", goal_line=False, red_zone=False, down=2,
        )
        book = playbook.load_books(self.db)["offense"]["formations"]
        base_opts = dict(
            formation=pair["formation"], play=pair["play"],
            sit=context, book=book, active=[], opponent_id="cpu",
            db=self.db, prediction={"probability": 0.6, "uncertainty": 0.25},
        )
        self.assertEqual(designer.verified_created_macros(self.db, "cpu"), [])
        result = choose_offense_action(**base_opts)
        self.assertNotEqual(result.get("id"), macro["name"])

        with self.assertRaises(ValueError):
            designer.verify_created_macro(
                self.db, name=macro["name"], opponent_id="cpu",
                attestation="not verified",
            )
        approved = designer.verify_created_macro(
            self.db, name=macro["name"], opponent_id="cpu",
            attestation="I built the generated macro with exact researched settings and armed its custom slot in Madden.",
        )
        self.assertTrue(approved["verified_armed"])
        self.assertEqual(len(designer.verified_created_macros(self.db, "cpu")), 1)
        self.assertEqual(len(designer.verified_created_macros(self.db, "gavin")), 0)
        context.coverage_hint = "Cover 1" if macro["coverage"] == "man" else "Blitz"
        result = choose_offense_action(**base_opts)
        self.assertIn(macro["name"], [c["id"] for c in result["candidates"]])
        context.coverage_source = "last"
        result = choose_offense_action(**base_opts)
        self.assertNotIn(macro["name"], [c["id"] for c in result["candidates"]])

    def test_macro_slot_cap_requires_explicit_retirement(self) -> None:
        from cfb_coach.madden import macros

        staged = designer.stage_design(self.db, self.plan())
        designer.confirm_installed(
            self.db, proposal_id=staged["proposal_id"],
            attestation="I installed and checked every play and formation in Madden.",
        )
        ids = ["MATCH", "C2", "O-RUN", "C3", "MAN", "RZ", "ZERO", "SHOT"]
        macros.store_selection(
            self.db, "cpu", {"offense": ids, "defense": []}
        )
        chosen = staged["macro_blueprints"][0]["name"]
        att = "I installed and armed this macro, replacing one existing slot inside Madden."
        with self.assertRaises(ValueError):
            designer.verify_created_macro(
                self.db, name=chosen, opponent_id="cpu", attestation=att,
            )
        applied = designer.verify_created_macro(
            self.db, name=chosen, opponent_id="cpu", attestation=att,
            retire_existing="MATCH",
        )
        self.assertEqual(applied["retired_existing"], "MATCH")
        self.assertNotIn("MATCH", macros.load_selection(self.db, "cpu")["offense"])
        self.assertEqual(
            len(macros.load_selection(self.db, "cpu")["offense"])
            + len(designer.verified_created_macros(self.db, "cpu")), 8
        )

    def test_reject_oversized_offense_book(self) -> None:
        with self.assertRaises(ValueError):
            self.plan(max_formations=100)


if __name__ == "__main__":
    unittest.main()
