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
from cfb_coach.madden.model.schema import (
    CoachingMode,
    MLStatus,
    PolicySource,
    PropensityMethod,
)
from cfb_coach.madden.playcaller import MaddenCall

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
        n = sum(len(ps) for ps in book.values())
        self.assertEqual(len(rows), n)
        self.assertEqual({(r["formation"], r["play"]) for r in rows},
                         {(f, p) for f, ps in book.items() for p in ps})

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
            "executed_verification": "verified", "executed_status": "identified",
            "label_available": True, "opponent_type": "human",
            "opponent_id": "gavin", "game_id": "user-match-1",
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
                preview = defense.confirm_design(self.db, plan["proposal_id"], "yes")
                self.assertFalse(preview["wrote_book"])
                self.assertTrue(preview["diff"])
                self.assertTrue(preview["install_instructions"])
                self.assertEqual(preview["generated_macros_armed"], 0)
                confirmed = defense.confirm_design(
                    self.db, plan["proposal_id"],
                    "All defensive formations and every source play have been installed and tested in Madden."
                )
                self.assertFalse(confirmed["wrote_book"])
                self.assertFalse(confirmed["confirmed"])
                self.assertEqual(confirmed["generated_macros_armed"], 0)
                self.assertEqual(playbook.load_books(self.db)["defense"]["name"], "49ers")

    def test_seeded_pick_unchanged_when_decision_is_logged(self):
        """The propensity log names the same call the selector already returned.

        Recorded before the log was added, on this catalog with no trained model:
        session ``seed-game`` snap 7 draws Nickel Over / Cover 3 Sky from the
        two-play near tie, and the applied joint sort shows 4-3 Over / Cover 3 Sky.
        """
        book = self.book
        situation = sit(extras={"session_id": "seed-game", "snap_seq": 7})
        rows = defense.rank_defense(situation, book)
        drawn = defense._stable_choice(rows, "seed-game", 7)
        drawn_again = defense._stable_choice(rows, "seed-game", 7)
        self.assertEqual(
            (drawn["formation"], drawn["play"]),
            (drawn_again["formation"], drawn_again["play"]),
        )
        self.assertEqual((drawn["formation"], drawn["play"]), ("Nickel Over", "Cover 3 Sky"))
        other = defense._stable_choice(rows, "seed-game", 1)
        self.assertEqual((other["formation"], other["play"]), ("4-3 Over", "Cover 3 Sky"))

        defense.set_mode(self.db, "experimental")
        logged = defense.live_pick(
            situation, self.db, "gavin", book, session_id="seed-game", snap_seq=7,
        )
        self.assertEqual(
            (logged["selected"]["formation"], logged["selected"]["play"]),
            (drawn["formation"], drawn["play"]),
        )
        decision = logged["decision"]
        self.assertEqual(decision.final_pick.formation, "Nickel Over")
        self.assertEqual(decision.final_pick.play, "Cover 3 Sky")
        self.assertIs(decision.policy_source, PolicySource.MODEL)
        self.assertIs(decision.propensity_method, PropensityMethod.EXACT)
        self.assertAlmostEqual(decision.behavior_propensity, 0.5)
        self.assertFalse(decision.propensity_override)
        self.assertIsNone(decision.propensity_override_reason)
        self.assertIs(decision.shadow_status, MLStatus.OK)
        self.assertFalse(decision.fell_back)
        self.assertIsNone(decision.fallback_reason)
        self.assertIs(decision.mode, CoachingMode.EXPERIMENTAL)

        heuristic = MaddenCall(
            "defense", "Nickel Over", "Tampa 2", "No adj", "User hook", "heuristic",
        )
        refused = defense.maybe_apply_defense(heuristic, situation, "gavin", self.db, book)
        self.assertIs(refused, heuristic)
        refused_decision = getattr(refused, defense.DECISION_ATTR)
        self.assertTrue(refused_decision.fell_back)
        self.assertIs(refused_decision.fallback_reason, MLStatus.MODEL_MISSING)
        self.assertEqual(refused.play, "Tampa 2")

        row = {
            "supervised_eligible": True, "side": "defense",
            "executed_play": "Cover 3 Sky", "success": "true",
            "executed_verification": "verified", "executed_status": "identified",
            "label_available": True, "opponent_type": "human",
            "opponent_id": "gavin", "game_id": "user-match-1",
        }
        art = defense.train_defense([row])
        dest = Path(self.tmp.name) / "defense.json"
        defense.save_model(art, dest)
        self.db.set_meta(defense.MODEL_META, str(dest))
        applied = defense.maybe_apply_defense(heuristic, situation, "gavin", self.db, book)
        applied_again = defense.maybe_apply_defense(heuristic, situation, "gavin", self.db, book)
        self.assertIn(applied.play, book[applied.formation])
        self.assertEqual(
            (applied.formation, applied.play, applied.macro, applied.adjustment),
            (applied_again.formation, applied_again.play, applied_again.macro, applied_again.adjustment),
        )
        applied_decision = getattr(applied, defense.DECISION_ATTR)
        self.assertEqual(applied_decision.final_pick.formation, applied.formation)
        self.assertEqual(applied_decision.final_pick.play, applied.play)
        self.assertEqual(applied_decision.heuristic_pick.play, "Tampa 2")
        self.assertIs(applied_decision.policy_source, PolicySource.MODEL)
        self.assertIs(applied_decision.propensity_method, PropensityMethod.DETERMINISTIC)
        self.assertEqual(applied_decision.behavior_propensity, 1.0)
        self.assertIs(applied_decision.shadow_status, MLStatus.OK)
        self.assertFalse(applied_decision.fell_back)

        defense.set_mode(self.db, "off")
        idle = MaddenCall(
            "defense", "Nickel Over", "Tampa 2", "No adj", "User hook", "heuristic",
        )
        kept = defense.maybe_apply_defense(idle, situation, "gavin", self.db, book)
        self.assertIs(kept, idle)
        self.assertFalse(hasattr(kept, defense.DECISION_ATTR))

        defense.set_mode(self.db, "experimental")
        tagged = sit(extras={
            "session_id": "seed-game", "snap_seq": 7, "snap_id": "snap-7", "game_id": "game-1",
        })
        persisted = defense.maybe_apply_defense(heuristic, tagged, "gavin", self.db, book)
        self.assertIn(persisted.play, book[persisted.formation])
        rows_logged = list(self.db.conn.execute("SELECT decision_json, policy_source FROM ml_decisions"))
        self.assertEqual(len(rows_logged), 1)
        self.assertEqual(rows_logged[0]["policy_source"], "model")
        import json
        body = json.loads(rows_logged[0]["decision_json"])
        self.assertEqual(body["propensity_method"], "deterministic")
        self.assertEqual(body["behavior_propensity"], 1.0)
        self.assertEqual(body["shadow_status"], "ok")
        self.assertFalse(body["fell_back"])
        self.assertIsNone(body["fallback_reason"])
        self.assertEqual(body["final_pick"]["play"], persisted.play)


if __name__ == "__main__":
    unittest.main()
