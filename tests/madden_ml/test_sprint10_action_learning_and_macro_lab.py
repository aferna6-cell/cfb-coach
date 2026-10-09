"""Sprint 10: action evidence, tiny observational model and novel macro drafts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model import offense_action_learning as learn
from cfb_coach.madden.model import offense_macro_lab as lab
from cfb_coach.madden.model.offense_designer import (
    META_BLUEPRINTS, META_APPROVED, verified_created_macros,
)


def _db(path):
    return CoachDB(path, seed={"opponents": {
        "cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu",
                "confidence": "low", "profile_json": "{}"}
    }})


def _record(db, *, ident, game, action="adjustment:PROT-EDGE", applied=True,
            executed_play="Mesh", result="+12", out_verified=True):
    kind, _, action_id = action.partition(":")
    db.conn.execute(
        "INSERT INTO ml_decisions "
        "(decision_ts,game_id,snap_id,session_id,mode,final_formation,final_play,"
        "decision_json,created_ts) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "2026-10-08T20:00:00+00:00", game, ident, game,
            "experimental", "Gun Bunch", "Mesh",
            json.dumps({"experimental_offense": {
                "offense_action": {"kind": kind, "id": action_id if kind != "none" else None}
            }}), "2026-10-08T20:00:00+00:00",
        ),
    )
    db.conn.commit()
    db.log_snap(
        opponent_id="cpu", side="offense", situation_raw="1&10",
        formation="Gun Bunch", play="Mesh",
        our_call="Gun Bunch — Mesh", down=1, distance=10,
        result=result, session_id=game, ml_snap_id=ident,
        executed_status="identified" if out_verified else "unknown",
        executed_formation="Gun Bunch" if out_verified else None,
        executed_play=executed_play if out_verified else None,
        executed_verification="verified" if out_verified else "unknown",
    )
    out = {
        "result": result, "offense_action_explicitly_confirmed": bool(applied),
        "executed_adjustment_id": (
            action_id if applied and kind == "adjustment" else None
        ),
        "executed_macro": action_id if applied and kind == "macro" else None,
    }
    db.log_ml_outcome(
        snap_id=ident, game_id=game,
        executed_status="identified" if out_verified else "unknown",
        executed_formation="Gun Bunch",
        executed_play=executed_play,
        executed_verification="verified" if out_verified else "unknown",
        outcome=out,
    )


class ActionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _db(Path(self.tmp.name) / "coach.db")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_exact_verified_applied_action_and_no_action_reference(self):
        _record(self.db, ident="g1-0001", game="g1")
        _record(self.db, ident="g1-0002", game="g1", action="none", applied=False)
        _record(self.db, ident="g1-0003", game="g1", applied=False)
        _record(self.db, ident="g1-0004", game="g1", out_verified=False)
        _record(self.db, ident="g1-0005", game="g1", executed_play="Flood")
        evidence = learn.collect_action_evidence(self.db)
        self.assertEqual(evidence["counts"]["verified_action_applications"], 1)
        self.assertEqual(evidence["counts"]["observed_no_recommended_action_references"], 1)
        self.assertEqual(len(evidence["rows"]), 2)
        self.assertGreaterEqual(
            evidence["counts"]["excluded"].get("suggested_action_not_explicitly_verified", 0), 1
        )
        self.assertNotIn("g1-0005", [x["snap_id"] for x in evidence["rows"]])

    def test_insufficient_evidence_stays_shadow_and_enable_refused(self):
        _record(self.db, ident="g1-0001", game="g1")
        fitted = learn.train_from_db(self.db)
        self.assertEqual(fitted["eligible_entries"], 0)
        self.assertEqual(fitted["examples"], 1)
        learn.save_artifact(self.db, fitted)
        self.assertEqual(self.db.get_meta(learn.META_MODE), "shadow")
        with self.assertRaises(ValueError):
            learn.set_mode(self.db, "enabled")
        signal = learn.action_signal(
            self.db, kind="adjustment", action_id="PROT-EDGE",
            play="Mesh", down=1, distance=10,
        )
        self.assertEqual(signal["ranking_shift"], 0.0)

    def test_multi_game_comparable_evidence_enables_bounded_observational_signal(self):
        examples = []
        for i in range(36):
            examples.append({
                "game_id": f"g{i % 3}",
                "context": "mesh|1_long",
                "action": (
                    "adjustment:PROT-EDGE" if i % 2 == 0
                    else "reference:no_action_recommended"
                ),
                "success": i % 2 == 0,
            })
        fitted = learn.fit_action_model({"rows": examples, "counts": {}})
        self.assertEqual(fitted["eligible_entries"], 1)
        entry = fitted["entries"][0]
        self.assertLessEqual(abs(entry["ranking_shift"]), learn.MAX_SCORE_INFLUENCE)
        learn.save_artifact(self.db, fitted)
        shadow = learn.action_signal(
            self.db, kind="adjustment", action_id="PROT-EDGE",
            play="Mesh", down=1, distance=10,
        )
        self.assertTrue(shadow["qualified"])
        self.assertEqual(shadow["ranking_shift"], 0.0)
        learn.set_mode(self.db, "enabled")
        active = learn.action_signal(
            self.db, kind="adjustment", action_id="PROT-EDGE",
            play="Mesh", down=1, distance=10,
        )
        self.assertNotEqual(active["ranking_shift"], 0)
        self.assertLessEqual(abs(active["ranking_shift"]), learn.MAX_SCORE_INFLUENCE)
        different = learn.action_signal(
            self.db, kind="adjustment", action_id="PROT-EDGE",
            play="Flood", down=1, distance=10,
        )
        self.assertEqual(different["ranking_shift"], 0)
        learn.set_mode(self.db, "shadow")
        self.assertEqual(learn.action_signal(
            self.db, kind="adjustment", action_id="PROT-EDGE",
            play="Mesh", down=1, distance=10,
        )["ranking_shift"], 0)

    def test_distinct_games_required_not_many_snaps_in_one_game(self):
        examples = [
            {"game_id": "g1", "context": "mesh|1_long",
             "action": ("adjustment:X" if i % 2 == 0
                        else "reference:no_action_recommended"), "success": True}
            for i in range(80)
        ]
        model = learn.fit_action_model({"rows": examples})
        self.assertEqual(model["eligible_entries"], 0)


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
