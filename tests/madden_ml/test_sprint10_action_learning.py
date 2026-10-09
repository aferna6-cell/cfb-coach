"""Sprint 10: only verified, explicitly applied offensive adjustments train actions."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden.model.offense_action_learning import (
    META_ACTION_MODEL, SCHEMA, collect_verified_action_rows,
    fit_action_evidence, load_action_evidence, promote_action_evidence,
    rollback_action_evidence, save_action_evidence, score_shift,
    train_action_evidence,
)
from cfb_coach.madden.model.offense_action_policy import choose_offense_action


class ActionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "actions.db", seed={
            "opponents": {
                "cpu": {"display_name":"CPU", "team_now":"DET",
                        "skill":"cpu","confidence":"low","profile_json":"{}"}
            }
        })

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def snap(
        self, key: str, *,
        action: str = "adjustment:SLIDE",
        confirmed_action: bool = False,
        confirmed_unchanged: bool = False,
        game: str = "game-a", executed_play: str = "Mesh",
        verified: bool = True, result: str = "+12",
        presnap_coverage: str | None = None,
        presnap_source: str | None = None,
        observed_coverage: str | None = None,
        red_zone: bool = False,
        goal_line: bool = False,
        down: int = 1,
        distance: int = 10,
    ):
        kind, _, act_id = action.partition(":")
        payload = {
            "experimental_offense": {
                "offense_action": {"kind": kind, "id": act_id if kind != "none" else None},
                "pre_snap_action_context": {
                    "down": down, "distance": distance,
                    "coverage_hint": presnap_coverage,
                    "coverage_source": presnap_source,
                    "red_zone": red_zone,
                    "goal_line": goal_line,
                },
            }
        }
        self.db.conn.execute(
            "INSERT INTO ml_decisions "
            "(decision_ts,game_id,snap_id,session_id,mode,final_formation,final_play,"
            "decision_json,created_ts) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "2026-10-08T20:00:00+00:00", game, key, game,
                "experimental", "Gun Bunch", "Mesh",
                json.dumps(payload), "2026-10-08T20:00:00+00:00",
            ),
        )
        self.db.conn.commit()
        self.db.log_snap(
            opponent_id="cpu", side="offense", situation_raw=f"{down}&{distance}",
            formation="Gun Bunch", play="Mesh",
            our_call="Gun Bunch — Mesh", down=down, distance=distance,
            yardline=15 if red_zone else (3 if goal_line else 35),
            result=result, session_id=game, ml_snap_id=key,
            coverage_seen=observed_coverage,
            executed_status="identified" if verified else "unknown",
            executed_formation="Gun Bunch" if verified else None,
            executed_play=executed_play if verified else None,
            executed_verification="verified" if verified else "unverified",
        )
        out_json = {
            "result": result, "yards": 12.0 if result.startswith("+") else -2.0,
            "kind": "gain" if result.startswith("+") else "loss",
            "executed_macro": act_id if kind == "macro" and confirmed_action else None,
            "executed_adjustment_id": act_id if kind == "adjustment" and confirmed_action else None,
            "offense_action_explicitly_confirmed": bool(confirmed_action),
            "no_adjustment_explicitly_confirmed": bool(confirmed_unchanged),
        }
        self.db.log_ml_outcome(
            snap_id=key, game_id=game,
            decision_id=1,
            executed_status="identified" if verified else "unknown",
            executed_formation="Gun Bunch" if verified else None,
            executed_play=executed_play if verified else None,
            executed_verification="verified" if verified else "unverified",
            outcome=out_json,
        )

    def test_collect_requires_explicit_confirmation(self):
        # 1. Action recommended, applied checkbox NOT clicked -> excluded
        self.snap("s1", action="adjustment:SLIDE", confirmed_action=False)
        # 2. None recommended, no-action checkbox clicked -> included as unchanged control
        self.snap("s2", action="none", confirmed_unchanged=True)
        # 3. Action recommended and checked -> included as action sample
        self.snap("s3", action="adjustment:SLIDE", confirmed_action=True)
        # 4. Unverified execution -> excluded
        self.snap("s4", action="adjustment:SLIDE", confirmed_action=True, verified=False)
        # 5. Play mismatch -> excluded
        self.snap("s5", action="adjustment:SLIDE", confirmed_action=True, executed_play="HB Dive")

        admitted, stats = collect_verified_action_rows(self.db)
        self.assertEqual(len(admitted), 2)
        self.assertEqual(stats["verified_applied_actions"], 1)
        self.assertEqual(stats["verified_unchanged"], 1)
        self.assertEqual(stats["excluded_action_confirmation"], 1)
        self.assertEqual(stats["excluded_play_mismatch"], 1)

    def test_fit_and_gates(self):
        # Generate 15 positive action samples and 15 control samples across 3 games
        for g in ("g1", "g2", "g3"):
            for i in range(5):
                self.snap(f"act_{g}_{i}", action="adjustment:SLIDE", confirmed_action=True,
                          game=g, result="+15")
                self.snap(f"ctrl_{g}_{i}", action="none", confirmed_unchanged=True,
                          game=g, result="-2")
        art = train_action_evidence(self.db)
        self.assertEqual(art["mode"], "shadow")
        self.assertEqual(art["ready_groups"], 1)
        grp = art["groups"][0]
        self.assertEqual(grp["action_id"], "SLIDE")
        self.assertTrue(grp["eligible_for_live_gate"])
        self.assertGreater(grp["score_shift"], 0.0)
        self.assertLessEqual(grp["score_shift"], 0.04)

        # Artifact must be explicitly promoted before score_shift activates
        save_action_evidence(self.db, art)
        shift, _ = score_shift(
            art, play="Mesh", down=1, distance=10, kind="adjustment", action_id="SLIDE",
        )
        self.assertEqual(shift, 0.0)  # shadow mode produces zero shift

        promoted = promote_action_evidence(self.db)
        self.assertEqual(promoted["mode"], "bounded_active")
        shift, meta = score_shift(
            promoted, play="Mesh", down=1, distance=10, kind="adjustment", action_id="SLIDE",
        )
        self.assertGreater(shift, 0.0)
        self.assertIsNotNone(meta)

        # Rollback returns it to shadow
        rolled = rollback_action_evidence(self.db)
        self.assertEqual(rolled["mode"], "shadow")
        shift, _ = score_shift(
            rolled, play="Mesh", down=1, distance=10, kind="adjustment", action_id="SLIDE",
        )
        self.assertEqual(shift, 0.0)

    def test_no_leakage_from_observed_postsnap_coverage(self):
        # Post-snap coverage is recorded, but no live pre-snap look was detected.
        # Context key must stay general, never bucketed by the later observed look.
        self.snap("s_blind", action="adjustment:SLIDE", confirmed_action=True,
                  presnap_coverage=None, presnap_source=None,
                  observed_coverage="cover2")
        admitted, _ = collect_verified_action_rows(self.db)
        self.assertEqual(len(admitted), 1)
        self.assertIsNone(admitted[0]["coverage_class"])

    def test_red_zone_context_isolation(self):
        # Red zone training data must not boost an open-field call
        for g in ("g1", "g2", "g3"):
            for i in range(5):
                self.snap(f"rz_act_{g}_{i}", action="adjustment:SLIDE", confirmed_action=True,
                          game=g, red_zone=True, result="+8")
                self.snap(f"rz_ctrl_{g}_{i}", action="none", confirmed_unchanged=True,
                          game=g, red_zone=True, result="-1")
        art = train_action_evidence(self.db)
        save_action_evidence(self.db, art)
        promote_action_evidence(self.db, force=True)
        active = load_action_evidence(self.db)
        # Open field check -> shift is 0
        shift_open, _ = score_shift(
            active, play="Mesh", down=1, distance=10, kind="adjustment", action_id="SLIDE",
            red_zone=False,
        )
        self.assertEqual(shift_open, 0.0)
        # Red zone check -> shift is active
        shift_rz, _ = score_shift(
            active, play="Mesh", down=1, distance=10, kind="adjustment", action_id="SLIDE",
            red_zone=True,
        )
        self.assertGreater(shift_rz, 0.0)

    def test_policy_applies_bounded_evidence(self):
        # When active model has positive evidence, choose_offense_action applies shift
        for g in ("g1", "g2", "g3"):
            for i in range(5):
                self.snap(f"act_{g}_{i}", action="adjustment:hot_slant_vs_man", confirmed_action=True,
                          game=g, presnap_coverage="man", presnap_source="live", result="+15")
                self.snap(f"ctrl_{g}_{i}", action="none", confirmed_unchanged=True,
                          game=g, presnap_coverage="man", presnap_source="live", result="-2")
        art = train_action_evidence(self.db)
        save_action_evidence(self.db, art)
        promote_action_evidence(self.db, force=True)

        book = {"Gun Bunch": ["Mesh"]}
        sit = SimpleNamespace(
            down=1, distance=10, red_zone=False, goal_line=False,
            coverage_hint="man", coverage_source="live",
        )
        active_macros = []
        choice = choose_offense_action(
            formation="Gun Bunch", play="Mesh", sit=sit, book=book,
            active=active_macros, db=self.db,
        )
        self.assertEqual(choice["observational_evidence_mode"], "bounded_active")
        self.assertTrue(choice["observation_not_causal"])
        # Action candidates list should reflect observational shift
        self.assertTrue(any("observational_model_shift" in c for c in choice.get("candidates", [])))


if __name__ == "__main__":
    unittest.main()
