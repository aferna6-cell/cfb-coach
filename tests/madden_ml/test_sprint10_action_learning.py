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
    ):
        did = self.db.log_ml_decision({
            "snap_id": key, "game_id": game, "session_id": game,
            "snap_seq": 1, "mode": "experimental",
            "final_pick": {"formation": "Gun Bunch", "play": "Mesh"},
        })
        kind, _, action_id = action.partition(":")
        decision_json = {"experimental_offense": {"offense_action": {
            "kind": kind, "id": action_id if kind != "none" else None,
        }}}
        self.db.conn.execute(
            "UPDATE ml_decisions SET decision_json=? WHERE id=?",
            (json.dumps(decision_json), did),
        )
        self.db.conn.commit()
        self.db.log_snap(
            opponent_id="cpu", side="offense", session_id=game,
            situation_raw="2&8", down=2, distance=8,
            our_call="Gun Bunch — Mesh",
            formation="Gun Bunch", play="Mesh", result=result,
            executed_status="identified" if verified else "unknown",
            executed_formation=executed_play if verified else None,
            executed_play=executed_play if verified else None,
            executed_verification="verified" if verified else "unknown",
            ml_snap_id=key, ml_decision_id=did,
        )
        self.db.log_ml_outcome(
            snap_id=key, game_id=game, decision_id=did,
            executed_status="identified" if verified else "unknown",
            executed_formation="Gun Bunch" if verified else None,
            executed_play=executed_play if verified else None,
            executed_verification="verified" if verified else "unknown",
            outcome={
                "result": result, "yards": 12 if result == "+12" else 0,
                "offense_action_explicitly_confirmed": confirmed_action,
                "no_adjustment_explicitly_confirmed": confirmed_unchanged,
                "executed_adjustment_id": "SLIDE" if confirmed_action else None,
            }
        )

    def test_collector_requires_separate_action_or_unchanged_confirmation(self):
        self.snap("applied", confirmed_action=True)
        self.snap("no-action", action="none", confirmed_unchanged=True)
        self.snap("just-displayed")
        self.snap("not-sure", action="none")
        self.snap("different-play", confirmed_action=True, executed_play="Four Verticals")
        self.snap("not-verified", confirmed_action=True, verified=False)
        self.snap("wrong-flag", action="none", confirmed_action=True)
        rows, audit = collect_verified_action_rows(self.db)
        self.assertEqual({r["snap_id"] for r in rows}, {"applied", "no-action"})
        self.assertEqual({r["action"] for r in rows},
                         {"adjustment:SLIDE", "none"})
        self.assertEqual(audit["verified_applied_actions"], 1)
        self.assertEqual(audit["verified_unchanged"], 1)
        self.assertEqual(audit["excluded_play_mismatch"], 1)
        self.assertGreater(audit["excluded_action_confirmation"], 0)
        self.assertGreater(audit["excluded_no_action_confirmation"], 0)

    def test_repeat_corrections_not_double_counted(self):
        self.snap("same-snap", confirmed_action=True)
        self.snap("same-snap", confirmed_action=True)
        rows, _ = collect_verified_action_rows(self.db)
        self.assertEqual(len(rows), 1)

    def test_sparse_data_stays_shadow_and_cannot_promote(self):
        self.snap("a", confirmed_action=True)
        self.snap("b", action="none", confirmed_unchanged=True)
        art = train_action_evidence(self.db)
        self.assertEqual(art["n_verified_action_rows"], 2)
        self.assertEqual(art["ready_groups"], 0)
        self.assertEqual(art["mode"], "shadow")
        save_action_evidence(self.db, art)
        with self.assertRaisesRegex(ValueError, "No action has enough"):
            promote_action_evidence(self.db)

    def test_only_multiple_games_and_both_conditions_gate(self):
        rows = []
        for n in range(12):
            game = f"game-{n % 3}"
            rows.append({
                "snap_id": f"a-{n}", "game_id": game,
                "action": "adjustment:SLIDE", "context": "mesh|2_long",
                "success": True,
            })
            rows.append({
                "snap_id": f"b-{n}", "game_id": game,
                "action": "none", "context": "mesh|2_long",
                "success": False,
            })
        art = fit_action_evidence(rows)
        self.assertEqual(art["ready_groups"], 1)
        save_action_evidence(self.db, art)
        active = promote_action_evidence(self.db)
        self.assertEqual(active["mode"], "bounded_active")
        shift, details = score_shift(
            active, play="Mesh", down=2, distance=8,
            kind="adjustment", action_id="SLIDE",
        )
        self.assertGreater(shift, 0)
        self.assertLessEqual(shift, 0.04)
        self.assertGreater(details["n_unchanged"], 0)
        self.assertEqual(score_shift(
            active, play="Mesh", down=3, distance=11,
            kind="adjustment", action_id="SLIDE",
        ), (0.0, None))
        self.assertEqual(rollback_action_evidence(self.db)["mode"], "shadow")
        self.assertEqual(score_shift(
            load_action_evidence(self.db),
            play="Mesh", down=2, distance=8,
            kind="adjustment", action_id="SLIDE",
        ), (0.0, None))

    def test_one_game_cannot_pass_even_if_sample_count_is_high(self):
        rows = []
        for n in range(80):
            rows.append({
                "snap_id": f"s-{n}", "game_id": "one-game",
                "context": "mesh|2_long",
                "action": "adjustment:SLIDE" if n%2 else "none",
                "success": bool(n%2),
            })
        self.assertEqual(fit_action_evidence(rows)["ready_groups"], 0)

    def test_missing_unchanged_controls_cannot_pass(self):
        rows = [
            {"snap_id": f"s-{n}", "game_id": f"game-{n%4}",
             "context": "mesh|2_long", "action": "adjustment:SLIDE",
             "success": True}
            for n in range(200)
        ]
        self.assertEqual(fit_action_evidence(rows)["ready_groups"], 0)

    def test_live_eligibility_still_required_even_if_artifact_promoted(self):
        art = fit_action_evidence([
            {"snap_id":f"a-{n}", "game_id":f"g{n%3}", "context":"mesh|2_long",
             "action":"adjustment:SLIDE", "success":True}
            for n in range(15)
        ] + [
            {"snap_id":f"b-{n}", "game_id":f"g{n%3}", "context":"mesh|2_long",
             "action":"none", "success":False}
            for n in range(15)
        ])
        art["mode"] = "bounded_active"
        save_action_evidence(self.db, art)
        sit = SimpleNamespace(
            down=2, distance=8, coverage_hint=None, coverage_source="none",
            goal_line=False, red_zone=False,
        )
        call = choose_offense_action(
            formation="Gun Bunch", play="Mesh", sit=sit,
            book={"Gun Bunch":["Mesh"]}, active=[], db=self.db,
            opponent_id="cpu",
        )
        self.assertEqual(call["kind"], "none")
        self.assertEqual(call["candidates"], [])
        # No learned signal can invent a live coverage read or arm a macro.


if __name__ == "__main__":
    unittest.main()
