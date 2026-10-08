"""Sprint 4.2: execution default fix + one-time game 141d4a16b4184ee7 recovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from cfb_coach.db import CoachDB
from cfb_coach.last_snap import LastSnapBook
from cfb_coach.live_server import LivePlayController, render_live_html
from cfb_coach.madden.model import (
    dataset as dataset_mod,
    execution_recovery,
    experimental_model,
    inference,
)
from cfb_coach.madden.model.schema import (
    AppliedPlaybookRef,
    CandidatePlay,
    CoachingDecision,
    CoachingMode,
    ExecutedStatus,
    ML_LATENCY_BUDGET_MS,
    MLStatus,
    PolicySource,
    Possession,
    PropensityMethod,
    Tri,
    Verification,
)
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.situation import Situation

GAME = execution_recovery.RECOVERY_GAME_ID
ATTEST = execution_recovery.ATTESTATION_PHRASE


def _seed() -> dict:
    return {
        "opponents": {
            "cpu": {
                "display_name": "CPU",
                "team_now": "DET",
                "skill": "cpu",
                "confidence": "low",
                "profile_json": "{}",
            },
            "other_franchise": {
                "display_name": "Other",
                "team_now": "KC",
                "skill": "human",
                "confidence": "medium",
                "profile_json": "{}",
            },
        }
    }


def _cand(form: str, play: str) -> CandidatePlay:
    return CandidatePlay(
        playbook=AppliedPlaybookRef(side=Possession.OFFENSE),
        formation=form,
        play=play,
        in_applied_book=Tri.TRUE,
    )


def _decision(
    *,
    game_id: str,
    snap_id: str,
    snap_seq: int,
    final_form: str,
    final_play: str,
    heur_form: str | None = None,
    heur_play: str | None = None,
    shadow_form: str | None = None,
    shadow_play: str | None = None,
    agree: int = 1,
) -> tuple[CoachingDecision, int | None]:
    heur = _cand(heur_form or final_form, heur_play or final_play)
    final = _cand(final_form, final_play)
    shadow = _cand(shadow_form or final_form, shadow_play or final_play)
    return CoachingDecision(
        decision_ts=datetime.now(timezone.utc).isoformat(),
        latency_ms=12.0,
        latency_ms_model=12.0,
        inference_budget_ms=ML_LATENCY_BUDGET_MS,
        mode=CoachingMode.EXPERIMENTAL,
        effective_mode=CoachingMode.EXPERIMENTAL,
        game_id=game_id,
        snap_id=snap_id,
        session_id=game_id,
        snap_seq=snap_seq,
        policy_source=PolicySource.MODEL,
        heuristic_pick=heur,
        final_pick=final,
        shadow_pick=shadow,
        shadow_status=MLStatus.OK,
        fell_back=False,
        propensity_method=PropensityMethod.UNKNOWN,
        accepted=Tri.UNKNOWN,
    ), agree


def _seed_recovery_game(db: CoachDB) -> dict[str, Any]:
    """Representative copy: 5 decisions, 4 linked unknown outcomes, 1 unlinked, 1 already verified."""
    # Other franchise game — must remain untouched.
    other_dec, other_agree = _decision(
        game_id="franchise-other-001",
        snap_id="franchise-other-001-0001",
        snap_seq=1,
        final_form="Gun Bunch",
        final_play="Mesh",
    )
    other_id = db.log_ml_decision(other_dec, agree=other_agree)
    db.log_snap(
        opponent_id="other_franchise",
        side="offense",
        situation_raw="1&10",
        our_call="Gun Bunch — Mesh",
        formation="Gun Bunch",
        play="Mesh",
        result="+4",
        session_id="franchise-other-001",
        ml_snap_id="franchise-other-001-0001",
        executed_status="unknown",
        executed_verification="unknown",
    )
    db.log_ml_outcome(
        snap_id="franchise-other-001-0001",
        game_id="franchise-other-001",
        decision_id=other_id,
        executed_status="unknown",
        executed_verification="unknown",
        outcome={"result": "+4"},
    )

    meta = {"snaps": []}
    plays = [
        ("Gun Bunch", "Mesh", "+7", True, "unknown"),  # recover
        ("Gun Doubles", "Y Cross", "incomplete", True, "unknown"),  # recover
        ("Singleback Ace", "PA Boot", "+12", True, "unknown"),  # recover
        ("I Form Twins", "Power O", "td", True, "verified_already"),  # already ok
        ("Pistol Trips", "RPO Peek", None, False, "unlinked"),  # no outcome
        (
            "Gun Bunch",
            "Texas Y-Stutter Wheel",
            "sack",
            True,
            "unknown",
        ),  # snap 0010-style override recover
    ]
    for i, (form, play, result, linked, kind) in enumerate(plays, start=1):
        snap_id = f"{GAME}-{i:04d}"
        # Snap 0010 disagreement: heuristic Mesh, final Texas Y-Stutter Wheel
        if i == 6:
            dec, agree = _decision(
                game_id=GAME,
                snap_id=snap_id,
                snap_seq=i,
                final_form=form,
                final_play=play,
                heur_form="Gun Bunch",
                heur_play="Mesh",
                shadow_form=form,
                shadow_play=play,
                agree=0,
            )
        else:
            dec, agree = _decision(
                game_id=GAME,
                snap_id=snap_id,
                snap_seq=i,
                final_form=form,
                final_play=play,
                agree=1,
            )
        did = db.log_ml_decision(dec, agree=agree)
        db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="1&10",
            our_call=f"{form} — {play}",
            formation=form,
            play=play,
            result=result,
            session_id=GAME,
            ml_snap_id=snap_id,
            snap_seq=i,
            executed_status="identified" if kind == "verified_already" else "unknown",
            executed_formation=form if kind == "verified_already" else None,
            executed_play=play if kind == "verified_already" else None,
            executed_verification="verified" if kind == "verified_already" else "unknown",
            ml_decision_id=did,
        )
        if linked:
            if kind == "verified_already":
                db.log_ml_outcome(
                    snap_id=snap_id,
                    game_id=GAME,
                    decision_id=did,
                    executed_status="identified",
                    executed_formation=form,
                    executed_play=play,
                    executed_verification="verified",
                    outcome={"result": result},
                )
            else:
                db.log_ml_outcome(
                    snap_id=snap_id,
                    game_id=GAME,
                    decision_id=did,
                    executed_status="unknown",
                    executed_verification="unknown",
                    outcome={"result": result, "yards": 7 if result and result.startswith("+") else None},
                )
        meta["snaps"].append({"snap_id": snap_id, "kind": kind, "decision_id": did})
    return meta


# typing for _seed_recovery_game annotation without importing Any at top messily
from typing import Any  # noqa: E402


class UsedDifferentVerificationTests(unittest.TestCase):
    def test_fully_specified_used_different_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                book = LastSnapBook(
                    db=db,
                    opponent_id="cpu",
                    parse_situation=parse_madden_situation,
                    session_id="s1",
                )
                call = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "a", "h")
                sit = Situation(raw="1&10", side="offense", down=1, distance=10, yardline=25)
                book.remember_call(call, sit, ml_snap_id="s1-0001", snap_seq=1)
                closed = book.close_from_form(
                    "+5",
                    executed_status="used_different",
                    executed_formation="Gun Trips",
                    executed_play="Inside Zone",
                )
                self.assertIsNotNone(closed)
                assert closed is not None
                self.assertEqual(closed["executed_status"], "identified")
                self.assertEqual(closed["executed_play"], "Inside Zone")
                self.assertEqual(closed["executed_verification"], "verified")
                snap = db.conn.execute(
                    "SELECT * FROM snaps WHERE ml_snap_id=?", ("s1-0001",)
                ).fetchone()
                self.assertEqual(snap["executed_verification"], "verified")
                self.assertEqual(snap["executed_status"], "identified")
            finally:
                db.close()

    def test_used_different_missing_play_stays_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                book = LastSnapBook(
                    db=db,
                    opponent_id="cpu",
                    parse_situation=parse_madden_situation,
                    session_id="s1",
                )
                call = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "a", "h")
                sit = Situation(raw="1&10", side="offense", down=1, distance=10)
                book.remember_call(call, sit, ml_snap_id="s1-0002", snap_seq=1)
                closed = book.close_from_form(
                    "incomplete",
                    executed_status="used_different",
                    executed_formation="Gun Trips",
                    executed_play=None,
                )
                self.assertEqual(closed["executed_status"], "unknown")
                self.assertEqual(closed["executed_verification"], "unknown")
            finally:
                db.close()

    def test_live_path_logs_verified_used_different(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:

                def _make(sit, **kwargs):
                    return MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "a", "h")

                ctrl = LivePlayController(
                    db=db,
                    opponent_id="cpu",
                    make_call=_make,
                    parse_situation=parse_madden_situation,
                    learn_summary=lambda: "ok",
                    brand="Madden 27 Franchise",
                    cpu_only=True,
                    enable_execution_verify=True,
                )
                ctrl.start()
                ctrl.call_only("1&10 my 25", side="offense")
                out = ctrl.result_and_call(
                    outcome="+3",
                    sit_raw="2&7 my 28",
                    executed_status="used_different",
                    executed_formation="Gun Trips",
                    executed_play="Inside Zone",
                )
                self.assertTrue(out["ok"], out)
                self.assertEqual(out["logged"]["executed_status"], "identified")
                self.assertEqual(out["logged"]["executed_play"], "Inside Zone")
                snap_id = out["logged"]["ml_snap_id"]
                ml = db.conn.execute(
                    "SELECT * FROM ml_outcomes WHERE snap_id=?", (snap_id,)
                ).fetchone()
                self.assertEqual(ml["executed_verification"], "verified")
                self.assertEqual(ml["executed_status"], "identified")
            finally:
                db.close()


class HtmlDefaultTests(unittest.TestCase):
    def test_default_radio_is_used_recommended(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                ctrl = LivePlayController(
                    db=db,
                    opponent_id="cpu",
                    make_call=lambda sit, **k: MaddenCall(
                        "offense", "Gun Bunch", "Mesh", "No adj", "a", "h"
                    ),
                    parse_situation=parse_madden_situation,
                    learn_summary=lambda: "ok",
                    brand="Madden 27 Franchise",
                    cpu_only=True,
                    enable_execution_verify=True,
                )
                html = render_live_html(ctrl)
                self.assertIn('value="used_recommended" checked', html)
                self.assertNotIn('value="unknown" checked', html)
                self.assertIn("restoreExecDefaultAfterSubmit", html)
                self.assertIn("setExecSelection", html)
            finally:
                db.close()


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())
        self.meta = _seed_recovery_game(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_refuses_other_game_id(self) -> None:
        plan = execution_recovery.preview_recovery(self.db, game_id="franchise-other-001")
        self.assertFalse(plan["ok"])

    def test_dry_run_no_mutation(self) -> None:
        before = execution_recovery._counts(self.db)
        plan = execution_recovery.preview_recovery(self.db, attestation=ATTEST)
        self.assertTrue(plan["ok"])
        self.assertTrue(plan["dry_run"])
        self.assertGreaterEqual(plan["n_would_recover"], 4)
        self.assertEqual(plan["n_unlinked"], 1)
        self.assertEqual(plan["n_already_ok"], 1)
        after = execution_recovery._counts(self.db)
        self.assertEqual(before["n_verified_executions"], after["n_verified_executions"])

    def test_apply_requires_attestation(self) -> None:
        out = execution_recovery.apply_recovery(
            self.db, attestation="nope", apply=True
        )
        self.assertFalse(out.get("applied"))
        self.assertFalse(out["ok"])

    def test_apply_recovery_idempotent_and_scoped(self) -> None:
        before = execution_recovery._counts(self.db)
        self.assertEqual(before["n_verified_executions"], 1)  # the already-ok snap

        # Capture other game outcome
        other = self.db.conn.execute(
            "SELECT * FROM ml_outcomes WHERE snap_id=?",
            ("franchise-other-001-0001",),
        ).fetchone()
        other_status = other["executed_status"]

        backup = Path(self.tmp.name) / "backups" / "pre.db"
        first = execution_recovery.apply_recovery(
            self.db, attestation=ATTEST, backup_path=backup, apply=True
        )
        self.assertTrue(first["applied"])
        self.assertTrue(backup.is_file())
        self.assertGreaterEqual(first["delta_verified"], 4)

        # Snap 0010-style override recovered to Texas Y-Stutter Wheel
        wheel = self.db.conn.execute(
            "SELECT * FROM ml_outcomes WHERE snap_id=?",
            (f"{GAME}-0006",),
        ).fetchone()
        self.assertEqual(wheel["executed_play"], "Texas Y-Stutter Wheel")
        self.assertEqual(wheel["executed_verification"], "verified")
        payload = json.loads(wheel["outcome_json"])
        self.assertEqual(payload["result"], "sack")
        self.assertIn("recovery", payload)

        # Unlinked still unlinked
        unlinked = self.db.conn.execute(
            "SELECT count(*) FROM ml_outcomes WHERE snap_id=?",
            (f"{GAME}-0005",),
        ).fetchone()[0]
        self.assertEqual(unlinked, 0)

        # Other franchise untouched
        other2 = self.db.conn.execute(
            "SELECT * FROM ml_outcomes WHERE snap_id=?",
            ("franchise-other-001-0001",),
        ).fetchone()
        self.assertEqual(other2["executed_status"], other_status)

        after = execution_recovery._counts(self.db)
        self.assertEqual(after["n_verified_executions"], 5)  # 1 already + 4 recovered
        self.assertGreater(after["n_game_supervised_rows"], before["n_game_supervised_rows"])

        # Idempotent second apply
        second = execution_recovery.apply_recovery(
            self.db, attestation=ATTEST, backup_path=Path(self.tmp.name) / "backups" / "pre2.db", apply=True
        )
        self.assertTrue(second["applied"])
        self.assertEqual(second["n_would_recover"] if "n_would_recover" in second else len(second.get("recovered") or []), 0)
        # Preview after recovery should show all already_ok
        plan2 = execution_recovery.preview_recovery(self.db, attestation=ATTEST)
        self.assertEqual(plan2["n_would_recover"], 0)
        self.assertEqual(plan2["n_already_ok"], 5)

    def test_conflicting_verified_skipped(self) -> None:
        # Plant a conflicting verified outcome
        self.db.log_ml_outcome(
            snap_id=f"{GAME}-0001",
            game_id=GAME,
            decision_id=1,
            executed_status="identified",
            executed_formation="Gun Bunch",
            executed_play="NOT THE FINAL PLAY",
            executed_verification="verified",
            outcome={"result": "+7"},
            replace=True,
        )
        plan = execution_recovery.preview_recovery(self.db, attestation=ATTEST)
        skipped = [s for s in plan["skipped"] if s["snap_id"] == f"{GAME}-0001"]
        self.assertEqual(len(skipped), 1)
        self.assertIn("conflict", skipped[0]["reason"].lower())

    def test_training_counts_after_recovery(self) -> None:
        before_rows = dataset_mod.build_rows(db=self.db)
        before_sup = sum(
            1
            for r in before_rows
            if (r.get("eligibility") or dataset_mod.classify_eligibility(r))
            in dataset_mod.SUPERVISED_ELIGIBLE
            and str(r.get("snap_id") or "").startswith(f"{GAME}-")
        )
        execution_recovery.apply_recovery(
            self.db,
            attestation=ATTEST,
            backup_path=Path(self.tmp.name) / "b.db",
            apply=True,
        )
        after_rows = dataset_mod.build_rows(db=self.db)
        after_sup = sum(
            1
            for r in after_rows
            if (r.get("eligibility") or dataset_mod.classify_eligibility(r))
            in dataset_mod.SUPERVISED_ELIGIBLE
            and str(r.get("snap_id") or "").startswith(f"{GAME}-")
        )
        self.assertGreater(after_sup, before_sup)
        art = experimental_model.train_experimental(after_rows, side="offense")
        self.assertGreaterEqual(art.n_supervised, after_sup)
        dest = Path(self.tmp.name) / "artifact.json"
        experimental_model.save_artifact(art, dest)
        self.assertTrue(dest.is_file())


class UndoRetryRegressionTests(unittest.TestCase):
    def test_undo_and_used_recommended_default_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:

                def _make(sit, **kwargs):
                    return MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "a", "h")

                ctrl = LivePlayController(
                    db=db,
                    opponent_id="cpu",
                    make_call=_make,
                    parse_situation=parse_madden_situation,
                    learn_summary=lambda: "ok",
                    brand="Madden 27",
                    cpu_only=True,
                    enable_execution_verify=True,
                )
                ctrl.start()
                ctrl.call_only("1&10 my 25")
                r = ctrl.result_and_call(
                    outcome="+5",
                    sit_raw="2&5 my 30",
                    executed_status="used_recommended",
                )
                self.assertEqual(r["logged"]["executed_status"], "identified")
                snap = r["logged"]["ml_snap_id"]
                undo = ctrl.undo_last()
                self.assertTrue(undo["ok"])
                self.assertEqual(
                    db.conn.execute(
                        "SELECT count(*) FROM ml_outcomes WHERE snap_id=?", (snap,)
                    ).fetchone()[0],
                    0,
                )
                r2 = ctrl.result_and_call(
                    outcome="+3",
                    sit_raw="2&7 my 28",
                    executed_status="used_recommended",
                )
                self.assertTrue(r2["ok"])
                # Experimental mode still settable / heuristic default intact
                inference.set_mode(db, CoachingMode.EXPERIMENTAL)
                self.assertIs(inference.resolve_mode(db), CoachingMode.EXPERIMENTAL)
                inference.set_mode(db, CoachingMode.HEURISTIC)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
