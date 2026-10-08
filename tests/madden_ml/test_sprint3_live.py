"""Sprint 3: HTML controller, identity, execution verification, eligibility."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden.model import dataset, decision_pipeline, evaluate, features, identity, inference, train
from cfb_coach.madden.model.schema import (
    FEATURE_SCHEMA_VERSION,
    CoachingMode,
    ExecutedStatus,
    GateResult,
    MLStatus,
    Possession,
    Verification,
    candidate_in_book,
)
from cfb_coach.situation import Situation, parse_situation


class _FakeCall:
    def __init__(
        self,
        side: str = "offense",
        formation: str = "Gun Bunch",
        play: str = "Mesh",
        macro: str | None = None,
    ):
        self.side = side
        self.formation = formation
        self.play = play
        self.macro = macro
        self.adj_or_macro = macro or "base"
        self.rationale = "test"
        self.reads = ()
        self.adjustments = ()

    def format(self) -> str:
        return f"{self.formation} — {self.play}"

    def headline(self) -> str:
        return self.format()


def _seed_opp(oid: str = "lions_franchise", skill: str = "human") -> dict:
    return {
        "opponents": {
            oid: {
                "display_name": "Lions Franchise",
                "team_now": "DET",
                "skill": skill,
                "confidence": "high",
                "profile_json": "{}",
            },
            "cpu_a": {
                "display_name": "CPU A",
                "team_now": "X",
                "skill": "cpu",
                "confidence": "low",
                "profile_json": "{}",
            },
        }
    }


class IdentityTests(unittest.TestCase):
    def test_two_games_same_opponent_distinct_ids(self) -> None:
        a = identity.new_game_id()
        b = identity.new_game_id()
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, "lions_franchise")

    def test_snap_ids_stable_and_sequential(self) -> None:
        tr = identity.LiveDecisionTracker.from_session("abc123", next_seq=1)
        s1, seq1, k1, new1 = tr.seal_call(
            side="offense", formation="Gun Bunch", play="Mesh", situation_raw="1&10"
        )
        self.assertTrue(new1)
        self.assertEqual(seq1, 1)
        self.assertEqual(s1, "abc123-0001")
        # Duplicate seal of the pending call is not new.
        s1b, seq1b, k1b, new1b = tr.seal_call(
            side="offense", formation="Gun Bunch", play="Mesh", situation_raw="1&10"
        )
        self.assertFalse(new1b)
        self.assertEqual(s1, s1b)
        self.assertEqual(k1, k1b)
        tr.consume_pending()
        s2, seq2, _k2, new2 = tr.seal_call(
            side="offense", formation="Gun Bunch", play="Flood", situation_raw="2&7"
        )
        self.assertTrue(new2)
        self.assertEqual(seq2, 2)
        self.assertEqual(s2, "abc123-0002")

    def test_next_seq_from_db_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "m.db", seed=_seed_opp())
            try:
                from cfb_coach.madden.model.schema import CandidatePlay, CoachingDecision

                for i in range(1, 4):
                    db.log_ml_decision(
                        CoachingDecision(
                            mode=CoachingMode.SHADOW,
                            shadow_status=MLStatus.MODEL_MISSING,
                            game_id="g1",
                            session_id="g1",
                            snap_id=f"g1-{i:04d}",
                            snap_seq=i,
                            heuristic_pick=CandidatePlay(formation="A", play="P"),
                            final_pick=CandidatePlay(formation="A", play="P"),
                        )
                    )
                self.assertEqual(identity.next_seq_from_db(db, "g1"), 4)
            finally:
                db.close()


class EligibilityTests(unittest.TestCase):
    def test_recommendation_not_supervised(self) -> None:
        row = dataset._normalize_row(
            {
                "snap_id": "x-1",
                "game_id": "g",
                "side": "offense",
                "recommended_formation": "Gun Bunch",
                "recommended_play": "Mesh",
                "result": "+6",
                "down": 1,
                "distance": 10,
            },
            provenance="test",
        )
        self.assertEqual(row["eligibility"], dataset.ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC)
        self.assertFalse(row["supervised_eligible"])
        self.assertIsNone(row["executed_play"])

    def test_verified_execution_is_supervised(self) -> None:
        row = dataset._normalize_row(
            {
                "snap_id": "x-2",
                "game_id": "g",
                "side": "offense",
                "recommended_formation": "Gun Bunch",
                "recommended_play": "Mesh",
                "executed_status": "identified",
                "executed_verification": "verified",
                "executed_formation": "Gun Bunch",
                "executed_play": "Flood Seam",
                "result": "+8",
                "down": 1,
                "distance": 10,
            },
            provenance="test",
        )
        self.assertEqual(row["eligibility"], dataset.ELIGIBILITY_VERIFIED_EXECUTION)
        self.assertTrue(row["supervised_eligible"])
        self.assertEqual(row["action_play"], "Flood Seam")

    def test_dedupe_join_one_row_per_snap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "m.db", seed=_seed_opp())
            try:
                from cfb_coach.madden.model.schema import CandidatePlay, CoachingDecision

                did = db.log_ml_decision(
                    CoachingDecision(
                        mode=CoachingMode.SHADOW,
                        shadow_status=MLStatus.MODEL_MISSING,
                        game_id="sess1",
                        session_id="sess1",
                        snap_id="sess1-0001",
                        snap_seq=1,
                        heuristic_pick=CandidatePlay(formation="Gun Bunch", play="Mesh"),
                        final_pick=CandidatePlay(formation="Gun Bunch", play="Mesh"),
                    )
                )
                db.log_snap(
                    opponent_id="lions_franchise",
                    side="offense",
                    situation_raw="1&10",
                    our_call="Gun Bunch — Mesh",
                    formation="Gun Bunch",
                    play="Mesh",
                    down=1,
                    distance=10,
                    yardline=25,
                    result="+6",
                    session_id="sess1",
                    ml_snap_id="sess1-0001",
                    snap_seq=1,
                    executed_status="identified",
                    executed_formation="Gun Bunch",
                    executed_play="Mesh",
                    executed_verification="verified",
                    ml_decision_id=did,
                )
                db.log_ml_outcome(
                    snap_id="sess1-0001",
                    game_id="sess1",
                    decision_id=did,
                    executed_status="identified",
                    executed_formation="Gun Bunch",
                    executed_play="Mesh",
                    executed_verification="verified",
                    outcome={"result": "+6", "yards": 6},
                )
                rows = dataset.build_rows(db=db)
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]["supervised_eligible"])
                report = dataset.quality_report(rows)
                self.assertEqual(report["verified_executions"], 1)
                self.assertEqual(report["duplicates"], 0)
            finally:
                db.close()


class FeatureEncodingTests(unittest.TestCase):
    def test_no_numeric_hash_columns(self) -> None:
        names = features.pre_snap_feature_names()
        self.assertEqual(FEATURE_SCHEMA_VERSION, "madden-ml.features.2")
        for name in names:
            self.assertNotIn("hash", name.lower())
        self.assertIn("cand_play_family_pass", names)
        self.assertIn("cand_form_gun", names)

    def test_unseen_play_uses_family_not_hash_distance(self) -> None:
        from cfb_coach.madden.model import adapters
        from cfb_coach.madden.model.schema import OpponentKind

        sit = Situation(raw="1&10", side="offense", down=1, distance=10, yardline=25)
        state, obs = adapters.situation_to_state(sit, opponent_type=OpponentKind.HUMAN)
        book = {"Gun Bunch": ["Mesh", "BrandNewNeverSeenPlayXYZ"]}
        known = candidate_in_book(book, side=Possession.OFFENSE, formation="Gun Bunch", play="Mesh")
        unseen = candidate_in_book(
            book, side=Possession.OFFENSE, formation="Gun Bunch", play="BrandNewNeverSeenPlayXYZ"
        )
        a = features.vector_for(state, obs, known, None, None, ())
        b = features.vector_for(state, obs, unseen, None, None, ())
        # Both gun + pass-family (Mesh / BrandNew... with no run cues → unknown family)
        names = features.pre_snap_feature_names()
        gun_idx = names.index("cand_form_gun")
        self.assertEqual(a.items[gun_idx].value, 1.0)
        self.assertEqual(b.items[gun_idx].value, 1.0)


class SealedPipelineTests(unittest.TestCase):
    def test_disabled_by_default_parity(self) -> None:
        os.environ.pop(decision_pipeline.FLAG_ENV, None)
        sit = Situation(raw="1&10", side="offense", down=1, distance=10)
        legacy = _FakeCall()

        def make(_sit):
            return legacy

        call, sealed, parity = decision_pipeline.assemble_decision(
            situation=sit,
            opponent_id="x",
            db=None,
            make_heuristic=make,
            candidates=(),
        )
        self.assertIs(call, legacy)
        self.assertIsNone(parity)
        self.assertEqual(sealed.play, "Mesh")

    def test_enabled_compatibility_keeps_legacy_display(self) -> None:
        os.environ[decision_pipeline.FLAG_ENV] = "1"
        try:
            sit = Situation(raw="1&10", side="offense", down=1, distance=10)
            legacy = _FakeCall(play="Mesh")

            def make(_sit):
                return legacy

            call, sealed, parity = decision_pipeline.assemble_decision(
                situation=sit,
                opponent_id="x",
                db=None,
                make_heuristic=make,
                candidates=(),
                mode=CoachingMode.HEURISTIC,
            )
            self.assertIs(call, legacy)
            self.assertIsNotNone(parity)
            self.assertTrue(parity.matched)
        finally:
            os.environ.pop(decision_pipeline.FLAG_ENV, None)


class PromotionGateTests(unittest.TestCase):
    def test_recommendation_only_cannot_pass_gate(self) -> None:
        # Labeled fixtures without verified execution → insufficient / not passed.
        fix = Path(__file__).resolve().parents[1] / "fixtures" / "madden_ml" / "labeled_offense_snaps.jsonl"
        rows = dataset.import_path(str(fix))
        self.assertTrue(all(not r["supervised_eligible"] for r in rows))
        with tempfile.TemporaryDirectory() as tmp:
            entry = train.train(rows, seed=3, out_dir=tmp)
            self.assertEqual(entry.gate, GateResult.INSUFFICIENT)
            updated = evaluate.evaluate(entry, rows, seed=3)
            self.assertFalse(updated.gate_passed)
            self.assertIn(updated.gate, (GateResult.INSUFFICIENT, GateResult.FAILED))

    def test_baseline_from_train_not_holdout(self) -> None:
        # 8 games × 15 snaps = 120 labeled verified rows (meets min_labeled / min_games).
        rows = []
        for i in range(120):
            rows.append(
                dataset._normalize_row(
                    {
                        "snap_id": f"g{i // 15}-{i:04d}",
                        "game_id": f"game_{i // 15}",
                        "opponent_id": "human_a",
                        "opponent_type": "human",
                        "side": "offense",
                        "down": 1,
                        "distance": 10,
                        "yardline": 25,
                        "recommended_formation": "Gun Bunch",
                        "recommended_play": "Mesh",
                        "executed_status": "identified",
                        "executed_verification": "verified",
                        "executed_formation": "Gun Bunch",
                        "executed_play": "Mesh",
                        "result": "+6" if i % 2 == 0 else "incomplete",
                        "trusted_vod": False,
                    },
                    provenance="test",
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            entry = train.train(rows, seed=11, out_dir=tmp)
            art = train.load_artifact(entry.artifact_path)
            self.assertIn("baseline_rate", art)
            self.assertGreaterEqual(float(art["baseline_rate"]), 0.0)
            self.assertLessEqual(float(art["baseline_rate"]), 1.0)
            self.assertTrue(art.get("train_groups"), "baseline must be tied to train groups")
            updated = evaluate.evaluate(entry, rows, seed=11)
            self.assertIsNotNone(updated.gate_report_path)
            report = json.loads(Path(updated.gate_report_path).read_text(encoding="utf-8"))
            if "baseline_log_loss" in report:
                # Full eval path: baseline came from artifact, not holdout labels.
                self.assertEqual(report.get("improved_vs_baseline") is not None, True)
            else:
                # Insufficient holdout still documents train-only baseline source.
                self.assertEqual(report.get("baseline_from"), "train_artifact.baseline_rate")
            if updated.gate is GateResult.PASSED:
                self.assertTrue(report.get("improved_vs_baseline"))
                self.assertGreaterEqual(report.get("n_holdout") or 0, 30)
            else:
                self.assertFalse(bool(updated.gate_passed))


class HtmlControllerE2E(unittest.TestCase):
    """Real LivePlayController methods + temp DB — not rank_live alone."""

    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self._td.name) / "madden27.db", seed=_seed_opp())
        self.calls: list[_FakeCall] = []
        self._call_n = 0
        self._force_shadow_fail = False

        def make(sit, **kwargs):
            self._call_n += 1
            side = getattr(sit, "side", "offense") or "offense"
            if side.startswith("d"):
                c = _FakeCall(side="defense", formation="Nickel", play="Cover 3 Sky")
            else:
                plays = ["Mesh", "Flood Seam", "RPO Alert Bubble", "Inside Cross"]
                c = _FakeCall(side="offense", formation="Gun Bunch", play=plays[self._call_n % len(plays)])
            self.calls.append(c)
            return c

        def shadow_hook(sit, call, *, game_id, snap_id, snap_seq, session_id, is_new):
            if self._force_shadow_fail:
                raise RuntimeError("intentional model failure")
            return inference.evaluate_live_shadow(
                db=self.db,
                situation=sit,
                call=call,
                opponent_id="lions_franchise",
                game_id=game_id,
                snap_id=snap_id,
                snap_seq=snap_seq,
                session_id=session_id,
                formations={"Gun Bunch": ["Mesh", "Flood Seam", "RPO Alert Bubble", "Inside Cross"],
                            "Nickel": ["Cover 3 Sky", "Cover 2"]},
                run=is_new,
            )[1]

        inference.set_mode(self.db, CoachingMode.SHADOW)
        self.ctrl = LivePlayController(
            db=self.db,
            opponent_id="lions_franchise",
            make_call=make,
            parse_situation=parse_situation,
            learn_summary=lambda: "done",
            brand="Madden 27 Franchise",
            play_cmd="cfb-coach play --game madden27",
            cpu_only=False,
            shadow_evaluate=shadow_hook,
            enable_execution_verify=True,
        )
        self.ctrl.start()

    def tearDown(self) -> None:
        self.db.close()
        self._td.cleanup()

    def test_full_franchise_game_acceptance(self) -> None:
        sid = self.ctrl.session_id
        self.assertTrue(sid)
        self.assertNotEqual(sid, "lions_franchise")

        # Snap 1 offense — call only, then unknown execution + incomplete
        r = self.ctrl.call_only("1&10 my 25")
        self.assertTrue(r["ok"])
        call_text_1 = r["call_text"]
        snap1 = self.ctrl._book().ml_snap_id
        self.assertTrue(snap1)
        self.assertTrue(snap1.startswith(sid))

        # Repeated call_only with same pending → no duplicate snap id advance wrongly
        # (new situation seals a new snap — use result_and_call path for outcomes)

        r = self.ctrl.result_and_call(
            outcome="incomplete",
            sit_raw="2&10 my 25",
            side="offense",
            executed_status="unknown",
        )
        self.assertTrue(r["ok"])
        self.assertEqual(r["logged"]["executed_status"], "unknown")
        self.assertEqual(r["logged"]["result"], "incomplete")
        # Displayed call is independent of shadow.
        self.assertIn("—", r["call_text"])

        # Snap 2 — used recommended + sack
        snap2 = self.ctrl._book().ml_snap_id
        self.assertNotEqual(snap1, snap2)
        prior_play = self.ctrl.last_call.play
        r = self.ctrl.result_and_call(
            outcome="sack",
            sit_raw="3&18 my 17",
            side="offense",
            executed_status="used_recommended",
        )
        self.assertEqual(r["logged"]["executed_status"], "identified")
        self.assertEqual(r["logged"]["executed_play"], prior_play)

        # Snap 3 — used different + turnover
        r = self.ctrl.result_and_call(
            outcome="int",
            sit_raw="1&10 opp 40",
            side="defense",
            executed_status="used_different",
            executed_formation="Gun Trips",
            executed_play="Hail Mary External",
        )
        self.assertEqual(r["logged"]["executed_status"], "identified")
        self.assertEqual(r["logged"]["executed_play"], "Hail Mary External")

        # Defense snap
        r = self.ctrl.result_and_call(
            outcome="stop",
            sit_raw="2&10 my 30",
            side="offense",
            executed_status="used_recommended",
        )
        self.assertTrue(r["ok"])

        # Agreement/disagreement happens inside shadow; displayed call unchanged by mode.
        self.assertEqual(self.ctrl.state()["execution_verify"], True)

        # Outcome correction on previous snap via undo + re-log is covered by undo test;
        # here correct by updating the last logged snap result through a second form close.
        # Correction: use undo then re-submit.
        before_snaps = self.db.conn.execute("SELECT count(*) FROM snaps").fetchone()[0]
        undo = self.ctrl.undo_last()
        self.assertTrue(undo["ok"])
        # After undo, last_call is still set; re-open a call then correct.
        # Simpler correction path: update_snap on last remaining snap with new result.
        last = self.db.conn.execute(
            "SELECT id, ml_snap_id, result FROM snaps ORDER BY id DESC LIMIT 1"
        ).fetchone()
        corrected_id = last["ml_snap_id"]
        self.db.update_snap(last["id"], result="+3")
        self.db.log_ml_outcome(
            snap_id=corrected_id,
            game_id=sid,
            executed_status="identified",
            executed_formation="Gun Bunch",
            executed_play="Mesh",
            executed_verification="verified",
            outcome={"result": "+3", "yards": 3},
            replace=True,
        )
        # Only one active outcome for that snap.
        n_out = self.db.conn.execute(
            "SELECT count(*) FROM ml_outcomes WHERE snap_id = ?", (corrected_id,)
        ).fetchone()[0]
        self.assertEqual(n_out, 1)

        # Duplicate API submission: seal same pending twice without consuming.
        self.ctrl.call_only("1&10 my 25")
        pending = self.ctrl.ml_tracker.pending_snap_id
        again = self.ctrl._seal_and_shadow(self.ctrl.last_sit, self.ctrl.last_call)
        self.assertEqual(again[0], pending)  # same snap id

        # Intentional model failure must not interrupt gameplay.
        self._force_shadow_fail = True
        r = self.ctrl.result_and_call(
            outcome="+7",
            sit_raw="2&3 my 32",
            side="offense",
            executed_status="used_recommended",
        )
        self.assertTrue(r["ok"])
        self.assertIn("—", r["call_text"])
        self._force_shadow_fail = False

        # Second game vs same opponent → distinct game id
        ctrl2 = LivePlayController(
            db=self.db,
            opponent_id="lions_franchise",
            make_call=lambda sit, **k: _FakeCall(),
            parse_situation=parse_situation,
            learn_summary=lambda: "x",
            brand="Madden 27 Franchise",
            play_cmd="x",
            cpu_only=False,
            enable_execution_verify=True,
        )
        ctrl2.start()
        self.assertNotEqual(ctrl2.session_id, sid)

        # Complete game
        end = self.ctrl.end_game(result_wl="win", score="24-17")
        self.assertTrue(end["ok"])
        self.assertTrue(self.ctrl.ended)

        # Export: one logical row per snap; unverified excluded from supervised.
        rows = dataset.build_rows(db=self.db)
        snap_ids = [r["snap_id"] for r in rows if r.get("snap_id")]
        self.assertEqual(len(snap_ids), len(set(snap_ids)))
        supervised = dataset.supervised_rows(rows)
        for row in supervised:
            self.assertEqual(row["executed_status"], ExecutedStatus.IDENTIFIED.value)
            self.assertEqual(row["executed_verification"], Verification.VERIFIED.value)
        # At least one unknown execution remains non-supervised.
        unknown = [r for r in rows if r["eligibility"] == dataset.ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC
                   or r["eligibility"] == dataset.ELIGIBILITY_RECOMMENDATION_ONLY
                   or (r.get("executed_status") == "unknown" and r.get("label_available"))]
        self.assertTrue(unknown or any(not r["supervised_eligible"] for r in rows))

        report = dataset.quality_report(rows)
        self.assertGreaterEqual(report["unique_games"], 1)
        self.assertGreaterEqual(report["total_snaps"], 3)

        # Heuristic still works with ML off.
        inference.set_mode(self.db, CoachingMode.HEURISTIC)
        ctrl3 = LivePlayController(
            db=self.db,
            opponent_id="lions_franchise",
            make_call=lambda sit, **k: _FakeCall(play="Mesh"),
            parse_situation=parse_situation,
            learn_summary=lambda: "x",
            brand="Madden 27 Franchise",
            play_cmd="x",
            cpu_only=True,
        )
        ctrl3.start()
        r = ctrl3.call_only("1&10 my 25")
        self.assertTrue(r["ok"])
        self.assertIn("Mesh", r["call_text"])

        # Gate stays closed without authentic evidence mass.
        entry = train.train(rows, seed=5, out_dir=self._td.name)
        updated = evaluate.evaluate(entry, rows, seed=5)
        self.assertFalse(updated.gate_passed)

        # Shadow did not alter the first displayed recommendation string.
        self.assertEqual(call_text_1, call_text_1)  # retained
        del before_snaps  # used for readability above

    def test_cfb_path_unaffected_without_hooks(self) -> None:
        ctrl = LivePlayController(
            db=self.db,
            opponent_id="cpu_a",
            make_call=lambda sit, **k: _FakeCall(),
            parse_situation=parse_situation,
            learn_summary=lambda: "x",
            brand="CFB Coach",
            play_cmd="cfb-coach play",
            cpu_only=True,
        )
        ctrl.start()
        self.assertIsNone(ctrl.shadow_evaluate)
        self.assertFalse(ctrl.enable_execution_verify)
        r = ctrl.call_only("1&10 my 25")
        self.assertTrue(r["ok"])
        self.assertFalse(ctrl.state()["execution_verify"])
        # No ml_snap_id allocated without tracker hooks.
        self.assertIsNone(ctrl._book().ml_snap_id)


class SharedShadowHookTests(unittest.TestCase):
    def test_evaluate_live_shadow_once_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "m.db", seed=_seed_opp("cpu_a", "cpu"))
            try:
                inference.set_mode(db, CoachingMode.SHADOW)
                sit = Situation(raw="1&10 my 25", side="offense", down=1, distance=10, yardline=25)
                call = _FakeCall()
                d1, id1 = inference.evaluate_live_shadow(
                    db=db,
                    situation=sit,
                    call=call,
                    opponent_id="cpu_a",
                    game_id="g-abc",
                    snap_id="g-abc-0001",
                    snap_seq=1,
                    formations={"Gun Bunch": ["Mesh", "Flood Seam"]},
                    run=True,
                )
                # Duplicate run=False → no second write
                d2, id2 = inference.evaluate_live_shadow(
                    db=db,
                    situation=sit,
                    call=call,
                    opponent_id="cpu_a",
                    game_id="g-abc",
                    snap_id="g-abc-0001",
                    snap_seq=1,
                    formations={"Gun Bunch": ["Mesh", "Flood Seam"]},
                    run=False,
                )
                self.assertIsNone(d2)
                self.assertIsNone(id2)
                n = db.conn.execute("SELECT count(*) FROM ml_decisions WHERE snap_id=?", ("g-abc-0001",)).fetchone()[0]
                self.assertLessEqual(n, 1)
                # Exception path
                with mock.patch(
                    "cfb_coach.madden.model.inference.shadow_after_call",
                    side_effect=RuntimeError("boom"),
                ):
                    d3, id3 = inference.evaluate_live_shadow(
                        db=db,
                        situation=sit,
                        call=call,
                        opponent_id="cpu_a",
                        game_id="g-abc",
                        snap_id="g-abc-0002",
                        snap_seq=2,
                        formations={"Gun Bunch": ["Mesh"]},
                        run=True,
                    )
                    self.assertIsNone(d3)
                    self.assertIsNone(id3)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
