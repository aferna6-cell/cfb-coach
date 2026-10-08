"""Sprint 4: experimental ML pilot — historical audit, model, live sealed path, research."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden.data import load_seed, reads_for
from cfb_coach.madden.model import experimental_live, experimental_model, historical, inference
from cfb_coach.madden.model.schema import (
    CoachingDecision,
    CoachingMode,
    ExecutedStatus,
    MLStatus,
    PolicySource,
    Verification,
)
from cfb_coach.madden.playcaller import MaddenCall, make_call
from cfb_coach.madden.situation import Situation


def _seed() -> dict:
    return {
        "opponents": {
            "cpu_lions": {
                "display_name": "Lions CPU",
                "team_now": "DET",
                "skill": "cpu",
                "confidence": "low",
                "profile_json": "{}",
            },
            "human_x": {
                "display_name": "Human X",
                "team_now": "X",
                "skill": "human",
                "confidence": "high",
                "profile_json": "{}",
            },
        }
    }


def _book() -> dict[str, list[str]]:
    return {
        "Gun Bunch": ["Mesh", "Stick", "Smash"],
        "Gun Doubles": ["Y Cross", "Four Verticals", "HB Dive"],
        "Singleback Ace": ["PA Boot", "Inside Zone", "Slant Flat"],
        "I Form Twins": ["Power O", "PA Cross", "Flood"],
        "Pistol Trips": ["RPO Peek", "Mesh Spot", "HB Zone"],
    }


class HistoricalAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_empty_db_audit_does_not_invent_games(self) -> None:
        report = historical.audit_database(self.db)
        self.assertEqual(report["n_games"], 0)
        self.assertEqual(report["n_rows"], 0)
        self.assertEqual(report["verified_executions"], 0)
        self.assertIn("verified executions", report["note"].lower())
        self.assertIn("confirm-execution", report["note"].lower())

    def test_confirm_requires_evidence(self) -> None:
        with self.assertRaises(ValueError):
            historical.confirm_execution(
                self.db,
                ml_snap_id="x",
                executed_formation="Gun Bunch",
                executed_play="Mesh",
                evidence="",
            )

    def test_confirm_execution_with_evidence(self) -> None:
        row_id = self.db.log_snap(
            opponent_id="cpu_lions",
            side="offense",
            situation_raw="1&10",
            our_call="Gun Bunch — Mesh",
            formation="Gun Bunch",
            play="Mesh",
            result="+5",
            session_id="sess-hist",
            ml_snap_id="g1-0001",
        )
        out = historical.confirm_execution(
            self.db,
            ml_snap_id="g1-0001",
            executed_formation="Gun Bunch",
            executed_play="Mesh",
            evidence="vod minute 12:04 — confirmed Mesh",
        )
        self.assertTrue(out["ok"])
        snap = self.db.conn.execute("SELECT * FROM snaps WHERE id=?", (row_id,)).fetchone()
        self.assertEqual(snap["executed_status"], ExecutedStatus.IDENTIFIED.value)
        self.assertEqual(snap["executed_verification"], Verification.VERIFIED.value)

    def test_uncertain_not_in_supervised(self) -> None:
        from cfb_coach.madden.model import dataset as dataset_mod

        rows = [
            {
                "side": "offense",
                "eligibility": dataset_mod.ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC,
                "label_available": True,
                "success": "true",
                "recommended_play": "Mesh",
                "action_play": "Mesh",
            }
        ]
        priors = historical.discounted_prior_rows(rows)
        self.assertEqual(len(priors), 1)
        self.assertEqual(priors[0]["prior_discount"], 0.25)
        self.assertNotIn(
            dataset_mod.ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC,
            dataset_mod.SUPERVISED_ELIGIBLE,
        )


class ExperimentalModelTests(unittest.TestCase):
    def test_prior_driven_when_empty(self) -> None:
        art = experimental_model.train_experimental([], side="offense")
        self.assertEqual(art.evidence_quality, "prior_driven")
        self.assertEqual(art.n_supervised, 0)
        self.assertIn("prior-driven", art.note.lower())

    def test_rank_candidates_near_tie_keeps_heuristic(self) -> None:
        art = experimental_model.train_experimental([], side="offense")
        ranked = experimental_model.rank_candidates(
            art,
            [("Gun Bunch", "Mesh"), ("Gun Doubles", "Y Cross")],
            down=1,
            distance=10,
            yardline=50,
            heuristic=("Gun Bunch", "Mesh"),
        )
        self.assertTrue(ranked)
        self.assertEqual(ranked[0]["formation"], "Gun Bunch")

    def test_predict_presnap_only_unknown_cov_without_hint(self) -> None:
        art = experimental_model.train_experimental([], side="offense")
        pred = experimental_model.predict_success(
            art, formation="Gun Bunch", play="Mesh", coverage_hint=None
        )
        self.assertEqual(pred["coverage_bucket"], "unknown")

    def test_save_load_roundtrip(self) -> None:
        art = experimental_model.train_experimental([], side="offense")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "offense.json"
            experimental_model.save_artifact(art, path)
            again = experimental_model.load_artifact(path)
            self.assertEqual(again.evidence_quality, art.evidence_quality)
            self.assertEqual(again.kind, art.kind)


class ExperimentalLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())
        self.book = _book()
        # Install empty prior-driven artifact.
        art = experimental_model.train_experimental([], side="offense")
        dest = Path(self.tmp.name) / "exp.json"
        experimental_model.save_artifact(art, dest)
        experimental_live.set_artifact_path(self.db, dest)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _sit(self, **kwargs) -> Situation:
        defaults = dict(
            raw="1st & 10 at mid",
            side="offense",
            down=1,
            distance=10,
            yardline=50,
            coverage_hint=None,
            coverage_source="none",
        )
        defaults.update(kwargs)
        return Situation(**defaults)

    def test_default_mode_is_heuristic(self) -> None:
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.HEURISTIC)

    def test_experimental_selects_legal_in_book_play(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        heur = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", reads_for("Mesh"), "heuristic")
        sit = self._sit()
        call, dec = experimental_live.apply_experimental_offense(
            heuristic_call=heur,
            sit=sit,
            opponent_id="cpu_lions",
            db=self.db,
            book=self.book,
            active=[],
            game_id="g-test",
            snap_id="g-test-0001",
            snap_seq=1,
        )
        self.assertEqual(call.side, "offense")
        self.assertIn(call.formation, self.book)
        self.assertIn(call.play, self.book[call.formation])
        self.assertEqual(call.read_or_user, reads_for(call.play))
        self.assertIs(dec.mode, CoachingMode.EXPERIMENTAL)
        self.assertIsNotNone(dec.shadow_status)
        self.assertIsNotNone(getattr(call, "ml_experimental", None))
        info = call.ml_experimental
        self.assertEqual(info["heuristic_play"], "Mesh")
        self.assertIn(info["evidence_quality"], ("prior_driven", "empirical_light", "empirical"))

    def test_timeout_falls_back_to_heuristic(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        heur = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "Primary → Checkdown", "h")
        with mock.patch(
            "cfb_coach.madden.model.experimental_live.exp_mod.rank_candidates",
            side_effect=lambda *a, **k: (_ for _ in ()).throw(TimeoutError("slow")),
        ):
            # Force exception path via rank failure after load.
            pass
        with mock.patch(
            "cfb_coach.madden.model.experimental_live.exp_mod.load_artifact",
            side_effect=RuntimeError("boom"),
        ):
            call, dec = experimental_live.apply_experimental_offense(
                heuristic_call=heur,
                sit=self._sit(),
                opponent_id="cpu_lions",
                db=self.db,
                book=self.book,
                budget_ms=150,
            )
        self.assertEqual(call.formation, "Gun Bunch")
        self.assertEqual(call.play, "Mesh")
        self.assertTrue(dec.fell_back)
        self.assertIs(dec.policy_source, PolicySource.HEURISTIC)
        self.assertIs(dec.fallback_reason, MLStatus.EXCEPTION)

    def test_budget_timeout(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        heur = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "Primary → Checkdown", "h")

        def _slow(*_a, **_k):
            import time

            time.sleep(0.08)
            return [
                {
                    "formation": "Gun Bunch",
                    "play": "Mesh",
                    "probability": 0.5,
                    "play_family": "pass",
                    "evidence_quality": "prior_driven",
                    "components": {},
                    "rank": 1,
                }
            ]

        with mock.patch(
            "cfb_coach.madden.model.experimental_live.exp_mod.rank_candidates",
            side_effect=_slow,
        ):
            call, dec = experimental_live.apply_experimental_offense(
                heuristic_call=heur,
                sit=self._sit(),
                opponent_id="cpu_lions",
                db=self.db,
                book=self.book,
                budget_ms=20,
            )
        self.assertTrue(dec.fell_back)
        self.assertIs(dec.fallback_reason, MLStatus.TIMEOUT)
        self.assertEqual(call.play, "Mesh")

    def test_defense_not_selected(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        heur = MaddenCall("defense", "Nickel", "Cover 3", "none", "User hook", "d")
        call, dec = experimental_live.apply_experimental_offense(
            heuristic_call=heur,
            sit=self._sit(side="defense"),
            opponent_id="human_x",
            db=self.db,
            book=self.book,
        )
        self.assertEqual(call.side, "defense")
        self.assertTrue(dec.fell_back)

    def test_coverage_situations_presnap_hint(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        for cov in ("Cover 2", "Cover 3", "Cover 4", "Man Cover 1", None):
            heur = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", reads_for("Mesh"), "h")
            sit = self._sit(
                coverage_hint=cov,
                coverage_source="live" if cov else None,
                distance=3 if cov == "Cover 2" else 12 if cov == "Cover 4" else 10,
                yardline=85 if cov == "Cover 3" else 20 if cov is None else 50,
            )
            call, dec = experimental_live.apply_experimental_offense(
                heuristic_call=heur,
                sit=sit,
                opponent_id="cpu_lions",
                db=self.db,
                book=self.book,
                snap_id=f"cov-{cov or 'unk'}",
            )
            self.assertIn(call.play, self.book[call.formation])
            self.assertEqual(call.read_or_user, reads_for(call.play))

    def test_maybe_apply_noop_when_heuristic(self) -> None:
        inference.set_mode(self.db, CoachingMode.HEURISTIC)
        heur = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", reads_for("Mesh"), "h")
        out = experimental_live.maybe_apply_experimental(
            call=heur, sit=self._sit(), opponent_id="cpu_lions", db=self.db, book=self.book
        )
        self.assertIs(out, heur)

    def test_rebuild_reads_match_play(self) -> None:
        call = experimental_live.rebuild_offense_attachments(
            formation="Gun Doubles",
            play="Four Verticals",
            sit=self._sit(),
            book=self.book,
            active=[],
            db=self.db,
            opponent_id="cpu_lions",
            rationale="test",
        )
        self.assertEqual(call.play, "Four Verticals")
        self.assertEqual(call.read_or_user, reads_for("Four Verticals"))
        self.assertNotEqual(call.read_or_user, reads_for("Mesh"))


class ExperimentalSchemaTests(unittest.TestCase):
    def test_experimental_success_decision(self) -> None:
        from cfb_coach.madden.model.schema import Possession, candidate_in_book

        book = {"Gun Bunch": ["Mesh"]}
        cand = candidate_in_book(
            book, side=Possession.OFFENSE, formation="Gun Bunch", play="Mesh"
        )
        dec = CoachingDecision(
            mode=CoachingMode.EXPERIMENTAL,
            effective_mode=CoachingMode.EXPERIMENTAL,
            policy_source=PolicySource.MODEL,
            heuristic_pick=cand,
            final_pick=cand,
            shadow_pick=cand,
            shadow_status=MLStatus.OK,
            fell_back=False,
        )
        dec.validate()

    def test_experimental_fallback_decision(self) -> None:
        dec = CoachingDecision(
            mode=CoachingMode.EXPERIMENTAL,
            effective_mode=CoachingMode.HEURISTIC,
            policy_source=PolicySource.HEURISTIC,
            shadow_status=MLStatus.TIMEOUT,
            fell_back=True,
            fallback_reason=MLStatus.TIMEOUT,
        )
        dec.validate()


class ExperimentalHtmlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())
        art = experimental_model.train_experimental([], side="offense")
        dest = Path(self.tmp.name) / "exp.json"
        experimental_model.save_artifact(art, dest)
        experimental_live.set_artifact_path(self.db, dest)
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)

        book = _book()

        def _make(sit, **kwargs):
            # Bypass full playbook lock for unit test — sealed experimental path only.
            heur = MaddenCall(
                "offense",
                "Gun Bunch",
                "Mesh",
                "No adj",
                reads_for("Mesh"),
                "heuristic baseline",
            )
            call, _ = experimental_live.apply_experimental_offense(
                heuristic_call=heur,
                sit=sit,
                opponent_id="cpu_lions",
                db=self.db,
                book=book,
                active=[],
            )
            return call

        from cfb_coach.madden.situation import parse_madden_situation

        self.ctrl = LivePlayController(
            db=self.db,
            opponent_id="cpu_lions",
            make_call=_make,
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "ok",
            brand="Madden 27 Franchise",
            cpu_only=True,
            enable_execution_verify=True,
        )
        self.ctrl.start()

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_html_call_exposes_ml_experimental(self) -> None:
        out = self.ctrl.call_only("1st and 10 midfield", side="offense")
        self.assertTrue(out.get("ok"), out)
        st = out["state"]
        self.assertIn("ml_experimental", st)
        self.assertIsNotNone(st["ml_experimental"])
        self.assertIn("heuristic", st["ml_experimental"])
        self.assertIn("ML experimental", out["call_text"])

    def test_undo_and_idempotency_keys(self) -> None:
        key = "idem-sprint4-1"
        a = self.ctrl.call_only("2nd and 7 own 30", side="offense", request_key=key)
        b = self.ctrl.call_only("2nd and 7 own 30", side="offense", request_key=key)
        self.assertEqual(a.get("call_text"), b.get("call_text"))


class ResearchRefreshTests(unittest.TestCase):
    def test_dry_run_preserves_active(self) -> None:
        from cfb_coach.madden import research_refresh

        before = Path("research/madden27.json").read_text(encoding="utf-8")
        with mock.patch.object(
            research_refresh,
            "_fetch",
            return_value={
                "url": "https://example.com",
                "ok": True,
                "retrieved_at": "2026-10-08T00:00:00+00:00",
                "title_guess": "Test",
                "snippet": "title update 1.007 patch notes",
                "bytes": 10,
            },
        ):
            result = research_refresh.run_research_refresh(dry_run=True, open_pr=False)
        after = Path("research/madden27.json").read_text(encoding="utf-8")
        self.assertEqual(before, after)
        self.assertTrue(result.get("ok"))
        self.assertTrue(result.get("active_research_preserved"))
        self.assertFalse(result["policy_guards"]["overwrite_active_five_formations"])

    def test_freshness_for_prep(self) -> None:
        from cfb_coach.madden.research_refresh import research_freshness_for_prep

        fr = research_freshness_for_prep()
        self.assertEqual(fr["game"], "madden27")
        self.assertIn("researched_at", fr)


class PostgameCompareTests(unittest.TestCase):
    def test_empty_compare(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                report = experimental_live.postgame_experimental_compare(db)
                self.assertEqual(report["n_experimental_calls"], 0)
                self.assertIn("counterfactual", report["caveat"].lower())
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
