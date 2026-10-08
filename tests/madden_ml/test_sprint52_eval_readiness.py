"""Sprint 5.2: Stage 3 evaluation, readiness, provenance, full budget fallback, control refuse."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import (
    defense_ca,
    defense_control,
    defense_eval,
    defense_opponent,
    defense_readiness,
    defense_shadow,
    inference,
)
from cfb_coach.madden.model.schema import CoachingMode, ML_LATENCY_BUDGET_MS
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.situation import Situation


def _seed() -> dict:
    return {
        "opponents": {
            "human_gavin": {
                "display_name": "Gavin",
                "team_now": "KC",
                "skill": "human",
                "confidence": "medium",
                "profile_json": "{}",
            },
            "human_alex": {
                "display_name": "Alex",
                "team_now": "BUF",
                "skill": "human",
                "confidence": "medium",
                "profile_json": "{}",
            },
            "cpu": {
                "display_name": "CPU",
                "team_now": "DET",
                "skill": "cpu",
                "confidence": "low",
                "profile_json": "{}",
            },
        }
    }


def _d_book() -> dict[str, list[str]]:
    return {
        "Nickel Over": [
            "Cover 3 Sky",
            "Cover 4 Quarters",
            "Tampa 2",
            "Cover 1 Hole",
            "Mid Blitz",
            "Cover 6",
        ],
        "Nickel Normal": ["Cover 2 Man", "Cover 3 Match", "Safe", "Pinch"],
        "4-3 Over": ["Cover 3 Sky", "Cover 1 Robber", "QB Spy"],
    }


ARMED = [
    "TAMPA MABLE",
    "SAFE DEEP",
    "PRESS SHADE IN",
    "LOOP MAN 0",
    "RZ COVER 2",
    "TEX 4 MAN",
    "TEX2 L CONT",
    "TEX2 R CONT",
]


def _log_shadow(
    db: CoachDB,
    *,
    game_id: str,
    snap_id: str,
    snap_seq: int,
    opponent_id: str = "human_gavin",
    verify: bool = False,
    executed_play: str = "Cover 3 Sky",
    executed_form: str = "Nickel Over",
    executed_macro: str | None = None,
) -> None:
    heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
    sit = Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50)
    call = defense_shadow.maybe_attach_defense_shadow(
        call=heur,
        sit=sit,
        opponent_id=opponent_id,
        db=db,
        book=_d_book(),
        armed=ARMED,
    )
    rid = defense_shadow.commit_defense_shadow(
        db, call, game_id=game_id, snap_id=snap_id, snap_seq=snap_seq, session_id=game_id
    )
    assert rid is not None
    if verify:
        db.log_ml_outcome(
            snap_id=snap_id,
            game_id=game_id,
            decision_id=rid,
            executed_status="identified",
            executed_formation=executed_form,
            executed_play=executed_play,
            executed_verification="verified",
            outcome={
                "executed_macro": executed_macro,
                "executed_verification": "verified",
                "concept_seen": "Crossers",
            },
        )


class ProvenanceTests(unittest.TestCase):
    def test_distinguishes_researched_vs_synthesized(self) -> None:
        view = defense_ca.inspect_armed_macro("TAMPA MABLE")
        self.assertTrue(view.valid)
        cov = view.source_coverage
        self.assertGreater(cov["researched"], 0)
        self.assertGreater(cov["synthesized_default"], 0)
        self.assertEqual(
            cov["researched"] + cov["explicit_default"] + cov["synthesized_default"],
            cov["total"],
        )
        kinds = {k["kind"] for k in view.setting_kinds}
        self.assertIn("researched", kinds)
        self.assertIn("synthesized_default", kinds)

    def test_control_eligible_requires_provenance(self) -> None:
        view = defense_ca.inspect_armed_macro("TAMPA MABLE")
        # Real macros with ≥3 researched fields are provenance-ok; control still off.
        self.assertTrue(view.provenance_ok_for_control)
        self.assertTrue(view.control_eligible)

        # Force thin research → not control-eligible.
        with mock.patch(
            "cfb_coach.madden.model.defense_ca._raw_researched_index",
            return_value={
                ("coverage", "coverage shell"): {
                    "section": "Coverage",
                    "setting": "Coverage Shell",
                    "value": "Cover 2",
                    "source": "test",
                }
            },
        ):
            thin = defense_ca.inspect_armed_macro("TAMPA MABLE")
            # settings_rows still full; but raw index only has 1 → researched count 1
            self.assertFalse(thin.provenance_ok_for_control)
            self.assertFalse(thin.control_eligible)

    def test_recommend_reports_provenance(self) -> None:
        out = defense_ca.recommend_adjustment(
            ARMED,
            concept_families={"cross": 0.6},
            formation="Nickel Over",
            play="Tampa 2",
            book=_d_book(),
            essential_only=False,
        )
        if out.get("macro"):
            self.assertIn("source_coverage", out)
            self.assertIn("control_eligible", out)
            self.assertIn("provenance_ok_for_control", out)


class FullBudgetFallbackTests(unittest.TestCase):
    def test_timeout_before_ca_full_heuristic(self) -> None:
        """Over-budget after ranking must not keep ranked play / drop-adj only."""
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
                sit = Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50)

                real_rank = defense_shadow.rank_defense_candidates

                def _slow_rank(*a, **k):
                    import time

                    time.sleep(0.05)
                    return real_rank(*a, **k)

                with mock.patch(
                    "cfb_coach.madden.model.defense_shadow.rank_defense_candidates",
                    side_effect=_slow_rank,
                ):
                    rec = defense_shadow.build_shadow_recommendation(
                        sit=sit,
                        heuristic_call=heur,
                        book=_d_book(),
                        armed_macros=ARMED,
                        opponent_id="human_gavin",
                        db=db,
                        budget_ms=10,
                    )
                self.assertEqual(rec.formation, "Nickel Over")
                self.assertEqual(rec.play, "Cover 3 Sky")
                self.assertIsNone(rec.adjustment)
                self.assertIn("timeout", rec.reason.lower())
                self.assertIn("fallback", rec.reason.lower())
                # Must not look like play-only / drop-adj partial path.
                self.assertNotIn("play only", (rec.adjustment_reason or "").lower())
                self.assertNotIn("dropped adjustment", (rec.adjustment_reason or "").lower())
            finally:
                db.close()

    def test_timeout_after_ca_full_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
                real_adj = defense_ca.recommend_adjustment

                def _slow_adj(*a, **k):
                    import time

                    time.sleep(0.05)
                    return real_adj(*a, **k)

                with mock.patch(
                    "cfb_coach.madden.model.defense_shadow.defense_ca.recommend_adjustment",
                    side_effect=_slow_adj,
                ):
                    rec = defense_shadow.build_shadow_recommendation(
                        sit=Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50),
                        heuristic_call=heur,
                        book=_d_book(),
                        armed_macros=ARMED,
                        opponent_id="human_gavin",
                        db=db,
                        budget_ms=10,
                    )
                self.assertEqual(rec.play, "Cover 3 Sky")
                self.assertEqual(rec.formation, "Nickel Over")
                self.assertIsNone(rec.adjustment)
                self.assertIn("timeout", rec.reason.lower())
            finally:
                db.close()

    def test_budget_constant(self) -> None:
        self.assertEqual(ML_LATENCY_BUDGET_MS, 150)


class EvalAndReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())
        for i in range(6):
            self.db.log_snap(
                opponent_id="human_gavin",
                side="defense",
                situation_raw="3&8",
                our_call="x",
                formation="Nickel Over",
                play="Cover 3 Sky",
                down=3,
                distance=8,
                yardline=50,
                result="incomplete",
                concept_seen="Crossers",
                session_id="hist",
            )

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_eval_report_verified_only(self) -> None:
        _log_shadow(self.db, game_id="g1", snap_id="g1-0001", snap_seq=1, verify=True)
        _log_shadow(self.db, game_id="g1", snap_id="g1-0002", snap_seq=2, verify=False)
        _log_shadow(
            self.db,
            game_id="g2",
            snap_id="g2-0001",
            snap_seq=1,
            opponent_id="human_alex",
            verify=True,
            executed_macro="TAMPA MABLE",
        )
        report = defense_eval.stage3_evaluation_report(self.db)
        self.assertEqual(report["kind"], "stage3_defense_evaluation")
        self.assertGreaterEqual(report["n_shadow_defense_calls"], 3)
        self.assertGreaterEqual(report["n_games"], 2)
        self.assertIn("heuristic_vs_shadow", report)
        self.assertIn("coverage_family_recommendations", report)
        self.assertTrue(report["verified_executed_plays"])
        for ex in report["verified_outcome_examples"]:
            self.assertFalse(ex["outcome_attributed_to_shadow"])
        self.assertIn("latency", report)
        self.assertIn("missing_evidence", report)

    def test_readiness_dashboard(self) -> None:
        for i in range(5):
            _log_shadow(
                self.db,
                game_id=f"g{i}",
                snap_id=f"g{i}-0001",
                snap_seq=1,
                opponent_id="human_gavin" if i % 2 == 0 else "human_alex",
                verify=True,
            )
        dash = defense_readiness.readiness_dashboard(self.db)
        self.assertEqual(dash["kind"], "stage3_readiness_dashboard")
        self.assertFalse(dash["control_ready"])
        self.assertIn("checks", dash)
        names = {c["name"] for c in dash["checks"]}
        self.assertIn("verified_defensive_snaps", names)
        self.assertIn("distinct_human_opponent_games", names)
        self.assertIn("control_path_disabled", names)
        # Volume targets not met with only 5 verified.
        verified_check = next(c for c in dash["checks"] if c["name"] == "verified_defensive_snaps")
        self.assertFalse(verified_check["ok"])
        self.assertTrue(dash["remaining_evidence"])
        rec = dash["next_human_opponent_recommendation"]
        self.assertEqual(rec["control"], "off")
        self.assertEqual(rec["mode"], "shadow_observation_only")
        text = defense_readiness.format_readiness_text(dash)
        self.assertIn("Stage 3 Defense Readiness", text)
        self.assertIn("NEED", text)


class ControlPilotTests(unittest.TestCase):
    def test_refuse_activation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                result = defense_control.try_activate_pilot(db, authorization="secret", force=True)
                self.assertFalse(result["activated"])
                self.assertFalse(result["ok"])
                self.assertFalse(defense_control.pilot_enabled(db))
                self.assertFalse(defense_shadow.control_enabled(db))
            finally:
                db.close()

    def test_meta_on_still_refuses_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                db.set_meta(defense_control.META_PILOT, "on")
                db.set_meta(defense_control.META_PILOT_AUTH, "token")
                db.set_meta(defense_shadow.META_CONTROL, "on")
                # try_activate clears metas
                defense_control.try_activate_pilot(db)
                self.assertFalse(defense_control.pilot_enabled(db))
                self.assertFalse(defense_shadow.control_enabled(db))
            finally:
                db.close()

    def test_select_controlled_always_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
                dec = defense_control.select_controlled_defense(
                    sit=Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50),
                    heuristic_call=heur,
                    book=_d_book(),
                    armed_macros=ARMED,
                    opponent_id="human_gavin",
                    db=db,
                )
                self.assertFalse(dec.controlled)
                self.assertTrue(dec.rolled_back)
                self.assertEqual(dec.formation, "Nickel Over")
                self.assertEqual(dec.play, "Cover 3 Sky")
                self.assertEqual(dec.source, "heuristic_rollback")
            finally:
                db.close()

    def test_instant_rollback(self) -> None:
        dec = defense_control.instant_rollback_to_heuristic(
            heuristic_formation="4-3 Over",
            heuristic_play="Cover 3 Sky",
            heuristic_adjustment=None,
        )
        self.assertFalse(dec.controlled)
        self.assertTrue(dec.rolled_back)
        self.assertEqual(dec.play, "Cover 3 Sky")


class RegressionIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_cpu_offense_only_unchanged(self) -> None:
        call = defense_shadow.maybe_attach_defense_shadow(
            call=MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h"),
            sit=Situation(raw="1&10", side="defense", down=1, distance=10),
            opponent_id="cpu",
            db=self.db,
            book=_d_book(),
            armed=ARMED,
        )
        self.assertIsNone(getattr(call, "ml_defense_shadow", None))
        ctrl = defense_control.select_controlled_defense(
            sit=Situation(raw="1&10", side="defense", down=1, distance=10),
            heuristic_call=MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h"),
            book=_d_book(),
            armed_macros=ARMED,
            opponent_id="cpu",
            db=self.db,
        )
        self.assertFalse(ctrl.controlled)
        self.assertEqual(ctrl.play, "Cover 3 Sky")

    def test_experimental_offense_mode_unchanged(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.EXPERIMENTAL)
        o_call = defense_shadow.maybe_attach_defense_shadow(
            call=MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "a", "h"),
            sit=Situation(raw="1&10", side="offense", down=1, distance=10),
            opponent_id="human_gavin",
            db=self.db,
            book=_d_book(),
            armed=ARMED,
        )
        self.assertIsNone(getattr(o_call, "ml_defense_shadow", None))
        inference.set_mode(self.db, CoachingMode.HEURISTIC)

    def test_incomplete_macro_and_incompatible_play(self) -> None:
        heat = defense_ca.inspect_armed_macro("HEAT")
        self.assertFalse(heat.valid)
        unknown = defense_ca.inspect_armed_macro("NOT_REAL")
        self.assertFalse(unknown.valid)
        view = defense_ca.inspect_armed_macro("TAMPA MABLE")
        # Force coverage mismatch via base play if possible
        ok, why = defense_ca.play_adjustment_compatible(
            view, formation="Missing Form", play="Not A Play", book=_d_book()
        )
        self.assertFalse(ok)

    def test_missing_evidence_logged(self) -> None:
        # Empty DB → prior-driven / missing concept observations
        heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
        rec = defense_shadow.build_shadow_recommendation(
            sit=Situation(raw="1&10", side="defense", down=1, distance=10, yardline=40),
            heuristic_call=heur,
            book=_d_book(),
            armed_macros=ARMED,
            opponent_id="human_gavin",
            db=self.db,
        )
        self.assertIsInstance(rec.missing, list)


if __name__ == "__main__":
    unittest.main()
