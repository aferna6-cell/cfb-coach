"""Sprint 5.1: harden defensive shadow — legality, CA completeness, budget, reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden.model import defense_ca, defense_opponent, defense_shadow, inference
from cfb_coach.madden.model.schema import CoachingMode
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.situation import parse_madden_situation
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


class ExperimentalMacroValidationTests(unittest.TestCase):
    def test_heat_ineligible(self) -> None:
        view = defense_ca.inspect_armed_macro("HEAT")
        self.assertFalse(view.valid)
        self.assertTrue(view.experimental or "not in research" in ";".join(view.reasons).lower())

    def test_unknown_macro_ineligible(self) -> None:
        view = defense_ca.inspect_armed_macro("NOT_A_REAL_MACRO_XYZ")
        self.assertFalse(view.valid)
        self.assertFalse(view.complete)

    def test_valid_armed_macro_complete(self) -> None:
        view = defense_ca.inspect_armed_macro("TAMPA MABLE")
        self.assertTrue(view.valid)
        self.assertTrue(view.complete)
        self.assertFalse(view.experimental)
        self.assertEqual(len(view.missing_fields), 0)

    def test_experimental_in_pool_still_invalid_without_auth(self) -> None:
        fake_meta = {
            "id": "HEAT",
            "families": ["pressure"],
            "when": "sim pressure",
            "base": {"formation": "Nickel", "play": "Mid Blitz"},
            "shell_pair": "Nickel — Mid Blitz",
        }
        with mock.patch("cfb_coach.madden.model.defense_ca.pool_macro", return_value=fake_meta):
            with mock.patch(
                "cfb_coach.madden.model.defense_ca.settings_rows",
                return_value=[
                    {"section": sec, "setting": name, "value": "Default", "source": "default"}
                    for sec, name in defense_ca.required_editor_fields()
                ],
            ):
                with mock.patch.dict("os.environ", {"CFB_COACH_ALLOW_EXPERIMENTAL_D_MACROS": ""}, clear=False):
                    view = defense_ca.inspect_armed_macro("HEAT")
        self.assertTrue(view.experimental)
        self.assertFalse(view.valid)
        self.assertTrue(any("experimental" in r.lower() for r in view.reasons))


class CompleteSettingsTests(unittest.TestCase):
    def test_inventory_covers_all_editor_fields(self) -> None:
        required = defense_ca.required_editor_fields()
        self.assertGreaterEqual(len(required), 40)
        view = defense_ca.inspect_armed_macro("SAFE DEEP")
        self.assertTrue(view.complete)
        # Every required field appears in settings_rows.
        keys = {(str(r.get("section")), str(r.get("setting"))) for r in view.settings}
        for sec, name in required:
            self.assertIn((sec, name), keys)

    def test_pool_membership_alone_insufficient(self) -> None:
        # Incomplete inventory → invalid even if meta exists.
        fake_meta = {
            "id": "FAKE",
            "families": ["cross"],
            "base": {"formation": "Nickel", "play": "Cover 3 Sky"},
        }
        with mock.patch("cfb_coach.madden.model.defense_ca.pool_macro", return_value=fake_meta):
            with mock.patch(
                "cfb_coach.madden.model.defense_ca.settings_rows",
                return_value=[{"section": "General", "setting": "QB Contain", "value": "On", "source": "x"}],
            ):
                view = defense_ca.inspect_armed_macro("FAKE")
        self.assertFalse(view.valid)
        self.assertFalse(view.complete)
        self.assertTrue(view.missing_fields)


class PlayAdjustmentCompatTests(unittest.TestCase):
    def test_cover6_play_rejects_cover2_adjustment(self) -> None:
        view = defense_ca.inspect_armed_macro("TAMPA MABLE")
        self.assertEqual(view.coverage_family_hint, "cover2")
        ok, why = defense_ca.play_adjustment_compatible(
            view,
            formation="Nickel Over",
            play="Cover 6",
            book=_d_book(),
        )
        self.assertFalse(ok)
        self.assertIn("coverage mismatch", why or "")

    def test_compatible_tampa2_accepted(self) -> None:
        view = defense_ca.inspect_armed_macro("TAMPA MABLE")
        ok, why = defense_ca.play_adjustment_compatible(
            view,
            formation="Nickel Over",
            play="Tampa 2",
            book=_d_book(),
        )
        self.assertTrue(ok, why)

    def test_recommend_withholds_incompatible_adj(self) -> None:
        rec = defense_ca.recommend_adjustment(
            ["TAMPA MABLE"],
            concept_families={"cross": 0.8},
            essential_only=False,
            formation="Nickel Over",
            play="Cover 6",
            book=_d_book(),
        )
        self.assertIsNone(rec["macro"])
        blob = (rec.get("incompatibility") or rec.get("reason") or "").lower()
        self.assertTrue("mismatch" in blob or "compat" in blob or "incompatible" in blob, blob)


class ReportingTests(unittest.TestCase):
    def test_verified_vs_unknown_outcome_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
                sit = Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50)
                call = defense_shadow.maybe_attach_defense_shadow(
                    call=heur,
                    sit=sit,
                    opponent_id="human_gavin",
                    db=db,
                    book=_d_book(),
                    armed=ARMED,
                )
                rid = defense_shadow.commit_defense_shadow(
                    db, call, game_id="g1", snap_id="g1-0001", snap_seq=1, session_id="g1"
                )
                self.assertIsNotNone(rid)
                # Unknown outcome row (not verified)
                db.log_ml_outcome(
                    snap_id="g1-0001",
                    game_id="g1",
                    decision_id=rid,
                    executed_status="unknown",
                    executed_formation=None,
                    executed_play=None,
                    executed_verification="unknown",
                )
                report = defense_shadow.postgame_defense_shadow_report(db, game_id="g1")
                self.assertEqual(report["linked_outcomes"], 1)
                self.assertEqual(report["verified_executions_linked"], 0)
                self.assertEqual(report["unknown_executions"], 1)

                # Replace with verified execution of the *shown* heuristic play
                db.log_ml_outcome(
                    snap_id="g1-0001",
                    game_id="g1",
                    decision_id=rid,
                    executed_status="identified",
                    executed_formation="Nickel Over",
                    executed_play="Cover 3 Sky",
                    executed_verification="verified",
                    replace=True,
                )
                report2 = defense_shadow.postgame_defense_shadow_report(db, game_id="g1")
                self.assertEqual(report2["verified_executions_linked"], 1)
                self.assertEqual(report2["unknown_executions"], 0)
                self.assertEqual(report2["user_used_shown_play"], 1)
                self.assertFalse(report2["examples"][0]["outcome_attributed_to_shadow"])
            finally:
                db.close()


class BudgetTests(unittest.TestCase):
    def test_timeout_preserves_heuristic_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
                sit = Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50)

                def _slow(*_a, **_k):
                    import time

                    time.sleep(0.05)
                    return defense_opponent.OpponentOffenseModel(opponent_id="human_gavin")

                with mock.patch(
                    "cfb_coach.madden.model.defense_shadow.defense_opponent.build_opponent_offense_model",
                    side_effect=_slow,
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
                self.assertGreaterEqual(float(rec.latency_ms or 0), 10.0)
            finally:
                db.close()

    def test_exception_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
                with mock.patch(
                    "cfb_coach.madden.model.defense_shadow.defense_opponent.build_opponent_offense_model",
                    side_effect=RuntimeError("boom"),
                ):
                    rec = defense_shadow.build_shadow_recommendation(
                        sit=Situation(raw="1&10", side="defense", down=1, distance=10),
                        heuristic_call=heur,
                        book=_d_book(),
                        armed_macros=ARMED,
                        opponent_id="human_gavin",
                        db=db,
                    )
                self.assertEqual(rec.play, "Cover 3 Sky")
                self.assertIn("exception", rec.reason.lower())
            finally:
                db.close()


class AcceptanceE2ETests(unittest.TestCase):
    """Demonstrate the ten Sprint 5.1 acceptance criteria."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())
        self.book = _d_book()
        # Seed opponent tendency: many crossers
        for i in range(8):
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
                result="+12" if i % 2 else "incomplete",
                concept_seen="Crossers",
                session_id="hist",
            )

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_acceptance_chain(self) -> None:
        model = defense_opponent.build_opponent_offense_model(self.db, "human_gavin")
        self.assertGreaterEqual(model.families.get("cross", defense_opponent.FamilyEvidence()).count, 1)

        # 1. Opponent tendencies affect ranking
        ranked_cross = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50),
            opponent_model=model,
            concept_weights={"cross": 0.7},
        )
        ranked_vert = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="3&12", side="defense", down=3, distance=12, yardline=40, long_yardage=True),
            opponent_model=defense_opponent.OpponentOffenseModel(
                opponent_id="human_gavin",
                families={"vert": defense_opponent.FamilyEvidence(count=10)},
                evidence_quality="empirical_light",
            ),
            concept_weights={"vert": 0.7},
        )
        self.assertNotEqual(
            (ranked_cross[0]["play"], ranked_cross[0]["coverage_family"]),
            (ranked_vert[0]["play"], ranked_vert[0]["coverage_family"]),
        )

        heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "User hooks", "h")
        sit = Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50)
        sit.concept_hint = "Crossers"
        sit.concept_source = "live"
        call = defense_shadow.maybe_attach_defense_shadow(
            call=heur,
            sit=sit,
            opponent_id="human_gavin",
            db=self.db,
            book=self.book,
            armed=ARMED + ["HEAT"],  # HEAT armed but must never be recommended
        )
        info = call.ml_defense_shadow
        self.assertIsInstance(info, dict)

        # 5. Live call remains heuristic
        self.assertEqual(call.formation, "Nickel Over")
        self.assertEqual(call.play, "Cover 3 Sky")
        self.assertFalse(info.get("controlled"))

        # 2. Selected shadow play in applied book
        shadow_form, shadow_play = info["formation"], info["play"]
        self.assertIn(shadow_play, self.book[shadow_form])

        # 3–4. Adjustment armed, complete, compatible; HEAT never auto-recommended
        adj = info.get("adjustment")
        if adj:
            self.assertIn(adj, ARMED)
            self.assertNotEqual(adj, "HEAT")
            view = defense_ca.inspect_armed_macro(adj)
            self.assertTrue(view.valid)
            self.assertTrue(view.complete)
            ok, why = defense_ca.play_adjustment_compatible(
                view, formation=shadow_form, play=shadow_play, book=self.book
            )
            self.assertTrue(ok, why)
        heat = defense_ca.inspect_armed_macro("HEAT")
        self.assertFalse(heat.valid)

        # 6. Seal + verified outcome join by snap id
        def _make(sit2, **kwargs):
            h = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "User hooks", "h")
            return defense_shadow.maybe_attach_defense_shadow(
                call=h,
                sit=sit2,
                opponent_id="human_gavin",
                db=self.db,
                book=self.book,
                armed=ARMED,
            )

        ctrl = LivePlayController(
            db=self.db,
            opponent_id="human_gavin",
            make_call=_make,
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "ok",
            brand="Madden 27 Franchise",
            cpu_only=False,
            enable_execution_verify=True,
        )
        ctrl.start()
        out = ctrl.call_only("3rd and 8 midfield defense", side="defense")
        self.assertTrue(out.get("ok"), out)
        rows = list(self.db.conn.execute("SELECT * FROM ml_decisions WHERE mode='shadow'"))
        self.assertGreaterEqual(len(rows), 1)
        dec = rows[-1]
        self.assertTrue(dec["snap_id"])
        self.assertEqual(dec["final_play"], "Cover 3 Sky")  # shown = heuristic
        nxt = ctrl.result_and_call(
            outcome="incomplete",
            sit_raw="1st and 10 midfield defense",
            side="defense",
            executed_status="used_recommended",
        )
        self.assertTrue(nxt.get("ok"), nxt)
        outc = self.db.conn.execute(
            "SELECT * FROM ml_outcomes WHERE snap_id=?", (dec["snap_id"],)
        ).fetchone()
        self.assertIsNotNone(outc)
        report = defense_shadow.postgame_defense_shadow_report(self.db, game_id=ctrl.session_id)
        self.assertGreaterEqual(report["verified_executions_linked"], 1)
        self.assertFalse(report["examples"][0]["outcome_attributed_to_shadow"])

        # 7. Undo / retry integrity
        key = "s51-idem"
        a = ctrl.call_only("2nd and 5 midfield defense", side="defense", request_key=key)
        b = ctrl.call_only("2nd and 5 midfield defense", side="defense", request_key=key)
        self.assertEqual(a.get("call_text"), b.get("call_text"))

        # 8. CPU offense-only
        cpu_call = defense_shadow.maybe_attach_defense_shadow(
            call=MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h"),
            sit=sit,
            opponent_id="cpu",
            db=self.db,
            book=self.book,
            armed=ARMED,
        )
        self.assertIsNone(getattr(cpu_call, "ml_defense_shadow", None))

        # 9. Experimental offense unchanged
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.EXPERIMENTAL)
        o_call = defense_shadow.maybe_attach_defense_shadow(
            call=MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "a", "h"),
            sit=Situation(raw="1&10", side="offense", down=1, distance=10),
            opponent_id="human_gavin",
            db=self.db,
            book=self.book,
            armed=ARMED,
        )
        self.assertIsNone(getattr(o_call, "ml_defense_shadow", None))
        inference.set_mode(self.db, CoachingMode.HEURISTIC)


if __name__ == "__main__":
    unittest.main()
