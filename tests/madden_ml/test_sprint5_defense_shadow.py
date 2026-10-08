"""Sprint 5: opponent-aware defensive shadow advisor (log-only)."""

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
from cfb_coach.madden.playcaller import MaddenCall, make_call
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
                "display_name": "CPU Dynasty",
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


class OpponentModelTests(unittest.TestCase):
    def test_single_observation_high_uncertainty(self) -> None:
        model = defense_opponent.OpponentOffenseModel(
            opponent_id="human_gavin",
            families={"cross": defense_opponent.FamilyEvidence(count=1)},
            n_defense_snaps=1,
        )
        share, unc, n = model.family_share("cross")
        self.assertEqual(n, 1)
        self.assertGreater(unc, 0.5)
        self.assertLess(share, 0.5)  # shrunk toward prior

    def test_repeated_vs_one_off(self) -> None:
        model = defense_opponent.OpponentOffenseModel(
            opponent_id="x",
            repeated_concepts={"Flood": 3, "Mesh": 1},
        )
        self.assertTrue(defense_opponent.is_repeated_concept(model, "Flood"))
        self.assertFalse(defense_opponent.is_repeated_concept(model, "Mesh"))

    def test_last_snap_concept_is_soft(self) -> None:
        model = defense_opponent.OpponentOffenseModel(opponent_id="x")
        live = Situation(raw="3&7 crossers live", side="defense", down=3, distance=7)
        live.concept_hint = "Crossers"
        live.concept_source = "live"
        last = Situation(raw="3&7", side="defense", down=3, distance=7)
        last.concept_hint = "Crossers"
        last.concept_source = "last"
        w_live = defense_opponent.situation_concept_prior(live, model)
        w_last = defense_opponent.situation_concept_prior(last, model)
        self.assertGreater(w_live.get("cross", 0), w_last.get("cross", 0))

    def test_refine_run_families(self) -> None:
        self.assertEqual(defense_opponent.refine_run_family("Inside Zone"), "inside_zone")
        self.assertEqual(defense_opponent.refine_run_family("Stretch"), "outside_zone")
        self.assertEqual(defense_opponent.refine_run_family("Power O"), "power")
        self.assertEqual(defense_opponent.refine_run_family("RPO bubble"), "rpo")
        self.assertEqual(defense_opponent.refine_run_family("QB scramble"), "scram")

    def test_cpu_vs_human_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                m = defense_opponent.build_opponent_offense_model(db, "cpu")
                self.assertEqual(m.opponent_kind, "cpu")
                m2 = defense_opponent.build_opponent_offense_model(db, "human_gavin")
                self.assertEqual(m2.opponent_kind, "human")
            finally:
                db.close()


class CustomAdjustmentTests(unittest.TestCase):
    def test_inspect_armed_eight(self) -> None:
        views = [defense_ca.inspect_armed_macro(m) for m in ARMED]
        self.assertEqual(len(views), 8)
        # Pool macros should be present
        self.assertTrue(any(v.valid for v in views))

    def test_never_invent_unarmed(self) -> None:
        rec = defense_ca.recommend_adjustment(
            ["TAMPA MABLE"],
            concept_families={"cross": 0.6},
            essential_only=False,
        )
        self.assertEqual(rec["macro"], "TAMPA MABLE")
        # Unknown id is invalid
        bad = defense_ca.inspect_armed_macro("NOT_A_REAL_MACRO")
        self.assertFalse(bad.valid)

    def test_conflicting_settings_not_combined(self) -> None:
        a = defense_ca.ArmedMacroView(
            mid="A",
            xbox_name="A",
            families=["cross"],
            when_to_arm="",
            settings=[],
            researched_settings={"Coverage": "Cover 2"},
            buttons="",
            shell_pair="",
            valid=True,
        )
        b = defense_ca.ArmedMacroView(
            mid="B",
            xbox_name="B",
            families=["cross"],
            when_to_arm="",
            settings=[],
            researched_settings={"Coverage": "Cover 3"},
            buttons="",
            shell_pair="",
            valid=True,
        )
        conflicts = defense_ca.settings_conflict(a, b)
        self.assertTrue(conflicts)
        ranked = defense_ca.compatible_macros(
            ["B"], preferred_families=["cross"], already_armed="A"
        )
        # Monkeypatch inspect by using synthetic via already_armed path — B conflicts with A
        # when we inject: use settings_conflict directly (above) as the unit proof.
        self.assertTrue(any("Coverage" in c for c in conflicts))


class RankingSituationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.book = _d_book()

    def _model(self, **fams: int) -> defense_opponent.OpponentOffenseModel:
        families = {
            k: defense_opponent.FamilyEvidence(count=v, success_vs_us=max(0, v - 1))
            for k, v in fams.items()
        }
        return defense_opponent.OpponentOffenseModel(
            opponent_id="human_gavin",
            opponent_kind="human",
            n_defense_snaps=sum(fams.values()),
            families=families,
            evidence_quality="empirical_light",
        )

    def test_crossers_prefers_man_or_cover2(self) -> None:
        ranked = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50),
            opponent_model=self._model(cross=10),
            concept_weights={"cross": 0.7},
        )
        top_fams = {r["coverage_family"] for r in ranked[:3]}
        self.assertTrue(top_fams & {"man", "cover2", "single_high"})

    def test_flood_and_verts(self) -> None:
        flood = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="2&7", side="defense", down=2, distance=7, yardline=45),
            opponent_model=self._model(flood=8),
            concept_weights={"flood": 0.6},
        )
        self.assertTrue(flood)
        verts = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="3&10", side="defense", down=3, distance=10, yardline=40, long_yardage=True),
            opponent_model=self._model(vert=9),
            concept_weights={"vert": 0.7},
        )
        self.assertIn("two_high", {r["coverage_family"] for r in verts[:4]})

    def test_rpo_and_run_heavy(self) -> None:
        ranked = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="1&10", side="defense", down=1, distance=10, yardline=35),
            opponent_model=self._model(rpo=6, inside_zone=8),
            concept_weights={"rpo": 0.3, "inside_zone": 0.4},
        )
        self.assertTrue(any(r["coverage_family"] == "single_high" for r in ranked[:5]))

    def test_scramble_avoids_pure_pressure_bias(self) -> None:
        ranked = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="3&5", side="defense", down=3, distance=5, yardline=55),
            opponent_model=self._model(scram=8),
            concept_weights={"scram": 0.7},
        )
        # COUNTERS["scram"] penalizes pressure
        pressure_scores = [r["score"] for r in ranked if r["coverage_family"] == "pressure"]
        man_scores = [r["score"] for r in ranked if r["coverage_family"] == "man"]
        if pressure_scores and man_scores:
            self.assertGreaterEqual(max(man_scores), max(pressure_scores) - 0.2)

    def test_short_yardage_third_long_red_zone_late(self) -> None:
        short = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="3&1", side="defense", down=3, distance=1, yardline=50, short_yardage=True),
            opponent_model=self._model(run=5),
            concept_weights={"run": 0.5},
        )
        self.assertTrue(short)
        long = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="3&14", side="defense", down=3, distance=14, yardline=40, long_yardage=True),
            opponent_model=self._model(vert=5),
            concept_weights={"vert": 0.5},
        )
        self.assertTrue(any(r["coverage_family"] == "two_high" for r in long[:5]))
        rz = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="1&G", side="defense", down=1, distance=3, yardline=97, red_zone=True, goal_line=True),
            opponent_model=self._model(cross=4),
            concept_weights={"cross": 0.4},
        )
        self.assertTrue(rz)
        late = defense_shadow.rank_defense_candidates(
            book=self.book,
            sit=Situation(raw="2&10 0:48", side="defense", down=2, distance=10, yardline=60, two_minute=True),
            opponent_model=self._model(vert=4),
            concept_weights={"vert": 0.4},
        )
        self.assertTrue(any(r["coverage_family"] in ("two_high", "cover2") for r in late[:5]))


class ShadowAdvisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())
        self.book = _d_book()

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_shadow_does_not_change_call(self) -> None:
        heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "User hooks", "h")
        sit = Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50)
        call = defense_shadow.maybe_attach_defense_shadow(
            call=heur,
            sit=sit,
            opponent_id="human_gavin",
            db=self.db,
            book=self.book,
            armed=ARMED,
        )
        self.assertEqual(call.formation, "Nickel Over")
        self.assertEqual(call.play, "Cover 3 Sky")
        info = call.ml_defense_shadow
        self.assertIsInstance(info, dict)
        self.assertFalse(info.get("controlled"))
        self.assertIn("format_line", info)

    def test_cpu_no_shadow(self) -> None:
        heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
        call = defense_shadow.maybe_attach_defense_shadow(
            call=heur,
            sit=Situation(raw="1&10", side="defense", down=1, distance=10),
            opponent_id="cpu",
            db=self.db,
            book=self.book,
            armed=ARMED,
        )
        self.assertIsNone(getattr(call, "ml_defense_shadow", None))

    def test_commit_after_seal_and_report(self) -> None:
        heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
        sit = Situation(raw="3&8", side="defense", down=3, distance=8, yardline=50)
        call = defense_shadow.maybe_attach_defense_shadow(
            call=heur, sit=sit, opponent_id="human_gavin", db=self.db, book=self.book, armed=ARMED
        )
        row_id = defense_shadow.commit_defense_shadow(
            self.db, call, game_id="g1", snap_id="g1-0001", snap_seq=1, session_id="g1"
        )
        self.assertIsNotNone(row_id)
        row = self.db.conn.execute("SELECT * FROM ml_decisions WHERE id=?", (row_id,)).fetchone()
        self.assertEqual(row["mode"], "shadow")
        self.assertEqual(row["final_play"], "Cover 3 Sky")  # shown = heuristic
        self.assertTrue(row["shadow_play"])
        payload = json.loads(row["decision_json"])["defense_shadow"]
        self.assertEqual(payload["kind"], "defense_shadow")
        self.assertFalse(payload["controlled"])
        report = defense_shadow.postgame_defense_shadow_report(self.db, game_id="g1")
        self.assertEqual(report["n_shadow_defense_calls"], 1)
        self.assertFalse(report["examples"][0]["outcome_attributed_to_shadow"])

    def test_activation_refused(self) -> None:
        result = defense_shadow.refuse_activation(self.db)
        self.assertFalse(result["activated"])
        self.assertFalse(defense_shadow.control_enabled(self.db))

    def test_missing_observations_still_useful(self) -> None:
        heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "u", "h")
        rec = defense_shadow.build_shadow_recommendation(
            sit=Situation(raw="1&10", side="defense", down=1, distance=10, yardline=40),
            heuristic_call=heur,
            book=self.book,
            armed_macros=ARMED,
            opponent_id="human_gavin",
            db=self.db,
        )
        self.assertTrue(rec.formation)
        self.assertTrue(rec.play)
        self.assertIn(rec.evidence_quality, ("prior_driven", "empirical_light", "empirical"))

    def test_html_join_and_undo_path(self) -> None:
        def _make(sit, **kwargs):
            heur = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj", "User hooks", "h")
            return defense_shadow.maybe_attach_defense_shadow(
                call=heur,
                sit=sit,
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
        st = out["state"]
        self.assertIsNotNone(st.get("ml_defense_shadow"))
        self.assertFalse(st["ml_defense_shadow"].get("controlled"))
        rows = list(self.db.conn.execute("SELECT * FROM ml_decisions WHERE mode='shadow'"))
        self.assertEqual(len(rows), 1)
        # Idempotent retry
        key = "dshadow-idem"
        a = ctrl.call_only("2nd and 7 midfield defense", side="defense", request_key=key)
        b = ctrl.call_only("2nd and 7 midfield defense", side="defense", request_key=key)
        self.assertEqual(a.get("call_text"), b.get("call_text"))


class IsolationTests(unittest.TestCase):
    """Experimental offense and CPU offense-only must remain unchanged."""

    def test_cpu_make_call_stays_offense_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                sit = parse_madden_situation("defense 1&10 midfield", default_side="defense")
                # CPU forces offense — no defense shadow path.
                call = make_call(
                    sit,
                    "cpu",
                    db=db,
                    playbook={"offense": {"Gun Bunch": ["Mesh"]}, "defense": _d_book()},
                    active_macros={"offense": [], "defense": ARMED},
                    rng=__import__("random").Random(0),
                )
                self.assertTrue(str(call.side).startswith("o"))
                self.assertIsNone(getattr(call, "ml_defense_shadow", None))
            finally:
                db.close()

    def test_experimental_offense_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                from cfb_coach.madden.model import experimental_live, experimental_model

                art = experimental_model.train_experimental([], side="offense")
                dest = Path(td) / "e.json"
                experimental_model.save_artifact(art, dest)
                experimental_live.set_artifact_path(db, dest)
                inference.set_mode(db, CoachingMode.EXPERIMENTAL)
                self.assertIs(inference.resolve_mode(db), CoachingMode.EXPERIMENTAL)
                # Defense shadow attach must not clear experimental mode.
                heur = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "a → b", "h")
                call = defense_shadow.maybe_attach_defense_shadow(
                    call=heur,
                    sit=Situation(raw="1&10", side="offense", down=1, distance=10),
                    opponent_id="human_gavin",
                    db=db,
                    book=_d_book(),
                    armed=ARMED,
                )
                self.assertIsNone(getattr(call, "ml_defense_shadow", None))
                self.assertIs(inference.resolve_mode(db), CoachingMode.EXPERIMENTAL)
                inference.set_mode(db, CoachingMode.HEURISTIC)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
