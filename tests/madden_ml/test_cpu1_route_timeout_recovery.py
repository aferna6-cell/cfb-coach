"""CPU Game 1 regressions: skip duplicated work and use ML on route timeout."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import offense_route_ranker as rr
from cfb_coach.madden.model.experimental_live import (
    _resolve_model_adjustment,
    apply_experimental_offense,
    set_artifact_path,
)
from cfb_coach.madden.model.schema import CoachingMode, MLStatus, PolicySource
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.situation import Situation


class RouteRankerHotPathTests(unittest.TestCase):
    def test_reuses_previously_scored_plays_with_inactive_residual(self):
        pairs = [(f"Gun {i}", f"Play {i}") for i in range(306)]
        baseline = [
            {"formation": f, "play": p, "probability": 0.4 + i / 10000,
             "uncertainty": 0.3, "components": {}}
            for i, (f, p) in enumerate(pairs)
        ]
        art = SimpleNamespace(residual_active=False)
        with (
            mock.patch.object(rr, "resolve_artifact_path", return_value=Path("unused")),
            mock.patch.object(rr, "load_artifact", return_value=art),
            mock.patch.object(rr, "_coverage", return_value=({}, {})),
            mock.patch.object(rr, "football_priors_on", return_value=False),
            mock.patch.object(rr.sm, "structure_any", return_value=SimpleNamespace(n_known=0)),
            mock.patch.object(rr.sm, "score", side_effect=AssertionError("zero residual must not score")),
            mock.patch.object(rr.em, "rank_candidates", side_effect=AssertionError("base already scored")),
        ):
            rows, selected_pairs, ctx = rr.apply(
                "route_aware", baseline, pairs, pairs,
                sit=SimpleNamespace(down=1, distance=10, yardline=35),
                opponent_id="cpu", db=None, session_id="cpu-game",
                snap_seq=1, rank_kwargs={}, started=time.perf_counter(),
                budget_ms=1000,
            )
        self.assertTrue(ctx.applied, ctx.error)
        self.assertEqual(len(rows), 306)
        self.assertEqual(len(selected_pairs), 306)
        self.assertEqual(ctx.n_scored, 306)
        self.assertTrue(all(r["components"]["route_aware_residual_logit"] == 0 for r in rows))

    def test_no_adjustment_must_not_be_overridden_by_heuristic_macro(self):
        sit = Situation(raw="1st and 10", down=1, distance=10, yardline=35, side="offense")
        with mock.patch(
            "cfb_coach.madden.model.experimental_live._heuristic_catalog_macro",
            side_effect=AssertionError("model said NO ADJUSTMENT"),
        ):
            macro, adjustment, resolved = _resolve_model_adjustment(
                {"kind": "none", "id": None, "reason": "stock is best"},
                sit=sit, zone="open", play="Mesh", coverage=None,
                coverage_source="none", active=["MAN"], repeated=False,
                book={"Gun 5WR Tight": ["Mesh"]}, weights={},
                score_phase=None, cooled=set(), db=None, opponent_id="cpu",
            )
        self.assertIsNone(macro)
        self.assertIsNone(adjustment)
        self.assertEqual(resolved["kind"], "none")


    def test_incompatible_macro_skips_expensive_suggestion_lookup(self):
        from cfb_coach.madden.model.offense_joint_decision import legal_plans_for_play

        sit = Situation(raw="1st and 10", down=1, distance=10, yardline=35, side="offense")
        book = {"Gun 5WR Tight": ["Mesh"]}
        index = {"active": ["MAN"], "pair_sets": {"MAN": set()}, "cap": 2, "details": {}}
        with mock.patch(
            "cfb_coach.madden.offense_macros.suggest_for_snap",
            side_effect=AssertionError("incompatible macro should never be evaluated"),
        ):
            plans = legal_plans_for_play(
                formation="Gun 5WR Tight", play="Mesh", sit=sit,
                book=book, active=["MAN"],
                prediction={"probability": 0.5, "uncertainty": 0.6},
                repeated=False, audibles=None, db=None, opponent_id="cpu",
                evaluation={"coverage": {"state": "missing"}},
                allow_macros=True, macro_index=index,
            )
        self.assertEqual([p["plan_id"] for p in plans], ["NO_ADJUSTMENT"])


class PreviousExperimentalFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(
            Path(self.tmp.name) / "madden.db",
            seed={"opponents": {"cpu": {
                "display_name": "CPU", "team_now": "DET",
                "skill": "cpu", "confidence": "low", "profile_json": "{}",
            }}},
        )
        artifact = Path(self.tmp.name) / "offense.json"
        em.save_artifact(em.ExperimentalArtifact(model_version="prior-test"), artifact)
        set_artifact_path(self.db, artifact)
        self.book = {"Gun 5WR Tight": ["HB Dive", "Four Verticals"]}
        self.heuristic = MaddenCall(
            "offense", "Gun 5WR Tight", "HB Dive",
            "No adj", "Primary", "heuristic",
        )
        self.sit = Situation(
            raw="1st and 10", down=1, distance=10,
            yardline=35, side="offense",
        )
        self.ranking = [
            {"formation": "Gun 5WR Tight", "play": "Four Verticals",
             "probability": 0.72, "uncertainty": 0.1,
             "evidence_quality": "empirical", "components": {}},
            {"formation": "Gun 5WR Tight", "play": "HB Dive",
             "probability": 0.3, "uncertainty": 0.2,
             "evidence_quality": "empirical", "components": {}},
        ]

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _call(self, route_apply):
        with (
            mock.patch.object(rr, "resolve_mode", return_value="route_aware"),
            mock.patch.object(rr, "apply", side_effect=route_apply),
            mock.patch.object(em, "rank_candidates", return_value=self.ranking),
        ):
            return apply_experimental_offense(
                heuristic_call=self.heuristic, sit=self.sit,
                opponent_id="cpu", db=self.db, book=self.book,
                active=["MAN"], session_id="cpu-game", snap_seq=1,
                commit=False, budget_ms=150,
            )

    def test_route_exception_uses_previous_ml_not_heuristic(self):
        call, decision = self._call(RuntimeError("simulated route error"))
        self.assertEqual(call.play, "Four Verticals")
        self.assertNotEqual(call.play, self.heuristic.play)
        self.assertIsNone(call.macro)
        self.assertIsNone(call.adjustment)
        self.assertEqual(decision.effective_mode, CoachingMode.EXPERIMENTAL)
        self.assertEqual(decision.policy_source, PolicySource.FALLBACK)
        self.assertEqual(call.ml_experimental["fallback_source"], "previous_experimental_model")
        self.assertEqual(decision.fallback_reason, MLStatus.EXCEPTION)

    def test_joint_timeout_with_armed_macros_uses_previous_ml(self):
        pairs = [(row["formation"], row["play"]) for row in self.ranking]

        def route_works(*args, **kwargs):
            return self.ranking, pairs, SimpleNamespace(error="")

        def joint_runs_out_of_time(**kwargs):
            time.sleep(0.18)
            return {}

        with mock.patch(
            "cfb_coach.madden.model.offense_joint_decision.choose_joint_action",
            side_effect=joint_runs_out_of_time,
        ):
            call, decision = self._call(route_works)
        self.assertEqual(call.play, "Four Verticals")
        self.assertEqual(decision.fallback_reason, MLStatus.TIMEOUT)
        self.assertEqual(
            call.ml_experimental["fallback_source"], "previous_experimental_model"
        )
        self.assertIsNone(call.macro)
        self.assertIsNone(call.adjustment)

    def test_route_timeout_uses_previous_ml_without_macros(self):
        def slow(*args, **kwargs):
            time.sleep(0.18)
            return args[1], args[2], None

        call, decision = self._call(slow)
        self.assertEqual(call.play, "Four Verticals")
        self.assertEqual(decision.fallback_reason, MLStatus.TIMEOUT)
        self.assertTrue(call.ml_experimental["timed_out"])
        self.assertEqual(call.ml_experimental["fallback_source"], "previous_experimental_model")
        self.assertIsNone(call.macro)


if __name__ == "__main__":
    unittest.main()
