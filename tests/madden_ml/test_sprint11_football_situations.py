"""Sprint 11 Stage B: football situation evaluator and pregame portfolio grid."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model.football_situations import (
    PREGAME_SITUATION_GRID,
    SITUATION_PRIORS,
    assess_situation,
    grid_cell_assessment,
    play_situation_fit,
)
from cfb_coach.situation import Situation


def _sit(down=1, distance=10, **kw) -> Situation:
    sit = Situation(raw="test")
    sit.down = down
    sit.distance = distance
    for key, value in kw.items():
        setattr(sit, key, value)
    return sit


class SituationAssessmentTests(unittest.TestCase):
    def test_third_and_long_phase_and_conversion_need(self):
        a = assess_situation(_sit(3, 10))
        self.assertIn("third_or_fourth_long", a["phases"])
        self.assertTrue(a["conversion"]["long_yardage"])
        self.assertFalse(a["conversion"]["short_yardage"])

    def test_short_yardage_and_goal_line_phases(self):
        self.assertIn("short_yardage", assess_situation(_sit(4, 1))["phases"])
        gl = assess_situation(_sit(2, 2, goal_line=True, red_zone=True))
        self.assertIn("goal_line", gl["phases"])
        self.assertTrue(gl["red_zone"])

    def test_backed_up_detected_from_own_territory(self):
        a = assess_situation(_sit(1, 10, yardline=4))
        self.assertIn("backed_up", a["phases"])
        self.assertTrue(a["backed_up"])

    def test_two_minute_posture_split_by_score(self):
        trail = assess_situation(_sit(2, 8, two_minute=True, score_us=14, score_them=21))
        lead = assess_situation(_sit(2, 8, two_minute=True, score_us=28, score_them=10))
        unknown = assess_situation(_sit(2, 8, two_minute=True))
        self.assertEqual(trail["clock_posture"], "preserve_clock")
        self.assertEqual(lead["clock_posture"], "consume_clock")
        self.assertEqual(unknown["clock_posture"], "unknown")
        self.assertEqual(unknown["score"]["posture"], "unknown")

    def test_unknown_coverage_never_becomes_certainty(self):
        a = assess_situation(_sit(1, 10))
        self.assertIsNone(a["defense"]["observed_look"])
        self.assertIsNone(a["defense"]["inferred_tendency"])
        self.assertEqual(a["defense"]["credibility"], "unknown")

    def test_last_snap_coverage_is_inferred_not_observed(self):
        sit = _sit(1, 10, coverage_hint="Cover 2", coverage_source="last")
        a = assess_situation(sit)
        self.assertIsNone(a["defense"]["observed_look"])
        self.assertEqual(a["defense"]["inferred_tendency"], "Cover 2")
        self.assertIn(a["defense"]["credibility"],
                      ("inferred_single", "inferred_repeated"))

    def test_live_coverage_is_observed(self):
        sit = _sit(1, 10, coverage_hint="pressure", coverage_source="live")
        a = assess_situation(sit)
        self.assertEqual(a["defense"]["observed_look"], "pressure")
        self.assertEqual(a["defense"]["credibility"], "observed_live")
        self.assertEqual(a["defense"]["observed_class"], "pressure")


class PlayFitTests(unittest.TestCase):
    def test_every_delta_is_a_named_documented_prior(self):
        for name, prior in SITUATION_PRIORS.items():
            self.assertIn("delta", prior, name)
            self.assertTrue(prior.get("rationale"), name)
            self.assertTrue(prior.get("basis"), name)

    def test_third_and_long_prioritizes_realistic_conversion(self):
        a = assess_situation(_sit(3, 10))
        run, run_why = play_situation_fit("HB Dive", a)
        screen, _ = play_situation_fit("HB Slip Screen", a)
        deep, deep_why = play_situation_fit("Four Verticals", a)
        self.assertLess(run, -0.3)
        self.assertLess(screen, 0)
        self.assertGreater(deep, 0)
        self.assertIn("long_down_run", " ".join(run_why))
        self.assertIn("long_down_route_past_sticks", " ".join(deep_why))

    def test_second_and_short_considers_upside(self):
        a = assess_situation(_sit(2, 2))
        shot, why = play_situation_fit("PA Deep Shot", a)
        self.assertGreater(shot, 0)
        self.assertIn("second_short_upside", " ".join(why))

    def test_goal_line_accounts_for_compressed_spacing(self):
        a = assess_situation(_sit(2, 1, goal_line=True, red_zone=True))
        deep, _ = play_situation_fit("Four Verticals", a)
        run, _ = play_situation_fit("HB Dive", a)
        self.assertLess(deep, 0)
        self.assertGreater(run, 0)

    def test_late_lead_values_possession_and_clock(self):
        sit = _sit(1, 10, score_us=24, score_them=17)
        sit.extras["quarter"] = 4
        a = assess_situation(sit)
        self.assertIn("protect_lead_late", a["phases"])
        run, why = play_situation_fit("Inside Zone", a)
        deep, _ = play_situation_fit("PA Deep Shot", a)
        self.assertGreater(run, 0)
        self.assertLess(deep, 0)
        self.assertIn("protect_lead_possession", " ".join(why))

    def test_two_minute_unknown_score_assumes_nothing(self):
        a = assess_situation(_sit(2, 6, two_minute=True))
        delta, _ = play_situation_fit("Inside Zone", a)
        self.assertEqual(delta, 0.0)

    def test_unknown_down_contributes_nothing(self):
        a = assess_situation(Situation(raw="x"))
        delta, why = play_situation_fit("Mesh", a)
        self.assertEqual(delta, 0.0)
        self.assertIn("down/distance unknown", why[0])


CATALOG = {
    "Buccaneers": {
        "Gun Bunch": ["Mesh", "Inside Zone", "Flood", "Four Verticals", "Quick Slants"],
        "Singleback Wing": ["Stretch", "PA Boot", "Quick Slants"],
    },
    "Lions": {
        "Gun Tight": ["Mesh", "HB Dive", "Smash", "Slants", "Flood"],
        "Gun Trips": ["Inside Zone", "Verticals", "PA Cross", "HB Draw"],
        "Goal Line Jumbo": ["QB Sneak", "HB Dive"],
    },
}


def _seed() -> dict:
    return {"opponents": {"cpu": {
        "display_name": "CPU", "team_now": "DET", "skill": "cpu",
        "confidence": "low", "profile_json": "{}"
    }}}


class PregamePortfolioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed=_seed())
        old = {
            "side": "offense", "mode": "custom", "name": "Buccaneers",
            "source_book": "Buccaneers", "rev": 3, "trimmed": True,
            "formations": {"Gun Bunch": ["Mesh", "Inside Zone"]},
            "audibles": {}, "core": ["Gun Bunch"], "locked_ts": "test",
        }
        playbook._save_state(self.db, {"applied": {"offense": copy.deepcopy(old)},
                                        "pending": {}})
        self.artifact = experimental_model.ExperimentalArtifact(
            model_version="test.model", n_supervised=3,
            evidence_quality="mixed", global_rate=0.5,
        )
        self.catalog_patch = mock.patch.object(
            designer.catalog, "book_formations",
            side_effect=lambda side, source: dict(CATALOG.get(source, {})),
        )
        self.catalog_patch.start()

    def tearDown(self) -> None:
        self.catalog_patch.stop()
        self.db.close()
        self.tmp.cleanup()

    def plan(self, **kwargs):
        opts = {"opponent_id": "cpu", "max_formations": 3,
                "artifact": self.artifact, "catalogue": CATALOG}
        opts.update(kwargs)
        return designer.design_offense(self.db, **opts)

    def test_grid_covers_broad_football_situations(self):
        names = {c["name"] for c in PREGAME_SITUATION_GRID}
        for needed in ("open_first_down", "short_yardage", "third_and_long",
                       "red_zone", "goal_line", "backed_up",
                       "two_minute_trailing", "two_minute_protect"):
            self.assertIn(needed, names)
        for cell in PREGAME_SITUATION_GRID:
            assessment = grid_cell_assessment(cell)
            self.assertEqual(assessment["schema"], "madden.football_situation.v1")

    def test_plan_reports_situation_coverage_and_formation_rationale(self):
        plan = self.plan()
        self.assertIn("situation_coverage", plan)
        self.assertIn("tendency_evidence", plan)
        self.assertIn("provenance", plan)
        covered = [n for n, c in plan["situation_coverage"].items() if c["covered"]]
        self.assertGreaterEqual(len(covered), 6)
        for entry in plan["chosen_formations"]:
            self.assertTrue(entry["situations_addressed"])
            self.assertTrue(entry["why_selected"])
            self.assertNotIn("cell_best", entry)

    def test_formation_value_spans_situations_not_single_play(self):
        plan = self.plan()
        for entry in plan["chosen_formations"]:
            addressed = {s["situation"] for s in entry["situations_addressed"]}
            self.assertGreater(len(addressed), 1, entry["formation"])

    def test_full_formations_preserved_and_unsupported_excluded_visibly(self):
        plan = self.plan()
        for form, plays in plan["book"]["formations"].items():
            src = plan["book"]["formation_sources"][form]
            self.assertEqual(plays, CATALOG[src][form])
        excluded = {e["formation"] for e in plan["excluded_unsupported_formations"]}
        self.assertIn("Goal Line Jumbo", excluded)

    def test_suggested_audibles_are_in_formation_plays(self):
        plan = self.plan()
        for form, audibles in plan["suggested_audibles"].items():
            self.assertTrue(audibles)
            src = plan["book"]["formation_sources"][form]
            for play in audibles:
                self.assertIn(play, CATALOG[src][form])

    def test_tendency_bonus_requires_sufficient_samples(self):
        plan = self.plan()
        evidence = plan["tendency_evidence"]
        self.assertEqual(evidence["total_live_observations"], 0)
        self.assertIsNone(evidence["dominant_class"])
        for entry in plan["chosen_formations"]:
            self.assertEqual(entry["tendency_bonus"], 0.0)

    def test_credible_tendency_shapes_portfolio_with_explicit_evidence(self):
        for _ in range(8):
            self.db.bump_tendency("cpu", "their_coverage", "pressure")
        plan = self.plan()
        evidence = plan["tendency_evidence"]
        self.assertEqual(evidence["dominant_class"], "pressure")
        self.assertGreaterEqual(evidence["total_live_observations"], 8)
        self.assertTrue(evidence["counter_families"])

    def test_max_formations_respected_and_configurable(self):
        plan = self.plan(max_formations=2)
        self.assertLessEqual(len(plan["book"]["formations"]), 2)
        with self.assertRaises(ValueError):
            self.plan(max_formations=designer.MAX_FORMATIONS + 1)


if __name__ == "__main__":
    unittest.main()
