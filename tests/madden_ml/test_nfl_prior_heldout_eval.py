"""Pure-Python pieces of the offline NFL-prior held-out comparison (no xgboost needed)."""

from __future__ import annotations

import importlib.util
import math
import random
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "madden_ml" / "nfl_prior_heldout_eval.py"
_spec = importlib.util.spec_from_file_location("nfl_prior_heldout_eval", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _offense_row(nfl: float, success: bool, down: int = 1, distance: int = 10) -> dict:
    return {
        "side": "offense", "eligibility": "verified_execution", "action_play": "Mesh",
        "down": down, "distance": distance, "yardline": 30, "success": "true" if success else "false",
        "_nfl_success": nfl,
    }


class YardlineTests(unittest.TestCase):
    def test_offense_and_defense_conventions_map_to_yards_to_goal(self):
        self.assertEqual(mod._yardline_100({"side": "offense", "yardline": 21}), 79.0)
        self.assertEqual(mod._yardline_100({"side": "offense", "yardline": 86}), 14.0)
        self.assertEqual(mod._yardline_100({"side": "defense", "yardline": 3}), 3.0)
        self.assertEqual(mod._yardline_100({"side": "defense", "yardline": None}), 70.0)


class CalibrationTests(unittest.TestCase):
    def test_platt_recovers_a_shifted_and_scaled_logit(self):
        rng = random.Random(7)
        xs, ys = [], []
        for _ in range(4000):
            x = rng.uniform(-2, 2)
            xs.append(x)
            ys.append(int(rng.random() < 1 / (1 + math.exp(-(0.4 + 0.5 * x)))))
        a, b = mod.fit_platt(xs, ys, l2=0.0)
        self.assertAlmostEqual(a, 0.4, delta=0.1)
        self.assertAlmostEqual(b, 0.5, delta=0.1)

    def test_calibrated_model_fits_only_verified_training_rows(self):
        train = [_offense_row(0.4, True) for _ in range(30)] + [_offense_row(0.4, False) for _ in range(10)]
        train.append({**_offense_row(0.4, False), "eligibility": "recommendation_only"})
        model = mod.NFLCalibrated()
        model.fit(train)
        p = model.predict(_offense_row(0.4, True))["success"]
        self.assertGreater(p, 0.6)
        self.assertLess(p, 0.8)

    def test_raw_model_passes_the_nfl_score_through(self):
        self.assertEqual(mod.NFLRaw().predict(_offense_row(0.37, True))["success"], 0.37)


class MappingTests(unittest.TestCase):
    def test_madden_plays_map_to_the_only_nfl_play_input_run_or_pass(self):
        self.assertEqual(mod.nfl_play_input("Inside Zone"), "run")
        self.assertEqual(mod.nfl_play_input("Four Verticals"), "pass")
        self.assertEqual(mod.nfl_play_input("HB Screen"), "pass")
        self.assertEqual(mod.nfl_play_input("Totally Unknown Name"), "mixture")
        self.assertEqual(mod.nfl_play_input(None), "mixture")
        self.assertEqual(set(mod.NFL_PLAY_INPUT.values()), {"run", "pass", "mixture"})

    def test_situation_slices(self):
        self.assertEqual(mod.situation_slices({"side": "offense", "distance": 1, "yardline": 85}),
                         {"short_yardage": "yes", "red_zone": "yes"})
        self.assertEqual(mod.situation_slices({"side": "offense", "distance": 10, "yardline": 30}),
                         {"short_yardage": "no", "red_zone": "no"})
        self.assertEqual(mod.situation_slices({"side": "offense", "distance": None, "yardline": None}),
                         {"short_yardage": "no", "red_zone": "no"})


class TunedBlendTests(unittest.TestCase):
    def test_zero_strength_is_the_experimental_model(self):
        self.assertAlmostEqual(mod.blend(0.62, 0.30, 0.5, 0.0), 0.62)

    def test_prior_at_the_base_rate_changes_nothing(self):
        self.assertAlmostEqual(mod.blend(0.62, 0.5, 0.5, 1.0), 0.62)

    def test_strength_is_chosen_without_the_test_game(self):
        rows = []
        for g in range(4):
            for i in range(40):
                success = i % 2 == 0
                rows.append({**_offense_row(0.8 if success else 0.2, success), "game_id": f"g{g}"})
        choices: list[float] = []
        model = mod.ExperimentalNFLTuned({f"g{g}": g for g in range(4)}, choices)
        model.fit(rows)
        # An informative NFL score earns the strongest allowed prior on the validation game.
        self.assertEqual(choices, [max(mod.PRIOR_STRENGTHS)])
        self.assertGreater(model.predict({**_offense_row(0.8, True), "game_id": "test"})["success"], 0.6)


if __name__ == "__main__":
    unittest.main()
