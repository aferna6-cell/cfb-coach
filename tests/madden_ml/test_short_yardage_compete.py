"""Short-yardage offense: passes, PA, and RPO compete with runs. No quotas."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from cfb_coach.madden.catalog import is_run
from cfb_coach.madden.model.experimental_model import (
    ExperimentalArtifact,
    predict_success,
    rank_candidates,
    train_experimental,
)
from cfb_coach.madden.model.football_situation import SELECTION_PRIORS, selection_adjustment
from cfb_coach.madden.model.offense_selection_policy import choose_model_play
from cfb_coach.madden.playcaller import _offense_pool, SIT_BONUS


def _sit(down=3, distance=1, **kwargs):
    fields = {
        "down": down, "distance": distance, "yardline": 45,
        "short_yardage": distance <= 2, "long_yardage": distance >= 7,
        "goal_line": False, "red_zone": False,
        "two_minute": False, "score_us": None, "score_them": None,
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


BOOK = {
    "Gun Bunch": [
        "Inside Zone", "HB Dive", "Mesh", "Stick", "PA Drag Wheel",
        "RPO Peek Bubble", "Four Verticals", "Mtn Flood",
    ],
    "Singleback Deuce Close": ["Counter Y", "Stick", "PA Boot"],
}


class CandidatePoolTests(unittest.TestCase):
    def test_short_yardage_pool_keeps_runs_and_passes(self):
        from cfb_coach.madden.data import load_meta_baseline
        from cfb_coach.madden.playcaller import situation_key

        sit = _sit()
        og = load_meta_baseline()["offense_gameplan"]
        pool, bonus = _offense_pool(sit, BOOK, og, situation_key(sit), None)
        plays = {play for _form, play in pool}
        self.assertIn("Inside Zone", plays)
        self.assertIn("Mesh", plays)
        self.assertIn("PA Drag Wheel", plays)
        self.assertIn("RPO Peek Bubble", plays)
        self.assertEqual(SIT_BONUS, 0.12)
        self.assertGreater(bonus.get(("Gun Bunch", "Inside Zone"), 0), 0)

    def test_third_and_long_pool_still_drops_runs_for_experimental_candidates(self):
        from cfb_coach.madden.model.experimental_live import situational_offense_candidates

        pairs, _bonus = situational_offense_candidates(_sit(down=3, distance=11), BOOK)
        self.assertTrue(pairs)
        self.assertTrue(all(not is_run(play) for _form, play in pairs))


class RankingPriorTests(unittest.TestCase):
    def test_quick_pass_pa_and_rpo_compete_with_runs(self):
        sit = _sit(down=3, distance=1)
        run, _ = selection_adjustment("Inside Zone", sit)
        quick, _ = selection_adjustment("Stick", sit)
        mesh, _ = selection_adjustment("Mesh", sit)
        pa, _ = selection_adjustment("PA Drag Wheel", sit)
        rpo, _ = selection_adjustment("RPO Peek Bubble", sit)
        deep, _ = selection_adjustment("Four Verticals", sit)
        self.assertEqual(run, SELECTION_PRIORS["short_yardage_run"]["delta"])
        self.assertEqual(quick, SELECTION_PRIORS["short_yardage_quick_pass"]["delta"])
        self.assertEqual(mesh, SELECTION_PRIORS["short_yardage_quick_pass"]["delta"])
        self.assertEqual(pa, SELECTION_PRIORS["short_yardage_pa"]["delta"])
        self.assertEqual(rpo, SELECTION_PRIORS["short_yardage_rpo"]["delta"])
        self.assertLess(deep, 0)
        self.assertGreater(quick, 0)
        self.assertGreater(pa, 0)
        self.assertGreater(rpo, 0)
        self.assertLess(abs(run - quick), 0.01)
        self.assertLess(abs(run - rpo), 0.01)

    def test_no_fixed_run_pass_quota_in_sampling(self):
        sit = _sit(down=3, distance=1)
        ranked = [
            {"formation": "Gun Bunch", "play": "Inside Zone", "probability": 0.51,
             "play_concept": "run_concept", "uncertainty": 0.8, "evidence_quality": "prior_driven"},
            {"formation": "Gun Bunch", "play": "Stick", "probability": 0.50,
             "play_concept": "stick", "uncertainty": 0.8, "evidence_quality": "prior_driven"},
            {"formation": "Gun Bunch", "play": "PA Drag Wheel", "probability": 0.50,
             "play_concept": "pa", "uncertainty": 0.8, "evidence_quality": "prior_driven"},
            {"formation": "Gun Bunch", "play": "RPO Peek Bubble", "probability": 0.50,
             "play_concept": "rpo", "uncertainty": 0.8, "evidence_quality": "prior_driven"},
        ]
        chosen_plays = set()
        for seq in range(12):
            ordered, _audit = choose_model_play(
                ranked, sit=sit, opponent_type="human",
                session_id="short-yard", snap_seq=seq,
            )
            chosen_plays.add(ordered[0]["play"])
        self.assertGreaterEqual(len(chosen_plays), 2)
        self.assertTrue(any(not is_run(play) for play in chosen_plays))

    def test_baseline_artifact_without_conversion_buckets_is_unchanged_on_early_down(self):
        baseline = ExperimentalArtifact(
            global_rate=0.5, evidence_quality="prior_driven",
            play_family={"run": [4, 8], "pass": [4, 8]},
            play_concept={"run_concept": [4, 8], "stick": [4, 8]},
        )
        early_run = predict_success(baseline, formation="Gun Bunch", play="Inside Zone",
                                    down=1, distance=10)
        early_pass = predict_success(baseline, formation="Gun Bunch", play="Stick",
                                     down=1, distance=10)
        self.assertEqual(early_run["components"].get("conversion_trials"), 0.0)
        self.assertEqual(early_pass["components"].get("conversion_trials"), 0.0)

    def test_conversion_layer_lets_short_passes_compete_when_evidence_exists(self):
        empty = ExperimentalArtifact(
            global_rate=0.5, evidence_quality="empirical_light",
            play_family={"run": [8, 16], "pass": [8, 16]},
            play_concept={"run_concept": [8, 16], "stick": [8, 16]},
        )
        with_conv = ExperimentalArtifact(
            global_rate=0.5, evidence_quality="empirical_light",
            play_family={"run": [8, 16], "pass": [8, 16]},
            play_concept={"run_concept": [8, 16], "stick": [8, 16]},
            conversion_family_dd={"run|3_short": [2, 10], "pass|3_short": [9, 10]},
            conversion_concept_dd={"run_concept|3_short": [2, 10], "stick|3_short": [9, 10]},
        )
        empty_run = predict_success(empty, formation="Gun Bunch", play="Inside Zone",
                                    down=3, distance=1)
        empty_pass = predict_success(empty, formation="Gun Bunch", play="Stick",
                                     down=3, distance=1)
        conv_run = predict_success(with_conv, formation="Gun Bunch", play="Inside Zone",
                                   down=3, distance=1)
        conv_pass = predict_success(with_conv, formation="Gun Bunch", play="Stick",
                                    down=3, distance=1)
        self.assertGreater(empty_run["probability"], empty_pass["probability"] - 0.05)
        self.assertGreater(conv_pass["probability"], conv_run["probability"])
        ranked = rank_candidates(
            with_conv,
            [("Gun Bunch", "Inside Zone"), ("Gun Bunch", "Stick")],
            down=3, distance=1,
        )
        self.assertEqual(ranked[0]["play"], "Stick")

    def test_train_experimental_records_conversion_buckets(self):
        rows = [
            {
                "supervised_eligible": True, "side": "offense",
                "executed_play": "Stick", "success": "true",
                "executed_verification": "verified", "executed_status": "identified",
                "label_available": True, "opponent_type": "human",
                "opponent_id": "john", "game_id": "wc1",
                "down": 3, "distance": 1, "yards": 2, "result": "+2",
                "eligibility": "verified_execution",
            },
            {
                "supervised_eligible": True, "side": "offense",
                "executed_play": "Inside Zone", "success": "false",
                "executed_verification": "verified", "executed_status": "identified",
                "label_available": True, "opponent_type": "human",
                "opponent_id": "john", "game_id": "wc1",
                "down": 3, "distance": 1, "yards": 0, "result": "stop",
                "eligibility": "verified_execution",
            },
        ]
        from cfb_coach.madden.model import dataset as dataset_mod
        with unittest.mock.patch.object(dataset_mod, "supervised_rows", return_value=rows):
            art = train_experimental(rows, side="offense")
        self.assertTrue(art.conversion_family_dd or art.conversion_concept_dd)


if __name__ == "__main__":
    unittest.main()
