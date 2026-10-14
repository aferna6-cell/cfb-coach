"""Short-yardage offense: conversion passes compete with runs; 3rd-long stays pass-first."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from cfb_coach.madden.catalog import is_run
from cfb_coach.madden.model.experimental_live import situational_offense_candidates
from cfb_coach.madden.model.experimental_model import rank_candidates, train_experimental
from cfb_coach.madden.model.football_situation import (
    is_short_yardage_pass,
    selection_adjustment,
)
from cfb_coach.madden.model.offense_selection_policy import choose_model_play
from cfb_coach.madden.playcaller import _offense_pool, situation_key
from cfb_coach.madden.data import load_meta_baseline

BOOK = {
    "Gun Doubles Clamp Stack": [
        "Inside Zone", "HB Mid Zone", "Four Verticals", "Flood Seam",
        "Stick", "RPO Alert Bubble", "PA Drag Wheel", "Mtn Shuffle Verts Smash",
    ],
    "Singleback Deuce Close": [
        "Counter Y", "Stick", "PA Drag Wheel", "Mesh HB Rail", "HB Slip Screen",
    ],
    "I Form Y Off Close": ["HB Toss", "DBL Outs", "PA X Deep Out"],
    "Maryland I Close": ["Mtn HB Zone WK", "Mesh HB Rail", "Mtn PA Sprint HB Flat"],
}


def sit(down: int, distance: int, **kwargs):
    fields = {
        "down": down, "distance": distance, "yardline": 40, "side": "offense",
        "short_yardage": distance <= 2 and down in (2, 3, 4),
        "long_yardage": distance >= 7 and down in (3, 4),
        "red_zone": False, "goal_line": False, "two_minute": False,
        "coverage_hint": None, "coverage_source": "none", "extras": {},
        "raw": f"{down}&{distance}",
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


class ShortYardagePoolTests(unittest.TestCase):
    def test_short_yardage_pool_keeps_runs_and_conversion_passes(self):
        pairs, bonus = situational_offense_candidates(sit(3, 1), BOOK)
        plays = {p for _, p in pairs}
        self.assertIn("Inside Zone", plays)
        self.assertIn("Stick", plays)
        self.assertIn("PA Drag Wheel", plays)
        self.assertIn("RPO Alert Bubble", plays)
        self.assertTrue(is_short_yardage_pass("Stick"))
        self.assertTrue(is_short_yardage_pass("PA Drag Wheel"))
        self.assertTrue(is_short_yardage_pass("RPO Alert Bubble"))
        self.assertFalse(is_short_yardage_pass("Four Verticals"))
        self.assertFalse(is_short_yardage_pass("Inside Zone"))

    def test_short_yardage_bonus_is_not_run_only(self):
        og = load_meta_baseline()["offense_gameplan"]
        situation = sit(3, 1)
        _pool, bonus = _offense_pool(situation, BOOK, og, situation_key(situation), None)
        run_b = bonus.get(("Gun Doubles Clamp Stack", "Inside Zone"), 0)
        stick_b = bonus.get(("Gun Doubles Clamp Stack", "Stick"), 0)
        pa_b = bonus.get(("Singleback Deuce Close", "PA Drag Wheel"), 0)
        deep_b = bonus.get(("Gun Doubles Clamp Stack", "Four Verticals"), 0)
        self.assertGreaterEqual(stick_b, 0.12)
        self.assertGreaterEqual(pa_b, 0.12)
        self.assertGreaterEqual(run_b, 0.12)
        self.assertLess(deep_b, stick_b)

    def test_third_and_long_still_excludes_runs(self):
        pairs, _ = situational_offense_candidates(sit(3, 12), BOOK)
        self.assertTrue(pairs)
        self.assertTrue(all(not is_run(p) for _, p in pairs))

    def test_selection_adjustment_lets_passes_compete_on_third_short(self):
        run_d, _ = selection_adjustment("Inside Zone", sit(3, 1))
        pass_d, reason = selection_adjustment("Stick", sit(3, 1))
        deep_d, _ = selection_adjustment("Four Verticals", sit(3, 1))
        self.assertEqual(run_d, pass_d)
        self.assertGreater(pass_d, deep_d)
        self.assertIn("play-action, or RPO", reason)

    def test_recent_run_exposure_does_not_use_a_quota(self):
        ranked = [
            {"formation": "Gun Doubles Clamp Stack", "play": "Inside Zone",
             "probability": 0.51, "play_concept": "run_concept", "uncertainty": 0.7,
             "evidence_quality": "prior_driven"},
            {"formation": "Gun Doubles Clamp Stack", "play": "Stick",
             "probability": 0.50, "play_concept": "stick", "uncertainty": 0.7,
             "evidence_quality": "prior_driven"},
        ]
        recent = [
            ("Gun Doubles Clamp Stack", "Inside Zone"),
            ("Gun Doubles Clamp Stack", "HB Mid Zone"),
            ("I Form Y Off Close", "HB Toss"),
        ]
        scored, _audit = choose_model_play(
            ranked, recent_calls=recent, sit=sit(3, 1), opponent_type="human",
            session_id="wc", snap_seq=9,
        )
        run = next(r for r in scored if r["play"] == "Inside Zone")
        stick = next(r for r in scored if r["play"] == "Stick")
        self.assertGreater(run["selection_penalty"], 0)
        self.assertGreater(stick["selection_score"], run["selection_score"] - 0.2)

    def test_baseline_artifact_still_ranks_eligible_short_yardage_passes(self):
        artifact = train_experimental([], side="offense")
        self.assertEqual(artifact.n_supervised, 0)
        pairs, bonuses = situational_offense_candidates(sit(3, 1), BOOK)
        ranked = rank_candidates(
            artifact, pairs, down=3, distance=1, heuristic_bonuses=bonuses,
        )
        families = {row["play_family"] for row in ranked[:8]}
        plays = [row["play"] for row in ranked]
        self.assertTrue(any(not is_run(p) for p in plays[:6]))
        self.assertTrue({"run", "pass", "rpo", "pa"} & {
            row["play_concept"] if row["play_concept"] != "run_concept" else "run"
            for row in ranked[:10]
        } or any(is_short_yardage_pass(p) for p in plays[:8]))
        self.assertIn("Stick", plays)
        self.assertTrue(any(is_short_yardage_pass(p) for p in plays))
        del families
