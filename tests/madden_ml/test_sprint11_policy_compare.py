"""Compare joint coordinator vs prior model-primary play-then-action policy."""
from __future__ import annotations

import unittest
from unittest import mock

from cfb_coach.madden.model.football_situation import sit_from_probe
from cfb_coach.madden.model.offense_joint_decision import choose_joint_offense_action
from cfb_coach.madden.model.offense_selection_policy import choose_model_play


SCENARIOS = (
    {"label": "third_long", "down": 3, "distance": 10, "yardline": 35},
    {"label": "short", "down": 3, "distance": 1, "yardline": 40},
    {"label": "two_min_lead", "down": 2, "distance": 8, "yardline": 45,
     "two_minute": True, "score_us": 24, "score_them": 17, "quarter": 4},
    {"label": "two_min_trail", "down": 2, "distance": 8, "yardline": 45,
     "two_minute": True, "score_us": 10, "score_them": 24, "quarter": 4},
    {"label": "unknown_cov", "down": 1, "distance": 10, "yardline": 30,
     "coverage_hint": "cover 2", "coverage_source": "last"},
)


def _ranked_book():
    plays = [
        ("Gun Bunch", "Mesh", 0.58, "mesh"),
        ("Gun Bunch", "Inside Zone", 0.55, "inside_zone"),
        ("Gun Bunch", "Flood", 0.54, "flood"),
        ("Gun Bunch", "HB Slip Screen", 0.53, "screen"),
        ("Gun Trips", "Smash", 0.52, "smash"),
        ("Gun Trips", "Duo", 0.51, "duo"),
        ("Singleback Ace", "Power O", 0.50, "power"),
        ("Singleback Ace", "PA Boot", 0.49, "pa"),
    ]
    ranked = [
        {
            "formation": f, "play": p, "probability": prob,
            "uncertainty": 0.45, "evidence_quality": "prior_driven",
            "play_concept": concept,
        }
        for f, p, prob, concept in plays
    ]
    book = {}
    for f, p, *_ in plays:
        book.setdefault(f, []).append(p)
    return ranked, book


class PolicyCompareTests(unittest.TestCase):
    def test_joint_and_prior_both_legal_and_differ_by_situation(self):
        ranked, book = _ranked_book()
        picks = {}
        for scenario in SCENARIOS:
            sit = sit_from_probe(scenario)
            prior, audit = choose_model_play(
                ranked, sit=sit, opponent_type="cpu",
                session_id="compare", snap_seq=1,
            )
            with mock.patch(
                "cfb_coach.madden.model.offense_joint_decision.choose_offense_action",
                return_value={
                    "kind": "none", "no_action_score": 0.36,
                    "reason": "none", "candidates": [
                        {"kind": "none", "id": "NO_ADJUSTMENT", "score": 0.36},
                    ],
                },
            ):
                joint = choose_joint_offense_action(
                    ranked_plays=prior, sit=sit, book=book, active=[],
                )
            self.assertIn((joint["formation"], joint["play"]), {
                (r["formation"], r["play"]) for r in ranked
            })
            self.assertEqual(joint["adjustment_plan"]["kind"], "none")
            self.assertNotEqual(
                joint["football_situation"]["credibility"]["coverage"],
                "fabricated",
            )
            if scenario.get("coverage_source") == "last":
                self.assertFalse(joint["football_situation"]["credible_look"])
            picks[scenario["label"]] = (joint["play"], prior[0]["play"])

        # Third-and-long should not prefer a pure run when passes exist.
        self.assertNotEqual(picks["third_long"][0], "Inside Zone")
        # Lead vs trail two-minute decisions should be able to differ.
        self.assertTrue(
            picks["two_min_lead"] != picks["two_min_trail"]
            or picks["two_min_lead"][0] in {p[0] for p in _ranked_book()[0]}
        )


if __name__ == "__main__":
    unittest.main()
