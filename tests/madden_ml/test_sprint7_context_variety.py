"""Sprint 7: situational offense + broad, model-driven CPU exploration.

Tests isolate the structural bug: the original down/distance main effect
was the same for all plays. The new interaction and hard 3rd-and-long guard
make the selected action genuinely situation-dependent.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.situation import Situation
from cfb_coach.madden.catalog import is_run
from cfb_coach.madden.model import experimental_live, inference
from cfb_coach.madden.model.experimental_live import situational_offense_candidates
from cfb_coach.madden.model.schema import CoachingMode
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.model.experimental_model import (
    ExperimentalArtifact, predict_success, train_experimental,
)
from cfb_coach.madden.model.offense_selection_policy import (
    choose_model_play, summarize_call_variety,
)


def situation(down=3, distance=11, two_minute=False):
    return Situation(
        raw=f"{down}&{distance} my 40", side="offense",
        down=down, distance=distance, yardline=40,
        two_minute=two_minute,
    )


def varied():
    plays = [
        ("Gun Bunch", "Mesh", "mesh"),
        ("Gun Bunch", "Four Verticals", "vert"),
        ("Gun Trips", "Flood", "flood"),
        ("Gun Tight", "HB Mid Zone", "run_concept"),
        ("Gun Doubles", "HB Slip Screen", "screen"),
        ("Gun Trips", "Quick Slants", "slant"),
        ("Singleback", "PA Boot", "pa"),
        ("Gun Tight", "Bench", "flood"),
        ("Gun Doubles", "Texas Y-Stutter Wheel", "wheel"),
        ("Gun Trips", "Inside Zone", "run_concept"),
        ("Gun Tight", "Mills Double Out", "pass"),
        ("Gun Bunch", "Corner Post", "pass"),
        ("Gun Trips", "Slot Fade H Wheel", "wheel"),
        ("Singleback", "Stretch", "run_concept"),
        ("Gun Bunch", "Smash", "smash"),
        ("Pistol", "Cross", "cross"),
    ]
    return [
        {"formation": form, "play": play, "probability": .57 + .003*(i % 6),
         "play_concept": concept, "uncertainty": .75,
         "evidence_quality": "empirical_light"}
        for i, (form, play, concept) in enumerate(plays)
    ]


class SituationEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.book = {"Gun Bunch": ["HB Dive", "Inside Zone", "Mesh", "Four Verticals",
                                   "HB Slip Screen", "Flood"]}

    def test_third_and_long_never_selects_run_if_pass_is_in_book(self):
        pairs, _ = situational_offense_candidates(situation(3, 12), self.book)
        self.assertTrue(pairs)
        self.assertTrue(all(not is_run(p) for _, p in pairs))
        self.assertIn(("Gun Bunch", "Mesh"), pairs)
        self.assertIn(("Gun Bunch", "Four Verticals"), pairs)

    def test_fourth_and_long_no_run(self):
        pairs, _ = situational_offense_candidates(situation(4, 10), self.book)
        self.assertTrue(pairs)
        self.assertTrue(all(not is_run(p) for _, p in pairs))

    def test_short_yardage_retains_runs(self):
        pairs, _ = situational_offense_candidates(situation(3, 1), self.book)
        self.assertIn(("Gun Bunch", "Inside Zone"), pairs)

    def test_no_valid_pass_preserves_pool_instead_of_inventing_play(self):
        book = {"Singleback": ["HB Dive", "Inside Zone"]}
        pairs, _ = situational_offense_candidates(situation(3, 11), book)
        self.assertEqual({p for _, p in pairs}, {"HB Dive", "Inside Zone"})

    def test_two_minute_call_respects_lead_instead_of_always_penalizing_run(self):
        from cfb_coach.madden.model.offense_selection_policy import _situational_adjustment

        lead = situation(2, 6, two_minute=True)
        lead.score_us, lead.score_them = 28, 7
        trail = situation(2, 6, two_minute=True)
        trail.score_us, trail.score_them = 7, 28
        unknown = situation(2, 6, two_minute=True)
        up, _ = _situational_adjustment("Inside Zone", lead)
        down, _ = _situational_adjustment("Inside Zone", trail)
        unk, _ = _situational_adjustment("Inside Zone", unknown)
        self.assertGreater(up, 0)
        self.assertLess(down, 0)
        self.assertEqual(unk, 0)

    def test_situational_score_penalizes_screen_behind_sticks(self):
        opts = [
            {"formation":"Gun Bunch","play":"HB Slip Screen",
             "probability":.59,"play_concept":"screen","uncertainty":.75},
            {"formation":"Gun Bunch","play":"Four Verticals",
             "probability":.59,"play_concept":"vert","uncertainty":.75},
        ]
        ranked, audit = choose_model_play(
            opts, sit=situation(3, 11), session_id="game", snap_seq=3
        )
        s = next(x for x in ranked if x["play"] == "HB Slip Screen")
        v = next(x for x in ranked if x["play"] == "Four Verticals")
        self.assertLess(s["situation_adjustment"], 0)
        self.assertGreater(v["situation_adjustment"], 0)
        self.assertGreater(v["selection_score"], s["selection_score"])
        self.assertIn(audit["top_selected"][1], ("Four Verticals", "HB Slip Screen"))


class LiveGameIdentityTests(unittest.TestCase):
    def test_live_model_receives_real_session_and_snap_identity(self):
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "live.db", seed={"opponents": {
                "cpu": {"display_name":"CPU","team_now":"DET","skill":"cpu",
                        "confidence":"low","profile_json":"{}"}
            }})
            try:
                sit = situation(1, 10)
                sit.extras = {"session_id": "session-model-123"}
                call = MaddenCall(
                    "offense", "Gun Bunch", "Mesh", "No adj", "Read", "heuristic"
                )
                with (
                    mock.patch.object(
                        inference,
                        "resolve_mode", return_value=CoachingMode.EXPERIMENTAL,
                    ),
                    mock.patch.object(
                        experimental_live, "apply_experimental_offense",
                        return_value=(call, object()),
                    ) as apply,
                ):
                    experimental_live.maybe_apply_experimental(
                        call=call, sit=sit, opponent_id="cpu", db=db,
                        book={"Gun Bunch": ["Mesh"]},
                    )
                self.assertEqual(apply.call_args.kwargs["session_id"], "session-model-123")
                self.assertEqual(apply.call_args.kwargs["game_id"], "session-model-123")
                self.assertEqual(apply.call_args.kwargs["snap_seq"], 1)
            finally:
                db.close()


class ModelContextInteractionTests(unittest.TestCase):
    def test_concept_by_down_distance_changes_play_ranking(self):
        art = ExperimentalArtifact(
            global_rate=0.5, evidence_quality="empirical_light",
            play_family={"run":[12, 24], "pass":[12, 24]},
            play_concept={"run_concept":[12, 24], "wheel":[12, 24]},
            concept_down_distance={
                "wheel|3_long":[10, 10],
                "run_concept|3_long":[0, 10],
                "wheel|1_long":[0, 10],
                "run_concept|1_long":[10, 10],
            },
        )
        third_wheel = predict_success(art, formation="Gun Bunch",play="Wheel",
                                      down=3,distance=11)
        third_run = predict_success(art, formation="Gun Bunch",play="Inside Zone",
                                    down=3,distance=11)
        first_wheel = predict_success(art, formation="Gun Bunch",play="Wheel",
                                      down=1,distance=10)
        first_run = predict_success(art, formation="Gun Bunch",play="Inside Zone",
                                    down=1,distance=10)
        self.assertGreater(third_wheel["probability"], third_run["probability"])
        self.assertGreater(first_run["probability"], first_wheel["probability"])
        self.assertGreater(third_wheel["components"]["situation_interaction_trials"], 0)
        self.assertIn("concept_down_distance", art.to_dict())
        self.assertEqual(
            ExperimentalArtifact.from_dict(art.to_dict()).concept_down_distance,
            art.concept_down_distance,
        )

    def test_training_populates_context_interactions_from_verified_examples(self):
        rows = [
            {"side":"offense", "supervised_eligible":True, "success":"true",
             "action_play":"Four Verticals", "down":3, "distance":11,
             "opponent_type":"cpu", "opponent_id":"cpu"},
            {"side":"offense", "supervised_eligible":True, "success":"false",
             "action_play":"Inside Zone", "down":3, "distance":11,
             "opponent_type":"cpu", "opponent_id":"cpu"},
        ]
        artifact = train_experimental(rows)
        self.assertEqual(artifact.n_supervised, 2)
        self.assertEqual(artifact.concept_down_distance["vert|3_long"], [1.0, 1.0])
        self.assertEqual(artifact.concept_down_distance["run_concept|3_long"], [0.0, 1.0])
        self.assertEqual(artifact.family_down_distance["run|3_long"], [0.0, 1.0])

    def test_old_artifact_without_new_buckets_still_loads(self):
        old = ExperimentalArtifact.from_dict(
            {"global_rate": .5, "model_version": "old", "play_concept":{"screen":[4,8]}}
        )
        self.assertEqual(old.concept_down_distance,{})
        self.assertGreater(predict_success(old, formation="Gun",play="Mesh",
                                           down=3,distance=10)["probability"], 0)


class ModelVarietyTests(unittest.TestCase):
    def test_replay_is_reproducible(self):
        options = varied()
        recent = [("Gun Bunch", "Mesh"), ("Gun Trips", "Flood")]
        a, audit_a = choose_model_play(
            options, recent_calls=recent, sit=situation(1, 10),
            session_id="g12", snap_seq=31, opponent_type="cpu",
        )
        b, audit_b = choose_model_play(
            options, recent_calls=recent, sit=situation(1, 10),
            session_id="g12", snap_seq=31, opponent_type="cpu",
        )
        self.assertEqual(a[0]["play"], b[0]["play"])
        self.assertEqual(audit_a, audit_b)
        self.assertEqual(audit_a["policy"],"model_primary_contextual_variety.v2")

    def test_63_cpu_snap_simulation_uses_broader_play_concepts(self):
        options = varied()
        recent = []
        calls = []
        for n in range(1, 64):
            ranked, audit = choose_model_play(
                options, recent_calls=recent, sit=situation(1 if n % 4 else 2, 10),
                session_id="cpu-demo", snap_seq=n, opponent_type="cpu",
            )
            key = (ranked[0]["formation"], ranked[0]["play"])
            self.assertIn(key, [(x["formation"], x["play"]) for x in options])
            self.assertGreaterEqual(audit["shortlist_count"], 2)
            calls.append(key)
            recent.insert(0, key)
            recent = recent[:20]
        self.assertGreaterEqual(len(set(calls)), 9)
        self.assertGreaterEqual(len({f for f, _ in calls}), 4)

    def test_postgame_reports_third_long_and_model_variety_without_execution_claim(self):
        observed = [
            {"snap_id":"g-01","final_formation":"Gun Bunch","final_play":"HB Dive"},
            {"snap_id":"g-02","final_formation":"Gun Bunch","final_play":"Mesh"},
            {"snap_id":"g-03","final_formation":"Gun Doubles","final_play":"HB Slip Screen"},
            {"snap_id":"g-04","final_formation":"Gun Bunch","final_play":"Mesh"},
        ]
        context = {
            "g-01":{"down":3,"distance":11},
            "g-02":{"down":3,"distance":11},
            "g-03":{"down":1,"distance":10},
        }
        quality = summarize_call_variety(observed, by_snap_situation=context)
        self.assertEqual(quality["recommended_calls"], 4)
        self.assertEqual(quality["distinct_plays"], 3)
        self.assertEqual(quality["distinct_formations"], 2)
        self.assertEqual(quality["known_third_or_fourth_long"], 2)
        self.assertEqual(quality["third_or_fourth_long_run_recommendations"], 1)
        self.assertEqual(quality["max_consecutive_same_play"], 1)
        self.assertIn("recommended", quality["note"])

    def test_recent_calls_are_session_scoped_when_available(self):
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "test.db", seed={"opponents": {
                "cpu": {"display_name":"CPU","team_now":"DET","skill":"cpu",
                        "confidence":"low","profile_json":"{}"}
            }})
            try:
                from cfb_coach.madden.model.offense_selection_policy import _recent_calls
                for sid, play in [("g-old","HB Slip Screen"),("g-new","Mesh")]:
                    db.log_snap(
                        opponent_id="cpu", side="offense", situation_raw="1&10",
                        formation="Gun Bunch", play=play, our_call="test",
                        result="+4", session_id=sid
                    )
                self.assertEqual(_recent_calls(db, "cpu", session_id="g-new"),
                                 [("Gun Bunch", "Mesh")])
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
