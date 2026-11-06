"""Per-opponent learned tendencies change ranking only when a sample exists."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import defense_v1, experimental_model
from cfb_coach.madden.model.experimental_model import ExperimentalArtifact
from cfb_coach.madden.model.opponent_tendencies import (
    OFFENSE_LOGIT_CAP,
    build_defense_profile,
    build_offense_profile,
    coverage_bucket_name,
    defense_success_delta,
    offense_tendency_delta,
    opponent_context_key,
)
from cfb_coach.situation import Situation


def _artifact() -> ExperimentalArtifact:
    return ExperimentalArtifact(
        evidence_quality="empirical",
        global_rate=0.5,
        concept_vs_cov={
            "mesh|cover2_man": [9.0, 10.0],
            "vert|cover2_man": [1.0, 10.0],
        },
    )


def _pairs() -> list[tuple[str, str]]:
    return [("Gun", "Four Verticals"), ("Gun", "Mesh")]


def _third() -> Situation:
    return Situation(raw="3&7", side="offense", down=3, distance=7, yardline=40)


class OffenseTendencyTests(unittest.TestCase):
    def test_cover_2_man_is_its_own_bucket(self) -> None:
        self.assertEqual(coverage_bucket_name("Cover 2 Man"), "cover2_man")

    def test_unknown_opponent_does_not_change_the_ranking(self) -> None:
        art = _artifact()
        sit = _third()
        plain = experimental_model.rank_candidates(
            art, _pairs(), down=3, distance=7, yardline=40, opponent_id="nobody",
        )
        empty = experimental_model.rank_candidates(
            art, _pairs(), down=3, distance=7, yardline=40, opponent_id="nobody",
            tendency={"n": 0, "mix": {}, "shrink": 0.0, "summary": "no learned tendencies"},
        )
        self.assertEqual([row["play"] for row in plain], [row["play"] for row in empty])
        self.assertEqual(
            [row["probability"] for row in plain],
            [row["probability"] for row in empty],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "empty.db")
            try:
                profile = build_offense_profile(db, "nobody", sit)
                self.assertEqual(profile["n"], 0)
                self.assertEqual(profile["summary"], "no learned tendencies")
                loaded = experimental_model.rank_candidates(
                    art, _pairs(), down=3, distance=7, yardline=40,
                    opponent_id="nobody", tendency=profile,
                )
            finally:
                db.close()
        self.assertEqual([row["play"] for row in plain], [row["play"] for row in loaded])
        self.assertEqual(
            [row["probability"] for row in plain],
            [row["probability"] for row in loaded],
        )
        self.assertEqual(opponent_context_key("nobody").opponent_id, "nobody")
        self.assertEqual(opponent_context_key("nobody").table_version, "learned_tendencies.v1")
        self.assertIsNone(opponent_context_key("nobody").roster_snapshot_id)

    def test_third_down_cover_2_man_lifts_the_beater(self) -> None:
        art = _artifact()
        plain = experimental_model.rank_candidates(
            art, _pairs(), down=3, distance=7, yardline=40, opponent_id="ryan",
        )
        self.assertEqual(plain[0]["play"], "Four Verticals")
        profile = {
            "n": 10,
            "shrink": 10 / 18,
            "mix": {"cover2_man": 1.0},
            "labels": {"cover2_man": "Cover 2 Man"},
            "scope": "this down",
            "play_rates": {},
            "summary": "",
        }
        profile["summary"] = (
            "this down: Cover 2 Man 100% "
            f"(n=10, shrink={profile['shrink']:.2f})"
        )
        shifted = experimental_model.rank_candidates(
            art, _pairs(), down=3, distance=7, yardline=40,
            opponent_id="ryan", tendency=profile,
        )
        self.assertEqual(shifted[0]["play"], "Mesh")
        self.assertEqual(shifted[1]["play"], "Four Verticals")
        self.assertIn("Cover 2 Man", shifted[0]["tendency_summary"])
        mesh = offense_tendency_delta(
            art, "Gun", "Mesh", baseline=0.5, profile=profile,
        )
        vert = offense_tendency_delta(
            art, "Gun", "Four Verticals", baseline=0.5, profile=profile,
        )
        self.assertGreater(mesh, 0)
        self.assertLess(vert, 0)
        self.assertLessEqual(abs(mesh), OFFENSE_LOGIT_CAP)
        self.assertLessEqual(abs(vert), OFFENSE_LOGIT_CAP)
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "ryan.db")
            try:
                db.bump_tendency("ryan", "their_coverage|open|normal|3", "Cover 2 Man", amount=8)
                db.bump_tendency("ryan", "their_coverage", "Cover 2 Man", amount=8)
                loaded = build_offense_profile(db, "ryan", _third())
            finally:
                db.close()
        self.assertGreaterEqual(loaded["n"], 8)
        self.assertGreater(loaded["mix"].get("cover2_man", 0), 0.5)
        self.assertIn("Cover 2 Man", loaded["summary"])
        from_db = experimental_model.rank_candidates(
            art, _pairs(), down=3, distance=7, yardline=40,
            opponent_id="ryan", tendency=loaded,
        )
        self.assertEqual(from_db[0]["play"], "Mesh")


def _defense_artifact() -> dict:
    candidates = [
        {"formation": "Dime 3-2 Odd", "play": "Cover 3", "family": "cover3", "package": "dime"},
        {"formation": "4-3 Over", "play": "Cover 1", "family": "man", "package": "base"},
    ]
    payload = {
        "schema": defense_v1.SCHEMA_V2,
        "policy": defense_v1.POLICY,
        "target": defense_v1.TARGET,
        "intercept": 0.0,
        "features": ["red_zone"],
        "coefficients": {"red_zone": 0.0},
        "call_families": ["cover3", "man"],
        "packages": ["dime", "base"],
        "family_coefficients": {"cover3": -0.2, "man": 0.0},
        "package_coefficients": {"dime": 0.0, "base": 0.0},
        "family_situation_coefficients": {
            name: {feature: 0.0 for feature in defense_v1.INTERACTION_FEATURES}
            for name in ("cover3", "man")
        },
        "package_situation_coefficients": {
            name: {feature: 0.0 for feature in defense_v1.INTERACTION_FEATURES}
            for name in ("dime", "base")
        },
        "candidates": candidates,
        "temperature": 0.0,
        "top_k": 2,
        "family_spread": 0.0,
    }
    return defense_v1.validate_artifact(payload)


class DefenseTendencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifact = _defense_artifact()
        self.book = {"Dime 3-2 Odd": ["Cover 3"], "4-3 Over": ["Cover 1"]}
        self.sit = Situation(raw="1&10", side="defense", down=1, distance=10, yardline=45)

    def _pick(self, profile: dict | None) -> str:
        return defense_v1.pick_call(
            self.artifact, self.sit, self.book, seed=1, temperature=0.0,
            opponent_profile=profile,
        )["pick"]["play"]

    def test_no_sample_keeps_the_model_call(self) -> None:
        self.assertEqual(self._pick(None), "Cover 3")
        self.assertEqual(self._pick({"n": 0, "shrink": 0.0, "run_share": None, "concepts": {}}), "Cover 3")

    def test_run_heavy_sample_leaves_the_light_box(self) -> None:
        heavy = {
            "n": 12,
            "shrink": 12 / 20,
            "run_share": 0.9,
            "concepts": {"run": 0.9},
            "scope": "this down",
            "summary": "run 90% this down (n=12, shrink=0.60)",
        }
        thin = {
            "n": 1,
            "shrink": 1 / 9,
            "run_share": 0.9,
            "concepts": {"run": 1.0},
            "summary": "run 90% (n=1)",
        }
        self.assertEqual(self._pick(thin), "Cover 3")
        picked = defense_v1.pick_call(
            self.artifact, self.sit, self.book, seed=1, temperature=0.0,
            opponent_profile=heavy,
        )
        self.assertEqual(picked["pick"]["play"], "Cover 1")
        self.assertNotIn("dime", picked["pick"]["formation"].lower())
        self.assertIn("run 90%", picked["tendency_summary"])
        dime = {"formation": "Dime 3-2 Odd", "play": "Cover 3", "family": "cover3", "package": "dime"}
        self.assertGreater(defense_success_delta(dime, heavy), 0)
        self.assertGreater(
            defense_success_delta(dime, heavy),
            defense_success_delta(dime, thin),
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "run.db")
            try:
                for _ in range(8):
                    db.bump_tendency("ryan", "offense_concept|open|normal|1", "Inside Zone", amount=1)
                    db.bump_tendency("ryan", "offense_concept", "Inside Zone", amount=1)
                profile = build_defense_profile(db, "ryan", self.sit)
                unknown = build_defense_profile(db, "nobody", self.sit)
            finally:
                db.close()
        self.assertGreaterEqual(profile["n"], 8)
        self.assertGreater(profile["run_share"], 0.5)
        self.assertIn("run", profile["summary"])
        self.assertEqual(unknown["n"], 0)
        self.assertEqual(
            defense_v1.pick_call(
                self.artifact, self.sit, self.book, seed=1, temperature=0.0,
                opponent_profile=profile,
            )["pick"]["play"],
            "Cover 1",
        )


class _FakeConn:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def execute(self, _sql: str, _params: tuple = ()):
        return self

    def fetchall(self) -> list[dict]:
        return list(self.rows)


class _FakeDB:
    def __init__(self, snaps: list[dict], tendencies: list[dict] | None = None) -> None:
        self.conn = _FakeConn(snaps)
        self.tendencies = tendencies or []

    def get_tendencies(self, _opponent_id: str) -> list[dict]:
        return list(self.tendencies)


def _defense_snap(concept: str, *, executed: str = "unknown") -> dict:
    return {
        "side": "defense", "down": 1, "distance": 10, "coverage_seen": None,
        "concept_seen": concept, "formation": None, "play": None, "result": "4",
        "executed_status": executed, "executed_verification": executed,
    }


class OpponentObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sit = Situation(raw="1&10", side="defense", down=1, distance=10, yardline=45)

    def test_unverified_execution_still_counts_the_opponents_observed_play(self) -> None:
        snaps = [_defense_snap("Inside Zone") for _ in range(6)] + [_defense_snap("Mesh") for _ in range(2)]
        profile = build_defense_profile(_FakeDB(snaps), "opp_a", self.sit)
        self.assertEqual(profile["n"], 8)
        self.assertAlmostEqual(profile["run_share"], 0.75)

    def test_a_logged_snap_and_its_tendency_bump_count_once(self) -> None:
        snaps = [_defense_snap("Inside Zone") for _ in range(6)] + [_defense_snap("Mesh") for _ in range(2)]
        tendencies = [
            {"bucket": "offense_concept|open|normal|1", "key": "Inside Zone", "count": 6},
            {"bucket": "offense_concept|open|normal|1", "key": "Mesh", "count": 2},
            {"bucket": "offense_concept", "key": "Inside Zone", "count": 6},
            {"bucket": "offense_concept", "key": "Mesh", "count": 2},
        ]
        profile = build_defense_profile(_FakeDB(snaps, tendencies), "opp_a", self.sit)
        self.assertEqual(profile["n"], 8)
        self.assertAlmostEqual(profile["run_share"], 0.75)

    def test_tendency_history_beyond_logged_snaps_is_kept(self) -> None:
        snaps = [_defense_snap("Inside Zone") for _ in range(4)]
        tendencies = [
            {"bucket": "offense_concept", "key": "Inside Zone", "count": 4},
            {"bucket": "offense_concept", "key": "Four Verticals", "count": 4},
        ]
        profile = build_defense_profile(_FakeDB(snaps, tendencies), "opp_a", None)
        self.assertEqual(profile["n"], 8)
        self.assertAlmostEqual(profile["run_share"], 0.5)

    def test_offense_profile_counts_observed_coverage_once(self) -> None:
        snaps = [{
            "side": "offense", "down": 3, "distance": 7, "coverage_seen": "Cover 2 Man",
            "concept_seen": None, "formation": "Gun", "play": "Mesh", "result": "4",
            "executed_status": "shown", "executed_verification": "shown_call",
        } for _ in range(5)]
        tendencies = [{"bucket": "their_coverage", "key": "Cover 2 Man", "count": 5}]
        profile = build_offense_profile(_FakeDB(snaps, tendencies), "opp_a", None)
        self.assertEqual(profile["n"], 5)
        self.assertEqual(profile["mix"], {"cover2_man": 1.0})
        self.assertEqual(profile["play_rates"], {})


if __name__ == "__main__":
    unittest.main()
