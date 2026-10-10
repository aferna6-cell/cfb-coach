"""Sprint 17: verified four-human-game learning and defensive action intelligence."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.madden.model import (
    defense_action_policy as actions,
    defense_coordinator as defense,
    defense_intelligence as intelligence,
    user_game_learning as history,
)
from cfb_coach.madden.model import offense_macro_lab


def sit(**kwargs):
    fields = {
        "down": 3, "distance": 8, "yardline": 37, "red_zone": False,
        "goal_line": False, "two_minute": False,
        "concept_hint": None, "concept_source": "none", "extras": {},
        "score_us": None, "score_them": None, "quarter": 2,
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def snap(game, opponent, side, play, *, concept=None, success="true"):
    row = {
        "opponent_type": "human", "opponent_id": opponent, "game_id": game,
        "side": side, "down": 3, "distance": 8,
        "executed_formation": "Nickel Over" if side == "defense" else "Gun Bunch",
        "executed_play": play, "action_play": play, "success": success,
        "executed_status": "identified", "executed_verification": "verified",
        "label_available": True, "supervised_eligible": True,
        "eligibility": "verified_execution", "concept_seen": concept,
    }
    return row


class FourGameLearningTests(unittest.TestCase):
    def test_latest_four_human_games_never_mix_cpu_or_recommendations(self):
        rows = [snap("old", "alpha", "defense", "Cover 3 Sky", concept="Mesh")]
        for i in range(1, 5):
            rows.append(snap(f"g{i}", f"user{i}", "offense", "Mesh"))
            rows.append(snap(f"g{i}", f"user{i}", "defense", "Cover 3 Sky",
                             concept="Four Verticals"))
        rows.append(dict(snap("cpu-match", "cpu", "defense", "Cover 3 Sky"),
                         opponent_type="cpu"))
        rows.append(dict(snap("g4", "user4", "defense", "Cover 1 Robber"),
                         executed_verification="unknown", supervised_eligible=False))
        report = history.select_recent_human_games(rows, max_games=4)
        self.assertEqual(report["found_human_games"], 4)
        self.assertEqual(report["selected_game_ids"], ["g1", "g2", "g3", "g4"])
        self.assertEqual(report["verified_defense"], 4)
        self.assertEqual(report["verified_offense"], 4)
        self.assertEqual(report["verified_labeled_human_executions"], 8)
        self.assertEqual(report["games"]["g4"]["unverified_or_unlabeled"], 1)

    def test_no_verified_game_does_not_train_fabricated_artifacts(self):
        rows = [dict(snap("g1", "x", "offense", "Mesh"),
                     supervised_eligible=False, executed_verification="unknown")]
        with tempfile.TemporaryDirectory() as tmp:
            result = history.train_shadow_models(rows, out_dir=tmp)
            self.assertFalse(result["trained"])
            self.assertFalse(result["installed"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_offense_defense_shadow_artifacts_from_four_user_games(self):
        rows = []
        for i in range(4):
            for n in range(4):
                rows.extend([
                    snap(f"g{i}", f"player{i}", "defense", "Cover 3 Sky",
                         concept="Four Verticals", success="false" if n == 1 else "true"),
                    snap(f"g{i}", f"player{i}", "offense", "Mesh", success="true"),
                ])
        with tempfile.TemporaryDirectory() as tmp:
            result = history.train_shadow_models(rows, out_dir=tmp)
            self.assertEqual(result["verified_defense"], 16)
            self.assertEqual(result["verified_offense"], 16)
            self.assertEqual(result["opponent_concepts_observed"], 16)
            self.assertTrue(result["trained"])
            self.assertFalse(result["installed"])
            self.assertFalse(result["live_mode_changed"])
            for name, path in result["artifacts"].items():
                self.assertIn(name, ("offense", "defense"))
                self.assertTrue(Path(path).is_file())

    def test_outcome_only_four_user_games_learn_risks_not_play_success(self):
        rows = []
        for i in range(4):
            for _ in range(12):
                row = dict(snap(f"user-game-{i}", f"player{i}",
                                "defense", "Cover 3 Sky", concept=None))
                row.update({
                    "supervised_eligible": False,
                    "executed_status": "unknown",
                    "executed_verification": "unknown",
                    "executed_play": None,
                    "action_play": None,
                    "yards": 25,
                    "success": "false",
                })
                rows.append(row)
        with tempfile.TemporaryDirectory() as tmp:
            out = history.train_shadow_models(rows, out_dir=tmp)
            self.assertTrue(out["trained"])
            self.assertEqual(out["verified_defense"], 0)
            self.assertEqual(out["verified_offense"], 0)
            self.assertEqual(out["opponent_outcome_rows_learned"], 48)
            self.assertEqual(
                out["defense_training_basis"],
                "observed_opponent_risks_only_no_defensive_play_attribution",
            )
            m = defense.load_model(out["artifacts"]["defense"])
            self.assertEqual(m["supervised_defensive_snaps"], 0)
            self.assertEqual(m["opponent_outcome_risks"]["observed_outcomes"], 48)
            risks = m["opponent_outcome_risks"]
            self.assertGreater(
                intelligence.defensive_risk_delta(risks, "player0", "long", "two_high")["delta"], 0
            )
            self.assertLess(
                intelligence.defensive_risk_delta(risks, "player0", "long", "pressure")["delta"], 0
            )
            self.assertEqual(
                intelligence.defensive_risk_delta(risks, "different-user", "long", "pressure")["delta"], 0
            )

    def test_defense_is_not_trained_on_cpu_or_trusted_unverified_video(self):
        good = snap("g1", "tiano", "defense", "Cover 3 Sky", concept="Mesh")
        unverified = dict(good, executed_verification="unknown")
        cpu = dict(good, opponent_type="cpu")
        trusted_video = dict(good, executed_verification="unknown", trusted_vod=True)
        model = defense.train_defense([good, unverified, cpu, trusted_video])
        self.assertEqual(model["supervised_defensive_snaps"], 1)
        self.assertEqual(model["verified_human_games"], 1)
        self.assertEqual(model["opponent_tendencies"]["observed_post_snap_concepts"], 1)

    def test_tendency_is_not_current_snap_and_does_not_leak_between_opponents(self):
        rows = [
            snap("g1", "gavin", "defense", "Cover 3 Sky", concept="Four Verticals")
            for _ in range(12)
        ]
        trained = intelligence.fit_opponent_offense(rows)
        gavin = intelligence.tendency(trained, "gavin", "long")
        tiano = intelligence.tendency(trained, "tiano", "long")
        self.assertTrue(gavin["ready"])
        self.assertEqual(gavin["family"], "vert")
        self.assertTrue(gavin["historical_only"])
        self.assertFalse(tiano["ready"])
        inferred = intelligence.score_defensive_knowledge(
            family="two_high", sit=sit(), opponent_id="gavin", tendencies=trained
        )
        self.assertIsNone(inferred["live_concept"])
        self.assertTrue(inferred["evidence"]["historical_tendency"])
        live = intelligence.score_defensive_knowledge(
            family="two_high",
            sit=sit(concept_hint="Four Verticals", concept_source="live"),
            opponent_id="gavin", tendencies=trained,
        )
        self.assertEqual(live["live_concept"], "vert")
        self.assertTrue(live["evidence"]["live"])

    def test_last_snap_does_not_trigger_defensive_adjustments(self):
        self.assertEqual(actions.researched_adjustments(
            sit(concept_hint="QB scramble", concept_source="last"), "gavin"
        ), [])

    def test_live_qb_scramble_can_source_a_verified_contain_adjustment(self):
        rows = actions.researched_adjustments(
            sit(concept_hint="QB scramble", concept_source="live"), "gavin"
        )
        self.assertTrue(any(r["id"] == "qb_contain" for r in rows))
        for row in rows:
            self.assertTrue(row["research_sources"])
            self.assertTrue(row["buttons"])
            self.assertLessEqual(row["influence"], actions.MAX_ADJ_BONUS)

    def test_joint_defense_scores_all_legal_plays(self):
        book = {
            "Nickel Over": ["Cover 3 Sky", "Tampa 2", "Cover 1 Robber"],
            "Dime": ["Cover 4 Drop", "Cover 2 Man", "Unknown Defensive Name"],
        }
        ranked = defense.rank_defense(sit(), book, opponent_id="gavin")
        self.assertEqual(len(ranked), 6)
        self.assertTrue(all("football_intelligence" in r for r in ranked))
        self.assertEqual({r["play"] for r in ranked},
                         {play for plays in book.values() for play in plays})

    def test_offense_macro_drafts_fall_back_if_all_safe_compositions_conflict(self):
        fake_book = {
            "offense": {
                "locked_ts": "installed", "formations": {"Gun Bunch": ["Mesh"]},
                "formation_sources": {"Gun Bunch": "Buccaneers"},
            }
        }
        draft = {
            "name": "ML-MAN-HR", "status": "DRAFT_NEEDS_IN_GAME_VERIFICATION",
            "source_ids": ["prodigy-beat-man"], "source_action_id": "hot_slant_vs_man",
            "base_pairs": [{"formation": "Gun Bunch", "play": "Mesh"}],
            "settings": [{"section": "Route assignments", "setting": "WR1", "value": "Slant"}],
        }
        with mock.patch.object(offense_macro_lab.playbook, "load_books", return_value=fake_book):
            with mock.patch.object(offense_macro_lab, "compose_drafts", return_value=[]):
                with mock.patch.object(
                    offense_macro_lab, "propose_variants",
                    return_value={"proposals": [draft]},
                ):
                    report = offense_macro_lab.composed_macro_report(object())
        self.assertEqual(report["two_action_compositions"], 0)
        self.assertEqual(report["single_action_variants"], 1)
        self.assertEqual(report["proposals"][0]["name"], "ML-MAN-HR")
        self.assertEqual(report["status"], "DRAFT_ONLY")

    def test_original_offense_macro_combinations_conflicts_and_drafts(self):
        fields = [
            {"id": "hr", "type": "hot_route", "target": "WR1",
             "route": "Slant", "vs": ["man"], "sources": ["civil-offense"]},
            {"id": "pro", "type": "pass_protection", "target": "OL",
             "route": "Slide Left", "vs": ["man"], "sources": ["civil-offense"]},
        ]
        draft = offense_macro_lab.compose_primitives(
            fields, formations={"Gun Bunch": ["Mesh", "Inside Zone"]},
            sources={"Gun Bunch": "Buccaneers"},
        )
        self.assertEqual(len(draft), 1)
        self.assertEqual(draft[0]["state"], "draft")
        self.assertEqual(len(draft[0]["source_action_ids"]), 2)
        self.assertTrue(all(p["play"] == "Mesh" for p in draft[0]["base_pairs"]))


if __name__ == "__main__":
    unittest.main()
