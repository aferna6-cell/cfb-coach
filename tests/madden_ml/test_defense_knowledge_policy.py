"""Part 1 defense policy: verified knowledge tags, profiles, and home isolation."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.madden.model import defense_coordinator as defense
from cfb_coach.madden.model import defense_knowledge as knowledge

EA_GAMEPLAY = "https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay"
EA_TITLE = (
    "https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-nfl-27-title-update-september-16"
)
HYPOTHESIS_IDS = (
    "db_fire2",
    "mug_over_contain",
    "base_c4",
    "stunt_vs_rpo",
    "unsourced_editor_fields",
    "mesh_answer",
)
HYPOTHESIS_PROVENANCE = {
    "db_fire2": [
        "https://themaddenacademy.com/2026/09/mastering-madden-27-best-defensive-macro-after-the-patch",
    ],
    "mug_over_contain": ["/workspace/madden/brief-2026-10-02.md"],
    "base_c4": ["/workspace/madden/brief-2026-10-02.md"],
    "stunt_vs_rpo": ["https://www.civil.gg/tips/madden-27-complete-offense-guide"],
    "unsourced_editor_fields": [],
    "mesh_answer": [EA_GAMEPLAY],
}
PROVENANCE = {
    "align_controls": [EA_GAMEPLAY],
    "smart_zones": [EA_GAMEPLAY],
    "plaster": [EA_GAMEPLAY],
    "roll_coverage": [EA_GAMEPLAY],
    "man_stack": [EA_GAMEPLAY],
    "man_bunch": [EA_GAMEPLAY],
    "match_checks": [EA_GAMEPLAY],
    "double_team": [EA_GAMEPLAY],
    "global_cov_hub": [EA_GAMEPLAY],
    "gap_integrity": [EA_GAMEPLAY],
    "def_aggression": [EA_GAMEPLAY],
    "custom_adj_limits": [EA_GAMEPLAY],
    "qb_contain_fix": [EA_TITLE],
    "shell_fix": [EA_TITLE],
    "stress_vs_4verts": [EA_GAMEPLAY],
    "leverage_press": [EA_GAMEPLAY],
}
BOOK = {
    "Nickel Over": ["Cover 3 Sky", "Tampa 2", "Cover 1 Robber", "Mid Blitz 0"],
    "Dime Normal": ["Cover 4 Drop", "Cover 2 Sink", "Cover 3 Match", "Cover 4 Quarters"],
    "4-3 Over": ["Cover 3 Sky", "Cover 1 Hole", "Mike Blitz"],
}
CATALOG = {"49ers": BOOK, "Lions": {"Dollar": ["Cover 6", "Palms", "Cover 2 Man"]}}
HYPOTHESES_PATH = (
    Path(__file__).resolve().parents[2]
    / "cfb_coach" / "data" / "madden27" / "defense_knowledge_hypotheses.json"
)


def sit(**kwargs):
    fields = {
        "down": 1, "distance": 10, "yardline": 40,
        "red_zone": False, "goal_line": False, "two_minute": False,
        "concept_hint": None, "concept_source": "none", "extras": {},
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _row(play: str, rows: list[dict]) -> dict:
    return next(row for row in rows if row["play"] == play)


class VerifiedKnowledgeTests(unittest.TestCase):
    def test_live_catalog_is_high_confidence_verified_m27_with_provenance(self):
        tags = knowledge.verified_tags()
        self.assertEqual([tag["id"] for tag in tags], list(PROVENANCE))
        for tag in tags:
            self.assertEqual(tag["label"], "VERIFIED-M27")
            self.assertEqual(tag["confidence"], "high")
            self.assertEqual(tag["sources"], PROVENANCE[tag["id"]])
            self.assertNotIn(tag["id"], HYPOTHESIS_IDS)

    def test_hypotheses_are_flagged_and_absent_from_the_live_module(self):
        document = json.loads(HYPOTHESES_PATH.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"], "madden27.defense_knowledge.v1")
        self.assertFalse(document["live_scoring"])
        self.assertEqual(document["status"], "hypothesis")
        self.assertEqual([item["id"] for item in document["items"]], list(HYPOTHESIS_IDS))
        for item in document["items"]:
            self.assertTrue(item["excluded_from_live_scoring"])
            self.assertTrue(item["exclusion_reason"])
            self.assertEqual(item["sources"], HYPOTHESIS_PROVENANCE[item["id"]])
        source = Path(knowledge.__file__).read_text(encoding="utf-8")
        executable = source.split("from __future__", 1)[1]
        self.assertNotIn("defense_knowledge_hypotheses", executable)
        self.assertNotIn("Path.home", source)
        self.assertNotIn(".cfb-coach", source)
        self.assertNotIn("research_db", source)
        live_ids = {tag["id"] for tag in knowledge.verified_tags()}
        self.assertTrue(live_ids.isdisjoint(HYPOTHESIS_IDS))

    def test_stress_check_scores_only_named_shells_on_a_live_vertical(self):
        live = sit(concept_hint="Four Verticals", concept_source="live")
        for play in ("Cover 3 Match", "Cover 4 Quarters", "Cover 6", "Palms"):
            scored = knowledge.live_verified_adjustment(live, play)
            self.assertEqual(scored["delta"], knowledge.STRESS_DELTA)
            self.assertEqual(scored["scored_ids"], ("stress_vs_4verts", "match_checks"))
            self.assertEqual(scored["sources"], (EA_GAMEPLAY,))
            self.assertEqual(scored["hypothesis_ids_used"], ())
            self.assertIn(EA_GAMEPLAY, scored["why"])
        for play in ("Cover 3 Sky", "Cover 1 Robber", "Cover 2 Sink", "Mid Blitz 0"):
            quiet = knowledge.live_verified_adjustment(live, play)
            self.assertEqual(quiet["delta"], 0.0)
            self.assertEqual(quiet["scored_ids"], ())
        last = knowledge.live_verified_adjustment(
            sit(concept_hint="Four Verticals", concept_source="last"), "Cover 3 Match",
        )
        self.assertEqual(last["delta"], 0.0)

    def test_rank_uses_the_verified_tag_and_leaves_other_plays_untagged(self):
        live = sit(down=3, distance=12, concept_hint="Four Verticals", concept_source="live")
        baseline = sit(down=3, distance=12)
        last = sit(down=3, distance=12, concept_hint="Four Verticals", concept_source="last")
        live_rows = defense.rank_defense(live, BOOK)
        base_rows = defense.rank_defense(baseline, BOOK)
        last_rows = defense.rank_defense(last, BOOK)
        self.assertEqual(
            [(row["formation"], row["play"], row["score"]) for row in last_rows],
            [(row["formation"], row["play"], row["score"]) for row in base_rows],
        )
        match = _row("Cover 3 Match", live_rows)
        robber = _row("Cover 1 Robber", live_rows)
        self.assertEqual(match["verified_knowledge"]["delta"], knowledge.STRESS_DELTA)
        self.assertEqual(match["verified_knowledge"]["sources"], (EA_GAMEPLAY,))
        self.assertEqual(robber["verified_knowledge"]["delta"], 0.0)
        self.assertEqual(robber["verified_knowledge"]["scored_ids"], ())
        for row in live_rows:
            self.assertIn("Hypothesis knowledge is not scored.", row["why"])
            for banned in HYPOTHESIS_IDS:
                self.assertNotIn(banned, row["why"])
                self.assertNotIn(banned, row["verified_knowledge"]["scored_ids"])
        base_match = _row("Cover 3 Match", base_rows)
        self.assertAlmostEqual(
            match["score"] - base_match["score"],
            knowledge.STRESS_DELTA
            + match["football_intelligence"]["delta"]
            - base_match["football_intelligence"]["delta"],
        )

    def test_packaged_settings_do_not_reorder_plays_by_themselves(self):
        plain = defense.rank_defense(sit(), BOOK)
        self.assertTrue(all(row["verified_knowledge"]["delta"] == 0.0 for row in plain))
        self.assertTrue(all(row["verified_knowledge"]["scored_ids"] == () for row in plain))


class PortfolioAndProfileTests(unittest.TestCase):
    def test_situation_packages_stay_inside_the_staged_proposal(self):
        proposal = defense.design_defense(max_formations=15, catalogue=CATALOG)
        self.assertLessEqual(proposal["n_formations"], 15)
        self.assertEqual(proposal["hypothesis_tags_scored"], [])
        self.assertEqual(
            [tag["id"] for tag in proposal["knowledge_tags"]],
            [tag["id"] for tag in knowledge.verified_tags()],
        )
        for tag in proposal["knowledge_tags"]:
            self.assertEqual(tag["sources"], PROVENANCE[tag["id"]])
        installed = set(proposal["formations"])
        packages = proposal["situation_packages"]
        self.assertEqual(
            set(packages),
            {
                "early_downs", "passing_downs", "goal_line", "two_minute",
                "spread", "heavy", "mobile_qb",
            },
        )
        for package in packages.values():
            self.assertFalse(package["playable"])
            self.assertEqual(package["status"], "staged")
            self.assertTrue(set(package["formations"]) <= installed)
        self.assertIn("4-3 Over", packages["heavy"]["formations"])
        self.assertNotIn("Goal Line", packages["goal_line"]["formations"])

    def test_profiles_shrink_weight_recency_and_do_not_score(self):
        def snap(game_id: str, opponent: str, concept: str) -> dict:
            return {
                "opponent_type": "human", "opponent_id": opponent, "game_id": game_id,
                "supervised_eligible": True, "executed_verification": "verified",
                "executed_status": "identified", "executed_play": "Cover 3 Sky",
                "label_available": True, "concept_seen": concept, "side": "defense",
            }

        one = knowledge.fit_opponent_profiles([snap("g1", "gavin", "Four Verticals")])
        short = one["opponents"]["gavin"]
        self.assertLess(short["share"], 0.5)
        self.assertGreater(short["uncertainty"], 0.5)
        self.assertEqual(short["live_score_delta"], 0.0)
        self.assertFalse(short["current_snap_known"])
        self.assertFalse(short["previous_snap_is_current_play"])
        self.assertEqual(short["previous_snap_concept"], "vert")

        shifted = [snap(f"old-{i}", "gavin", "Four Verticals") for i in range(8)]
        shifted += [snap(f"new-{i}", "gavin", "Flood") for i in range(8)]
        shifted += [snap("tiano-1", "tiano", "Mesh")]
        fitted = knowledge.fit_opponent_profiles(shifted)
        gavin = fitted["opponents"]["gavin"]
        tiano = fitted["opponents"]["tiano"]
        self.assertEqual(gavin["family"], "flood")
        self.assertTrue(gavin["change_detected"])
        self.assertEqual(gavin["older_family"], "vert")
        self.assertEqual(gavin["newer_family"], "flood")
        self.assertEqual(gavin["live_score_delta"], 0.0)
        self.assertEqual(tiano["n"], 1)
        self.assertNotEqual(tiano["family"], gavin["family"])
        self.assertEqual(fitted["live_score_delta"], 0.0)
        self.assertFalse(fitted["uses_hypothesis_tags"])

        bare = {"families": {}, "contexts": {}, "plays": {}}
        with_profile = {**bare, "opponent_profiles": fitted}
        without = defense.rank_defense(sit(), BOOK, bare, opponent_id="gavin")
        with_rows = defense.rank_defense(sit(), BOOK, with_profile, opponent_id="gavin")
        self.assertEqual(
            [(row["play"], row["score"]) for row in without],
            [(row["play"], row["score"]) for row in with_rows],
        )
        self.assertEqual(with_rows[0]["opponent_profile"]["live_score_delta"], 0.0)
        self.assertFalse(with_rows[0]["opponent_profile"]["current_snap_known"])


class HomeIsolationTests(unittest.TestCase):
    def test_policy_paths_use_an_empty_temporary_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "empty-home"
            home.mkdir()
            keys = ("HOME", "USERPROFILE", "CFB_COACH_DB", "CFB_COACH_MADDEN_DB")
            saved = {key: os.environ.get(key) for key in keys}
            os.environ["HOME"] = str(home)
            os.environ["USERPROFILE"] = str(home)
            os.environ.pop("CFB_COACH_DB", None)
            os.environ.pop("CFB_COACH_MADDEN_DB", None)
            try:
                with mock.patch("pathlib.Path.home", side_effect=AssertionError("touched HOME")):
                    tags = knowledge.verified_tags()
                    scored = knowledge.live_verified_adjustment(
                        sit(concept_hint="Four Verticals", concept_source="live"),
                        "Cover 3 Match",
                    )
                    ranked = defense.rank_defense(
                        sit(down=3, distance=8, concept_hint="Four Verticals", concept_source="live"),
                        BOOK,
                    )
                    proposal = defense.design_defense(max_formations=4, catalogue=CATALOG)
                    profiles = knowledge.fit_opponent_profiles([])
                self.assertEqual(len(tags), len(PROVENANCE))
                self.assertEqual(scored["sources"], (EA_GAMEPLAY,))
                self.assertEqual(len(ranked), sum(len(plays) for plays in BOOK.values()))
                self.assertFalse(proposal["situation_packages"]["early_downs"]["playable"])
                self.assertEqual(profiles["opponents"], {})
                self.assertFalse((home / ".cfb-coach").exists())
                self.assertEqual(list(home.iterdir()), [])
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
