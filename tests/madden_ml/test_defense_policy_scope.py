"""Scope tests for defensive intelligence on the b987e3f baseline.

Each test id is listed in ``NEW_TEST_SCOPES`` with the scope item it covers:
1 opponent context, 2 joint ranking, 3 personnel and set, 4 staged install,
5 context coverage, 6 hybrid gate, plus the knowledge layer.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.defense_select import select_defense
from cfb_coach.madden.model import defense_coordinator as defense
from cfb_coach.madden.model.opponent_learning import build_opponent_context
from cfb_coach.madden.model.schema import (
    CoachingMode,
    EditorFieldAssignment,
    EditorFieldSource,
    GateResult,
    MLStatus,
    MacroJointFeatures,
    MacroSide,
    MacroState,
    ModelRegistryEntry,
    OpponentContextKey,
    PolicySource,
    Possession,
    PropensityMethod,
    Tri,
    defensive_editor_field_names,
)
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.situation import Situation

ROOT = Path(__file__).resolve().parents[2]
BOTH_NULL_SNAPSHOT = Path(__file__).with_name("fixtures") / "both_null_defense_rank_2b372e2.json"
_RANK_KEYS_ADDED_AFTER_BASE = ("personnel", "verified_knowledge", "opponent_profile", "why")
KNOWLEDGE = "tests/madden_ml/test_defense_knowledge_policy.py"
SCOPE = "tests/madden_ml/test_defense_policy_scope.py"

NEW_TEST_SCOPES = {
    f"{KNOWLEDGE}::VerifiedKnowledgeTests::test_live_catalog_is_high_confidence_verified_m27_with_provenance": "knowledge layer",
    f"{KNOWLEDGE}::VerifiedKnowledgeTests::test_hypotheses_are_flagged_and_absent_from_the_live_module": "knowledge layer",
    f"{KNOWLEDGE}::VerifiedKnowledgeTests::test_stress_check_scores_only_named_shells_on_a_live_vertical": "knowledge layer",
    f"{KNOWLEDGE}::VerifiedKnowledgeTests::test_rank_uses_the_verified_tag_and_leaves_other_plays_untagged": "knowledge layer",
    f"{KNOWLEDGE}::VerifiedKnowledgeTests::test_packaged_settings_do_not_reorder_plays_by_themselves": "knowledge layer",
    f"{KNOWLEDGE}::PortfolioAndProfileTests::test_situation_packages_stay_inside_the_staged_proposal": "knowledge layer",
    f"{KNOWLEDGE}::PortfolioAndProfileTests::test_profiles_shrink_weight_recency_and_do_not_score": "knowledge layer",
    f"{KNOWLEDGE}::HomeIsolationTests::test_policy_paths_use_an_empty_temporary_home": "knowledge layer",
    f"{SCOPE}::OpponentContextTests::test_seed_weight_zero_clears_both_seed_fields": "1",
    f"{SCOPE}::OpponentContextTests::test_builder_drops_cpu_coverage_and_recommendation_only_rows": "1",
    f"{SCOPE}::JointRankingTests::test_shadow_logs_the_product_and_keeps_the_shown_call": "2",
    f"{SCOPE}::JointRankingTests::test_armed_macros_compete_with_no_adjustment": "2",
    f"{SCOPE}::JointRankingTests::test_exact_requires_every_stage_exact": "2",
    f"{SCOPE}::JointRankingTests::test_heuristic_call_matches_the_fixed_seed": "2",
    f"{SCOPE}::JointRankingTests::test_hybrid_above_forty_shows_the_joint_product": "2",
    f"{SCOPE}::PersonnelEncodingTests::test_both_null_fields_contribute_zero": "3",
    f"{SCOPE}::PersonnelEncodingTests::test_empty_string_is_not_null_and_is_not_verified": "3",
    f"{SCOPE}::PersonnelEncodingTests::test_none_blank_and_empty_token_are_distinct": "3",
    f"{SCOPE}::BothNullIdentityTests::test_both_null_rank_matches_the_2b372e2_snapshot": "3",
    f"{SCOPE}::DesignInstallTests::test_confirm_leaves_the_database_hash_until_aidan_applies": "4",
    f"{SCOPE}::ContextCoverageTests::test_proposal_names_each_portfolio_situation": "5",
    f"{SCOPE}::ContextCoverageTests::test_sixteen_formations_is_refused": "5",
    f"{SCOPE}::HybridGateTests::test_below_forty_snaps_returns_no_metrics": "6",
    f"{SCOPE}::HybridGateTests::test_forty_snaps_runs_the_standard_gate": "6",
    f"{SCOPE}::HybridGateTests::test_live_hybrid_does_not_score_below_forty": "6",
    f"{SCOPE}::HybridGateTests::test_user_entered_look_verification_does_not_count": "6",
    f"{SCOPE}::DatabasePathTests::test_db_paths_use_an_explicit_argument_or_default_db_path": "knowledge layer",
    f"{SCOPE}::DatabasePathTests::test_readonly_path_is_the_madden_database_and_opens_read_only": "knowledge layer",
}

CATALOG = {
    "49ers": {
        "Nickel Over": ["Cover 3 Sky", "Tampa 2", "Cover 1 Robber", "Mid Blitz 0"],
        "Dime Normal": ["Cover 4 Drop", "Cover 2 Sink", "Cover 3 Match"],
        "4-3 Over": ["Cover 3 Sky", "Cover 1 Hole", "Mike Blitz"],
    },
}
BOOK = CATALOG["49ers"]
LONG_ATTESTATION = "All defensive formations and every source play have been installed and tested in Madden."


def sit(**kwargs):
    fields = {
        "down": 1, "distance": 10, "yardline": 40,
        "red_zone": False, "goal_line": False, "two_minute": False,
        "concept_hint": None, "concept_source": "none", "extras": {},
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def snap(index: int, **overrides) -> dict:
    row = {
        "opponent_type": "human",
        "opponent_id": "gavin",
        "game_id": "g1",
        "side": "defense",
        "supervised_eligible": True,
        "executed_verification": "verified",
        "executed_status": "identified",
        "executed_play": "Cover 3 Sky",
        "label_available": True,
        "concept_seen": "Four Verticals",
        "down": 1,
        "distance": 10,
        "yards": 4,
        "snap_id": f"snap-{index}",
    }
    row.update(overrides)
    return row


def _sheet() -> dict:
    return {
        name: EditorFieldAssignment(value="Default", source=EditorFieldSource.DEFAULT)
        for name in defensive_editor_field_names()
    }


def _candidate(macro_id: str, formation: str, play: str, state: MacroState, *, no_adj: bool = False):
    from cfb_coach.madden.model.schema import MacroCandidate
    from cfb_coach.madden.adjustments.synthesis import design_record_id_for

    return MacroCandidate(
        macro_id=macro_id,
        design_record_id=design_record_id_for(macro_id),
        version="1",
        side=MacroSide.DEF,
        state=state,
        formation=formation,
        play=play,
        adjustment_primitives=() if no_adj else (),
        editor_fields=None if no_adj else _sheet(),
        tactical_purpose="Fixture defensive adjustment.",
        situations=(),
        conflicts=(),
        joint=MacroJointFeatures(compatibility=1.0, action_count=0 if no_adj else 1, uncertainty=0.2),
        is_no_adjustment=no_adj,
    )


def _record(candidate):
    from cfb_coach.madden.adjustments.synthesis import DesignRecord

    return DesignRecord(
        design_record_id=candidate.design_record_id,
        candidate=candidate,
        downside=None,
        uncertainty_note=None,
        catalog_comparison=(),
        install_instructions=("Install the sheet in the Madden editor.",),
    )


def _advance(record, *states: MacroState):
    from cfb_coach.madden.adjustments.synthesis import transition

    current = record
    for state in states:
        current = transition(current, state, "FIXTURE: user confirmed this defensive sheet")
    return current


class OpponentContextTests(unittest.TestCase):
    def test_seed_weight_zero_clears_both_seed_fields(self) -> None:
        weight = inspect.signature(build_opponent_context).parameters["seed_shrink_weight"]
        self.assertIs(weight.default, inspect.Parameter.empty)
        seed = OpponentContextKey(
            opponent_id="gavin", patch_id="1.007", roster_snapshot_id="roster-a", table_version="v1",
        )
        with mock.patch("pathlib.Path.home", side_effect=AssertionError("home")):
            cleared = build_opponent_context(
                [],
                opponent_id="gavin",
                seed_shrink_weight=0,
                seeded_from_key=seed,
                patch_id="1.008",
                roster_snapshot_id="roster-b",
                table_version="v1",
            )
        self.assertIsNone(cleared.seeded_from_key)
        self.assertIsNone(cleared.seed_shrink_weight)
        self.assertEqual(cleared.n_snaps, 0)
        self.assertIs(cleared.tendencies_known, Tri.FALSE)
        self.assertEqual(cleared.tendencies, ())
        self.assertIsNone(cleared.shrinkage_weight)

        kept = build_opponent_context(
            [],
            opponent_id="gavin",
            seed_shrink_weight=0.25,
            seeded_from_key=seed,
            patch_id="1.008",
            roster_snapshot_id="roster-b",
            table_version="v1",
        )
        self.assertEqual(kept.seeded_from_key, seed)
        self.assertEqual(kept.seed_shrink_weight, 0.25)
        with self.assertRaises(ValueError):
            build_opponent_context([], opponent_id="gavin", seed_shrink_weight=1.2)
        with self.assertRaises(ValueError):
            build_opponent_context([], opponent_id="gavin", seed_shrink_weight=0.4)
        same = OpponentContextKey(
            opponent_id="gavin", patch_id="1.008", roster_snapshot_id="roster-b", table_version="v1",
        )
        with self.assertRaises(ValueError):
            build_opponent_context(
                [],
                opponent_id="gavin",
                seed_shrink_weight=0.4,
                seeded_from_key=same,
                patch_id="1.008",
                roster_snapshot_id="roster-b",
                table_version="v1",
            )

    def test_builder_drops_cpu_coverage_and_recommendation_only_rows(self) -> None:
        rows = [snap(i) for i in range(6)]
        rows.append(snap(6, concept_seen=None, coverage_seen="Cover 3 Match"))
        rows.append(snap(7, opponent_type="cpu", concept_seen="Flood"))
        rows.append(snap(8, opponent_id="other", concept_seen="Inside Zone"))
        rows.append(snap(
            9, recommendation_only=True, concept_seen="Mesh", executed_play="Cover 2",
        ))
        rows.append({
            "opponent_type": "human",
            "opponent_id": "gavin",
            "game_id": "g1",
            "side": "defense",
            "recommended_play": "Mesh",
            "concept_seen": "Mesh",
            "label_available": True,
        })
        with mock.patch("pathlib.Path.home", side_effect=AssertionError("home")):
            context = build_opponent_context(
                rows, opponent_id="gavin", seed_shrink_weight=0,
            )
        self.assertEqual(context.n_snaps, 7)
        self.assertAlmostEqual(context.shrinkage_weight, 7 / 15)
        self.assertIs(context.tendencies_known, Tri.TRUE)
        self.assertEqual([item.key for item in context.tendencies], ["vert"])
        self.assertEqual(context.tendencies[0].count, 6)
        thin = build_opponent_context(
            [snap(i) for i in range(5)], opponent_id="gavin", seed_shrink_weight=0,
        )
        self.assertEqual(thin.n_snaps, 5)
        self.assertEqual(thin.tendencies, ())
        self.assertIs(thin.tendencies_known, Tri.FALSE)
        self.assertAlmostEqual(thin.shrinkage_weight, 5 / 13)


class JointRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed={"opponents": {}})

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _call(self) -> MaddenCall:
        return MaddenCall(
            "defense", "Nickel Over", "Tampa 2", "No adj", "User hook", "heuristic",
            macro="ALREADY",
        )

    def test_shadow_logs_the_product_and_keeps_the_shown_call(self) -> None:
        defense.set_mode(self.db, "shadow")
        call = self._call()
        shown = (call.formation, call.play, call.macro, call.adjustment, call.adj_or_macro, call.rationale)
        situation = sit(extras={"session_id": "seed-game", "snap_seq": 7})
        with mock.patch(
            "cfb_coach.madden.model.defense_action_policy.select_defensive_action",
            side_effect=AssertionError("shadow must not call the researched action policy"),
        ):
            result = defense.maybe_apply_defense(call, situation, "gavin", self.db, BOOK)
        self.assertIs(result, call)
        self.assertEqual(
            (call.formation, call.play, call.macro, call.adjustment, call.adj_or_macro, call.rationale),
            shown,
        )
        ranking = call.joint_ranking
        self.assertEqual([stage["name"] for stage in ranking["stages"]], ["play", "action"])
        play_stage, action_stage = ranking["stages"]
        self.assertIs(play_stage["method"], PropensityMethod.EXACT)
        self.assertAlmostEqual(play_stage["propensity"], 0.5)
        self.assertIs(action_stage["method"], PropensityMethod.DETERMINISTIC)
        self.assertEqual(action_stage["propensity"], 1.0)
        self.assertIs(ranking["method"], PropensityMethod.DETERMINISTIC)
        self.assertAlmostEqual(ranking["behavior_propensity"], 0.5)
        self.assertAlmostEqual(ranking["log_product"], math.log(ranking["behavior_propensity"]))
        self.assertEqual(ranking["actions"], [{"id": "NO_ADJUSTMENT", "kind": "none"}])
        decision = getattr(call, defense.DECISION_ATTR)
        self.assertIs(decision.mode, CoachingMode.SHADOW)
        self.assertIs(decision.policy_source, PolicySource.HEURISTIC)
        self.assertIs(decision.shadow_status, MLStatus.OK)
        self.assertFalse(decision.fell_back)
        self.assertIsNone(decision.behavior_propensity)
        self.assertEqual(decision.final_pick.play, "Tampa 2")
        self.assertEqual(decision.shadow_pick.play, ranking["winner"]["play"])

    def test_armed_macros_compete_with_no_adjustment(self) -> None:
        armed = _advance(
            _record(_candidate("ML-D-ARMED", "Nickel Over", "Cover 3 Sky", MacroState.DRAFT)),
            MacroState.EDITOR_VERIFIED, MacroState.INSTALLED, MacroState.ARMED,
        )
        editor = _advance(
            _record(_candidate("ML-D-EDITOR", "Nickel Over", "Cover 3 Sky", MacroState.DRAFT)),
            MacroState.EDITOR_VERIFIED,
        )
        installed = _advance(
            _record(_candidate("ML-D-INSTALLED", "Nickel Over", "Cover 3 Sky", MacroState.DRAFT)),
            MacroState.EDITOR_VERIFIED, MacroState.INSTALLED,
        )
        draft = _record(_candidate("ML-D-DRAFT", "Nickel Over", "Cover 3 Sky", MacroState.DRAFT))
        ranking = defense.joint_rank(
            sit(), BOOK, [armed, editor, installed, draft],
            session_id="seed-game", snap_seq=7, opponent_id="gavin",
        )
        ids = [row["id"] for row in ranking["actions"]]
        self.assertIn("NO_ADJUSTMENT", ids)
        self.assertIn("ML-D-ARMED", ids)
        self.assertNotIn("ML-D-EDITOR", ids)
        self.assertNotIn("ML-D-INSTALLED", ids)
        self.assertNotIn("ML-D-DRAFT", ids)
        action = ranking["stages"][1]
        self.assertIs(action["method"], PropensityMethod.EXACT)
        self.assertAlmostEqual(action["propensity"], 0.5)

    def test_exact_requires_every_stage_exact(self) -> None:
        armed = _advance(
            _record(_candidate("ML-D-ARMED", "Nickel Over", "Cover 3 Sky", MacroState.DRAFT)),
            MacroState.EDITOR_VERIFIED, MacroState.INSTALLED, MacroState.ARMED,
        )
        mixed = defense.joint_rank(
            sit(), {"Nickel Over": ["Cover 3 Sky"]}, [armed],
            session_id="solo", snap_seq=1,
        )
        self.assertIs(mixed["stages"][0]["method"], PropensityMethod.DETERMINISTIC)
        self.assertEqual(mixed["stages"][0]["propensity"], 1.0)
        self.assertIs(mixed["stages"][1]["method"], PropensityMethod.EXACT)
        self.assertIs(mixed["method"], PropensityMethod.DETERMINISTIC)
        self.assertAlmostEqual(mixed["behavior_propensity"], 0.5)
        self.assertAlmostEqual(mixed["log_product"], math.log(0.5))

        both = defense.joint_rank(
            sit(), BOOK, [armed], session_id="seed-game", snap_seq=7,
        )
        self.assertIs(both["stages"][0]["method"], PropensityMethod.EXACT)
        self.assertIs(both["stages"][1]["method"], PropensityMethod.EXACT)
        self.assertIs(both["method"], PropensityMethod.EXACT)
        self.assertAlmostEqual(both["log_product"], math.log(both["behavior_propensity"]))

    def test_heuristic_call_matches_the_fixed_seed(self) -> None:
        situation = Situation(
            raw="1st and 10", side="defense", down=1, distance=10, yardline=35,
        )
        book = {
            "Nickel Over": ["Cover 3 Sky", "Tampa 2", "Cover 1 Robber"],
            "4-3 Over": ["Cover 3 Sky", "Cover 1 Hole"],
        }
        pick = select_defense(situation, "gavin", None, book, random.Random(20261011))
        self.assertEqual(pick.formation, "Nickel Over")
        self.assertEqual(pick.play, "Cover 3 Sky")
        self.assertEqual(pick.family, "single_high")
        self.assertEqual(pick.user_job, "User HB")
        self.assertEqual(
            pick.rationale,
            "D mix (early): single-high 55% · Cover 2 / Tampa 27% · man 18% → single-high (C3/match)",
        )
        self.assertEqual(pick.call_probs, {
            ("Nickel Over", "Tampa 2"): 0.2727272727272727,
            ("Nickel Over", "Cover 3 Sky"): 0.2727272727272727,
            ("4-3 Over", "Cover 3 Sky"): 0.2727272727272727,
            ("Nickel Over", "Cover 1 Robber"): 0.09090909090909091,
            ("4-3 Over", "Cover 1 Hole"): 0.09090909090909091,
        })

    def test_hybrid_above_forty_shows_the_joint_product(self) -> None:
        defense.set_mode(self.db, "hybrid")
        situation = sit(extras={
            "session_id": "seed-game",
            "snap_seq": 7,
            "verified_defensive_snaps": [snap(i) for i in range(40)],
        })
        with mock.patch(
            "cfb_coach.madden.model.defense_action_policy.select_defensive_action",
            side_effect=AssertionError("hybrid must not call the researched action policy"),
        ):
            shown = defense.maybe_apply_defense(self._call(), situation, "gavin", self.db, BOOK)
        ranking = shown.joint_ranking
        decision = getattr(shown, defense.DECISION_ATTR)
        self.assertIs(decision.mode, CoachingMode.HYBRID)
        self.assertIs(decision.policy_source, PolicySource.MODEL)
        self.assertFalse(decision.fell_back)
        self.assertIs(decision.shadow_status, MLStatus.OK)
        self.assertAlmostEqual(decision.behavior_propensity, ranking["behavior_propensity"])
        self.assertIs(decision.propensity_method, ranking["method"])
        self.assertEqual(decision.final_pick.formation, ranking["winner"]["formation"])
        self.assertEqual(decision.final_pick.play, ranking["winner"]["play"])


class PersonnelEncodingTests(unittest.TestCase):
    def test_both_null_fields_contribute_zero(self) -> None:
        plain = defense.rank_defense(sit(down=2, distance=7), BOOK)
        explicit = defense.rank_defense(
            sit(down=2, distance=7, opp_personnel=None, opp_set=None), BOOK,
        )
        self.assertEqual(
            [(row["formation"], row["play"], row["score"]) for row in plain],
            [(row["formation"], row["play"], row["score"]) for row in explicit],
        )
        encoding = explicit[0]["personnel"]
        self.assertEqual(encoding["delta"], 0.0)
        self.assertFalse(encoding["verified"])
        self.assertEqual(encoding["personnel"]["missing"], 1)
        self.assertEqual(encoding["set"]["missing"], 1)
        self.assertEqual(encoding["personnel"]["one_hot"], {})
        self.assertEqual(encoding["set"]["one_hot"], {})

    def test_empty_string_is_not_null_and_is_not_verified(self) -> None:
        ranked = defense.rank_defense(
            sit(down=2, distance=7, opp_personnel="", opp_set="bunch"), BOOK,
        )
        encoding = ranked[0]["personnel"]
        self.assertFalse(encoding["verified"])
        self.assertEqual(encoding["personnel"], {
            "missing": 1, "empty": 0, "unknown": 0, "one_hot": {},
        })
        self.assertEqual(encoding["set"]["missing"], 0)
        self.assertEqual(encoding["set"]["empty"], 0)
        self.assertEqual(encoding["set"]["unknown"], 0)
        self.assertEqual(encoding["set"]["one_hot"], {"bunch": 1})
        self.assertNotEqual(encoding["delta"], 0.0)
        unknown = defense.rank_defense(
            sit(down=2, distance=7, opp_personnel="ZZZ", opp_set=None), BOOK,
        )[0]["personnel"]
        self.assertEqual(unknown["personnel"]["unknown"], 1)
        self.assertEqual(unknown["personnel"]["missing"], 0)
        self.assertEqual(unknown["personnel"]["empty"], 0)
        self.assertFalse(unknown["verified"])
        self.assertFalse(ranked[0]["verified_knowledge"].get("scored_ids"))

    def test_none_blank_and_empty_token_are_distinct(self) -> None:
        missing = {"missing": 1, "empty": 0, "unknown": 0, "one_hot": {}}
        for value in (None, "", "   ", "\t"):
            encoding = defense.personnel_features(sit(opp_personnel=value, opp_set=value))
            self.assertEqual(encoding["personnel"], missing)
            self.assertEqual(encoding["set"], missing)
            self.assertEqual(encoding["delta"], 0.0)
            self.assertFalse(encoding["verified"])
        token = defense.personnel_features(sit(opp_personnel=None, opp_set="empty"))
        self.assertEqual(token["personnel"], missing)
        self.assertEqual(token["set"]["missing"], 0)
        self.assertEqual(token["set"]["empty"], 0)
        self.assertEqual(token["set"]["unknown"], 0)
        self.assertEqual(token["set"]["one_hot"], {"empty": 1})
        self.assertFalse(token["verified"])
        self.assertNotEqual(token["delta"], 0.0)


class DesignInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed={"opponents": {}})
        playbook._save_state(self.db, {"applied": {
            "defense": {
                "side": "defense", "mode": "stock", "name": "49ers",
                "source_book": "49ers", "rev": 1, "locked_ts": "test",
                "formations": BOOK,
            },
        }, "pending": {}})
        self.db.set_meta(defense.MODEL_META, str(Path(self.tmp.name) / "missing-model.json"))

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_confirm_leaves_the_database_hash_until_aidan_applies(self) -> None:
        with mock.patch.object(defense.catalog, "book_names", return_value=list(CATALOG)):
            with mock.patch.object(
                defense.catalog, "book_formations", side_effect=lambda side, name: CATALOG[name],
            ):
                plan = defense.stage_design(self.db, max_formations=2)
                before = hashlib.sha256(Path(self.db.path).read_bytes()).hexdigest()
                preview = defense.confirm_design(self.db, plan["proposal_id"], "yes")
                self.assertFalse(preview["wrote_book"])
                self.assertTrue(preview["diff"]["to"]["formations"])
                self.assertTrue(preview["install_instructions"])
                self.assertEqual(
                    hashlib.sha256(Path(self.db.path).read_bytes()).hexdigest(), before,
                )
                self.assertEqual(playbook.load_books(self.db)["defense"]["name"], "49ers")
                with self.assertRaises(ValueError):
                    defense.apply_design(
                        self.db, plan["proposal_id"], "yes", operator="Aidan",
                    )
                with self.assertRaises(ValueError):
                    defense.apply_design(
                        self.db, plan["proposal_id"], LONG_ATTESTATION, operator="Sam",
                    )
                self.assertEqual(
                    hashlib.sha256(Path(self.db.path).read_bytes()).hexdigest(), before,
                )
                applied = defense.apply_design(
                    self.db, plan["proposal_id"], LONG_ATTESTATION, operator="Aidan",
                )
        self.assertTrue(applied["wrote_book"])
        self.assertTrue(applied["confirmed"])
        self.assertEqual(applied["generated_macros_armed"], 0)
        self.assertNotEqual(
            hashlib.sha256(Path(self.db.path).read_bytes()).hexdigest(), before,
        )
        self.assertEqual(
            playbook.load_books(self.db)["defense"]["name"], "ML Designed Defense (custom)",
        )


class ContextCoverageTests(unittest.TestCase):
    def test_proposal_names_each_portfolio_situation(self) -> None:
        proposal = defense.design_defense(max_formations=15, catalogue=CATALOG)
        coverage = proposal["context_coverage"]
        self.assertFalse(coverage["playable"])
        self.assertEqual(coverage["status"], "proposal")
        self.assertEqual(coverage["max_formations"], 15)
        self.assertLessEqual(len(proposal["formations"]), 15)
        self.assertEqual(list(coverage["situations"]), [
            "early downs", "passing downs", "goal line", "two-minute",
            "spread", "heavy", "mobile QB",
        ])
        for label, row in coverage["situations"].items():
            self.assertFalse(row["playable"])
            self.assertEqual(row["status"], "proposal")
            for formation in row["formations"]:
                self.assertIn(formation, proposal["formations"])

    def test_sixteen_formations_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            defense.design_defense(max_formations=16, catalogue=CATALOG)
        self.assertIn("1..15", str(caught.exception))


class HybridGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed={"opponents": {}})

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _entry(self) -> ModelRegistryEntry:
        return ModelRegistryEntry(
            artifact_path=str(Path(self.tmp.name) / "missing.json"),
            side=Possession.DEFENSE,
        )

    def test_below_forty_snaps_returns_no_metrics(self) -> None:
        report = Path(self.tmp.name) / "eval_report.json"
        result = defense.defense_hybrid_gate(self._entry(), [snap(i) for i in range(39)], seed=7)
        self.assertIs(result.gate, GateResult.INSUFFICIENT)
        self.assertFalse(result.gate_passed)
        self.assertEqual(result.metrics, ())
        self.assertFalse(report.exists())

    def test_forty_snaps_runs_the_standard_gate(self) -> None:
        result = defense.defense_hybrid_gate(self._entry(), [snap(i) for i in range(40)], seed=7)
        self.assertIs(result.gate, GateResult.FAILED)
        self.assertTrue(result.metrics)
        self.assertTrue(any(metric.name == "error" for metric in result.metrics))

    def test_live_hybrid_does_not_score_below_forty(self) -> None:
        defense.set_mode(self.db, "hybrid")
        call = MaddenCall(
            "defense", "Nickel Over", "Tampa 2", "No adj", "User hook", "heuristic",
        )
        situation = sit(extras={
            "session_id": "seed-game",
            "snap_seq": 7,
            "verified_defensive_snaps": [snap(i) for i in range(39)],
        })
        result = defense.maybe_apply_defense(call, situation, "gavin", self.db, BOOK)
        self.assertIs(result, call)
        self.assertFalse(hasattr(call, "joint_ranking"))
        decision = getattr(call, defense.DECISION_ATTR)
        self.assertTrue(decision.fell_back)
        self.assertIs(decision.mode, CoachingMode.HYBRID)
        self.assertIs(decision.policy_source, PolicySource.HEURISTIC)
        self.assertIsNot(decision.fallback_reason, MLStatus.OK)
        self.assertIs(decision.fallback_reason, MLStatus.LOW_CONFIDENCE)

    def test_user_entered_look_verification_does_not_count(self) -> None:
        from cfb_coach.madden.model.defense_intelligence import is_counted_defensive_snap

        look = snap(
            99,
            executed_verification="",
            opp_look_verification="verified",
        )
        self.assertFalse(is_counted_defensive_snap(look))
        rows = [snap(i) for i in range(39)]
        rows.append(look)
        self.assertEqual(defense.count_verified_defensive_snaps(rows), 39)
        report = Path(self.tmp.name) / "eval_report.json"
        result = defense.defense_hybrid_gate(self._entry(), rows, seed=7)
        self.assertIs(result.gate, GateResult.INSUFFICIENT)
        self.assertFalse(result.gate_passed)
        self.assertEqual(result.metrics, ())
        self.assertFalse(report.exists())


def _strip_rank_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in _RANK_KEYS_ADDED_AFTER_BASE}


class BothNullIdentityTests(unittest.TestCase):
    def test_both_null_rank_matches_the_2b372e2_snapshot(self) -> None:
        import json

        snapshot = json.loads(BOTH_NULL_SNAPSHOT.read_text(encoding="utf-8"))
        self.assertEqual(snapshot["source_commit"], "2b372e2")
        self.assertGreaterEqual(len(snapshot["cases"]), 3)
        book = snapshot["book"]
        for case in snapshot["cases"]:
            fields = case["situation"]
            situation = Situation(
                raw=fields["raw"],
                side=fields["side"],
                down=fields["down"],
                distance=fields["distance"],
                yardline=fields["yardline"],
                red_zone=fields["red_zone"],
                goal_line=fields["goal_line"],
                two_minute=fields["two_minute"],
            )
            self.assertIsNone(getattr(situation, "opp_personnel", None))
            self.assertIsNone(getattr(situation, "opp_set", None))
            ranks = defense.rank_defense(
                situation, book, None, opponent_id=snapshot["opponent_id"],
            )
            displayed = defense._stable_choice(ranks, snapshot["session_id"], case["snap_seq"])
            stripped = [_strip_rank_row(row) for row in ranks]
            shown = _strip_rank_row(displayed)
            self.assertEqual(stripped, case["ranks"])
            self.assertEqual(shown, case["displayed"])
            self.assertEqual(
                [(row["formation"], row["play"], row["score"]) for row in stripped],
                [(row["formation"], row["play"], row["score"]) for row in case["ranks"]],
            )
            self.assertEqual(shown["formation"], case["displayed"]["formation"])
            self.assertEqual(shown["play"], case["displayed"]["play"])


class DatabasePathTests(unittest.TestCase):
    def test_db_paths_use_an_explicit_argument_or_default_db_path(self) -> None:
        import os

        from cfb_coach.db import default_db_path
        from cfb_coach.madden.model.opponent_learning import readonly_madden_db_path

        for relative in (
            "cfb_coach/madden/model/opponent_learning.py",
            "cfb_coach/madden/model/defense_coordinator.py",
            "cfb_coach/madden/model/defense_intelligence.py",
            "cfb_coach/madden/model/defense_knowledge.py",
        ):
            body = (ROOT / relative).read_text(encoding="utf-8").split("from __future__", 1)[1]
            self.assertNotIn("Path.home", body)
            self.assertNotIn(".cfb-coach", body)
            self.assertNotIn("sqlite3.connect", body)
            self.assertNotIn("CoachDB(", body)

        for fn, name in (
            (defense.save_model, "path"),
            (defense.apply_design, "db"),
            (defense.stage_design, "db"),
            (defense.set_mode, "db"),
        ):
            param = inspect.signature(fn).parameters[name]
            self.assertIs(param.default, inspect.Parameter.empty)
        confirm_src = inspect.getsource(defense.confirm_design)
        self.assertNotIn("set_meta", confirm_src)
        self.assertNotIn("_save_state", confirm_src)
        self.assertNotIn("sqlite3", confirm_src)
        self.assertNotIn("CoachDB", confirm_src)

        explicit = Path("/tmp/explicit-policy.db")
        with mock.patch("pathlib.Path.home", side_effect=AssertionError("home")):
            self.assertEqual(readonly_madden_db_path(explicit), explicit)

        with tempfile.TemporaryDirectory() as tmp:
            chosen = Path(tmp) / "coach.db"
            saved = os.environ.get("CFB_COACH_DB")
            saved_madden = os.environ.get("CFB_COACH_MADDEN_DB")
            os.environ["CFB_COACH_DB"] = str(chosen)
            os.environ.pop("CFB_COACH_MADDEN_DB", None)
            try:
                with mock.patch("pathlib.Path.home", side_effect=AssertionError("home")):
                    self.assertEqual(default_db_path(), chosen)
                    self.assertNotEqual(readonly_madden_db_path(), chosen)
                    self.assertEqual(
                        defense._data_path(),
                        chosen.parent / "madden_ml_experimental" / "defense_coordinator.json",
                    )
                    self.assertIsNone(defense.load_model(None))
                    self.assertIsNone(defense.load_model(Path(tmp) / "missing-artifact.json"))
            finally:
                if saved is None:
                    os.environ.pop("CFB_COACH_DB", None)
                else:
                    os.environ["CFB_COACH_DB"] = saved
                if saved_madden is None:
                    os.environ.pop("CFB_COACH_MADDEN_DB", None)
                else:
                    os.environ["CFB_COACH_MADDEN_DB"] = saved_madden

    def test_readonly_path_is_the_madden_database_and_opens_read_only(self) -> None:
        import os
        import sqlite3

        from cfb_coach import db as coach_db
        from cfb_coach.games import madden_db_path as games_madden_db_path
        from cfb_coach.madden.model.opponent_learning import readonly_madden_db_path

        with tempfile.TemporaryDirectory() as tmp:
            coach = Path(tmp) / "coach.db"
            madden = Path(tmp) / "madden27.db"
            sqlite3.connect(coach).close()
            sqlite3.connect(madden).close()
            coach_hash = hashlib.sha256(coach.read_bytes()).hexdigest()
            madden_hash = hashlib.sha256(madden.read_bytes()).hexdigest()
            saved_coach = os.environ.get("CFB_COACH_DB")
            saved_madden = os.environ.get("CFB_COACH_MADDEN_DB")
            os.environ["CFB_COACH_DB"] = str(coach)
            os.environ["CFB_COACH_MADDEN_DB"] = str(madden)
            patcher = None
            try:
                if not callable(getattr(coach_db, "madden_db_path", None)):
                    # #46 has not added db.madden_db_path. The games helper
                    # already honors CFB_COACH_MADDEN_DB, then the patch
                    # installs that same callable where #46 will put it.
                    self.assertEqual(readonly_madden_db_path(), madden)
                    patcher = mock.patch.object(
                        coach_db, "madden_db_path", games_madden_db_path, create=True,
                    )
                    patcher.start()
                resolved = readonly_madden_db_path()
                self.assertEqual(resolved, madden)
                self.assertNotEqual(resolved, coach)
                opened = CoachDB.open_read_only(resolved)
                try:
                    self.assertTrue(opened.read_only)
                    self.assertEqual(Path(opened.path), madden)
                    with self.assertRaises(sqlite3.OperationalError):
                        opened.conn.execute("CREATE TABLE should_fail (id INTEGER)")
                finally:
                    opened.close()
                self.assertEqual(hashlib.sha256(madden.read_bytes()).hexdigest(), madden_hash)
                self.assertEqual(hashlib.sha256(coach.read_bytes()).hexdigest(), coach_hash)
            finally:
                if patcher is not None:
                    patcher.stop()
                if saved_coach is None:
                    os.environ.pop("CFB_COACH_DB", None)
                else:
                    os.environ["CFB_COACH_DB"] = saved_coach
                if saved_madden is None:
                    os.environ.pop("CFB_COACH_MADDEN_DB", None)
                else:
                    os.environ["CFB_COACH_MADDEN_DB"] = saved_madden


if __name__ == "__main__":
    unittest.main()
