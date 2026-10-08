"""Round-trip the Madden ML contracts. Importing them must not load an ML stack."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import unittest
from enum import Enum
from pathlib import Path
from typing import get_type_hints

from cfb_coach.game_score import GameScore
from cfb_coach.madden.model import policy
from cfb_coach.madden.model.schema import (
    CONTRACT_VERSION,
    CSV_UNKNOWN,
    ML_LATENCY_BUDGET_MS,
    ML_LOW_CONFIDENCE,
    PROVISIONAL_SCHEMA_VERSION,
    PROVISIONAL_TYPES,
    RANKING_STATUSES,
    SCHEMA_RECORD_TYPES,
    YARDS_MAX,
    AdjustmentRef,
    AppliedPlaybookRef,
    CandidatePlay,
    CandidateScore,
    CoachingDecision,
    CoachingMode,
    MLStatus,
    DataHash,
    DecisionStage,
    ExecutedPlay,
    ExecutedStatus,
    FeatureValue,
    FeatureVector,
    GameState,
    MetricRecord,
    MLConfig,
    ModelRegistryEntry,
    ObservedLook,
    OpponentContext,
    OpponentContextKey,
    OpponentKind,
    OutcomeEstimates,
    ParsedOutcome,
    PenaltyCall,
    PlayRanking,
    PolicySource,
    Possession,
    PreSnapObservation,
    PropensityMethod,
    PropensityOverrideReason,
    RankedPlay,
    RecentSnap,
    RosterFeature,
    RosterSnapshot,
    ScorePart,
    ShadowScore,
    Situation,
    SnapOutcome,
    StagePick,
    TendencyCount,
    Tri,
    Verification,
    candidate_in_book,
    csv_cell,
    from_dict,
    legal_candidates,
    parse_csv_cell,
    pre_snap_field_names,
    schema_version_for,
    to_dict,
)
from cfb_coach.outcome import ParsedOutcome as ParsedOutcomeType

ROOT = Path(__file__).resolve().parents[2]


def _round_trip(obj: object) -> object:
    raw = json.dumps(to_dict(obj))
    return from_dict(json.loads(raw))


class SchemaRoundTripTest(unittest.TestCase):
    def test_every_record_round_trips(self) -> None:
        samples = _samples()
        self.assertEqual(set(samples), set(SCHEMA_RECORD_TYPES))
        for cls, objs in samples.items():
            self.assertTrue(objs, cls.__name__)
            for obj in objs:
                self.assertIsInstance(obj, cls)
                again = _round_trip(obj)
                self.assertEqual(again, obj)
                self.assertEqual(to_dict(again), to_dict(obj))

    def test_provisional_types_use_their_own_version(self) -> None:
        self.assertEqual(PROVISIONAL_TYPES, (PlayRanking, OpponentContext))
        for cls in PROVISIONAL_TYPES:
            self.assertEqual(schema_version_for(cls), PROVISIONAL_SCHEMA_VERSION)
            payload = to_dict(cls())
            self.assertEqual(payload["__schema__"], PROVISIONAL_SCHEMA_VERSION)
        decision = to_dict(CoachingDecision())
        self.assertEqual(decision["__schema__"], CONTRACT_VERSION)
        leaked = dict(to_dict(PlayRanking()))
        leaked["__schema__"] = CONTRACT_VERSION
        with self.assertRaises(ValueError):
            from_dict(leaked)

    def test_propensity_fields_round_trip(self) -> None:
        exact = CoachingDecision(
            propensity_method=PropensityMethod.EXACT,
            behavior_propensity=0.25,
        )
        forced = CoachingDecision(
            propensity_method=PropensityMethod.DETERMINISTIC,
            behavior_propensity=1.0,
        )
        overridden = CoachingDecision(
            propensity_method=PropensityMethod.OVERRIDE,
            propensity_override=True,
            propensity_override_reason=PropensityOverrideReason.UNEXPRESSIBLE,
            behavior_propensity=0.4,
        )
        for obj in (exact, forced, overridden, CoachingDecision()):
            again = _round_trip(obj)
            self.assertEqual(again, obj)
        self.assertIs(forced.propensity_method, PropensityMethod.DETERMINISTIC)
        self.assertEqual(forced.behavior_propensity, 1.0)
        self.assertFalse(forced.propensity_override)
        self.assertIsNone(forced.propensity_override_reason)
        self.assertTrue(overridden.propensity_override)
        self.assertIs(overridden.propensity_override_reason, PropensityOverrideReason.UNEXPRESSIBLE)

    def test_legacy_decision_without_propensity_fields(self) -> None:
        payload = to_dict(
            CoachingDecision(mode=CoachingMode.SHADOW, shadow_status=MLStatus.OK, ml_seed=7)
        )
        payload.pop("propensity_method")
        payload.pop("propensity_override_reason")
        loaded = from_dict(json.loads(json.dumps(payload)))
        self.assertIs(loaded.propensity_method, PropensityMethod.UNKNOWN)
        self.assertIsNone(loaded.propensity_override_reason)
        self.assertFalse(loaded.propensity_override)
        self.assertEqual(loaded.ml_seed, 7)
        self.assertIs(loaded.mode, CoachingMode.SHADOW)

        bare = from_dict({"__type__": "CoachingDecision"})
        self.assertEqual(bare, CoachingDecision())
        self.assertIs(bare.propensity_method, PropensityMethod.UNKNOWN)
        self.assertIsNone(bare.propensity_override_reason)
        self.assertIs(bare.accepted, Tri.UNKNOWN)
        self.assertIs(bare.executed_status, ExecutedStatus.UNKNOWN)
        self.assertIsNone(bare.executed)

    def test_propensity_flag_must_match_method(self) -> None:
        with self.assertRaises(ValueError):
            CoachingDecision(propensity_method=PropensityMethod.OVERRIDE, propensity_override=False)
        with self.assertRaises(ValueError):
            CoachingDecision(
                propensity_method=PropensityMethod.EXACT,
                propensity_override=True,
                propensity_override_reason=PropensityOverrideReason.CAP,
            )
        with self.assertRaises(ValueError):
            CoachingDecision(propensity_override_reason=PropensityOverrideReason.OTHER)
        with self.assertRaises(ValueError):
            CoachingDecision(
                propensity_method=PropensityMethod.OVERRIDE,
                propensity_override=True,
                propensity_override_reason=None,
            )
        bad = to_dict(
            CoachingDecision(
                propensity_method=PropensityMethod.OVERRIDE,
                propensity_override=True,
                propensity_override_reason=PropensityOverrideReason.CAP,
            )
        )
        bad["propensity_override"] = False
        with self.assertRaises(ValueError):
            from_dict(bad)

    def test_ml_status_round_trip_and_legacy(self) -> None:
        shadow = CoachingDecision(
            mode=CoachingMode.SHADOW,
            shadow_status=MLStatus.OK,
            ml_seed=3,
        )
        hybrid = CoachingDecision(
            mode=CoachingMode.HYBRID,
            policy_source=PolicySource.HEURISTIC,
            shadow_status=MLStatus.TIMEOUT,
            fell_back=True,
            fallback_reason=MLStatus.TIMEOUT,
        )
        model_call = CoachingDecision(
            mode=CoachingMode.HYBRID,
            policy_source=PolicySource.MODEL,
            shadow_status=MLStatus.LOW_CONFIDENCE,
            fell_back=False,
        )
        for obj in (shadow, hybrid, model_call, CoachingDecision()):
            again = _round_trip(obj)
            self.assertEqual(again, obj)
        encoded = to_dict(shadow)
        self.assertEqual(encoded["shadow_status"], "ok")
        self.assertIsNone(encoded["fallback_reason"])
        self.assertEqual(to_dict(hybrid)["fallback_reason"], "timeout")
        for status in MLStatus:
            row = CoachingDecision(mode=CoachingMode.SHADOW, shadow_status=status)
            self.assertIs(_round_trip(row).shadow_status, status)

        legacy = to_dict(CoachingDecision(ml_seed=9))
        legacy.pop("shadow_status")
        legacy.pop("fallback_reason")
        legacy.pop("mode")
        loaded = from_dict(json.loads(json.dumps(legacy)))
        self.assertEqual(loaded, CoachingDecision(ml_seed=9))
        self.assertIs(loaded.mode, CoachingMode.HEURISTIC)
        self.assertIs(loaded.policy_source, PolicySource.HEURISTIC)
        self.assertFalse(loaded.fell_back)
        self.assertIsNone(loaded.fallback_reason)
        self.assertIsNone(loaded.shadow_status)

        bare = from_dict({"__type__": "CoachingDecision"})
        self.assertIsNone(bare.shadow_status)
        self.assertIsNone(bare.fallback_reason)
        self.assertFalse(bare.fell_back)
        self.assertIs(bare.mode, CoachingMode.HEURISTIC)

    def test_ml_status_invalid_combinations(self) -> None:
        shadow_ok = dict(mode=CoachingMode.SHADOW, shadow_status=MLStatus.OK)
        hybrid_ok = dict(
            mode=CoachingMode.HYBRID,
            shadow_status=MLStatus.OK,
            policy_source=PolicySource.MODEL,
        )
        invalid = [
            dict(fallback_reason=MLStatus.OK),
            dict(fell_back=True, fallback_reason=MLStatus.OK, mode=CoachingMode.HYBRID, shadow_status=MLStatus.OK),
            dict(mode=CoachingMode.HEURISTIC, fell_back=True, fallback_reason=MLStatus.TIMEOUT),
            dict(mode=CoachingMode.HEURISTIC, fell_back=None),
            dict(mode=CoachingMode.HEURISTIC, fallback_reason=MLStatus.TIMEOUT),
            dict(mode=CoachingMode.HEURISTIC, policy_source=PolicySource.MODEL),
            dict(mode=CoachingMode.HEURISTIC, policy_source=PolicySource.FALLBACK),
            dict(mode=CoachingMode.HEURISTIC, policy_source=PolicySource.UNKNOWN),
            dict(mode=CoachingMode.HEURISTIC, shadow_status=MLStatus.OK),
            dict(mode=CoachingMode.HEURISTIC, shadow_status=MLStatus.TIMEOUT),
            dict(shadow_ok, fell_back=True, fallback_reason=MLStatus.TIMEOUT),
            dict(shadow_ok, fell_back=None),
            dict(shadow_ok, fallback_reason=MLStatus.WORKER_BUSY),
            dict(shadow_ok, policy_source=PolicySource.MODEL),
            dict(mode=CoachingMode.SHADOW, shadow_status=None),
            dict(mode=CoachingMode.HYBRID, shadow_status=None, fell_back=False),
            dict(mode=CoachingMode.HYBRID, shadow_status=MLStatus.TIMEOUT, fell_back=True, fallback_reason=None),
            dict(
                mode=CoachingMode.HYBRID,
                shadow_status=MLStatus.TIMEOUT,
                fell_back=True,
                fallback_reason=MLStatus.TIMEOUT,
                policy_source=PolicySource.MODEL,
            ),
            dict(
                mode=CoachingMode.HYBRID,
                shadow_status=MLStatus.CIRCUIT_OPEN,
                fell_back=True,
                fallback_reason=MLStatus.CIRCUIT_OPEN,
                policy_source=PolicySource.FALLBACK,
            ),
            dict(
                mode=CoachingMode.HYBRID,
                shadow_status=MLStatus.EXCEPTION,
                fell_back=True,
                fallback_reason=MLStatus.EXCEPTION,
                policy_source=PolicySource.UNKNOWN,
            ),
            dict(hybrid_ok, fell_back=False, fallback_reason=MLStatus.MODEL_MISSING),
            dict(mode=CoachingMode.UNKNOWN, fell_back=True, fallback_reason=MLStatus.INVALID_OUTPUT),
        ]
        for kwargs in invalid:
            with self.assertRaises(ValueError, msg=kwargs):
                CoachingDecision(**kwargs)
            payload = to_dict(CoachingDecision())
            for key, value in kwargs.items():
                payload[key] = value.value if isinstance(value, Enum) else value
            with self.assertRaises(ValueError, msg=kwargs):
                from_dict(payload)

    def test_rank_plays_signature(self) -> None:
        params = list(inspect.signature(policy.rank_plays).parameters)
        self.assertEqual(
            params,
            [
                "game_state",
                "candidate_plays",
                "roster_context",
                "opponent_context",
                "recent_history",
            ],
        )
        self.assertNotIn("recent_history", {item.name for item in GameState.__dataclass_fields__.values()})
        self.assertNotIn("recent", {item.name for item in GameState.__dataclass_fields__.values()})
        history = get_type_hints(policy.rank_plays)["recent_history"]
        self.assertIn("CoachingDecision", str(history))
        self.assertIn("SnapOutcome", str(history))
        self.assertNotIn("RecentSnap", str(history))

    def test_rank_plays_history_and_ranking_status(self) -> None:
        self.assertEqual(
            RANKING_STATUSES,
            frozenset(
                {
                    MLStatus.OK,
                    MLStatus.LOW_CONFIDENCE,
                    MLStatus.MODEL_MISSING,
                    MLStatus.INVALID_OUTPUT,
                }
            ),
        )
        self.assertNotIn("fallback_reason", PlayRanking.__dataclass_fields__)
        self.assertNotIn("stale", OpponentContext.__dataclass_fields__)
        detail = PlayRanking(status=MLStatus.LOW_CONFIDENCE, status_detail="confidence=0.41")
        self.assertEqual(_round_trip(detail), detail)
        self.assertEqual(to_dict(detail)["status"], "low_confidence")
        for status in RANKING_STATUSES:
            self.assertEqual(_round_trip(PlayRanking(status=status)).status, status)

        legacy = to_dict(PlayRanking(policy_version="policy-0.1.0"))
        legacy.pop("status")
        legacy.pop("status_detail")
        loaded = from_dict(json.loads(json.dumps(legacy)))
        self.assertIs(loaded.status, MLStatus.OK)
        self.assertIsNone(loaded.status_detail)
        self.assertEqual(loaded.policy_version, "policy-0.1.0")

        bare = from_dict({"__type__": "PlayRanking"})
        self.assertIs(bare.status, MLStatus.OK)
        self.assertIsNone(bare.status_detail)

        for status in (
            MLStatus.WORKER_BUSY,
            MLStatus.TIMEOUT,
            MLStatus.EXCEPTION,
            MLStatus.CIRCUIT_OPEN,
            MLStatus.UNKNOWN,
        ):
            with self.assertRaises(ValueError):
                PlayRanking(status=status)
            payload = to_dict(PlayRanking())
            payload["status"] = status.value
            with self.assertRaises(ValueError):
                from_dict(payload)

    def test_opponent_seed_pair(self) -> None:
        row_key = dict(opponent_id="james", patch_id="1.007", roster_snapshot_id="r2", table_version="1")
        prior = OpponentContextKey(opponent_id="james", patch_id="1.006", roster_snapshot_id="r1", table_version="1")
        seeded = OpponentContext(**row_key, seeded_from_key=prior, seed_shrink_weight=0.25)
        self.assertEqual(_round_trip(seeded), seeded)
        self.assertEqual(_round_trip(OpponentContext(**row_key, seeded_from_key=prior, seed_shrink_weight=0.0)).seed_shrink_weight, 0.0)
        self.assertEqual(_round_trip(OpponentContext(**row_key, seeded_from_key=prior, seed_shrink_weight=1.0)).seed_shrink_weight, 1.0)

        legacy = to_dict(OpponentContext(opponent_id="james"))
        legacy.pop("seeded_from_key")
        legacy.pop("seed_shrink_weight")
        loaded = from_dict(json.loads(json.dumps(legacy)))
        self.assertIsNone(loaded.seeded_from_key)
        self.assertIsNone(loaded.seed_shrink_weight)
        self.assertEqual(loaded.opponent_id, "james")
        bare = from_dict({"__type__": "OpponentContext"})
        self.assertIsNone(bare.seeded_from_key)
        self.assertIsNone(bare.seed_shrink_weight)

        invalid = [
            dict(row_key, seeded_from_key=prior, seed_shrink_weight=None),
            dict(row_key, seeded_from_key=None, seed_shrink_weight=0.25),
            dict(row_key, seeded_from_key=prior, seed_shrink_weight=-0.01),
            dict(row_key, seeded_from_key=prior, seed_shrink_weight=1.01),
            dict(row_key, seeded_from_key=OpponentContextKey(opponent_id="cpu", patch_id="1.006"), seed_shrink_weight=0.25),
            dict(row_key, seeded_from_key=OpponentContextKey(**row_key), seed_shrink_weight=0.25),
        ]
        for kwargs in invalid:
            with self.assertRaises(ValueError, msg=kwargs):
                OpponentContext(**kwargs)
            payload = to_dict(OpponentContext(**row_key))
            payload["seeded_from_key"] = to_dict(kwargs["seeded_from_key"]) if kwargs["seeded_from_key"] else None
            payload["seed_shrink_weight"] = kwargs["seed_shrink_weight"]
            with self.assertRaises(ValueError, msg=kwargs):
                from_dict(payload)

    def test_unknown_convention_and_enums(self) -> None:
        self.assertEqual(csv_cell(None), CSV_UNKNOWN)
        self.assertIsNone(parse_csv_cell(CSV_UNKNOWN))
        self.assertEqual(csv_cell(PropensityMethod.UNKNOWN), "unknown")
        self.assertNotEqual(csv_cell(PropensityMethod.UNKNOWN), CSV_UNKNOWN)
        for cls in SCHEMA_RECORD_TYPES:
            module = sys.modules[cls.__module__]
            break
        enum_types = [
            obj
            for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, Enum) and obj is not Enum
        ]
        self.assertGreaterEqual(len(enum_types), 8)
        for enum_type in enum_types:
            self.assertIn("UNKNOWN", enum_type.__members__, enum_type.__name__)
        self.assertEqual(ML_LOW_CONFIDENCE, 0.6)
        self.assertEqual(ML_LATENCY_BUDGET_MS, 150)
        self.assertIs(CoachingDecision().mode, CoachingMode.HEURISTIC)
        with self.assertRaises(ValueError):
            MLConfig(latency_budget_ms=10)
        with self.assertRaises(ValueError):
            MLConfig(low_confidence=0.2)

    def test_guards(self) -> None:
        book = {"Gun Bunch": ["Mesh Post"]}
        ok = candidate_in_book(
            book,
            side=Possession.OFFENSE,
            formation="Gun Bunch",
            play="Mesh Post",
            revision=2,
            adjustment_slot=7,
            adjustment_id="MAN",
        )
        self.assertIs(ok.in_applied_book, Tri.TRUE)
        self.assertEqual(legal_candidates((ok, CandidatePlay())), (ok,))
        with self.assertRaises(ValueError):
            candidate_in_book(book, side=Possession.OFFENSE, formation="Gun Bunch", play="Not A Play")
        with self.assertRaises(ValueError):
            candidate_in_book(
                book,
                side=Possession.OFFENSE,
                formation="Gun Bunch",
                play="Mesh Post",
                adjustment_slot=8,
            )
        with self.assertRaises(ValueError):
            SnapOutcome(yards=YARDS_MAX + 1)
        with self.assertRaises(ValueError):
            SnapOutcome(penalty=PenaltyCall.ACCEPTED, success=Tri.TRUE)
        with self.assertRaises(ValueError):
            SnapOutcome(no_play=Tri.TRUE, success=Tri.FALSE)
        with self.assertRaises(ValueError):
            SnapOutcome(sack=Tri.TRUE, yards=4, success=Tri.FALSE)
        with self.assertRaises(ValueError):
            CoachingDecision(
                executed_status=ExecutedStatus.UNKNOWN,
                executed=ExecutedPlay(formation="Gun Bunch", play="Mesh Post", verification=Verification.VERIFIED),
            )
        self.assertTrue(LEAKAGE_DISJOINT)

    def test_import_does_not_pull_optional_ml(self) -> None:
        script = """
import sys
mods = [
    "cfb_coach.madden.model",
    "cfb_coach.madden.model.schema",
    "cfb_coach.madden.model.features",
    "cfb_coach.madden.model.dataset",
    "cfb_coach.madden.model.train",
    "cfb_coach.madden.model.evaluate",
    "cfb_coach.madden.model.inference",
    "cfb_coach.madden.model.policy",
    "cfb_coach.madden.model.registry",
    "cfb_coach.madden.model.adapters",
]
for name in mods:
    __import__(name)
banned = {"numpy", "sklearn", "lightgbm", "catboost", "xgboost", "torch", "pandas"}
hit = sorted(key for key in sys.modules if key.split(".")[0] in banned)
if hit:
    raise SystemExit("optional ML imported: " + ", ".join(hit))
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


LEAKAGE_DISJOINT = pre_snap_field_names().isdisjoint(
    {
        "yards",
        "result",
        "touchdown",
        "sack",
        "interception",
        "fumble",
        "explosive",
        "first_down",
        "stop",
        "incompletion",
        "drive_result",
        "success",
        "executed",
        "parsed",
    }
)


def _samples() -> dict[type, list[object]]:
    book = candidate_in_book(
        {"Gun Bunch": ["Mesh Post", "Inside Zone"]},
        side=Possession.OFFENSE,
        formation="Gun Bunch",
        play="Mesh Post",
        revision=3,
        source_book="Buccaneers",
        mode="custom",
        adjustment_slot=0,
        adjustment_id="MAN",
        audible=Tri.FALSE,
        heuristic_score=0.2,
        heuristic_p=0.1,
    )
    estimates = OutcomeEstimates(success_probability=0.55, expected_yards=4.0, n=12)
    ranked = RankedPlay(
        candidate=book,
        estimates=estimates,
        confidence=0.7,
        reasons=("in book",),
        rank=1,
        sampling_p=0.1,
        sample_size=12,
        shrinkage_weight=0.4,
        score_breakdown=(ScorePart(name="sit", value=0.12),),
    )
    decision = CoachingDecision(
        mode=CoachingMode.SHADOW,
        policy_source=PolicySource.HEURISTIC,
        propensity_method=PropensityMethod.DETERMINISTIC,
        behavior_propensity=1.0,
        ml_seed=None,
        final_pick=book,
        fell_back=False,
        shadow_status=MLStatus.OK,
        opponent_context_key=OpponentContextKey(opponent_id="james", patch_id="1.0", roster_snapshot_id="r1"),
        shadow_scores=(ShadowScore(candidate_key="Gun Bunch::Mesh Post", score=0.4, p=0.55),),
        inference_budget_ms=ML_LATENCY_BUDGET_MS,
    )
    hybrid_fallback = CoachingDecision(
        mode=CoachingMode.HYBRID,
        policy_source=PolicySource.HEURISTIC,
        shadow_status=MLStatus.TIMEOUT,
        fell_back=True,
        fallback_reason=MLStatus.TIMEOUT,
        final_pick=book,
    )
    context = OpponentContext(
        opponent_id="james",
        patch_id="tu",
        roster_snapshot_id="r1",
        table_version="1",
        opponent_type=OpponentKind.HUMAN,
        n_snaps=4,
        tendencies_known=Tri.TRUE,
        tendencies=(TendencyCount(bucket="coverage", key="cover 3", count=4, success=2, sample_size=4),),
    )
    return {
        AppliedPlaybookRef: [AppliedPlaybookRef(), AppliedPlaybookRef(side=Possession.DEFENSE, revision=1, mode="stock")],
        AdjustmentRef: [AdjustmentRef(), AdjustmentRef(side=Possession.OFFENSE, slot_index=1, macro_id="MAN")],
        CandidatePlay: [CandidatePlay(), book],
        GameState: [GameState(), GameState(opponent_type=OpponentKind.CPU, possession=Possession.OFFENSE, quarter=4, situation=Situation(raw="1&10", down=1, distance=10), score=GameScore(us=7, them=3))],
        PreSnapObservation: [PreSnapObservation(), PreSnapObservation(coverage_prediction="Cover 3", coverage_prediction_source=__import__("cfb_coach.madden.model.schema", fromlist=["HintSource"]).HintSource.LIVE)],
        ObservedLook: [ObservedLook(), ObservedLook(coverage="Cover 3", verification=Verification.VERIFIED)],
        SnapOutcome: [SnapOutcome(), SnapOutcome(yards=-6, sack=Tri.TRUE, success=Tri.FALSE, verification=Verification.VERIFIED)],
        ExecutedPlay: [ExecutedPlay(), ExecutedPlay(formation="Gun Bunch", play="Mesh Post", verification=Verification.VERIFIED)],
        RosterFeature: [RosterFeature(key="qb_throw_power"), RosterFeature(key="qb_throw_power", value_number=1.0, missing=False)],
        RosterSnapshot: [RosterSnapshot(), RosterSnapshot(roster_version="v1", completeness=Tri.FALSE, features=(RosterFeature(key="qb_throw_power"),))],
        TendencyCount: [TendencyCount(), TendencyCount(bucket="coverage", key="cover 3", count=3, sample_size=3, shrinkage_weight=0.5)],
        OpponentContextKey: [OpponentContextKey(), OpponentContextKey(opponent_id="james", patch_id="tu", roster_snapshot_id="r1", table_version="1")],
        OpponentContext: [OpponentContext(), context],
        RecentSnap: [RecentSnap(), RecentSnap(snap_id="s1", possession=Possession.OFFENSE, play="Mesh Post", outcome=SnapOutcome(yards=6, success=Tri.TRUE))],
        OutcomeEstimates: [OutcomeEstimates(), estimates],
        CandidateScore: [CandidateScore(), CandidateScore(candidate_index=0, source="heuristic", total=0.2)],
        StagePick: [StagePick(stage=DecisionStage.VOD_PRIOR), StagePick(stage=DecisionStage.FINAL, formation="Gun Bunch", play="Mesh Post", changed=Tri.TRUE)],
        ScorePart: [ScorePart(name="sit"), ScorePart(name="sit", value=0.12)],
        ShadowScore: [ShadowScore(), ShadowScore(candidate_key="k", score=0.2, p=0.3)],
        RankedPlay: [RankedPlay(candidate=CandidatePlay(), estimates=OutcomeEstimates()), ranked],
        PlayRanking: [PlayRanking(), PlayRanking(policy_source=PolicySource.HEURISTIC, plays=(ranked,), low_confidence=Tri.FALSE)],
        CoachingDecision: [CoachingDecision(), decision, hybrid_fallback],
        FeatureValue: [FeatureValue(name="down"), FeatureValue(name="down", value=1.0, missing=False)],
        FeatureVector: [FeatureVector(), FeatureVector(items=(FeatureValue(name="down", value=1.0, missing=False),))],
        DataHash: [DataHash(label="snaps"), DataHash(label="snaps", sha256="abc", uri="file://snaps")],
        MetricRecord: [MetricRecord(name="log_loss"), MetricRecord(name="log_loss", value=0.5, split="held_out", n=10)],
        ModelRegistryEntry: [ModelRegistryEntry(), ModelRegistryEntry(model_version="none", gate=__import__("cfb_coach.madden.model.schema", fromlist=["GateResult"]).GateResult.NOT_RUN, seed=1)],
        MLConfig: [MLConfig(), MLConfig(mode=CoachingMode.SHADOW, seed=1)],
        Situation: [Situation(raw=""), Situation(raw="1&10", down=1, distance=10, side="offense")],
        GameScore: [GameScore(us=0, them=0), GameScore(us=7, them=3)],
        ParsedOutcome: [ParsedOutcome(raw="", kind="unknown"), ParsedOutcomeType(raw="gain 6", kind="gain", yards=6, success=True, label="+6")],
    }


if __name__ == "__main__":
    unittest.main()
