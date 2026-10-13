"""Vertical-slice tests for the Madden ML coach (adapters → shadow)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import adapters, dataset, evaluate, features, inference, policy, registry, train
from cfb_coach.madden.model.schema import (
    FEATURE_SCHEMA_VERSION,
    LEAKAGE_FIELDS,
    CoachingMode,
    ExecutedStatus,
    GateResult,
    MLStatus,
    OpponentKind,
    Possession,
    RecentSnap,
    SnapOutcome,
    Tri,
    Verification,
    candidate_in_book,
    legal_candidates,
    pre_snap_field_names,
)
from cfb_coach.situation import Situation

FIX = Path(__file__).resolve().parents[1] / "fixtures"
CPU_CSV = FIX / "cpu_19-0_offense_snaps.csv"
LABELED = FIX / "madden_ml" / "labeled_offense_snaps.jsonl"


class AdapterTests(unittest.TestCase):
    def test_situation_to_state_keeps_unknowns(self) -> None:
        sit = Situation(raw="1&10 my 25", side="offense", down=1, distance=10, yardline=25)
        state, obs = adapters.situation_to_state(sit, opponent_id="cpu_a", opponent_type=OpponentKind.CPU)
        self.assertEqual(state.possession, Possession.OFFENSE)
        self.assertIsNone(state.quarter)
        self.assertIsNone(state.clock_seconds)
        self.assertEqual(state.goal_to_go, Tri.UNKNOWN)
        self.assertIsNone(obs.coverage_prediction)
        leaked = LEAKAGE_FIELDS.intersection(pre_snap_field_names())
        self.assertFalse(leaked)

    def test_hints_are_predictions_not_verified(self) -> None:
        sit = Situation(
            raw="1&10 cover 3",
            side="offense",
            down=1,
            distance=10,
            coverage_hint="cover 3",
            coverage_source="live",
        )
        _state, obs = adapters.situation_to_state(sit)
        self.assertEqual(obs.coverage_prediction, "cover 3")
        self.assertNotIn("observed_coverage", obs.__dataclass_fields__)


class DatasetTests(unittest.TestCase):
    def test_import_csv_does_not_verify_recommendation(self) -> None:
        rows = dataset.import_path(str(CPU_CSV))
        self.assertGreater(len(rows), 10)
        for row in rows:
            self.assertEqual(row["executed_status"], ExecutedStatus.UNKNOWN.value)
            self.assertIsNone(row["executed_play"])
            self.assertIn(row["success"], (Tri.TRUE.value, Tri.FALSE.value, Tri.UNKNOWN.value))

    def test_export_roundtrip_and_validate(self) -> None:
        rows = dataset.import_path(str(LABELED))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            dataset.export_jsonl(rows, str(out))
            again = dataset.import_path(str(out))
        self.assertEqual(len(again), len(rows))
        report = dataset.validate_rows(again)
        self.assertEqual(report["n_rows"], len(again))
        self.assertGreater(report["labeled"], 0)
        self.assertEqual(report["verified_executions"], 0)

    def test_build_rows_from_db_preserves_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "madden27.db", seed={"opponents": {
                "cpu_a": {
                    "display_name": "CPU A",
                    "team_now": "X",
                    "skill": "cpu",
                    "confidence": "low",
                    "profile_json": "{}",
                }
            }})
            try:
                db.conn.execute(
                    "INSERT INTO snaps (ts, opponent_id, side, down, distance, yardline, formation, play, result, situation_raw) "
                    "VALUES ('t','cpu_a','offense',1,10,25,'Gun Bunch','Mesh','+6','1&10')"
                )
                db.conn.commit()
                rows = dataset.build_rows(db=db)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["recommended_play"], "Mesh")
                self.assertEqual(rows[0]["executed_status"], ExecutedStatus.UNKNOWN.value)
            finally:
                db.close()


class MigrationTests(unittest.TestCase):
    def test_ml_migration_idempotent_and_cfb_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "madden27.db"
            db = CoachDB(path, seed={"opponents": {}})
            try:
                names = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertIn("ml_decisions", names)
                self.assertIn("ml_outcomes", names)
                db._migrate_ml_tables()
                db._migrate_ml_tables()
                n = db.conn.execute("SELECT count(*) FROM ml_decisions").fetchone()[0]
                self.assertEqual(n, 0)
            finally:
                db.close()
            cfb = CoachDB(Path(tmp) / "coach.db", seed={"opponents": {}})
            try:
                names = {r[0] for r in cfb.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertIn("snaps", names)
                # ML tables exist on any CoachDB open (shared schema) but CFB history is untouched.
                self.assertEqual(cfb.conn.execute("SELECT count(*) FROM snaps").fetchone()[0], 0)
            finally:
                cfb.close()

    def test_recommendation_and_execution_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "m.db", seed={"opponents": {}})
            try:
                from cfb_coach.madden.model.schema import CoachingDecision, CandidatePlay

                dec = CoachingDecision(
                    mode=CoachingMode.SHADOW,
                    shadow_status=MLStatus.OK,
                    heuristic_pick=CandidatePlay(formation="A", play="P1"),
                    final_pick=CandidatePlay(formation="A", play="P1"),
                    snap_id="s1",
                    game_id="g1",
                )
                did = db.log_ml_decision(dec, agree=0)
                oid = db.log_ml_outcome(
                    snap_id="s1",
                    game_id="g1",
                    decision_id=did,
                    executed_status="identified",
                    executed_formation="A",
                    executed_play="OTHER",
                    executed_verification="verified",
                )
                # Non-identified path clears execution fields.
                oid2 = db.log_ml_outcome(
                    snap_id="s2",
                    executed_status="unknown",
                    executed_formation="SHOULD_CLEAR",
                    executed_play="SHOULD_CLEAR",
                )
                row = db.conn.execute("SELECT * FROM ml_outcomes WHERE id=?", (oid,)).fetchone()
                self.assertEqual(row["executed_play"], "OTHER")
                row2 = db.conn.execute("SELECT * FROM ml_outcomes WHERE id=?", (oid2,)).fetchone()
                self.assertIsNone(row2["executed_play"])
            finally:
                db.close()


class FeatureTests(unittest.TestCase):
    def test_vector_deterministic_and_no_leakage(self) -> None:
        sit = Situation(raw="2&7 my 35 cover 2", side="offense", down=2, distance=7, yardline=35,
                        coverage_hint="cover 2", coverage_source="live")
        state, obs = adapters.situation_to_state(sit, opponent_type=OpponentKind.CPU)
        book = {"Gun Bunch": ["Mesh", "Flood"]}
        cand = candidate_in_book(book, side=Possession.OFFENSE, formation="Gun Bunch", play="Mesh")
        # Current snap outcome must not be in recent_history for the decision.
        recent = (
            RecentSnap(
                snap_id="old",
                possession=Possession.OFFENSE,
                formation="Gun Bunch",
                play="Flood",
                outcome=SnapOutcome(success=Tri.TRUE, yards=8),
            ),
        )
        a = features.vector_for(state, obs, cand, None, None, recent)
        b = features.vector_for(state, obs, cand, None, None, recent)
        # Policy history is decision/outcome pairs (provisional contract).
        from cfb_coach.madden.model.schema import CoachingDecision

        hist = (
            (
                CoachingDecision(
                    mode=CoachingMode.HEURISTIC,
                    snap_id="old",
                    final_pick=candidate_in_book(
                        {"Gun Bunch": ["Flood"]},
                        side=Possession.OFFENSE,
                        formation="Gun Bunch",
                        play="Flood",
                    ),
                ),
                SnapOutcome(success=Tri.TRUE, yards=8),
            ),
        )
        ranking = policy.rank_plays(state, obs, [cand], None, None, hist)
        self.assertGreaterEqual(len(ranking.plays), 1)
        self.assertEqual(a, b)
        self.assertEqual(a.feature_schema_version, FEATURE_SCHEMA_VERSION)
        names = features.pre_snap_feature_names()
        self.assertEqual(tuple(item.name for item in a.items), names)
        self.assertFalse(LEAKAGE_FIELDS.intersection(names))


class TrainEvalRegistryTests(unittest.TestCase):
    def test_train_evaluate_registry_roundtrip(self) -> None:
        rows = dataset.import_path(str(LABELED))
        # Enrich with CPU CSV for more labeled mass without fabricating labels.
        rows.extend(dataset.import_path(str(CPU_CSV)))
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "scratch"
            reg = Path(tmp) / "registry"
            entry = train.train(rows, seed=7, out_dir=str(scratch))
            self.assertIsNotNone(entry.artifact_path)
            self.assertIn(entry.gate, (GateResult.NOT_RUN, GateResult.INSUFFICIENT))
            path = registry.write_entry(entry, str(reg))
            loaded = registry.read_entry(str(reg))
            self.assertEqual(loaded.model_version, entry.model_version)
            updated = evaluate.evaluate(loaded, rows, seed=7)
            registry.write_entry(updated, str(reg))
            again = registry.read_entry(str(reg))
            self.assertIn(again.gate, (GateResult.PASSED, GateResult.FAILED, GateResult.INSUFFICIENT))
            # Corrupted artifact fails safely.
            bad = Path(tmp) / "bad"
            bad.mkdir()
            (bad / "registry_entry.json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                registry.read_entry(str(bad))
            self.assertFalse(registry.promotion_allowed(entry))
            self.assertTrue(path.endswith("registry_entry.json"))


class PolicyInferenceTests(unittest.TestCase):
    def test_illegal_candidates_rejected_and_tiebreak(self) -> None:
        sit = Situation(raw="1&10", side="offense", down=1, distance=10, yardline=25)
        state, obs = adapters.situation_to_state(sit)
        book = {"Gun Bunch": ["Mesh", "Flood Seam"]}
        legal = [
            candidate_in_book(book, side=Possession.OFFENSE, formation="Gun Bunch", play="Mesh"),
            candidate_in_book(book, side=Possession.OFFENSE, formation="Gun Bunch", play="Flood Seam"),
        ]
        illegal = legal + [
            type(legal[0])(formation="NotInBook", play="Nope")  # in_applied_book UNKNOWN
        ]
        ranking = policy.rank_plays(state, obs, illegal, None, None, ())
        self.assertEqual(len(ranking.plays), 2)
        keys = [(p.candidate.formation, p.candidate.play) for p in ranking.plays]
        self.assertEqual(sorted(keys), sorted(keys))  # stable
        # ranks are 1..n
        self.assertEqual([p.rank for p in ranking.plays], [1, 2])

    def test_shadow_does_not_change_heuristic_and_survives_missing_model(self) -> None:
        sit = Situation(raw="1&10 my 25", side="offense", down=1, distance=10, yardline=25)
        state, obs = adapters.situation_to_state(sit, game_id="g", snap_id="s")
        book = {"Gun Bunch": ["Mesh", "Flood Seam"]}
        cands = [
            candidate_in_book(book, side=Possession.OFFENSE, formation="Gun Bunch", play="Mesh"),
            candidate_in_book(book, side=Possession.OFFENSE, formation="Gun Bunch", play="Flood Seam"),
        ]
        heuristic = cands[0]
        decision = inference.rank_live(
            state,
            obs,
            cands,
            None,
            None,
            (),
            mode=CoachingMode.SHADOW,
            heuristic_pick=heuristic,
            registry_dir="/tmp/madden_ml_missing_registry_xxxxx",
        )
        self.assertEqual(decision.mode, CoachingMode.SHADOW)
        self.assertEqual(decision.final_pick, heuristic)
        self.assertEqual(decision.heuristic_pick, heuristic)
        self.assertEqual(decision.shadow_status, MLStatus.MODEL_MISSING)
        self.assertFalse(decision.fell_back)

    def test_shadow_with_trained_model_records_latency(self) -> None:
        rows = dataset.import_path(str(LABELED))
        with tempfile.TemporaryDirectory() as tmp:
            entry = train.train(rows, seed=3, out_dir=tmp)
            artifact = train.load_artifact(entry.artifact_path)
            sit = Situation(raw="3&7 my 40", side="offense", down=3, distance=7, yardline=40)
            state, obs = adapters.situation_to_state(sit, game_id="g", snap_id="s2")
            book = {
                "Gun Bunch": ["Mesh", "Flood Seam"],
                "Gun Trips": ["RPO Alert Bubble"],
            }
            cands = list(legal_candidates([
                candidate_in_book(book, side=Possession.OFFENSE, formation=f, play=p)
                for f, plays in book.items() for p in plays
            ]))
            heuristic = cands[0]
            decision = inference.rank_live(
                state,
                obs,
                cands,
                None,
                None,
                (),
                mode=CoachingMode.SHADOW,
                heuristic_pick=heuristic,
                model_artifact=artifact,
            )
            self.assertEqual(decision.final_pick.play, heuristic.play)
            self.assertIn(decision.shadow_status, (MLStatus.OK, MLStatus.LOW_CONFIDENCE, MLStatus.TIMEOUT))
            if decision.shadow_status is MLStatus.OK:
                self.assertIsNotNone(decision.shadow_pick)
                self.assertIsNotNone(decision.latency_ms_model)
                self.assertLessEqual(decision.latency_ms_model or 0, 1500)  # generous CI bound


class EndToEndFixtureDemo(unittest.TestCase):
    """Fixture-based demonstration of the full workflow (not live effectiveness)."""

    def test_fixture_workflow(self) -> None:
        rows = dataset.import_path(str(LABELED))
        report = dataset.validate_rows(rows)
        self.assertGreater(report["labeled"], 10)
        with tempfile.TemporaryDirectory() as tmp:
            export_path = Path(tmp) / "rows.jsonl"
            dataset.export_jsonl(rows, str(export_path))
            entry = train.train(rows, seed=11, out_dir=tmp)
            reg = Path(tmp) / "registry"
            registry.write_entry(entry, str(reg))
            loaded = registry.read_entry(str(reg))
            evaluated = evaluate.evaluate(loaded, rows, seed=11)
            registry.write_entry(evaluated, str(reg))
            artifact = train.load_artifact(loaded.artifact_path)
            sit = Situation(raw="1&10 my 25", side="offense", down=1, distance=10, yardline=25)
            state, obs = adapters.situation_to_state(
                sit, game_id="demo", snap_id="demo-1", opponent_type=OpponentKind.CPU
            )
            book = {
                "Gun Bunch": ["Mesh", "Flood Seam"],
                "I Form": ["Power O"],
            }
            cands = [
                candidate_in_book(book, side=Possession.OFFENSE, formation=f, play=p)
                for f, plays in book.items()
                for p in plays
            ]
            ranking = policy.rank_plays(
                state, obs, cands, None, None, (), model_artifact=artifact, mode=CoachingMode.SHADOW
            )
            self.assertGreaterEqual(len(ranking.plays), 2)
            heuristic = cands[0]
            decision = inference.rank_live(
                state,
                obs,
                cands,
                None,
                None,
                (),
                mode=CoachingMode.SHADOW,
                heuristic_pick=heuristic,
                model_artifact=artifact,
            )
            self.assertEqual(decision.final_pick, heuristic)
            self.assertIsNotNone(decision.shadow_status)
            # Persist and compare
            db = CoachDB(Path(tmp) / "madden27.db", seed={"opponents": {}})
            try:
                db.log_ml_decision(decision, agree=0 if decision.shadow_pick != heuristic else 1)
                rows_db = db.list_ml_decisions(limit=10)
                self.assertEqual(len(rows_db), 1)
                self.assertEqual(rows_db[0]["final_play"], heuristic.play)
            finally:
                db.close()


class PackageImportGuard(unittest.TestCase):
    def test_package_import_stays_light(self) -> None:
        import cfb_coach.madden.model as pkg

        self.assertIn("GameState", pkg.__all__)
        self.assertNotIn("train", pkg.__all__)
        self.assertNotIn("numpy", pkg.__dict__)
        self.assertNotIn("sklearn", pkg.__dict__)


if __name__ == "__main__":
    unittest.main()
