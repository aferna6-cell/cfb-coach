"""Sprint 15A: expert VOD foundation and personalized offensive learning."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cfb_coach.madden.model.expert_film import (
    annotate_expert_snap,
    export_expert_evidence,
    import_expert_recording,
    load_expert_manifest,
)
from cfb_coach.madden.model.expert_policy import (
    prepare_expert_dataset,
    train_expert_outcome_model,
    train_expert_policy,
)
from cfb_coach.madden.model.expert_signal import (
    EXPERT_SIGNAL_CAP,
    evaluation_gates_passed,
    expert_learning_adjustment,
    promote_expert_signal,
    register_shadow_artifacts,
    rollback_expert_signal,
)
from cfb_coach.madden.model.learning_eval import shadow_evaluation
from cfb_coach.madden.model.learning_reports import (
    expert_learning_summary,
    learning_compare_report,
    personal_learning_report,
)
from cfb_coach.madden.model.learning_sources import (
    VOD_PRIOR_INTERACTION,
    evidence_row,
    ingest_general_from_football_knowledge,
    list_evidence,
    personal_rows_from_verified_snaps,
    retain_evidence,
    source_summary,
)
from cfb_coach.madden.model.offense_joint_decision import choose_joint_action
from cfb_coach.madden.model.personalization import (
    chronological_personal_split,
    fit_personalization,
    ingest_personal_from_snaps,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "expert_film"


class _MetaDB:
    def __init__(self):
        self._meta: dict[str, str] = {}

    def get_meta(self, key: str):
        return self._meta.get(key)

    def set_meta(self, key: str, value: str) -> None:
        self._meta[key] = value


def _seed_expert_store(film_store: Path, learning_store: Path) -> None:
    for name in ("pilot_match_a.json", "pilot_match_b.json"):
        payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        match_id = payload["match_id"]
        import_expert_recording(
            FIXTURES / name,
            expert_id=payload["expert_id"],
            match_id=match_id,
            store=film_store,
            dry_run=False,
            skip_decode=True,
            game_version=payload.get("game_version", "madden27"),
            patch=payload.get("patch"),
            competitive_mode=payload.get("competitive_mode", "unknown"),
            opponent_type=payload.get("opponent_type", "unknown"),
            permission_status="fixture_synthetic",
        )
        ann_dir = film_store / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        (ann_dir / f"{match_id}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
    export_expert_evidence(film_store, learning_store)


class SourceSeparationTests(unittest.TestCase):
    def test_three_categories_retained_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "learning"
            ingest_general_from_football_knowledge(store)
            retain_evidence(store, evidence_row(
                category="expert_evidence",
                evidence_id="expert:x:1",
                match_id="m1",
                situation={"down": 1, "distance": 10},
                observed_action={"play": "Mesh", "concept_family": "mesh", "demonstrated_action": True},
                verified_outcome=None,
                provenance={"expert_id": "e1", "source": "test"},
                confidence=0.8,
                human_verification="verified_human",
                executed=False,
            ))
            snaps = json.loads((FIXTURES / "personal_snaps.json").read_text(encoding="utf-8"))
            ingest_personal_from_snaps(store, snaps)
            summary = source_summary(store)
            self.assertGreater(summary["counts"]["general_football_evidence"], 0)
            self.assertEqual(summary["counts"]["expert_evidence"], 1)
            self.assertGreaterEqual(summary["counts"]["personal_evidence"], 3)
            self.assertFalse(summary["undifferentiated_mix"])
            self.assertIn("vod_model", VOD_PRIOR_INTERACTION)

    def test_personal_excludes_unexecuted_recommendations(self):
        snaps = json.loads((FIXTURES / "personal_snaps.json").read_text(encoding="utf-8"))
        rows = personal_rows_from_verified_snaps(snaps)
        self.assertTrue(all(row["executed"] for row in rows))
        self.assertFalse(any(row["evidence_id"].endswith("rec-only") for row in rows))

    def test_duplicate_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            row = evidence_row(
                category="expert_evidence",
                evidence_id="expert:dup:1",
                match_id="m1",
                situation={"down": 1, "distance": 10},
                observed_action={"play": "Mesh", "concept_family": "mesh"},
                verified_outcome=None,
                provenance={"expert_id": "e1"},
                confidence=0.7,
                human_verification="verified_human",
                executed=False,
            )
            first = retain_evidence(store, row)
            second = retain_evidence(store, row)
            self.assertTrue(first["ok"])
            self.assertFalse(second["ok"])
            self.assertEqual(second["error"], "duplicate_evidence")
            self.assertFalse(second["independent_evidence"])


class ExpertFilmTests(unittest.TestCase):
    def test_import_refuses_denied_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = import_expert_recording(
                FIXTURES / "pilot_match_a.json",
                expert_id="x",
                match_id="denied-1",
                store=tmp,
                dry_run=False,
                skip_decode=True,
                permission_status="denied",
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "permission_denied")

    def test_fixture_import_writes_manifest_without_claiming_auto_understanding(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            result = import_expert_recording(
                FIXTURES / "pilot_match_a.json",
                expert_id="fixture-elite-1",
                match_id="expert-pilot-a",
                store=store,
                dry_run=False,
                skip_decode=True,
                permission_status="fixture_synthetic",
                competitive_mode="ultimate_team",
                opponent_type="human",
            )
            self.assertTrue(result["ok"])
            self.assertFalse(result["downloaded"])
            self.assertFalse(result["raw_video_automatically_understood"])
            manifest = load_expert_manifest(store, "expert-pilot-a")
            self.assertEqual(manifest["expert_id"], "fixture-elite-1")
            self.assertEqual(manifest["permission_status"], "fixture_synthetic")

    def test_invisible_fields_remain_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            import_expert_recording(
                FIXTURES / "pilot_match_a.json",
                expert_id="fixture-elite-1",
                match_id="expert-pilot-a",
                store=store,
                dry_run=False,
                skip_decode=True,
                permission_status="fixture_synthetic",
            )
            from cfb_coach.madden.model.expert_film import empty_expert_annotations
            from cfb_coach.madden.model.film_review import save_annotations

            save_annotations(store, empty_expert_annotations(
                match_id="expert-pilot-a",
                recording_id="fx",
                segments=[{
                    "candidate_id": "manual-0",
                    "start_s": 0,
                    "end_s": 2,
                    "confidence": 0.5,
                    "boundary_status": "corrected",
                    "source": "manual",
                }],
                expert_id="fixture-elite-1",
            ))
            result = annotate_expert_snap(
                store,
                "expert-pilot-a",
                "manual-0",
                {
                    "formation": "Gun Bunch",
                    "play": "unknown",
                    "invisible_fields": ["play", "offensive_adjustments"],
                    "down": 1,
                    "distance": 10,
                    "human_verification_status": "verified_human",
                    "observation_confidence": 0.6,
                    "concept_family": "mesh",
                    "outcome_visible": False,
                },
            )
            self.assertTrue(result["ok"])
            payload = json.loads(
                (store / "annotations" / "expert-pilot-a.json").read_text(encoding="utf-8")
            )
            row = next(c for c in payload["candidates"] if c["candidate_id"] == "manual-0")
            self.assertIsNone(row["play"])
            self.assertIn("play", row["invisible_fields"])


class ExpertPolicyAndPersonalizationTests(unittest.TestCase):
    def test_train_policy_and_personalization_shadow_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            film = root / "film"
            learning = root / "learning"
            _seed_expert_store(film, learning)
            snaps = json.loads((FIXTURES / "personal_snaps.json").read_text(encoding="utf-8"))
            ingest_personal_from_snaps(learning, snaps)
            dataset = prepare_expert_dataset(learning)
            self.assertGreaterEqual(dataset["n_examples"], 4)
            self.assertGreaterEqual(dataset["n_experts"], 2)
            policy = train_expert_policy(learning)
            outcome = train_expert_outcome_model(learning)
            personal = fit_personalization(learning, expert_policy=policy)
            self.assertEqual(policy["mode"], "shadow")
            self.assertTrue(policy["demonstrated_action_only"])
            self.assertFalse(policy["unchosen_failure_claim"])
            self.assertEqual(personal["mode"], "shadow")
            self.assertFalse(personal["fixed_game_threshold"])
            self.assertFalse(personal["fixed_percentage_schedule"])
            # Outcome model may publish zero cells with tiny fixture n; that is ok.
            self.assertIn("verified_outcome_examples", outcome)
            report = shadow_evaluation(str(learning))
            self.assertTrue(report["ok"])
            self.assertEqual(report["mode"], "shadow")
            self.assertFalse(report["gate"]["counterfactual_success_claimed"])
            self.assertEqual(len(report["comparisons"]), 3)

    def test_chronological_personal_split(self):
        snaps = json.loads((FIXTURES / "personal_snaps.json").read_text(encoding="utf-8"))
        verified = [s for s in snaps if s.get("executed_verification") == "verified"]
        views = chronological_personal_split(verified)
        self.assertEqual(views[0]["eligible_n"], 0)
        self.assertEqual(views[1]["eligible_n"], 1)
        self.assertLess(views[-1]["eligible_n"], len(verified))


class ExpertSignalJointDecisionTests(unittest.TestCase):
    def test_shadow_signal_does_not_change_live_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            learning = Path(tmp) / "learning"
            film = Path(tmp) / "film"
            _seed_expert_store(film, learning)
            policy = train_expert_policy(learning)
            personal = fit_personalization(learning, expert_policy=policy)
            db = _MetaDB()
            register_shadow_artifacts(db, expert_policy=policy, personalization=personal)
            sit = SimpleNamespace(down=1, distance=10, coverage_hint=None, coverage_source="none")
            adj = expert_learning_adjustment(
                "Mesh", sit, db=db, expert_policy=policy, personalization=personal,
                opponent_type="cpu",
                book={"Gun Bunch": ["Mesh", "Flood"]},
                formation="Gun Bunch",
            )
            self.assertEqual(adj["mode"], "shadow")
            self.assertEqual(adj["delta"], 0.0)
            self.assertNotEqual(adj["shadow_delta"], None)
            self.assertLessEqual(abs(adj["shadow_delta"]), EXPERT_SIGNAL_CAP + 1e-9)
            self.assertTrue(adj["model_primary_retained"])

    def test_signal_withholds_play_outside_installed_book(self):
        db = _MetaDB()
        register_shadow_artifacts(db, expert_policy={"path": None}, personalization=None)
        sit = SimpleNamespace(down=1, distance=10)
        adj = expert_learning_adjustment(
            "Secret Cheese", sit, db=db,
            book={"Gun Bunch": ["Mesh"]},
            formation="Gun Bunch",
        )
        self.assertEqual(adj["delta"], 0.0)
        self.assertIn("unverified_or_missing_playbook_entry", adj["withheld"])

    def test_promote_requires_gates_and_rollback_works(self):
        db = _MetaDB()
        register_shadow_artifacts(db, expert_policy={"path": "p", "fingerprint": "f"}, personalization=None)
        blocked = promote_expert_signal(db, evaluation={"match_holdout": False})
        self.assertFalse(blocked["ok"])
        ok, failures = evaluation_gates_passed({
            "match_holdout": True,
            "expert_holdout": True,
            "insufficient_data": False,
            "coordinator_baseline_measured": True,
            "personalized_outcome_eval_measured": True,
            "real_expert_footage_verified": True,
            "expert_minus_baseline": 0.1,
            "personalized_minus_expert": 0.0,
        })
        self.assertTrue(ok)
        self.assertEqual(failures, [])
        promoted = promote_expert_signal(db, evaluation={
            "match_holdout": True,
            "expert_holdout": True,
            "insufficient_data": False,
            "coordinator_baseline_measured": True,
            "personalized_outcome_eval_measured": True,
            "real_expert_footage_verified": True,
            "expert_minus_baseline": 0.05,
            "personalized_minus_expert": 0.01,
        })
        self.assertTrue(promoted["ok"])
        self.assertEqual(promoted["mode"], "bounded_active")
        rolled = rollback_expert_signal(db)
        self.assertEqual(rolled["mode"], "shadow")

    def test_joint_decision_keeps_model_primary_with_expert_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            learning = Path(tmp) / "learning"
            film = Path(tmp) / "film"
            _seed_expert_store(film, learning)
            policy = train_expert_policy(learning)
            personal = fit_personalization(learning, expert_policy=policy)
            db = _MetaDB()
            register_shadow_artifacts(db, expert_policy=policy, personalization=personal)
            sit = SimpleNamespace(
                down=1, distance=10, yards_to_goal=70, clock_seconds=300,
                quarter=1, coverage_hint=None, coverage_source="none",
                score_diff=0, hash_mark=None,
            )
            book = {"Gun Bunch": ["Mesh", "Flood"], "Gun Trips": ["Flood", "Mesh"]}
            ranked = [
                {"formation": "Gun Bunch", "play": "Mesh", "selection_score": 0.62, "probability": 0.62},
                {"formation": "Gun Trips", "play": "Flood", "selection_score": 0.55, "probability": 0.55},
            ]
            decision = choose_joint_action(
                ranked=ranked,
                anchor=ranked[0],
                sit=sit,
                book=book,
                db=db,
                opponent_type="cpu",
                use_knowledge=False,
                use_strategy=False,
                use_opponent_learning=False,
                use_expert_learning=True,
                use_diversity=False,
            )
            self.assertEqual(decision["expert_learning"]["mode"], "shadow")
            self.assertFalse(decision["expert_learning"]["live_influence"])
            self.assertFalse(decision["expert_learning"]["vod_prior_on_joint_path"])
            self.assertTrue(decision["expert_learning"]["model_primary_retained"])
            self.assertEqual(decision["joint"]["formation"], "Gun Bunch")
            self.assertEqual(decision["joint"]["play"], "Mesh")


class ReportCliContractTests(unittest.TestCase):
    def test_reports_and_cli_help(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            film = root / "film"
            learning = root / "learning"
            _seed_expert_store(film, learning)
            snaps = json.loads((FIXTURES / "personal_snaps.json").read_text(encoding="utf-8"))
            ingest_personal_from_snaps(learning, snaps)
            train_expert_policy(learning)
            fit_personalization(learning)
            expert = expert_learning_summary(
                learning_store=str(learning), film_store=str(film),
            )
            personal = personal_learning_report(
                learning_store=str(learning), opponent="cpu",
            )
            compare = learning_compare_report(
                learning_store=str(learning), concept="mesh",
            )
            self.assertEqual(expert["report"], "expert-learning")
            self.assertGreaterEqual(expert["reviewed_snaps"], 4)
            self.assertEqual(personal["report"], "personal-learning")
            self.assertGreaterEqual(personal["personal_snaps"], 3)
            self.assertEqual(compare["report"], "learning-compare")
            self.assertEqual(compare["concept"], "mesh")
            self.assertFalse(compare["expert_signal_changes_decisions"])

        from contextlib import redirect_stdout
        import io
        from cfb_coach.cli import main

        buf = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(buf):
            main(["ml", "--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = buf.getvalue()
        for name in (
            "expert-learning",
            "personal-learning",
            "learning-compare",
            "expert-film-import",
            "train-expert-policy",
            "learning-eval",
        ):
            self.assertIn(name, help_text)


class EliteVodValidationTests(unittest.TestCase):
    def test_eval_does_not_replace_active_model_artifacts(self):
        from cfb_coach.madden.model.learning_eval import shadow_evaluation
        from cfb_coach.madden.model.expert_policy import train_expert_policy

        with tempfile.TemporaryDirectory() as tmp:
            film = Path(tmp) / "film"
            learn = Path(tmp) / "learning"
            _seed_expert_store(film, learn)
            ingest_personal_from_snaps(
                learn, json.loads((FIXTURES / "personal_snaps.json").read_text())
            )
            policy = train_expert_policy(learn)
            personal = fit_personalization(learn, expert_policy=policy)
            fingerprints = {
                policy["path"]: Path(policy["path"]).read_bytes(),
                personal["path"]: Path(personal["path"]).read_bytes(),
            }
            report = shadow_evaluation(str(learn))
            self.assertTrue(report["ok"])
            self.assertTrue(report["gate"]["insufficient_data"])
            self.assertFalse(report["gate"]["coordinator_baseline_measured"])
            self.assertFalse(report["personal_eval"]["future_personal_rows_used"])
            for path, before in fingerprints.items():
                self.assertEqual(Path(path).read_bytes(), before)

    def test_no_train_test_overlap_when_experts_share_match(self):
        from cfb_coach.madden.model.learning_eval import _match_expert_holdout

        mixed = [
            {"evidence_id": "a", "expert_id": "pro1", "match_id": "same"},
            {"evidence_id": "b", "expert_id": "pro2", "match_id": "same"},
        ]
        train, test, split = _match_expert_holdout(mixed)
        self.assertEqual(train, [])
        self.assertEqual(test, [])
        self.assertFalse(split["disjoint"])

    def test_real_footage_and_measured_baseline_are_required(self):
        ok, failures = evaluation_gates_passed({
            "match_holdout": True, "expert_holdout": True,
            "insufficient_data": False,
            "expert_minus_baseline": 0.2,
            "personalized_minus_expert": 0.05,
        })
        self.assertFalse(ok)
        self.assertIn("coordinator_baseline_required", failures)
        self.assertIn("real_reviewed_expert_footage_required", failures)

    def test_only_verified_permitted_expert_footage_is_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            film = Path(tmp) / "film"
            learn = Path(tmp) / "learn"
            _seed_expert_store(film, learn)
            # Revoke processing status in the manifest for a previously
            # reviewed synthetic fixture. No new training rows may be exported.
            manifest = film / "expert_manifests" / "expert-pilot-a.json"
            payload = json.loads(manifest.read_text())
            payload["permission_status"] = "denied"
            manifest.write_text(json.dumps(payload))
            count = len(list_evidence(learn, category="expert_evidence"))
            result = export_expert_evidence(film, learn, match_id="expert-pilot-a")
            self.assertEqual(result["exported"], 0)
            self.assertGreater(result["excluded_unverified_or_unpermitted"], 0)
            self.assertEqual(len(list_evidence(learn, category="expert_evidence")), count)


if __name__ == "__main__":
    unittest.main()
