"""Test the NEW 650-ish row training path, without touching any owner files."""
from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from scripts import train_lions_legacy_offense as lt
from scripts import restore_lions_baseline as rb


def fixture_source(tmp_path: Path) -> Path:
    db = tmp_path / "source.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        c.execute("CREATE TABLE snaps (id INTEGER PRIMARY KEY)")
        c.execute("CREATE TABLE game_sessions (id INTEGER PRIMARY KEY)")
        c.execute("CREATE TABLE ml_decisions (id INTEGER PRIMARY KEY)")
        c.execute("CREATE TABLE ml_outcomes (id INTEGER PRIMARY KEY)")
        c.execute("INSERT INTO snaps VALUES (1)")
        c.execute("INSERT INTO meta VALUES (?, ?)", ("ml_mode", "heuristic"))
        c.execute("INSERT INTO meta VALUES (?, ?)", (rb.MODE_DEFENSE, "experimental"))
        c.execute("INSERT INTO meta VALUES (?, ?)", ("ml_experimental_artifact", str(tmp_path / "newer-655.json")))
    (tmp_path / "newer-655.json").write_text(json.dumps({"n_supervised": 655}), encoding="utf-8")
    return db


def args(src: Path, dst: Path, minimum: int = 40) -> Namespace:
    return Namespace(source_db=src, workdir=dst, min_supervised=minimum,
                     check=False, apply=False)


def fake_rows(count: int) -> list[dict]:
    # Unit-test-only synthetic labels (not production training examples).
    return [{
        "side": "offense", "supervised_eligible": True,
        "eligibility": "verified_execution", "label_available": True,
        "success": "true" if i % 3 else "false",
        "action_play": ["Mesh", "Flood", "HB Stretch"][i % 3],
        "executed_play": ["Mesh", "Flood", "HB Stretch"][i % 3],
        "down": i % 3 + 1, "distance": 3 + i % 11,
        "yardline": 20 + i % 55, "opponent_type": "cpu",
        "opponent_id": "cpu", "coverage_seen": "cover 3 sky",
        "result": "gain 7" if i % 3 else "incomplete",
    } for i in range(count)]


def test_retrain_exact_historical_algorithm_on_new_examples(monkeypatch):
    from cfb_coach.madden.model import experimental_model as old_ml
    monkeypatch.setattr(old_ml, "_load_research_boosts",
                        lambda: ({}, {}, "offline", "test"))
    artifact = lt._train(fake_rows(55), min_supervised=40)
    assert artifact.model_version == "madden-ml.experimental.offense.55.empirical"
    assert artifact.kind == "hierarchical_shrinkage.v1"
    assert artifact.n_supervised == 55
    assert artifact.evidence_quality == "empirical"


def test_training_and_preview_only_use_database_copy(tmp_path, monkeypatch):
    from cfb_coach.madden.model import experimental_model as old_ml
    src = fixture_source(tmp_path)
    workdir = tmp_path / "fresh-lions"
    original_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    model_655 = tmp_path / "newer-655.json"
    existing_sha = hashlib.sha256(model_655.read_bytes()).hexdigest()
    rows = fake_rows(53)
    monkeypatch.setattr(lt, "_rows", lambda path: (rows, 53, 0))
    monkeypatch.setattr(old_ml, "_load_research_boosts",
                        lambda: ({}, {}, "offline", "test"))

    cfg = args(src, workdir)
    assert lt.preview(cfg) == 0
    assert not workdir.exists()
    assert lt.train(cfg) == 0

    output_db, model_path, report_path = lt._destination_paths(workdir)
    assert output_db.exists() and model_path.exists() and report_path.exists()
    assert hashlib.sha256(src.read_bytes()).hexdigest() == original_sha
    assert hashlib.sha256(model_655.read_bytes()).hexdigest() == existing_sha
    with sqlite3.connect(src) as c:
        assert rb._get_meta(c, rb.MODE_DEFENSE) == "experimental"
    with sqlite3.connect(output_db) as c:
        st = rb._read_state(c)
        assert rb._counts(st["applied"]["offense"]["formations"]) == (5, 93)
        assert rb._counts(st["applied"]["defense"]["formations"]) == (6, 123)
        assert rb._get_meta(c, rb.MODE_DEFENSE) == "off"
        assert rb._get_meta(c, "ml_mode") == "experimental"
        assert rb._get_meta(c, "ml_experimental_artifact") == str(model_path)
        assert c.execute("SELECT count(*) FROM snaps").fetchone()[0] == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["trained_supervised_rows"] == 53
    assert report["original_model_recreated"] is False
    assert report["model_algorithm"] == "hierarchical_shrinkage.v1"
    assert report["model_sha256"] == rb.digest(model_path)
    with pytest.raises(rb.RestoreError, match="Refusing to overwrite"):
        lt.train(cfg)


def test_fewer_verified_rows_causes_no_write(tmp_path, monkeypatch):
    src = fixture_source(tmp_path)
    workdir = tmp_path / "lions"
    monkeypatch.setattr(lt, "_rows", lambda path: (fake_rows(3), 3, 0))
    cfg = args(src, workdir)
    assert lt.preview(cfg) == 2
    assert not workdir.exists()
    with pytest.raises(rb.RestoreError, match="Only 3 verified offensive"):
        lt.train(cfg)
    assert not workdir.exists()


def test_protected_folders_and_source_never_targets(tmp_path, monkeypatch):
    src = fixture_source(tmp_path)
    with pytest.raises(rb.RestoreError, match="Output must be separate"):
        lt._protect(src, tmp_path)
    with pytest.raises(rb.RestoreError, match="Output must be separate"):
        lt._protect(src, src)
    with pytest.raises(rb.RestoreError, match="Output must be separate"):
        lt._protect(src, Path.home() / ".cfb-coach" / "retrained")
    with pytest.raises(rb.RestoreError, match="Output must be separate"):
        lt._protect(src, Path.home() / "unified_v1_cpu1" / "retrained")


def test_any_newer_model_version_not_falsely_called_original(monkeypatch):
    from cfb_coach.madden.model import experimental_model as old_ml
    monkeypatch.setattr(old_ml, "_load_research_boosts",
                        lambda: ({}, {}, "offline", "test"))
    trained = lt._train(fake_rows(44), 40)
    assert trained.model_version == "madden-ml.experimental.offense.44.empirical"
    assert trained.model_version != rb.EXPECTED_MODEL_VERSION
