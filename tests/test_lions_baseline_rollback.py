"""Regression checks for the isolated Lions 2026-11-04 rollback.

Tests never open the user's actual ~/.cfb-coach database.
"""
from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import sqlite3

from scripts import restore_lions_baseline as rb


def make_db(tmp_path: Path, *, valid_model: bool = True):
    src = tmp_path / "live-source.db"
    model = tmp_path / "old-artifact.json"
    data = {
        "kind": "hierarchical_shrinkage.v1",
        "model_version": rb.EXPECTED_MODEL_VERSION if valid_model else "wrong-version",
        "side": "offense",
        "n_supervised": rb.EXPECTED_SUPERVISED,
        "n_discounted_priors": 405,
        "evidence_quality": "empirical",
    }
    model.write_text(json.dumps(data), encoding="utf-8")
    with sqlite3.connect(src) as c:
        c.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        c.execute("CREATE TABLE snaps(id INTEGER PRIMARY KEY)")
        c.execute("CREATE TABLE game_sessions(id INTEGER PRIMARY KEY)")
        c.execute("CREATE TABLE ml_decisions(id INTEGER PRIMARY KEY)")
        c.execute("CREATE TABLE ml_outcomes(id INTEGER PRIMARY KEY)")
        c.execute("INSERT INTO snaps(id) VALUES (1)")
        c.execute("INSERT INTO meta VALUES (?, ?)", ("ml_experimental_artifact", str(model)))
        c.execute("INSERT INTO meta VALUES (?, ?)", ("ml_mode", "experimental"))
        c.execute("INSERT INTO meta VALUES (?, ?)", ("ml_defense_coordinator_mode.v1", "experimental"))
        current = {"applied": {"offense": {"formations": {
            "Gun Doubles": ["Inside Zone"]
        }}, "defense": {"formations": {"Nickel Over": ["Cover 3 Sky"]}}},
            "pending": {"offense": {"rev": 7}}}
        c.execute("INSERT INTO meta VALUES (?, ?)", ("active_playbook_json", json.dumps(current)))
    return src, model


def args_for(src: Path, dest: Path, *, apply=True, model=None):
    return Namespace(source_db=src, workdir=dest, historical_db=None,
                     model=model, apply=apply, check=False)


def test_saved_manifest_is_five_formations_93_plays_and_six_defensive_sets():
    manifest = rb.load_manifest()
    assert rb._counts(manifest["sides"]["offense"]["formations"]) == (5, 93)
    assert rb._counts(manifest["sides"]["defense"]["formations"]) == (6, 123)
    assert "Gun Trips X Nasty" in manifest["sides"]["offense"]["formations"]
    assert "Nickel Double Mug" in manifest["sides"]["defense"]["formations"]


def test_restoration_changes_only_new_copy(tmp_path):
    src, model = make_db(tmp_path)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    assert rb.restore(args_for(src, tmp_path / "lions")) == 0
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before
    with sqlite3.connect(src) as c:
        assert rb._get_meta(c, "ml_defense_coordinator_mode.v1") == "experimental"
        assert len(rb._read_state(c)["applied"]["offense"]["formations"]) == 1
        assert rb._get_meta(c, "ml_experimental_artifact") == str(model)
        assert c.execute("SELECT count(*) FROM snaps").fetchone()[0] == 1
    copy_db = tmp_path / "lions" / "madden27-lions.db"
    with sqlite3.connect(copy_db) as c:
        state = rb._read_state(c)
        assert rb._counts(state["applied"]["offense"]["formations"]) == (5, 93)
        assert rb._counts(state["applied"]["defense"]["formations"]) == (6, 123)
        assert state["pending"].get("offense") is None
        assert rb._get_meta(c, "ml_mode") == "experimental"
        assert rb._get_meta(c, "ml_defense_coordinator_mode.v1") == "off"
        assert rb._get_meta(c, "ml_experimental_artifact").startswith(str(tmp_path / "lions"))
        assert c.execute("SELECT count(*) FROM snaps").fetchone()[0] == 1
    assert (tmp_path / "lions" / "RESTORE_REPORT.json").is_file()


def test_default_is_read_only_and_does_not_write(tmp_path):
    src, _ = make_db(tmp_path)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    assert rb.restore(args_for(src, tmp_path / "lions", apply=False)) == 0
    assert not (tmp_path / "lions").exists()
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before


def test_mismatched_artifact_refuses_to_retrain_or_create_clone(tmp_path, monkeypatch):
    # This test must not discover the owner's actual ~/.cfb-coach artifact.
    monkeypatch.setenv("HOME", str(tmp_path))
    src, bad = make_db(tmp_path, valid_model=False)
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as c:
        found, findings = rb.find_model(c, bad)
    assert found is None and findings
    try:
        rb.restore(args_for(src, tmp_path / "lions", model=bad))
    except rb.RestoreError as exc:
        assert "Original 428-row empirical artifact not found" in str(exc)
    else:
        assert False, "Must refuse to install the wrong offense model"
    assert not (tmp_path / "lions").exists()


def test_second_restoration_refuses_to_overwrite_game_progress(tmp_path):
    src, _ = make_db(tmp_path)
    a = args_for(src, tmp_path / "lions")
    rb.restore(a)
    try:
        rb.restore(a)
    except rb.RestoreError as exc:
        assert "Refusing to overwrite" in str(exc)
    else:
        assert False, "Must refuse clobbering existing game DB"


def test_cpu_defense_is_opt_in_and_still_uses_old_defensive_caller():
    from cfb_coach.madden.playcaller import make_call
    from cfb_coach.madden.situation import parse_madden_situation
    plays = {
        "offense": {"Gun 5WR Tight": ["Mesh", "Spacing"]},
        "defense": {"4-3 Over": ["Cover 3 Sky", "Cover 2 Man", "Tampa 2"]},
    }
    # Original default is untouched: CPU games used to force defense to O.
    old_sit = parse_madden_situation("d 3&6 opp 45", default_side="defense")
    old = make_call(old_sit, "cpu", playbook=plays, live_macros=False)
    assert old.side == "offense"
    sit = parse_madden_situation("d 3&6 opp 45", default_side="defense")
    defense = make_call(sit, "cpu", playbook=plays, live_macros=False,
                        coach_cpu_defense=True)
    assert defense.side == "defense"
    assert defense.formation in plays["defense"]
    assert defense.play in plays["defense"][defense.formation]
