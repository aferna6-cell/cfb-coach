"""The CPU-experiment driver: gate-checked enable, status, rollback and report against a temp database (SYNTHETIC; plumbing only)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cfb_coach.madden.model import offense_route_ranker as rr
from cfb_coach.madden.model import inference
from tests.madden_ml.test_defense_knowledge_live import _ctrl, _prime, _world, artifacts, rows  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts" / "madden_ml" / "cpu_experiment.py"


def _run(tmp, db_path, *args, expect=0):
    env = {**os.environ, "HOME": str(tmp), "CFB_COACH_DB": str(tmp / "coach.db"), "CFB_COACH_MADDEN_DB": str(db_path), "PYTHONPATH": str(ROOT)}
    env.pop(rr.ENV_MODE, None)
    r = subprocess.run([sys.executable, str(DRIVER), *args], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == expect, (r.stdout, r.stderr)
    return r.stdout + r.stderr


GOOD_GATES = {g: {"status": "PASS", "detail": "ok", "kind": "safety"} for g in (
    "coverage_feedback_connected", "no_cross_game_or_future_leakage", "no_invalid_formations_plays_or_adjustments", "adjustment_gate",
    "latency_within_150ms", "interface_displays_recommendation", "shadow_replay_clean", "regression_and_integration_tests")}
GOOD_GATES["model_quality_vs_existing_coordinator"] = {"status": "NOT EVALUATED", "detail": "no significant lift", "kind": "quality"}


def _sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _prepared(d, db_path, art, gates=None, **tamper):
    """Fake the output of ONE successful `prepare` (work dir with manifest, report, artifact, copy)."""
    import shutil
    from cfb_coach.madden.model import offense_structural_model as sm
    work = d / "work"
    (work / "readiness").mkdir(parents=True, exist_ok=True)
    pa = work / "readiness" / "offense_route_aware.json"
    shutil.copy2(art, pa)
    copy = work / "experiment_copy.db"
    copy.write_bytes(b"copy")
    digest = sm.artifact_digest(rr.load_artifact(pa))
    rep = {"real_data": True, "source": f"real database copy: {copy}", "artifact": {"valid": True, "digest": digest},
           "readiness": {"verdict": "CONDITIONAL GO", "gates": gates if gates is not None else GOOD_GATES}}
    rp = work / "readiness" / "readiness_report.json"
    rp.write_text(json.dumps(rep))
    sys.path.insert(0, str(ROOT / "scripts" / "madden_ml"))
    import importlib
    drv = importlib.import_module("cpu_experiment")
    m = {"source_db": str(tamper.get("source_db", db_path)), "copy": str(copy), "copy_sha256": _sha(copy), "report": str(rp), "report_sha256": _sha(rp),
         "artifact": str(pa), "artifact_sha256": _sha(pa), "artifact_digest": digest, "art_bank": drv._art_bank()}
    (work / "prepare_manifest.json").write_text(json.dumps(m))
    return work, pa


@pytest.fixture()
def world(tmp_path, monkeypatch, artifacts):
    db = _world(tmp_path, monkeypatch, artifacts, "default")
    db.close()
    d = tmp_path / "default"
    return d, d / "madden27.db", d / "default.json"


def _enable(d, db, pa, work, expect=0):
    return _run(d, db, "enable", "--artifact", str(pa), "--work", str(work), "--i-approve-cpu-experiment", expect=expect)


def test_enable_requires_approval_and_a_manifest(world):
    d, db, art = world
    assert "i-approve-cpu-experiment" in _run(d, db, "enable", "--artifact", str(art), "--work", str(d / "work"), expect=1)
    assert "no prepare manifest" in _enable(d, db, art, d / "work", expect=1)


def test_enable_is_fail_closed_on_missing_failed_stale_or_tampered_inputs(world):
    d, db, art = world
    work, pa = _prepared(d, db, art, gates={k: v for k, v in GOOD_GATES.items() if k != "shadow_replay_clean"})
    assert "required gate missing: shadow_replay_clean" in _enable(d, db, pa, work, expect=1)
    bad = {**GOOD_GATES, "latency_within_150ms": {"status": "NOT EVALUATED", "detail": "x", "kind": "performance"}}
    work, pa = _prepared(d, db, art, gates=bad)
    assert "latency_within_150ms is NOT EVALUATED" in _enable(d, db, pa, work, expect=1)          # not-FAIL is not enough
    work, pa = _prepared(d, db, art, source_db=d / "other.db")
    assert "not" in _enable(d, db, pa, work, expect=1) and "prepare was run against" in _enable(d, db, pa, work, expect=1)
    work, pa = _prepared(d, db, art)
    pa.write_text(pa.read_text() + " ")                                                           # tamper with the artifact after prepare
    assert "changed or is missing since prepare" in _enable(d, db, pa, work, expect=1)
    work, pa = _prepared(d, db, art)
    (work / "readiness" / "readiness_report.json").write_text("{}")
    assert "changed or is missing since prepare" in _enable(d, db, pa, work, expect=1)


def test_worse_model_is_refused_but_not_evaluated_or_nonsignificant_is_allowed(world):
    d, db, art = world
    worse = {**GOOD_GATES, "model_quality_vs_existing_coordinator": {"status": "FAIL", "detail": "WORSE +0.01", "kind": "quality"}}
    work, pa = _prepared(d, db, art, gates=worse)
    assert "explicitly WORSE" in _enable(d, db, pa, work, expect=1)
    work, pa = _prepared(d, db, art)                                                              # NOT EVALUATED / nonsignificant
    assert "ENABLED" in _enable(d, db, pa, work)


def test_enable_status_live_call_report_and_exact_rollback(world, monkeypatch):
    d, db_path, art = world
    from cfb_coach.db import CoachDB
    pre = CoachDB(db_path)
    pre.set_meta(rr.META_MODE, "route_aware_preview")
    pre.set_meta(rr.META_ARTIFACT, str(art))
    pre.set_meta("ml_mode", "shadow")
    pre.close()
    work, pa = _prepared(d, db_path, art)
    out = _enable(d, db_path, pa, work)
    assert "ENABLED" in out
    assert "ENABLED" in _enable(d, db_path, pa, work)                                              # re-enable must not clobber the ORIGINAL record
    st = _run(d, db_path, "status")
    assert "effective in this shell: route_aware" in st and "loads OK" in st and "93/93" in st
    monkeypatch.setenv("HOME", str(d))
    monkeypatch.setenv("CFB_COACH_DB", str(d / "coach.db"))
    db = CoachDB(db_path)
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    info = rr.info_of(c.last_call)
    assert info["applied"] is True and info["n_scored"] == 93 and not info.get("fallback_reason") and info["latency_ms"] < 150
    c.result_and_call(outcome="+6", sit_raw="2&4 my 31", their="Cover 3 Cloud", executed_status="used_recommended")
    db.close()
    _run(d, db_path, "report", "--out", str(d / "rep.json"))
    rep = json.loads((d / "rep.json").read_text())
    assert rep["offense_snaps"] == 1 and rep["exact_defensive_calls_logged"][0][0] == "Cover 3 Cloud" and rep["illegal_recommendations"] == []
    assert rep["model"]["records"] >= 1 and rep["model"]["fallbacks"] == []
    assert "exact previous state" in _run(d, db_path, "rollback")
    after = CoachDB(db_path)
    assert (after.get_meta(rr.META_MODE), after.get_meta(rr.META_ARTIFACT), after.get_meta("ml_mode")) == ("route_aware_preview", str(art), "shadow")
    after.close()


def test_safe_rollback_only_forces_the_ranker_back(world):
    d, db_path, art = world
    work, pa = _prepared(d, db_path, art)
    _enable(d, db_path, pa, work)
    assert "rolled back (safe)" in _run(d, db_path, "rollback", "--safe")
    assert "stored: experimental" in _run(d, db_path, "status")


def test_status_does_not_write_to_the_database(world):
    d, db_path, art = world
    before = db_path.read_bytes()
    _run(d, db_path, "status")
    assert db_path.read_bytes() == before


def test_prepare_fails_closed_and_discards_stale_results(tmp_path):
    from cfb_coach.db import CoachDB
    src = tmp_path / "madden27.db"
    CoachDB(src).close()                                                                           # an empty database: nothing to evaluate
    work = tmp_path / "work"
    (work / "readiness").mkdir(parents=True)
    (work / "readiness" / "readiness_report.json").write_text('{"stale": true}')
    (work / "prepare_manifest.json").write_text("{}")
    env = {**os.environ, "HOME": str(tmp_path / "h"), "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, str(DRIVER), "prepare", "--db", str(src), "--work", str(work)], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode != 0 and ("PREPARE FAILED" in r.stderr or "PREPARE INCOMPLETE" in r.stderr)
    stale = work / "readiness" / "readiness_report.json"
    assert not stale.exists() or "stale" not in stale.read_text()                                  # the old report was discarded, never reused


def test_football_priors_are_opt_in_visible_in_status_and_rolled_back(world):
    d, db_path, art = world
    work, pa = _prepared(d, db_path, art)
    _enable(d, db_path, pa, work)
    assert "football priors in the live score: off" in _run(d, db_path, "status")             # default enable leaves them off
    _run(d, db_path, "rollback")
    work, pa = _prepared(d, db_path, art)
    out = _run(d, db_path, "enable", "--artifact", str(pa), "--work", str(work), "--i-approve-cpu-experiment", "--football-priors")
    assert "ENABLED" in out and "football priors in the live score: ON" in _run(d, db_path, "status")
    _run(d, db_path, "rollback")
    assert "football priors in the live score: off" in _run(d, db_path, "status")
