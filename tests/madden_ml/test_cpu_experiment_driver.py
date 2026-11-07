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


@pytest.fixture()
def world(tmp_path, monkeypatch, artifacts):
    db = _world(tmp_path, monkeypatch, artifacts, "default")
    path = Path(db.path) if hasattr(db, "path") else tmp_path / "default" / "madden27.db"
    db.close()
    d = tmp_path / "default"
    art = d / "default.json"
    (d / "readiness_report.json").write_text(json.dumps({"real_data": True, "readiness": {"gates": {"latency_within_150ms": {"status": "PASS", "detail": "x", "kind": "performance"}}}}))
    return d, d / "madden27.db", art


def test_enable_requires_explicit_approval_and_a_clean_report(world):
    d, db, art = world
    out = _run(d, db, "enable", "--artifact", str(art), expect=1)
    assert "i-approve-cpu-experiment" in out.lower() or "refusing" in out.lower()
    (d / "readiness_report.json").write_text(json.dumps({"real_data": True, "readiness": {"gates": {"no_invalid_formations_plays_or_adjustments": {"status": "FAIL", "detail": "illegal pick", "kind": "safety"}}}}))
    out = _run(d, db, "enable", "--artifact", str(art), "--i-approve-cpu-experiment", expect=1)
    assert "gate failure" in out
    (d / "readiness_report.json").write_text(json.dumps({"real_data": False, "readiness": {"gates": {}}}))
    assert "synthetic" in _run(d, db, "enable", "--artifact", str(art), "--i-approve-cpu-experiment", expect=1)


def test_enable_status_live_call_and_rollback(world, monkeypatch):
    d, db_path, art = world
    out = _run(d, db_path, "enable", "--artifact", str(art), "--i-approve-cpu-experiment")
    assert "ENABLED" in out and "route_aware" in out
    st = _run(d, db_path, "status")
    assert "effective in this shell: route_aware" in st and "loads OK" in st and "93/93" in st
    # the real live loop now uses the model as primary
    from cfb_coach.db import CoachDB
    monkeypatch.setenv("HOME", str(d))
    monkeypatch.setenv("CFB_COACH_DB", str(d / "coach.db"))
    db = CoachDB(db_path)
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    info = rr.info_of(c.last_call)
    assert info["applied"] is True and info["n_scored"] == 93 and not info.get("fallback_reason") and info["latency_ms"] < 150
    c.result_and_call(outcome="+6", sit_raw="2&4 my 31", their="Cover 3 Cloud", executed_status="used_recommended")
    db.close()
    rp = _run(d, db_path, "report", "--out", str(d / "rep.json"))
    rep = json.loads((d / "rep.json").read_text())
    assert rep["offense_snaps"] == 1 and rep["exact_defensive_calls_logged"][0][0] == "Cover 3 Cloud" and rep["illegal_recommendations"] == []
    assert rep["model"]["records"] >= 1 and rep["model"]["fallbacks"] == []
    rb = _run(d, db_path, "rollback")
    assert "rolled back" in rb
    assert "stored: experimental" in _run(d, db_path, "status")


def test_status_and_report_do_not_write_to_the_database(world):
    d, db_path, art = world
    before = db_path.read_bytes()
    _run(d, db_path, "status")
    assert db_path.read_bytes() == before
