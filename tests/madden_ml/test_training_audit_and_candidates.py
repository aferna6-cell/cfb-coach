"""Read-only training audit on a synthetic database, and candidate-only experimental training."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.madden.data import load_seed
from cfb_coach.madden.model import cli as ml_cli

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "audit_training_eligibility",
    _ROOT / "scripts" / "madden_ml" / "audit_training_eligibility.py",
)
audit = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(audit)

_TS = "2026-11-01T00:00:00+00:00"


def _snap(db: CoachDB, *, sid: str, side: str, result: str, verified: bool,
          play: str = "Mesh", attestation: str | None = None) -> None:
    ml_snap_id = f"g1-{sid}"
    status = "identified" if verified else "unknown"
    verification = "verified" if verified else "unknown"
    db.conn.execute(
        "INSERT INTO snaps (ts, opponent_id, side, down, distance, result, session_id, ml_snap_id, "
        "formation, play, executed_status, executed_formation, executed_play, executed_verification) "
        "VALUES (?, 'dillon', ?, 1, 10, ?, 'g1', ?, 'Gun Bunch', ?, ?, ?, ?, ?)",
        (_TS, side, result, ml_snap_id, play, status,
         "Gun Bunch" if verified else None, play if verified else None, verification),
    )
    payload: dict = {"result": result}
    if attestation:
        payload["execution_attestation_source"] = attestation
    db.conn.execute(
        "INSERT INTO ml_outcomes (snap_id, game_id, executed_status, executed_formation, "
        "executed_play, executed_verification, outcome_json, created_ts) VALUES (?, 'g1', ?, ?, ?, ?, ?, ?)",
        (ml_snap_id, status, "Gun Bunch" if verified else None, play if verified else None,
         verification, json.dumps(payload), _TS),
    )


@pytest.fixture()
def synthetic_db(tmp_path: Path) -> Path:
    path = tmp_path / "madden.db"
    db = CoachDB(path, seed=load_seed())
    _snap(db, sid="01", side="offense", result="td", verified=True)
    _snap(db, sid="02", side="offense", result="+3", verified=True,
          attestation="standing_user_attestation")
    _snap(db, sid="03", side="offense", result="td", verified=False, play="Y Cross")
    _snap(db, sid="04", side="defense", result="incomplete", verified=False, play="Cover 3")
    db.conn.commit()
    db.close()
    return path


def test_audit_counts_verified_tds_and_separates_attestation(synthetic_db: Path) -> None:
    before = synthetic_db.read_bytes()
    report = audit.summarize(synthetic_db)
    assert synthetic_db.read_bytes() == before
    assert report["logical_snap_count"] == 4
    assert report["duplicate_snap_ids"] == 0
    offense = report["by_side"]["offense"]
    assert report["trainable_verified_offense"] == 2
    assert offense["verified_touchdowns"] == 1
    assert offense["known_result_execution_uncertain"] == 1
    assert offense["verified_by_basis"] == {
        "per_snap_or_legacy": 1, "standing_user_attestation": 1,
    }
    assert offense["verified_by_opponent_type"] == {"human": 2}
    assert report["trainable_verified_defense"] == 0
    assert report["by_side"]["defense"]["known_result_execution_uncertain"] == 1


def _train_args(**overrides) -> argparse.Namespace:
    values = dict(seed=7, side="offense", path=[], out=None, install=False, db=False, db_path=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_train_experimental_writes_a_new_candidate_not_the_live_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ml_cli, "data_dir", lambda: tmp_path)
    live = tmp_path / "madden_ml_experimental" / "offense.json"
    assert ml_cli.cmd_ml_train_experimental(_train_args()) == 0
    assert not live.exists()
    candidates = list((tmp_path / "madden_ml_experimental" / "candidates").glob("offense.*.json"))
    assert len(candidates) == 1


def test_train_experimental_refuses_live_fallback_and_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ml_cli, "data_dir", lambda: tmp_path)
    live = tmp_path / "madden_ml_experimental" / "offense.json"
    assert ml_cli.cmd_ml_train_experimental(_train_args(out=str(live))) == 2
    assert not live.exists()
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    assert ml_cli.cmd_ml_train_experimental(_train_args(out=str(existing))) == 2
    assert existing.read_text(encoding="utf-8") == "{}"
