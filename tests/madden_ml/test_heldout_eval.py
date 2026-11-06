"""Held-out evaluator: metrics, leakage-free folds, read-only data access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.madden.data import load_seed
from cfb_coach.madden.model import cli as ml_cli
from cfb_coach.madden.model import heldout_eval as he


def test_binary_metrics_known_values() -> None:
    m = he.binary_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.3])
    assert m["n"] == 4
    assert m["auc"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx((0.01 + 0.01 + 0.04 + 0.09) / 4)
    assert he.auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert he.auc([1, 1], [0.2, 0.9]) is None


def test_wilson_interval_brackets_rate() -> None:
    w = he.wilson(5, 10)
    assert w["lo"] < 0.5 < w["hi"]
    assert he.wilson(0, 0)["rate"] is None


def _row(game: str, opp: str, *, verified: bool = True, result: str = "+6", play: str = "Mesh") -> dict[str, Any]:
    raw = {
        "session_id": game,
        "ml_snap_id": f"{game}-{result}-{play}-{verified}",
        "opponent_id": opp,
        "side": "offense",
        "down": 1,
        "distance": 10,
        "yardline": 30,
        "result": result,
        "formation": "Gun Bunch",
        "play": play,
        "executed_status": "identified" if verified else "unknown",
        "executed_verification": "verified" if verified else "unknown",
        "executed_formation": "Gun Bunch" if verified else None,
        "executed_play": play if verified else None,
    }
    return he.dataset_mod._normalize_row(raw, provenance="test")


class _Spy:
    name = "spy"
    targets = ("success",)

    def __init__(self, log: list[set[str]]):
        self.log = log

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        self.log.append({str(r["game_id"]) for r in train_rows})

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        return {"success": 0.5}


def test_chronological_folds_only_train_on_earlier_games() -> None:
    games = ["g0", "g1", "g2", "g3", "g4"]
    rows = [_row(g, "cpu") for g in games for _ in range(3)]
    order = {g: i for i, g in enumerate(games)}
    folds = he.make_folds(rows, order, is_test_row=he.is_verified_offense, min_train_games=2)
    assert [f.test_game for f in folds] == ["g2", "g3", "g4"]
    seen: list[set[str]] = []
    preds, _ = he.run_folds(rows, folds, lambda: [_Spy(seen)], targets=("success",),
                            label_fn=he.offense_label, is_test_row=he.is_verified_offense)
    for fold, trained_on in zip(folds, seen):
        assert fold.test_game not in trained_on
        assert all(order[g] < order[fold.test_game] for g in trained_on)
    assert {p["game"] for p in preds} == {"g2", "g3", "g4"}


def test_exclude_opponent_drops_every_game_against_the_test_opponent() -> None:
    plan = [("g0", "ann"), ("g1", "bob"), ("g2", "cat"), ("g3", "ann"), ("g4", "dan")]
    rows = [_row(g, o) for g, o in plan]
    order = {g: i for i, (g, _) in enumerate(plan)}
    folds = he.make_folds(rows, order, is_test_row=he.is_verified_offense, mode="exclude_opponent", min_train_games=2)
    by_test = {f.test_game: f.train_games for f in folds}
    assert "g0" not in by_test["g3"]
    assert by_test["g3"] == ["g1", "g2"]


def test_recommendation_only_rows_are_never_scored_but_train_context() -> None:
    games = ["g0", "g1", "g2"]
    rows = [_row(g, "cpu") for g in games] + [_row("g2", "cpu", verified=False, result="td", play="Y Cross")]
    order = {g: i for i, g in enumerate(games)}
    folds = he.make_folds(rows, order, is_test_row=he.is_verified_offense, min_train_games=2)
    preds, _ = he.run_folds(rows, folds, he.default_offense_models, targets=he.OFFENSE_TARGETS,
                            label_fn=he.offense_label, is_test_row=he.is_verified_offense)
    success = [p for p in preds if p["target"] == "success"]
    assert len(success) == 1
    assert set(success[0]["p"]) == {"base_rate", "situation_table", "situation_play_family", "experimental_shrinkage"}


def test_explosive_and_turnover_labels() -> None:
    assert he.explosive_label(12, None, "HB Stretch") == 1
    assert he.explosive_label(12, None, "PA Boot Over") == 0
    assert he.run_or_pass("Stretch") == "run"
    assert he.run_or_pass("PA Boot Over") == "pass"
    assert he.run_or_pass("Hail Mary Prayer") == "unknown"
    assert he.explosive_label(None, "td", "Mesh") is None
    assert he.explosive_label(30, "int", "Mesh") == 0
    assert he.offense_label(_row("g", "cpu", result="int"), "turnover") == 1
    assert he.offense_label(_row("g", "cpu"), "conversion") is None


def _insert(db: CoachDB, game: str, i: int, *, side: str, result: str, opponent: str, verified: bool,
            concept_seen: str | None = None) -> None:
    sid = f"{game}-{i:04d}"
    ts = f"2026-10-{1 + int(game[1:]):02d}T00:{i:02d}:00+00:00"
    play = "Mesh" if i % 2 else "HB Zone"
    db.conn.execute(
        "INSERT INTO snaps (ts, opponent_id, side, down, distance, yardline, result, session_id, ml_snap_id, "
        "formation, play, concept_seen, executed_status, executed_formation, executed_play, executed_verification) "
        "VALUES (?, ?, ?, 1, 10, 30, ?, ?, ?, 'Gun Bunch', ?, ?, ?, ?, ?, ?)",
        (ts, opponent, side, result, game, sid, play, concept_seen,
         "identified" if verified else "unknown", "Gun Bunch" if verified else None,
         play if verified else None, "verified" if verified else "unknown"),
    )
    db.conn.execute(
        "INSERT INTO ml_outcomes (snap_id, game_id, executed_status, executed_formation, executed_play, "
        "executed_verification, outcome_json, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, game, "identified" if verified else "unknown", "Gun Bunch" if verified else None,
         play if verified else None, "verified" if verified else "unknown", json.dumps({"result": result}), ts),
    )


@pytest.fixture()
def synthetic_db(tmp_path: Path) -> Path:
    path = tmp_path / "madden.db"
    db = CoachDB(path, seed=load_seed())
    results = ["+6", "+2", "incomplete", "+15", "+4", "sack", "td", "+1"]
    for g in range(5):
        game = f"g{g}"
        for i, res in enumerate(results):
            _insert(db, game, i, side="offense", result=res, opponent="cpu", verified=True)
        _insert(db, game, 20, side="defense", result="+8", opponent="cpu", verified=False, concept_seen="Inside Zone")
        _insert(db, game, 21, side="defense", result="incomplete", opponent="cpu", verified=False, concept_seen="Mesh")
    db.conn.commit()
    db.close()
    return path


def test_evaluate_end_to_end_is_read_only(synthetic_db: Path, tmp_path: Path) -> None:
    before = synthetic_db.read_bytes()
    data = he.load_eval_data(synthetic_db)
    assert list(data.game_order) == ["g0", "g1", "g2", "g3", "g4"]
    report = he.evaluate(data, n_boot=50)
    assert synthetic_db.read_bytes() == before
    assert report["data"]["verified_offense_rows"] == 40
    chron = report["offense"]["chronological"]
    assert chron["folds"] == 2
    assert chron["metrics"]["success"]["models"]["experimental_shrinkage"]["n"] == 16
    # Every synthetic game is against the same opponent, so no other-opponent prior exists.
    assert report["opponent_run_pass"]["chronological"]["folds"] == []
    assert report["defense_outcomes"]["all"]["plays"] == 10
    jp, mp = he.write_report(report, tmp_path / "out")
    assert json.loads(jp.read_text())["eval_version"] == he.EVAL_VERSION
    assert "held-out" in mp.read_text()


def test_cli_refuses_output_inside_live_data_dir(synthetic_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ml_cli, "data_dir", lambda: tmp_path / "live")
    args = argparse.Namespace(db_path=str(synthetic_db), out=str(tmp_path / "live" / "eval"),
                              min_train_games=3, bootstrap=20, seed=1)
    assert ml_cli.cmd_ml_heldout_eval(args) == 2
    assert not (tmp_path / "live" / "eval").exists()
    args.out = str(tmp_path / "elsewhere")
    assert ml_cli.cmd_ml_heldout_eval(args) == 0
    assert (tmp_path / "elsewhere" / "heldout_eval.json").exists()
