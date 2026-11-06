"""Research-only attestation weighting: synthetic rows, no private database."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cfb_coach.madden.model import dataset, experimental_model

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "attestation_weight_eval", ROOT / "scripts" / "madden_ml" / "attestation_weight_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(play: str, result: str, snap: str) -> dict:
    return dataset._normalize_row({
        "session_id": "synthetic-game", "snap_id": snap, "side": "offense", "down": 1, "distance": 10,
        "formation": "Gun Bunch", "play": play, "result": result,
        "executed_status": "identified", "executed_verification": "verified",
        "executed_formation": "Gun Bunch", "executed_play": play,
    }, provenance="synthetic.test")


def test_default_training_is_unchanged_by_the_weight_option() -> None:
    rows = [_row("Mesh", "td", "a"), _row("Y Cross", "incomplete", "b")]
    plain = experimental_model.train_experimental(rows).to_dict()
    unit = experimental_model.train_experimental(rows, row_weight=lambda r: 1.0).to_dict()
    for key in ("play_concept", "down_distance", "global_rate", "n_supervised"):
        assert plain[key] == unit[key]


def test_row_weight_scales_and_zero_weight_drops_a_play() -> None:
    rows = [_row("Mesh", "td", "a"), _row("Y Cross", "incomplete", "b")]
    half = experimental_model.train_experimental(
        rows, row_weight=lambda r: 0.5 if r["snap_id"] == "a" else 1.0)
    assert half.play_concept["mesh"] == pytest.approx([0.5, 0.5])
    dropped = experimental_model.train_experimental(
        rows, row_weight=lambda r: 0.0 if r["snap_id"] == "a" else 1.0)
    assert "mesh" not in dropped.play_concept
    assert "cross" in dropped.play_concept


def test_summary_reports_each_policy_against_the_reference_and_by_stratum() -> None:
    mod = _load_script()
    preds = [
        {"game": g, "y": y, "p": {"w0": 0.5, "w100": 0.8 if y else 0.2},
         "confidence": "high" if i % 2 else "medium", "source": "retrospective_statement"}
        for i, (g, y) in enumerate([("g1", 1), ("g1", 0), ("g2", 1), ("g2", 0), ("g3", 1), ("g3", 0)])
    ]
    out = mod.summarize(preds, "w0", n_boot=50, seed=1)
    assert set(out["all"]["models"]) == {"w0", "w100"}
    assert out["all"]["vs_reference"]["w100"]["mean_diff"] < 0
    assert {"confidence=high", "confidence=medium", "source=retrospective_statement"} <= set(out["strata"])


def test_refuses_to_write_inside_the_live_data_directory(tmp_path, monkeypatch) -> None:
    mod = _load_script()
    monkeypatch.setattr("cfb_coach.games.data_dir", lambda: tmp_path)
    with pytest.raises(SystemExit, match="live data directory"):
        mod.main(["--db", str(tmp_path / "x.db"), "--defense-template", "t.json",
                  "--defense-current", "c.json", "--out", str(tmp_path / "research")])
