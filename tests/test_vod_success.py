"""VOD success model writes the box v0.7 files and keeps CPU and human apart."""

from __future__ import annotations

import json
from pathlib import Path

import math

from cfb_coach.vod_success.features import feature_map, snap_from_row
from cfb_coach.vod_success.train import (
    DEFAULT_HP,
    OUTPUTS,
    SCRIPT_VERSION,
    default_grid,
    train_and_write,
)
from cfb_coach.vod_success.__main__ import main

TINY_GRID = [(2.0, 16.0, 64.0), (4.0, 16.0, 16.0)]


def _prob(model: dict, row: dict) -> float:
    snap = snap_from_row(row)
    logit = model["intercept"]
    for group, level in feature_map(snap).items():
        logit += model["coef"].get(group, {}).get(level, 0.0)
    logit = max(-15.0, min(15.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))

BEATER_CALL_KEYS = {
    "call", "call_key", "call_class", "n", "successes", "raw_success",
    "shrunk_success", "lower_bound_90", "wilson_lb_90_raw", "situation_expected",
    "lift_vs_situation", "n_with_off_adjustment", "n_streamer_offense",
    "n_streamer_defense", "n_vods", "mean_conf_play", "mean_conf_success",
    "tentative", "confidence",
}
COEF_GROUPS = {
    "call", "call_class", "call_x_family", "class_x_family", "def_adj", "dist",
    "down", "down_x_dist", "look", "look_family", "off_adj", "pressure", "score",
    "side", "time", "zone",
}
SITUATION_GROUPS = {"dist", "down", "down_x_dist", "score", "side", "time", "zone"}


def _row(**kwargs):
    row = {
        "game": "madden27",
        "opponent_type": "human",
        "side": "offense",
        "play": "HB Dive",
        "coverage_seen": "Cover 3",
        "down": 1,
        "distance": 10,
        "result": "+8",
        "success": True,
        "video_id": "vod-a",
        "conf_play": 0.9,
        "conf_success": 0.8,
    }
    row.update(kwargs)
    return row


def _fit(tmp_path: Path, rows: list[dict], **kwargs):
    return train_and_write(
        rows,
        tmp_path / "models",
        label_version="v0.7",
        input_paths=[],
        grid=kwargs.pop("grid", TINY_GRID),
        min_vods_eval=kwargs.pop("min_vods_eval", 3),
        min_rows_eval=kwargs.pop("min_rows_eval", 6),
        **kwargs,
    )


def test_default_grid_matches_the_box_search():
    grid = default_grid()
    assert len(grid) == 45
    assert grid[0] == (1.0, 1.0, 16.0)
    assert grid[1] == (1.0, 1.0, 128.0)
    assert grid[-1] == (16.0, 256.0, 1024.0)
    assert DEFAULT_HP == (2.0, 16.0, 64.0)
    assert SCRIPT_VERSION == "train_vod_model.v2"


def test_coach_success_rule_when_the_label_is_absent():
    made = snap_from_row({**{k: v for k, v in _row().items() if k != "success"}, "result": "+4"})
    short_rule = snap_from_row({**{k: v for k, v in _row().items() if k != "success"}, "result": "+3"})
    third = snap_from_row({
        **{k: v for k, v in _row().items() if k != "success"},
        "down": 3,
        "result": "+9",
    })
    assert short_rule.success is False and short_rule.success_source == "coach_rule"
    assert made.success is True
    assert third.success is False
    assert short_rule.dist == "long"
    assert short_rule.look_family == "cover_3"
    assert short_rule.call_class == "run"


def test_outputs_use_the_box_names_and_keys(tmp_path: Path):
    rows = []
    for vod, success in (("va", True), ("vb", False), ("vc", True)):
        for _ in range(4):
            rows.append(_row(video_id=vod, success=success, play="HB Dive", coverage_seen="Cover 3"))
    written = _fit(tmp_path, rows)
    dest = Path(written["model_dir"])
    assert dest.name == "v0.7"
    for name in OUTPUTS:
        assert (dest / name).is_file(), name
    last = json.loads((tmp_path / "models" / "LAST_TRAINED.json").read_text())
    assert set(last) == {
        "label_version", "latest_sha256", "labels_data_sha256", "input_signature",
        "script_version", "script_sha256", "model_dir", "trained_at",
    }
    assert last["label_version"] == "v0.7"
    assert last["script_version"] == SCRIPT_VERSION
    assert last["model_dir"].endswith("/v0.7")

    params = json.loads((dest / "model_params.json").read_text())
    assert set(params) == {"label_version", "script_version", "strata"}
    stratum = params["strata"]["madden27/human"]
    assert set(stratum) == {
        "n_train_rows", "base_rate", "hp_selection", "model", "situation_only_model",
    }
    assert set(stratum["model"]) == {"intercept", "coef", "hyperparams"}
    assert set(stratum["model"]["hyperparams"]) == {
        "lambda_situation", "lambda_entity", "lambda_interaction",
    }
    assert set(stratum["model"]["coef"]) == COEF_GROUPS
    assert set(stratum["situation_only_model"]["coef"]) == SITUATION_GROUPS
    assert isinstance(stratum["hp_selection"], list)
    assert stratum["hp_selection"][0]["hp"] == [2.0, 16.0, 64.0]

    metrics = json.loads((dest / "metrics.json").read_text())
    assert set(metrics) == {
        "label_version", "script_version", "counts", "strata", "settings",
        "success_source", "excluded_unknown_opponent", "trained_at",
    }
    counts = metrics["counts"]["madden27/human"]
    assert set(counts) == {
        "rows", "scrimmage_rows", "known_success_rows", "success_rate_known",
        "clean_pairs", "clean_pairs_known_success", "rows_with_call_known_success",
        "rows_with_look_known_success", "vods", "vods_with_known_success",
        "known_success_conf_ge_0.6", "opponent_type_sources",
        "streamer_side_known_success", "trained",
    }
    assert counts["trained"] is True
    evaluated = metrics["strata"]["madden27/human"]
    assert evaluated["status"] == "evaluated"
    assert set(evaluated) == {"n_train_rows", "vods", "status", "held_out", "folds", "final_hp"}
    held = evaluated["held_out"]["all_known"]
    assert set(held) == {
        "n", "positives", "obs_rate", "model", "situation_only_model",
        "overall_base_rate", "situation_base_rate", "call_base_rate",
    }
    assert set(held["model"]) == {"log_loss", "brier", "ece", "reliability"}
    assert set(evaluated["folds"][0]) == {
        "held_out_vod", "n_test", "n_train", "hp", "test_obs_rate",
        "model_log_loss", "overall_base_log_loss",
    }
    assert set(metrics["settings"]) == {
        "min_vods_eval", "min_rows_eval", "grid", "default_hp",
        "conf_min_names", "n_bins", "split",
    }
    assert metrics["settings"]["default_hp"] == [2.0, 16.0, 64.0]
    assert metrics["settings"]["min_vods_eval"] == 3

    beaters = json.loads((dest / "beaters.json").read_text())
    assert set(beaters) == {"label_version", "tentative_n", "strata"}
    assert beaters["tentative_n"] == 15
    view = beaters["strata"]["madden27/human"]["views"]["all_clean_pairs"]
    assert set(view) == {"n_rows", "stratum_clean_success_rate", "look_family", "look"}
    family = view["look_family"]
    assert set(family) == {
        "prior_strength_m", "prior_strength_note", "n_cells",
        "cells_n_ge_tentative", "largest_cells", "looks",
    }
    look = family["looks"]["cover_3"]
    assert set(look) == {"look", "n", "successes", "raw_success", "look_prior", "calls"}
    assert set(look["calls"][0]) == BEATER_CALL_KEYS

    manifest = json.loads((dest / "manifest.json").read_text())
    assert set(manifest) == {
        "label_version", "trained_at", "script", "script_version", "script_sha256",
        "latest_json", "latest_json_sha256", "labels_dir", "label_files",
        "labels_data_sha256", "label_side_files_sha256", "snaps_notes_sha256",
        "rows_loaded", "outputs", "rerun",
    }
    assert manifest["outputs"] == OUTPUTS
    assert set(manifest["latest_json"]) == {
        "latest", "path", "schema_version", "vods", "rows", "clean_pairs",
        "script_version", "input_signature", "created_at", "note",
    }


def test_cpu_and_human_are_separate_and_unknown_is_not_trained(tmp_path: Path):
    rows = []
    for _ in range(8):
        rows.append(_row(opponent_type="human", success=False, result="+0", video_id="h1"))
        rows.append(_row(opponent_type="cpu", success=True, result="+8", video_id="c1", side="offense"))
    rows.append(_row(opponent_type="", success=True, video_id="u1"))
    written = _fit(tmp_path, rows, min_vods_eval=3, min_rows_eval=40)
    params = written["params"]
    assert set(params["strata"]) == {"madden27/cpu", "madden27/human"}
    assert "madden27/unknown" not in params["strata"]
    assert params["strata"]["madden27/human"]["base_rate"] == 0.0
    assert params["strata"]["madden27/cpu"]["base_rate"] == 1.0
    human_p = _prob(params["strata"]["madden27/human"]["model"], rows[0])
    cpu_p = _prob(params["strata"]["madden27/cpu"]["model"], rows[1])
    assert human_p < 0.5 < cpu_p
    metrics = written["metrics"]
    assert metrics["counts"]["madden27/unknown"]["trained"] is False
    assert metrics["excluded_unknown_opponent"]["rows"] == 1
    assert metrics["excluded_unknown_opponent"]["known_success"] == 1
    cpu_views = written["beaters"]["strata"]["madden27/cpu"]["views"]
    human_views = written["beaters"]["strata"]["madden27/human"]["views"]
    assert "vs_cpu_defense (streamer on offense)" in cpu_views
    assert "vs_cpu_defense (streamer on offense)" not in human_views
    assert metrics["strata"]["madden27/cpu"]["status"] == "not_evaluated"
    assert "leave-one-VOD-out" in metrics["strata"]["madden27/cpu"]["reason"]
    assert metrics["strata"]["madden27/cpu"]["final_hp"] == [2.0, 16.0, 64.0]
    assert params["strata"]["madden27/cpu"]["hp_selection"] == (
        "default (fewer than 3 VODs for inner CV)"
    )


def test_low_name_confidence_is_not_a_clean_pair():
    snap = snap_from_row(_row(conf_play=0.2, coverage_seen="Cover 3"))
    assert snap.call_key == ""
    assert snap.clean is False


def test_missing_yardline_is_unk_not_open_field():
    snap = snap_from_row(_row(yardline=""))
    assert snap.zone == "unk"
    red = snap_from_row(_row(yardline=85))
    assert red.zone == "red_zone"


def test_cli_writes_last_trained(tmp_path: Path, capsys):
    batch = tmp_path / "snaps.json"
    rows = [_row(video_id="only", success=True) for _ in range(4)]
    batch.write_text(json.dumps({"snaps": rows}), encoding="utf-8")
    models = tmp_path / "models"
    code = main([
        "train", str(batch),
        "--models-dir", str(models),
        "--label-version", "0.7",
    ])
    assert code == 0
    text = capsys.readouterr().out
    assert "madden27/human" in text
    assert (models / "v0.7" / "beaters.json").is_file()
    last = json.loads((models / "LAST_TRAINED.json").read_text())
    assert last["label_version"] == "v0.7"
    assert "Wrote" in text
