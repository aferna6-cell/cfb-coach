"""VOD training eligibility. Default stays reviewed; unreviewed is opt-in."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import subprocess
from pathlib import Path

import pytest

from cfb_coach.madden.model.dataset import (
    export_vod_training_dataset,
    reviewed_only_subset,
    vod_training_eligible,
)

# #44 head. dataset.py matches 5d9c566; that older SHA is not in this history.
_BASELINE_REV = "ae24c7933be599a7a389e1901ca7d0c5964f60a5"
_REPO = Path(__file__).resolve().parents[2]


def _row(**overrides):
    row = {
        "snap_id": "snap-1",
        "vod_id": "vod-a",
        "game": "madden",
        "opponent_type": "h2h",
        "permitted_use": "training",
        "review_status": "reviewed",
        "trainable": True,
        "source": "pilot-vod",
    }
    row.update(overrides)
    return row


def _ids(rows):
    return [row["snap_id"] for row in rows]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_hashes(folder: Path) -> dict[str, str]:
    return {
        str(path.relative_to(folder)): _sha256(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _load_dataset_at(rev: str, dest: Path):
    source = subprocess.check_output(
        ["git", "show", f"{rev}:cfb_coach/madden/model/dataset.py"],
        cwd=_REPO,
    )
    path = dest / f"dataset_{rev}.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(f"dataset_{rev}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flag_off_by_default_keeps_reviewed_trainable_only(tmp_path):
    signature = inspect.signature(export_vod_training_dataset)
    assert signature.parameters["include_unreviewed_permitted"].default is False
    rows = [
        _row(snap_id="reviewed", review_status="reviewed", trainable=True),
        _row(snap_id="confirmed", review_status="confirmed", trainable=True),
        _row(snap_id="unreviewed", review_status="unreviewed", trainable=True),
        _row(snap_id="not-trainable", review_status="reviewed", trainable=False),
    ]
    report = export_vod_training_dataset(rows, tmp_path / "out")
    assert report["include_unreviewed_permitted"] is False
    assert _ids(report["rows"]) == ["reviewed", "confirmed"]


def test_flag_on_adds_only_permitted_unreviewed_rows(tmp_path):
    rows = [
        _row(snap_id="reviewed"),
        _row(snap_id="unreviewed-ok", review_status="unreviewed", trainable=False),
        _row(
            snap_id="unreviewed-research",
            review_status="unreviewed",
            permitted_use="research",
        ),
        _row(snap_id="rejected", review_status="rejected", trainable=True),
        _row(
            snap_id="blocked-source",
            review_status="unreviewed",
            source="https://www.youtube.com/watch?v=F8Hm324SMtg",
        ),
    ]
    off = export_vod_training_dataset(rows, tmp_path / "off")
    on = export_vod_training_dataset(
        rows, tmp_path / "on", include_unreviewed_permitted=True
    )
    assert _ids(off["rows"]) == ["reviewed"]
    assert _ids(on["rows"]) == ["reviewed", "unreviewed-ok"]


def test_excluded_source_f8hm324smtg_always_excluded(tmp_path):
    rows = [
        _row(snap_id="clean"),
        _row(snap_id="id-field", vod_id="F8Hm324SMtg"),
        _row(
            snap_id="url",
            review_status="unreviewed",
            url="https://youtu.be/F8Hm324SMtg",
        ),
    ]
    for flag in (False, True):
        report = export_vod_training_dataset(
            rows, tmp_path / str(flag), include_unreviewed_permitted=flag
        )
        assert _ids(report["rows"]) == ["clean"]
        assert vod_training_eligible(rows[1], include_unreviewed_permitted=flag) is False


def test_non_training_permitted_use_always_excluded(tmp_path):
    rows = [
        _row(snap_id="training"),
        _row(snap_id="research", permitted_use="research"),
        _row(snap_id="blank", permitted_use=""),
        _row(snap_id="missing", permitted_use=None),
        _row(snap_id="unreviewed-analysis", permitted_use="analysis", review_status="unreviewed"),
    ]
    for flag in (False, True):
        report = export_vod_training_dataset(
            rows, tmp_path / str(flag), include_unreviewed_permitted=flag
        )
        assert _ids(report["rows"]) == ["training"]


def test_unpermitted_rows_excluded_when_flag_on(tmp_path):
    rows = [
        _row(snap_id="allowed-unreviewed", review_status="unreviewed"),
        _row(
            snap_id="unpermitted-flag",
            review_status="unreviewed",
            permitted=False,
        ),
        _row(
            snap_id="unpermitted-reviewed",
            review_status="reviewed",
            trainable=True,
            permitted=False,
        ),
        _row(
            snap_id="unpermitted-use",
            review_status="unreviewed",
            permitted_use="internal",
        ),
    ]
    report = export_vod_training_dataset(
        rows, tmp_path / "on", include_unreviewed_permitted=True
    )
    assert _ids(report["rows"]) == ["allowed-unreviewed"]


def test_auto_review_status_counts_as_unreviewed(tmp_path):
    auto = _row(
        snap_id="auto",
        review_status="auto",
        trainable=False,
        permitted_use="training",
        source="pilot-vod",
    )
    before = dict(auto)
    off = export_vod_training_dataset([dict(auto)], tmp_path / "off")
    on = export_vod_training_dataset(
        [dict(auto)],
        tmp_path / "on",
        include_unreviewed_permitted=True,
    )
    assert off["include_unreviewed_permitted"] is False
    assert on["include_unreviewed_permitted"] is True
    assert off["rows"] == []
    assert vod_training_eligible(auto, include_unreviewed_permitted=False) is False
    assert vod_training_eligible(auto, include_unreviewed_permitted=True) is True
    assert _ids(on["rows"]) == ["auto"]
    assert on["rows"][0]["review_status"] == "auto"
    assert on["rows"][0]["trainable"] is False
    assert auto == before


def test_rejected_and_unknown_status_excluded_when_flag_on(tmp_path):
    rows = [
        _row(snap_id="reviewed"),
        _row(snap_id="rejected", review_status="rejected", trainable=True),
        _row(snap_id="typo", review_status="reviewd_typo", trainable=True),
        _row(snap_id="null-status", review_status=None, trainable=True),
        _row(snap_id="auto-capital", review_status="Auto", trainable=True),
        _row(snap_id="auto", review_status="auto", trainable=False),
        _row(snap_id="unreviewed", review_status="unreviewed", trainable=False),
    ]
    on = export_vod_training_dataset(
        rows, tmp_path / "on", include_unreviewed_permitted=True
    )
    off = export_vod_training_dataset(rows, tmp_path / "off")
    assert _ids(on["rows"]) == ["reviewed", "auto", "unreviewed"]
    assert _ids(off["rows"]) == ["reviewed"]
    excluded_on = {"rejected", "typo", "null-status", "auto-capital"}
    for row in rows:
        if row["snap_id"] == "reviewed":
            continue
        assert vod_training_eligible(row, include_unreviewed_permitted=False) is False
        if row["snap_id"] in excluded_on:
            assert vod_training_eligible(row, include_unreviewed_permitted=True) is False


def test_review_status_and_trainable_pass_through_unchanged(tmp_path):
    untouched = _row(snap_id="reviewed", review_status="reviewed", trainable=True)
    unreviewed = _row(snap_id="unreviewed", review_status="unreviewed", trainable=False)
    missing = _row(snap_id="missing-status")
    del missing["review_status"]
    explicit_none = _row(snap_id="none-status", review_status=None, trainable=None)
    before = [
        dict(untouched),
        dict(unreviewed),
        dict(missing),
        dict(explicit_none),
    ]
    report = export_vod_training_dataset(
        [untouched, unreviewed, missing, explicit_none],
        tmp_path / "out",
        include_unreviewed_permitted=True,
    )
    assert [untouched, unreviewed, missing, explicit_none] == before
    exported = {row["snap_id"]: row for row in report["rows"]}
    assert exported["reviewed"]["review_status"] == "reviewed"
    assert exported["reviewed"]["trainable"] is True
    assert exported["unreviewed"]["review_status"] == "unreviewed"
    assert exported["unreviewed"]["trainable"] is False
    assert "missing-status" not in exported
    assert "none-status" not in exported


def test_group_does_not_span_train_and_held_out(tmp_path):
    rows = [
        _row(snap_id="a1", vod_id="vod-a"),
        _row(snap_id="a2", vod_id="vod-a"),
        _row(snap_id="b1", vod_id="vod-b"),
        _row(snap_id="b2", vod_id="vod-b", review_status="unreviewed", trainable=False),
    ]
    report = export_vod_training_dataset(
        rows, tmp_path / "out", include_unreviewed_permitted=True
    )
    train_groups = {_group(row) for row in report["train"]}
    held_groups = {_group(row) for row in report["held_out"]}
    assert train_groups.isdisjoint(held_groups)
    assert train_groups == {"vod-a"}
    assert held_groups == {"vod-b"}
    assert _ids(report["held_out"]) == ["b1", "b2"]


def test_cpu_h2h_and_madden_cfb_stay_separated(tmp_path):
    rows = [
        _row(snap_id="madden-h2h", game="madden", opponent_type="h2h", vod_id="same"),
        _row(snap_id="madden-cpu", game="madden", opponent_type="cpu", vod_id="same"),
        _row(snap_id="cfb-h2h", game="cfb", opponent_type="h2h", vod_id="same"),
        _row(snap_id="cfb-cpu", game="cfb27", opponent_type="cpu", vod_id="same"),
    ]
    report = export_vod_training_dataset(rows, tmp_path / "out")
    strata = report["strata"]
    expected = {
        ("madden", "h2h"): ["madden-h2h"],
        ("madden", "cpu"): ["madden-cpu"],
        ("cfb", "h2h"): ["cfb-h2h"],
        ("cfb", "cpu"): ["cfb-cpu"],
    }
    assert set(strata) == set(expected)
    for key, bucket in strata.items():
        pooled = bucket["train"] + bucket["held_out"]
        assert {_stratum(row) for row in pooled} == {key}
        assert _ids(pooled) == expected[key]


def test_reviewed_only_held_out_subset(tmp_path):
    rows = [
        _row(snap_id="train-reviewed", vod_id="vod-a"),
        _row(snap_id="held-reviewed", vod_id="vod-b", review_status="confirmed"),
        _row(
            snap_id="held-unreviewed",
            vod_id="vod-b",
            review_status="unreviewed",
            trainable=False,
        ),
    ]
    report = export_vod_training_dataset(
        rows, tmp_path / "out", include_unreviewed_permitted=True
    )
    assert _ids(report["held_out"]) == ["held-reviewed", "held-unreviewed"]
    subset = report["held_out_reviewed_only"]
    assert subset == reviewed_only_subset(report["held_out"])
    assert _ids(subset) == ["held-reviewed"]
    assert subset[0]["review_status"] == "confirmed"
    assert subset[0]["trainable"] is True
    assert report["held_out"][1]["review_status"] == "unreviewed"


def test_flag_off_export_matches_5d9c566_bytes(tmp_path):
    """Default export files match #44's JSONL writer on the same fixture.

    The writer is the one on the current #44 head. Its ``export_jsonl`` matches
    ``5d9c566``.
    """
    rows = [
        _row(snap_id="m1", vod_id="vod-a", game="madden", opponent_type="h2h"),
        _row(snap_id="m2", vod_id="vod-b", game="madden", opponent_type="cpu", review_status="confirmed"),
        _row(snap_id="c1", vod_id="vod-c", game="cfb", opponent_type="h2h", trainable=True),
    ]
    historical = _load_dataset_at(_BASELINE_REV, tmp_path)
    old_dir = tmp_path / "5d9c566"
    new_dir = tmp_path / "head"
    old_dir.mkdir()
    historical.export_jsonl(rows, str(old_dir / "dataset.jsonl"))
    export_vod_training_dataset(rows, new_dir, include_unreviewed_permitted=False)
    old_hashes = _file_hashes(old_dir)
    new_hashes = _file_hashes(new_dir)
    assert old_hashes.keys() == new_hashes.keys()
    assert old_hashes == new_hashes
    assert set(old_hashes) == {"dataset.jsonl"}


def _group(row) -> str:
    return str(row["vod_id"])


def _stratum(row) -> tuple[str, str]:
    game = "madden" if str(row["game"]).startswith("madden") else "cfb"
    return game, row["opponent_type"]
