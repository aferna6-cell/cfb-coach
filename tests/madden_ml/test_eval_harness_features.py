"""Offline scorer feature keys follow the vod_success trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfb_coach.vod_success.features import feature_map, snap_from_row
from scripts.madden_ml import eval_harness
from scripts.madden_ml.eval_harness import score_offline_model

pytestmark = pytest.mark.filterwarnings(
    "error:CFB_COACH_MADDEN_DB and CFB_COACH_DB resolve to the same file:UserWarning"
)

_GROUPS = (
    "call",
    "call_class",
    "look",
    "look_family",
    "call_x_family",
    "class_x_family",
    "down",
    "dist",
    "down_x_dist",
    "zone",
    "score",
    "time",
    "side",
    "pressure",
    "off_adj",
    "def_adj",
)


@pytest.fixture(autouse=True)
def separate_coach_databases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coach_dir = tmp_path / "coach-db"
    madden_dir = tmp_path / "madden-db"
    coach_dir.mkdir()
    madden_dir.mkdir()
    monkeypatch.setenv("CFB_COACH_DB", str(coach_dir / "coach.db"))
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", str(madden_dir / "live.db"))


@pytest.fixture
def tiny_book(monkeypatch: pytest.MonkeyPatch):
    original = eval_harness.prepare_session

    def wrapped(db) -> None:
        original(db)
        from cfb_coach.madden.playbook import lock_books

        record = {
            "side": "offense",
            "mode": "stock",
            "name": "fixture",
            "formations": {"Gun": ["Inside Zone", "Mesh", "0 1 Trap"]},
            "audibles": {},
            "core": ["Gun"],
            "rev": 1,
        }
        lock_books(
            db,
            {"offense": {"record": record, "change": "none", "untouched": False}},
            applied=True,
        )

    monkeypatch.setattr(eval_harness, "prepare_session", wrapped)


def _write_model(folder: Path, coef: dict, *, intercept: float = 0.0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_params.json").write_text(
        json.dumps({
            "strata": {
                "madden27/human/offense": {
                    "model": {"intercept": intercept, "coef": coef},
                },
            },
        }),
        encoding="utf-8",
    )
    (folder / "manifest.json").write_text(
        json.dumps({"model_version": "feature-fixture", "scored_on": "unreviewed"}),
        encoding="utf-8",
    )
    return folder


def _score(model: Path, sit, **overrides):
    params = {
        "side": "offense",
        "model_dir": model,
        "strata": "madden27/human/offense",
        "opp_look": "Cover 3",
        "seed": 1,
        "offline_ok": True,
    }
    params.update(overrides)
    return score_offline_model(sit, **params)


def test_situation_keys_differ_and_call_x_family_flips_the_argmax(tmp_path: Path, tiny_book) -> None:
    del tiny_book
    model = _write_model(
        tmp_path / "model",
        {
            "call": {"0 1 trap": 0.1, "mesh": 0.1, "inside zone": 0.1},
            "call_x_family": {
                "0 1 trap|cover_3": 5.0,
                "inside zone|cover_3": 1.0,
                "inside zone|cover_2": 1.0,
                "mesh|cover_2": 5.0,
            },
            "class_x_family": {"pass|cover_2": 0.5},
            "down_x_dist": {"1|long": 1.5, "3|xlong": -2.0},
        },
    )
    first = _score(model, "1st and 10", opp_look="Cover 3")
    third = _score(model, "3rd and 12", opp_look="Cover 3")
    cover2 = _score(model, "1st and 10", opp_look="Cover 2")

    assert first["features"]["groups"]["down"]["key"] == "1"
    assert first["features"]["groups"]["dist"]["key"] == "long"
    assert first["features"]["groups"]["down_x_dist"] == {
        "key": "1|long",
        "matched": True,
        "weight": 1.5,
    }
    assert third["features"]["groups"]["down"]["key"] == "3"
    assert third["features"]["groups"]["dist"]["key"] == "xlong"
    assert third["features"]["groups"]["down_x_dist"] == {
        "key": "3|xlong",
        "matched": True,
        "weight": -2.0,
    }
    assert first["features"]["groups"]["down_x_dist"]["key"] != third["features"]["groups"]["down_x_dist"]["key"]
    assert first["play"] == "0 1 Trap"
    assert cover2["play"] == "Mesh"
    assert first["play"] != cover2["play"]
    assert first["features"]["groups"]["call_x_family"] == {
        "key": "0 1 trap|cover_3",
        "matched": True,
        "weight": 5.0,
    }
    assert first["features"]["groups"]["look_family"]["key"] == "cover_3"
    assert cover2["features"]["groups"]["call_x_family"]["key"] == "mesh|cover_2"
    assert cover2["features"]["groups"]["look_family"]["key"] == "cover_2"
    assert cover2["features"]["groups"]["class_x_family"] == {
        "key": "pass|cover_2",
        "matched": True,
        "weight": 0.5,
    }
    assert "0 1 Trap" in first["features"]["call_matches"]
    assert "Mesh" in cover2["features"]["call_matches"]
    for name in _GROUPS:
        assert name in first["features"]["groups"]
        assert set(first["features"]["groups"][name]) == {"key", "matched", "weight"}


def test_real_key_format_strings_and_csv_coercion(tmp_path: Path, tiny_book) -> None:
    del tiny_book
    model = _write_model(
        tmp_path / "model",
        {
            "call": {"0 1 trap": 3.0, "mesh": 1.0},
            "call_x_family": {"0 1 trap|cover_3": 4.0, "mesh|cover_2": 4.0},
            "class_x_family": {"pass|cover_2": 2.0, "run|cover_3": 0.25},
            "look": {"cover 3": 0.1},
            "look_family": {"cover_3": 0.2, "cover_2": 0.2},
        },
    )
    trap = _score(model, "1st and 10 own 25", opp_look="Cover 3")
    mesh = _score(model, "1st and 10 own 25", opp_look="Cover 2")
    trainer = feature_map(snap_from_row({
        "game": "madden27",
        "opponent_type": "human",
        "side": "offense",
        "play": "0 1 Trap",
        "formation": "Gun",
        "down": 1,
        "distance": 10,
        "yardline": 25,
        "look": "Cover 3",
        "coverage_seen": "Cover 3",
        "situation_raw": "1st and 10 own 25",
        "success": None,
    }))
    groups = trap["features"]["groups"]
    assert trap["play"] == "0 1 Trap"
    assert groups["call"]["key"] == "0 1 trap" == trainer["call"]
    assert groups["call_x_family"]["key"] == "0 1 trap|cover_3" == trainer["call_x_family"]
    assert groups["class_x_family"]["key"] == "run|cover_3" == trainer["class_x_family"]
    assert groups["look"]["key"] == "cover 3" == trainer["look"]
    assert groups["look_family"]["key"] == "cover_3"
    assert groups["zone"]["key"] == "own" == trainer["zone"]
    assert groups["down_x_dist"]["key"] == "1|long"
    assert groups["call"]["matched"] is True
    assert groups["call"]["weight"] == 3.0
    assert mesh["play"] == "Mesh"
    assert mesh["features"]["groups"]["class_x_family"]["key"] == "pass|cover_2"
    assert mesh["features"]["groups"]["call_x_family"]["key"] == "mesh|cover_2"

    csv = _score(
        model,
        {"down": "3", "distance": "12", "yardline": "25", "personnel": "   "},
        opp_look="  ",
    )
    assert csv["features"]["raw"]["down"] == {"value": "3", "type": "str"}
    assert csv["features"]["raw"]["distance"] == {"value": "12", "type": "str"}
    assert csv["features"]["raw"]["yardline"] == {"value": "25", "type": "str"}
    assert csv["features"]["raw"]["personnel"] == {"value": "   ", "type": "str"}
    assert csv["features"]["raw"]["look"] == {"value": "  ", "type": "str"}
    assert csv["features"]["coerced"] == {"look": None, "personnel": None}
    assert csv["opp_look"] is None
    assert csv["features"]["groups"]["down"]["key"] == "3"
    assert csv["features"]["groups"]["dist"]["key"] == "xlong"
    assert csv["features"]["groups"]["down_x_dist"]["key"] == "3|xlong"
    assert csv["features"]["groups"]["zone"]["key"] == "own"
    assert csv["features"]["groups"]["look"] == {"key": None, "matched": False, "weight": 0.0}
    assert csv["features"]["groups"]["look_family"]["key"] == "none"
    assert csv["features"]["groups"]["pressure"] == {"key": None, "matched": False, "weight": 0.0}


def test_trainer_builds_pressure_and_adjustment_keys(tmp_path: Path, tiny_book) -> None:
    del tiny_book
    model = _write_model(
        tmp_path / "model",
        {
            "pressure": {"yes": 1.0},
            "off_adj": {"yes": 1.0},
            "def_adj": {"yes": 1.0},
        },
    )
    pressed = _score(model, "1st and 10", opp_look="Fire Blitz")
    trainer = feature_map(snap_from_row({
        "game": "madden27",
        "opponent_type": "human",
        "side": "offense",
        "play": pressed["play"],
        "formation": pressed["formation"],
        "down": 1,
        "distance": 10,
        "look": "Fire Blitz",
        "coverage_seen": "Fire Blitz",
        "situation_raw": "1st and 10",
        "success": None,
    }))
    assert pressed["features"]["groups"]["pressure"] == {
        "key": "yes",
        "matched": True,
        "weight": 1.0,
    }
    assert pressed["features"]["groups"]["pressure"]["key"] == trainer["pressure"]
    assert pressed["features"]["groups"]["look_family"]["key"] == trainer["look_family"] == "pressure_other"
    assert pressed["features"]["groups"]["off_adj"]["key"] is None
    assert pressed["features"]["groups"]["def_adj"]["key"] is None

    flagged = _score(
        model,
        {
            "down": "1",
            "distance": "10",
            "off_adj": "yes",
            "def_adjustment": "1",
            "adjustment": "   ",
        },
        opp_look="Cover 3",
    )
    assert flagged["features"]["groups"]["off_adj"] == {"key": "yes", "matched": True, "weight": 1.0}
    assert flagged["features"]["groups"]["def_adj"] == {"key": "yes", "matched": True, "weight": 1.0}
    assert flagged["features"]["groups"]["pressure"]["key"] is None
    assert flagged["features"]["raw"]["off_adj"] == {"value": "yes", "type": "str"}
    assert flagged["features"]["raw"]["def_adjustment"] == {"value": "1", "type": "str"}
    assert flagged["side"] == "offense"
    shared = feature_map(snap_from_row({
        "game": "madden27",
        "opponent_type": "human",
        "side": "offense",
        "play": flagged["play"],
        "formation": flagged["formation"],
        "down": "1",
        "distance": "10",
        "look": "Cover 3",
        "coverage_seen": "Cover 3",
        "off_adj": "yes",
        "def_adjustment": "1",
        "situation_raw": "1&10",
        "success": None,
    }))
    assert flagged["features"]["groups"]["off_adj"]["key"] == shared["off_adj"]
    assert flagged["features"]["groups"]["def_adj"]["key"] == shared["def_adj"]

    blank = _score(
        model,
        {"down": "1", "distance": "10", "off_adj": "  ", "def_adj": "none"},
        opp_look="Cover 3",
    )
    assert blank["features"]["groups"]["off_adj"]["key"] is None
    assert blank["features"]["groups"]["def_adj"]["key"] is None

    defense = _score(model, "d 1st and 10", side="defense", opp_look="Cover 3")
    assert defense["side"] == "defense"
    assert defense["features"]["groups"]["side"]["key"] == "defense"
    assert defense["features"]["groups"]["off_adj"]["key"] is None
    assert defense["features"]["groups"]["def_adj"]["key"] is None
    assert defense["features"]["groups"]["pressure"]["key"] is None
