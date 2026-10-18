"""Defense offline scores minimize predicted offense success.

The offense call stays empty. Our defensive candidate is the look.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfb_coach.vod_success import shared_features as sf
from scripts.madden_ml import eval_harness
from scripts.madden_ml.eval_harness import score_offline_model

pytestmark = pytest.mark.filterwarnings(
    "error:CFB_COACH_MADDEN_DB and CFB_COACH_DB resolve to the same file:UserWarning"
)

_CALL_DERIVED = ("call", "call_class", "call_x_family")


@pytest.fixture(autouse=True)
def separate_coach_databases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coach_dir = tmp_path / "coach-db"
    madden_dir = tmp_path / "madden-db"
    coach_dir.mkdir()
    madden_dir.mkdir()
    monkeypatch.setenv("CFB_COACH_DB", str(coach_dir / "coach.db"))
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", str(madden_dir / "live.db"))


@pytest.fixture
def defense_book(monkeypatch: pytest.MonkeyPatch):
    original = eval_harness.prepare_session

    def wrapped(db) -> None:
        original(db)
        from cfb_coach.madden.playbook import lock_books

        record = {
            "side": "defense",
            "mode": "stock",
            "name": "fixture",
            "formations": {"Nickel": ["Cover 3", "Cover 2"]},
            "audibles": {},
            "core": ["Nickel"],
            "rev": 1,
        }
        lock_books(
            db,
            {"defense": {"record": record, "change": "none", "untouched": False}},
            applied=True,
        )

    monkeypatch.setattr(eval_harness, "prepare_session", wrapped)


def _write_model(folder: Path, coef: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_params.json").write_text(
        json.dumps({
            "strata": {
                "madden27/human/defense": {
                    "model": {"intercept": 0.0, "coef": coef},
                },
            },
        }),
        encoding="utf-8",
    )
    (folder / "manifest.json").write_text(
        json.dumps({"model_version": "defense-fixture", "scored_on": "unreviewed"}),
        encoding="utf-8",
    )
    return folder


def _assert_call_empty(row: dict, play: str) -> None:
    groups = row["features"]["groups"]
    assert groups["call"]["key"] is None
    assert groups["call_x_family"]["key"] is None
    # call_class is dropped on defense (recipe D2), so it is not built at all.
    assert groups["call_class"]["key"] is None
    folded = play.casefold()
    for name in _CALL_DERIVED:
        key = groups[name]["key"] or ""
        assert folded not in key
        assert groups[name]["matched"] is False
    assert row["features"]["ranking"] == "min_offense_success"
    assert row["features"]["call_matches"] == []


def _score(model: Path, **overrides):
    params = {
        "side": "defense",
        "model_dir": model,
        "strata": "madden27/human/defense",
        "opp_look": "Mesh",
        "seed": 1,
        "offline_ok": True,
    }
    params.update(overrides)
    return score_offline_model("1st and 10", **params)


def _direction_models(tmp_path: Path) -> tuple[Path, Path]:
    # Cover 2 lowers offense success. Cover 3 raises it. A max ranking would
    # pick Cover 3. Stuffing our play into ``call`` would add the +100 weights
    # and, under a min ranking, would also pick Cover 3.
    low_cover_2 = _write_model(
        tmp_path / "low-cover-2",
        {
            "look_family": {"cover_2": -4.0, "cover_3": 5.0},
            "call": {"cover 2": 100.0, "cover 3": 100.0},
            "call_class": {"unknown": 50.0},
            "call_x_family": {
                "cover 2|cover_2": 80.0,
                "cover 3|cover_3": 80.0,
            },
        },
    )
    low_cover_3 = _write_model(
        tmp_path / "low-cover-3",
        {"look_family": {"cover_2": 3.0, "cover_3": -8.0}},
    )
    return low_cover_2, low_cover_3


def test_lower_offense_success_look_ranks_first(tmp_path: Path, defense_book) -> None:
    del defense_book
    low_cover_2, low_cover_3 = _direction_models(tmp_path)
    low = _score(low_cover_2)
    assert low["play"] == "Cover 2"
    assert low["play"] != "Cover 3"
    assert low["legal"] is True
    assert low["side"] == "defense"
    assert low["features"]["groups"]["look"]["key"] == "cover 2"
    assert low["features"]["groups"]["look_family"] == {
        "key": "cover_2",
        "matched": True,
        "weight": -4.0,
    }
    assert low["features"]["groups"]["look"]["key"] != "mesh"
    assert low["features"]["ranking"] == "min_offense_success"

    other = _score(low_cover_3)
    assert other["play"] == "Cover 3"
    assert other["features"]["groups"]["look"]["key"] == "cover 3"
    assert other["features"]["groups"]["look_family"]["key"] == "cover_3"
    assert other["features"]["groups"]["look_family"]["weight"] == -8.0


def test_defense_rows_leave_call_and_call_x_family_empty(tmp_path: Path, defense_book) -> None:
    del defense_book
    low_cover_2, low_cover_3 = _direction_models(tmp_path)
    rows = (_score(low_cover_2), _score(low_cover_3))
    plays = ("Cover 2", "Cover 3")
    for row, play in zip(rows, plays, strict=True):
        _assert_call_empty(row, play)
        assert row["features"]["groups"]["call"]["key"] is None
        assert row["features"]["groups"]["call_x_family"]["key"] is None
    # Training side: a defense label row whose own call (def_call) is Cover 2.
    trainer = sf.feature_levels(sf.training_inputs({
        "side": "defense",
        "offense_call": {"play": "HB Dive"},
        "defense_look": {"def_call": "Cover 2", "coverage_seen": "Cover 4"},
        "conf": {"play": 1.0, "def_call": 1.0, "coverage_seen": 1.0},
        "down": 1,
        "distance": 10,
    }))
    for dropped in ("call", "call_x_family", "call_class", "class_x_family"):
        assert dropped not in trainer
    low = rows[0]
    assert low["features"]["groups"]["look_family"]["key"] == trainer["look_family"] == "cover_2"
    assert low["features"]["groups"]["look"]["key"] == trainer["look"] == "cover 2"
    assert low["features"]["groups"]["class_x_family"]["key"] is None
