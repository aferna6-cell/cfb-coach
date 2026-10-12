"""Live eval harness: contract, reproducibility, legality, hybrid guard, home, budget."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cfb_coach.madden.model.inference import ManualClock
from cfb_coach.madden.model.schema import ML_LATENCY_BUDGET_MS, MacroState
from cfb_coach.madden.playcaller import MaddenCall
from scripts.madden_ml import eval_harness
from scripts.madden_ml.eval_harness import main, run_situation

_TOP = (
    "seed",
    "new",
    "old",
    "fallback_reason",
    "model_version",
    "scored_on",
    "latency_ms",
    "budget_ms",
    "over_budget",
    "legal",
    "propensity",
    "opp_look",
)
_SIDE = ("formation", "play", "adjustment_id", "macro_state", "line", "mode")
_OFFENSE_EXACT = r"^.+ — .+ \| .+ \| .+ → .+$"


def _row(**overrides):
    params = {
        "mode": "heuristic",
        "opp_look": None,
        "side": "offense",
        "seed": 1,
    }
    params.update(overrides)
    return run_situation("1&10", **params)


def test_contract_fields_and_blank_look_versus_empty() -> None:
    blank = _row(opp_look="   ")
    empty = _row(opp_look="empty", seed=1)
    missing = _row(opp_look=None)
    for row in (blank, empty, missing):
        for key in _TOP:
            assert key in row
        for label in ("new", "old"):
            for key in _SIDE:
                assert key in row[label]
        assert set(row["latency_ms"]) >= {"whole_call", "model"}
        assert row["budget_ms"] == 150
        assert row["budget_ms"] == ML_LATENCY_BUDGET_MS
        assert isinstance(row["legal"], bool)
        assert isinstance(row["over_budget"], bool)
        assert row["new"]["line"].count("\n") == 0
        assert re.fullmatch(_OFFENSE_EXACT, row["new"]["line"])
        assert row["seed"] == 1
    assert blank["opp_look"] is None
    assert missing["opp_look"] is None
    assert empty["opp_look"] == "empty"
    assert blank["opp_look"] != empty["opp_look"]
    # A blank argument does not pick up the word empty sitting in the raw line.
    spaced = run_situation(
        "1&10 empty", mode="heuristic", opp_look="  ", side="offense", seed=1,
    )
    assert spaced["opp_look"] is None


def test_same_seed_reproduces_the_decision() -> None:
    kwargs = {"mode": "heuristic", "opp_look": None, "side": "offense", "seed": 11}
    first = run_situation("2&7 my 35", **kwargs)
    second = run_situation("2&7 my 35", **kwargs)
    assert first["new"] == second["new"]
    assert first["old"] == second["old"]
    assert first["legal"] == second["legal"]
    assert first["opp_look"] == second["opp_look"]
    assert first["new"]["formation"]
    assert first["new"]["play"]


def test_illegal_pick_and_unarmed_macro_are_not_legal() -> None:
    def illegal(*_args, **_kwargs):
        return MaddenCall("offense", "Nope", "Nope Play", "No adj", "Spot → Drag")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("cfb_coach.madden.playcaller.make_call", illegal)
        row = _row()
    assert row["legal"] is False
    assert row["new"]["formation"] == "Nope"
    assert row["new"]["adjustment_id"] is None

    legal = _row(seed=2)
    form, play = legal["new"]["formation"], legal["new"]["play"]

    def unarmed(*_args, **_kwargs):
        call = MaddenCall(
            "offense", form, play, "No adj", "Spot → Drag", macro="SECRET-MACRO",
        )
        call.macro_state = MacroState.DRAFT
        return call

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("cfb_coach.madden.playcaller.make_call", unarmed)
        row = _row(seed=2)
    assert row["legal"] is False
    assert row["new"]["adjustment_id"] == "SECRET-MACRO"
    assert row["new"]["macro_state"] == MacroState.DRAFT.value
    assert "SECRET-MACRO" not in row["new"]["line"]
    assert "SECRET-MACRO" not in row["old"]["line"]
    assert "| No adj |" in row["new"]["line"]


def test_hybrid_refused_for_unreviewed_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = eval_harness.prepare_session

    def wrapped(db) -> None:
        original(db)
        folder = Path(db.path).parent / "registry"
        artifact_path = Path(db.path).parent / "unreviewed.json"
        artifact_path.write_text(
            json.dumps({
                "model_type": "intercept_prior",
                "weights": [],
                "intercept": 0.0,
                "training_report": {"tags": {"scored_on": "unreviewed"}},
            }),
            encoding="utf-8",
        )
        from cfb_coach.madden.model.inference import set_model_version, set_registry_dir
        from cfb_coach.madden.model.registry import write_entry
        from cfb_coach.madden.model.schema import ModelRegistryEntry

        write_entry(
            ModelRegistryEntry(model_version="unreviewed-1", artifact_path=str(artifact_path), seed=1),
            str(folder),
        )
        set_registry_dir(db, str(folder))
        set_model_version(db, "unreviewed-1")

    monkeypatch.setattr(eval_harness, "prepare_session", wrapped)
    del tmp_path
    row = _row(mode="hybrid", seed=4)
    assert row["scored_on"] == "unreviewed"
    assert row["fallback_reason"] == "circuit_open"
    assert row["mode_downgrade_reason"] == "hybrid refused: scored_on is unreviewed"
    assert row["new"]["formation"] == row["old"]["formation"]
    assert row["new"]["play"] == row["old"]["play"]
    assert row["new"]["line"] == row["old"]["line"]
    assert row["over_budget"] is False


def test_home_stays_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    row = _row(seed=3)
    assert list(home.rglob("*")) == []
    assert row["budget_ms"] == 150
    assert row["new"]["line"]


def test_over_budget_model_call_falls_back_on_a_fake_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    tick = ManualClock()
    monkeypatch.setattr(eval_harness, "clock", tick)

    def slow(*_args, **_kwargs):
        tick.advance(1.0)
        raise AssertionError("a late model result must not be used")

    monkeypatch.setattr(eval_harness, "score_model", slow)
    row = _row(mode="shadow", seed=5)
    assert row["over_budget"] is True
    assert row["fallback_reason"] == "timeout"
    assert row["budget_ms"] == 150
    assert row["latency_ms"]["model"] <= 150
    assert row["new"]["formation"] == row["old"]["formation"]
    assert row["new"]["play"] == row["old"]["play"]
    assert row["new"]["line"] == row["old"]["line"]
    assert row["new"]["line"]


def test_cli_writes_jsonl_and_latency_summary(tmp_path: Path) -> None:
    situations = tmp_path / "situations.json"
    situations.write_text(
        json.dumps([
            {"situation": "1&10", "side": "offense", "mode": "heuristic", "seed": 9, "opp_look": ""},
        ]),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    code = main(["--situations", str(situations), "--passes", "2", "--out", str(out)])
    assert code == 0
    lines = [json.loads(line) for line in (out / "situations.jsonl").read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["reproduced"] is True
    assert lines[0]["opp_look"] is None
    summary = json.loads((out / "summary.json").read_text())
    assert summary["passes"] == 2
    assert summary["samples"] == 2
    assert summary["pass"] is True
    assert summary["budget_ms"] == 150
    for block in ("whole_call_ms", "model_ms"):
        assert set(summary[block]) == {"p50", "p95", "max"}
        assert summary[block]["max"] >= summary[block]["p95"] >= summary[block]["p50"]
    assert summary["model_ms"]["max"] <= 150
    assert "fallback_counts" in summary
