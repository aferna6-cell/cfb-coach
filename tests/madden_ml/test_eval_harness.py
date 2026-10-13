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
from scripts.madden_ml.eval_harness import main, run_situation, score_offline_model

pytestmark = pytest.mark.filterwarnings(
    "error:CFB_COACH_MADDEN_DB and CFB_COACH_DB resolve to the same file:UserWarning"
)


@pytest.fixture(autouse=True)
def separate_coach_databases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two different temp files in two directories. A shared file fails the test."""
    coach_dir = tmp_path / "coach-db"
    madden_dir = tmp_path / "madden-db"
    coach_dir.mkdir()
    madden_dir.mkdir()
    coach = coach_dir / "coach.db"
    madden = madden_dir / "live.db"
    assert coach.parent != madden.parent
    assert madden.resolve() != coach.resolve()
    assert madden.name != "madden27.db"
    monkeypatch.setenv("CFB_COACH_DB", str(coach))
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", str(madden))

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


def _write_model(folder: Path, strata: dict, *, model_version: str = "v0-test", scored_on: str = "unreviewed") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_params.json").write_text(
        json.dumps({"strata": strata}),
        encoding="utf-8",
    )
    (folder / "manifest.json").write_text(
        json.dumps({
            "model_version": model_version,
            "label_version": "labels-ignored-when-model-version-set",
            "scored_on": scored_on,
            "training_report": {"tags": {"scored_on": scored_on}},
        }),
        encoding="utf-8",
    )
    return folder


def _flat_stratum() -> dict:
    return {"model": {"intercept": 0.0, "coef": {"call": {"ZZZ Illegal": 1000.0}}}}


def test_offline_hook_raises_unless_offline_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    model = _write_model(tmp_path / "model", {"madden27/cpu": _flat_stratum()})
    with pytest.raises(RuntimeError, match="offline_ok=True"):
        score_offline_model(
            "1&10", side="offense", model_dir=model, strata="madden27/cpu", seed=1,
        )
    with pytest.raises(RuntimeError, match="offline_ok=True"):
        score_offline_model(
            "1&10",
            side="offense",
            model_dir=tmp_path / "missing",
            strata="cfb",
            offline_ok=False,
        )
    assert list(home.rglob("*")) == []


def test_offline_hook_refuses_non_madden_strata(tmp_path: Path) -> None:
    model = _write_model(
        tmp_path / "model",
        {
            "madden27/cpu": _flat_stratum(),
            "madden27/human": _flat_stratum(),
            "madden27/cpu/redzone": _flat_stratum(),
            "cfb": _flat_stratum(),
            "cfb/human": _flat_stratum(),
            "cfb27/human": _flat_stratum(),
        },
    )
    for strata in ("cfb", "cfb/human", "cfb27/human", "madden27/unknown", ["madden27/cpu", "cfb"]):
        with pytest.raises(ValueError, match="refused stratum"):
            score_offline_model(
                "1&10",
                side="offense",
                model_dir=model,
                strata=strata,
                offline_ok=True,
            )
    with pytest.raises(ValueError, match="refused stratum"):
        score_offline_model(
            "1&10", side="offense", model_dir=model, strata="*", offline_ok=True,
        )
    with pytest.raises(ValueError, match="missing stratum"):
        score_offline_model(
            "1&10", side="offense", model_dir=model, strata="madden27/human/missing", offline_ok=True,
        )


def test_offline_hook_ranks_only_the_installed_book(tmp_path: Path) -> None:
    model = _write_model(
        tmp_path / "model",
        {
            "madden27/cpu": _flat_stratum(),
            "madden27/human": _flat_stratum(),
            "cfb/human": {"model": {"intercept": 0.0, "coef": {"call": {"CFB Only": 9999.0}}}},
        },
        scored_on="unreviewed",
    )
    flat = score_offline_model(
        "1&10",
        side="offense",
        model_dir=model,
        strata=["madden27/cpu", "madden27/human"],
        opp_look=None,
        seed=3,
        offline_ok=True,
    )
    assert flat["legal"] is True
    assert flat["play"] != "ZZZ Illegal"
    assert flat["play"] != "CFB Only"
    assert flat["gate_bypassed"] == "offline_eval"
    assert flat["scored_on"] == "unreviewed"
    assert flat["model_version"] == "v0-test"
    assert flat["strata_used"] == ["madden27/cpu", "madden27/human"]
    assert flat["mode"] == "offline"
    assert flat["adjustment_id"] is None
    assert flat["budget_ms"] == 150
    assert flat["over_budget"] is False
    assert set(flat["latency_ms"]) >= {"whole_call", "model"}
    assert flat["heuristic"]["mode"] == "heuristic"
    assert flat["heuristic"]["formation"]
    assert flat["heuristic"]["play"]
    for key in _SIDE:
        assert key in flat
        assert key in flat["heuristic"]

    boosted = _write_model(
        tmp_path / "boosted",
        {
            "madden27/cpu": {
                "model": {
                    "intercept": 0.0,
                    "coef": {
                        "call": {
                            "ZZZ Illegal": 5000.0,
                            flat["play"].lower(): -50.0,
                        },
                    },
                },
            },
        },
    )
    moved = score_offline_model(
        "1&10",
        side="offense",
        model_dir=boosted,
        strata="madden27/cpu",
        seed=3,
        offline_ok=True,
    )
    assert moved["legal"] is True
    assert moved["play"] != "ZZZ Illegal"
    assert (moved["formation"], moved["play"]) != (flat["formation"], flat["play"])
    assert moved["heuristic"] == flat["heuristic"]

    globed = _write_model(
        tmp_path / "glob",
        {
            "madden27/cpu/redzone": {
                "model": {
                    "intercept": 0.0,
                    "coef": {"call": {moved["play"].lower(): 4.0, "ZZZ Illegal": 80.0}},
                },
            },
            "cfb/human": _flat_stratum(),
        },
    )
    picked = score_offline_model(
        "1&10",
        side="offense",
        model_dir=globed,
        strata="madden27/cpu/*",
        seed=3,
        offline_ok=True,
    )
    assert picked["strata_used"] == ["madden27/cpu/redzone"]
    assert picked["legal"] is True
    assert picked["play"] != "ZZZ Illegal"
    assert "cfb" not in " ".join(picked["strata_used"])


def test_offline_hook_same_seed_reproduces_and_leaves_home_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    def refuse_live(*_args, **_kwargs):
        raise AssertionError("offline scoring reached the live gate")

    monkeypatch.setattr("cfb_coach.madden.model.inference.set_mode", refuse_live)
    monkeypatch.setattr("cfb_coach.madden.model.inference.rank_live", refuse_live)
    model = _write_model(tmp_path / "model", {"madden27/human": _flat_stratum()}, model_version="fixture-1")
    kwargs = {
        "side": "offense",
        "model_dir": model,
        "strata": "madden27/human",
        "opp_look": "empty",
        "seed": 11,
        "offline_ok": True,
    }
    first = score_offline_model("2&7", **kwargs)
    second = score_offline_model("2&7", **kwargs)
    assert first["formation"] == second["formation"]
    assert first["play"] == second["play"]
    assert first["line"] == second["line"]
    assert first["heuristic"] == second["heuristic"]
    assert first["opp_look"] == "empty"
    assert first["seed"] == 11
    assert list(home.rglob("*")) == []


def test_live_paths_do_not_import_the_offline_hook() -> None:
    root = Path(__file__).resolve().parents[2] / "cfb_coach"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "score_offline_model" not in text
        assert "eval_harness" not in text
    live = run_situation("1&10", mode="hybrid", opp_look=None, side="offense", seed=1)
    assert "gate_bypassed" not in live
    assert live["fallback_reason"] == "circuit_open"
