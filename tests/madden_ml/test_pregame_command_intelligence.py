"""Pregame command opt-in, empirical loading, and bounded draft packages."""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.madden import catalog
from cfb_coach.madden.model import cli, experimental_model as em
from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model import offense_structural_model as sm

HISTORY = "cfb_coach.madden.model.historical_coverage_mix"


def args_for(command, *flags):
    parser = argparse.ArgumentParser()
    cli.build_ml_subparser(parser.add_subparsers())
    extras = ["--text"] if command == "offense-design" else ["--full"]
    return parser.parse_args(["ml", command, *extras, *flags])


@pytest.fixture
def db(tmp_path):
    connection = CoachDB(tmp_path / "preview.db")
    yield connection
    connection.close()


@pytest.fixture
def structural_path(tmp_path):
    path = tmp_path / "structural.json"
    sm.save_artifact(sm.StructuralArtifact(base=em.ExperimentalArtifact(
        evidence_quality="empirical", global_rate=.67).to_dict()), path)
    return path


@pytest.mark.parametrize("command", ["offense-design", "offense-coordinator"])
@pytest.mark.parametrize("control", ["off", "empty", "file", "history"])
def test_real_commands_forward_explicit_options(command, control, db, tmp_path, monkeypatch, capsys):
    mix = {"mix": {"cover_3_match": 1.0}, "mix_source": "opponent_history", "unknown_mass": .2}
    flags = []
    extractor = Mock(return_value=mix)
    monkeypatch.setitem(sys.modules, HISTORY, None)
    if control != "off":
        flags.append("--football-intelligence")
    if control == "empty":
        flags.append("--empty-coverage-mix")
    elif control == "file":
        path = tmp_path / "mix.json"
        path.write_text(json.dumps(mix))
        flags += ["--coverage-mix", str(path)]
    elif control == "history":
        module = ModuleType(HISTORY)
        module.extract_opponent_history_mix = extractor
        monkeypatch.setitem(sys.modules, HISTORY, module)
    args = args_for(command, *flags)
    sentinel = "CFB_COACH_FOOTBALL_PRIORS"
    # Both reads and writes would fail, even if a caller had set the live flag.
    monkeypatch.setenv(sentinel, "1")
    original_get = type(os.environ).__getitem__
    original_set = type(os.environ).__setitem__
    def guarded_get(self, key):
        assert key != sentinel, "pregame command read the live opt-in"
        return original_get(self, key)
    def guarded_set(self, key, value):
        assert key != sentinel, "pregame command changed the live opt-in"
        return original_set(self, key, value)
    monkeypatch.setattr(type(os.environ), "__getitem__", guarded_get)
    monkeypatch.setattr(type(os.environ), "__setitem__", guarded_set)
    captured = {}
    def design(opened, **kwargs):
        assert opened is db
        captured.update(kwargs)
        return {"football_intelligence": kwargs}
    monkeypatch.setattr(cli, "open_madden_db", lambda **kwargs: db)
    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(designer, "design_offense", design)
    monkeypatch.setattr(designer, "plan_inspection", lambda plan: plan)
    assert args.func(args) == 0
    assert captured["priors_on"] is (control != "off")
    assert captured["coverage_mix"] == (mix["mix"] if control in ("file", "history") else {})
    assert captured["mix_source"] == ({"off": "unknown", "empty": "empty_control"}.get(control, "opponent_history"))
    assert captured["unknown_mass"] == (.2 if control in ("file", "history") else 1.)
    assert not captured["skip_adjustment_design"]
    if control == "history":
        extractor.assert_called_once_with(db.conn, opponent_id="cpu")
    else:
        extractor.assert_not_called()
    assert "football_intelligence" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["offense-design", "offense-coordinator"])
@pytest.mark.parametrize("flags,match", [
    (["--empty-coverage-mix"], "require --football-intelligence"),
    (["--coverage-mix", "absent.json"], "require --football-intelligence"),
    (["--football-intelligence", "--coverage-mix", "absent.json", "--empty-coverage-mix"], "mutually exclusive"),
    (["--football-intelligence"], "Historical coverage extractor is missing"),
])
def test_invalid_mix_flags_and_missing_history_fail_clearly(command, flags, match, db, monkeypatch):
    monkeypatch.setitem(sys.modules, HISTORY, None)
    with pytest.raises(ValueError, match=match):
        cli._pregame_design_options(args_for(command, *flags), db)


@pytest.mark.parametrize("command", ["offense-design", "offense-coordinator"])
def test_command_loads_nested_base_and_skip_flag(command, structural_path, db):
    opts = cli._pregame_design_options(args_for(command, "--artifact", str(structural_path),
                                              "--skip-adjustment-design"), db)
    assert opts["skip_adjustment_design"]
    assert opts["artifact"].evidence_quality == "empirical"
    assert opts["artifact"].global_rate == .67
    assert opts["artifact"]._pregame_artifact_loader == "structural_nested_base"


def test_shared_loader_preserves_empirical_base_and_contrasts_hollow_load(structural_path, db, monkeypatch):
    monkeypatch.setattr(designer, "resolve_artifact_path", lambda db: structural_path)
    loaded = designer._load_artifact(db)
    assert loaded.evidence_quality == "empirical" and loaded.global_rate == .67
    assert loaded._pregame_artifact_loader == "structural_nested_base"
    hollow = em.load_artifact(structural_path)
    assert hollow.evidence_quality == "prior_driven" and hollow.global_rate == .5
    structural_path.write_text(json.dumps(em.ExperimentalArtifact(global_rate=.61).to_dict()))
    assert designer._load_artifact(db)._pregame_artifact_loader == "experimental"
    assert designer._load_artifact(db).global_rate == .61
    monkeypatch.setattr(designer, "resolve_artifact_path", lambda db: None)
    assert designer._load_artifact(db)._pregame_artifact_loader == "prior_only"


def test_play_rankings_forwards_intelligence(monkeypatch):
    rate = Mock(return_value=.6)
    monkeypatch.setattr(designer, "_rate", rate)
    mix = {"cover_1_robber": 1.}
    rows = designer._play_rankings(None, "Buccaneers", "Gun Bunch", ["Mesh"], "cpu", {},
                                    priors_on=True, coverage_mix=mix)
    assert rows[0]["play"] == "Mesh"
    rate.assert_called_once_with(None, "Gun Bunch", "Mesh", "cpu", priors_on=True, coverage_mix=mix)


def test_auto_selector_scores_only_exact_stock_passes_and_forwards_intelligence(monkeypatch):
    formations = {"Gun Bunch": ["Mesh", "Flood", "Inside Zone", "RPO Read", "Invented"],
                  "Gun Trips": ["Slants", "HB Draw"], "Run Only": ["HB Dive"]}
    stock = {**formations, "Gun Bunch": formations["Gun Bunch"][:-1]}
    books = {"Buccaneers": stock, "Lions": {"Gun Bunch": ["Invented"]}}
    monkeypatch.setattr(catalog, "book_formations", lambda side, source: books[source])
    mix = {"cover_1_robber": 1.}
    calls = []
    def rate(art, form, play, opponent, **kwargs):
        calls.append((form, play, kwargs))
        return .8 if play == "Flood" else .6
    monkeypatch.setattr(designer, "_rate", rate)
    assert designer._auto_adjustment_pairs(None, formations, dict.fromkeys(formations, "Buccaneers"),
        "cpu", priors_on=True, coverage_mix=mix) == [("Gun Bunch", "Flood"), ("Gun Trips", "Slants")]
    assert {(f, p) for f, p, _ in calls} == {("Gun Bunch", "Flood"), ("Gun Bunch", "Mesh"), ("Gun Trips", "Slants")}
    assert all(k == {"priors_on": True, "coverage_mix": mix} for _, _, k in calls)


@pytest.mark.parametrize("explicit,skip", [(False, False), (True, False), (False, True), (True, True)])
def test_proposal_generates_draft_package_or_explicit_override(explicit, skip, db, structural_path, monkeypatch):
    stock = catalog.book_formations("offense", "Buccaneers")
    form = "Gun Trips X Nasty"
    plays = stock[form]
    pair = (form, next(p for p in plays if not catalog.is_run(p) and "rpo" not in p.lower()))
    mix = {"cover_3_match": 1.}
    if explicit:
        monkeypatch.setattr(designer, "_auto_adjustment_pairs", Mock(side_effect=AssertionError("Explicit pairs must win")))
    plan = designer.design_offense(db, max_formations=1, artifact=designer.load_design_artifact(structural_path),
        catalogue={"Buccaneers": {form: plays}}, priors_on=True, coverage_mix=mix,
        mix_source="opponent_history", unknown_mass=.2,
        adjustment_pairs=[pair] if explicit else (), skip_adjustment_design=skip)
    assert plan["football_intelligence"] == {"priors_on": True, "mix": mix,
        "mix_source": "opponent_history", "unknown_mass": .2, "artifact_loader": "structural_nested_base"}
    designs = plan["adjustment_designs"]
    assert len(designs) == (0 if skip and not explicit else 1)
    assert plan["preparation_guide"]["adjustment_designs"] == designs
    for result in designs:
        assert result["play"] in stock[result["formation"]]
        assert not catalog.is_run(result["play"]) and "rpo" not in result["play"].lower()
        if explicit:
            assert (result["formation"], result["play"]) == pair
        else:
            expected = designer._auto_adjustment_pairs(designer.load_design_artifact(structural_path),
                plan["book"]["formations"], plan["book"]["formation_sources"], "cpu",
                priors_on=True, coverage_mix=mix)
            assert [(result["formation"], result["play"])] == expected
        for candidate in result["candidates"]:
            if candidate.get("score"):
                assert candidate["score"]["mix_source"] == "opponent_history"
            assert not candidate["executable_now"]
            if candidate["kind"] != "none":
                assert "DRAFT" in candidate["status"] and candidate["activation"] == "NOT_ARMED"
                assert "UNKNOWN" in candidate["controls"]
        selected = result["selected"]
        if selected["kind"] != "none":
            assert selected["beneficial"] and selected["net_gain"] > .02
    assert designer.plan_inspection(plan)["football_intelligence"] == plan["football_intelligence"]


@pytest.mark.parametrize("command", ["offense-design", "offense-coordinator"])
@pytest.mark.parametrize("enabled", [False, True])
def test_command_preview_runs_real_designer_read_only(command, enabled, tmp_path, structural_path, monkeypatch, capsys):
    import hashlib
    path = tmp_path / "command.db"
    connection = CoachDB(path)
    connection.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    stock = catalog.book_formations("offense", "Buccaneers")
    original = designer.design_offense
    def limited_catalog(db, **kwargs):
        assert db.read_only
        return original(db, catalogue={"Buccaneers": {"Gun Trips X Nasty": stock["Gun Trips X Nasty"]}}, **kwargs)
    monkeypatch.setattr(designer, "design_offense", limited_catalog)
    monkeypatch.setitem(sys.modules, HISTORY, None)
    flags = ["--db", str(path), "--artifact", str(structural_path), "--max-formations", "1"]
    if enabled:
        flags += ["--football-intelligence", "--empty-coverage-mix"]
    args = args_for(command, *flags)
    assert args.func(args) == 0
    text = capsys.readouterr().out
    payload, _ = json.JSONDecoder().raw_decode(text)
    plan = payload["plan"] if command == "offense-coordinator" else payload
    assert plan["football_intelligence"] == {"priors_on": enabled, "mix": {},
        "mix_source": "empty_control" if enabled else "unknown", "unknown_mass": 1.,
        "artifact_loader": "structural_nested_base"}
    assert len(plan["adjustment_designs"]) == 1
    assert plan["preparation_guide"]["adjustment_designs"] == plan["adjustment_designs"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("payload", [[], {}, {"mix": []},
    {"mix": {"cover_3_match": -1}, "mix_source": "opponent_history", "unknown_mass": .1},
    {"mix": {}, "mix_source": "opponent_history", "unknown_mass": 2.},
    {"mix": {"cover_3_match": float("nan")}, "mix_source": "unknown", "unknown_mass": 1.}])
def test_invalid_coverage_payload_is_refused(payload, tmp_path, db):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Coverage"):
        cli._pregame_design_options(args_for("offense-design", "--football-intelligence", "--coverage-mix", str(path)), db)


def test_coordinator_compare_refuses_design_flags(capsys):
    args = args_for("offense-coordinator", "--compare", "--empty-coverage-mix")
    assert args.func(args) == 2
    assert "cannot be combined" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["offense-design", "offense-coordinator"])
def test_command_historical_default_uses_landed_extractor(command, tmp_path, structural_path, monkeypatch, capsys):
    pytest.importorskip(HISTORY)
    path = tmp_path / "history_preview.db"
    connection = CoachDB(path)
    connection.close()
    stock = catalog.book_formations("offense", "Buccaneers")
    original = designer.design_offense
    def limited_catalog(db, **kwargs):
        assert db.read_only
        return original(db, catalogue={"Buccaneers": {"Gun Trips X Nasty": stock["Gun Trips X Nasty"]}}, **kwargs)
    monkeypatch.setattr(designer, "design_offense", limited_catalog)
    args = args_for(command, "--db", str(path), "--artifact", str(structural_path),
                    "--max-formations", "1", "--football-intelligence")
    assert args.func(args) == 0
    payload, _ = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    plan = payload["plan"] if command == "offense-coordinator" else payload
    intelligence = plan["football_intelligence"]
    assert set(intelligence) == {"priors_on", "mix", "mix_source", "unknown_mass", "artifact_loader"}
    assert intelligence["priors_on"] and intelligence["mix_source"] == "opponent_history"
    assert intelligence["mix"] == {} and intelligence["unknown_mass"] == 0.
    assert len(plan["adjustment_designs"]) == 1


@pytest.mark.parametrize("command", ["offense-design", "offense-coordinator"])
def test_mix_flags_refused_before_database_open(command, monkeypatch, capsys):
    monkeypatch.setattr(cli, "open_madden_db", Mock(side_effect=AssertionError("Invalid flags opened a database")))
    args = args_for(command, "--empty-coverage-mix")
    assert args.func(args) == 2
    assert "require --football-intelligence" in capsys.readouterr().err


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("stored", [False, True])
def test_adjustment_comparison_receives_stored_selection_and_blueprints(
        explicit, stored, db, structural_path, monkeypatch):
    from cfb_coach.madden import macros, offense_macros
    from cfb_coach.madden.model import offense_adjustment_designer as adjustments

    name = "MATCH"
    assert offense_macros.known(name)
    if stored:
        macros.store_selection(db, "cpu", {"offense": [name], "defense": []})
    stock = catalog.book_formations("offense", "Buccaneers")
    form = "Gun Trips X Nasty"
    pair = (form, next(p for p in stock[form] if not catalog.is_run(p) and "rpo" not in p.lower()))
    captured = []
    original = adjustments.design_adjustments
    def capture(*args, **kwargs):
        captured.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(adjustments, "design_adjustments", capture)
    plan = designer.design_offense(
        db, opponent_id="cpu", max_formations=1,
        artifact=designer.load_design_artifact(structural_path),
        catalogue={"Buccaneers": {form: stock[form]}},
        adjustment_pairs=[pair] if explicit else (),
    )
    assert len(captured) == len(plan["adjustment_designs"]) == 1
    kwargs = captured[0]
    assert kwargs["proposed_macros"] is plan["macro_blueprints"]
    assert kwargs["proposed_macros"]
    assert kwargs["confirmations"] is None
    assert kwargs["macro_effects"] is None
    if stored:
        entries = kwargs["loadout"].entries
        assert len(entries) == 1 and entries[0].name == name
        assert entries[0].side == "offense" and entries[0].slot == ""
        assert entries[0].installed and entries[0].active
        assert not entries[0].confirmed_by_user
        assert kwargs["loadout"].usable("offense", name) is None
    else:
        assert kwargs["loadout"] is None
    candidates = plan["adjustment_designs"][0]["candidates"]
    kinds = {row["kind"] for row in candidates}
    assert "proposed_macro" in kinds
    assert ("existing_macro" in kinds) is stored
    for blueprint in plan["macro_blueprints"]:
        row = next(row for row in candidates
                   if row["kind"] == "proposed_macro" and row["label"] == blueprint["name"])
        assert "DRAFT" in row["status"] and row["activation"] == "NOT_ARMED"
        assert not row["executable_now"] and "UNKNOWN" in row["controls"]


def test_adjustment_comparison_uses_only_sourced_verified_effects(db, structural_path, monkeypatch):
    from cfb_coach.madden.model import offense_adjustment_designer as adjustments

    records = [
        {"name": "verified", "verified_armed": True,
         "changes": {"WR1": "Slant"}, "source": "fixture route attestation"},
        {"name": "opaque", "verified_armed": True, "settings": ["Slant"], "source": "fixture"},
        {"name": "unsourced", "verified_armed": True, "changes": {"WR1": "Slant"}},
        {"name": "unarmed", "verified_armed": False,
         "changes": {"WR1": "Slant"}, "source": "fixture"},
    ]
    db.set_meta(designer.META_APPROVED.format(opponent="cpu"), json.dumps({"macros": records}))
    original = adjustments.design_adjustments
    captured = []
    def capture(*args, **kwargs):
        captured.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(adjustments, "design_adjustments", capture)
    stock = catalog.book_formations("offense", "Buccaneers")
    form = "Gun Trips X Nasty"
    designer.design_offense(db, opponent_id="cpu", max_formations=1,
        artifact=designer.load_design_artifact(structural_path),
        catalogue={"Buccaneers": {form: stock[form]}})
    effects = captured[0]["macro_effects"]
    assert set(effects) == {"verified"}
    assert effects["verified"].changes == records[0]["changes"]
    assert effects["verified"].source == records[0]["source"]
    assert effects["verified"].verified
    assert captured[0]["confirmations"] is None
