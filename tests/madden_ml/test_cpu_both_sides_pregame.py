"""Two-sided CPU pregame must never confuse proposals with physical installation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.madden_ml import cpu_both_sides_pregame as prep


def test_model_generated_both_sides_proposals_are_not_installed(tmp_path, monkeypatch):
    from cfb_coach.madden.model import offense_designer, defense_v2_playbook, defense_v2_pilot
    from cfb_coach.madden import catalog, playbook

    artifact = tmp_path / "offense_learned.json"
    artifact.write_text("synthetic-offense-artifact")
    monkeypatch.setattr(offense_designer, "load_design_artifact",
                        lambda path: SimpleNamespace(model_version="learned_test"))
    monkeypatch.setattr(offense_designer, "design_offense",
                        lambda db, **kw: {"book": {"formations": {
                            "Gun Trips": ["Mesh", "Inside Zone"]},
                            "formation_sources": {"Gun Trips": "Bears"}}})
    monkeypatch.setattr(defense_v2_pilot, "verified_rows",
                        lambda conn, franchise: ([], {"admitted_verified": 0}))
    monkeypatch.setattr(catalog, "book_names", lambda side: ["Ravens"])
    monkeypatch.setattr(catalog, "book_formations", lambda side, name: {
        "Nickel Over": ["Cover 3 Sky"]})
    monkeypatch.setattr(defense_v2_playbook, "propose",
                        lambda **kw: {"formations": {"Nickel Over": ["Cover 3 Sky"]},
                                      "formation_sources": {"Nickel Over": "Ravens"},
                                      "n_formations": 1, "n_plays": 1,
                                      "installed": False})
    monkeypatch.setattr(playbook, "load_books", lambda db: {})

    db_path = tmp_path / "test.db"
    db_path.write_bytes(b"placeholder")
    db = SimpleNamespace(path=db_path, conn=object())
    dest = tmp_path / "plans"
    report = prep.generate(db=db, franchise="test", opponent="cpu",
                           offense_artifact=artifact,
                           outdir=dest, allow_cold_start=True)
    assert report["status"] == "PREGAME_PROPOSALS_NOT_INSTALLED"
    assert report["ready_to_kickoff"] is False
    assert report["defensive_prior_only"] is True
    assert report["offense_proposal"]["formations"] == 1
    assert report["defense_proposal"]["formations"] == 1
    assert report["stored_book_audit"]["offense"]["ready_for_first_live_test"] is False
    assert report["stored_book_audit"]["defense"]["ready_for_first_live_test"] is False
    assert json.loads((dest / "defense_cold_start_model.json").read_text())["n_verified"] == 0
    assert (dest / "offense_proposal.json").is_file()
    assert (dest / "cpu_both_sides_pregame_report.json").is_file()


def test_cannot_generate_without_actual_offense_model(tmp_path):
    db = SimpleNamespace(path=tmp_path / "game.db", conn=object())
    with pytest.raises(FileNotFoundError):
        prep.generate(db=db, franchise="test", opponent="cpu",
                      offense_artifact=tmp_path / "missing_model.json",
                      outdir=tmp_path / "plans", allow_cold_start=True)


def test_audit_requires_locked_exact_installed_formations(monkeypatch):
    from cfb_coach.madden import playbook
    monkeypatch.setattr(playbook, "load_books", lambda db: {
        "offense": {"locked_ts": "today",
                    "formations": {"Gun Trips": ["Mesh"]}},
        "defense": {"locked_ts": "today",
                    "formations": {"Nickel Over": ["Cover 3 Sky"]}},
    })
    offense = {"book": {"formations": {"Gun Trips": ["Mesh"]}}}
    defense = {"formations": {"Nickel Over": ["Cover 3 Sky"]}}
    result = prep.audit(db=object(), offense=offense, defense=defense)
    assert result["offense"]["ready_for_first_live_test"]
    assert result["defense"]["ready_for_first_live_test"]
    assert result["offense"]["physical_editor_confirmation_still_required"]

    defense["formations"]["Nickel Over"].append("Cover 2 Man")
    assert prep.audit(db=object(), offense=offense,
                      defense=defense)["defense"]["ready_for_first_live_test"] is False
