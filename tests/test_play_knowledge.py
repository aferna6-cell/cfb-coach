"""Standalone research-schema tests; no live model, game database or Xbox required."""
import pytest

from cfb_coach.madden.model.play_knowledge import (
    AdjustmentCandidate, AssignmentChange, KnowledgeError, MacroDefinition,
    PlayKnowledgeDB, PlayerJob, apply_macro, choose_adjustment, defensive_user,
    offense_read, play_key, transform,
)


CATALOG = {"source_url": "https://example.org/catalog", "books": {
    "offense": {"Alpha": {"formations": {
        "Gun Trips": {"plays": ["Mesh", "Mesh", "Spacing"]},
        "Singleback Ace": {"plays": ["Inside Zone"]}}},
        "Beta": {"formations": {"Gun Trips": {"plays": ["Mesh"]}}}},
    "defense": {"D": {"formations": {"Nickel Over": {"plays": ["Cover 3 Sky"]}}}},
}}


def _offense_jobs():
    return [PlayerJob("WR1", "route", "curl", "source_documented", "annotator:diagram-1"),
            PlayerJob("WR2", "route", "streak", "source_documented", "annotator:diagram-1"),
            PlayerJob("QB", "block", "pass_protection", "annotated", "annotator:diagram-1")]


def _defense_jobs():
    return [PlayerJob("MLB", "zone", "hook_curl", "source_documented", "diagram-2"),
            PlayerJob("CB1", "man", "WR1", "source_documented", "diagram-2")]


def test_import_retains_source_book_identity_and_no_guessed_assignments(tmp_path):
    with PlayKnowledgeDB(tmp_path / "research.db") as db:
        counts = db.import_catalog(CATALOG)
        assert counts == {"offense": 4, "defense": 1}
        a = db.get(play_key("offense", "Alpha", "Gun Trips", "Mesh"))
        b = db.get(play_key("offense", "Beta", "Gun Trips", "Mesh"))
        assert a.key != b.key
        assert a.completeness == b.completeness == "catalog_only"
        assert a.jobs == ()
        assert db.coverage()["offense"]["catalogued"] == 4
        db.import_catalog(CATALOG)
        assert db.coverage()["offense"]["catalogued"] == 4


def test_offense_annotation_and_exact_one_route_edit(tmp_path):
    key = play_key("offense", "Alpha", "Gun Trips", "Mesh")
    with PlayKnowledgeDB(tmp_path / "research.db") as db:
        db.import_catalog(CATALOG)
        db.annotate(key, _offense_jobs(), read_order=["WR1", "WR2"], read_source="diagram-1")
        original = db.get(key)
        assert offense_read(original) == "READ: WR1 -> WR2"
        assert original.completeness == "partial"
        updated = transform(original, [AssignmentChange("hot_route", "WR1", "route", "drag",
                  "off.hot_route", True, "user-test-1")], eligible_targets=["WR1", "WR2"])
        assert updated.jobs_for("WR1")[0].task == "drag"
        assert updated.jobs_for("WR2")[0].task == "streak"
        assert offense_read(updated) == "READ UNKNOWN"
        assert db.get(key).jobs_for("WR1")[0].task == "curl"  # read-only transform


def test_unverified_or_unknown_edits_rejected(tmp_path):
    key = play_key("offense", "Alpha", "Gun Trips", "Mesh")
    with PlayKnowledgeDB(tmp_path / "r.db") as db:
        db.import_catalog(CATALOG)
        db.annotate(key, _offense_jobs())
        original = db.get(key)
        for target, verified in [("WR4", True), ("WR1", False)]:
            with pytest.raises(KnowledgeError):
                transform(original, [AssignmentChange("hot_route", target, "route", "drag",
                          "off.hot_route", verified, "user-test")], eligible_targets=["WR1", "WR2", "WR4"])
        with pytest.raises(KnowledgeError):
            transform(original, [AssignmentChange("hot_route", "WR1", "route", "drag", "off.hot_route", True, "source"),
                                 AssignmentChange("hot_route", "WR1", "route", "slant", "off.hot_route", True, "source")],
                      eligible_targets=["WR1", "WR2"])


def test_macro_must_be_installed_and_have_confirmed_effects(tmp_path):
    key = play_key("offense", "Alpha", "Gun Trips", "Mesh")
    with PlayKnowledgeDB(tmp_path / "r.db") as db:
        db.import_catalog(CATALOG)
        db.annotate(key, _offense_jobs())
        origin = db.get(key)
        change = AssignmentChange("hot_route", "WR1", "route", "slant", "off.hot_route", True, "verified macro")
        pending = MacroDefinition("Beat Man", "offense", "LB 2", True, False, True, (change,), "user")
        with pytest.raises(KnowledgeError):
            apply_macro(origin, pending, eligible_targets=["WR1"])
        ready = MacroDefinition("Beat Man", "offense", "LB 2", True, True, True, (change,), "user")
        assert apply_macro(origin, ready, eligible_targets=["WR1"]).jobs_for("WR1")[0].task == "slant"
        with pytest.raises(KnowledgeError):
            apply_macro(origin, MacroDefinition("Unknown", "offense", "LB 2", True, True, True, (), "user"),
                        eligible_targets=["WR1"])


def test_defensive_assignment_and_user_advice(tmp_path):
    key = play_key("defense", "D", "Nickel Over", "Cover 3 Sky")
    with PlayKnowledgeDB(tmp_path / "r.db") as db:
        db.import_catalog(CATALOG)
        db.annotate(key, _defense_jobs())
        original = db.get(key)
        assert defensive_user(original, "MLB") == "USER MLB: zone hook_curl"
        assert defensive_user(original, "FS") == "USER RESPONSIBILITY UNKNOWN"
        updated = transform(original, [AssignmentChange("defender_assignment", "MLB", "zone", "flat",
                  "def.hot_route", True, "user-test")], eligible_targets=["MLB", "FS"])
        assert defensive_user(updated, "MLB") == "USER MLB: zone flat"


def test_decisive_no_adjustment_or_verified_winner_only():
    options = [AdjustmentCandidate("WR1 -> Drag", .65, True, True, "known"),
               AdjustmentCandidate("speculative Cover 2 Beater", .95, False, True, "prior"),
               AdjustmentCandidate("illegal macro", .99, True, False, "prior")]
    assert choose_adjustment(.60, options) == "WR1 -> Drag"
    assert choose_adjustment(.66, options) == "NO ADJUSTMENT"
    assert choose_adjustment(.10, [AdjustmentCandidate("maybe a slant", .90, False, True, "")]) == "NO ADJUSTMENT"


def test_reject_missing_source_unknown_taxonomy_and_unsourced_reads(tmp_path):
    key = play_key("offense", "Alpha", "Gun Trips", "Mesh")
    with PlayKnowledgeDB(tmp_path / "r.db") as db:
        db.import_catalog(CATALOG)
        with pytest.raises(KnowledgeError):
            db.annotate(key, [PlayerJob("WR1", "route", "laser_beam", "source_documented", "diagram")])
        with pytest.raises(KnowledgeError):
            db.annotate(key, _offense_jobs(), read_order=["WR3"], read_source="diagram")
        with pytest.raises(KnowledgeError):
            db.annotate(key, _offense_jobs(), read_order=["WR1"])