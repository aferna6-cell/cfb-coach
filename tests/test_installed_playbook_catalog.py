"""Installed-play inventory tests; use an immutable copy of the Nov 6 backup snapshot."""
import copy
import csv
import json
from pathlib import Path

import pytest

from cfb_coach.madden.model.installed_playbook_catalog import (
    annotation_queue, import_installed, installed_coverage, validate_snapshot, write_queue_csv,
)
from cfb_coach.madden.model.play_knowledge import KnowledgeError, PlayKnowledgeDB, play_key

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "cfb_coach" / "data" / "madden27"


@pytest.fixture
def snapshot():
    return json.loads((DATA / "installed_playbook_snapshot.json").read_text())


@pytest.fixture
def annotations():
    return json.loads((DATA / "installed_play_annotations.json").read_text())


def test_exact_installed_inventory_matches_recorded_fingerprints(snapshot):
    c = validate_snapshot(snapshot)
    assert c["offense"]["fingerprint"] == "OFF-3C6D-E543"
    assert c["defense"]["fingerprint"] == "DEF-F1CF-1862"
    assert [len(snapshot["sides"][x]["formations"]) for x in ("offense","defense")] == [5,6]
    assert [sum(map(len,snapshot["sides"][x]["formations"].values())) for x in ("offense","defense")] == [93,123]
    assert {v for v in snapshot["sides"]["offense"]["formation_sources"].values()} == {"Buccaneers"}
    assert set(snapshot["sides"]["defense"]["formation_sources"].values()) == {"49ers","Chargers"}


def test_inventory_fingerprint_catches_even_one_modified_play(snapshot):
    bad=copy.deepcopy(snapshot)
    bad["sides"]["offense"]["formations"]["Pistol Trips"][0]="not the actual play"
    with pytest.raises(KnowledgeError,match="fingerprint"):
        validate_snapshot(bad)
    bad=copy.deepcopy(snapshot)
    bad["sides"]["defense"]["formations"]["Nickel Over"].append("Tampa 2")
    with pytest.raises(KnowledgeError,match="duplicate"):
        validate_snapshot(bad)


def test_import_all_216_preserves_source_book_names_and_is_idempotent(snapshot,tmp_path):
    with PlayKnowledgeDB(tmp_path/"installed.db") as db:
        status=import_installed(db,snapshot)
        assert status["counts"]=={"offense":{"formations":5,"plays":93},"defense":{"formations":6,"plays":123}}
        assert status["console_reverified"] is False
        assert installed_coverage(db)["offense"]["catalogued"]==93
        assert installed_coverage(db)["defense"]["catalogued"]==123
        assert db.get(play_key("offense","Buccaneers","Gun Trips X Nasty","Mtn Stick Wheel")).completeness=="catalog_only"
        assert db.get(play_key("defense","Chargers","46 Normal","Cover 3 Match")).completeness=="catalog_only"
        assert db.get(play_key("defense","49ers","Nickel Over","Tampa 2")) is not None
        assert len(annotation_queue(db))==216
        import_installed(db,snapshot)
        assert len(annotation_queue(db))==216
        assert db.conn.execute("select count(*) from recorded_inventories").fetchone()[0]==2


def test_cross_check_catches_mismatches_without_erasing_recorded_play(snapshot,tmp_path):
    source={"books":{"offense":{"Buccaneers":{"formations":{ "Gun Trips X Nasty":{
        "url":"https://example.org/formation","plays":["Mtn Stick Wheel"]}}}},"defense":{}}}
    with PlayKnowledgeDB(tmp_path/"s.db") as db:
        result=import_installed(db,snapshot,full_catalog=source)
        checks=result["source_validation"]["offense"]
        assert checks["source_match"]==1
        assert checks["source_formation_not_found"]>0
        assert checks["source_play_not_found"]>0
        assert len(annotation_queue(db))==216
        r=next(x for x in annotation_queue(db) if x["play"]=="Hi Lo Cross")
        assert r["source_catalog_check"]=="source_play_not_found"


def test_sourced_annotations_are_subset_of_installed_and_not_xbox_verified(snapshot,annotations,tmp_path):
    from scripts.madden_ml.build_installed_play_knowledge import _insert_annotations
    with PlayKnowledgeDB(tmp_path/"s.db") as db:
        import_installed(db,snapshot)
        seeded=_insert_annotations(db,annotations)
        assert seeded=={"offense":8,"defense":3}
        coverage=installed_coverage(db)
        assert coverage["offense"]["partially_annotated"]==8
        assert coverage["defense"]["partially_annotated"]==3
        assert coverage["offense"]["all_11_verified"]==0
        assert coverage["defense"]["all_11_verified"]==0
        p=db.get(play_key("offense","Buccaneers","Gun Trips X Nasty","Mtn Stick Wheel"))
        assert p.completeness=="partial"
        assert p.jobs_for("SLWR")[0].task=="wheel"
        assert p.read_order==( "TE1","WR2","HB1","WR1","SLWR")
        assert all(j.evidence=="source_documented" for j in p.jobs)
        d=db.get(play_key("defense","49ers","Nickel Over","Tampa 2"))
        assert d.jobs_for("MLB")[0].task=="vertical_middle_seam"
        assert db.get(play_key("defense","49ers","Nickel Double Mug","Tampa 2")).completeness=="catalog_only"
        assert len(annotation_queue(db))==216


def test_export_csv_has_all_216_with_unknown_assignments(snapshot,tmp_path):
    with PlayKnowledgeDB(tmp_path/"s.db") as db:
        import_installed(db,snapshot)
        out=tmp_path/"review.csv"
        write_queue_csv(annotation_queue(db),out)
        with out.open(newline="") as handle:rows=list(csv.DictReader(handle))
        assert len(rows)==216
        assert all(r["status"]=="catalog_only" for r in rows)
        assert all(r["unrecorded_player_assignments"]=="11" for r in rows)


def test_malformed_snapshot_never_leaves_partial_import(snapshot,tmp_path):
    bad=copy.deepcopy(snapshot)
    bad["sides"]["offense"]["formation_sources"].pop("Pistol Trips")
    with PlayKnowledgeDB(tmp_path/"s.db") as db:
        with pytest.raises(KnowledgeError):import_installed(db,bad)
        assert db.conn.execute("select count(*) from plays").fetchone()[0]==0


def test_all_installed_names_match_checked_in_huddle_reference(snapshot,tmp_path):
    source=json.loads((DATA/"installed_source_catalog_reference.json").read_text())
    with PlayKnowledgeDB(tmp_path/"crosscheck.db") as db:
        out=import_installed(db,snapshot,full_catalog=source)
        assert out["source_validation"]=={"offense":{"source_match":93},"defense":{"source_match":123}}
        assert installed_coverage(db)["offense"]["source_matches"]==93
        assert installed_coverage(db)["defense"]["source_matches"]==123