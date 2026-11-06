"""Coverage distinction regression tests — every test checks knowledge, not win rate."""
import json
import sqlite3
from pathlib import Path
import pytest
from cfb_coach.madden.model.coverage_intelligence import (
    SCHEMES,MATCH_CHECKS,classify_defensive_play,coverage_read,
)
from scripts.madden_ml.build_installed_semantics import build, classify_offense

@pytest.mark.parametrize("name,expected",[
    ("Cover 4 Palms","cover_4_palms"),
    ("Cover 4 Quarters","cover_4_quarters"),
    ("Cover 4 Drop Field","cover_4_drop"),
    ("Cover 1 Hole","cover_1_hole"),
    ("1 Hole Press","cover_1_hole"),
    ("Cover 1 Robber Press","cover_1_robber"),
    ("1 Robber Press","cover_1_robber"),
    ("Cover 1 Contain","cover_1_contain"),
    ("Cover 2 Man","cover_2_man"),
    ("Cover 2 Invert","cover_2_invert"),
    ("Tampa 2","tampa_2"),
    ("Cover 3 Match","cover_3_match"),
    ("Cover 3 Buzz Match","cover_3_match"),
    ("Cov 3 Buzz Match","cover_3_match"),
    ("Cover 3 Sky","cover_3_spot"),
    ("Cover 6 Willie","cover_6"),
    ("Cover 9 Show 2","cover_9"),
    ("Nickel 2 Trap","trap_2"),
    ("Cover 3 Lock","unknown"),
    ("Overload 3 Seam","unknown"),
    ("Fire Zone 3","unknown"),
    ("Cover 4 Press","unknown"),
])
def test_coverage_family_distinction(name,expected):
    v=classify_defensive_play(name)
    assert v.scheme_id==expected
    assert not v.allowed_as_verified_player_duties

def test_match_is_not_spot_drop():
    assert SCHEMES["cover_4_quarters"].family=="pattern_match"
    assert SCHEMES["cover_4_palms"].family=="pattern_match"
    assert SCHEMES["cover_4_drop"].family=="spot_zone"
    assert SCHEMES["cover_3_match"].family=="pattern_match"
    assert SCHEMES["cover_3_spot"].family=="spot_zone"

def test_palms_vs_quarters_specific_release():
    a=coverage_read("cover_4_palms",{"release":"receiver_2_quick_out_or_flat"})
    b=coverage_read("cover_4_quarters",{"release":"receiver_2_quick_out_or_flat"})
    assert len(a["triggered_rules"])==1
    assert len(b["triggered_rules"])==0
    assert a["specific_defender_assignments"].startswith("unknown")

def test_zone_it_overrides_only_when_confirmed():
    assert coverage_read("cover_4_quarters",{})["effective_family"]=="pattern_match"
    assert coverage_read("cover_4_quarters",{"verified_match_check":"zone_it"})["effective_family"]=="spot_zone"
    assert coverage_read("cover_2_man",{"verified_match_check":"zone_it"})["effective_family"]=="man"

def test_split_field_orientation_and_hole_robber():
    assert SCHEMES["cover_6"].quarter_side != SCHEMES["cover_9"].quarter_side
    assert SCHEMES["cover_1_hole"].bonus_defender != SCHEMES["cover_1_robber"].bonus_defender
    assert "stubbie" in MATCH_CHECKS and "solo_cut" in MATCH_CHECKS

def test_offense_name_tags_never_imply_player_routes():
    assert "mesh" in classify_offense("Mesh Post")
    assert "read_option" in classify_offense("Read Option")
    assert "mesh" not in classify_offense("HB Dive")

def test_build_216_with_separate_rules(tmp_path):
    snap=Path(__file__).resolve().parents[1]/"cfb_coach/data/madden27/installed_playbook_snapshot.json"
    output=tmp_path/"semantics.sqlite"
    r=build(snap,output)
    assert r["catalog_rows"]=={"offense":93,"defense":123}
    assert r["complete_player_assigned_plays"]==0
    with sqlite3.connect(output) as conn:
        assert conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]==216
        assert conn.execute("SELECT COUNT(*) FROM plays WHERE side='defense' AND scheme_id='cover_4_palms'").fetchone()[0]>0
        assert conn.execute("SELECT COUNT(*) FROM plays WHERE verified_player_duties > 0").fetchone()[0]==0
        assert conn.execute("SELECT COUNT(*) FROM match_checks").fetchone()[0]==len(MATCH_CHECKS)
    assert output.with_suffix(".review.csv").exists()
    with pytest.raises(FileExistsError):build(snap,output)

def test_all_research_features_are_unverified(tmp_path):
    snap=Path(__file__).resolve().parents[1]/"cfb_coach/data/madden27/installed_playbook_snapshot.json"
    out=tmp_path/"research.sqlite"
    build(snap,out)
    lines=out.with_suffix(".research_features.jsonl").read_text().splitlines()
    assert len(lines)==216
    for line in lines:
        obj=json.loads(line)
        assert obj["interpretation"]=="football_prior_only_not_observed_player_assignments"
        assert obj["evidence"].startswith("name_inferred_")