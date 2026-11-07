"""Issue #127: staged catalogue semantics never upgrades names/art/proposals to verified; human review is the only path; e2e per side."""
from __future__ import annotations

import csv
import json

import pytest

from cfb_coach.madden.model import catalog_semantics as cs

SMASH_KEY = "madden27|offense|gun_doubles_clamp_stack|mtn_shuffle_verts_smash"
SLANTS_KEY = "madden27|offense|pistol_trips|slants"


def _rev(key, slot, decision="confirm", ev="user_screenshot:own_capture_001.png", **kw):
    return {"play_key": key, "slot": slot, "decision": decision, "reviewer": "aidan", "reviewed_at": "2026-11-07", "evidence_ref": ev, **kw}


def test_committed_state_is_honest_zero_verified():
    o, d = cs.offense_records(reviews=[]), cs.defense_records()
    fo, fd = cs.funnel(o), cs.funnel(d)
    assert fo["indexed"] == 93 and fd["indexed"] == sum(len(v) for v in cs.installed_snapshot()["defense"]["formations"].values())
    for f in (fo, fd):
        assert f["assignment_verified"] == f["concept_verified"] == f["model_feature_eligible"] == f["downloaded"] == 0
    assert fo["geometry_extracted"] == 0 and fo["occupancy_features_only"] == 93   # occupancy is not route geometry
    assert all(r["defenders"] == "unknown" and r["coverage_verified"] == "unknown" for r in d)
    assert all(not row["eligible"] and not row["features"] for row in cs.export(reviews=[])["feature_rows"])


def test_third_party_documentation_is_a_proposal_not_verification():
    rec = next(r for r in cs.offense_records(reviews=[]) if r["key"] == SMASH_KEY)
    assert rec["slots"]["WR1"]["tier"] == "third_party_documented" and rec["slots"]["WR1"]["assignment"] == "unknown"
    assert rec["concepts_verified"] == [] and rec["stage"] == "source_image"
    assert rec["membership"] == {"owner_installed_book": True, "second_source_corroborated": True}


@pytest.mark.parametrize("bad", [
    {"evidence_ref": "https://acemadden.com/play-art/x.png"},     # art URL
    {"evidence_ref": "extractor:route_polyline_0.93"},             # unreviewed geometry
    {"evidence_ref": "name:Slants"},                               # play name
    {"reviewer": "extractor"},
    {"decision": "correct"},                                       # correction without an assignment
    {"reviewer": ""},
])
def test_reviews_that_cannot_certify_are_rejected(bad):
    with pytest.raises(cs.ReviewError):
        cs.validate_review({**_rev(SLANTS_KEY, "WR1"), **bad})


def test_offense_end_to_end_listing_to_feature_row():
    revs = [_rev(SLANTS_KEY, s) for s in ("WR1", "WR2", "WR3", "TE1")]
    rec = next(r for r in cs.offense_records(reviews=revs) if r["key"] == SLANTS_KEY)
    assert rec["source_image"]["sha256"] and rec["membership"]["owner_installed_book"]
    assert {c["concept"] for c in rec["concepts_verified"]} >= {"slants"} and rec["stage"] == "model_feature_eligible"
    row = next(x for x in cs.export(reviews=revs)["feature_rows"] if x["key"] == SLANTS_KEY)
    assert row["eligible"] and row["features"]["concept_slants"] == 1.0 and row["features"]["route_slant"] == 2.0
    assert all(s["reviews"][0]["evidence_ref"].startswith("user_screenshot:") for n, s in rec["slots"].items() if s["reviews"])


def test_disagreement_blocks_eligibility_and_is_kept():
    revs = [_rev(SLANTS_KEY, s) for s in ("WR1", "WR2", "WR3")] + [_rev(SLANTS_KEY, "WR1", "correct", assignment="hitch", reviewer="second")]
    rec = next(r for r in cs.offense_records(reviews=revs) if r["key"] == SLANTS_KEY)
    assert rec["slots"]["WR1"]["disagreement"] and rec["slots"]["WR1"]["tier"] not in cs.VERIFIED_SLOT
    assert rec["stage"] != "model_feature_eligible"


def test_correction_replaces_proposal_and_can_remove_a_concept():
    revs = [_rev(SMASH_KEY, "WR1", "correct", assignment="streak"), _rev(SMASH_KEY, "SLWR"), _rev(SMASH_KEY, "HB1")]
    rec = next(r for r in cs.offense_records(reviews=revs) if r["key"] == SMASH_KEY)
    assert rec["slots"]["WR1"]["assignment"] == "streak" and rec["slots"]["WR1"]["corrected_from"] == "corner"
    assert "smash" not in {c["concept"] for c in rec["concepts_verified"]}


def test_ambiguous_partial_review_never_invents_a_concept():
    # one reviewed corner and nothing else: not smash, not flood, not anything
    rec = next(r for r in cs.offense_records(reviews=[_rev(SMASH_KEY, "WR1")]) if r["key"] == SMASH_KEY)
    assert rec["concepts_verified"] == [] and rec["stage"] == "assignment_reviewed"


def test_defense_end_to_end_stays_unknown_without_defender_evidence():
    d = cs.defense_records()
    c3m = [r for r in d if r["play"] == "Cover 3 Match"]
    assert c3m and all(r["coverage_proposed"]["family"] == "cover3" and r["coverage_verified"] == "unknown" for r in c3m)
    rows = {x["key"]: x for x in cs.export(reviews=[])["feature_rows"]}
    assert all(not rows[r["key"]]["eligible"] for r in d)
    q = [x for x in cs.review_queue(d) if x["slot"] == "coverage_shell"]
    assert len(q) == len(d) and all(x["proposed_tier"] == "name_inference" for x in q)


def test_review_csv_import_is_idempotent(tmp_path):
    f, out = tmp_path / "q.csv", tmp_path / "r.jsonl"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["play_key", "slot", "decision", "assignment", "reviewer", "reviewed_at", "evidence_ref", "note"])
        w.writeheader(); w.writerow(_rev(SLANTS_KEY, "WR1")); w.writerow({"play_key": SLANTS_KEY, "slot": "WR2"})
    assert cs.import_review_csv(f, out) == 1 and cs.import_review_csv(f, out) == 0
    assert len(cs.load_reviews(out)) == 1


def test_queue_puts_fast_confirmations_first_and_skips_linemen():
    q = cs.review_queue(cs.offense_records(reviews=[]))
    assert q[0]["priority"] == 1 and sum(x["priority"] == 1 for x in q) == 40
    assert not {x["slot"] for x in q} & {"C", "LG", "RG", "LT", "RT", "QB"}


def test_export_is_versioned_and_records_rights():
    e = cs.export(reviews=[])
    assert e["schema"] == "madden27.catalog_semantics.v1" and "NOT granted" in e["rights"]["acemadden_site"] and e["digest"]


def test_name_hint_accuracy_is_measured_not_assumed():
    acc = cs.name_hint_accuracy()
    assert acc["plays"] == 8 and acc["any_overlap"] < acc["plays"]   # names are weak evidence; this is why they never certify
