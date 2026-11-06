"""Offense assignment database: identity, 11-slot roles, evidence tiers, taxonomy, transformations.

Synthetic data only where noted. Nothing here touches ~/.cfb-coach or a game DB.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import pytest

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_taxonomy as tx
from cfb_coach.madden.model.coverage_intelligence import MATCH_CHECKS, SCHEMES

SNAP = json.loads(oa.SNAPSHOT_PATH.read_text(encoding="utf-8"))["sides"]["offense"]


@pytest.fixture(scope="module")
def db():
    return oa.load_database()


# 1-4 ---------------------------------------------------------------- inventory identity

def test_exactly_93_installed_offensive_plays(db):
    assert len(db["plays"]) == 93 == sum(len(v) for v in SNAP["formations"].values())
    assert db["inventory_fingerprint"] == oa.FINGERPRINT == "OFF-3C6D-E543"


def test_formation_counts_and_source_book_mapping(db):
    want = {"Gun 5WR Tight": 15, "Gun Doubles Clamp Stack": 27, "Gun Trips X Nasty": 21,
            "Pistol Deuce Close": 12, "Pistol Trips": 18}
    assert dict(Counter(p["formation"] for p in db["plays"])) == want
    assert {p["book"] for p in db["plays"]} == {"Buccaneers"}
    for p in db["plays"]:
        assert SNAP["formation_sources"][p["formation"]] == p["book"]
        assert p["play"] in SNAP["formations"][p["formation"]]


def test_no_duplicate_or_mismatched_play_identity(db):
    ids = [p["play_id"] for p in db["plays"]]
    assert len(set(ids)) == 93
    assert len({(p["formation"], p["play"]) for p in db["plays"]}) == 93
    # same play name in different formations is a DIFFERENT record
    names = Counter(p["play"] for p in db["plays"])
    same = [n for n, c in names.items() if c > 1]
    assert "Inside Zone" in same and "HB Slip Screen" in same
    for n in same:
        assert len({p["play_id"] for p in db["plays"] if p["play"] == n}) == names[n]
    for p in db["plays"]:
        assert p["play_id"] == oa.play_id(p["formation"], p["play"], p["book"])


def test_data_file_matches_a_fresh_build(db):
    ann = json.loads(oa.ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    snap = json.loads(oa.SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert oa.build_database(snap, ann) == db


def test_every_play_has_eleven_valid_slots(db):
    for p in db["plays"]:
        slots = p["slots"]
        assert len(slots) == 11
        assert set(tx.FIXED_SLOTS) <= set(slots)               # QB + LT LG C RG RT always present
        assert len(p["personnel"]["skill_slots"]) == 5
        assert set(slots) == set(tx.FIXED_SLOTS) | set(p["personnel"]["skill_slots"])


def test_wrong_inventory_is_rejected(db):
    bad = copy.deepcopy(db)
    bad["plays"].pop()
    with pytest.raises(oa.AssignmentError):
        oa.validate(bad)
    dup = copy.deepcopy(db)
    dup["plays"][1] = copy.deepcopy(dup["plays"][0])
    with pytest.raises(oa.AssignmentError):
        oa.validate(dup)
    snap = json.loads(oa.SNAPSHOT_PATH.read_text(encoding="utf-8"))
    snap["sides"]["offense"]["inventory_fingerprint"] = "OFF-0000-0000"
    with pytest.raises(oa.AssignmentError):
        oa.build_database(snap, {"annotations": []})


# 10 ------------------------------------------------------------------ unknown never becomes verified

def test_unknown_stays_unknown_and_nothing_is_verified(db):
    stats = oa.coverage_stats(db)
    assert stats["fully_verified_plays"] == 0
    assert stats["slots_by_tier"].get("diagram_verified", 0) == 0
    assert stats["slots_by_tier"].get("in_game_verified", 0) == 0
    assert stats["partial_plays"] == 8 and stats["unknown_plays"] == 85
    for p in db["plays"]:
        for s in p["slots"].values():
            if s["tier"] == "unknown":
                assert s["task"] == "UNKNOWN" and s["category"] == "unknown"
        assert p["name_hint"]["tier"] == "football_inference"


def test_name_hints_never_create_player_assignments(db):
    mesh = next(p for p in db["plays"] if p["play"] == "Mesh" and p["formation"] == "Gun 5WR Tight")
    assert "mesh" in mesh["name_hint"]["concepts"]
    assert all(s["tier"] == "unknown" for s in mesh["slots"].values())
    assert pf.structure_from_record(mesh).n_known == 0


def test_validator_rejects_unknown_slot_with_a_task_and_sourceless_tier(db):
    bad = copy.deepcopy(db)
    bad["plays"][0]["slots"]["LT"]["task"] = "pass_set"
    with pytest.raises(oa.AssignmentError):
        oa.validate(bad)
    bad = copy.deepcopy(db)
    s = bad["plays"][0]["slots"]["LT"]
    s.update(tier="football_inference", category="block", task="pass_set", source="")
    with pytest.raises(oa.AssignmentError):
        oa.validate(bad)


def _overlay(play="Mesh", formation="Gun 5WR Tight", slot="LT", tier="diagram_verified",
             source="user_screenshot:mesh.png", category="block", task="pass_set", params=None):
    return {"schema": oa.OVERLAY_SCHEMA, "annotations": [
        {"formation": formation, "play": play, "slots": {slot: {
            "category": category, "task": task, "params": params or {}, "tier": tier, "source": source}}}]}


def test_overlay_verified_requires_the_users_own_evidence(db):
    merged, log = oa.merge_overlay(db, _overlay())
    p = oa.index(merged)[("Gun 5WR Tight", "Mesh")]
    assert p["slots"]["LT"]["tier"] == "diagram_verified"
    assert oa.play_status(p) == "partial"                        # one verified slot is not a verified play
    with pytest.raises(oa.AssignmentError):                      # third-party URL cannot be 'verified'
        oa.merge_overlay(db, _overlay(source="https://madden.tools/x"))
    with pytest.raises(oa.AssignmentError):
        oa.merge_overlay(db, _overlay(source=""))
    with pytest.raises(oa.AssignmentError):                      # not an installed play
        oa.merge_overlay(db, _overlay(play="Not A Play"))
    with pytest.raises(oa.AssignmentError):                      # slot that does not exist
        oa.merge_overlay(db, _overlay(slot="HB1"))


def test_weaker_evidence_never_replaces_stronger_and_conflicts_are_recorded(db):
    strong, _ = oa.merge_overlay(db, _overlay(category="route", task="slant", slot="SKILL_1",
                                              params={"side": "left"}))
    weak, log = oa.merge_overlay(strong, _overlay(category="route", task="post", slot="SKILL_1", tier="football_inference",
                                                  source="my guess", params={"side": "left"}))
    s = oa.index(weak)[("Gun 5WR Tight", "Mesh")]["slots"]["SKILL_1"]
    assert s["tier"] == "diagram_verified" and s["task"] == "slant"
    clash, log = oa.merge_overlay(strong, _overlay(category="route", task="post", slot="SKILL_1", params={"side": "left"}))
    assert oa.index(clash)[("Gun 5WR Tight", "Mesh")]["disputes"]
    assert any("DISPUTE" in x for x in log)


def test_all_eleven_verified_is_the_only_way_to_fully_verified(db):
    p = copy.deepcopy(oa.index(db)[("Gun 5WR Tight", "Mesh")])
    for n, s in p["slots"].items():
        s.update(tier="diagram_verified", category="block", task="pass_set", source="user_screenshot:x.png")
    assert oa.play_status(p) == "fully_verified"
    p["slots"]["QB"].update(tier="source_documented")
    assert oa.play_status(p) == "partial"


# 8-9 ----------------------------------------------------------------- blocking, protection, run/RPO/PA

def test_blocking_stays_unknown_and_is_reported_exactly(db):
    gaps = oa.audit_gaps(db)
    assert gaps["unknown_ol_slots"] == gaps["ol_slots_total"] == 465
    assert gaps["unknown_qb_slots"] == 93
    assert all(s["ol_known"] == 0 for s in [{"ol_known": pf.structure_from_record(p).ol_known} for p in db["plays"]])
    assert set(gaps["formations_with_unnamed_skill_slots"]) == {"Gun 5WR Tight", "Pistol Deuce Close"}


def test_run_rpo_option_screen_play_action_representation(db):
    by = {(p["formation"], p["play"]): p["name_hint"] for p in db["plays"]}
    assert by[("Gun Trips X Nasty", "Inside Zone")]["run_scheme"] == "inside_zone"
    assert by[("Pistol Trips", "Strong Power")]["run_scheme"] == "power"
    assert by[("Pistol Trips", "Pin Pull Sweep")]["run_scheme"] == "sweep"
    assert by[("Pistol Deuce Close", "Y Lead Read Option")]["play_type"] == "option"
    assert by[("Pistol Deuce Close", "Y Lead Read Option")]["run_scheme"] == "lead_read_option"
    assert by[("Gun Doubles Clamp Stack", "RPO Alert Out")]["play_type"] == "rpo"
    assert by[("Pistol Trips", "HB Slip Screen")]["play_type"] == "screen"
    assert by[("Pistol Trips", "PA Boot")]["play_type"] == "play_action"
    assert by[("Gun 5WR Tight", "QB Draw")]["run_scheme"] == "qb_draw"
    assert by[("Gun Doubles Clamp Stack", "Mtn Counter Y")]["motion"] is True
    assert by[("Gun Doubles Clamp Stack", "Counter Y")]["motion"] is False
    kinds = Counter(h["play_type"] for h in by.values())
    assert sum(kinds.values()) == 93 and kinds["unknown"] == 0


def test_block_and_release_job_is_kept_as_a_block_not_counted_as_a_route(db):
    p = oa.index(db)[("Pistol Trips", "Inside Cross")]
    hb = p["slots"]["HB1"]
    assert hb["category"] == "block" and hb["task"] == "block_and_release_hook"
    assert hb["tier"] == "source_documented" and "route" not in hb
    s = pf.structure_from_record(p)
    assert s.route_for("HB1") is None and s.n_known == 4          # HB's hook is a block job, not a stock route


# taxonomy ------------------------------------------------------------------------------------------

def test_hot_route_vocabulary_equals_the_registry():
    reg = ex.registry()["off.hot_route"]
    assert set(reg.params["route"]["options"]) == set(tx.HOT_ROUTE_OPTIONS)
    for opt in tx.HOT_ROUTE_OPTIONS:
        assert tx.ROUTE_FAMILIES[tx.HOT_ROUTE_TO_FAMILY[opt]].hot_route_option == opt


def test_taxonomy_covers_required_routes_and_every_sourced_route():
    need = {"slant", "drag", "streak", "seam", "post", "corner", "in", "out", "hitch", "curl", "comeback", "fade",
            "wheel", "flat", "angle", "pivot", "zig", "option", "texas", "swing", "screen"}
    assert need <= set(tx.ROUTE_FAMILIES)
    ann = json.loads(oa.ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    for a in ann["annotations"]:
        for j in a["jobs"]:
            if a["side"] == "offense" and j["kind"] == "route":
                assert tx.route_from_job(j["task"], j.get("params")).family in tx.ROUTE_FAMILIES


def test_five_yard_in_and_fifteen_yard_dig_share_a_family_but_are_not_identical():
    short = tx.route_from_job("in", {"side": "left", "published_route": "Short In"})
    dig = tx.route_from_job("in", {"side": "left", "depth": "deep", "published_route": "Deep In"})
    assert short.family == dig.family == "in" and short != dig
    assert (short.band, dig.band) == ("short", "deep") and dig.yards > short.yards


def test_route_combinations_are_interacting_not_a_list():
    post = tx.RouteSpec("post", "right", "deep")
    dig = tx.RouteSpec("in", "left", "deep")
    flat = tx.RouteSpec("flat", "left", "shallow")
    a, b = set(tx.detect_concepts([post, dig])), set(tx.detect_concepts([post, flat]))
    assert "post_dig" in a and "post_dig" not in b and a != b
    two_verts = [tx.RouteSpec("streak", s, "deep") for s in ("left", "right")]
    assert "four_verticals" not in tx.detect_concepts(two_verts)
    assert "four_verticals" in tx.detect_concepts([tx.RouteSpec(f, s, "deep") for f, s in
                                                   (("streak", "left"), ("streak", "right"), ("seam", "left"), ("seam", "right"))])


def test_mesh_detected_only_from_opposite_side_shallow_crossers(db):
    s = pf.structure_lookup("Gun Trips X Nasty", "Hi Lo Cross")
    assert "mesh" in s.concepts()
    same_side = [tx.RouteSpec("drag", "left", "shallow"), tx.RouteSpec("drag", "left", "shallow")]
    assert "mesh" not in tx.detect_concepts(same_side)


# transformations -----------------------------------------------------------------------------------

def test_hot_route_changes_exactly_one_route_and_clears_stock_read_order():
    s = pf.structure_lookup("Gun Trips X Nasty", "Hi Lo Cross")
    assert s.read_order
    m = pf.apply_hot_route(s, "WR1", "Slant")
    assert m.route_for("WR1").family == "slant" and m.route_for("WR1").side == s.route_for("WR1").side
    for slot in ("HB1", "SLWR", "WR2", "TE1"):
        assert m.route_for(slot) == s.route_for(slot)
    assert m.read_order == () and m.sourced_concepts == ()
    diff = pf.compare_structures(s, m)
    assert "mesh" in diff["lost_concepts"] and "slant_flat" in diff["gained_concepts"]
    assert s.route_for("WR1").family == "drag"                       # the stock structure is untouched


def test_hot_route_legality_unknown_original_bad_option_and_non_eligible_slot():
    unknown = pf.structure_lookup("Gun Trips X Nasty", "Mtn Triple Slants")
    with pytest.raises(pf.StructureError):
        pf.apply_hot_route(unknown, "WR1", "Slant")                  # original assignment unknown
    s = pf.structure_lookup("Gun Trips X Nasty", "Hi Lo Cross")
    with pytest.raises(pf.StructureError):
        pf.apply_hot_route(s, "WR1", "Wheel")                        # not a Madden hot-route option
    with pytest.raises(pf.StructureError):
        pf.apply_hot_route(s, "LT", "Slant")                         # lineman is not a receiver
    with pytest.raises(pf.StructureError):
        pf.apply_hot_route(s, "WR9", "Slant")
    assert "WR1" in pf.eligible_hot_route_targets(s) and "LT" not in pf.eligible_hot_route_targets(s)


def test_registry_rejects_a_non_eligible_hot_route_target():
    s = pf.structure_lookup("Gun Trips X Nasty", "Hi Lo Cross")
    ctx = ex.PlayContext("offense", s.formation, s.play, "pass", eligible_receivers=s.skill_slots)
    bad = ex.validate_plan([ex.PlannedAction("off.hot_route", {"target": "LT", "route": "Slant"})], ctx)
    assert any("not an eligible receiver" in e for e in bad)
    ok = ex.validate_plan([ex.PlannedAction("off.hot_route", {"target": "WR1", "route": "Slant"})], ctx)
    assert ok == []
    live = ex.validate_plan([ex.PlannedAction("off.hot_route", {"target": "WR1", "route": "Slant"})], ctx, require_live=True)
    assert live and "not yet confirmed" in live[0]                   # nothing confirmed in-game yet
    two_same = ex.validate_plan([ex.PlannedAction("off.hot_route", {"target": "WR1", "route": "Slant"}),
                                 ex.PlannedAction("off.hot_route", {"target": "WR1", "route": "Post"})], ctx)
    assert any("cannot both be applied" in e for e in two_same)


# coverage features ---------------------------------------------------------------------------------

def test_coverage_shells_keep_palms_quarters_drop_and_hole_robber_distinct():
    sh = pf.SHELL_OF
    assert len({sh["cover_4_palms"], sh["cover_4_quarters"], sh["cover_4_drop"]}) == 3
    assert sh["cover_2_man"] != sh["cover_2_spot"]
    assert sh["cover_3_match"] != sh["cover_3_spot"]
    assert sh["cover_6"] == sh["cover_9"] == "split_field"
    assert set(sh) - {"cover_0"} <= set(SCHEMES)                      # every scheme id is a real PR #107 scheme
    assert set(MATCH_CHECKS) and len(MATCH_CHECKS) == 13


def test_matchup_features_are_priors_zero_without_routes_and_respect_uncertainty():
    s = pf.structure_lookup("Gun Trips X Nasty", "Hi Lo Cross")
    unknown = pf.structure_lookup("Gun 5WR Tight", "Mesh")
    d_man, _ = pf.scheme_distribution({"cover1": 1.0})
    d_none, src = pf.scheme_distribution(None)
    assert src == "no_pre_snap_information" and d_none == {"unknown": 1.0}
    assert pf.matchup_features(s, d_man)["m_one_high_man"] > 0
    assert all(v == 0.0 for k, v in pf.matchup_features(unknown, d_man).items() if k not in ("m_info", "m_entropy"))
    assert pf.matchup_features(s, d_none)["m_info"] == 0.0
    d_q, _ = pf.scheme_distribution({"cover4": 1.0})
    f = pf.matchup_features(s, d_q)
    assert f["m_quarters"] == 0.0 or f["m_quarters"] != f["m_palms"]
    d_p, _ = pf.scheme_distribution({"cover4": 1.0})
    assert f["m_info"] == pytest.approx(1.0) and f["m_entropy"] > 0   # uniform within the cover-4 bucket: uncertain
    assert pf.matchup_features(s, {"cover_4_palms": 1.0})["m_quarters"] == 0.0
    assert pf.matchup_features(s, {"cover_4_quarters": 1.0})["m_palms"] == 0.0


def test_exported_taxonomy_file_is_current():
    assert json.loads((oa.DATA_DIR / "offense_taxonomy.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(tx.as_dict()))


def test_annotation_cli_template_import_roundtrip(tmp_path):
    import csv
    import subprocess
    import sys
    from pathlib import Path
    root = Path(oa.__file__).resolve().parents[3]
    cli = [sys.executable, str(root / "scripts/madden_ml/offense_annotation.py")]
    tpl = tmp_path / "t.csv"
    assert subprocess.run(cli + ["template", "--out", str(tpl)], capture_output=True).returncode == 0
    rows = list(csv.DictReader(tpl.open()))
    assert len(rows) == 1023 and {r["formation"] for r in rows} == set(SNAP["formations"])

    def fill(source, tier="diagram_verified"):
        for r in rows:
            if (r["formation"], r["play"], r["slot"]) == ("Gun 5WR Tight", "Mesh", "SKILL_1"):
                r.update(category="route", task="drag", side="left", tier=tier, source=source)
        out = tmp_path / "f.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return out

    ov = tmp_path / "overlay.json"
    bad = subprocess.run(cli + ["import", "--csv", str(fill("https://example.com/x")), "--out", str(ov)],
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "IMPORT REJECTED" in bad.stdout and not ov.exists()
    good = subprocess.run(cli + ["import", "--csv", str(fill("user_screenshot:mesh.png")), "--out", str(ov)],
                          capture_output=True, text=True)
    assert good.returncode == 0, good.stdout + good.stderr
    merged, _ = oa.merge_overlay(oa.load_database(), json.loads(ov.read_text()))
    assert oa.index(merged)[("Gun 5WR Tight", "Mesh")]["slots"]["SKILL_1"]["tier"] == "diagram_verified"
    assert oa.coverage_stats(merged)["partial_plays"] == 9 and oa.coverage_stats(merged)["fully_verified_plays"] == 0
    assert not oa.OVERLAY_PATH.exists()          # the import never wrote into the repo data dir by itself here
