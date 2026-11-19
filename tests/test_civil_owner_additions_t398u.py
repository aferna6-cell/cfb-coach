"""Owner t398u plays that remain on the *shared* catalog after t414u: Pistol Strong Y Off Close PA Slide,
red-zone Gun Bunch TE Mesh, Pistol Strong Y Off Close HB Zone / HB Dive.
Gun Bunch TE Mtn Spacing 1-4 and Mtn Fork Y-Corner 1-3 stay out of the shared book (Chiefs/Titans).
They are re-added only on the Lions/CPU book (t445u); see tests/test_lions_cpu_book_t445u.py."""
from __future__ import annotations

import collections
import copy
import json

import pytest

from cfb_coach.madden.scheme import adjustments as adjm
from cfb_coach.madden.scheme import beliefs as bl
from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import coordinator as co
from cfb_coach.madden.scheme import design as dz
from cfb_coach.madden.scheme import legality as lg
from cfb_coach.madden.scheme import model as sm
from cfb_coach.madden.scheme import play_cooldown as pcd
from tests.test_civil_scheme_only import CAT, CAT_LIONS, _artifact, _close_snap, _db, _sit, _verify_all
from tests.test_civil_scheme_only import civil_env  # noqa: F401 — pytest fixture
from tests.test_titans_play_cooldown import _titans

GB, PS = "Gun Bunch TE", "Pistol Strong Y Off Close"
RZ_MESH = "civil_rz_gun_bunch_te_mesh"
HB_ZONE, HB_DIVE = "civil_run_pistol_strong_y_off_close_hb_zone", "civil_run_pistol_strong_y_off_close_hb_dive"
PA_SLIDE = "civil_t398_pistol_strong_y_off_close_pa_slide"
KEPT = {PA_SLIDE, RZ_MESH, HB_ZONE, HB_DIVE}
MTN_SPACING = [f"civil_t398_gun_bunch_te_mtn_spacing_{i}" for i in (1, 2, 3, 4)]
MTN_FORK = [f"civil_t398_gun_bunch_te_mtn_fork_y_corner_{i}" for i in (1, 2, 3)]
REMOVED = set(MTN_SPACING) | set(MTN_FORK)

# The owner's setup steps, as sent (guide wording, macro flag) — kept plays only.
WANT_STEPS = {
    PA_SLIDE: [("Streak Iso WR", False), ("Slant TE", False), ("Slant WR Next To TE", False),
               ("When You Snap IMMEDIATELY Hit RT/R1 To Cancel Playaction", False)],
    RZ_MESH: [("Crosser Outside Bunch WR", False)],
}


def _owner():
    return json.loads((sc.repo_root() / sc.ADDITIONS_RELPATH).read_text())


def test_t414_mtn_plays_are_absent_and_kept_t398_plays_remain():
    ids = set(CAT.ids)
    assert REMOVED.isdisjoint(ids)
    assert CAT.get(MTN_SPACING[0]) is None and CAT.get(MTN_FORK[0]) is None
    assert {v.variant_id for v in CAT.variants if v.owner_source == "t398u"} == KEPT
    assert KEPT <= ids
    owner = _owner()
    owner_ids = {v["variant_id"] for v in owner["pass_variants"] + owner["red_zone_variants"] + owner["designed_runs"]}
    assert owner_ids == KEPT
    assert not owner.get("name_not_in_stock_data")
    assert "Mtn Spacing" not in owner["cooldown_families"] and "Mtn Fork Y-Corner" not in owner["cooldown_families"]
    assert not any("Mtn Spacing" in (v.get("base_play") or "") or "Mtn Fork" in (v.get("base_play") or "")
                   or "Mtn Spacing" in (v.get("variant_name") or "") or "Mtn Fork" in (v.get("variant_name") or "")
                   for v in owner["pass_variants"] + owner["red_zone_variants"] + owner["designed_runs"])
    assert len(CAT.variants) == 49 and sc.contract_errors(CAT) == []
    assert {pcd.family_of_variant(v) for v in CAT.variants} == set(pcd.catalog_families(CAT))
    assert len(pcd.catalog_families(CAT)) == 29
    assert {"mtn spacing", "mtn fork y corner"}.isdisjoint(pcd.catalog_families(CAT))


def test_t398_kept_plays_load_with_exact_names_macros_and_steps():
    names = {vid: CAT.get(vid).variant_name for vid in KEPT}
    assert names[PA_SLIDE] == "PA Slide" and names[RZ_MESH] == "Gun Bunch TE Mesh (Redzone Pass)"
    assert names[HB_ZONE] == "HB Zone" and names[HB_DIVE] == "HB Dive"
    for vid, steps in WANT_STEPS.items():
        v = CAT.get(vid)
        assert [(a.guide_text, a.macro_marked) for a in v.adjustments] == steps, vid
        if v.group == sc.OWNER_PASS:
            assert v.raw["macro_name"] == v.variant_name
    assert CAT.get(PA_SLIDE).formation == PS and CAT.get(PA_SLIDE).base_play == "PA Slide"
    assert CAT.get(PA_SLIDE).field_zone == "any" and not CAT.get(PA_SLIDE).is_run
    # catalog bytes and every confirmation pinned to them are unchanged
    assert CAT.version() == "lions_civil_bucs_5wr_v4@15c120decc4b" and lg.owner_confirmation_applies(CAT)


def test_t398_kept_names_are_in_the_bucs_book():
    stock = sc.stock_offense_plays()
    for vid in KEPT:
        v = CAT.get(vid)
        assert v.base_play in stock[v.formation] and v.name_in_stock_data, vid
    assert {"HB Zone", "HB Dive", "PA Slide"} <= set(stock[PS])
    assert "Mesh" in stock[GB]
    # the removed Mtn names still are not stock, and must not be callable anyway
    assert "Mtn Spacing" not in stock[GB] and "Mtn Fork Y-Corner" not in stock[GB]


@pytest.mark.parametrize("edit", ["unlisted_name", "run_adjustment", "run_not_stock", "pass_no_steps",
                                  "rz_no_gate", "wrong_version", "duplicate_id", "bad_formation"])
def test_t398_loader_fails_closed(edit):
    base = sc.parse(json.loads((sc.repo_root() / sc.DATASET_RELPATH).read_bytes()), digest=CAT.digest)
    o = copy.deepcopy(_owner())
    if edit == "unlisted_name":
        o["pass_variants"][0]["base_play"] = "PA Slide Weak"
    elif edit == "run_adjustment":
        o["designed_runs"][0]["adjustments"] = [{"target": "HB", "action": "Stretch"}]
    elif edit == "run_not_stock":
        o["designed_runs"][1]["base_play"] = o["designed_runs"][1]["variant_name"] = "HB Dive Weak"
    elif edit == "pass_no_steps":
        o["pass_variants"][0]["adjustments"] = []
    elif edit == "rz_no_gate":
        o["red_zone_variants"][0]["field_zone"] = "any"
    elif edit == "wrong_version":
        o["catalog_version"] = "lions_civil_bucs_5wr_v4@000000000000"
    elif edit == "duplicate_id":
        o["pass_variants"].append(dict(o["pass_variants"][0]))
    elif edit == "bad_formation":
        o["pass_variants"][0]["formation"] = "Gun Bunch TE Flex"
    with pytest.raises(sc.CatalogError):
        sc.with_owner_additions(base, o, digest="x")


def test_t398_situational_tags_on_kept_plays():
    v = CAT.get(PA_SLIDE)
    assert v.scheme_category == "Gotcha Pass"
    assert not v.expert_coverage_tags and not v.coverage_tags_known
    assert list(CAT.get(RZ_MESH).expert_situation_tags) == ["Redzone Inside 10", "Redzone 10-20"]


def test_t398_field_gates_red_zone_mesh_and_pistol_runs():
    for y in range(0, 101):
        ids = {v.variant_id for v in lg.legal_set(CAT, db=None, yardline=y)[0]}
        band = sc.field_band(y)
        assert (RZ_MESH in ids) == (band in ("inside_10", "ten_to_twenty")), y
        for vid in (HB_ZONE, HB_DIVE, PA_SLIDE):
            assert (vid in ids) == (band != "inside_10"), (vid, y)
        assert ids.isdisjoint(REMOVED)
    rz = CAT.get(RZ_MESH)
    assert rz.field_zone == sc.RED_ZONE and rz.group == sc.REDZONE_GROUP and rz.base_play == "Mesh"
    for vid in (HB_ZONE, HB_DIVE):
        v = CAT.get(vid)
        assert v.is_run and v.group == sc.DESIGNED_RUN and not v.adjustments and v.unadjusted_allowed


def test_t398_cooldown_families_keep_pa_slide_runs_and_mesh_drop_mtn():
    fam = {vid: pcd.family_of_variant(CAT.get(vid)) for vid in KEPT}
    assert fam[PA_SLIDE] == "pa slide" and fam[HB_DIVE] == "hb dive"
    assert fam[HB_ZONE] == pcd.family_of_variant(CAT.get("civil_rz_pistol_bunch_te_hb_zone")) == "hb zone"
    assert fam[RZ_MESH] == pcd.family_of_variant(CAT.get("civil_bucs_mesh_3")) == "mesh"
    old = {pcd.family_of_variant(v) for v in CAT.variants if v.owner_source != "t398u"}
    assert {"pa slide", "hb dive"}.isdisjoint(old)
    assert {"mtn spacing", "mtn fork y corner"}.isdisjoint(old | set(fam.values()))
    # any Mesh call blocks the new red-zone Mesh; PA Slide stays legal
    legal = [CAT.get(i) for i in (RZ_MESH, PA_SLIDE, HB_ZONE, HB_DIVE)]
    kept, _ = pcd.apply_play_cooldown(legal, ["mesh", "hb zone"], cat=CAT)
    assert {v.variant_id for v in kept} == {PA_SLIDE, HB_DIVE}


def test_t398_kept_variants_are_independent_and_rz_mesh_is_not_a_regular_sibling(tmp_path):
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, dz.GeometrySource(enabled=False))
    assert not s.has_siblings[PA_SLIDE] and not s.has_siblings[HB_ZONE] and not s.has_siblings[HB_DIVE]
    # the red-zone Mesh is not lifted by the regular Mesh variants (never legal next to it inside the 10)
    assert not s.has_siblings[RZ_MESH] and s.has_siblings["civil_bucs_mesh_3"]
    assert co.repeat_penalty(PA_SLIDE, [PA_SLIDE], {PA_SLIDE: 1}) > 0
    assert co.repeat_penalty(HB_DIVE, [PA_SLIDE], {PA_SLIDE: 1}) == 0


def test_t398_steps_shown_verbatim_on_the_card(tmp_path):
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, dz.GeometrySource(enabled=False))
    bel = bl.combine({}, 0.1, 0, "none", [])
    for vid in KEPT:
        v = CAT.get(vid)
        top = s.score([v], down=1, distance=10, yardline=85 if v.field_zone != "any" else 40, opponent_id="cpu",
                      opponent_type="cpu", belief=bel, offset_override={vid: 0.5})
        call = co.build_call(top[0], top, bel, book=None, shadow=False)
        text = call.format()
        for a in v.adjustments:
            assert a.guide_text in text, (vid, a.guide_text)
        w = " ".join(call.scheme_variant["warnings"])
        if v.group == sc.OWNER_PASS:
            assert "OWNER-ADDED PLAY" in w and f"macro name: {v.variant_name}" in w
        assert "spells it differently" not in w
        if v.is_run:
            assert "t398u" in w
    # play-action cancel is a real step but not route geometry: shown, never dropped
    sp = adjm.parse_variant(CAT.get(PA_SLIDE))[3]
    assert sp.kind == "unmodeled" and not sp.representable


def test_t398_live_calls_include_red_zone_mesh_inside_10_and_pistol_runs(tmp_path, civil_env):
    from cfb_coach.madden import playcaller
    db = _db(tmp_path, formations=sorted({v.formation for v in CAT.variants}))
    inside, outside = collections.Counter(), collections.Counter()
    for i in range(120):
        y = 91 + i % 8
        call = playcaller.make_call(_sit(1 + i % 3, min(1 + i % 9, 100 - y), y, session=f"rz{i}"), "cpu", db)
        vid = call.scheme_variant["variant_id"]
        assert vid in {v.variant_id for v in CAT_LIONS.redzone_variants}
        assert vid not in REMOVED
        inside[vid] += 1
    for i in range(240):
        call = playcaller.make_call(_sit(1 + i % 4, (3, 7, 10, 2)[i % 4], 25 + i % 50, session=f"mid{i}"), "cpu", db)
        vid = call.scheme_variant["variant_id"]
        # CPU sessions use the Lions/CPU book, so the Mtn plays may be called here.
        assert call.scheme_variant.get("catalog_book") == sc.LIONS_CPU_BOOK
        outside[vid] += 1
    assert inside[RZ_MESH] > 0, inside
    assert inside[RZ_MESH] < sum(inside.values()) * 0.6, inside        # mixed with the other red-zone plays
    assert outside[HB_ZONE] + outside[HB_DIVE] > 0, outside
    midfield_legal = {v.variant_id for v in lg.legal_set(CAT, db=db, yardline=40)[0]}
    assert {PA_SLIDE, HB_ZONE, HB_DIVE} <= midfield_legal
    assert midfield_legal.isdisjoint(REMOVED)
    rz_legal = {v.variant_id for v in lg.legal_set(CAT, db=db, yardline=95)[0]}
    assert RZ_MESH in rz_legal and rz_legal.isdisjoint(REMOVED | {PA_SLIDE, HB_ZONE, HB_DIVE})


def test_t398_titans_cooldown_blocks_kept_new_families(tmp_path, civil_env):
    from cfb_coach.madden import playcaller
    db = _db(tmp_path, formations=sorted({v.formation for v in CAT.variants}))
    _verify_all(db)
    _titans(db, "t398")
    _close_snap(db, "t398", PA_SLIDE, 40)
    _close_snap(db, "t398", HB_DIVE, 42)
    for i in range(3):
        call = playcaller.make_call(_sit(1, 10, 45 + i, session="t398"), "cpu", db)
        vid = call.scheme_variant["variant_id"]
        assert vid not in {PA_SLIDE, HB_DIVE} | REMOVED, vid
        cand = set(call.scheme_variant.get("candidates") or ())
        assert not (cand & ({PA_SLIDE, HB_DIVE} | REMOVED)), cand
        _close_snap(db, "t398", vid, 45 + i)
