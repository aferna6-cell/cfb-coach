"""Owner t398u: new civil.gg plays (Gun Bunch TE Mtn Spacing / Mtn Fork Y-Corner, Pistol Strong Y Off Close PA Slide,
red-zone Gun Bunch TE Mesh, Pistol Strong Y Off Close HB Zone / HB Dive)."""
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
from tests.test_civil_scheme_only import CAT, _artifact, _close_snap, _db, _sit, _verify_all
from tests.test_civil_scheme_only import civil_env  # noqa: F401 — pytest fixture
from tests.test_titans_play_cooldown import _titans

GB, PS = "Gun Bunch TE", "Pistol Strong Y Off Close"
RZ_MESH = "civil_rz_gun_bunch_te_mesh"
HB_ZONE, HB_DIVE = "civil_run_pistol_strong_y_off_close_hb_zone", "civil_run_pistol_strong_y_off_close_hb_dive"
PA_SLIDE = "civil_t398_pistol_strong_y_off_close_pa_slide"
SPACING = [f"civil_t398_gun_bunch_te_mtn_spacing_{i}" for i in (1, 2, 3, 4)]
FORK = [f"civil_t398_gun_bunch_te_mtn_fork_y_corner_{i}" for i in (1, 2, 3)]
NEW = set(SPACING) | set(FORK) | {PA_SLIDE, RZ_MESH, HB_ZONE, HB_DIVE}
EV = ["Cover 2", "Cover 3", "Cover 4", "Man Coverage"]

# The owner's setup steps, as sent (guide wording, macro flag)
WANT_STEPS = {
    SPACING[0]: [("Streak TE", True), ("Drag Inside Bunch WR", False), ("In Route Outside Bunch WR", False)],
    SPACING[1]: [("Drag TE", True), ("Corner Route Inside Bunch WR & Custom Stem Down", False),
                 ("Streak Middle Bunch WR", True), ("Slot Fade Outside Bunch WR", False)],
    SPACING[2]: [("Trail TE", True), ("Corner Route Inside Bunch WR & Custom Stem Down", False),
                 ("Streak Middle Bunch WR", True), ("Drag Outside Bunch WR", False)],
    SPACING[3]: [("In Route TE & Custom Stem Up 2x On Dpad", True), ("Slant Inside Bunch WR", False),
                 ("Streak Middle Bunch WR", True), ("In Route Outside Bunch WR", False)],
    FORK[0]: [("In Route TE", True), ("Flat Inside Bunch WR", False), ("Drag Middle Bunch WR", True),
              ("Flat HB", True), ("Custom Stem Post Down 1x On Dpad", False)],
    FORK[1]: [("In Route Inside Bunch WR", False), ("Streak Middle Bunch WR", True), ("Flat HB", True),
              ("Custom Stem Post Down 1x On Dpad", False)],
    FORK[2]: [("Streak TE", True), ("Flat Route Inside Bunch WR", False), ("Drag Middle Bunch WR", True),
              ("In Route HB", True), ("Custom Stem Post Down 1x On Dpad", False)],
    PA_SLIDE: [("Streak Iso WR", False), ("Slant TE", False), ("Slant WR Next To TE", False),
               ("When You Snap IMMEDIATELY Hit RT/R1 To Cancel Playaction", False)],
    RZ_MESH: [("Crosser Outside Bunch WR", False)],
}


def _owner():
    return json.loads((sc.repo_root() / sc.ADDITIONS_RELPATH).read_text())


def test_t398_plays_load_with_exact_names_macros_and_steps():
    assert {v.variant_id for v in CAT.variants if v.owner_source == "t398u"} == NEW
    assert len(CAT.variants) == 56 and sc.contract_errors(CAT) == []
    names = {vid: CAT.get(vid).variant_name for vid in NEW}
    assert [names[i] for i in SPACING] == [f"Mtn Spacing {i}" for i in (1, 2, 3, 4)]
    assert [names[i] for i in FORK] == [f"Mtn Fork Y-Corner {i}" for i in (1, 2, 3)]
    assert names[PA_SLIDE] == "PA Slide" and names[RZ_MESH] == "Gun Bunch TE Mesh (Redzone Pass)"
    assert names[HB_ZONE] == "HB Zone" and names[HB_DIVE] == "HB Dive"
    for vid, steps in WANT_STEPS.items():
        v = CAT.get(vid)
        assert [(a.guide_text, a.macro_marked) for a in v.adjustments] == steps, vid
        if v.group == sc.OWNER_PASS:
            assert v.raw["macro_name"] == v.variant_name
    for vid in SPACING + FORK:
        assert CAT.get(vid).formation == GB and CAT.get(vid).field_zone == "any" and not CAT.get(vid).is_run
    assert CAT.get(PA_SLIDE).formation == PS and CAT.get(PA_SLIDE).base_play == "PA Slide"
    # catalog bytes and every confirmation pinned to them are unchanged
    assert CAT.version() == "lions_civil_bucs_5wr_v4@15c120decc4b" and lg.owner_confirmation_applies(CAT)


def test_t398_names_checked_against_the_bucs_book_and_mismatches_flagged():
    stock = sc.stock_offense_plays()
    for vid in (PA_SLIDE, RZ_MESH, HB_ZONE, HB_DIVE):
        v = CAT.get(vid)
        assert v.base_play in stock[v.formation] and v.name_in_stock_data, vid
    # HB Zone and HB Dive both exist under Pistol Strong Y Off Close in the Bucs book
    assert {"HB Zone", "HB Dive", "PA Slide"} <= set(stock[PS])
    for vid in SPACING + FORK:
        v = CAT.get(vid)
        assert v.base_play not in stock[GB] and not v.name_in_stock_data, vid
    assert "Spacing" in stock[GB]          # closest stock name, recorded in the owner file
    flagged = {(e["formation"], e["base_play"]) for e in _owner()["name_not_in_stock_data"]}
    assert flagged == {(GB, "Mtn Spacing"), (GB, "Mtn Fork Y-Corner")}


@pytest.mark.parametrize("edit", ["unlisted_name", "run_adjustment", "run_not_stock", "pass_no_steps",
                                  "rz_no_gate", "wrong_version", "duplicate_id", "bad_formation"])
def test_t398_loader_fails_closed(edit):
    base = sc.parse(json.loads((sc.repo_root() / sc.DATASET_RELPATH).read_bytes()), digest=CAT.digest)
    o = copy.deepcopy(_owner())
    if edit == "unlisted_name":
        o["pass_variants"][0]["base_play"] = "Mtn Spacing Weak"
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
        o["pass_variants"][1]["variant_id"] = o["pass_variants"][0]["variant_id"]
    elif edit == "bad_formation":
        o["pass_variants"][0]["formation"] = "Gun Bunch TE Flex"
    with pytest.raises(sc.CatalogError):
        sc.with_owner_additions(base, o, digest="x")


def test_t398_situational_tags_match_the_beater_labels():
    want = {"Everything Beater": EV, "Zone Beater": ["Cover 2", "Cover 3", "Cover 4"],
            "Man/Match Beater": ["Cover 4", "Man Coverage"]}
    labels = {SPACING[0]: "Everything Beater", SPACING[1]: "Zone Beater", SPACING[2]: "Man/Match Beater",
              SPACING[3]: "Everything Beater", FORK[0]: "Everything Beater", FORK[1]: "Zone Beater"}
    for vid, lab in labels.items():
        v = CAT.get(vid)
        assert v.scheme_category == lab and list(v.expert_coverage_tags) == want[lab], vid
        assert v.coverage_tags_known
    for vid in (FORK[2], PA_SLIDE):        # no beater label / gotcha pass: unknown coverage, not negative
        v = CAT.get(vid)
        assert not v.expert_coverage_tags and not v.coverage_tags_known
    assert CAT.get(PA_SLIDE).scheme_category == "Gotcha Pass"
    assert list(CAT.get(RZ_MESH).expert_situation_tags) == ["Redzone Inside 10", "Redzone 10-20"]


def test_t398_field_gates_red_zone_mesh_and_pistol_runs():
    for y in range(0, 101):
        ids = {v.variant_id for v in lg.legal_set(CAT, db=None, yardline=y)[0]}
        band = sc.field_band(y)
        assert (RZ_MESH in ids) == (band in ("inside_10", "ten_to_twenty")), y
        for vid in (HB_ZONE, HB_DIVE, PA_SLIDE, *SPACING, *FORK):
            assert (vid in ids) == (band != "inside_10"), (vid, y)
    rz = CAT.get(RZ_MESH)
    assert rz.field_zone == sc.RED_ZONE and rz.group == sc.REDZONE_GROUP and rz.base_play == "Mesh"
    for vid in (HB_ZONE, HB_DIVE):
        v = CAT.get(vid)
        assert v.is_run and v.group == sc.DESIGNED_RUN and not v.adjustments and v.unadjusted_allowed


def test_t398_cooldown_families():
    fam = {vid: pcd.family_of_variant(CAT.get(vid)) for vid in NEW}
    assert {fam[i] for i in SPACING} == {"mtn spacing"}
    assert {fam[i] for i in FORK} == {"mtn fork y corner"}
    assert fam[PA_SLIDE] == "pa slide" and fam[HB_DIVE] == "hb dive"
    assert fam[HB_ZONE] == pcd.family_of_variant(CAT.get("civil_rz_pistol_bunch_te_hb_zone")) == "hb zone"
    assert fam[RZ_MESH] == pcd.family_of_variant(CAT.get("civil_bucs_mesh_3")) == "mesh"
    # new families are distinct from the old ones they resemble
    old = {pcd.family_of_variant(v) for v in CAT.variants if v.owner_source != "t398u"}
    assert {"mtn spacing", "mtn fork y corner", "pa slide", "hb dive"}.isdisjoint(old)
    # any Mesh call blocks the new red-zone Mesh; a Mtn Spacing call blocks every Mtn Spacing variant
    legal = [CAT.get(i) for i in (RZ_MESH, *SPACING, *FORK, PA_SLIDE)]
    kept, _ = pcd.apply_play_cooldown(legal, ["mesh", "mtn spacing"], cat=CAT)
    assert {v.variant_id for v in kept} == set(FORK) | {PA_SLIDE}


def test_t398_variants_are_independent_for_repetition_and_siblings_mix(tmp_path):
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, dz.GeometrySource(enabled=False))
    for group in (SPACING, FORK):
        assert all(s.has_siblings[v] for v in group)
    assert not s.has_siblings[PA_SLIDE] and not s.has_siblings[HB_ZONE] and not s.has_siblings[HB_DIVE]
    # the red-zone Mesh is not lifted by the regular Mesh variants (never legal next to it inside the 10)
    assert not s.has_siblings[RZ_MESH] and s.has_siblings["civil_bucs_mesh_3"]
    # repeat penalty is per variant_id: calling Mtn Spacing 1 does not penalize Mtn Spacing 2
    assert co.repeat_penalty(SPACING[0], [SPACING[0]], {SPACING[0]: 1}) > 0
    assert co.repeat_penalty(SPACING[1], [SPACING[0]], {SPACING[0]: 1}) == 0
    # sibling mix: draw among near-best Mtn Spacing variants
    bel = bl.combine({}, 0.1, 0, "none", [])
    vs = [CAT.get(i) for i in SPACING]
    ranked = s.score(vs, down=1, distance=10, yardline=40, opponent_id="cpu", opponent_type="cpu", belief=bel,
                     offset_override={v: 0.5 for v in SPACING})
    chosen = collections.Counter(co.pick(ranked, f"seed{i}")[0].variant.variant_id for i in range(80))
    assert len(chosen) >= 3, chosen


def test_t398_steps_shown_verbatim_on_the_card(tmp_path):
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, dz.GeometrySource(enabled=False))
    bel = bl.combine({}, 0.1, 0, "none", [])
    for vid in NEW:
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
        if not v.name_in_stock_data:
            assert "spells it differently" in w      # name mismatch vs the repo's stock data is flagged on the card
        if v.is_run:
            assert "t398u" in w
    # Trail and the play-action cancel are real steps but not route geometry: shown, never dropped
    for vid, idx in ((SPACING[2], 0), (PA_SLIDE, 3)):
        sp = adjm.parse_variant(CAT.get(vid))[idx]
        assert sp.kind == "unmodeled" and not sp.representable


def test_t398_live_calls_include_red_zone_mesh_inside_10_and_pistol_runs(tmp_path, civil_env):
    from cfb_coach.madden import playcaller
    db = _db(tmp_path, formations=sorted({v.formation for v in CAT.variants}))
    inside, outside = collections.Counter(), collections.Counter()
    for i in range(120):
        y = 91 + i % 8
        call = playcaller.make_call(_sit(1 + i % 3, min(1 + i % 9, 100 - y), y, session=f"rz{i}"), "cpu", db)
        assert call.scheme_variant["variant_id"] in {v.variant_id for v in CAT.redzone_variants}
        inside[call.scheme_variant["variant_id"]] += 1
    for i in range(240):
        call = playcaller.make_call(_sit(1 + i % 4, (3, 7, 10, 2)[i % 4], 25 + i % 50, session=f"mid{i}"), "cpu", db)
        outside[call.scheme_variant["variant_id"]] += 1
    assert inside[RZ_MESH] > 0, inside
    assert inside[RZ_MESH] < sum(inside.values()) * 0.6, inside        # mixed with the other red-zone plays
    assert outside[HB_ZONE] + outside[HB_DIVE] > 0, outside
    assert sum(outside[v] for v in SPACING + FORK) > 0, outside


def test_t398_titans_cooldown_blocks_new_families(tmp_path, civil_env):
    from cfb_coach.madden import playcaller
    db = _db(tmp_path, formations=sorted({v.formation for v in CAT.variants}))
    _verify_all(db)
    _titans(db, "t398")
    _close_snap(db, "t398", SPACING[1], 40)
    _close_snap(db, "t398", HB_DIVE, 42)
    for i in range(3):
        call = playcaller.make_call(_sit(1, 10, 45 + i, session="t398"), "cpu", db)
        vid = call.scheme_variant["variant_id"]
        assert vid not in set(SPACING) | {HB_DIVE}, vid
        cand = set(call.scheme_variant.get("candidates") or ())
        assert not (cand & (set(SPACING) | {HB_DIVE})), cand
        _close_snap(db, "t398", vid, 45 + i)
