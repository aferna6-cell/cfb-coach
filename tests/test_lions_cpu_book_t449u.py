"""t449u: Mesh TE red-zone variant, Civil-only runs on Lions/CPU, shared book unchanged."""
from __future__ import annotations

import hashlib
import json

from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import legality as lg
from tests.test_civil_scheme_only import CAT, CAT_LIONS, REG_LIONS, RZ_LIONS
from tests.test_civil_scheme_only import _db, _sit, _verify_all, civil_env  # noqa: F401
from tests.test_lions_cpu_book_t445u import _forms

SHARED_ADDITIONS_SHA256 = "22a93b0f3e3e644de22e2d2c69d10ff39c813a3bdd08e391259996e5a4e972e3"
MESH_TE = "civil_t449_rz_gun_bunch_te_mesh_crosser_outside_bunch_te"
MESH_WR = "civil_rz_gun_bunch_te_mesh"
OLD_RUNS = (
    "civil_run_gun_doubles_clamp_stack_counter_y",
    "civil_run_gun_doubles_clamp_stack_hb_mid_zone",
    "civil_run_gun_bunch_te_inside_zone",
    "civil_run_gun_bunch_te_read_option",
    "civil_run_gun_bunch_te_hb_off_tackle",
    "civil_run_gun_bunch_te_boomerang_hb_off_tackle",
    "civil_run_gun_bunch_te_same_side_zone",
    "civil_run_gun_tight_doubles_inside_zone",
    "civil_run_gun_tight_doubles_hb_draw",
    "civil_run_gun_5wr_tight_qb_draw",
)
KEPT_CLAMP = (
    "civil_run_gun_doubles_clamp_stack_inside_zone",
    "civil_run_gun_doubles_clamp_stack_same_side_zone",
)
KEPT_PISTOL = (
    "civil_run_pistol_strong_y_off_close_hb_zone",
    "civil_run_pistol_strong_y_off_close_hb_dive",
)
RPO = "civil_rz_gun_empty_quads_rpo_alert_qb_draw"


def _owner():
    return json.loads((sc.repo_root() / sc.LIONS_CPU_RELPATH).read_text())


def test_shared_book_is_byte_identical_and_still_has_old_runs():
    raw = (sc.repo_root() / sc.ADDITIONS_RELPATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHARED_ADDITIONS_SHA256
    assert len(CAT.variants) == 49 and sc.contract_errors(CAT) == []
    assert CAT.book_id == sc.SHARED_BOOK
    for vid in OLD_RUNS + KEPT_CLAMP + KEPT_PISTOL + (RPO, MESH_WR):
        assert CAT.get(vid) is not None, vid
    assert CAT.get(MESH_TE) is None


def test_lions_cpu_adds_mesh_te_and_drops_only_old_runs():
    owner = _owner()
    assert owner["owner_instruction"]["source"] == "t449u"
    assert set(owner["exclude_inherited_variant_ids"]) == set(OLD_RUNS)
    assert len(CAT_LIONS.variants) == 115 == sc.EXPECTED_LIONS_CPU_VARIANTS
    assert sc.contract_errors(CAT_LIONS) == []
    assert CAT_LIONS.version() == CAT.version() == "lions_civil_bucs_5wr_v4@15c120decc4b"
    te = CAT_LIONS.get(MESH_TE)
    wr = CAT_LIONS.get(MESH_WR)
    assert te is not None and wr is not None and te.variant_id != wr.variant_id
    assert te.formation == wr.formation == "Gun Bunch TE" and te.base_play == wr.base_play == "Mesh"
    assert te.field_zone == sc.RED_ZONE and te.play_family == sc.PASS
    assert [a.guide_text for a in te.adjustments] == ["Crosser Outside Bunch TE"]
    assert [a.guide_text for a in wr.adjustments] == ["Crosser Outside Bunch WR"]
    for vid in OLD_RUNS:
        assert CAT_LIONS.get(vid) is None, vid
    for vid in KEPT_CLAMP + KEPT_PISTOL:
        assert CAT_LIONS.get(vid) is not None and CAT_LIONS.get(vid).is_run, vid
    assert CAT_LIONS.get(RPO) is not None and CAT_LIONS.get(RPO).is_run
    assert CAT_LIONS.get(RPO).field_zone == sc.INSIDE_10
    assert len(CAT_LIONS.run_variants) == 9 == sc.EXPECTED_LIONS_CPU_RUN_VARIANTS
    assert len(CAT_LIONS.redzone_variants) == 18 == sc.EXPECTED_LIONS_CPU_REDZONE_VARIANTS
    assert len(REG_LIONS) == 88 + 9
    assert MESH_TE in RZ_LIONS


def test_lions_still_has_a_legal_run_in_every_field_band():
    for y, band in ((40, "outside"), (85, "ten_to_twenty"), (95, "inside_10")):
        legal = [v for v in lg.legal_set(CAT_LIONS, db=None, yardline=y)[0] if v.is_run]
        assert legal, f"no legal Lions/CPU run at yardline {y} ({band})"
        if y >= 90:
            assert {v.variant_id for v in legal} <= RZ_LIONS
        assert RPO not in {v.variant_id for v in legal} or y >= 90


def test_mesh_te_is_legal_inside_the_20_only():
    mid = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=None, yardline=40)[0]}
    ten = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=None, yardline=85)[0]}
    inside = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=None, yardline=95)[0]}
    assert MESH_TE not in mid and MESH_WR not in mid
    assert {MESH_TE, MESH_WR} <= ten and {MESH_TE, MESH_WR} <= inside


def test_commanders_best_inside_the_5_is_not_in_either_book():
    for cat in (CAT, CAT_LIONS):
        assert not [v for v in cat.variants if "heavy redzone" in (v.variant_name or "").lower()
                    or "best inside the 5" in (v.verification_notes or "").lower()]
        assert "Gun Heavy Redzone" not in {v.formation for v in cat.variants}
