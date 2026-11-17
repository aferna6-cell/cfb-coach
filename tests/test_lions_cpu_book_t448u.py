"""t448u: Lions/CPU book adds the rest of the owner PDF; Chiefs/Titans stay on the shared 49-play book."""
from __future__ import annotations

import collections
import hashlib
import json

from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import legality as lg
from tests.test_civil_scheme_only import CAT, CAT_LIONS, LIONS_ONLY, REG_LIONS, RZ_LIONS, civil_env  # noqa: F401
from tests.test_lions_cpu_book_t445u import SNAPS, _forms, _simulate
from tests.test_civil_scheme_only import _db, _sit, _verify_all
from tests.test_titans_play_cooldown import _lions, _online, _titans
from tests.test_chiefs_play_cooldown import ravens_env  # noqa: F401

SHARED_ADDITIONS_SHA256 = "22a93b0f3e3e644de22e2d2c69d10ff39c813a3bdd08e391259996e5a4e972e3"
NEW_IDS = {v["variant_id"] for v in json.loads((sc.repo_root() / sc.LIONS_CPU_RELPATH).read_text())["pass_variants"]
           + json.loads((sc.repo_root() / sc.LIONS_CPU_RELPATH).read_text())["red_zone_variants"]
           + json.loads((sc.repo_root() / sc.LIONS_CPU_RELPATH).read_text())["designed_runs"]
           if str(v["variant_id"]).startswith("civil_t448_")}


def test_shared_additions_file_is_byte_identical():
    raw = (sc.repo_root() / sc.ADDITIONS_RELPATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHARED_ADDITIONS_SHA256
    assert len(CAT.variants) == 49 and sc.contract_errors(CAT) == []
    assert set(CAT.ids).isdisjoint(NEW_IDS)
    assert set(CAT.ids).isdisjoint(LIONS_ONLY)


def test_lions_cpu_book_has_the_t448u_additions():
    assert CAT_LIONS.book_id == sc.LIONS_CPU_BOOK
    assert len(CAT_LIONS.variants) == sc.EXPECTED_LIONS_CPU_VARIANTS
    assert sc.contract_errors(CAT_LIONS) == []
    assert CAT_LIONS.version() == CAT.version() == "lions_civil_bucs_5wr_v4@15c120decc4b"
    assert NEW_IDS and NEW_IDS <= set(CAT_LIONS.ids) - set(CAT.ids)
    assert len(NEW_IDS) == 54 + 9 + 5
    regular_passes = {v.variant_id for v in CAT_LIONS.variants if v.field_zone == "any" and not v.is_run}
    assert regular_passes and len(regular_passes) == 88
    # REG_LIONS is regular passes + designed runs (t449u dropped 10 inherited t349u runs).
    assert len(REG_LIONS) == 88 + sc.EXPECTED_LIONS_CPU_RUN_VARIANTS
    assert len(RZ_LIONS) == sc.EXPECTED_LIONS_CPU_REDZONE_VARIANTS
    # Every new pass has setup steps; every new designed run has none.
    owner = json.loads((sc.repo_root() / sc.LIONS_CPU_RELPATH).read_text())
    for row in owner["pass_variants"]:
        if row["variant_id"].startswith("civil_t448_"):
            assert row["adjustments"]
    for row in owner["designed_runs"]:
        assert row["variant_id"].startswith("civil_t448_")
        assert row["adjustments"] == [] and row["variant_name"] == row["base_play"]


def test_chiefs_and_titans_legal_sets_never_include_new_plays():
    for y in (40, 85, 95):
        ids = {v.variant_id for v in lg.legal_set(CAT, db=None, yardline=y)[0]}
        assert ids.isdisjoint(NEW_IDS) and ids.isdisjoint(LIONS_ONLY)


def test_lions_legal_sets_include_new_regular_and_red_zone_plays():
    mid = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=None, yardline=40)[0]}
    ten = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=None, yardline=85)[0]}
    inside = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=None, yardline=95)[0]}
    new_reg = {v.variant_id for v in CAT_LIONS.variants if v.variant_id in NEW_IDS and v.field_zone == "any"}
    new_rz20 = {v.variant_id for v in CAT_LIONS.variants if v.variant_id in NEW_IDS and v.field_zone == sc.RED_ZONE}
    new_rz10 = {v.variant_id for v in CAT_LIONS.variants if v.variant_id in NEW_IDS and v.field_zone == sc.INSIDE_10}
    assert new_reg <= mid and mid.isdisjoint(RZ_LIONS)
    assert new_reg | new_rz20 <= ten
    assert new_rz20 | new_rz10 <= inside and inside.isdisjoint(new_reg)
