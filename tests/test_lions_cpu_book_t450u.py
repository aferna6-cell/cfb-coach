"""t450u: p06_top Civil red-zone runs in already-used formations; t451u: Steelers+NSM defense in every league."""
from __future__ import annotations

import hashlib
import json

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import legality as lg
from tests.test_civil_scheme_only import CAT, CAT_LIONS, _db, _sit, civil_env  # noqa: F401
from tests.test_chiefs_play_cooldown import ravens_env  # noqa: F401
from tests.test_titans_play_cooldown import _lions, _online, _titans

SHARED_ADDITIONS_SHA256 = "22a93b0f3e3e644de22e2d2c69d10ff39c813a3bdd08e391259996e5a4e972e3"
RAVENS_DATASET_SHA256 = "5e0673b91112b272b19e715c1153bfb04bf9f109cc66e43cddfe1e7c2b960788"

NEW_IDS = {
    "civil_t450_rz_gun_bunch_te_hb_off_tackle",
    "civil_t450_rz_gun_bunch_te_wk_hb_dive",
    "civil_t450_rz_pistol_strong_y_off_close_hb_dive",
}

# Existing identical red-zone entries that p06_top would otherwise have duplicated.
DUPLICATE_RZ = {
    ("Gun Bunch TE", "HB Draw"): "civil_t448_rz_gun_bunch_te_hb_draw",
    ("Gun Empty Quads", "QB Blast"): "civil_rz_gun_empty_quads_qb_blast",
    ("Gun Heavy Saint", "QB Blast"): "civil_t448_rz_gun_heavy_saint_qb_blast",
    ("Pistol Deuce Close", "HB Dive"): "civil_t448_rz_pistol_deuce_close_hb_dive",
}


def test_shared_book_and_ravens_dataset_are_byte_identical():
    raw = (sc.repo_root() / sc.ADDITIONS_RELPATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHARED_ADDITIONS_SHA256
    assert len(CAT.variants) == 49 and sc.contract_errors(CAT) == []
    assert set(CAT.ids).isdisjoint(NEW_IDS)
    ravens = rc.DATASET.read_bytes()
    assert hashlib.sha256(ravens).hexdigest() == RAVENS_DATASET_SHA256
    assert rc.load().sha256 == RAVENS_DATASET_SHA256


def test_lions_cpu_gains_three_p06_red_zone_runs_and_skips_identical_duplicates():
    assert len(CAT_LIONS.variants) == sc.EXPECTED_LIONS_CPU_VARIANTS
    assert sc.contract_errors(CAT_LIONS) == []
    assert NEW_IDS <= set(CAT_LIONS.ids)
    regular_passes = {v.variant_id for v in CAT_LIONS.variants if v.field_zone == "any" and not v.is_run}
    assert len(regular_passes) == 88
    assert len(CAT_LIONS.run_variants) == sc.EXPECTED_LIONS_CPU_RUN_VARIANTS
    assert len(CAT_LIONS.redzone_variants) == 21 == sc.EXPECTED_LIONS_CPU_REDZONE_VARIANTS

    off = CAT_LIONS.get("civil_t450_rz_gun_bunch_te_hb_off_tackle")
    assert off.formation == "Gun Bunch TE" and off.base_play == "HB Off Tackle"
    assert off.field_zone == sc.RED_ZONE and off.is_run and not off.adjustments
    # t453u restored the t349u designed run alongside the t450u Civil red-zone copy.
    designed = CAT_LIONS.get("civil_run_gun_bunch_te_hb_off_tackle")
    assert designed is not None and designed.field_zone == "any" and designed.is_run
    assert CAT.get("civil_run_gun_bunch_te_hb_off_tackle") is not None
    assert designed.variant_id != off.variant_id

    wk = CAT_LIONS.get("civil_t450_rz_gun_bunch_te_wk_hb_dive")
    assert wk.formation == "Gun Bunch TE Wk" and wk.base_play == "HB Dive"
    assert wk.field_zone == sc.RED_ZONE and wk.is_run and not wk.adjustments
    designed_wk = CAT_LIONS.get("civil_t448_run_gun_bunch_te_wk_hb_dive")
    assert designed_wk is not None and designed_wk.field_zone == "any"

    pistol = CAT_LIONS.get("civil_t450_rz_pistol_strong_y_off_close_hb_dive")
    assert pistol.formation == "Pistol Strong Y Off Close" and pistol.base_play == "HB Dive"
    assert pistol.field_zone == sc.RED_ZONE and pistol.is_run and not pistol.adjustments
    designed_pistol = CAT_LIONS.get("civil_run_pistol_strong_y_off_close_hb_dive")
    assert designed_pistol is not None and designed_pistol.field_zone == "any"

    # Identical red-zone placements are not added a second time.
    rz_pairs = [(v.formation, v.base_play) for v in CAT_LIONS.redzone_variants if v.is_run]
    for pair, existing_id in DUPLICATE_RZ.items():
        assert CAT_LIONS.get(existing_id) is not None
        assert rz_pairs.count(pair) == 1
    assert all(vid.startswith("civil_t450_") is False or vid in NEW_IDS for vid in CAT_LIONS.ids)


def test_new_red_zone_runs_are_legal_only_inside_the_20():
    mid = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=None, yardline=40)[0]}
    ten = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=None, yardline=85)[0]}
    inside = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=None, yardline=95)[0]}
    assert mid.isdisjoint(NEW_IDS)
    assert NEW_IDS <= ten and NEW_IDS <= inside
    chiefs_rz = {v.variant_id for v in lg.legal_set(CAT, db=None, yardline=95)[0]}
    assert chiefs_rz.isdisjoint(NEW_IDS)


def test_shared_ravens_dataset_stays_two_formation_but_live_catalog_is_steelers():
    assert hasattr(rc, "load_lions_franchise") and hasattr(rc, "NSM_CALL_IDS")
    base = rc.load()
    assert rc.contract_errors(base.raw) == []
    assert hashlib.sha256(rc.DATASET.read_bytes()).hexdigest() == RAVENS_DATASET_SHA256
    assert tuple(f["name"] for f in base.raw["formations"]) == rc.FORMATIONS
    assert base.raw.get("source_defensive_playbook") == "Baltimore Ravens"
    assert "Nickel Single Mug" not in {f["name"] for f in base.raw["formations"]}

    cat = rc.catalog_for_defense_session(None, None, {}, None)
    assert cat is not base or cat.sha256 != base.sha256
    assert cat.stock_team == rc.STEELERS and cat.is_lions_franchise
    assert cat.allowed_formations == rc.LIONS_FRANCHISE_FORMATIONS
    assert {c.formation for c in cat.candidates} == {rc.NICKEL, rc.DIME, rc.NSM}
    assert set(rc.NSM_CALL_IDS) <= set(cat.by_id())
    plays = {c.base_play for c in cat.candidates if c.formation == rc.NSM}
    assert plays == {"Cover 4 Quarters", "Tampa 2"}
    assert "Steelers" in (cat.raw.get("source_defensive_playbook") or "")


def test_lions_chiefs_titans_and_cpu_defense_calls_include_nsm(tmp_path, civil_env, ravens_env):
    from cfb_coach.madden import playcaller

    db = _db(tmp_path, formations=["Gun Bunch TE"])
    _lions(db, "lions-human")
    _online(db, "online1")
    _titans(db, "titans1")
    db.conn.execute(
        "INSERT INTO game_sessions (session_id, opponent_id, dynasty, started_ts, "
        "league, user_team, opponent_team, opponent_type) "
        "VALUES ('cpu1', 'cpu', 'primary', '2026-11-17T16:00:00+00:00', "
        "NULL, 'Detroit Lions', 'Dallas Cowboys', 'cpu')"
    )
    db.conn.commit()

    sessions = (
        ("lions-human", "jaxon", False),
        ("online1", "online_indianapolis_colts", False),
        ("titans1", "titans_indianapolis_colts", False),
        ("cpu1", "cpu", True),
    )
    nsm_ids = set(rc.NSM_CALL_IDS)
    for sid, opponent, cpu_defense in sessions:
        seen = set()
        forms = set()
        for i in range(80):
            s = _sit(1, 10, 40, session=sid)
            s.side = "defense"
            s.extras = {"session_id": sid, "snap_seq": i}
            call = playcaller.make_call(s, opponent, db, cpu_defense=cpu_defense)
            rd = call.ravens_defense
            assert rd["status"] == "CALL", (sid, rd)
            assert call.formation in rc.LIONS_FRANCHISE_FORMATIONS
            text = "\n".join(rd["card"])
            assert "Lions franchise only" not in text
            seen.add(rd["candidate"]["call_id"])
            forms.add(call.formation)
            if call.formation == rc.NSM:
                assert "PACKAGE: Nickel Single Mug" in text and "UNVERIFIED" in text
        assert seen & nsm_ids, (sid, seen)
        assert rc.NSM in forms, (sid, forms)
