"""t445u: Lions and CPU use a separate book (shared catalog + Mtn Spacing / Mtn Fork Y-Corner).

Chiefs online and Titans franchise stay on the current shared book, which does not include those
Eagles plays. Existing cooldowns stay: Chiefs 3, Titans 5, none for Lions.
"""
from __future__ import annotations

import collections
import copy
import json

import pytest

from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import legality as lg
from cfb_coach.madden.scheme import play_cooldown as pcd
from tests.test_civil_scheme_only import (
    CAT,
    CAT_LIONS,
    MTN_PASS,
    REG,
    REG_LIONS,
    RZ,
    _close_snap,
    _db,
    _sit,
    _verify_all,
    civil_env,  # noqa: F401
)
from tests.test_chiefs_play_cooldown import _defense_call, ravens_env  # noqa: F401
from tests.test_titans_play_cooldown import _giants, _lions, _online, _titans

MTN_SPACING = [f"civil_t398_gun_bunch_te_mtn_spacing_{i}" for i in (1, 2, 3, 4)]
MTN_FORK = [f"civil_t398_gun_bunch_te_mtn_fork_y_corner_{i}" for i in (1, 2, 3)]
MTN_IDS = set(MTN_SPACING) | set(MTN_FORK)
SNAPS = 600


def _owner():
    return json.loads((sc.repo_root() / sc.LIONS_CPU_RELPATH).read_text())


def _forms():
    return sorted({v.formation for v in CAT_LIONS.variants})


def test_shared_book_stays_at_49_without_mtn_plays():
    assert set(CAT.ids).isdisjoint(MTN_IDS)
    assert CAT.get(MTN_SPACING[0]) is None and CAT.book_id == sc.SHARED_BOOK
    assert len(CAT.variants) == 49 and sc.contract_errors(CAT) == []
    assert len(pcd.catalog_families(CAT)) == 29
    assert {"mtn spacing", "mtn fork y corner"}.isdisjoint(pcd.catalog_families(CAT))
    for y in (40, 85, 95):
        assert {v.variant_id for v in lg.legal_set(CAT, db=None, yardline=y)[0]}.isdisjoint(MTN_IDS)


def test_lions_cpu_book_is_the_shared_book_plus_the_seven_mtn_plays():
    assert CAT_LIONS.book_id == sc.LIONS_CPU_BOOK
    excluded = set(_owner().get("exclude_inherited_variant_ids") or ())
    assert excluded <= set(CAT.ids) and excluded.isdisjoint(CAT_LIONS.ids)
    assert set(CAT.ids) - excluded < set(CAT_LIONS.ids)
    assert MTN_IDS == MTN_PASS <= set(CAT_LIONS.ids) - set(CAT.ids)
    assert len(CAT_LIONS.variants) == sc.EXPECTED_LIONS_CPU_VARIANTS and sc.contract_errors(CAT_LIONS) == []
    assert CAT_LIONS.version() == CAT.version() == "lions_civil_bucs_5wr_v4@15c120decc4b"
    fam = pcd.catalog_families(CAT_LIONS)
    assert len(fam) == 64 and {"mtn spacing", "mtn fork y corner", "hb off tackle", "qb draw", "x post return"} <= set(fam)
    owner = _owner()
    assert [v["variant_id"] for v in owner["pass_variants"][:7]] == MTN_SPACING + MTN_FORK
    assert owner["red_zone_variants"] and owner["designed_runs"]
    names = {(e["formation"], e["base_play"]) for e in owner["name_not_in_stock_data"]}
    assert {("Gun Bunch TE", "Mtn Spacing"), ("Gun Bunch TE", "Mtn Fork Y-Corner")} <= names
    stock = sc.stock_offense_plays()
    assert "Mtn Spacing" not in stock["Gun Bunch TE"] and "Mtn Fork Y-Corner" not in stock["Gun Bunch TE"]
    for vid in MTN_IDS:
        v = CAT_LIONS.get(vid)
        assert v is not None and v.group == sc.OWNER_PASS and v.field_zone == "any" and not v.is_run
        assert v.formation == "Gun Bunch TE" and not v.name_in_stock_data
        row = next(r for r in owner["pass_variants"] if r["variant_id"] == vid)
        assert v.variant_name == row["variant_name"] == row["macro_name"]
        assert [(a.guide_text, a.macro_marked) for a in v.adjustments] == [
            (a["guide_text"], bool(a["macro_marked"])) for a in row["adjustments"]
        ]
        assert v.raw["guide_setup_text"] == row["guide_setup_text"]


def test_lions_cpu_slot_does_not_change_the_shared_catalog():
    # Applying a subset of the real Mtn list only affects a Lions/CPU catalog copy.
    owner = copy.deepcopy(_owner())
    dropped = owner["pass_variants"].pop()
    cat = sc.with_lions_cpu_additions(sc.load(), owner, digest="test-slot")
    assert dropped["variant_id"] not in cat.ids
    assert dropped["variant_id"] not in CAT.ids
    excluded = set(owner.get("exclude_inherited_variant_ids") or ())
    assert set(CAT.ids) - excluded < set(cat.ids)
    assert len(cat.variants) == len(CAT_LIONS.variants) - 1
    assert sc.load_cached() is CAT or set(sc.load_cached().ids) == set(CAT.ids)
    assert set(sc.load_cached().ids).isdisjoint(MTN_IDS)


@pytest.mark.parametrize("league,make_session,book", [
    ("lions", _lions, sc.LIONS_CPU_BOOK),
    ("cpu", None, sc.LIONS_CPU_BOOK),
    ("online", _online, sc.SHARED_BOOK),
    ("titans", _titans, sc.SHARED_BOOK),
    ("giants", _giants, sc.LIONS_CPU_BOOK),
])
def test_catalog_for_call_routes_by_session(tmp_path, league, make_session, book):
    from cfb_coach.madden.scheme.coordinator import catalog_for_call

    db = _db(tmp_path, formations=_forms())
    sid = f"route-{league}"
    if make_session:
        make_session(db, sid)
    sit = _sit(1, 10, 40, session=sid)
    assert catalog_for_call(db, sit).book_id == book


def _simulate(db, session, opponent, n=SNAPS):
    from cfb_coach.madden import playcaller

    counts = collections.Counter()
    statuses = []
    fams = []
    fallbacks = 0
    books = set()
    cooldowns = []
    for i in range(n):
        down = 1 + i % 4
        dist = (10, 7, 3, 1)[i % 4]
        y = 15 + (i * 5) % 80          # 15..94, covers midfield, 10-20, and inside the 10
        call = playcaller.make_call(_sit(down, min(dist, 100 - y), y, session=session), opponent, db)
        sv = call.scheme_variant
        statuses.append(sv["status"])
        assert sv["status"] == "CALL" and sv.get("variant_id") and sv["variant_id"] != lg.NO_LEGAL, (i, sv)
        books.add(sv.get("catalog_book"))
        vid = sv["variant_id"]
        counts[vid] += 1
        fam = pcd.play_family_key(sv["base_play"])
        cd = sv.get("play_cooldown") or {}
        cooldowns.append(cd)
        if cd.get("fallback"):
            fallbacks += 1
        elif cd.get("active") and cd.get("window"):
            assert fam not in set(fams[-cd["window"]:]), (session, i, fam, fams[-cd["window"]:], cd)
        fams.append(fam)
        _close_snap(db, session, vid, y)
    return {
        "counts": counts, "statuses": statuses, "fams": fams, "fallbacks": fallbacks,
        "books": books, "cooldowns": cooldowns,
    }


def test_lions_and_cpu_call_mtn_plays_with_no_cooldown(tmp_path, civil_env):
    from cfb_coach.madden import playcaller

    db = _db(tmp_path, formations=_forms())
    _verify_all(db)
    for vid in MTN_IDS:
        lg.record_verified(db, CAT_LIONS, vid)

    _lions(db, "lions-sim")
    lions = _simulate(db, "lions-sim", "jaxon")
    cpu = _simulate(db, "cpu-sim", "cpu")

    for label, rec in (("lions", lions), ("cpu", cpu)):
        assert rec["statuses"].count("CALL") == SNAPS and lg.NO_LEGAL not in rec["statuses"], label
        assert rec["books"] == {sc.LIONS_CPU_BOOK}, (label, rec["books"])
        mtn_calls = sum(rec["counts"][vid] for vid in MTN_IDS)
        assert mtn_calls > 0, (label, rec["counts"])
        assert set(rec["counts"]) <= set(CAT_LIONS.ids)
        assert all(not (cd or {}).get("active") for cd in rec["cooldowns"]), label
        # Midfield legality includes the Mtn plays; inside the 10 does not.
        mid = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=db, yardline=40)[0]}
        rz = {v.variant_id for v in lg.legal_set(CAT_LIONS, db=db, yardline=95)[0]}
        assert MTN_IDS <= mid and mid == REG_LIONS
        assert RZ <= rz and rz.isdisjoint(MTN_IDS)
        assert rz == {v.variant_id for v in CAT_LIONS.redzone_variants}

    # One card still shows the owner's setup steps and macro flags verbatim.
    call = playcaller.make_call(_sit(1, 10, 35, session="lions-card"), "jaxon", db)
    # force a known Mtn variant onto the scorer? just check the catalog card path in a later unit-style loop
    from cfb_coach.madden.scheme import beliefs as bl
    from cfb_coach.madden.scheme import coordinator as co
    from cfb_coach.madden.scheme import design as dz
    from cfb_coach.madden.scheme import model as sm
    from tests.test_civil_scheme_only import _artifact
    s = co.Scorer(sm.load(_artifact(tmp_path / "card")), CAT_LIONS, dz.GeometrySource(enabled=False))
    bel = bl.combine({}, 0.1, 0, "none", [])
    v = CAT_LIONS.get("civil_t398_gun_bunch_te_mtn_spacing_1")
    top = s.score([v], down=1, distance=10, yardline=40, opponent_id="cpu", opponent_type="cpu",
                  belief=bel, offset_override={v.variant_id: 0.5})
    text = co.build_call(top[0], top, bel, book=None, shadow=False).format()
    for a in v.adjustments:
        assert a.guide_text in text
    assert "OWNER-ADDED PLAY" in text and "macro name: Mtn Spacing 1" in text


def test_chiefs_and_titans_never_call_mtn_and_keep_cooldowns(tmp_path, civil_env, ravens_env):
    db = _db(tmp_path, formations=_forms())
    _verify_all(db)
    _online(db, "chiefs-sim")
    _titans(db, "titans-sim")

    chiefs = _simulate(db, "chiefs-sim", "online_indianapolis_colts")
    titans = _simulate(db, "titans-sim", "titans_indianapolis_colts")

    for label, rec, window, league in (
        ("chiefs", chiefs, 3, "Chiefs"),
        ("titans", titans, 5, "Titans"),
    ):
        assert rec["statuses"].count("CALL") == SNAPS and lg.NO_LEGAL not in rec["statuses"], label
        assert rec["books"] == {sc.SHARED_BOOK}, (label, rec["books"])
        lions_only = set(CAT_LIONS.ids) - set(CAT.ids)
        assert set(rec["counts"]).isdisjoint(lions_only), (label, set(rec["counts"]) & lions_only)
        assert set(rec["counts"]) <= set(CAT.ids)
        # Family cooldown is on and the window matches; non-fallback snaps do not repeat inside the window
        # (already asserted inside _simulate). Confirm the metadata.
        active = [cd for cd in rec["cooldowns"] if cd.get("active")]
        assert active and all(cd.get("window") == window and cd.get("league") == league for cd in active), label

    # Chiefs defense also keeps the 3-snap family cooldown and never uses a civil Mtn play.
    def_fams, def_status, def_fallback = [], [], 0
    for i in range(SNAPS):
        call = _defense_call(db, "chiefs-sim", down=1 + i % 3, distance=5 + i % 8, yardline=30 + i % 40)
        rd = call.ravens_defense
        def_status.append(rd["status"])
        assert rd["status"] == "CALL", rd
        fam = pcd.defense_family_key(rd["candidate"]["base_play"])
        cd = rd.get("play_cooldown") or {}
        if cd.get("fallback"):
            def_fallback += 1
        else:
            assert fam not in set(def_fams[-3:]), (i, fam, def_fams[-3:], cd)
            assert cd.get("active") is True and cd.get("window") == 3 and cd.get("league") == "Chiefs"
        def_fams.append(fam)
    assert def_status.count("CALL") == SNAPS and "NO_LEGAL_RAVENS_CALL" not in def_status

    # Titans defense is still not cooled down.
    call = _defense_call(db, "titans-sim")
    # Titans sessions shouldn't be using the Chiefs defense path with an online opponent id;
    # the cooldown spec is what matters.
    assert pcd.cooldown_spec(db, "titans-sim", side="defense") is None
    assert pcd.cooldown_spec(db, "lions-absent", side="offense") is None
