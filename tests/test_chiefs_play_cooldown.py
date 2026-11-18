"""Chiefs online 3-snap play-family cooldown on offense and defense (owner t403u)."""
from __future__ import annotations

import pytest

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.scheme import play_cooldown as pcd
from cfb_coach.madden.scheme.coordinator import format_lines
from cfb_coach.situation import Situation
from tests.test_civil_scheme_only import (
    CAT,
    _close_snap,
    _db,
    _sit,
    _verify_all,
    civil_env,  # noqa: F401 — pytest fixture
)
from tests.test_titans_play_cooldown import MESH_IDS, MESH_POST, _giants, _lions, _online, _titans

CAT_D = rc.load()
CAT_STEEL = rc.load_lions_franchise()
CAT_LIVE = rc.load_61_defense()
COVER2 = [c for c in CAT_D.candidates if c.base_play == "Cover 2 Match"]
QUARTERS = [c for c in CAT_D.candidates if c.base_play == "Cover 4 Quarters"]


@pytest.fixture
def ravens_env(monkeypatch):
    from cfb_coach.madden.ravens_defense import mode as rmd
    monkeypatch.setenv(rmd.ENV_MODE, rmd.RAVENS)


def _dsit(session, down=1, distance=10, yardline=40):
    s = Situation(raw="t", down=down, distance=distance, yardline=yardline)
    s.side = "defense"
    s.extras = {"session_id": session}
    return s


def _log_defense(db, session, candidate_id):
    from cfb_coach.madden.ravens_defense.coordinator import log_decision
    c = CAT_LIVE.by_id().get(candidate_id) or CAT_D.by_id()[candidate_id]
    log_decision(db.conn, {
        "session_id": session, "status": "CALL",
        "candidate_id": c.candidate_id, "call_id": c.call_id,
        "formation": c.formation, "base_play": c.base_play,
    })


def _defense_call(db, session, **kw):
    from cfb_coach.madden import playcaller
    return playcaller.make_call(_dsit(session, **kw), "online_indianapolis_colts", db)


# --------------------------------------------------------------------------- session detection / windows
def test_cooldown_spec_windows_and_sides(tmp_path):
    db = _db(tmp_path)
    _titans(db, "t1")
    _online(db, "o1")
    _lions(db, "l1")
    _giants(db, "g1")

    t_off = pcd.cooldown_spec(db, "t1", side="offense")
    t_def = pcd.cooldown_spec(db, "t1", side="defense")
    assert t_off == {"window": 5, "league": "Titans", "league_key": "titans", "side": "offense"}
    assert t_def is None  # Titans cooldown is still offense-only

    c_off = pcd.cooldown_spec(db, "o1", side="offense")
    c_def = pcd.cooldown_spec(db, "o1", side="defense")
    assert c_off["window"] == c_def["window"] == 3
    assert c_off["league"] == c_def["league"] == "Chiefs"
    assert c_off["side"] == "offense" and c_def["side"] == "defense"

    g_off = pcd.cooldown_spec(db, "g1", side="offense")
    g_def = pcd.cooldown_spec(db, "g1", side="defense")
    assert g_off == {"window": 3, "league": "Giants", "league_key": "giants", "side": "offense"}
    assert g_def is None  # Giants cooldown is offense-only

    assert pcd.cooldown_spec(db, "l1", side="offense") is None
    assert pcd.cooldown_spec(db, "l1", side="defense") is None
    assert pcd.cooldown_spec(db, "missing", side="offense") is None

    assert pcd.is_online_session(db, "o1")
    assert not pcd.is_online_session(db, "t1")
    assert not pcd.is_online_session(db, "l1")
    assert pcd.is_online_session(db, "ghost", {"league": "online"})
    assert not pcd.is_online_session(db, "t1", {"league": "online"})  # DB row wins
    assert pcd.is_titans_session(db, "t1")
    assert not pcd.is_titans_session(db, "o1")


def test_defense_families_group_zone_it_and_shared_base_plays():
    assert len(QUARTERS) == 2
    assert {pcd.family_of_defense_candidate(c) for c in QUARTERS} == {"cover 4 quarters"}
    assert {c.candidate_id for c in QUARTERS} == {"match_quarters", "match_quarters+zone_it"}
    # match_cover2 and nickel_cover2 share the Cover 2 Match family (Titans base-play semantics)
    assert len(COVER2) == 2
    assert {pcd.family_of_defense_candidate(c) for c in COVER2} == {"cover 2 match"}
    cover9 = [c for c in CAT_D.candidates if c.base_play == "Cover 9 Show 2"]
    assert len(cover9) == 2
    assert {pcd.family_of_defense_candidate(c) for c in cover9} == {"cover 9 show 2"}
    # Distinct base plays stay distinct.
    assert pcd.defense_family_key("Cover 3 Match") != pcd.defense_family_key("Cover 3 Cloud")
    assert pcd.defense_family_key("Cover 4 Quarters") != pcd.defense_family_key("Cover 4 Palms")
    assert pcd.defense_family_key("Cover 4 Quarters") != pcd.defense_family_key("Cover 4 Zone")
    assert pcd.defense_family_key("DB Fire 2") == "db fire 2"
    assert pcd.defense_family_key("Cover 9 Show 2") == "cover 9 show 2"
    # Offense-style trailing-number stripping must not be applied to these names.
    assert pcd.defense_family_key("DB Fire 2") != pcd.play_family_key("DB Fire 2")


def test_apply_cooldown_window_3_and_defense_lru():
    mesh2 = CAT.get("civil_bucs_mesh_2_provisional")
    post = MESH_POST[0]
    zspot = CAT.get("civil_5wr_z_spot_1")
    dive = CAT.get("civil_t398_pistol_strong_y_off_close_hb_dive") or CAT.get(
        next(v.variant_id for v in CAT.variants if pcd.family_of_variant(v) == "hb dive")
    )
    legal = [mesh2, post, zspot, dive]
    recent = ["mesh"]
    kept, info = pcd.apply_play_cooldown(legal, recent, window=3, cat=CAT, league="Chiefs")
    assert info["window"] == 3 and info["league"] == "Chiefs" and info["fallback"] is False
    assert {pcd.family_of_variant(v) for v in kept} == {"mesh post", "z spot", pcd.family_of_variant(dive)}

    # Window of 3: a family four snaps ago is legal again.
    hist = ["mesh post", "z spot", pcd.family_of_variant(dive), "mesh"]  # newest first
    kept, info = pcd.apply_play_cooldown(legal, hist, window=3, cat=CAT, league="Chiefs")
    assert "mesh" not in {pcd.family_of_variant(v) for v in (post, zspot, dive)}  # sanity
    assert any(pcd.family_of_variant(v) == "mesh" for v in kept)
    assert "mesh post" not in {pcd.family_of_variant(v) for v in kept}

    # Defense LRU: three legal families, all cooling down, oldest wins.
    a, b, c = (
        CAT_D.by_id()["match_cover3"],
        CAT_D.by_id()["nickel_cloud"],
        CAT_D.by_id()["dime_dbfire2"],
    )
    # Also include the Zone It sibling of a family not in this legal set, and Cover 2 both packages.
    legal_d = [a, b, c]
    hist_d = ["cover 3 match", "cover 3 cloud", "db fire 2"]  # newest first → LRU is DB Fire 2
    kept, info = pcd.apply_play_cooldown(
        legal_d, hist_d, window=3, family_of=pcd.family_of_defense_candidate,
        label_of=lambda k: {"cover 3 match": "Cover 3 Match", "cover 3 cloud": "Cover 3 Cloud",
                            "db fire 2": "DB Fire 2"}[k],
        league="Chiefs",
    )
    assert info["fallback"] is True and info["window"] == 3 and info["league"] == "Chiefs"
    assert {pcd.family_of_defense_candidate(v) for v in kept} == {"db fire 2"}
    assert info["fallback_family"] == "DB Fire 2"
    assert "3-snap Chiefs cooldown" in info["fallback_rule"]

    # Zone It of a blocked family is also blocked.
    legal_q = list(QUARTERS) + [CAT_D.by_id()["match_cover3"]]
    kept, info = pcd.apply_play_cooldown(
        legal_q, ["cover 4 quarters"], window=3, family_of=pcd.family_of_defense_candidate, league="Chiefs")
    assert {c.candidate_id for c in kept} == {"match_cover3"}


# --------------------------------------------------------------------------- live offense
def test_chiefs_offense_mesh_blocked_for_three_snaps(tmp_path, civil_env):
    from cfb_coach.madden import playcaller

    db = _db(tmp_path)
    _verify_all(db)
    _online(db, "cmesh")
    _close_snap(db, "cmesh", "civil_bucs_mesh_2_provisional", 40)

    for i in range(3):
        call = playcaller.make_call(_sit(1, 10, 35 + i, session="cmesh"), "cpu", db)
        vid = call.scheme_variant["variant_id"]
        assert vid not in MESH_IDS, (i, vid, call.scheme_variant.get("play_cooldown"))
        _close_snap(db, "cmesh", vid, 35 + i)
        cd = call.scheme_variant.get("play_cooldown") or {}
        assert cd.get("active") is True and cd.get("window") == 3 and cd.get("league") == "Chiefs"
        assert cd.get("fallback") is False

    # After three non-Mesh offensive snaps the Mesh family is legal again.
    fourth = playcaller.make_call(_sit(1, 10, 40, session="cmesh"), "cpu", db)
    assert any(v in MESH_IDS for v in fourth.scheme_variant["candidates"])
    # Titans still uses a window of 5: three fillers would not yet free Mesh there.
    assert pcd.CHIEFS_COOLDOWN_SNAPS == 3 and pcd.TITANS_COOLDOWN_SNAPS == 5


def test_chiefs_red_zone_exhaustion_falls_back_to_lru(tmp_path, civil_env, monkeypatch):
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.scheme import legality as lg

    db = _db(tmp_path)
    _verify_all(db)
    _online(db, "crz")
    subset = [
        CAT.get("civil_rz_pistol_bunch_te_hb_zone"),
        CAT.get("civil_rz_gun_empty_quads_qb_blast"),
        CAT.get("civil_rz_i_form_close_flex_mesh"),
    ]
    real_legal_set = lg.legal_set

    def _restricted(cat, **kw):
        legal, verdicts = real_legal_set(cat, **kw)
        keep = {v.variant_id for v in subset}
        return [v for v in legal if v.variant_id in keep], verdicts

    monkeypatch.setattr(lg, "legal_set", _restricted)
    order = [
        "civil_rz_pistol_bunch_te_hb_zone",
        "civil_rz_gun_empty_quads_qb_blast",
        "civil_rz_i_form_close_flex_mesh",
    ]
    for vid in order:
        _close_snap(db, "crz", vid, 95)

    call = playcaller.make_call(_sit(1, 5, 95, session="crz"), "cpu", db)
    sv = call.scheme_variant
    assert sv["status"] == "CALL"
    cd = sv["play_cooldown"]
    assert cd["active"] is True and cd["fallback"] is True and cd["window"] == 3
    assert cd["league"] == "Chiefs"
    assert cd["fallback_family"] == "HB Zone"
    assert sv["variant_id"] == "civil_rz_pistol_bunch_te_hb_zone"
    assert "PLAY COOLDOWN FALLBACK" in sv["explanation"]
    assert any("PLAY COOLDOWN FALLBACK" in line for line in format_lines(call))
    lg.revalidate_call(db, call, CAT, yardline=95)


# --------------------------------------------------------------------------- live defense
def test_chiefs_defense_family_blocked_for_three_snaps(tmp_path, civil_env, ravens_env):
    db = _db(tmp_path)
    _online(db, "cdef")
    # Seed Cover 3 Match (and confirm Zone It is not a separate escape hatch).
    _log_defense(db, "cdef", "match_cover3")
    blocked = "cover 3 match"
    seen = []
    for i in range(3):
        call = _defense_call(db, "cdef", yardline=40 + i)
        rd = call.ravens_defense
        assert rd["status"] == "CALL", rd
        cd = rd.get("play_cooldown") or {}
        assert cd.get("active") is True and cd.get("window") == 3 and cd.get("league") == "Chiefs"
        assert cd.get("fallback") is False
        fam = pcd.defense_family_key(rd["candidate"]["base_play"])
        assert fam != blocked, (i, rd["candidate"], cd)
        # The filtered legal set must not include the blocked family.
        ranked_fams = {pcd.defense_family_key((CAT_LIVE.by_id().get(r["candidate_id"]) or CAT_STEEL.by_id().get(r["candidate_id"]) or CAT_D.by_id()[r["candidate_id"]]).base_play) for r in rd["ranked"]}
        assert blocked not in ranked_fams
        seen.append(fam)
    # After three other defensive calls, Cover 3 Match is legal again.
    fourth = _defense_call(db, "cdef", yardline=50)
    rd = fourth.ravens_defense
    assert rd["status"] == "CALL"
    ranked_fams = {pcd.defense_family_key((CAT_LIVE.by_id().get(r["candidate_id"]) or CAT_STEEL.by_id().get(r["candidate_id"]) or CAT_D.by_id()[r["candidate_id"]]).base_play) for r in rd["ranked"]}
    assert blocked in ranked_fams
    # History is the decisions table, not process memory.
    hist = pcd.recent_defense_families(db, "cdef")
    assert blocked in hist and hist[0] != blocked


def test_chiefs_defense_exhaustion_falls_back_to_lru_and_logs(tmp_path, civil_env, ravens_env, monkeypatch):
    from cfb_coach.madden.ravens_defense import legality as lg
    from cfb_coach.madden.ravens_defense.coordinator import card_lines

    db = _db(tmp_path)
    _online(db, "cdefb")
    subset = [CAT_LIVE.by_id()[i] for i in ("match_cover3", "nickel_cloud", "even61_base_swblitz_61_murple")]
    real = lg.legal_set

    def _restricted(*a, **k):
        legal, verdicts = real(*a, **k)
        keep = {c.candidate_id for c in subset}
        return [c for c in legal if c.candidate_id in keep], verdicts

    monkeypatch.setattr(lg, "legal_set", _restricted)
    # Newest-first history will be Cover 3 Match, Cover 3 Cloud, Sam Will Blitz → LRU is Sam Will Blitz.
    for cid in ("even61_base_swblitz_61_murple", "nickel_cloud", "match_cover3"):
        _log_defense(db, "cdefb", cid)

    call = _defense_call(db, "cdefb")
    rd = call.ravens_defense
    assert rd["status"] == "CALL"
    cd = rd["play_cooldown"]
    assert cd["active"] is True and cd["fallback"] is True and cd["window"] == 3
    assert cd["fallback_family"] == "Sam Will Blitz"
    assert rd["candidate"]["candidate_id"] == "even61_base_swblitz_61_murple"
    assert "PLAY COOLDOWN FALLBACK" in "\n".join(rd["card"])
    assert any("PLAY COOLDOWN FALLBACK" in line for line in card_lines(rd))


def test_titans_and_lions_defense_have_no_play_cooldown(tmp_path, civil_env, ravens_env):
    db = _db(tmp_path)
    _titans(db, "tdef")
    _lions(db, "ldef")
    _log_defense(db, "tdef", "match_cover3")
    _log_defense(db, "ldef", "match_cover3")
    for sid in ("tdef", "ldef"):
        call = _defense_call(db, sid)
        rd = call.ravens_defense
        assert rd["status"] == "CALL"
        assert rd.get("play_cooldown") is None
        # The just-called family remains in the scored legal set.
        ranked_fams = {pcd.defense_family_key((CAT_LIVE.by_id().get(r["candidate_id"]) or CAT_STEEL.by_id().get(r["candidate_id"]) or CAT_D.by_id()[r["candidate_id"]]).base_play) for r in rd["ranked"]}
        assert "cover 3 match" in ranked_fams


# --------------------------------------------------------------------------- simulation: no repeats within 3, 0 NO_LEGAL
def test_chiefs_game_sim_no_repeat_within_3_and_zero_no_legal(tmp_path, civil_env, ravens_env):
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.scheme.legality import NO_LEGAL

    db = _db(tmp_path)
    _verify_all(db)
    _online(db, "csim")

    off_fams, def_fams = [], []
    off_status, def_status = [], []
    off_fallback = def_fallback = 0

    for i in range(36):
        call = playcaller.make_call(_sit(1 + i % 4, (10, 7, 3, 1)[i % 4], 25 + (i * 3) % 50, session="csim"), "cpu", db)
        sv = call.scheme_variant
        off_status.append(sv["status"])
        assert sv["status"] == "CALL", sv
        fam = pcd.play_family_key(sv["base_play"])
        cd = sv.get("play_cooldown") or {}
        if cd.get("fallback"):
            off_fallback += 1
        else:
            assert fam not in set(off_fams[-3:]), (i, fam, off_fams[-3:], cd)
        off_fams.append(fam)
        _close_snap(db, "csim", sv["variant_id"], 25 + (i * 3) % 50)

    for i in range(36):
        call = _defense_call(db, "csim", down=1 + i % 3, distance=5 + i % 8, yardline=30 + i % 40)
        rd = call.ravens_defense
        def_status.append(rd["status"])
        assert rd["status"] == "CALL", rd
        fam = pcd.defense_family_key(rd["candidate"]["base_play"])
        cd = rd.get("play_cooldown") or {}
        if cd.get("fallback"):
            def_fallback += 1
        else:
            assert fam not in set(def_fams[-3:]), (i, fam, def_fams[-3:], cd)
        def_fams.append(fam)

    assert NO_LEGAL not in off_status and "NO_LEGAL_RAVENS_CALL" not in def_status
    assert off_status.count("CALL") == 36 and def_status.count("CALL") == 36
    # Midfield, 10+ families, window 3: fallback should not be needed.
    assert off_fallback == 0 and def_fallback == 0
    assert len(set(off_fams)) >= 6 and len(set(def_fams)) >= 4
