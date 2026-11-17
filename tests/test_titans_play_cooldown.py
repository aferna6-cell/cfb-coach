"""Titans-only 5-snap offensive play-family cooldown (owner t390u)."""
from __future__ import annotations

import pytest

from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import play_cooldown as pcd
from cfb_coach.madden.scheme.coordinator import format_lines
from tests.test_civil_scheme_only import (
    CAT,
    _close_snap,
    _db,
    _sit,
    _verify_all,
    civil_env,  # noqa: F401 — pytest fixture
)

MESH = [v for v in CAT.variants if v.base_play == "Mesh"]
MESH_POST = [v for v in CAT.variants if v.base_play == "Mesh Post"]
MESH_IDS = {v.variant_id for v in MESH}
MESH_POST_IDS = {v.variant_id for v in MESH_POST}
INSIDE_ZONE = [v for v in CAT.variants if v.base_play == "Inside Zone"]

def _session(db, sid, *, league, dynasty=None, user_team=None, opponent="cpu"):
    dynasty = dynasty if dynasty is not None else league
    db.conn.execute(
        "INSERT INTO game_sessions (session_id, opponent_id, dynasty, started_ts, "
        "league, user_team, opponent_team, opponent_type) "
        "VALUES (?, ?, ?, '2026-11-15T21:00:00+00:00', ?, ?, 'Indianapolis Colts', 'human')",
        (sid, opponent, dynasty, league, user_team),
    )
    db.conn.commit()


def _titans(db, sid="titans1"):
    _session(db, sid, league="titans", user_team="Tennessee Titans",
             opponent="titans_indianapolis_colts")


def _online(db, sid="online1"):
    _session(db, sid, league="online", user_team="Kansas City Chiefs",
             opponent="online_indianapolis_colts")


def _lions(db, sid="lions1"):
    _session(db, sid, league=None, dynasty="primary", user_team="Detroit Lions",
             opponent="cpu")


def _giants(db, sid="giants1"):
    _session(db, sid, league="giants", user_team="New York Giants",
             opponent="giants_indianapolis_colts")


# --------------------------------------------------------------------------- family grouping
def test_family_keys_group_variants_not_distinct_plays():
    key = pcd.play_family_key
    # Mesh family: numbered variants, the Outside Bunch WR Post variant, and Redzone Mesh.
    mesh_keys = {pcd.family_of_variant(v) for v in MESH}
    assert mesh_keys == {"mesh"}
    assert key("Mesh 2") == "mesh"
    assert key("Mesh - Outside Bunch WR Post") == "mesh"
    # The red-zone play's catalog base_play is "Mesh" (variant name is "Redzone Mesh").
    assert pcd.family_of_variant(CAT.get("civil_rz_i_form_close_flex_mesh")) == "mesh"
    # Mesh Post is its own play, not a Mesh variant.
    assert {pcd.family_of_variant(v) for v in MESH_POST} == {"mesh post"}
    assert key("Mesh Post 1") == "mesh post"
    assert key("Mesh Post") != key("Mesh")
    # Z Spot vs Z Spot Shake stay distinct (Shake is part of the play name).
    assert key("Z Spot") == "z spot"
    assert key("Z Spot 1") == "z spot"
    assert key("Z Spot Shake") == "z spot shake"
    assert key("Z Spot Shake 1") == "z spot shake"
    assert key("Z Spot") != key("Z Spot Shake")
    # Return Motion / Corner-Flat are variant suffixes of the same in-game play.
    assert key("MTN Shuffle Verts Smash") == key("MTN Shuffle Verts Smash — Return Motion") == "mtn shuffle verts smash"
    assert key("MTN Cross HB Sneak") == key("MTN Cross HB Sneak — Corner/Flat") == "mtn cross hb sneak"
    # Inside Zone (and Same Side Zone) share a family across formations.
    assert len({pcd.family_of_variant(v) for v in INSIDE_ZONE}) == 1
    assert key("Inside Zone") == "inside zone"
    # Nearby run names stay distinct.
    assert key("Inside Zone") != key("HB Mid Zone") != key("HB Zone")
    assert key("HB Off Tackle") != key("Boomerang HB Off Tackle")
    assert key("QB Draw") != key("RPO Alert QB Draw") != key("HB Draw")
    # Hyphenated catalog spelling.
    assert key("Texas Y-Stutter Wheel") == key("Texas Y Stutter Wheel 2") == "texas y stutter wheel"
    # Every catalog variant lands in exactly one family.
    families = pcd.catalog_families(CAT)
    # 27 + the remaining t398u families PA Slide and HB Dive (HB Zone and Mesh already existed;
    # t414u dropped Mtn Spacing and Mtn Fork Y-Corner)
    assert len(families) == 29
    assert "mtn spacing" not in families and "mtn fork y corner" not in families
    assert {"pa slide", "hb dive", "hb zone", "mesh"} <= set(families)
    assert sum(len([v for v in CAT.variants if pcd.family_of_variant(v) == k]) for k in families) == 49


def test_apply_cooldown_blocks_family_and_falls_back_to_lru():
    mesh2 = CAT.get("civil_bucs_mesh_2_provisional")
    mesh5 = CAT.get("civil_bucs_mesh_5")
    post = MESH_POST[0]
    zspot = CAT.get("civil_5wr_z_spot_1")
    legal = [mesh2, mesh5, post, zspot]
    recent = ["mesh"]  # called last snap
    kept, info = pcd.apply_play_cooldown(legal, recent, cat=CAT)
    assert {v.variant_id for v in kept} == {post.variant_id, zspot.variant_id}
    assert info["fallback"] is False
    assert "Mesh" in info["blocked_families"]

    # Exhaust the legal set: only Mesh variants, all cooling down.
    kept, info = pcd.apply_play_cooldown([mesh2, mesh5], ["mesh", "clear deep"], cat=CAT)
    assert info["fallback"] is True
    assert {v.variant_id for v in kept} <= MESH_IDS
    assert info["fallback_family"] == "Mesh"
    assert "least-recently-used" in info["fallback_rule"]

    # LRU among several blocked families: newest-first history [A, B, C] → C is oldest.
    a, b, c = post, zspot, mesh2
    legal = [a, b, c]
    hist = [pcd.family_of_variant(a), pcd.family_of_variant(b), pcd.family_of_variant(c)]
    kept, info = pcd.apply_play_cooldown(legal, hist, window=5, cat=CAT)
    assert info["fallback"] is True
    assert {pcd.family_of_variant(v) for v in kept} == {pcd.family_of_variant(c)}
    assert info["fallback_family"] == "Mesh"


def test_is_titans_only_when_the_session_is_titans(tmp_path):
    db = _db(tmp_path)
    _titans(db, "t1")
    _online(db, "o1")
    _lions(db, "l1")
    assert pcd.is_titans_session(db, "t1")
    assert not pcd.is_titans_session(db, "o1")
    assert not pcd.is_titans_session(db, "l1")
    assert not pcd.is_titans_session(db, "missing")
    assert not pcd.is_titans_session(db, None)
    # extras only apply when there is no session row
    assert pcd.is_titans_session(db, "ghost", {"league": "titans"})
    assert not pcd.is_titans_session(db, "o1", {"league": "titans"})  # DB row wins


# --------------------------------------------------------------------------- live coordinator
def test_titans_mesh_call_blocks_all_mesh_variants_for_five_snaps(tmp_path, civil_env):
    from cfb_coach.madden import playcaller

    db = _db(tmp_path)
    _verify_all(db)
    _titans(db, "tmesh")
    first = playcaller.make_call(_sit(1, 10, 40, session="tmesh"), "cpu", db)
    assert first.scheme_variant["status"] == "CALL"
    # Force a Mesh call into history (whatever the first pick was).
    mesh_id = "civil_bucs_mesh_2_provisional"
    _close_snap(db, "tmesh", mesh_id, 40)

    picks = []
    for i in range(5):
        call = playcaller.make_call(_sit(1, 10, 35 + i, session="tmesh"), "cpu", db)
        vid = call.scheme_variant["variant_id"]
        assert vid not in MESH_IDS, (i, vid, call.scheme_variant.get("play_cooldown"))
        # Mesh Post is a different family and may still be called.
        picks.append(vid)
        _close_snap(db, "tmesh", vid, 35 + i)
        cd = call.scheme_variant.get("play_cooldown") or {}
        assert cd.get("active") is True
        assert cd.get("fallback") is False

    # After five non-Mesh offensive snaps the Mesh family is legal again.
    sixth = playcaller.make_call(_sit(1, 10, 40, session="tmesh"), "cpu", db)
    # It is allowed, not required, to be Mesh; the legal set must include Mesh again.
    assert "civil_bucs_mesh_2_provisional" in sixth.scheme_variant["candidates"]
    assert any(v in MESH_IDS for v in sixth.scheme_variant["candidates"])
    # And Mesh Post was never blocked by the Mesh call itself (candidates on snap 1 after Mesh).
    # Restart safety: history is the snaps table, not process memory.
    db2_families = pcd.recent_offense_families(db, "tmesh", CAT)
    assert db2_families[0] != "mesh"  # last 5 were the fillers
    assert "mesh" in db2_families  # the original Mesh call is still on the session


def test_lions_sessions_have_no_play_cooldown(tmp_path, civil_env):
    from cfb_coach.madden import playcaller

    db = _db(tmp_path)
    _verify_all(db)
    _lions(db, "lions1")
    _close_snap(db, "lions1", "civil_bucs_mesh_2_provisional", 40)
    call = playcaller.make_call(_sit(1, 10, 40, session="lions1"), "cpu", db)
    assert call.scheme_variant.get("play_cooldown") is None
    # Mesh remains a legal candidate immediately after a Mesh call.
    assert any(v in MESH_IDS for v in call.scheme_variant["candidates"])


def test_red_zone_exhaustion_falls_back_to_lru_and_logs(tmp_path, civil_env, monkeypatch):
    """Inside the 10 there are 7 red-zone plays; if every *currently legal* family is
    cooling down, call the least-recently-used legal one and log the fallback."""
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.scheme import legality as lg

    db = _db(tmp_path)
    _verify_all(db)
    _titans(db, "trz")
    # Restrict the live legal set to three red-zone families so a window of 5 can exhaust them.
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

    # Call the three families in order; history newest-first will be Mesh, QB Blast, HB Zone.
    order = [
        "civil_rz_pistol_bunch_te_hb_zone",
        "civil_rz_gun_empty_quads_qb_blast",
        "civil_rz_i_form_close_flex_mesh",
    ]
    for vid in order:
        _close_snap(db, "trz", vid, 95)

    call = playcaller.make_call(_sit(1, 5, 95, session="trz"), "cpu", db)
    sv = call.scheme_variant
    assert sv["status"] == "CALL"
    cd = sv["play_cooldown"]
    assert cd["active"] is True and cd["fallback"] is True
    # LRU among the three is HB Zone (called earliest).
    assert cd["fallback_family"] == "HB Zone"
    assert sv["variant_id"] == "civil_rz_pistol_bunch_te_hb_zone"
    assert sv["base_play"] == "HB Zone"
    assert "PLAY COOLDOWN FALLBACK" in sv["explanation"]
    assert any("PLAY COOLDOWN FALLBACK" in line for line in format_lines(call))
    # Revalidation still accepts the fallback (cooldown is not a book-legality failure).
    lg.revalidate_call(db, call, CAT, yardline=95)
