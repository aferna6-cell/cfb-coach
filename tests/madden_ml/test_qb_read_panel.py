"""Issue #118 deliverable A: the compact Madden HTML shows the read ORDER for the call on screen, only from its finalized card.

SYNTHETIC artifacts; proves display and safety, not that any receiver is open.
"""
from __future__ import annotations

import types

import pytest

from cfb_coach.live_server import render_live_html
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_presnap_advice as advice
from cfb_coach.madden.model import offense_route_ranker as rr
from tests.madden_ml.test_defense_knowledge_live import _ctrl, _prime, _world, artifacts, rows  # noqa: F401

KNOWN = ("Gun Trips X Nasty", "Hi Lo Cross")        # 5 documented routes with a read order
UNKNOWN = ("Pistol Trips", "Smash")                  # no route annotations


@pytest.fixture()
def world(tmp_path, monkeypatch, artifacts):
    db = _world(tmp_path, monkeypatch, artifacts, "default")
    yield db, artifacts["default"]
    db.close()


def _card(art, fp):
    sit = od.Situation(down=1, distance=10, yardline=25, opponent_id="cpu", opponent_type="cpu", mix={})
    return advice.build_card(od.decide(art, sit, [fp], mode="live", allow_adjustments=False), {})


def _stub(fp, ra, side="offense"):
    return types.SimpleNamespace(formation=fp[0], play=fp[1], side=side, ml_experimental={"route_aware": ra}, format=lambda: f"{fp[0]} / {fp[1]}")


def _state(c, call):
    c.last_call = call
    return c._qb_reads_state()


def test_known_play_shows_ordered_reads_and_no_reasoning(world):
    db, art = world
    c = _ctrl(db)
    q = _state(c, _stub(KNOWN, {"mode": "route_aware", "applied": True, "card": _card(art, KNOWN)}))
    assert q["status"] == "known" and q["label"] == "ROUTE-AWARE"
    assert [r["order"] for r in q["reads"]][:2] == ["1st", "2nd"] and all(r["text"] for r in q["reads"])
    assert "why" not in str(q).lower() and q["basis"] in ("documented order", "inferred for expected coverage", "name-level hint")


def test_unknown_play_never_fabricates_a_target(world):
    db, art = world
    c = _ctrl(db)
    q = _state(c, _stub(UNKNOWN, {"mode": "route_aware", "applied": True, "card": _card(art, UNKNOWN)}))
    assert q["status"] == "unknown" and q["reads"] == [] and "Target unknown" in q["note"]


def test_preview_fallback_missing_and_mismatched_cards(world):
    db, art = world
    c = _ctrl(db)
    card = _card(art, KNOWN)
    assert "PREVIEW" in _state(c, _stub(KNOWN, {"mode": "route_aware_preview", "applied": False, "card": card}))["label"]
    f = _state(c, _stub(KNOWN, {"mode": "route_aware", "applied": False, "fallback_reason": "latency budget exhausted", "card": card}))
    assert f["status"] == "fallback" and "latency budget exhausted" in f["note"] and f["reads"] == []
    assert _state(c, types.SimpleNamespace(formation=KNOWN[0], play=KNOWN[1], side="offense", ml_experimental={}, format=lambda: ""))["status"] == "unavailable"
    stale = _state(c, _stub(UNKNOWN, {"mode": "route_aware", "applied": True, "card": card}))     # a DIFFERENT play's card
    assert stale["status"] == "unavailable" and stale["reads"] == []


def test_state_is_cleared_after_game_over_and_defense_calls(world):
    db, art = world
    c = _ctrl(db)
    call = _stub(KNOWN, {"mode": "route_aware", "applied": True, "card": _card(art, KNOWN)})
    assert _state(c, call)["status"] == "known"
    assert _state(c, _stub(KNOWN, {"card": _card(art, KNOWN)}, side="defense")) is None
    c.last_call = call
    c.ended = True
    assert c._qb_reads_state() is None


def test_live_flow_and_undo_never_carry_the_previous_calls_reads(world):
    db, art = world
    _prime(db)
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    q1 = c.state()["qb_reads"]
    assert q1["status"] in ("known", "unknown", "fallback", "unavailable") and (q1["status"] != "known" or q1["reads"])
    c.result_and_call(outcome="+6", sit_raw="2&4 my 31", their="Cover 3 Cloud")
    q2 = c.state()["qb_reads"]
    assert q2 is not None
    shown = (c.last_call.formation, c.last_call.play)
    assert (c.last_call.ml_experimental or {}).get("route_aware", {}).get("card", {}).get("PLAY", shown[1]) == shown[1]
    c.undo_last()
    q3 = c.state()["qb_reads"]
    assert q3 is None or q3["status"] in ("known", "unknown", "fallback", "unavailable")


def test_compact_html_has_the_panel_renders_with_textcontent_and_normal_mode_does_not(world):
    db, art = world
    c = _ctrl(db)
    html = render_live_html(c)
    assert 'id="qb-reads"' in html and "renderQbReads(st.qb_reads)" in html
    body = html[html.index("function renderQbReads"):html.index("function renderState")]
    assert "innerHTML" not in body and "textContent" in body and "createTextNode" in body       # escaped by construction
    c.brand = "College Football"
    assert 'id="qb-reads"' not in render_live_html(c)


def test_hostile_text_is_data_not_markup(world):
    db, art = world
    c = _ctrl(db)
    card = {**_card(art, KNOWN), "PRIMARY READ": "<img src=x onerror=alert(1)> Z", }
    q = _state(c, _stub(KNOWN, {"mode": "route_aware", "applied": True, "card": card}))
    assert q["reads"][0]["text"] == "<img src=x onerror=alert(1)> Z"          # passed as a string; the page sets it via textContent only
