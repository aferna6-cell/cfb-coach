"""The finalized card describes the attached call, never a second action search."""
from __future__ import annotations

from copy import deepcopy

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden.model import experimental_live as xl
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_presnap_advice as advice
from cfb_coach.madden.model import offense_route_ranker as rr
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.situation import parse_madden_situation

PAIR = ("Gun Trips X Nasty", "Hi Lo Cross")
HOT = {"id": "hot_slant_vs_man", "kind": "hot_route", "label": "Hot route WR1 → Slant",
       "buttons": "test", "why": "selected live man adjustment"}
PROTECT = {"id": "max_protect_vs_pressure", "kind": "pass_protection",
           "label": "Max protect", "buttons": "test", "why": "selected protection"}


@pytest.fixture
def finalize(monkeypatch):
    # Fix only the numeric estimate; keep actual structure projection, decision,
    # card builder and QB panel. No training, DB snapshots or timing dependency.
    monkeypatch.setattr(sm, "score", lambda *a, **kw: sm.Score(.6, .6, 0., .1, "test"))
    monkeypatch.setattr(rr, "_write_shadow", lambda rec: None)
    actual_decide = od.decide

    searches = []

    def stock_only(*args, **kwargs):
        searches.append(kwargs["allow_adjustments"])
        # A deterministic competing decision: the independent search would
        # choose stock. This must not replace the attached hot route or macro.
        kwargs["allow_adjustments"] = False
        return actual_decide(*args, **kwargs)

    monkeypatch.setattr(od, "decide", stock_only)

    def run(call, *, priors=False):
        ctx = rr.RouteContext("route_aware", art=sm.StructuralArtifact(), applied=True,
                              priors=priors, mix={"cover_1_man": 1.0})
        result = rr.finalize(ctx, call, sit=parse_madden_situation("2&8 my 35"),
                             opponent_id="cpu", game_id=None, snap_seq=None)
        assert "display_error" not in result, result
        call.ml_experimental = {"route_aware": result}
        return result["card"]
    run.searches = searches
    return run


def _call(adjustment=None, macro=None):
    call = MaddenCall("offense", *PAIR, "No adj", "", adjustment=deepcopy(adjustment), macro=macro)
    call.ml_offense_action = {"kind": "macro" if macro else "adjustment" if adjustment else "none",
                              "id": macro or (adjustment or {}).get("id"),
                              "reason": (adjustment or {}).get("why") or "run the play unchanged"}
    return call


def _panel(call):
    controller = object.__new__(LivePlayController)
    controller.last_call, controller.ended = call, False
    return controller._qb_reads_state()


@pytest.mark.parametrize("priors", [False, True])
def test_attached_joint_hot_route_drives_card_and_qb_reads(finalize, tmp_path, monkeypatch, priors):
    monkeypatch.setenv("HOME", str(tmp_path))
    db = CoachDB(tmp_path / "test.db")
    try:
        # Exercise the actual reconstruction seam that precedes finalize.
        call = xl.rebuild_offense_attachments(
            formation=PAIR[0], play=PAIR[1], sit=parse_madden_situation("2&8 my 35"),
            book={PAIR[0]: [PAIR[1]]}, active=[], db=db, opponent_id="cpu", rationale="test",
            model_action_policy=True,
            preset_decision={"kind": "adjustment", "id": HOT["id"], "adjustment": deepcopy(HOT),
                             "reason": HOT["why"]})
    finally:
        db.close()
    card = finalize(call, priors=priors)
    assert card["ADJUSTMENT"] == call.adjustment["label"] == HOT["label"]
    assert card["CALL ACTION"]["adjustment"] == call.adjustment
    assert card["CALL ACTION"] == advice.call_action_identity(call)
    assert "WR1:drag->slant" in card["WHY"] and HOT["why"] in card["WHY"]
    assert "gain +0.0%" not in card["WHY"]
    assert card["READ BASIS"].startswith("INFERRED")
    reads = [card[k] for k in ("PRIMARY READ", "SECONDARY READ", "CHECKDOWN")]
    assert call.read_or_user == " → ".join(reads)
    assert all("drag" not in r for r in reads if r.startswith("WR1 "))
    assert any(r.startswith("WR1 slant") for r in reads)
    assert finalize.searches == [False], "finalize must not choose another adjustment"
    assert [r["text"] for r in _panel(call)["reads"]] == [r for r in reads if not r.startswith("UNKNOWN")]


def test_selected_no_action_stays_stock(finalize):
    call = _call()
    card = finalize(call)
    assert card["ADJUSTMENT"] == od.NO_ADJUSTMENT
    assert card["CALL ACTION"] == advice.call_action_identity(call)
    assert card["READ BASIS"].startswith("DOCUMENTED")
    assert _panel(call)["status"] == "known"


@pytest.mark.parametrize("adjustment,macro", [
    (PROTECT, None),
    ({"id": "package", "kind": "multi_adjustment", "label": "selected package",
      "parts": [HOT["id"], PROTECT["id"]]}, None),
    ({"id": "hot_hb_flat_vs_pressure", "kind": "hot_route", "label": "HB flat"}, None),
    (None, "selected_catalog_macro"),
])
def test_unrepresentable_selected_effects_never_leak_stock_reads(finalize, adjustment, macro):
    call = _call(adjustment, macro)
    card = finalize(call)
    assert card["ADJUSTMENT"] == ("MACRO " + macro if macro else adjustment["label"])
    assert card["CALL ACTION"] == advice.call_action_identity(call)
    assert "receiver effects are not representable" in card["WHY"]
    assert _panel(call)["status"] == "unknown" and not _panel(call)["reads"]


@pytest.mark.parametrize("change", ["adjustment", "macro", "missing_identity"])
def test_same_play_stale_action_card_is_rejected(finalize, change):
    call = _call(HOT)
    card = finalize(call)
    if change == "adjustment":
        # Even an in-place mutation must not mutate the finalized snapshot.
        call.adjustment["label"] = "different action"
        call.adjustment["target"] = "WR2"
    elif change == "macro":
        call.macro = "new macro"
    else:
        card.pop("CALL ACTION")
    panel = _panel(call)
    assert panel["status"] == "unavailable" and panel["reads"] == []
    assert "action" in panel["note"]


def test_explicit_attached_route_parameters_override_catalog_defaults(finalize):
    call = _call({**HOT, "target": "WR2", "route": "Slant", "label": "Hot route WR2 → Slant"})
    card = finalize(call)
    assert "WR2:in->slant" in card["WHY"]
    assert "WR1:drag->slant" not in card["WHY"]
    assert card["CALL ACTION"]["adjustment"]["target"] == "WR2"
