"""One defensive SHADOW decision, and one fully traced live offensive decision (inputs -> coverage mixture -> ranking -> adjustments ->
transformed play -> reads -> UI state -> logged result). Synthetic artifact and hand-fed history: this shows the chain is connected and consistent,
not that the football hypotheses win in Madden."""
from __future__ import annotations

import json

import pytest

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import defense_shadow as dsh
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_route_ranker as rr
from cfb_coach.madden.model import offense_structural_model as sm
from tests.madden_ml.test_football_priors import live_artifacts, art, rows, sit  # noqa: F401

import tests.madden_ml.test_offense_intel_model as T
from cfb_coach.madden.model import offense_assignments as oa

SNAP = json.loads(oa.SNAPSHOT_PATH.read_text(encoding="utf-8"))["sides"]["defense"]["formations"]
LEGAL = [(f, p) for f, ps in SNAP.items() for p in ps]


def test_defensive_shadow_decision_is_legal_distinct_per_opponent_tendency_and_never_applied(tmp_path):
    mesh = dsh.shadow_decision(LEGAL, ["Mesh"] * 8)
    smash = dsh.shadow_decision(LEGAL, ["Smash", "Smash", "Four Verticals", "Smash"] * 2)
    screens = dsh.shadow_decision(LEGAL, ["HB Slip Screen", "Quick Slants"] * 4)
    for r in (mesh, smash, screens):
        assert r["applied"] is False and r["kind"] == "shadow_decision_not_an_outcome" and r["ranked"] > 20
        assert all((t["formation"], t["play"]) in set(LEGAL) for t in r["top"])                 # only installed (legal) calls
        assert r["latency_ms"] < 150
    tops = [tuple((t["formation"], t["play"]) for t in r["top"]) for r in (mesh, smash, screens)]
    assert len(set(tops)) == 3                                                                    # the opponent's tendencies change the ranking
    assert smash["top"][0]["offense_edge_allowed"] < 0 and smash["top"][0]["stops"]               # grounded in a named concept it stops
    assert dsh.shadow_decision(LEGAL, [])["evidence_snaps"] == 0                                  # no history: shrinks to the uniform prior, still answers
    p = dsh.write_shadow(smash, tmp_path / "d.jsonl")
    assert json.loads(p.read_text().splitlines()[0])["applied"] is False


def test_thin_evidence_cannot_dominate_the_defensive_shadow():
    one = dsh.offense_distribution(["Mesh"])
    many = dsh.offense_distribution(["Mesh"] * 40)
    assert one["mesh"] < many["mesh"] < 1.0 and one["mesh"] < 0.2


def test_full_trace_of_one_live_offensive_decision(art, monkeypatch, tmp_path):
    """coverage probabilities -> ranked plays -> selected adjustments -> transformed play -> updated reads -> UI state -> logged result."""
    from cfb_coach.db import CoachDB
    from cfb_coach.live_server import LivePlayController
    from cfb_coach.madden.model import offense_presnap_advice as adv
    mesh = ("Gun Trips X Nasty", "Hi Lo Cross")
    s0 = sm.structure_any(*mesh)

    class Stub:                                           # learned hot-route gain stub (the real model rejects hot routes as out of support)
        def __init__(self, st):
            pass

        def score(self, mod):
            return 0.05 * len(mod.adjustments), 0.01, []

    monkeypatch.setattr(od.sm, "DeltaScorer", lambda artifact, st, **k: Stub(st))
    trace = {}
    for look, mix in (("Cover 1 (man)", {"cover_1_unspecified": 0.7, "cover_2_man": 0.3}), ("Cover 4 Quarters", {"cover_4_quarters": 1.0})):
        # 1) coverage probabilities (time-correct mixture) -> 2) ranked plays
        d = od.decide(art, sit(mix), T.ANNOTATED, mode="research", margin=0.0, z=0.0, priors=True, budget_ms=3000,
                      extra_actions=(ex.PlannedAction("off.pass_protection", {"protection": "Half Slide"}),))
        ranking = [(r["formation"], r["play"], round(r["probability"], 3)) for r in d.ranking[:3]]
        # 3) selected adjustments -> 4) transformed play (from the VERIFIED stock structure of the selected play) -> 5) reads
        stock = sm.structure_any(d.formation, d.play)
        transformed = od._apply_actions(stock, d.actions, {})
        card = adv.build_card(d, mix)
        assert d.final_structure.routes == transformed.routes                                    # the card reads the SAME transformed play
        assert (card["FORMATION"], card["PLAY"]) == (d.formation, d.play)
        trace[look] = {"mix": mix, "ranking": ranking, "play": f"{d.formation} / {d.play}", "adjustment": d.adjustment,
                       "matchup_edge": round(d.matchup["edge"], 3) if d.matchup else None, "changed": list(transformed.adjustments),
                       "reads": [card["PRIMARY READ"], card["SECONDARY READ"], card["CHECKDOWN"]], "conditioned": card["COVERAGE-CONDITIONED ORDER"],
                       "basis": card["READ BASIS"], "executable_now": card["EXECUTABLE NOW"]}
    a, b = trace.values()
    assert a["ranking"] != b["ranking"] and a["matchup_edge"] != b["matchup_edge"]               # the defense changed the scores/ranking
    assert a["conditioned"] != b["conditioned"] or a["reads"] != b["reads"]                      # ... and the read plan
    json.dumps(trace)                                                                             # fully serialisable for the log


def test_full_trace_through_the_ui_state_and_the_logged_result(tmp_path, monkeypatch, live_artifacts):
    """The same chain through the real live controller: card/UI state for the call, then the typed defense + result are stored with the call."""
    from tests.madden_ml.test_defense_knowledge_live import _ctrl, _prime, _world
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = _world(tmp_path, monkeypatch, live_artifacts, "default")
    monkeypatch.setenv(rr.ENV_PRIORS, "1")
    _prime(db)
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    for sit_raw in ("2&4 my 31", "1&10 my 41"):
        c.result_and_call(outcome="+6", sit_raw=sit_raw, their="Cover 1")
    info = rr.info_of(c.last_call)
    shown = (c.last_call.formation, c.last_call.play)
    ui = c._qb_reads_state()                                                                      # what the compact UI renders for this call
    assert info["applied"] and "football_priors" in info and info["coverage"]["named_n"] == 2
    assert ui is not None and ui["status"] in ("known", "unknown", "fallback", "unavailable")
    assert (info["card"]["FORMATION"], info["card"]["PLAY"]) == shown
    c.result_and_call(outcome="+3", sit_raw="2&7 my 44", their="Cover 1 Robber", executed_status="used_recommended")
    row = [dict(r) for r in db.conn.execute("SELECT * FROM snaps WHERE session_id=? AND side='offense' AND result IS NOT NULL ORDER BY id", (c.session_id,))][-1]
    assert (row["formation"], row["play"]) == shown and row["coverage_typed"] == "Cover 1 Robber" and row["result"] == "+3"
    assert row["executed_status"] == "identified" and row["executed_verification"] == "verified"
    shadow = [json.loads(l) for l in rr.shadow_log_path().read_text().splitlines()]
    assert shadow and shadow[-1].get("football_priors") is not None or any("football_priors" in r for r in shadow)   # decision + priors logged
    db.close()
