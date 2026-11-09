"""Mandatory acceptance tests (spec §15) for unified_v1."""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from cfb_coach.madden.unified import guard, knowledge as K, postgame as PG, pregame as PRE, solver as SV
from cfb_coach.madden.unified.adjustments import AdjustmentPlanner
from cfb_coach.madden.unified.coordinator import COORDINATOR_NAME, InstalledBookError, UnifiedCoordinator, inventory_fp
from cfb_coach.madden.unified.opponent import OpponentModel
from cfb_coach.madden.unified.reads import defense_user, offense_reads
from cfb_coach.madden.unified.solver import Solver, SolverConfig
from cfb_coach.madden.unified.store import UnifiedStore
from cfb_coach.madden.unified.types import (DEF, EXEC_CONFIRMED, EXEC_NOT_RUN, EXEC_UNKNOWN, OFF, Adjustment, FootballAction,
                                            GameState, OpponentAction, Outcome, SnapObservation)
from cfb_coach.madden.unified.world_model import DefSig, Profile, WorldModel
from tests.unified.helpers import coordinator, inventory, play_snap, state

REPO = Path(__file__).resolve().parents[2]


# 1 -------------------------------------------------------------------------------------------
def test_offense_and_defense_consume_the_same_game_state_schema():
    off, de = coordinator(OFF), coordinator(DEF)
    s_o, s_d = state(OFF), state(DEF)
    assert type(s_o) is type(s_d) is GameState
    assert set(asdict(s_o)) == set(asdict(s_d))
    assert off.recommend(s_o).side == OFF and de.recommend(s_d).side == DEF
    with pytest.raises(ValueError):  # a state for the other side is rejected, not silently coerced
        off.recommend(s_d)


# 2 -------------------------------------------------------------------------------------------
def test_shared_model_gives_compatible_predictions_to_both_sides():
    m, st = WorldModel(), state(OFF)
    o = K.offense_play("Gun Trips X Nasty", "Mesh Spot")
    d = K.defense_play("4-3 Over", "Cover 1 Hole")
    pred = m.predict(st, FootballAction(o), DefSig.from_signature(d.signature, d.play_id, d.features))
    # the defensive coordinator queries the SAME interface for the same matchup
    prof = Profile.off(FootballAction(o))
    pred2 = m.predict(replace(st, our_side=DEF), prof, DefSig.from_signature(d.signature, d.play_id, d.features))
    assert pred.as_dict().keys() == pred2.as_dict().keys()
    assert pred.value == pytest.approx(pred2.value)
    for f in ("p_first_down", "p_td", "p_turnover", "p_explosive", "p_sack"):
        assert 0.0 <= getattr(pred, f) <= 1.0
    assert pred.calibrated is False and pred.source == "cold_start_prior"  # never presented as trained
    # one table: offense maximises the value, defense minimises the same number
    co_o, co_d = coordinator(OFF), coordinator(DEF)
    assert co_o.model.predict is not None and type(co_o.model) is type(co_d.model)


# 3 -------------------------------------------------------------------------------------------
def test_observed_opponent_call_is_not_mislabelled_as_our_execution():
    co = coordinator(DEF)
    st = state(DEF)
    co.recommend(st)
    co.record_opponent(st, OpponentAction(OFF, "Gun Trips X Nasty", "Mesh Spot", K.offense_play("Gun Trips X Nasty", "Mesh Spot").signature, "typed_exact", 1.0))
    snap = co.snaps[st.snap_id]
    assert snap.executed is None and snap.executed_status == EXEC_UNKNOWN
    co.close_snap(st, Outcome(4.0, False, False, False, False, False))
    assert co.model.n_records == 0  # opponent info alone never becomes OUR labelled defensive execution
    assert co.opp.records  # ...but it does inform the shared opponent knowledge
    assert co.opp.records[0].sig == K.offense_play("Gun Trips X Nasty", "Mesh Spot").signature


# 4 -------------------------------------------------------------------------------------------
def test_opponent_tendencies_move_with_repeated_observations_and_keep_uncertainty():
    co = coordinator(OFF)
    st0 = state(OFF, seq=1)
    b0 = co.opp.beliefs(st0)
    call = ("4-3 Over", "Cover 1 Hole")
    sig = K.defense_play(*call).signature
    for i in range(1, 9):
        play_snap(co, state(OFF, seq=i), opp_call=call)
    b1 = co.opp.beliefs(state(OFF, seq=9))
    assert b1.dist[sig] > b0.dist[sig] + 0.15
    assert b1.dist[sig] < 0.95  # uncertainty preserved: never certain
    assert b1.n_effective > b0.n_effective and b1.tv_radius < b0.tv_radius
    assert sum(b1.dist.values()) == pytest.approx(1.0)


# 5 -------------------------------------------------------------------------------------------
def test_future_snaps_never_influence_earlier_recommendations():
    base, fut = coordinator(OFF), coordinator(OFF)
    call = ("4-3 Over", "Cover 1 Hole")
    for i in range(1, 3):
        play_snap(base, state(OFF, seq=i), opp_call=call)
        play_snap(fut, state(OFF, seq=i), opp_call=call)
    for i in range(3, 12):  # a long run of later snaps seen only by `fut`
        play_snap(fut, state(OFF, seq=i), opp_call=("Nickel Over", "Cover 4 Palms"))
    q = state(OFF, seq=3)
    assert base.opp.beliefs(q).dist == fut.opp.beliefs(q).dist  # snaps 3+ are invisible at seq 3
    # unclosed snaps never count either
    c3 = coordinator(OFF)
    c3.recommend(state(OFF, seq=1))
    c3.record_opponent(state(OFF, seq=1), OpponentAction(DEF, "4-3 Over", "Cover 1 Hole", K.defense_play(*call).signature, "typed_exact", 1.0))
    assert not c3.opp.records


# 6 -------------------------------------------------------------------------------------------
def _trained_model():
    """A model that has learned: mesh beats man (cover1), four_verts beats cover3 shells."""
    m = WorldModel()
    mesh, verts = K.offense_play("Gun Trips X Nasty", "Hi Lo Cross"), K.offense_play("Gun Trips X Nasty", "Verticals X Shallow")
    c1, c3 = K.defense_play("4-3 Over", "Cover 1 Hole"), K.defense_play("4-3 Over", "Cover 3 Sky")
    st = state(OFF)
    for _ in range(60):
        for off, d, y in ((mesh, c1, 11.0), (mesh, c3, 3.0), (verts, c3, 11.0), (verts, c1, 3.0)):
            m.update(st, Profile.off(FootballAction(off)), DefSig.from_signature(d.signature, d.play_id, d.features), Outcome(y, y >= 6, False, False, False, y >= 15))
    return m, c1, c3


def test_solver_changes_selection_when_beliefs_materially_change():
    m, c1, c3 = _trained_model()
    inv = {"Gun Trips X Nasty": ["Hi Lo Cross", "Verticals X Shallow"]}

    def pick(sig):
        om = OpponentModel(DEF, prior={sig: 1.0})
        co = UnifiedCoordinator(OFF, inv, confirmed_fingerprint=inventory_fp(OFF, inv), model=m, opponent_model=om)
        return co.recommend(state(OFF)).play

    assert pick(c1.signature) == "Hi Lo Cross"
    assert pick(c3.signature) == "Verticals X Shallow"


# 7 -------------------------------------------------------------------------------------------
def test_coordinator_never_recommends_same_exact_play_on_consecutive_cpu_snaps():
    m, c1, _ = _trained_model()
    inv = {"Gun Trips X Nasty": ["Hi Lo Cross", "Verticals X Shallow"]}
    om = OpponentModel(DEF, prior={c1.signature: 1.0})
    co = UnifiedCoordinator(OFF, inv, confirmed_fingerprint=inventory_fp(OFF, inv), model=m, opponent_model=om)
    plays = [co.recommend(state(OFF, seq=i)).play for i in range(1, 8)]
    assert set(plays) == {"Hi Lo Cross", "Verticals X Shallow"}
    assert all(a != b for a, b in zip(plays, plays[1:]))
    src = (REPO / "cfb_coach/madden/unified/solver.py").read_text()
    # This is a pre-solver legal-candidate constraint, not a random changeup,
    # a synthetic score bonus, or an override after the model has chosen.
    assert "random.choice" not in src and "shuffle" not in src and "changeup" not in src.lower()


def test_cpu_defense_never_calls_free_fire_twice_when_an_alternative_exists():
    inv = {"4-3 Over": ["Free Fire", "Cover 3 Sky"]}
    co = UnifiedCoordinator(DEF, inv, confirmed_fingerprint=inventory_fp(DEF, inv))
    calls = [co.recommend(state(DEF, seq=i, down=3, dist=6)).play for i in range(1, 7)]
    assert all(a != b for a, b in zip(calls, calls[1:]))


def test_diverse_installed_book_never_alternates_only_two_plays():
    inv = {"Gun Tight Doubles": [
        "Bench", "Mesh Spot", "Mtn Cross HB Sneak", "HB Draw",
        "Inside Zone", "PA WR Cross", "Tight Flood"
    ]}
    co = UnifiedCoordinator(OFF, inv, confirmed_fingerprint=inventory_fp(OFF, inv))
    picks = [co.recommend(state(OFF, seq=i, down=2, dist=7)).play for i in range(1, 13)]
    for j, play in enumerate(picks):
        assert play not in picks[max(0, j - 4):j], picks
    assert len(set(picks)) >= 5


def test_mtn_cross_hb_sneak_is_a_pass_concept_not_qb_run():
    from cfb_coach.madden.unified.knowledge import offense_play
    from cfb_coach.madden.model.football_knowledge import profile_for_play
    p = offense_play("Gun Tight Doubles", "Mtn Cross HB Sneak")
    assert p.play_type == "pass"
    assert p.concept == "deep_cross"
    assert profile_for_play("Mtn Cross HB Sneak")["concept_id"] == "deep_cross"


def test_one_play_book_is_only_allowed_repeat_exception():
    inv = {"4-3 Over": ["Free Fire"]}
    co = UnifiedCoordinator(DEF, inv, confirmed_fingerprint=inventory_fp(DEF, inv))
    calls = [co.recommend(state(DEF, seq=i)).play for i in range(1, 4)]
    assert calls == ["Free Fire"] * 3


def test_no_repeat_filter_is_per_game_not_previous_game():
    inv = {"4-3 Over": ["Free Fire", "Cover 3 Sky"]}
    co = UnifiedCoordinator(DEF, inv, confirmed_fingerprint=inventory_fp(DEF, inv))
    first = co.recommend(state(DEF, seq=1, game="g1")).play
    second = co.recommend(state(DEF, seq=1, game="g2")).play
    # Initial state and opponent are unchanged; previous-game recommendations
    # do not forbid the preferred opening play in a new game.
    assert second == first


# 8 -------------------------------------------------------------------------------------------
def _hot_route_play():
    from cfb_coach.madden.unified import knowledge as KK
    for (f, p), rec in KK.offense_assignment_index().items():
        pl = KK.offense_play(f, p)
        if len(pl.assignments) >= 2:
            return pl
    pytest.skip("no documented-route play in catalog")


def test_uncertain_estimates_cannot_force_custom_adjustments():
    pl = _hot_route_play()
    conf = {"off.hot_route": True, **{f"off.hot_route|{r}": True for r in ("Slant", "Flat", "Streak", "Curl", "Out", "In", "Drag", "Fade", "Post", "Corner", "Comeback")}}
    planner = AdjustmentPlanner(conf)
    acts = planner.candidates(OFF, pl)
    assert any(a.adjustments for a in acts), "confirmed hot routes should produce candidates"
    inv = {pl.formation: [pl.play]}
    co = UnifiedCoordinator(OFF, inv, confirmed_fingerprint=inventory_fp(OFF, inv), planner=planner)
    card = co.recommend(state(OFF))
    assert card.adjustments == []  # weak-prior benefit does not survive uncertainty -> NO ADJUSTMENT
    rejected = co.solver.decide(state(OFF), OFF, co.legal_candidates(), co.opp.beliefs(state(OFF))).rejected_adjustments
    assert rejected and all("does not survive" in why for _, why in rejected)


def test_unconfirmed_adjustments_are_never_live_selectable():
    pl = K.defense_play("4-3 Over", "Cover 3 Sky")
    planner = AdjustmentPlanner(include_unconfirmed=True)
    acts = planner.candidates(DEF, pl)
    assert any(a.adjustments and not a.executable for a in acts)
    inv = {pl.formation: [pl.play]}
    co = UnifiedCoordinator(DEF, inv, confirmed_fingerprint=inventory_fp(DEF, inv), planner=planner)
    assert any(not a.executable for a in co.legal_candidates())  # present for analysis...
    card = co.recommend(state(DEF))
    assert card.adjustments == [] and co.snaps[card.snap_id].recommended.executable  # ...never chosen live


def test_well_supported_adjustment_can_still_win():
    pl = K.defense_play("4-3 Over", "Cover 3 Sky")
    planner = AdjustmentPlanner({"def.coverage_hub": True, "def.coverage_hub|Press": True})
    # learned (not prior) evidence: Press stops man-ish offence a lot and robustly
    for tag in ("cross", "pass_other", "run", "vert", "screen", "flood", "rpo"):
        planner.learned[("def.coverage_hub", tag)] = [40.0, 40.0 * 3.0]
    inv = {pl.formation: [pl.play]}
    co = UnifiedCoordinator(DEF, inv, confirmed_fingerprint=inventory_fp(DEF, inv), planner=planner)
    cands = co.legal_candidates()
    assert any("Press" in a.key and a.executable for a in cands)
    card = co.recommend(state(DEF))
    assert card.adjustments == ["Global Coverage: Press"]


# 9 -------------------------------------------------------------------------------------------
def test_no_adjustment_is_always_a_legal_candidate():
    for side in (OFF, DEF):
        co = coordinator(side)
        plays = {p.play_id for p in co.plays}
        assert plays == {a.play.play_id for a in co.legal_candidates() if a.no_adjustment}
    co = coordinator(OFF)
    no_noadj = [a for a in co.legal_candidates() if not a.no_adjustment]
    with pytest.raises(ValueError):
        co.solver.decide(state(OFF), OFF, no_noadj or co.legal_candidates()[:0] + [replace(co.legal_candidates()[0], adjustments=(Adjustment("x", "x"),))],
                         co.opp.beliefs(state(OFF)))


# 10 ------------------------------------------------------------------------------------------
def test_adjusted_offensive_card_reflects_actual_changed_routes():
    pl = _hot_route_play()
    slot = pl.assignments[0].slot
    adj = Adjustment("off.hot_route", "Hot Route: Streak", {"target": slot, "route": "Streak"})
    act = AdjustmentPlanner._apply(pl, (adj,), OFF, executable=True)
    plan = offense_reads(act, OpponentModel(DEF).beliefs(state(OFF)))
    assert plan.changed_routes == {slot: "Streak"}
    assert f"{slot} (streak)" in (plan.primary, plan.secondary) or "hot_route" in plan.basis
    assert "+hot_route" in plan.basis


def test_receiver_identity_is_never_invented_for_ambiguous_art():
    pl = K.offense_play("Gun Trips X Nasty", "Mesh Spot")  # no documented routes
    plan = offense_reads(FootballAction(pl), OpponentModel(DEF).beliefs(state(OFF)))
    assert plan.identity_known is False and "UNKNOWN" in plan.basis
    assert not any(tok in plan.primary for tok in ("WR1", "WR2", "TE", "HB", "X ", "Y ", "Z "))


# 11 ------------------------------------------------------------------------------------------
def test_defensive_user_advice_respects_verified_assignment_constraints(tmp_path):
    play = K.defense_play("4-3 Over", "Cover 3 Sky")
    slots = [{"slot_id": f"S{i}", "role": "Deep third" if i == 1 else "Hook/curl" if i == 2 else "Run fit", "responsibility":
              f"R{i}: " + ("deep third, never leave" if i == 1 else "hook/curl zone" if i == 2 else "gap fit"), "verified": True} for i in range(1, 12)]
    f = tmp_path / "roles.json"
    f.write_text(json.dumps({"schema": "madden.defense.v2.verified_roles.v1", "plays": [
        {"formation": "4-3 Over", "play": "Cover 3 Sky", "verified_in_madden": True, "source": "owner_xbox_play_art", "verified_on": "2026-11-01", "slots": slots}]}))
    belief = OpponentModel(OFF).beliefs(state(DEF))
    ua = defense_user(FootballAction(play), belief, str(f))
    assert ua.verified is True
    assert ua.primary in {s["responsibility"] for s in slots}  # only a role the owner verified
    plain = defense_user(FootballAction(K.defense_play("4-3 Over", "Cover 2 Man")), belief, None)
    assert plain.verified is False and "UNKNOWN" in plain.basis


# 12 ------------------------------------------------------------------------------------------
def test_uninstalled_plays_and_wrong_fingerprints_are_rejected():
    inv = inventory(OFF)
    with pytest.raises(InstalledBookError):
        UnifiedCoordinator(OFF, inv, confirmed_fingerprint="OFF-INST-DEADBEEF00")
    co = coordinator(OFF)
    installed = {(f, p) for f, ps in inv.items() for p in ps}
    for i in range(1, 12):
        card = co.recommend(state(OFF, seq=i, down=1 + i % 3, dist=2 + i, yl=10 + 7 * i))
        assert (card.formation, card.play) in installed
    assert all((a.play.formation, a.play.play) in installed for a in co.legal_candidates())
    with pytest.raises(InstalledBookError):
        co.recommend(state(OFF, seq=40, playbook_fingerprint="OFF-INST-0000000000"))


# 13 ------------------------------------------------------------------------------------------
def test_fifteen_formation_limit_applies_per_side():
    big = {f"F{i}": ["Mesh"] for i in range(16)}
    for side in (OFF, DEF):
        with pytest.raises(InstalledBookError):
            UnifiedCoordinator(side, big, confirmed_fingerprint=inventory_fp(side, big))
    recs = []
    for i in range(30):
        r = PRE.Rec(f"Gun Set {i}", "Bk", ["Mesh Spot", "Inside Zone"], "Gun", None)
        r.footballplays = [K.offense_play(r.formation, p) for p in r.plays]
        r.values = {s: (i % 7) * 0.1 for s in PRE.SITUATIONS}
        recs.append(r)
    sel, _ = PRE.select_portfolio(OFF, recs)
    assert len(sel) <= 15
    d = [PRE.Rec(f"Nickel Set {i}", "Bk", ["Cover 3 Sky", "Cover 1 Hole"], "Nickel", None) for i in range(30)]
    for r in d:
        r.footballplays = [K.defense_play("4-3 Over", p) for p in r.plays]
        r.values = {s: (hash(r.formation) % 5) * 0.1 for s in PRE.SITUATIONS}
    assert len(PRE.select_portfolio(DEF, d)[0]) <= 15  # independent limit for the other side


# 14 ------------------------------------------------------------------------------------------
def test_recommendation_execution_and_revealed_action_stay_distinct(tmp_path):
    store = UnifiedStore(tmp_path / "u.db")
    co = coordinator(OFF, store=store)
    st = state(OFF)
    card = co.recommend(st)
    assert store.counts() == {"decisions": 1, "executions": 0, "opponent_obs": 0, "outcomes": 0}
    co.record_opponent(st, OpponentAction(DEF, "4-3 Over", "Cover 1 Hole", K.defense_play("4-3 Over", "Cover 1 Hole").signature, "typed_exact", 1.0))
    o = [x for x in store.load_observations() if x.snap_id == st.snap_id][0]
    assert o.recommended is not None and o.executed is None and o.executed_status == EXEC_UNKNOWN and o.opponent_action is not None
    ran_other = co.snaps[st.snap_id].recommended
    co.record_execution(st, None, EXEC_NOT_RUN)
    o = [x for x in store.load_observations() if x.snap_id == st.snap_id][0]
    assert o.executed is None and o.executed_status == EXEC_NOT_RUN and not o.labelled_for_own_learning
    assert o.recommended.key == card.action_key
    assert ran_other.key == card.action_key


# 15 ------------------------------------------------------------------------------------------
def test_both_coordinators_identify_their_decision_model():
    for side in (OFF, DEF):
        card = coordinator(side).recommend(state(side))
        txt = card.render()
        assert card.decision_model == COORDINATOR_NAME == "unified_v1"
        assert card.model_version and card.artifact and "cold_start" in card.artifact
        assert card.opponent_prediction_source
        for needle in ("unified_v1", "OPPONENT MODEL", "EVIDENCE CONFIDENCE", "FALLBACK", "NO ADJUSTMENT"):
            assert needle in txt
        assert set(card.fallback) == {"occurred", "reason"}


# 16 ------------------------------------------------------------------------------------------
def test_postgame_training_cannot_silently_modify_the_champion(tmp_path):
    from cfb_coach.madden.unified import sim
    games = sim.gen_games(6, 30, seed=3)
    rows, excluded = PG.validate_labels(games)
    champ_dir = tmp_path / "champ"
    champ_dir.mkdir()
    champ = champ_dir / "champion.json"
    champ.write_text(json.dumps(PG.fit_on(rows[:100]).to_dict()))
    before = hashlib.sha256(champ.read_bytes()).hexdigest()
    rep = PG.train_challenger(rows, tmp_path / "chall", champion_path=champ)
    assert rep["champion_unchanged"] and hashlib.sha256(champ.read_bytes()).hexdigest() == before
    assert rep["champion_modified"] is False and "NOT PERFORMED" in rep["promotion"]
    with pytest.raises(ValueError):
        PG.train_challenger(rows, champ_dir, champion_path=champ)
    assert not any(n.startswith("promote") for n in dir(PG))


# 17 ------------------------------------------------------------------------------------------
def test_protected_assets_are_unchanged_and_isolated_stores_are_enforced(tmp_path, monkeypatch):
    prot = guard.protected_paths((), REPO)
    assert prot, "repo data assets must be checksummed"
    before = guard.snapshot(prot)
    store = UnifiedStore(tmp_path / "iso.db")
    co = coordinator(OFF, store=store)
    for i in range(1, 6):
        play_snap(co, state(OFF, seq=i), opp_call=("4-3 Over", "Cover 3 Sky"))
    d = guard.diff(before, guard.snapshot(prot))
    assert not (d["changed"] or d["added"] or d["removed"])
    home = tmp_path / "home" / ".cfb-coach"
    home.mkdir(parents=True)
    monkeypatch.setattr(guard, "DEFAULT_HOME", home)
    with pytest.raises(guard.ProtectedPathError):
        UnifiedStore(home / "coach.db")
