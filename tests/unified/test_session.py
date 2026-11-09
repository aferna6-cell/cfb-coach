"""Continuous two-sided game: shared state, ids, history, learning provenance, and the scripted live CLI."""
from __future__ import annotations

import json
import types

import pytest

from cfb_coach.madden.unified import knowledge as K
from cfb_coach.madden.unified.cli import cmd_game
from cfb_coach.madden.unified.coordinator import inventory_fp
from cfb_coach.madden.unified.inputs import InputError
from cfb_coach.madden.unified.session import UnifiedSession
from cfb_coach.madden.unified.store import AlreadyRecorded, DuplicateSnapError, UnifiedStore
from cfb_coach.madden.unified.types import DEF, EXEC_CONFIRMED, OFF, GameState, Outcome
from cfb_coach.madden.unified.world_model import DefSig, Profile
from tests.unified.helpers import inventory

COVER1 = "4-3 Over | Cover 1 Hole"


def books(sides=(OFF, DEF)):
    return {s: {"inventory": inventory(s), "fingerprint": inventory_fp(s, inventory(s))} for s in sides}


def session(tmp_path, **kw):
    kw.setdefault("game_id", "g1")
    return UnifiedSession(books=kw.pop("books", books()), opponent_id="cpu_x", store=UnifiedStore(tmp_path / "u.db"), **kw)


def step(S, line, ex="y", res="+5", opp=""):
    card = S.start_snap(line)
    return card, S.finish_snap(executed=ex, result=res, opponent_call=opp)


def test_one_game_switches_sides_without_relaunch_and_ids_never_repeat(tmp_path):
    S = session(tmp_path)
    plan = [("o 1&10 my 22", OFF), ("o 2&3 my 29", OFF), ("d 1&10 opp 40", DEF), ("d 2&6 opp 36", DEF), ("o 1&10 my 30", OFF), ("d 3&8 my 20", DEF)]
    cards = [step(S, line)[0] for line, _ in plan]
    assert [c.side for c in cards] == [s for _, s in plan]
    assert [c.snap_id for c in cards] == [f"g1-{i:04d}" for i in range(1, 7)]
    assert len({c.snap_id for c in cards}) == 6 and all(c.decision_model == "unified_v1" for c in cards)
    counts = S.store.counts()
    assert counts["decisions"] == counts["executions"] == counts["outcomes"] == 6
    # the side follows the last typed possession when the line omits it
    assert S.start_snap("2&4 my 35").side == DEF


def test_duplicate_ids_and_overwritten_rows_are_refused(tmp_path):
    S = session(tmp_path)
    card, _ = step(S, "o 1&10 my 22")
    st = S.store.load_observations()[0].state
    with pytest.raises(DuplicateSnapError):
        S.coord[OFF].recommend(st)  # same snap id again
    with pytest.raises(DuplicateSnapError):
        UnifiedSession(books=books(), opponent_id="cpu_x", store=S.store, game_id="g1")  # game id reuse without resume
    S.start_snap("o 2&5 my 27")
    S.finish_snap(executed="y", result="+3")
    snap = S.store.load_observations()[1]
    with pytest.raises(AlreadyRecorded):
        S.coord[OFF].record_execution(snap.state, None, "not_executed")  # would overwrite the recorded execution
    assert [o.executed_status for o in S.store.load_observations()][1] == EXEC_CONFIRMED  # memory and store untouched
    rev = S.coord[OFF].record_execution(snap.state, None, "not_executed", amend=True)  # explicit, audited correction
    assert S.store.db.execute("select count(*) from executions where snap_id=?", (snap.snap_id,)).fetchone()[0] == 2
    assert S.store.load_observations()[1].executed_status == "not_executed"


def test_previous_snap_context_is_same_side_across_possession_changes(tmp_path):
    S = session(tmp_path)
    step(S, "o 1&10 my 22", opp=COVER1)
    step(S, "d 1&10 opp 40", opp="Gun Trips X Nasty | Mesh Spot")  # a defensive snap in between
    step(S, "o 1&10 my 40", opp="4-3 Over | Cover 3 Sky")
    recs = [r for r in S.coord[OFF].opp.records]
    assert len(recs) == 2
    assert recs[1].prev_sig == recs[0].sig  # sequence context skips the intervening defensive snap
    assert S.coord[OFF].last_same_side("g1").snap_id == "g1-0003" and S.coord[DEF].last_same_side("g1").snap_id == "g1-0002"


def test_no_future_information_reaches_earlier_decisions(tmp_path):
    A, B = session(tmp_path / "a"), session(tmp_path / "b")
    for S in (A, B):
        step(S, "o 1&10 my 22", opp=COVER1)
    q = GameState("g1", 2, OFF, "cpu_x", 1, 10, 30, opponent_class="cpu")
    before = A.coord[OFF].opp.beliefs(q).dist
    for _ in range(6):  # later snaps, only in B
        step(B, "o 1&10 my 30", opp="Nickel Over | Cover 4 Palms")
    assert B.coord[OFF].opp.beliefs(q).dist == before


def test_shared_learning_crosses_sides_but_provenance_stays_separate(tmp_path):
    S = session(tmp_path)
    mesh = K.offense_play("Gun Trips X Nasty", "Hi Lo Cross")
    c1 = K.defense_play("4-3 Over", "Cover 1 Hole")
    st = GameState("x", 1, DEF, "cpu_x", 2, 6, 45, opponent_class="cpu")
    prof, ds = Profile("pass", mesh.concept, None, mesh.features), DefSig.from_signature(c1.signature, c1.play_id, c1.features)
    before = S.model.predict(st, prof, ds).e_yards
    # a real OFFENSIVE possession: our offense executes big gains against a typed cover-1 call
    off_play = S.coord[OFF].plays[0]
    for i in range(12):
        card = S.start_snap("o 1&10 my 22")
        S.finish_snap(executed="y", result="+14", opponent_call=COVER1)
    mid = S.model.predict(st, Profile("pass", S.coord[OFF].snaps["g1-0001"].recommended.play.concept, None, S.coord[OFF].snaps["g1-0001"].recommended.play.features), ds)
    shown = S.coord[OFF].snaps["g1-0001"].recommended.play
    cold = type(S.model)().predict(st, Profile("pass", shown.concept, None, shown.features), ds).e_yards
    assert mid.e_yards > cold + 0.5  # the DEFENSIVE coordinator's query now reflects what our offense learned
    assert S.model.provenance == {"our_offense_executed": 12}
    # a DEFENSIVE possession next: unknown execution + observed opponent offense must not become 'our executed defense'
    S.start_snap("d 1&10 opp 40")
    out = S.finish_snap(executed="u", result="+9", opponent_call="Gun Trips X Nasty | Hi Lo Cross")
    assert out["world_model_row"] is False and S.model.provenance == {"our_offense_executed": 12}
    assert S.coord[DEF].opp.records and not S.coord[OFF].opp.records[-1].sig == S.coord[DEF].opp.records[-1].sig
    S.start_snap("d 2&1 opp 49")
    assert S.finish_snap(executed="y", result="+1", opponent_call="Gun Trips X Nasty | Hi Lo Cross")["world_model_row"] is True
    assert S.model.provenance == {"our_offense_executed": 12, "our_defense_executed": 1}
    assert {k[0] for k in S.model.ctx_resid} == {OFF, DEF}  # side-specific execution effects kept apart


def test_history_replay_feeds_both_sides_and_both_opponent_models(tmp_path):
    S1 = session(tmp_path, game_id="g1")
    for i in range(4):
        step(S1, "o 1&10 my 22", res="+8", opp=COVER1)
        step(S1, "d 1&10 opp 40", res="+3", opp="Gun Trips X Nasty | Hi Lo Cross")
    S1.end_game()
    S2 = UnifiedSession(books=books(), opponent_id="cpu_x", store=S1.store, game_id="g2")
    rep = S2.history_report
    assert rep["games"] == 1 and rep["world_model_rows"] == 8
    assert rep["provenance"] == {"our_offense_executed": 4, "our_defense_executed": 4}
    assert all(n > 0 for n in rep["opponent_records"].values())
    card = S2.start_snap("o 1&10 my 22")
    assert "hist(" in card.opponent_prediction_source  # the new game starts from last game's evidence, not zero


def test_resume_continues_the_sequence(tmp_path):
    S = session(tmp_path)
    step(S, "o 1&10 my 22")
    step(S, "d 1&10 opp 40")
    R = UnifiedSession(books=books(), opponent_id="cpu_x", store=S.store, game_id="g1", resume=True)
    assert R.start_snap("o 1&10 my 30").snap_id == "g1-0003"


def test_no_consecutive_exact_play_survives_defensive_possession_and_resume(tmp_path):
    S = session(tmp_path)
    first_off = S.start_snap("o 1&10 my 22")
    S.finish_snap(executed="y", result="+2")
    # Switching to defense must not forget the previous offensive call.
    S.start_snap("d 1&10 opp 40")
    S.finish_snap(executed="y", result="+3")
    R = UnifiedSession(books=books(), opponent_id="cpu_x", store=S.store, game_id="g1", resume=True)
    next_off = R.start_snap("o 2&8 my 24")
    assert (first_off.formation, first_off.play) != (next_off.formation, next_off.play)


def test_inputs_clock_and_presnap_showing_are_ignored_and_bad_input_changes_nothing(tmp_path):
    A, B = session(tmp_path / "a"), session(tmp_path / "b")
    c1 = A.start_snap("o 3&7 my 45")
    c2 = B.start_snap("o 3&7 my 45 showing cover 3 pre-snap 2 minute")
    assert (c1.formation, c1.play, c1.adjustments) == (c2.formation, c2.play, c2.adjustments)
    with pytest.raises(InputError):
        A.finish_snap(executed="Gun Nope | Fake Play", result="+3")  # not in the installed book
    with pytest.raises(InputError):
        A.finish_snap(executed="y", result="+3", opponent_call="zzz qqq")
    assert A.store.counts()["executions"] == 0 and A.pending is not None  # nothing written, snap still waiting
    with pytest.raises(InputError):
        A.start_snap("3&7") if False else B.start_snap("hello")


def test_executed_grammar_records_what_was_really_run(tmp_path):
    S = session(tmp_path)
    c = S.start_snap("d 1&10 opp 40")
    assert S.finish_snap(executed="n", result="+3")["executed_status"] == "not_executed"
    c = S.start_snap("o 1&10 my 22")
    other = next(p for p in inventory(OFF)["Gun Trips X Nasty"] if p != c.play)
    r = S.finish_snap(executed=f"Gun Trips X Nasty | {other}", result="+3", opponent_call=COVER1)
    ob = S.store.load_observations()[1]
    assert r["executed_status"] == "modified" and ob.recommended.play.play == c.play and ob.executed.play.play == other


def test_both_books_are_required_for_a_two_sided_game(tmp_path):
    from cfb_coach.madden.unified.coordinator import InstalledBookError
    with pytest.raises(InstalledBookError):
        session(tmp_path, books=books((OFF,)))


def test_cli_game_runs_a_scripted_two_sided_game_and_next_game_remembers_it(tmp_path, capsys):
    bk = {s: {"side": s, "fingerprint": inventory_fp(s, inventory(s)), "inventory": inventory(s)} for s in (OFF, DEF)}
    bf = tmp_path / "books.json"
    bf.write_text(json.dumps(bk))
    script = iter(["o 1&10 my 22", "y", "+6", COVER1, "o 2&4 my 28", "y", "inc", "cover 3 sky", "d 1&10 opp 40", "y", "+4",
                   "Gun Trips X Nasty | Hi Lo Cross", "oops", "status", "end"])
    out = []
    ns = types.SimpleNamespace(opponent="cpu_x", opponent_class=None, confirmed_books=str(bf), store=str(tmp_path / "s.db"), model=None,
                               game_id="e2e", resume=False, verified_roles=None, adjustment_confirmations=None)
    assert cmd_game(ns, input_fn=lambda p="": next(script), out=out.append) == 0
    text = "\n".join(out)
    assert "OFFENSE" in text and "DEFENSE" in text and "[unified_v1" in text and "BOTH coordinators active" in text
    assert "who has the ball" not in text.split("sit> type")[0]
    assert "? " in text  # 'oops' was rejected politely
    st = UnifiedStore(tmp_path / "s.db")
    obs = st.load_observations()
    assert [o.state.our_side for o in obs] == [OFF, OFF, DEF] and len({o.snap_id for o in obs}) == 3
    assert all(o.closed and o.executed_status == EXEC_CONFIRMED for o in obs)
    out2 = []
    s2 = iter(["o 1&10 my 22", "y", "+3", "", "quit"])
    ns.game_id = "e2e-2"
    assert cmd_game(ns, input_fn=lambda p="": next(s2), out=out2.append) == 0
    assert any("hist(" in line for line in out2)  # game 2 starts from game 1's closed evidence
    ns.game_id = "e2e"
    assert cmd_game(ns, input_fn=lambda p="": "quit", out=lambda *_: None) == 2  # duplicate game id refused
