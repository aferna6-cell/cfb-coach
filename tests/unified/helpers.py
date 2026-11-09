"""Shared builders for the unified-coordinator tests (repo snapshot books, no owner data)."""
from __future__ import annotations

import json
from pathlib import Path

from cfb_coach.madden.unified import knowledge as K
from cfb_coach.madden.unified.coordinator import UnifiedCoordinator, inventory_fp
from cfb_coach.madden.unified.opponent import OpponentModel
from cfb_coach.madden.unified.types import (DEF, EXEC_CONFIRMED, OFF, GameState, OpponentAction, Outcome)

SNAP = json.loads((Path(__file__).resolve().parents[2] / "cfb_coach/data/madden27/installed_playbook_snapshot.json").read_text())["sides"]


def inventory(side):
    return {k: list(v) for k, v in SNAP[side]["formations"].items()}


def coordinator(side, **kw):
    inv = kw.pop("inv", None) or inventory(side)
    return UnifiedCoordinator(side, inv, confirmed_fingerprint=inventory_fp(side, inv), **kw)


def state(side, seq=1, down=3, dist=6, yl=45, game="g1", opp="cpu_x", cls="cpu", **kw):
    return GameState(game, seq, side, opp, down, dist, yl, opponent_class=cls, **kw)


def play_snap(co, st, opp_call=None, yards=5.0, execute=True):
    """Run one full recommend -> execute -> reveal -> close cycle; returns the card."""
    card = co.recommend(st)
    rec = co.snaps[st.snap_id].recommended
    if execute:
        co.record_execution(st, rec, EXEC_CONFIRMED)
    if opp_call:
        f, p = opp_call
        mk = K.defense_play if co.side == OFF else K.offense_play
        co.record_opponent(st, OpponentAction(DEF if co.side == OFF else OFF, f, p, mk(f, p).signature, "typed_exact", 1.0))
    co.close_snap(st, Outcome(yards, yards >= (st.distance or 99), False, False, False, yards >= 15))
    return card
