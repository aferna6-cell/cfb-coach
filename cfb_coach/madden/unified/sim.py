"""Synthetic game generator with a KNOWN matchup truth.

Used only to validate the learning machinery (does the hierarchical model recover a planted effect,
do opponent beliefs beat a frequency baseline when the opponent has a planted sequential habit).
It says nothing about real Madden football and every report that uses it says so.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from cfb_coach.madden.unified import knowledge as K
from cfb_coach.madden.unified.types import (DEF, EXEC_CONFIRMED, OFF, FootballAction, GameState, OpponentAction, Outcome,
                                            SnapObservation)

# planted truth: (offense concept, defense family) -> true mean-yard shift
TRUTH = {("mesh", "cover1"): +2.2, ("mesh", "cover3"): -0.8, ("four_verts", "cover3"): -1.8, ("four_verts", "cover1"): +1.5,
         ("inside_zone", "cover6"): -1.0}
DEF_CHOICES = [("Cover 3 Sky", "4-3 Over"), ("Cover 1 Hole", "4-3 Over"), ("Cover 6", "4-3 Over"), ("Cover 2 Man", "4-3 Over")]
OFF_PLAYS = [("Gun Trips X Nasty", "Mtn Stick Wheel"), ("Gun Trips X Nasty", "WR Smash"), ("Gun Trips X Nasty", "Inside Zone"),
             ("Gun Trips X Nasty", "Verticals X Shallow"), ("Gun Trips X Nasty", "Quick Unders")]


def _effect(concept: str, fam: str, sd: float = 1.3) -> float:
    """Deterministic pseudo-random planted effect per (concept, family) cell; the cold prior does not know it."""
    import hashlib
    h = int(hashlib.sha256(f"{concept}|{fam}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    h2 = int(hashlib.sha256(f"{fam}|{concept}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    import math
    return sd * math.sqrt(-2 * math.log(max(h, 1e-9))) * math.cos(2 * math.pi * h2)


def true_mean(concept: str, fam: str, base: float) -> float:
    return base + TRUTH.get((concept, fam), _effect(concept, fam))


def gen_games(n_games: int, plays_per_game: int = 40, *, seed: int = 0, our_side: str = OFF,
              habit: float = 0.55) -> list[list[SnapObservation]]:
    rng = random.Random(seed)
    games: list[list[SnapObservation]] = []
    for g in range(n_games):
        obs: list[SnapObservation] = []
        prev_def = None
        for i in range(1, plays_per_game + 1):
            st = GameState(f"sim{g:03d}", i, OFF, "cpu_sim", rng.choice((1, 2, 3)), rng.choice((3, 6, 10)), rng.randint(20, 80),
                           opponent_class="cpu")
            if prev_def is not None and rng.random() < habit:
                d_form, d_play = prev_def
            else:
                d_play, d_form = rng.choice(DEF_CHOICES)
                d_form, d_play = d_form, d_play
            prev_def = (d_form, d_play)
            f, p = rng.choice(OFF_PLAYS)
            off, dfn = K.offense_play(f, p), K.defense_play(d_form, d_play)
            base = 4.0 if off.play_type == "run" else 6.0
            mean = true_mean(off.concept, dfn.signature[1], base)
            yards = round(rng.gauss(mean, 5.0))
            out = Outcome(float(yards), yards >= st.distance, yards >= (st.to_goal or 99), rng.random() < 0.02, False, yards >= 15)
            o = SnapObservation(st, recommended=FootballAction(off), executed=FootballAction(off), executed_status=EXEC_CONFIRMED,
                                opponent_action=OpponentAction(DEF, d_form, d_play, dfn.signature, "typed_exact", 1.0),
                                outcome=out, closed=True, closed_ts="t", cutoff_seq=i)
            obs.append(o)
        games.append(obs)
    return games
