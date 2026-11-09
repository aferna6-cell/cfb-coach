"""Live-input parsing for the continuous session. Reuses the existing situation and outcome parsers.

Conventions (same as the established live loop):
* situation text:  ``o 1&10 my 22``  /  ``d 3&7 opp 40``  (o/d = who has the ball; yardlines from OUR goal)
* result text is the OFFENSE's result on that snap ("+7", "inc", "sack -6", "int", "td") — on a defensive snap it is what
  the opponent's offense got.
* clock, punt and pre-snap "showing" text are accepted but never used by the decision (owner preference).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from cfb_coach.madden import catalog_full
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.outcome import parse_outcome
from cfb_coach.madden.unified import knowledge as K
from cfb_coach.madden.unified.types import DEF, OFF, GameState, OpponentAction, Outcome

_SIDE_TOKEN = re.compile(r"^\s*(o|off|offense|d|def|defense)\b[\s:,-]*", re.I)


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class SituationInput:
    side: str
    down: int | None
    distance: int | None
    yardline: int | None  # as GameState wants it: from the goal the OFFENSE defends
    score_us: int | None
    score_opp: int | None
    ignored: tuple[str, ...]


def parse_snap_line(text: str, last_side: str | None) -> SituationInput:
    t = text.strip()
    m = _SIDE_TOKEN.match(t)
    if m:
        side = OFF if m.group(1).lower().startswith("o") else DEF
        body = t[m.end():]
    elif re.search(r"\b(their|opp(?:onent)?)\s*ball\b", t, re.I):
        side, body = DEF, t
    elif re.search(r"\b(our|we\s*have)\s*(ball)?\b", t, re.I) and "ball" in t.lower():
        side, body = OFF, t
    elif last_side:
        side, body = last_side, t
    else:
        raise InputError("who has the ball? start the line with o (we have it) or d (they have it), e.g. 'o 1&10 my 22'")
    sit = parse_madden_situation(body, default_side="offense" if side == OFF else "defense")
    if sit.down is None or sit.distance is None:
        raise InputError("need down and distance, e.g. '3&7' or '3rd and 7'")
    if sit.yardline is None:
        raise InputError("need the ball position, e.g. 'my 35' or 'opp 40'")
    # parser yardline = yards from OUR goal. GameState wants yards from the goal the offense defends.
    yl = sit.yardline if side == OFF else 100 - sit.yardline
    yl = max(1, min(99, int(yl)))
    ignored = []
    if sit.two_minute:
        ignored.append("clock")
    if sit.coverage_source == "live" or sit.concept_source == "live":
        ignored.append("pre-snap showing")
    return SituationInput(side, sit.down, sit.distance, yl, sit.score_us, sit.score_them, tuple(ignored))


def parse_result(text: str, st: GameState) -> Outcome:
    po = parse_outcome(text)
    low = text.lower()
    if po.kind == "unknown":
        raise InputError("could not read the result; use e.g. '+7', '-2', 'inc', 'sack -6', 'int', 'fumble', 'td'")
    td = po.kind == "td" or bool(re.search(r"\b(td|touchdown)\b", low))
    yards = po.yards
    if td and yards is None:
        yards = st.to_goal
    if po.kind in ("int", "fumble", "incomplete") and yards is None:
        yards = 0
    fd_flag = bool(re.search(r"\b(fd|1st|first\s*down|convert(?:ed)?)\b", low))
    first = td or fd_flag or (yards is not None and st.distance is not None and yards >= st.distance)
    return Outcome(None if yards is None else float(yards), bool(first), td, po.kind in ("int", "fumble"), po.kind == "sack",
                   None if yards is None else yards >= 15, po.to_result_text())


@lru_cache(maxsize=2)
def _catalog_pairs(side: str) -> frozenset[tuple[str, str]]:
    out = set()
    for plays in catalog_full.book_plays(side).values():
        for f, ps in plays.items():
            out.update((f, p) for p in ps)
    return frozenset(out)


def parse_opponent_call(text: str, opp_side: str) -> OpponentAction:
    """'formation | play' (exact, only if that pair exists in the Madden catalog) or free text (family-level, inferred)."""
    t = text.strip()
    if not t:
        raise InputError("empty opponent call")
    mk = K.defense_play if opp_side == DEF else K.offense_play
    if "|" in t:
        f, p = (x.strip() for x in t.split("|", 1))
        exact = (f, p) in _catalog_pairs(opp_side)
        play = mk(f, p)
        if opp_side == DEF and play.signature == ("unknown", "unknown", "unknown"):
            raise InputError(f"{p!r} is not a coverage/play I can classify; check the spelling")
        return OpponentAction(opp_side, f, p, play.signature, "typed_exact" if exact else "visual_inferred", 1.0 if exact else 0.5)
    if opp_side == DEF:
        fam, typ, toks = K.parse_defense_name(t)
        if fam == "unknown":
            raise InputError("could not classify that coverage; type 'formation | play' or a name like 'cover 3 sky'")
        sig = K.defense_signature_by_name(t)
        if sig is not None:  # the name maps to exactly one catalog signature
            return OpponentAction(DEF, None, t, sig, "visual_inferred", 0.7)
        play = K.defense_play("", t)  # family-level only; no formation or exact variant claimed
        return OpponentAction(DEF, None, t, play.signature, "visual_inferred", 0.5)
    play = K.offense_play("", t)
    if play.signature[1] in ("unknown",) and play.play_type == "unknown":
        raise InputError("could not classify that offensive call; type 'formation | play'")
    return OpponentAction(OFF, None, t, play.signature, "visual_inferred", 0.5)
