"""A typed play or result with no new down is the snap that just happened.

``mesh``, ``4 verts``, ``cover 2``, ``+7``, ``td``, and ``int`` record that snap.
They do not ask for another play call. ``undo`` removes the last one.
A line that still has a down (``2&7 mesh``) stays a new situation, and the play
name on it stays the previous-snap context for that call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from cfb_coach.outcome import outcome_success, parse_outcome

_RESULT_HEAD = re.compile(
    r"^(?P<head>[+-]\d+|gain\s+\d+|loss\s+\d+|touchdown|td|interception|int|"
    r"sack|incomplete|inc|stop|convert(?:ed)?|fumble(?:\s+lost)?)\b\s*(?P<rest>.*)$",
    re.I,
)
_BARE_YARDS = re.compile(r"^\d{1,2}$")


@dataclass
class BallSpot:
    """Where the ball is after the last logged result. Yards are from our goal."""

    down: int | None = None
    distance: int | None = None
    yardline: int | None = None
    score_us: int | None = None
    score_them: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "down": self.down,
            "distance": self.distance,
            "yardline": self.yardline,
            "score_us": self.score_us,
            "score_them": self.score_them,
        }


def _parse(parse_situation: Callable[..., Any], text: str, side: str) -> Any:
    return parse_situation(text, default_side=side) if text else None


def classify_last_snap(
    raw: str, parse_situation: Callable[..., Any], *, side: str = "offense",
) -> dict[str, Any] | None:
    """A completed snap (play, look, or result) with no new down. None = a new situation.

    A live marker (``showing`` / ``live`` / ``pre-snap``) is the look for the next
    call, so it stays a situation even with no down.
    """
    text = (raw or "").strip()
    if not text:
        return None
    sit = _parse(parse_situation, text, side)
    if getattr(sit, "down", None) is not None:
        return None
    if getattr(sit, "coverage_source", None) == "live" or getattr(sit, "concept_source", None) == "live":
        return None

    result: str | None = None
    rest = text
    matched = _RESULT_HEAD.match(text)
    if matched:
        result = matched.group("head")
        rest = (matched.group("rest") or "").strip()
    elif _BARE_YARDS.match(text):
        result = text
        rest = ""

    look = _parse(parse_situation, rest, side) if rest else None
    if look is not None and (
        getattr(look, "coverage_source", None) == "live" or getattr(look, "concept_source", None) == "live"
    ):
        return None
    cov = getattr(look, "coverage_hint", None) if look is not None else None
    concept = getattr(look, "concept_hint", None) if look is not None else None
    if result is None and not cov and not concept:
        return None
    canon = parse_outcome(result).to_result_text() if result else None
    if canon == "unknown":
        canon = result
    return {"result": canon, "coverage": cov, "concept": concept, "raw": text}


def refresh_marks(sit: Any) -> None:
    """Recompute short/long and field zone after a spot fill."""
    dist = getattr(sit, "distance", None)
    down = getattr(sit, "down", None)
    if dist is not None:
        sit.short_yardage = dist <= 3
        sit.long_yardage = (down in (2, 3, 4) and dist >= 8) or (down == 1 and dist >= 15)
    yl = getattr(sit, "yardline", None)
    if yl is None:
        return
    from cfb_coach.zones import GOAL_LINE, RED_ZONE, field_zone

    zone = field_zone(yl, dist)
    sit.goal_line = zone == GOAL_LINE
    sit.red_zone = zone in (RED_ZONE, GOAL_LINE)


def advance_ball(sit: Any, result: str, side: str, spot: BallSpot) -> tuple[BallSpot, str]:
    """Next down, distance, yardline, and score after ``result`` on ``sit``."""
    oc = parse_outcome(result)
    new = replace(spot)
    offense = not (side or "offense").lower().startswith("d")
    yl = getattr(sit, "yardline", None)
    if yl is None:
        yl = spot.yardline
    down = getattr(sit, "down", None) or spot.down
    dist = getattr(sit, "distance", None)
    if dist is None:
        dist = spot.distance

    if oc.kind == "td":
        if offense:
            new.score_us = (spot.score_us or 0) + 6
        else:
            new.score_them = (spot.score_them or 0) + 6
        return new, f"TD — score {new.score_us or 0}-{new.score_them or 0} (type the next down)"

    if oc.kind in ("int", "fumble"):
        return new, "turnover — type the next down"

    yards = oc.yards
    if oc.kind in ("incomplete", "stop"):
        yards = 0
    elif oc.kind == "sack" and yards is None:
        yards = 0
    elif oc.kind == "convert" and yards is None and dist is not None:
        yards = dist
    if yards is None or down is None or dist is None:
        return new, ""

    gain = yards
    if yl is not None:
        moved = yl + gain if offense else yl - gain
        new.yardline = max(1, min(99, moved))
    if gain >= dist:
        new.down = 1
        if new.yardline is None:
            new.distance = 10
        elif offense:
            new.distance = min(10, max(1, 100 - new.yardline))
        else:
            new.distance = min(10, max(1, new.yardline))
    else:
        nxt = down + 1
        if nxt > 4:
            new.down = None
            new.distance = None
            where = f" yl{new.yardline}" if new.yardline is not None else ""
            return new, f"turnover on downs{where} — type the next down"
        new.down = nxt
        new.distance = max(1, dist - gain)
    where = f" yl{new.yardline}" if new.yardline is not None else ""
    return new, f"now {new.down}&{new.distance}{where}"


def shown_macro(call: Any) -> str:
    """The Custom Adjustment this call puts on screen, or ``none``."""
    if call is None:
        return "none"
    mid = getattr(call, "macro", None)
    if mid and str(mid).strip().lower() not in ("", "none", "null"):
        return str(mid).strip()
    from cfb_coach.madden.macro_policy import macro_id_from_suggest

    return macro_id_from_suggest(getattr(call, "suggest_macro", None)) or "none"


def _macro_of(call: Any) -> str | None:
    if call is None:
        return None
    return shown_macro(call)


def _prev_look(parse_situation: Callable[..., Any], text: str, side: str) -> tuple[str | None, str | None]:
    """Coverage/concept on ``text`` only when it is a previous snap, not a live look."""
    sit = _parse(parse_situation, text, side)
    if sit is None:
        return None, None
    cov = sit.coverage_hint if getattr(sit, "coverage_source", None) == "last" else None
    concept = sit.concept_hint if getattr(sit, "concept_source", None) == "last" else None
    return cov, concept


@dataclass
class _Undo:
    snap_id: int
    inserted: bool
    prev: dict[str, Any]
    bumps: list[tuple[str, str, int, int]]
    spot: BallSpot
    logged_id: int | None
    result: str | None
    concept: str | None
    coverage: str | None


@dataclass
class LastSnapBook:
    """The call waiting on its result, and the ball spot after that result."""

    db: Any
    opponent_id: str
    parse_situation: Callable[..., Any]
    session_id: str | None = None
    call: Any = None
    sit: Any = None
    logged_id: int | None = None
    result: str | None = None
    concept: str | None = None
    coverage: str | None = None
    spot: BallSpot = field(default_factory=BallSpot)
    wrote: bool = False
    _undos: list[_Undo] = field(default_factory=list)

    def remember_call(self, call: Any, sit: Any) -> None:
        self.call = call
        self.sit = sit
        self.logged_id = None
        self.result = None
        self.concept = None
        self.coverage = None
        self.wrote = False
        # The call they just asked for is where the ball is until a result moves it.
        if getattr(sit, "down", None):
            self.spot.down = sit.down
            self.spot.distance = sit.distance
        if getattr(sit, "yardline", None) is not None:
            self.spot.yardline = sit.yardline
        if getattr(sit, "score_us", None) is not None:
            self.spot.score_us = sit.score_us
            self.spot.score_them = sit.score_them

    def stamp_situation(self, sit: Any) -> None:
        """Fill a new situation from the ball spot when the line left a field out."""
        filled = False
        if getattr(sit, "down", None) is None and self.spot.down:
            sit.down = self.spot.down
            sit.distance = self.spot.distance
            filled = True
        if (
            getattr(sit, "yardline", None) is None
            and self.spot.yardline is not None
            and getattr(sit, "down", None)
        ):
            sit.yardline = self.spot.yardline
            filled = True
        if self.spot.score_us is not None and getattr(sit, "score_us", None) is None:
            sit.score_us = self.spot.score_us
            sit.score_them = self.spot.score_them
        if filled:
            refresh_marks(sit)

    def handle(self, raw: str) -> str | None:
        """Record ``raw`` as the last snap. None means it is a new situation."""
        side = getattr(self.call, "side", None) or getattr(self.sit, "side", None) or "offense"
        spec = classify_last_snap(raw, self.parse_situation, side=side)
        if spec is None:
            return None
        if self.call is None or self.sit is None:
            return "  No call to log yet."
        return "  " + self._record(spec, inherit_sit_look=True)

    def close_from_form(
        self, outcome: str, their: str = "", next_raw: str | None = None,
    ) -> dict[str, Any] | None:
        """Browser submit: outcome, optional 'they ran', and the next line's last-play field."""
        if self.call is None or self.sit is None or not (outcome or "").strip():
            return None
        side = getattr(self.call, "side", None) or "offense"
        cov, concept = _prev_look(self.parse_situation, their, side)
        if next_raw and not cov and not concept:
            cov, concept = _prev_look(self.parse_situation, next_raw, side)
        canon = parse_outcome(outcome).to_result_text()
        spec = {
            "result": outcome.strip() if canon == "unknown" else canon,
            "coverage": cov,
            "concept": concept,
            "raw": outcome,
        }
        note = self._record(spec, inherit_sit_look=next_raw is None)
        call = self.call
        return {
            "formation": call.formation or "?",
            "play": call.play or "?",
            "result": self.result or "",
            "look": self.concept or self.coverage or "",
            "call_text": call.format(),
            "side": call.side,
            "note": note,
        }

    def undo(self) -> str:
        if not self._undos:
            return "  Nothing to undo."
        frame = self._undos.pop()
        for bucket, key, amount, succ in frame.bumps:
            self.db.adjust_tendency(
                self.opponent_id, bucket, key, amount=-amount, success_delta=-succ,
            )
        if frame.inserted:
            self.db.delete_snap(frame.snap_id)
        else:
            self.db.update_snap(frame.snap_id, **frame.prev)
        self.logged_id = frame.logged_id
        self.result = frame.result
        self.concept = frame.concept
        self.coverage = frame.coverage
        self.spot = frame.spot
        self.wrote = False
        return f"  undo: removed the last snap note (id {frame.snap_id})"

    def _record(self, spec: dict[str, Any], *, inherit_sit_look: bool) -> str:
        from cfb_coach.game_score import snap_notes_for
        from cfb_coach.tendency import mild_bump_concept, mild_bump_coverage, situation_bucket

        prev_result, prev_concept, prev_coverage = self.result, self.concept, self.coverage
        prev_spot = replace(self.spot)
        result = spec.get("result") or self.result
        concept = spec.get("concept") or self.concept
        coverage = spec.get("coverage") or self.coverage
        if inherit_sit_look and not spec.get("concept") and not spec.get("coverage"):
            concept = concept or getattr(self.sit, "concept_hint", None)
            coverage = coverage or getattr(self.sit, "coverage_hint", None)

        side = getattr(self.call, "side", None) or "offense"
        success = bool(outcome_success(result, side))
        bumps: list[tuple[str, str, int, int]] = []
        sit = self.sit
        typed_look = bool(spec.get("concept") or spec.get("coverage"))
        inherit = inherit_sit_look and spec.get("result") and not typed_look

        def take(kind: str, name: str | None, old: str | None) -> None:
            if not name or name == old:
                return
            if kind == "concept" and str(side).startswith("d"):
                mild_bump_concept(self.db, self.opponent_id, name, sit, success=success)
                delta = 1 if success else 0
                bumps.append(("offense_concept", name, 1, delta))
                bumps.append((f"offense_concept|{situation_bucket(sit)}", name, 1, delta))
            elif kind == "coverage" and not str(side).startswith("d"):
                mild_bump_coverage(self.db, self.opponent_id, name, sit)
                bumps.append(("their_coverage", name, 1, 0))
                bumps.append((f"their_coverage|{situation_bucket(sit)}", name, 1, 0))

        if spec.get("concept") or inherit:
            take("concept", concept, prev_concept)
        if spec.get("coverage") or inherit:
            take("coverage", coverage, prev_coverage)

        inserted = self.logged_id is None
        prev_row = {"result": prev_result, "coverage_seen": prev_coverage, "concept_seen": prev_concept}
        if inserted:
            quarter = (getattr(sit, "extras", None) or {}).get("quarter")
            self.logged_id = self.db.log_snap(
                opponent_id=self.opponent_id,
                side=side,
                situation_raw=getattr(sit, "raw", "") or "",
                our_call=self.call.format().split("\n")[0],
                formation=getattr(self.call, "formation", None),
                play=getattr(self.call, "play", None),
                macro=_macro_of(self.call),
                down=getattr(sit, "down", None),
                distance=getattr(sit, "distance", None),
                yardline=getattr(sit, "yardline", None),
                quarter=quarter,
                result=result,
                coverage_seen=coverage,
                concept_seen=concept,
                notes=snap_notes_for(sit),
                session_id=self.session_id,
            )
        else:
            self.db.update_snap(
                self.logged_id, result=result, coverage_seen=coverage, concept_seen=concept,
            )
        self.result, self.concept, self.coverage = result, concept, coverage
        note = ""
        if spec.get("result"):
            self.spot, note = advance_ball(sit, spec["result"], side, self.spot)
        self._undos.append(_Undo(
            snap_id=self.logged_id,
            inserted=inserted,
            prev=prev_row,
            bumps=bumps,
            spot=prev_spot,
            logged_id=None if inserted else self.logged_id,
            result=prev_result,
            concept=prev_concept,
            coverage=prev_coverage,
        ))
        self.wrote = True
        bits = [b for b in (result, concept, coverage) if b]
        head = "logged last snap: " + " | ".join(bits) if bits else "logged last snap"
        if note:
            head += f" | {note}"
        return head


__all__ = [
    "BallSpot", "LastSnapBook", "advance_ball", "classify_last_snap", "refresh_marks",
    "shown_macro",
]
