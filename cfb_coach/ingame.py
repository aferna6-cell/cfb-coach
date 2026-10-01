"""In-game adjustment: bench calls that keep failing this game (CFB + Madden, O + D).

A call (formation + play) is BENCHED once it has failed ``BENCH_FAILS`` times in the
current half with fewer successes than failures. Benched calls are dropped from the
live menu until halftime; the second half starts clean. When the quarter isn't logged
the window is the last ``WINDOW_SNAPS`` snaps on that side of this game instead.

Defense also tracks coverage *families* (Cover 4 / Cover 2 / Cover 3 / man / pressure):
a family that keeps getting beaten gets a ranking penalty, so one bad Quarters variant
doesn't just get swapped for another Quarters variant.

Needs the live session id (``sit.extras["session_id"]``); without one nothing is benched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from cfb_coach.outcome import outcome_success

BENCH_FAILS = 3
WINDOW_SNAPS = 16
FAMILY_FAILS = 4
FAMILY_MAX_RATE = 0.4
FAMILY_PENALTY = 0.3


@dataclass
class CallRecord:
    formation: str
    play: str
    success: bool | None
    quarter: int | None
    concept: str = ""
    coverage: str = ""


@dataclass
class BenchReport:
    side: str
    scope: str  # "1st half" | "2nd half" | "last N snaps" | ""
    benched: dict[tuple[str, str], str] = field(default_factory=dict)  # (form, play) -> "0/3 this half"
    family_penalty: dict[str, float] = field(default_factory=dict)  # coverage family -> penalty (defense)

    def is_benched(self, formation: str, play: str) -> bool:
        return (formation, play) in self.benched

    def lines(self) -> list[str]:
        return [f"{f} — {p} ({why})" for (f, p), why in self.benched.items()]

    def note(self) -> str:
        if not self.benched:
            return ""
        names = ", ".join(p for _, p in list(self.benched)[:4])
        until = {"1st half": "halftime", "2nd half": "end of game"}.get(self.scope, "it drops out of the last snaps")
        return f"BENCHED until {until}: {names}"


def half_of(quarter: int | None) -> int | None:
    if not quarter:
        return None
    return 1 if int(quarter) <= 2 else 2


def situation_context(sit: Any) -> tuple[str | None, int | None]:
    extras = getattr(sit, "extras", None) or {}
    q = extras.get("quarter")
    try:
        q = int(q) if q not in (None, "") else None
    except (TypeError, ValueError):
        q = None
    return extras.get("session_id") or None, q


def _row(s: Any, key: str) -> Any:
    try:
        return s[key]
    except (KeyError, IndexError, TypeError):
        return getattr(s, key, None)


def session_records(db: Any, session_id: str | None, side: str) -> list[CallRecord]:
    if db is None or not session_id:
        return []
    try:
        snaps = db.get_session_snaps(session_id)
    except Exception:  # noqa: BLE001 — never break a live call
        return []
    out: list[CallRecord] = []
    for s in snaps:
        if (_row(s, "side") or "offense") != side:
            continue
        out.append(CallRecord(
            formation=_row(s, "formation") or "",
            play=_row(s, "play") or "",
            success=outcome_success(_row(s, "result"), side),
            quarter=_row(s, "quarter"),
            concept=_row(s, "concept_seen") or "",
            coverage=_row(s, "coverage_seen") or "",
        ))
    return out


def current_window(records: list[CallRecord], quarter: int | None) -> tuple[list[CallRecord], str]:
    half = half_of(quarter)
    if half is not None and any(r.quarter for r in records):
        return [r for r in records if half_of(r.quarter) == half], ("1st half" if half == 1 else "2nd half")
    return records[-WINDOW_SNAPS:], f"last {WINDOW_SNAPS} snaps"


def bench_report(
    db: Any,
    session_id: str | None,
    side: str,
    quarter: int | None = None,
    *,
    family_of: Callable[[str], str | None] | None = None,
    records: Iterable[CallRecord] | None = None,
) -> BenchReport:
    recs = list(records) if records is not None else session_records(db, session_id, side)
    window, scope = current_window(recs, quarter)
    rep = BenchReport(side=side, scope=scope if recs else "")
    tally: dict[tuple[str, str], list[int]] = {}
    for r in window:
        if r.success is None or not r.play:
            continue
        t = tally.setdefault((r.formation, r.play), [0, 0])
        t[0 if r.success else 1] += 1
    for key, (wins, fails) in tally.items():
        if fails >= BENCH_FAILS and wins < fails:
            rep.benched[key] = f"{wins}/{wins + fails} worked, {scope}"
    if family_of is not None:
        fam: dict[str, list[int]] = {}
        for r in window:
            f = family_of(r.play) if r.play else None
            if not f or r.success is None:
                continue
            t = fam.setdefault(f, [0, 0])
            t[0 if r.success else 1] += 1
        for f, (wins, fails) in fam.items():
            if fails >= FAMILY_FAILS and wins / (wins + fails) < FAMILY_MAX_RATE:
                rep.family_penalty[f] = FAMILY_PENALTY
    return rep


def bench_for_situation(db: Any, sit: Any, side: str, **kw: Any) -> BenchReport:
    session_id, quarter = situation_context(sit)
    return bench_report(db, session_id, side, quarter, **kw)


def drop_benched(rows: list[dict[str, Any]], rep: BenchReport) -> list[dict[str, Any]]:
    """Ranked rows minus benched calls; the full list if benching would empty the menu."""
    if not rep.benched:
        return rows
    kept = [r for r in rows if not rep.is_benched(r.get("formation") or "", r.get("play") or "")]
    return kept or rows


__all__ = [
    "BENCH_FAILS",
    "BenchReport",
    "CallRecord",
    "bench_for_situation",
    "bench_report",
    "current_window",
    "drop_benched",
    "half_of",
    "session_records",
    "situation_context",
]
