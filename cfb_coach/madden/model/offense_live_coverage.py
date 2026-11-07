"""Optional CPU pilot bridge: previous-play Madden coverage -> next-snap ML mix.

Reads the SAME `snaps.coverage_seen` records already written by LastSnapBook.
Does not add a second logger or label a pre-snap prediction as actual coverage.
Current/future snaps are excluded by an explicit monotone snap_seq fence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import sqlite3
from typing import Any, Mapping

from cfb_coach.madden.model import historical_defense_baseline as hb

# Before enough evidence accumulates, keep the learned historical prior dominant.
PRIOR_STRENGTH = 8.0
MAX_PRIOR_SNAPS = 40


@dataclass(frozen=True)
class CoverageContext:
    mix: dict[str, float]
    history_n: int
    named_n: int
    pressure_n: int
    unknown_n: int
    recent_named: tuple[str, ...]
    session_id: str
    next_snap_seq: int
    basis: str = "closed_manual_postplay_coverage_with_historical_prior"

    def as_log(self) -> dict[str, Any]:
        return {"basis": self.basis, "history_n": self.history_n, "named_n": self.named_n,
                "pressure_n": self.pressure_n, "unknown_n": self.unknown_n,
                "recent_named": list(self.recent_named), "mix": dict(self.mix),
                "session_id": self.session_id, "next_snap_seq": self.next_snap_seq}


def _prior(prior_mix: Mapping[str, float] | None) -> dict[str, float]:
    if not prior_mix:
        return {"unknown": 1.0}
    cleaned: dict[str, float] = {}
    for k, v in prior_mix.items():
        p = float(v)
        if not math.isfinite(p) or p < 0:
            raise ValueError("invalid historical coverage prior")
        if p:
            # A pressure *observation* is NOT an exact Cover 0 label.
            key = "unknown" if k in ("pressure", hb.PRESSURE_UNKNOWN) else k
            cleaned[key] = cleaned.get(key, 0.0) + p
    total = sum(cleaned.values())
    return {k: v / total for k, v in cleaned.items()} if total else {"unknown": 1.0}


def blend_counts(counts: Mapping[str, int], prior_mix: Mapping[str, float] | None) -> dict[str, float]:
    """Shared by live inference AND training: same-game counts blended into the historical prior.

    ``counts`` are earlier, closed, named coverage observations of THIS game and opponent only.
    """
    n = sum(counts.values())
    p0 = _prior(prior_mix)
    ratio = n / (n + PRIOR_STRENGTH) if n else 0.0
    p = {k: (1 - ratio) * p0.get(k, 0.0) + ratio * counts.get(k, 0) / (n or 1)
         for k in set(p0) | set(counts)}
    total = sum(p.values())
    return {k: v / total for k, v in sorted(p.items()) if v > 0}


def count_labels(labels: "list[str]") -> tuple[Counter, int, int, int]:
    """(named counts by bucket, n_named, n_pressure_only, n_unknown) for normalized labels."""
    named = [x for x in labels if x in hb.KNOWN_COVERAGES]
    counts = Counter(hb.GENERIC_TO_BUCKET.get(x, x) for x in named)
    return counts, len(named), sum(x == hb.PRESSURE_UNKNOWN for x in labels), sum(x == hb.UNRESOLVED for x in labels)


def coverage_for_next_snap(conn: sqlite3.Connection, *, session_id: str,
                           opponent_id: str, next_snap_seq: int | None,
                           prior_mix: Mapping[str, float] | None = None) -> CoverageContext:
    """Consume only *closed* prior offense snaps in this session/opponent.

    The caller must supply the NEXT snap's sequence number, not the last
    closed snap's. This is a read-only SQL SELECT; writes remain in the existing
    LastSnapBook's `close_from_form` / `_record` path.
    """
    if not session_id or not opponent_id or (next_snap_seq is not None and int(next_snap_seq) <= 0):
        raise ValueError("session_id, opponent_id and (when given) a positive next_snap_seq required")
    if next_snap_seq is None:
        # LIVE: the next call is made after every previous snap of this game was closed through the form,
        # so every closed row already in the DB is strictly earlier. (snap_seq can be NULL on older rows.)
        history = conn.execute(
            "SELECT snap_seq, coverage_seen FROM snaps WHERE session_id = ? AND opponent_id = ? "
            "AND side = 'offense' AND result IS NOT NULL AND trim(result) != '' "
            "ORDER BY id DESC LIMIT ?", (session_id, opponent_id, MAX_PRIOR_SNAPS)).fetchall()
    else:
        history = conn.execute(
            "SELECT snap_seq, coverage_seen FROM snaps WHERE session_id = ? AND opponent_id = ? "
            "AND side = 'offense' AND snap_seq IS NOT NULL AND snap_seq < ? "
            "AND result IS NOT NULL AND trim(result) != '' "
            "ORDER BY snap_seq DESC, id DESC LIMIT ?",
            (session_id, opponent_id, int(next_snap_seq), MAX_PRIOR_SNAPS),
        ).fetchall()
    labels = [hb.normalize_label(r[1]) for r in history]
    named = [x for x in labels if x in hb.KNOWN_COVERAGES]
    # A generic 'Cover 2' remains a generic Cover 2 probability. Do not turn
    # it into spot zone or Cover 2 Man. The structural model can marginalize.
    counts = Counter(hb.GENERIC_TO_BUCKET.get(x, x) for x in named)
    n = sum(counts.values())
    mix = blend_counts(counts, prior_mix)
    return CoverageContext(mix=mix, history_n=len(labels), named_n=n,
                           pressure_n=sum(x == hb.PRESSURE_UNKNOWN for x in labels),
                           unknown_n=sum(x == hb.UNRESOLVED for x in labels),
                           recent_named=tuple(named[:5]), session_id=session_id,
                           next_snap_seq=int(next_snap_seq or 0))