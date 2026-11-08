"""Historical opponent coverage mix from recorded ``coverage_seen`` labels (issue #130).

``mix_source`` is always ``opponent_history``: this is what was observed, never a counterfactual
profile. Only labels specific enough for a ``football_priors.descriptor`` key map to that key; coarse
shells map to ``cover_N_unspecified``; empty / unrecognized labels are unknown (never guessed).

Filter: ``snaps`` rows with ``side = 'offense'`` (we had the ball, so ``coverage_seen`` is the
opponent defense). For the CPU alias (``opponents.is_cpu_opponent(opponent_id)`` is true, e.g.
``"cpu"``), every snap whose ``opponent_id`` satisfies ``is_cpu_opponent`` is included; otherwise
``opponent_id`` must equal the given id (trimmed, case-insensitive). No other filters.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from cfb_coach.opponents import is_cpu_opponent

MIX_SOURCE = "opponent_history"
UNKNOWN = "UNKNOWN"

_SHELL = r"\b(?:cover|cov|c) ?{n}\b"


def _shell(t: str, n: str) -> bool:
    return re.search(_SHELL.format(n=n), t) is not None


def map_coverage_seen(raw: str | None) -> str | None:
    """Raw ``coverage_seen`` -> official descriptor key, or ``None`` when empty / unrecognized."""
    t = " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(raw or "").lower().replace("_", " ").replace("-", " ")).split())
    if not t:
        return None
    # Specific phrases first.
    if "tampa" in t:
        return "tampa_2"
    if "palm" in t:
        return "cover_4_palms"
    if "quarter" in t:
        return "cover_4_quarters"
    if _shell(t, "4") and "drop" in t:
        return "cover_4_drop"
    if "robber" in t:
        return "cover_1_robber"
    if "hole" in t:
        return "cover_1_hole"
    if "trap" in t:
        return "trap_2"
    if _shell(t, "0"):
        return "cover_0"
    if _shell(t, "6"):
        return "cover_6"
    if _shell(t, "1") and ("pressure" in t or "blitz" in t):
        return "cover_1_pressure"
    if _shell(t, "2"):
        if "invert" in t:
            return "cover_2_invert"
        if "man" in t:
            return "cover_2_man"
        if "spot" in t:
            return "cover_2_spot"
    if _shell(t, "3"):
        if "match" in t:
            return "cover_3_match"
        if "sky" in t:
            return "cover_3_sky"
        if "spot" in t:
            return "cover_3_spot"
    # Coarse buckets: never invent a variant.
    if _shell(t, "1") or t in ("man", "single high", "singlehigh", "man free", "cover one"):
        return "cover_1_unspecified"
    if _shell(t, "2"):
        return "cover_2_unspecified"
    if _shell(t, "3"):
        return "cover_3_unspecified"
    if _shell(t, "4"):
        return "cover_4_unspecified"
    return None   # includes bare "pressure"/"blitz" (no shell) and Cover 9 (no descriptor entry)


def _connect(db: Any) -> tuple[sqlite3.Connection, bool]:
    if isinstance(db, sqlite3.Connection):
        return db, False
    p = Path(db).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True), True


def extract_opponent_history_mix(db: Any, *, opponent_id: str = "cpu") -> dict[str, Any]:
    """Observed coverage mix vs ``opponent_id`` (CPU alias by default). ``db`` is a path (opened read-only) or connection."""
    con, owned = _connect(db)
    try:
        rows = con.execute("SELECT opponent_id, coverage_seen FROM snaps WHERE side = 'offense'").fetchall()
    finally:
        if owned:
            con.close()
    want = (opponent_id or "").strip().lower()
    cpu = is_cpu_opponent(want)
    labels = [cov for opp, cov in rows
              if (is_cpu_opponent(opp) if cpu else (opp or "").strip().lower() == want)]
    inv: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for raw in labels:
        key = map_coverage_seen(raw)
        label = "" if raw is None else str(raw)
        e = inv.setdefault(label, {"key": key or UNKNOWN, "count": 0})
        e["count"] += 1
        if key:
            counts[key] = counts.get(key, 0) + 1
    total, mapped = len(labels), sum(counts.values())
    return {
        "mix": {k: v / mapped for k, v in sorted(counts.items())} if mapped else {},
        "mix_source": MIX_SOURCE,
        "unknown_mass": (total - mapped) / total if total else 0.0,
        "opponent_id": "cpu" if cpu else want,
        "total_snaps": total, "mapped_snaps": mapped, "unmapped_snaps": total - mapped,
        "key_counts": dict(sorted(counts.items())),
        "raw_inventory": dict(sorted(inv.items())),
    }
