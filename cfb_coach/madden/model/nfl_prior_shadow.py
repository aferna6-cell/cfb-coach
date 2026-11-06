"""Shadow-only NFL situational prior for the live offense model.

Off unless ``CFB_COACH_NFL_SHADOW`` names an exported table (see
``scripts/madden_ml/export_nfl_prior_table.py``). When on, it re-scores the
model's ranked candidates with the NFL prior blended in and appends one JSON
line per snap. It never changes the call, the ranking or the logged
decision, and any error or slow lookup simply skips the shadow record.

Pure Python: the table is a static JSON lookup, so the live process needs no
xgboost, numpy or pandas.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ENV_TABLE = "CFB_COACH_NFL_SHADOW"
ENV_LOG = "CFB_COACH_NFL_SHADOW_LOG"
TABLE_SCHEMA = "madden.nfl_prior_shadow_table.v1"
SHADOW_BUDGET_MS = 5.0
MAX_CANDIDATES = 8
EPS = 1e-6

_lock = threading.Lock()
_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, float(p)))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def enabled_table_path() -> Path | None:
    raw = os.environ.get(ENV_TABLE, "").strip()
    return Path(raw).expanduser() if raw else None


def validate_table(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("schema") != TABLE_SCHEMA:
        raise ValueError(f"unexpected NFL shadow table schema: {raw.get('schema')!r}")
    for key in ("model_version", "distance_edges", "yardline_edges", "cells", "platt", "base_rate", "prior_strength"):
        if key not in raw:
            raise ValueError(f"NFL shadow table missing {key}")
    a, b = (float(v) for v in raw["platt"])
    base = float(raw["base_rate"])
    if not 0.0 < base < 1.0 or not all(math.isfinite(v) for v in (a, b)):
        raise ValueError("NFL shadow table has invalid calibration")
    return dict(raw)


def load_table(path: Path) -> dict[str, Any]:
    key = str(path)
    mtime = path.stat().st_mtime
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    table = validate_table(json.loads(path.read_text(encoding="utf-8")))
    with _lock:
        _cache[key] = (mtime, table)
    return table


def _bin(value: float, edges: Sequence[float]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def yards_to_goal(yardline: Any) -> float | None:
    """Live offense yardline is yards from our own goal."""
    try:
        yl = int(yardline)
    except (TypeError, ValueError):
        return None
    return float(min(99, max(1, 100 - yl)))


def play_input(play: str | None) -> str:
    from cfb_coach.madden.model.heldout_eval import run_or_pass

    return {"run": "run", "pass": "pass", "screen": "pass"}.get(run_or_pass(play), "mixture")


def nfl_probability(table: Mapping[str, Any], *, down: Any, distance: Any, yardline: Any,
                    play: str | None) -> tuple[float, str, str] | None:
    """Calibrated NFL success probability for this play, or None outside the table."""
    try:
        d = int(down)
        dist = float(distance)
    except (TypeError, ValueError):
        return None
    ytg = yards_to_goal(yardline)
    if d not in (1, 2, 3, 4) or ytg is None:
        return None
    cell_key = f"{d}|{_bin(dist, table['distance_edges'])}|{_bin(ytg, table['yardline_edges'])}"
    cell = table["cells"].get(cell_key)
    if not cell:
        return None
    kind = play_input(play)
    if kind == "mixture":
        pp = float(cell["p_pass"])
        raw = pp * float(cell["pass"]) + (1 - pp) * float(cell["run"])
    else:
        raw = float(cell[kind])
    a, b = (float(v) for v in table["platt"])
    return _sigmoid(a + b * _logit(raw)), kind, cell_key


def blend(p_model: float, p_nfl: float, base: float, w: float) -> float:
    return _sigmoid(_logit(p_model) + w * (_logit(p_nfl) - _logit(base)))


def score_candidates(table: Mapping[str, Any], ranked: Sequence[Mapping[str, Any]], *, down: Any,
                     distance: Any, yardline: Any) -> list[dict[str, Any]]:
    base = float(table["base_rate"])
    w = float(table["prior_strength"])
    out = []
    for r in ranked[:MAX_CANDIDATES]:
        p_model = float(r["probability"])
        hit = nfl_probability(table, down=down, distance=distance, yardline=yardline, play=r.get("play"))
        row = {"formation": r.get("formation"), "play": r.get("play"), "p_model": round(p_model, 4)}
        if hit is not None:
            p_nfl, kind, cell = hit
            row.update(p_nfl=round(p_nfl, 4), p_blend=round(blend(p_model, p_nfl, base, w), 4),
                       nfl_input=kind, cell=cell)
        out.append(row)
    return out


def _log_path(table_path: Path) -> Path:
    raw = os.environ.get(ENV_LOG, "").strip()
    return Path(raw).expanduser() if raw else table_path.with_name("nfl_prior_shadow_log.jsonl")


def observe(*, ranked: Sequence[Mapping[str, Any]], chosen: tuple[str, str], sit: Any, opponent_type: str,
            session_id: str | None, snap_id: str | None, snap_seq: int | None,
            remaining_ms: float) -> dict[str, Any] | None:
    """Shadow record for one live offense decision, or None when off, slow or broken.

    Never raises. The caller's pick is passed in only so it can be logged.
    """
    path = enabled_table_path()
    if path is None:
        return None
    started = time.perf_counter()
    try:
        if remaining_ms < SHADOW_BUDGET_MS:
            return {"status": "skipped_budget", "remaining_ms": round(remaining_ms, 3)}
        table = load_table(path)
        down, distance, yardline = (getattr(sit, k, None) for k in ("down", "distance", "yardline"))
        scored = score_candidates(table, ranked, down=down, distance=distance, yardline=yardline)
        blended = [r for r in scored if "p_blend" in r]
        shadow_top = max(blended, key=lambda r: r["p_blend"]) if blended else None
        chosen_row = next((r for r in scored if (r["formation"], r["play"]) == tuple(chosen)), None)
        latency = (time.perf_counter() - started) * 1000.0
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": "ok" if latency <= SHADOW_BUDGET_MS else "over_budget",
            "model_version": table["model_version"],
            "prior_strength": float(table["prior_strength"]),
            "in_validated_scope": opponent_type == "cpu",
            "session_id": session_id, "snap_id": snap_id, "snap_seq": snap_seq,
            "situation": {"down": down, "distance": distance, "yardline": yardline,
                          "yards_to_goal": yards_to_goal(yardline)},
            "opponent_type": opponent_type,
            "live_pick": {"formation": chosen[0], "play": chosen[1]},
            "live_pick_scores": chosen_row,
            "shadow_top": shadow_top,
            "shadow_would_differ": bool(shadow_top and (shadow_top["formation"], shadow_top["play"]) != tuple(chosen)),
            "confidence": None if chosen_row is None or "p_blend" not in chosen_row
            else round(abs(chosen_row["p_blend"] - 0.5) * 2, 4),
            "candidates": scored,
            "latency_ms": round(latency, 3),
        }
        log = _log_path(path)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return {k: record[k] for k in ("status", "model_version", "shadow_would_differ", "latency_ms")}
    except Exception as exc:  # noqa: BLE001 — the shadow must never affect the live call
        return {"status": "error", "error": type(exc).__name__,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3)}
