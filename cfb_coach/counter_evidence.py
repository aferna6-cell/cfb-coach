"""Counter evidence API — LiveTendency list for playcaller (no hardcoded finals)."""

from __future__ import annotations

from typing import Any

from cfb_coach.live_tendency import LiveTendency, LiveTendencyEngine, SAMPLE_ACTIONABLE, SAMPLE_MILD
from cfb_coach.situation import Situation
from cfb_coach.vision.play_record import PlayRecord


def tendencies_for_situation(
    engine: LiveTendencyEngine | None,
    sit: Situation,
    *,
    formation: str = "",
    shell: str = "",
    pressure: str = "",
    side: str | None = None,
) -> list[LiveTendency]:
    if engine is None:
        return []
    side = side or sit.side or "offense"
    zone = "gl" if sit.goal_line else ("rz" if sit.red_zone else "open")
    return engine.tendencies_for_context(
        down=sit.down,
        distance=sit.distance,
        field_zone=zone,
        formation=formation,
        shell=shell,
        pressure=pressure,
        side=side,
    )


def weight_for_tendency(t: LiveTendency) -> float:
    """Soft weight for playcaller mix — grows with sample, never single-snap."""
    if t.sample_size < SAMPLE_MILD:
        return 0.0
    base = {2: 0.25, 3: 0.55, 4: 0.7, 5: 0.85}.get(min(t.sample_size, 5), 0.9)
    return base * (0.5 + 0.5 * t.hit_rate)


def strongest_signal(tendencies: list[LiveTendency]) -> LiveTendency | None:
    actionable = [t for t in tendencies if t.sample_size >= SAMPLE_ACTIONABLE]
    pool = actionable or [t for t in tendencies if t.sample_size >= SAMPLE_MILD]
    if not pool:
        return None
    return max(pool, key=lambda t: (t.sample_size, t.hit_rate, weight_for_tendency(t)))


def live_concept_from_engine(
    engine: LiveTendencyEngine | None,
    sit: Situation,
) -> str | None:
    """Optional concept_hint boost from live engine (does not overwrite live look)."""
    ts = tendencies_for_situation(engine, sit, side=sit.side)
    best = strongest_signal(ts)
    if best is None:
        return None
    sig = best.signal
    # Map family/tag → concept string for existing playcaller paths
    mapping = {
        "CROSSERS": "crossers",
        "VERTICALS": "verticals",
        "FLOOD": "flood",
        "MESH": "mesh",
        "RUN_INSIDE": "inside zone",
        "RUN_OUTSIDE": "outside zone",
        "RPO": "rpo",
        "SCREEN": "screen",
        "SCRAMBLE": "scramble",
        "DROPBACK_PASS": "dropback",
        "QUICK_PASS": "quick pass",
        "PLAY_ACTION": "play action",
    }
    if sig in mapping:
        return mapping[sig]
    if sig.startswith("SHELL:") or sig.startswith("PRESSURE:") or sig.startswith("MOTION"):
        return None
    return sig.lower().replace("_", " ")


def ingest_play(engine: LiveTendencyEngine, rec: PlayRecord) -> list[Any]:
    return engine.add_play(rec)
