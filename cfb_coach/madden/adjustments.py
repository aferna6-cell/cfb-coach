"""Madden 27 pre-snap adjustments (v1.17) — with the exact Xbox buttons for each.

Offense uses no macros: when a look calls for it, the live call adds ONE adjustment (hot route,
audible or protection) from the research DB's offense adjustments. Defense adds a non-macro
adjustment (contain, shade, back off, pinch…) when a repeated tendency has no macro in the
defense 10 that answers it. Buttons come from the DB's cited Xbox controls; a button no source
confirms says VERIFY instead of guessing. Most snaps get no adjustment.
"""

from __future__ import annotations

import re
from typing import Any

from cfb_coach.madden import research_db as rdb

_CONTROL = {"hot_route": "hot_route", "audible": "audible", "pass_protection": "pass_protection",
            "motion": "motion", "fake_snap": "fake_snap", "flip_run": "flip_run"}


def _sources(ids: list[str]) -> list[str]:
    srcs = rdb.sources()
    return [srcs[i]["title"] for i in ids or [] if i in srcs]


def _run_audible(formation: str, audibles: dict[str, list[str]] | None) -> str | None:
    from cfb_coach.madden.catalog import is_run

    return next((p for p in (audibles or {}).get(formation) or [] if is_run(p)), None)


def offense_adjustment(
    *,
    play: str,
    formation: str,
    coverage_class: str | None,
    coverage_source: str,
    repeated: bool,
    audibles: dict[str, list[str]] | None = None,
) -> dict[str, Any] | None:
    """At most one offense adjustment for this snap, or None (most snaps).

    Needs a LIVE look or a look REPEATED this game (never one previous-snap tell). Pass plays
    get the hot route / protection the research names for that look; a pass vs a two-high look
    audibles to the formation's run audible when it has one."""
    from cfb_coach.madden.catalog import is_run

    look_ok = bool(coverage_class) and (coverage_source == "live" or repeated)
    if not look_ok or not play:
        return None
    rpo = bool(re.search(r"rpo|alert", play, re.I))
    passing = not is_run(play) and not rpo
    for a in rdb.offense_adjustments():
        if coverage_class not in (a.get("vs") or []):
            continue
        kind = a.get("type")
        if kind == "audible":
            run = _run_audible(formation, audibles) if passing else None
            if not run:
                continue
            label = f"Audible → {run}"
            btn = rdb.buttons("offense", "audible").replace("that audible", run)
        elif not passing:
            continue
        elif kind == "hot_route":
            label = f"Hot route {a.get('target')} → {a.get('route')}"
            btn = (rdb.buttons("offense", "hot_route")
                   .replace("the receiver's icon button", f"{a.get('target')}'s icon button")
                   .replace("pick the route", f"pick {a.get('route')}"))
        else:
            label = f"{a.get('route') or kind.replace('_', ' ').title()}"
            btn = rdb.buttons("offense", _CONTROL.get(kind, kind)).replace("pick the protection", f"pick {a.get('route')}")
        return {"id": a.get("id"), "side": "offense", "kind": kind, "label": label, "buttons": btn,
                "why": f"{coverage_source} {coverage_class.replace('_', ' ')} look — {a.get('why')}",
                "sources": _sources(a.get("sources") or [])}
    return None


def defense_adjustment(family: str | None, *, why: str = "") -> dict[str, Any] | None:
    """A non-macro defensive adjustment the research names for this concept family, or None."""
    if not family:
        return None
    for a in rdb.defense_adjustments():
        if family not in (a.get("answers") or []):
            continue
        btn = rdb.buttons("defense", a["control"])
        if a.get("extra_control"):
            btn += " ; then " + rdb.buttons("defense", a["extra_control"])
        return {"id": a.get("id"), "side": "defense", "kind": "adjustment", "label": a.get("label"),
                "buttons": btn, "why": (why + " — " if why else "") + str(a.get("why") or ""),
                "sources": _sources(a.get("sources") or [])}
    return None


def offense_plan() -> list[dict[str, Any]]:
    """Every researched offense adjustment with its buttons (the prep page's offense list)."""
    out = []
    for a in rdb.offense_adjustments():
        kind = a.get("type")
        label = (f"Hot route {a.get('target')} → {a.get('route')}" if kind == "hot_route"
                 else "Audible to the formation's run audible" if kind == "audible"
                 else str(a.get("route") or kind))
        out.append({"label": label, "vs": ", ".join(v.replace("_", " ") for v in a.get("vs") or []),
                    "buttons": rdb.buttons("offense", _CONTROL.get(kind, kind)), "why": a.get("why") or "",
                    "sources": _sources(a.get("sources") or [])})
    return out


def controls_table() -> list[dict[str, str]]:
    """Every researched Xbox control (both sides) — prep details page / `macro-settings` listing."""
    db = rdb.load()
    rows = []
    for side in ("offense", "defense"):
        for key, c in ((db.get("controls_xbox") or {}).get(side) or {}).items():
            rows.append({"side": side, "action": key.replace("_", " "), "buttons": c.get("buttons", ""),
                         "confidence": c.get("confidence", ""), "sources": ", ".join(_sources(c.get("sources") or []))})
    return rows


__all__ = ["controls_table", "defense_adjustment", "offense_adjustment", "offense_plan"]
