"""Authoritative complete Madden offensive inventory and call-coverage reporting.

A model-designed book is not a handful of 'core' plays or the heuristic
situational menu. Every formation/play pair in the confirmed, applied DB
playbook is an ML candidate (subject only to football situational eligibility).
This module never silently installs a draft or infers execution from a call.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from cfb_coach.madden import playbook


def pairs_in_inventory(formations: Mapping[str, list[str]]) -> list[tuple[str, str]]:
    """Preserve every confirmed play exactly once, in formation order."""
    return list(dict.fromkeys(
        (str(form), str(play))
        for form, plays in formations.items()
        for play in plays
        if form and play
    ))


def inventory_fingerprint(
    formations: Mapping[str, list[str]],
    sources: Mapping[str, str] | None = None,
) -> str:
    """Identifies the exact playbook revision/content, independent of game."""
    payload = {
        "formations": {
            str(f): list(dict.fromkeys(ps)) for f, ps in sorted(formations.items())
        },
        "source_books": dict(sorted((sources or {}).items())),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def installed_inventory(db: Any) -> dict[str, Any]:
    """Read only the *confirmed applied* offensive record, never staged design."""
    rec = playbook.load_books(db).get("offense") or {}
    formations = rec.get("formations") or {}
    sources = rec.get("formation_sources") or {}
    pairs = pairs_in_inventory(formations)
    return {
        "installed": bool(rec),
        "name": rec.get("name") or "No confirmed offensive playbook",
        "revision": rec.get("rev"),
        "formation_sources": dict(sources),
        "formations": {f: list(dict.fromkeys(ps)) for f, ps in formations.items()},
        "formation_count": len(formations),
        "play_count": len(pairs),
        "inventory_id": inventory_fingerprint(formations, sources),
        "notes": (
            "Only confirmed applied formation/plays can be called. Proposed or "
            "staged designs do not enter live ML."
        ),
    }


def inventory_report(
    db: Any, *, opponent_id: str | None = None, game_id: str | None = None,
) -> dict[str, Any]:
    """Report **recommended** calls for every play, including never-called plays.

    Counts are recommendations, not verification of executed plays. Historical
    recommendations from a prior inventory are omitted if that pair is not
    installed now. Optionally scope to one opponent or game/session.
    """
    inventory = installed_inventory(db)
    forms = inventory["formations"]
    counts: dict[tuple[str, str], int] = {}
    clauses = ["side = 'offense'"]
    params: list[Any] = []
    if game_id:
        clauses.append("session_id = ?")
        params.append(game_id)
    elif opponent_id:
        clauses.append("opponent_id = ?")
        params.append(opponent_id)
    if inventory["installed"]:
        query = (
            "SELECT formation, play, COUNT(*) AS n FROM snaps WHERE "
            + " AND ".join(clauses)
            + " GROUP BY formation, play"
        )
        try:
            for row in db.conn.execute(query, params):
                counts[(str(row["formation"]), str(row["play"]))] = int(row["n"])
        except Exception:  # noqa: BLE001 — old/read-only DBs may lack snaps
            pass

    details = []
    used_count = 0
    for form, plays in forms.items():
        for play in plays:
            calls = counts.get((form, play), 0)
            if calls:
                used_count += 1
            details.append({
                "formation": form, "play": play,
                "source_book": inventory["formation_sources"].get(form),
                "recommended_calls": calls,
                "called_at_least_once": bool(calls),
            })
    return {
        **inventory,
        "scope": {
            "game_id": game_id, "opponent_id": opponent_id if not game_id else None,
        },
        "used_plays": used_count,
        "unused_plays": len(details) - used_count,
        "coverage_pct": round(100 * used_count / len(details), 1) if details else 0.0,
        "play_usage": details,
        "counting_note": (
            "Recommended calls from snap logs. These are not proof of execution. "
            "Zero recommendations does not imply a play was unavailable to the model."
        ),
    }
