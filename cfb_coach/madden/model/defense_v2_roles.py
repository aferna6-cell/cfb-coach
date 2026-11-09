"""First-party verified Madden 27 defensive play assignments.

External/private JSON annotations only; no generated defender jobs.
The structural model uses sourced coverage-family features even if all eleven
assignment slots remain UNKNOWN; this verifier closes the gap progressively.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "madden.defense.v2.verified_roles.v1"
REQUIRED = frozenset({"formation", "play", "verified_in_madden", "source", "slots"})


def verify_document(data: Mapping[str, Any]) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("unrecognized defensive role annotation schema")
    approved: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
    for raw in data.get("plays") or []:
        if not isinstance(raw, dict) or not REQUIRED <= set(raw):
            raise ValueError("role entry missing required fields")
        pair = (raw["formation"], raw["play"])
        if not all(isinstance(x, str) and x.strip() for x in pair):
            raise ValueError("formation and play must be exact nonempty strings")
        if pair in approved:
            raise ValueError("duplicate defensive play role entry")
        if raw["verified_in_madden"] is not True:
            raise ValueError("unverified defender roles must not be represented as verified")
        if raw["source"] != "owner_xbox_play_art" or not str(raw.get("verified_on") or "").strip():
            raise ValueError("explicit owner in-game source and verification date required")
        slots = raw["slots"]
        if not isinstance(slots, list) or len(slots) != 11:
            raise ValueError("an exact eleven-defender ledger is required")
        seen = set()
        cleaned = []
        for slot in slots:
            if not isinstance(slot, dict):
                raise ValueError("every defender must be a verified record")
            sid = slot.get("slot_id")
            responsibility = slot.get("responsibility")
            role = slot.get("role")
            if not (isinstance(sid, str) and sid.strip() and
                    isinstance(responsibility, str) and responsibility.strip() and
                    isinstance(role, str) and role.strip()):
                raise ValueError("each defender needs a unique slot, role and explicit assignment")
            if sid in seen:
                raise ValueError("defender roles cannot contain duplicate slot IDs")
            seen.add(sid)
            if slot.get("verified") is not True:
                raise ValueError("every defender assignment must be first-party verified")
            cleaned.append({"slot_id": sid, "role": role,
                            "responsibility": responsibility, "verified": True})
        approved[pair] = tuple(cleaned)
    return approved


@lru_cache(maxsize=8)
def _load(path: str, mtime_ns: int, size: int) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return verify_document(raw)


def verified_roles(file: str | Path | None, formation: str, play: str) -> dict[str, Any]:
    if not file:
        return {"verification": "UNKNOWN", "verified_defenders": 0,
                "slots": [], "missing_roles": 11}
    path = Path(file).expanduser().resolve()
    stat = path.stat()
    approved = _load(str(path), stat.st_mtime_ns, stat.st_size)
    slots = approved.get((formation, play))
    if slots is None:
        return {"verification": "UNKNOWN", "verified_defenders": 0,
                "slots": [], "missing_roles": 11}
    return {"verification": "OWNER_XBOX_VERIFIED", "verified_defenders": 11,
            "slots": list(slots), "missing_roles": 0}
