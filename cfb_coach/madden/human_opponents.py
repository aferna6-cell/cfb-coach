"""Madden-only cold-start human opponents.

John is a playoff user with no film. He is not a CFB persona and he is not
an alias of Gavin, Tiano, or anyone else. The profile starts neutral. Live
snaps update him through the existing tendency bumps. Nothing here is copied
from another opponent row.
"""

from __future__ import annotations

import json
import re
from typing import Any

from cfb_coach.opponents import resolve_opponent

_ID = re.compile(r"[a-z][a-z0-9_]{1,31}")


def _norm(raw: str) -> str:
    return " ".join(str(raw).strip().lower().replace("_", " ").replace("-", " ").split())


def cold_start_profile(oid: str, *, display_name: str | None = None) -> dict[str, Any]:
    """Neutral human profile. No coverages, concepts, snaps, or team."""
    key = _norm(oid).replace(" ", "_")
    name = (display_name or "").strip() or key[:1].upper() + key[1:]
    return {
        "id": key,
        "display_name": name,
        "team_now": None,
        "nfl_team": None,
        "skill": "unknown",
        "archetype": "unknown",
        "persona_confidence": "none",
        "confidence": "none",
        "traits": {},
        "offense": {},
        "defense_vs_us": {
            "archetype": "unknown",
            "coverages": [],
            "man_blitz": "unknown",
        },
        "sample_snaps": [],
        "madden_only": True,
        "cold_start": True,
        "history": "none",
        "persona_source": "madden cold-start human (no borrowed priors)",
    }


# Built in. Resolved by exact id only. Not written into the shared seed.
BUILTIN: dict[str, dict[str, Any]] = {
    "john": cold_start_profile("john", display_name="John"),
}


def builtin_profiles() -> list[tuple[str, dict[str, Any]]]:
    return [(oid, dict(profile)) for oid, profile in BUILTIN.items()]


def is_builtin_cold_start(raw: str) -> bool:
    return _norm(raw).replace(" ", "_") in BUILTIN


def is_cold_start_profile(profile: dict[str, Any] | None) -> bool:
    if not profile:
        return False
    return bool(profile.get("madden_only") and profile.get("cold_start"))


def is_cold_start_opponent(opponent_id: str, db: Any = None) -> bool:
    """True for John and for later Madden-only humans stored on this DB."""
    key = _norm(opponent_id).replace(" ", "_")
    if key in BUILTIN:
        return True
    if db is None:
        return False
    try:
        return is_cold_start_profile(db.get_opponent(key))
    except Exception:  # noqa: BLE001 — a bad row is not a cold start
        return False


def resolve_madden_opponent(raw: str, db: Any = None) -> str | None:
    """Shared personas first, except an exact cold-start id.

    ``john`` is never a prefix of another persona and never falls through to
    one. ``jo`` does not resolve to John. A shared alias such as ``auburn``
    still resolves to that persona.
    """
    key = _norm(raw).replace(" ", "_")
    if key in BUILTIN:
        return key
    shared = resolve_opponent(raw)
    if shared:
        return shared
    if db is not None and _stored_cold_start(db, key):
        return key
    return None


def _stored_cold_start(db: Any, oid: str) -> bool:
    try:
        return is_cold_start_profile(db.get_opponent(oid))
    except Exception:  # noqa: BLE001
        return False


def _insert_profile(db: Any, profile: dict[str, Any]) -> None:
    """Insert one opponent row. Does not write tendencies or weights."""
    oid = str(profile["id"])
    db.conn.execute(
        """
        INSERT INTO opponents
            (id, display_name, team_now, skill, confidence, profile_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            oid,
            profile.get("display_name") or oid,
            profile.get("team_now"),
            profile.get("skill"),
            profile.get("confidence"),
            json.dumps(profile),
        ),
    )
    db.conn.commit()


def register_cold_start(
    db: Any,
    raw: str,
    *,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Register a Madden-only human. Refuses a shared persona id or alias.

    An existing cold-start row is returned unchanged. This does not copy
    another opponent's profile, tendencies, or weights.
    """
    key = _norm(raw).replace(" ", "_")
    if not _ID.fullmatch(key):
        raise ValueError(f"opponent id must be a short lowercase token, got {raw!r}")
    if key not in BUILTIN:
        shared = resolve_opponent(raw)
        if shared:
            raise ValueError(f"{raw!r} already resolves to shared persona {shared}")
    existing = db.get_opponent(key) if db is not None else None
    if existing:
        if is_cold_start_profile(existing):
            return existing
        raise ValueError(f"{key} exists and is not a Madden cold-start profile")
    if key in BUILTIN:
        profile = cold_start_profile(key, display_name=BUILTIN[key]["display_name"])
    else:
        profile = cold_start_profile(key, display_name=display_name)
    _insert_profile(db, profile)
    return profile


def ensure_madden_opponent(db: Any, oid: str) -> dict[str, Any] | None:
    """Insert John (or another registered cold start) if the row is missing.

    Shared personas are left alone. A second call does not rewrite the row
    and does not add seed tendencies.
    """
    if db is None or not oid:
        return None
    key = _norm(oid).replace(" ", "_")
    if key in BUILTIN or _stored_cold_start(db, key):
        return register_cold_start(db, key)
    return None
