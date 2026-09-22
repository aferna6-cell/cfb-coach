"""Dynasty modes: alabama (serious) vs ohio_state (experimental)."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

ALABAMA = "alabama"
OHIO_STATE = "ohio_state"
DEFAULT_DYNASTY = ALABAMA
VALID_DYNASTIES = frozenset({ALABAMA, OHIO_STATE})

META_KEY = "dynasty_mode"


@lru_cache(maxsize=1)
def _exact() -> dict[str, Any]:
    try:
        ref = resources.files("cfb_coach").joinpath("data/aidan_exact_macros.json")
        with resources.as_file(ref) as p:
            return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        p = Path(__file__).resolve().parent / "data" / "aidan_exact_macros.json"
        return json.loads(p.read_text(encoding="utf-8"))


def normalize_dynasty(raw: str | None) -> str:
    s = (raw or DEFAULT_DYNASTY).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "alabama": ALABAMA,
        "bama": ALABAMA,
        "ua": ALABAMA,
        "serious": ALABAMA,
        "ohio_state": OHIO_STATE,
        "ohiostate": OHIO_STATE,
        "osu": OHIO_STATE,
        "ohio": OHIO_STATE,
        "experimental": OHIO_STATE,
    }
    out = aliases.get(s, s)
    if out not in VALID_DYNASTIES:
        raise ValueError(
            f"Unknown dynasty {raw!r}. Use: alabama | ohio_state (default alabama)."
        )
    return out


def dynasty_config(mode: str | None = None) -> dict[str, Any]:
    mid = normalize_dynasty(mode)
    block = (_exact().get("dynasties") or {}).get(mid) or {}
    experimental = bool(block.get("allow_experimental_active", mid == OHIO_STATE))
    return {
        "id": mid,
        "label": block.get("label") or ("Alabama" if mid == ALABAMA else "Ohio State"),
        "mode": block.get("mode") or ("serious" if mid == ALABAMA else "experimental"),
        "description": block.get("description") or "",
        "default_active": list(
            block.get("default_active") or _exact().get("active_8") or []
        ),
        "default_benched": list(
            block.get("default_benched") or _exact().get("benched") or []
        ),
        "allow_experimental_active": experimental,
        "experimental_badge": experimental and mid == OHIO_STATE,
    }


def allow_experimental(mode: str | None) -> bool:
    return bool(dynasty_config(mode)["allow_experimental_active"])


def get_session_dynasty(db: Any) -> str:
    if db is None:
        return DEFAULT_DYNASTY
    try:
        raw = db.get_meta(META_KEY)
    except Exception:
        return DEFAULT_DYNASTY
    if not raw:
        return DEFAULT_DYNASTY
    try:
        return normalize_dynasty(raw)
    except ValueError:
        return DEFAULT_DYNASTY


def set_session_dynasty(db: Any, mode: str) -> str:
    mid = normalize_dynasty(mode)
    if db is not None:
        db.set_meta(META_KEY, mid)
        db.set_meta("dynasty_config_json", json.dumps(dynasty_config(mid)))
    return mid


def doctrine_line() -> str:
    return (
        "Doctrine: do NOT auto-use macros from one concept appearance — "
        "most snaps Cover 3 Sky / Quarters / Tampa 2 with no macro."
    )
