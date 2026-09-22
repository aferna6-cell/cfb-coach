"""Dynasty modes: alabama (serious USER) vs ohio_state (experimental lab).

ohio_state = practice / freer strategies.
alabama = serious online dynasty.
When an experimental strategy works in ohio_state postgame, promote it into
Alabama recommendations (macro loadout change or gameplan overlay).
"""

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


PROMOTIONS_META_KEY = "alabama_promotions_json"

# Strong positive weight delta from ohio_state postgame → consider Alabama promote
PROMOTE_WEIGHT_THRESHOLD = 0.30


def list_alabama_promotions(db: Any) -> list[dict[str, Any]]:
    """Pending / accepted promotions from ohio_state lab → Alabama recommendations."""
    if db is None:
        return []
    raw = db.get_meta(PROMOTIONS_META_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    return list(data) if isinstance(data, list) else []


def save_alabama_promotions(db: Any, promotions: list[dict[str, Any]]) -> None:
    if db is None:
        return
    db.set_meta(PROMOTIONS_META_KEY, json.dumps(promotions))


def record_promotions(db: Any, new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge new promotion candidates (dedupe by kind+target). Returns full list."""
    if not new_items:
        return list_alabama_promotions(db)
    existing = list_alabama_promotions(db)
    keys = {
        (p.get("kind"), p.get("target") or p.get("macro") or p.get("key"))
        for p in existing
    }
    for item in new_items:
        k = (item.get("kind"), item.get("target") or item.get("macro") or item.get("key"))
        if k in keys:
            # refresh delta / note
            for i, old in enumerate(existing):
                ok = (old.get("kind"), old.get("target") or old.get("macro") or old.get("key"))
                if ok == k:
                    existing[i] = {**old, **item}
                    break
        else:
            existing.append(item)
            keys.add(k)
    save_alabama_promotions(db, existing)
    return existing


def promote(
    db: Any,
    *,
    kind: str | None = None,
    target: str | None = None,
    accept_all: bool = False,
) -> dict[str, Any]:
    """Mark pending promotion(s) accepted for Alabama recommendations.

    Returns summary with accepted + remaining pending.
    """
    items = list_alabama_promotions(db)
    accepted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for p in items:
        match = False
        if accept_all:
            match = p.get("status") != "accepted"
        elif kind and target:
            t = p.get("target") or p.get("macro") or p.get("key")
            match = p.get("kind") == kind and t == target and p.get("status") != "accepted"
        elif target:
            t = p.get("target") or p.get("macro") or p.get("key")
            match = t == target and p.get("status") != "accepted"
        if match:
            p = dict(p)
            p["status"] = "accepted"
            accepted.append(p)
            remaining.append(p)
        else:
            remaining.append(p)
    save_alabama_promotions(db, remaining)
    return {"accepted": accepted, "pending": [p for p in remaining if p.get("status") != "accepted"], "all": remaining}


def format_promotions(promotions: list[dict[str, Any]] | None = None, db: Any = None) -> str:
    items = promotions if promotions is not None else list_alabama_promotions(db)
    if not items:
        return "No Alabama promotions pending (ohio_state lab successes promote here)."
    lines = ["## Alabama promotions (from ohio_state lab)"]
    for p in items:
        st = p.get("status") or "pending"
        note = p.get("note") or ""
        tgt = p.get("target") or p.get("macro") or p.get("key") or "?"
        lines.append(f"  [{st}] {p.get('kind', '?')}: {tgt} — {note}")
    lines.append("  Use: cfb_coach promote --accept-all   (or promote --target MACRO)")
    return "\n".join(lines)



def doctrine_line() -> str:
    return (
        "Doctrine: do NOT auto-use macros from one concept appearance — "
        "most snaps Cover 3 Sky / Quarters / Tampa 2 with no macro."
    )
