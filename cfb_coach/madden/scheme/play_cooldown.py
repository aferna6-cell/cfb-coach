"""Titans-only offensive play-family cooldown (owner t390u).

In ``--franchise titans`` sessions, once a play family is called it cannot be
recommended again for the next ``COOLDOWN_SNAPS`` (5) offensive snaps of that
session. The key is the catalog *base play* family, not the variant: calling
any Mesh variant blocks every Mesh variant (including Redzone Mesh). Lions
and Chiefs online sessions are untouched.

If every currently legal family is inside the window, fall back to the
least-recently-used legal family rather than returning no call.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from cfb_coach.madden.franchise import DEFAULT_TITANS_TEAM, TITANS
from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import evidence as ev
from cfb_coach.madden.scheme.legality import NO_LEGAL

COOLDOWN_SNAPS = 5

# Variant qualifiers that appear on a name after the in-game play (never part of the family).
# Do not strip semantic play-name words such as "Post" or "Shake".
_VARIANT_TAILS = (
    r"return\s+motion",
    r"corner\s*/?\s*flat",
    r"outside\s+bunch\s+wr\s+post",
)
_TAIL = re.compile(
    r"(?:[\s]*[-—–][\s]*|\s+)(?:" + "|".join(_VARIANT_TAILS) + r")\s*$",
    re.I,
)
_TRAILING_NUM = re.compile(r"\s+\d+\s*$")


def play_family_key(base_play: str | None) -> str:
    """Normalize a catalog base-play name into a cooldown family key.

    Strips trailing variant numbers and known variant suffixes. Leaves distinct
    play names intact (Mesh vs Mesh Post, Z Spot vs Z Spot Shake).
    """
    text = " ".join(str(base_play or "").strip().split())
    if not text:
        return ""
    text = _TAIL.sub("", text)
    text = _TRAILING_NUM.sub("", text)
    text = text.replace("—", " ").replace("–", " ").replace("-", " ").replace("_", " ")
    return " ".join(text.lower().split())


def family_of_variant(v: sc.SchemeVariant) -> str:
    return play_family_key(v.base_play)


def family_label(key: str, cat: sc.SchemeCatalog | None = None) -> str:
    """Human-readable family name (catalog base_play spelling when known)."""
    if cat is not None:
        for v in cat.variants:
            if family_of_variant(v) == key:
                return v.base_play
    return key


def catalog_families(cat: sc.SchemeCatalog) -> dict[str, list[str]]:
    """family key -> sorted unique catalog base_play spellings (usually one)."""
    out: dict[str, set[str]] = {}
    for v in cat.variants:
        out.setdefault(family_of_variant(v), set()).add(v.base_play)
    return {k: sorted(vs) for k, vs in out.items()}


def is_titans_session(db: Any, session_id: str | None, extras: Mapping[str, Any] | None = None) -> bool:
    """True only for a Titans franchise session. Never inferred from env alone."""
    extras = extras or {}
    if session_id and db is not None:
        row = _session_row(db, session_id)
        if row is not None:
            league = str(row.get("league") or "").strip().lower()
            dynasty = str(row.get("dynasty") or "").strip().lower()
            user = str(row.get("user_team") or "").strip()
            return league == TITANS or dynasty == TITANS or user == DEFAULT_TITANS_TEAM
    # Tests (and a session not yet written) may tag extras; a real session row wins above.
    tag = str(extras.get("league") or extras.get("franchise") or "").strip().lower()
    return tag == TITANS


def _conn(db: Any) -> Any:
    return getattr(db, "conn", db)


def _session_row(db: Any, session_id: str) -> dict[str, Any] | None:
    conn = _conn(db)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(game_sessions)").fetchall()}
        if "session_id" not in cols:
            return None
        row = conn.execute("SELECT * FROM game_sessions WHERE session_id = ?", (session_id,)).fetchone()
    except Exception:  # noqa: BLE001 — a missing table must not enable the cooldown
        return None
    if row is None:
        return None
    if isinstance(row, Mapping):
        return dict(row)
    keys = row.keys() if hasattr(row, "keys") else []
    return {k: row[k] for k in keys}


def _has_snap_col(conn: Any, col: str) -> bool:
    try:
        return col in {r[1] for r in conn.execute("PRAGMA table_info(snaps)").fetchall()}
    except Exception:  # noqa: BLE001
        return False


def family_from_logged_call(
    *,
    adjustment_id: str | None,
    play: str | None = None,
    formation: str | None = None,
    cat: sc.SchemeCatalog | None = None,
) -> str | None:
    """Recover the family of a logged offensive recommendation."""
    cat = cat or sc.load_cached()
    adj = str(adjustment_id or "")
    if adj.startswith(ev.ADJ_PREFIX):
        vid = adj[len(ev.ADJ_PREFIX):]
        if not vid or vid == NO_LEGAL:
            return None
        v = cat.get(vid)
        if v is not None:
            return family_of_variant(v)
    if play:
        key = play_family_key(play)
        # Prefer a catalog family when the logged play name maps onto one.
        if key and any(family_of_variant(v) == key for v in cat.variants):
            return key
        return key or None
    return None


def recent_offense_families(
    db: Any,
    session_id: str | None,
    cat: sc.SchemeCatalog | None = None,
    *,
    limit: int | None = None,
) -> list[str]:
    """Family keys of this session's logged offensive calls, newest first.

    Reads closed snaps (``recommended_adjustment_id``) so a restart of the same
    session keeps the cooldown history.
    """
    if not session_id or db is None:
        return []
    conn = _conn(db)
    try:
        if not ev._has_table(conn, "snaps"):
            return []
        cols = {r[1] for r in conn.execute("PRAGMA table_info(snaps)").fetchall()}
        if "recommended_adjustment_id" not in cols:
            return []
        fields = ["recommended_adjustment_id"]
        for c in ("play", "formation", "our_call", "side"):
            if c in cols:
                fields.append(c)
        where = "session_id = ?"
        args: list[Any] = [session_id]
        if "side" in cols:
            where += " AND (side IS NULL OR side = '' OR side LIKE 'o%')"
        sql = f"SELECT {', '.join(fields)} FROM snaps WHERE {where} ORDER BY id DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        rows = conn.execute(sql, args).fetchall()
    except Exception:  # noqa: BLE001
        return []
    cat = cat or sc.load_cached()
    out: list[str] = []
    for row in rows:
        rec = row if isinstance(row, Mapping) else {k: row[k] if k in row.keys() else None for k in fields}
        # sqlite Row supports keys()
        if not isinstance(row, Mapping) and hasattr(row, "keys"):
            rec = {k: row[k] for k in row.keys()}
        fam = family_from_logged_call(
            adjustment_id=rec.get("recommended_adjustment_id"),
            play=rec.get("play") or rec.get("our_call"),
            formation=rec.get("formation"),
            cat=cat,
        )
        if fam:
            out.append(fam)
    return out


def apply_play_cooldown(
    legal: Sequence[sc.SchemeVariant],
    recent_families: Sequence[str],
    *,
    window: int = COOLDOWN_SNAPS,
    cat: sc.SchemeCatalog | None = None,
) -> tuple[list[sc.SchemeVariant], dict[str, Any]]:
    """Filter legal variants by the Titans play-family cooldown.

    If the filter would empty the set, return the least-recently-used legal
    family instead and mark ``fallback`` on the info dict.
    """
    legal = list(legal)
    blocked = [f for f in list(recent_families)[:window] if f]
    blocked_set = set(blocked)
    info: dict[str, Any] = {
        "active": True,
        "window": window,
        "blocked_families": [family_label(f, cat) for f in dict.fromkeys(blocked)],
        "fallback": False,
    }
    if not legal:
        return legal, info
    kept = [v for v in legal if family_of_variant(v) not in blocked_set]
    if kept:
        info["candidates"] = [v.variant_id for v in kept]
        return kept, info
    # Every currently legal family is inside the window: LRU fallback.
    last_age: dict[str, int] = {}
    for i, fam in enumerate(recent_families):
        if fam and fam not in last_age:
            last_age[fam] = i
    legal_fams = {family_of_variant(v) for v in legal}

    def _lru_key(fam: str) -> tuple[int, str]:
        # Larger age = older last use; never-seen families (shouldn't happen here) win.
        return (last_age.get(fam, 10**9), fam)

    lru = max(legal_fams, key=_lru_key)
    kept = [v for v in legal if family_of_variant(v) == lru]
    label = family_label(lru, cat)
    info.update({
        "fallback": True,
        "fallback_family": label,
        "fallback_rule": (
            f"every legal play is inside the {window}-snap Titans cooldown; "
            f"used least-recently-used family {label}"
        ),
        "candidates": [v.variant_id for v in kept],
    })
    return kept, info


def cooldown_explanation(info: Mapping[str, Any] | None) -> str | None:
    if not info or not info.get("fallback"):
        return None
    return "PLAY COOLDOWN FALLBACK: " + str(info.get("fallback_rule") or "least-recently-used legal play")
