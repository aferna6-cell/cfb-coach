"""Play-family cooldown for Titans, Chiefs, and Giants sessions (owner t390u / t403u / t456u).

In ``--franchise titans`` sessions, once a play family is called it cannot be
recommended again for the next ``TITANS_COOLDOWN_SNAPS`` (5) *offensive* snaps
of that session. Titans defense is not cooled down.

In ``--franchise online`` (Chiefs) sessions, the same family-key rule applies
on *both* offense and defense, with ``CHIEFS_COOLDOWN_SNAPS`` (3). Offense and
defense histories are tracked separately.

In ``--franchise giants`` sessions, the same family-key rule applies on
*offense only*, with ``GIANTS_COOLDOWN_SNAPS`` (3). Giants defense is not
cooled down.

The key is the catalog *base play* family, not the variant: calling any Mesh
variant blocks every Mesh variant (including Redzone Mesh); on defense, a
base play and its Zone It (or other package) siblings share one family.
Lions sessions are untouched.

If every currently legal family is inside the window, fall back to the
least-recently-used legal family rather than returning no call.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from cfb_coach.madden.franchise import (
    DEFAULT_GIANTS_TEAM,
    DEFAULT_ONLINE_TEAM,
    DEFAULT_TITANS_TEAM,
    GIANTS,
    ONLINE,
    TITANS,
)
from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import evidence as ev
from cfb_coach.madden.scheme.legality import NO_LEGAL

COOLDOWN_SNAPS = 5  # Titans default; kept as the apply_play_cooldown window default
TITANS_COOLDOWN_SNAPS = 5
CHIEFS_COOLDOWN_SNAPS = 3
GIANTS_COOLDOWN_SNAPS = 3

_LEAGUE_LABEL = {TITANS: "Titans", ONLINE: "Chiefs", GIANTS: "Giants"}

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


def defense_family_key(base_play: str | None) -> str:
    """Normalize a defensive base-play name into a cooldown family key.

    Unlike ``play_family_key``, this does not strip trailing numbers: names such
    as ``DB Fire 2`` and ``Cover 9 Show 2`` are the in-game plays, not variants.
    Zone It / package siblings still share a family because they share base_play.
    """
    text = " ".join(str(base_play or "").strip().split())
    if not text:
        return ""
    text = text.replace("—", " ").replace("–", " ").replace("-", " ").replace("_", " ")
    return " ".join(text.lower().split())


def family_of_defense_candidate(c: Any) -> str:
    """Family key of a Ravens defensive candidate (catalog base_play)."""
    return defense_family_key(getattr(c, "base_play", None))


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


def _franchise_from_fields(league: Any, dynasty: Any, user: Any) -> str | None:
    league_s = str(league or "").strip().lower()
    dynasty_s = str(dynasty or "").strip().lower()
    user_s = str(user or "").strip()
    if league_s == TITANS or dynasty_s == TITANS or user_s == DEFAULT_TITANS_TEAM:
        return TITANS
    if league_s == GIANTS or dynasty_s == GIANTS or user_s == DEFAULT_GIANTS_TEAM:
        return GIANTS
    if league_s == ONLINE or dynasty_s == ONLINE or user_s == DEFAULT_ONLINE_TEAM:
        return ONLINE
    return None


def session_franchise(db: Any, session_id: str | None, extras: Mapping[str, Any] | None = None) -> str | None:
    """``titans`` / ``giants`` / ``online`` / None. A real session row always wins over extras."""
    extras = extras or {}
    if session_id and db is not None:
        row = _session_row(db, session_id)
        if row is not None:
            return _franchise_from_fields(row.get("league"), row.get("dynasty"), row.get("user_team"))
    return _franchise_from_fields(
        extras.get("league") or extras.get("franchise"),
        extras.get("dynasty"),
        extras.get("user_team") or extras.get("user"),
    )


def is_titans_session(db: Any, session_id: str | None, extras: Mapping[str, Any] | None = None) -> bool:
    """True only for a Titans franchise session. Never inferred from env alone."""
    return session_franchise(db, session_id, extras) == TITANS


def is_online_session(db: Any, session_id: str | None, extras: Mapping[str, Any] | None = None) -> bool:
    """True only for a Chiefs online-league session. Never inferred from env alone."""
    return session_franchise(db, session_id, extras) == ONLINE


def is_giants_session(db: Any, session_id: str | None, extras: Mapping[str, Any] | None = None) -> bool:
    """True only for a Giants franchise session. Never inferred from env alone."""
    return session_franchise(db, session_id, extras) == GIANTS


def cooldown_spec(
    db: Any,
    session_id: str | None,
    extras: Mapping[str, Any] | None = None,
    *,
    side: str = "offense",
) -> dict[str, Any] | None:
    """Return ``{window, league, league_key, side}`` if a play-family cooldown applies.

    Titans: offense only, window 5. Giants: offense only, window 3.
    Chiefs online: offense and defense, window 3.
    Lions (and any other profile): no cooldown.
    """
    key = session_franchise(db, session_id, extras)
    side_name = "defense" if str(side or "").lower().startswith("d") else "offense"
    if key == TITANS and side_name == "offense":
        return {"window": TITANS_COOLDOWN_SNAPS, "league": _LEAGUE_LABEL[TITANS],
                "league_key": TITANS, "side": side_name}
    if key == GIANTS and side_name == "offense":
        return {"window": GIANTS_COOLDOWN_SNAPS, "league": _LEAGUE_LABEL[GIANTS],
                "league_key": GIANTS, "side": side_name}
    if key == ONLINE:
        return {"window": CHIEFS_COOLDOWN_SNAPS, "league": _LEAGUE_LABEL[ONLINE],
                "league_key": ONLINE, "side": side_name}
    return None


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


def recent_defense_families(
    db: Any,
    session_id: str | None,
    *,
    limit: int | None = None,
) -> list[str]:
    """Family keys of this session's logged Ravens defensive calls, newest first.

    Reads ``ravens_defense_decisions`` (written when the call is recommended) so a
    restart of the same session keeps the cooldown history.
    """
    if not session_id or db is None:
        return []
    conn = _conn(db)
    try:
        if not ev._has_table(conn, "ravens_defense_decisions"):
            return []
        sql = ("SELECT base_play, call_id FROM ravens_defense_decisions "
               "WHERE session_id = ? AND status = 'CALL' ORDER BY id DESC")
        args: list[Any] = [session_id]
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        rows = conn.execute(sql, args).fetchall()
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            rec = dict(row)
        elif hasattr(row, "keys"):
            rec = {k: row[k] for k in row.keys()}
        else:
            rec = {"base_play": row[0], "call_id": row[1]}
        fam = defense_family_key(rec.get("base_play")) or defense_family_key(rec.get("call_id"))
        if fam:
            out.append(fam)
    return out


def apply_play_cooldown(
    legal: Sequence[Any],
    recent_families: Sequence[str],
    *,
    window: int = COOLDOWN_SNAPS,
    cat: sc.SchemeCatalog | None = None,
    family_of: Callable[[Any], str] | None = None,
    label_of: Callable[[str], str] | None = None,
    league: str = "Titans",
) -> tuple[list[Any], dict[str, Any]]:
    """Filter legal candidates by the play-family cooldown.

    If the filter would empty the set, return the least-recently-used legal
    family instead and mark ``fallback`` on the info dict.
    """
    fam_of = family_of or family_of_variant

    def _label(key: str) -> str:
        if label_of is not None:
            return label_of(key)
        return family_label(key, cat)

    legal = list(legal)
    blocked = [f for f in list(recent_families)[:window] if f]
    blocked_set = set(blocked)
    info: dict[str, Any] = {
        "active": True,
        "window": window,
        "league": league,
        "blocked_families": [_label(f) for f in dict.fromkeys(blocked)],
        "fallback": False,
    }
    if not legal:
        return legal, info
    kept = [v for v in legal if fam_of(v) not in blocked_set]
    if kept:
        info["candidates"] = [_candidate_id(v) for v in kept]
        return kept, info
    # Every currently legal family is inside the window: LRU fallback.
    last_age: dict[str, int] = {}
    for i, fam in enumerate(recent_families):
        if fam and fam not in last_age:
            last_age[fam] = i
    legal_fams = {fam_of(v) for v in legal}

    def _lru_key(fam: str) -> tuple[int, str]:
        # Larger age = older last use; never-seen families (shouldn't happen here) win.
        return (last_age.get(fam, 10**9), fam)

    lru = max(legal_fams, key=_lru_key)
    kept = [v for v in legal if fam_of(v) == lru]
    label = _label(lru)
    info.update({
        "fallback": True,
        "fallback_family": label,
        "fallback_rule": (
            f"every legal play is inside the {window}-snap {league} cooldown; "
            f"used least-recently-used family {label}"
        ),
        "candidates": [_candidate_id(v) for v in kept],
    })
    return kept, info


def _candidate_id(v: Any) -> str:
    vid = getattr(v, "variant_id", None)
    if vid:
        return str(vid)
    cid = getattr(v, "candidate_id", None)
    if cid:
        return str(cid)
    return play_family_key(getattr(v, "base_play", None)) or ""


def cooldown_explanation(info: Mapping[str, Any] | None) -> str | None:
    if not info or not info.get("fallback"):
        return None
    return "PLAY COOLDOWN FALLBACK: " + str(info.get("fallback_rule") or "least-recently-used legal play")
