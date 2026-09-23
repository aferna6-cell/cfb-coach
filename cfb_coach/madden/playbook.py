"""Madden playbook of record — one active book per side, locked by the latest prep.

Contract (owner review on PR #2):
  - Every prep chooses a mode per side: STOCK (an existing in-game book by exact
    name) or CUSTOM (a book Aidan builds).
  - First custom / switch-to-custom → full formation checklist to install.
  - Successive preps on a custom book → only ADD / REMOVE of entire formations.
  - Switch any time (custom ↔ stock, stock → other stock) via --o-book / --d-book.
  - Live `play` may only call formation+play pairs inside the locked (applied) book;
    with no locked book it refuses to call (run prep first).
  - Stock picks need no building → locked (applied) at prep. Custom builds/diffs are
    PENDING until `prep --mark-applied` confirms Aidan installed them.

Persisted in the Madden DB meta key `active_playbook_json`:
  {"applied": {side: record}, "pending": {side: record}}
  record = {side, mode, name, formations: {formation: [plays]}, core, rev, locked_ts, reason}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from cfb_coach.madden.data import _load_json

META_KEY = "active_playbook_json"
SIDES = ("offense", "defense")
STOCK, CUSTOM = "stock", "custom"
DEFAULT_STOCK = {"offense": "Buccaneers", "defense": "49ers"}
CUSTOM_NAME = {"offense": "PRIMARY META O (custom)", "defense": "PRIMARY META D (custom)"}

_BOOK_ALIASES = {
    "bucs": "Buccaneers", "buccaneers": "Buccaneers", "tb": "Buccaneers", "tampa": "Buccaneers",
    "tampa bay": "Buccaneers", "tampa bay buccaneers": "Buccaneers",
    "titans": "Titans", "tennessee": "Titans", "ten": "Titans", "tennessee titans": "Titans",
    "shotgun": "Shotgun Classic", "shotgun classic": "Shotgun Classic", "classic": "Shotgun Classic",
    "49ers": "49ers", "niners": "49ers", "sf": "49ers", "saleh": "49ers",
    "san francisco 49ers": "49ers",
}


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------

def formation_catalog(side: str) -> dict[str, list[str]]:
    """Every formation we can install, with its verified callable plays."""
    seed = _load_json("seed.json")
    out: dict[str, list[str]] = {}
    if side == "offense":
        for name, meta in seed["playbooks"]["offense_formations"].items():
            out[name] = list(meta.get("verified") or meta.get("core") or [])
        for name, meta in (seed.get("delta_formations") or {}).items():
            out[name] = list(meta.get("verified") or meta.get("core") or [])
    else:
        for name, meta in seed["playbooks"]["defense_packages"].items():
            out[name] = list(meta.get("calls") or [])
    return out


def formation_books(side: str, formation: str) -> str:
    seed = _load_json("seed.json")
    pool = (
        {**seed["playbooks"]["offense_formations"], **(seed.get("delta_formations") or {})}
        if side == "offense"
        else seed["playbooks"]["defense_packages"]
    )
    meta = pool.get(formation) or {}
    books = meta.get("books")
    return ", ".join(books) if isinstance(books, list) else str(meta.get("book") or "")


def stock_books(side: str) -> dict[str, dict[str, Any]]:
    return dict((_load_json("seed.json").get("stock_books") or {}).get(side) or {})


def resolve_stock_book(side: str, raw: str) -> str:
    key = " ".join((raw or "").strip().lower().split())
    name = _BOOK_ALIASES.get(key) or next((b for b in stock_books(side) if b.lower() == key), None)
    if not name or name not in stock_books(side):
        raise ValueError(
            f"Unknown {side} stock book {raw!r}. Known: {', '.join(stock_books(side))}"
        )
    return name


def parse_choice(side: str, raw: str | None) -> tuple[str, str | None]:
    """--o-book / --d-book value → ("auto"|"custom"|"stock", book name|None)."""
    val = (raw or "auto").strip()
    low = val.lower()
    if low in ("", "auto"):
        return "auto", None
    if low == "custom":
        return CUSTOM, None
    if low == "stock":
        return STOCK, DEFAULT_STOCK[side]
    if low.startswith("stock:"):
        return STOCK, resolve_stock_book(side, val.split(":", 1)[1])
    return STOCK, resolve_stock_book(side, val)  # bare book name


def _stock_formations(side: str, name: str) -> dict[str, list[str]]:
    cat = formation_catalog(side)
    return {f: cat[f] for f in stock_books(side)[name]["formations"] if f in cat}


def _custom_formations(side: str, names: list[str]) -> dict[str, list[str]]:
    cat = formation_catalog(side)
    return {f: cat[f] for f in names if f in cat}


def custom_core(side: str) -> list[str]:
    return list((_load_json("seed.json").get("custom_core") or {}).get(side) or [])


def desired_custom(side: str, opp: dict[str, Any]) -> list[str]:
    """Core formations + persona formations (offense only)."""
    names = custom_core(side)
    if side == "offense":
        extras = (_load_json("seed.json").get("persona_custom_formations") or {}).get(
            (opp.get("archetype") or "").lower()
        ) or []
        names += [f for f in extras if f not in names]
    return names


# ---------------------------------------------------------------------------
# Record persistence
# ---------------------------------------------------------------------------

def make_record(side: str, mode: str, name: str | None, *, rev: int = 1, reason: str = "",
                formations: list[str] | None = None) -> dict[str, Any]:
    if mode == STOCK:
        book = name or DEFAULT_STOCK[side]
        forms = _stock_formations(side, book)
    else:
        book = CUSTOM_NAME[side]
        forms = _custom_formations(side, formations or custom_core(side))
    return {
        "side": side,
        "mode": mode,
        "name": book,
        "formations": forms,
        "core": custom_core(side) if mode == CUSTOM else list(forms),
        "rev": rev,
        "locked_ts": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }


class NoActivePlaybook(RuntimeError):
    """Live calls refused: no playbook of record has been locked by prep yet."""


def _load_state(db: Any) -> dict[str, dict[str, dict[str, Any]]]:
    state: dict[str, dict[str, dict[str, Any]]] = {"applied": {}, "pending": {}}
    if db is None:
        return state
    raw = db.get_meta(META_KEY)
    if not raw:
        return state
    try:
        data = dict(json.loads(raw))
    except ValueError:
        return state
    if "applied" in data or "pending" in data:
        for k in ("applied", "pending"):
            state[k] = {s: r for s, r in dict(data.get(k) or {}).items() if s in SIDES}
    else:  # pre-pending format: {side: record} == applied
        state["applied"] = {s: r for s, r in data.items() if s in SIDES}
    return state


def _save_state(db: Any, state: dict[str, dict[str, dict[str, Any]]]) -> None:
    db.set_meta(META_KEY, json.dumps(state))


def load_books(db: Any) -> dict[str, dict[str, Any]]:
    """Applied (locked) books — the only ones live calls may use."""
    return _load_state(db)["applied"]


def load_pending(db: Any) -> dict[str, dict[str, Any]]:
    """Custom builds/diffs proposed by prep but not yet confirmed installed."""
    return _load_state(db)["pending"]


def active_books(db: Any, sides: tuple[str, ...] = SIDES) -> dict[str, dict[str, Any]]:
    """Locked books for `sides`; raise NoActivePlaybook when any is missing."""
    books = load_books(db)
    missing = [s for s in sides if s not in books]
    if missing:
        pending = load_pending(db)
        hint = (
            " A custom book is pending — build it, then run `prep --game madden27 -o <opp> --mark-applied`."
            if any(s in pending for s in missing)
            else " Run `prep --game madden27 -o <opp>` first."
        )
        raise NoActivePlaybook(f"No {' / '.join(missing)} playbook locked yet.{hint}")
    return books


def eligible(books: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    return {side: dict(books[side]["formations"]) for side in SIDES if side in books}


# ---------------------------------------------------------------------------
# Prep decision (per side)
# ---------------------------------------------------------------------------

def _book_delta(action: str, target: str, detail: str, *, side: str, why: str,
                plays: list[str] | None = None) -> dict[str, Any]:
    return {
        "action": action,
        "kind": "playbook",
        "target": target,
        "field": side.title(),
        "before": "",
        "after": ", ".join(plays or []),
        "detail": detail,
        "why": why,
        "side": side,
        "validated_status": "meta_grounded",
    }


def _team_stock(side: str, team: str | None) -> str | None:
    if not team:
        return None
    return next((b for b, m in stock_books(side).items() if m.get("team") == team), None)


def plan_side(
    side: str,
    current: dict[str, Any] | None,
    *,
    opp: dict[str, Any],
    choice: str | None,
    team: str | None,
) -> dict[str, Any]:
    """Decide stock vs custom for one side; return the new record + install view."""
    mode_req, book_req = parse_choice(side, choice)
    desired = desired_custom(side, opp)
    team_book = _team_stock(side, team)
    cur_mode = (current or {}).get("mode")

    if mode_req == STOCK:
        mode, name, reason = STOCK, book_req, f"explicit --{side[0]}-book stock:{book_req}"
    elif mode_req == CUSTOM:
        mode, name, reason = CUSTOM, None, f"explicit --{side[0]}-book custom"
    elif cur_mode == CUSTOM:
        mode, name, reason = CUSTOM, None, "staying on your custom book (formation ADD/REMOVE only — no rebuild churn)"
    elif cur_mode == STOCK and (current or {}).get("rev", 0) > 0 and set(
        f for f in desired if f not in custom_core(side)
    ) <= set((current or {}).get("formations") or {}):
        mode, name = STOCK, current["name"]
        reason = f"staying on stock {name} (this opponent needs nothing it lacks)"
    else:
        fits = [
            b for b in ([team_book] if team_book else []) + [(current or {}).get("name")] + list(stock_books(side))
            if b in stock_books(side) and set(desired) <= set(stock_books(side)[b]["formations"])
        ]
        if fits:
            mode, name = STOCK, fits[0]
            reason = (
                f"stock {name} covers every formation this opponent needs"
                + (" (matches primary team)" if name == team_book else "")
            )
        else:
            mode, name = CUSTOM, None
            missing = [f for f in desired if all(f not in m["formations"] for m in stock_books(side).values())]
            reason = f"no stock book has {', '.join(missing) or 'the needed set'} — create custom"

    rev = int((current or {}).get("rev") or 0)
    deltas: list[dict[str, Any]] = []
    checklist: list[dict[str, Any]] = []

    if mode == STOCK:
        new = make_record(side, STOCK, name, rev=rev, reason=reason)
        changed = not current or cur_mode != STOCK or current.get("name") != name
        if changed:
            new["rev"] = rev + 1
            verb = "USE STOCK" if not current or current.get("rev", 0) == 0 else "SWITCH"
            deltas.append(_book_delta(
                verb, name, f"{side.title()} playbook of record → in-game stock book \"{name}\" (no building needed)",
                side=side, why=reason, plays=list(new["formations"]),
            ))
        change = "stock_select" if changed else "none"
    else:
        new = make_record(side, CUSTOM, None, rev=rev, reason=reason, formations=desired)
        if cur_mode != CUSTOM:
            new["rev"] = rev + 1
            change = "switch" if current and current.get("rev", 0) > 0 else "first_custom"
            deltas.append(_book_delta(
                "CREATE CUSTOM", new["name"],
                f"Build a custom {side} playbook with these {len(new['formations'])} formations (full list below)",
                side=side, why=reason, plays=list(new["formations"]),
            ))
            for f, plays in new["formations"].items():
                checklist.append({"formation": f, "plays": plays, "books": formation_books(side, f)})
        else:
            before = set(current.get("formations") or {})
            after = set(new["formations"])
            for f in [x for x in new["formations"] if x not in before]:
                deltas.append(_book_delta(
                    "ADD", f, f"Add formation {f} (from {formation_books(side, f)}) → plays: {', '.join(new['formations'][f])}",
                    side=side, why="needed vs this opponent's persona", plays=new["formations"][f],
                ))
            for f in [x for x in current.get("formations") or {} if x not in after]:
                deltas.append(_book_delta(
                    "REMOVE", f, f"Remove formation {f} (opponent-specific; not needed this week)",
                    side=side, why="keeps the custom book lean", plays=list((current.get("formations") or {}).get(f) or []),
                ))
            change = "diff" if deltas else "none"
            if deltas:
                new["rev"] = rev + 1
    return {
        "side": side,
        "record": new,
        "status": "applied" if mode == STOCK or change == "none" else "pending",
        "change": change,
        "deltas": deltas,
        "checklist": checklist,
        "reason": reason,
        "previous": {"mode": cur_mode, "name": (current or {}).get("name")} if current else None,
    }


def plan_books(
    db: Any,
    *,
    opp: dict[str, Any],
    team: str | None,
    offense_only: bool,
    o_book: str | None = None,
    d_book: str | None = None,
) -> dict[str, dict[str, Any]]:
    current = load_books(db)
    out = {"offense": plan_side("offense", current.get("offense"), opp=opp,
                                choice=o_book, team=team)}
    if offense_only:
        # CPU = offense-only: defense book untouched (report the locked/default one)
        rec = current.get("defense") or make_record("defense", STOCK, DEFAULT_STOCK["defense"], rev=0)
        out["defense"] = {"side": "defense", "record": rec, "change": "none", "deltas": [],
                          "checklist": [], "reason": "CPU = offense-only (defense book unchanged)",
                          "previous": None, "untouched": True}
    else:
        out["defense"] = plan_side("defense", current.get("defense"), opp=opp,
                                   choice=d_book, team=team)
    return out


def lock_books(db: Any, plans: dict[str, dict[str, Any]], *, applied: bool = False) -> None:
    """Persist prep's book decision. Stock (nothing to build) locks now; custom
    builds/diffs stay pending until `applied=True` (prep --mark-applied)."""
    state = _load_state(db)
    for side, p in plans.items():
        if p.get("untouched"):
            continue
        rec = p["record"]
        if rec["mode"] == STOCK or p["change"] == "none" or applied:
            state["applied"][side] = rec
            state["pending"].pop(side, None)
        else:
            state["pending"][side] = rec
        p["status"] = "applied" if state["applied"].get(side) is rec else "pending"
    _save_state(db, state)


def apply_pending(db: Any) -> list[str]:
    """Promote pending custom books to applied (Aidan built them). Returns sides applied."""
    state = _load_state(db)
    done = []
    for side, rec in list(state["pending"].items()):
        state["applied"][side] = rec
        del state["pending"][side]
        done.append(side)
    _save_state(db, state)
    return done


# ---------------------------------------------------------------------------
# Text views
# ---------------------------------------------------------------------------

def format_book(rec: dict[str, Any]) -> str:
    lines = [f"{rec['side'].title()}: {rec['name']} [{rec['mode']}] rev {rec.get('rev', 0)}"]
    if rec.get("reason"):
        lines.append(f"  ({rec['reason']})")
    for f, plays in rec["formations"].items():
        lines.append(f"  - {f}: {', '.join(plays)}")
    return "\n".join(lines)


def format_books(books: dict[str, dict[str, Any]], *, offense_only: bool = False) -> str:
    sides = ("offense",) if offense_only else SIDES
    return "\n".join(format_book(books[s]) for s in sides if s in books)


__all__ = [
    "CUSTOM",
    "STOCK",
    "NoActivePlaybook",
    "active_books",
    "apply_pending",
    "eligible",
    "load_pending",
    "format_book",
    "format_books",
    "lock_books",
    "parse_choice",
    "plan_books",
    "resolve_stock_book",
]
