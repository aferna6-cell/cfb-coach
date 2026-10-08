"""Playbook edits driven by a VOD model, with an audit log and a revert snapshot.

Only names that already exist in the stock book the trimmed book was built
from, or in another catalogued book for that game, can be added. A miss is
skipped. Nothing here invents a play name.

Madden writes the new record straight into ``active_playbook_json`` applied
state. There is no pending confirmation: the next live snap reads that book.
CFB inserts a new current revision the same way (the previous revision stays
in the history table). With the prior off, or the book frozen, or every cell
under the sample bar, this step does not touch the database.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

from cfb_coach.vod_model.adapter import VodCell, VodModel
from cfb_coach.vod_model.flags import book_frozen, prior_enabled
from cfb_coach.vod_model.prior import cell_tier, cells_for_hint, expected_coverage, select_cell

AUDIT_KEY = "vod_book_audit"
HISTORY_KEY = "vod_book_history"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_list(db: Any, key: str) -> list[dict[str, Any]]:
    if db is None:
        return []
    try:
        raw = db.get_meta(key)
        data = json.loads(raw) if raw else []
    except (ValueError, TypeError, AttributeError):
        return []
    return list(data) if isinstance(data, list) else []


def _save_list(db: Any, key: str, rows: list[dict[str, Any]]) -> None:
    db.set_meta(key, json.dumps(rows))


def read_audit(db: Any) -> list[dict[str, Any]]:
    return _load_list(db, AUDIT_KEY)


def _audit_row(
    cell: VodCell,
    *,
    game: str,
    opponent_id: str,
    action: str,
    changed: str,
    replaced: str,
    why: str,
    tier: str,
) -> dict[str, Any]:
    return {
        "ts": _now(),
        "model_version": cell.model_version,
        "game": game,
        "opponent_id": opponent_id,
        "side": "offense",
        "action": action,
        "changed": changed,
        "replaced": replaced,
        "why": why,
        "n": cell.n,
        "n_vods": cell.n_vods,
        "success": round(cell.shrunk_success, 4),
        "raw_success": round(cell.raw_success, 4),
        "lower_bound": round(cell.lower_bound, 4),
        "baseline": round(cell.baseline, 4),
        "tier": tier,
        "look": cell.look,
    }


def _why(cell: VodCell, tier: str, look: str) -> str:
    return (
        f"VOD vs {look}: {cell.call} shrunk {cell.shrunk_success:.3f} "
        f"(raw {cell.raw_success:.3f}, base {cell.baseline:.3f}) "
        f"lb {cell.lower_bound:.3f} tier {tier}, n={cell.n} across {cell.n_vods} VODs. "
        "Logged tendencies picked the look. Logs nudge a near-tie and do not override this rate."
    )


def _own_norm(db: Any, opponent_id: str, formation: str, play: str) -> float:
    if db is None:
        return 0.0
    try:
        from cfb_coach.learning import LearnedWeights

        learned = LearnedWeights.load(db, opponent_id, side="offense")
        key = f"play::{formation}::{play}"
        if learned.n(key):
            return float(learned.norm(key))
    except Exception:  # noqa: BLE001
        return 0.0
    return 0.0


def _in_book(forms: dict[str, list[str]], call_name: str) -> tuple[str, str] | None:
    from cfb_coach.madden.catalog import norm

    key = norm(call_name)
    for formation, plays in forms.items():
        for play in plays:
            if norm(play) == key:
                return formation, play
    return None


def resolve_madden(side: str, source_book: str | None, forms: dict[str, list[str]], call_name: str) -> dict[str, Any] | None:
    """A catalogued formation+play, or None. Never a guessed name.

    Prefers a formation already in the trimmed book. Otherwise the trimmed
    book's source stock book, then any other catalogued book.
    """
    from cfb_coach.madden import catalog

    if _in_book(forms, call_name):
        return None
    books: list[str] = []
    if source_book:
        books.append(source_book)
    for name in catalog.book_names(side):
        if name not in books:
            books.append(name)
    fallback: dict[str, Any] | None = None
    for book in books:
        stock = catalog.book_formations(side, book)
        for formation in stock:
            play = catalog.canonical_play(side, book, formation, call_name)
            if not play:
                continue
            if formation in forms:
                if play in forms[formation]:
                    return None
                return {"formation": formation, "play": play, "book": book, "action": "add_play",
                        "plays": list(forms[formation]) + [play]}
            if fallback is None:
                fallback = {
                    "formation": formation,
                    "play": play,
                    "book": book,
                    "action": "add_formation",
                    "plays": list(stock[formation]),
                }
    return fallback


def _apply_madden_edit(record: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    """Return the edit description after mutating ``record`` in place."""
    from cfb_coach.madden.playbook import MAX_FOCUS

    side = record.get("side") or "offense"
    forms = {f: list(ps) for f, ps in (record.get("formations") or {}).items()}
    formation = resolved["formation"]
    replaced = ""
    action = resolved["action"]
    if action == "add_play":
        forms[formation] = list(resolved["plays"])
    else:
        cap = int(MAX_FOCUS.get(side) or 5)
        if len(forms) >= cap:
            replaced = list(forms)[-1]
            forms.pop(replaced, None)
            auds = dict(record.get("audibles") or {})
            auds.pop(replaced, None)
            record["audibles"] = auds
            action = "swap_formation"
        forms[formation] = list(resolved["plays"])
    record["formations"] = forms
    record["rev"] = int(record.get("rev") or 1) + 1
    record["locked_ts"] = _now()
    note = record.get("reason") or ""
    extra = f"VOD {action} {formation}"
    record["reason"] = (note + " | " + extra).strip(" |") if note else extra
    return {"action": action, "formation": formation, "play": resolved["play"], "replaced": replaced,
            "book": resolved["book"]}


def consider_madden(
    db: Any,
    opponent_id: str,
    record: dict[str, Any],
    model: VodModel,
    *,
    game: str = "madden27",
    persist: bool = False,
) -> list[dict[str, Any]]:
    """Maybe edit one offense formation. No-op (and no meta write) when gated."""
    if not prior_enabled() or book_frozen() or model is None or not record:
        return []
    from cfb_coach.opponents import is_cpu_opponent

    opponent_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
    look = expected_coverage(db, opponent_id)
    if not look:
        return []
    cells = list(cells_for_hint(model, game, opponent_type, look))
    if not cells:
        return []
    forms = {f: list(ps) for f, ps in (record.get("formations") or {}).items()}
    source = record.get("source_book") or (record.get("name") if record.get("mode") == "stock" else None)
    actionable = [c for c in cells if cell_tier(c, tentative_n=model.tentative_n) in ("med", "high")]
    chosen = select_cell(actionable, lambda _cell: 0.0, tentative_n=model.tentative_n)
    if chosen is None or _in_book(forms, chosen.call):
        return []
    resolved = resolve_madden("offense", source if isinstance(source, str) else None, forms, chosen.call)
    if resolved is None:
        return []
    if persist and db is not None:
        history = _load_list(db, HISTORY_KEY)
        history.append({
            "ts": _now(),
            "game": game,
            "opponent_id": opponent_id,
            "side": "offense",
            "record": copy.deepcopy(record),
        })
        _save_list(db, HISTORY_KEY, history[-20:])
    edit = _apply_madden_edit(record, resolved)
    tier = cell_tier(chosen, tentative_n=model.tentative_n)
    changed = f"{edit['formation']} / {edit['play']}"
    row = _audit_row(
        chosen, game=game, opponent_id=opponent_id, action=edit["action"], changed=changed,
        replaced=edit["replaced"], why=_why(chosen, tier, look), tier=tier,
    )
    row["source_book"] = edit["book"]
    if persist and db is not None:
        audit = _load_list(db, AUDIT_KEY)
        audit.append(row)
        _save_list(db, AUDIT_KEY, audit[-50:])
    return [row]


def revert_last(db: Any, *, game: str = "madden27") -> dict[str, Any] | None:
    """Put back the book snapshot taken before the latest VOD edit for ``game``."""
    if db is None:
        return None
    if game == "cfb27":
        return _revert_cfb(db)
    history = _load_list(db, HISTORY_KEY)
    idx = next((i for i in range(len(history) - 1, -1, -1) if history[i].get("game") == game), None)
    if idx is None:
        return None
    snap = history.pop(idx)
    record = snap.get("record") or {}
    side = snap.get("side") or record.get("side") or "offense"
    from cfb_coach.madden.playbook import _load_state, _save_state

    state = _load_state(db)
    if record:
        state["applied"][side] = record
        state["pending"].pop(side, None)
        _save_state(db, state)
    _save_list(db, HISTORY_KEY, history)
    row = {
        "ts": _now(),
        "model_version": "",
        "game": game,
        "opponent_id": snap.get("opponent_id") or "",
        "side": side,
        "action": "revert",
        "changed": record.get("name") or side,
        "replaced": "",
        "why": "Reverted the latest VOD playbook edit. Live calls use this restored book.",
        "n": 0,
        "n_vods": 0,
        "success": 0.0,
        "raw_success": 0.0,
        "lower_bound": 0.0,
        "baseline": 0.0,
        "tier": "",
        "look": "",
    }
    audit = _load_list(db, AUDIT_KEY)
    audit.append(row)
    _save_list(db, AUDIT_KEY, audit[-50:])
    return row


def resolve_cfb(call_name: str) -> tuple[str, str, str | None] | None:
    from cfb_coach.cfb_catalog import (
        book_plays,
        canonical_formation,
        canonical_play,
        formation_books,
        formations_with_play,
    )

    if not (call_name or "").strip():
        return None
    if "/" in call_name:
        left, right = [part.strip() for part in call_name.split("/", 1)]
        formation = canonical_formation(left)
        if formation:
            play = canonical_play(formation, right)
            if play and book_plays(formation, None):
                books = formation_books(formation)
                return formation, play, (books[0] if books else None)
    forms = formations_with_play(call_name)
    if not forms:
        return None
    formation = forms[0]
    play = canonical_play(formation, call_name)
    if not play:
        return None
    books = formation_books(formation)
    return formation, play, (books[0] if books else None)


def consider_cfb(
    db: Any,
    opponent_id: str,
    model: VodModel,
    *,
    dynasty: str,
    game: str = "cfb27",
    persist: bool = False,
) -> list[dict[str, Any]]:
    """Add one catalog formation to the current CFB book, or do nothing.

    No applied book means no edit (VOD does not seed one). The new revision
    is current immediately, so live calls use it without ``book apply``.
    """
    if not prior_enabled() or book_frozen() or model is None or db is None:
        return []
    from cfb_coach.cfb_catalog import book_plays
    from cfb_coach.cfb_playbook import PRACTICAL, _insert, _set_status, current_rev, form_plays
    from cfb_coach.opponents import is_cpu_opponent

    opponent_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
    look = expected_coverage(db, opponent_id)
    if not look:
        return []
    cur = current_rev(db, dynasty)
    if not cur:
        return []
    have = form_plays(cur["book"].get("formations"))
    cells = list(cells_for_hint(model, game, opponent_type, look))
    chosen_cell = select_cell(
        [c for c in cells if cell_tier(c, tentative_n=model.tentative_n) in ("med", "high")],
        lambda _c: 0.0,
        tentative_n=model.tentative_n,
    )
    if chosen_cell is None:
        return []
    resolved = resolve_cfb(chosen_cell.call)
    if resolved is None:
        return []
    formation, play, source = resolved
    if formation in have and play in (have.get(formation) or []):
        return []
    plays = list(book_plays(formation, source))
    if play not in plays:
        return []
    formations = copy.deepcopy(cur["book"].get("formations") or {})
    replaced = ""
    cap = int(PRACTICAL.get("max_formations") or 8)
    names = list(form_plays(formations))
    if formation not in names and len(names) >= cap:
        replaced = names[-1]
        formations.pop(replaced, None)
        action = "swap_formation"
    else:
        action = "add_formation" if formation not in names else "add_play"
    formations[formation] = {"source_book": source, "plays": plays}
    tier = cell_tier(chosen_cell, tentative_n=model.tentative_n)
    changed = f"{formation} / {play}"
    row = _audit_row(
        chosen_cell, game=game, opponent_id=opponent_id, action=action, changed=changed,
        replaced=replaced, why=_why(chosen_cell, tier, look), tier=tier,
    )
    if not persist:
        return [row]
    history = _load_list(db, HISTORY_KEY)
    history.append({
        "ts": _now(),
        "game": game,
        "dynasty": dynasty,
        "opponent_id": opponent_id,
        "rev": cur.get("rev"),
        "book": copy.deepcopy(cur["book"]),
    })
    _save_list(db, HISTORY_KEY, history[-20:])
    book = dict(cur["book"], formations=formations)
    _set_status(db, dynasty, cur["rev"], "superseded")
    _insert(
        db, dynasty, status="current", kind="vod", book=book,
        edits=[{
            "op": "add_formation" if not replaced else "swap_formation",
            "formation": formation,
            "replaced": replaced,
            "reason": row["why"],
        }],
        summary=f"VOD {action}: {changed}",
        parent_rev=cur.get("rev"),
        applied=True,
    )
    audit = _load_list(db, AUDIT_KEY)
    audit.append(row)
    _save_list(db, AUDIT_KEY, audit[-50:])
    return [row]


def _revert_cfb(db: Any) -> dict[str, Any] | None:
    from cfb_coach.cfb_playbook import _insert, _set_status, current_rev

    history = _load_list(db, HISTORY_KEY)
    idx = next((i for i in range(len(history) - 1, -1, -1) if history[i].get("game") == "cfb27"), None)
    if idx is None:
        return None
    snap = history.pop(idx)
    dynasty = snap.get("dynasty") or "alabama"
    previous = snap.get("book") or {}
    cur = current_rev(db, dynasty)
    if cur:
        _set_status(db, dynasty, cur["rev"], "superseded")
    _insert(
        db, dynasty, status="current", kind="vod_revert", book=previous, edits=[],
        summary="Reverted the latest VOD playbook edit",
        parent_rev=(cur or {}).get("rev"),
        applied=True,
    )
    _save_list(db, HISTORY_KEY, history)
    row = {
        "ts": _now(), "model_version": "", "game": "cfb27", "opponent_id": snap.get("opponent_id") or "",
        "side": "offense", "action": "revert", "changed": dynasty, "replaced": "",
        "why": "Reverted the latest VOD playbook edit. Live calls use this restored book.",
        "n": 0, "n_vods": 0, "success": 0.0, "raw_success": 0.0, "lower_bound": 0.0,
        "baseline": 0.0, "tier": "", "look": "",
    }
    audit = _load_list(db, AUDIT_KEY)
    audit.append(row)
    _save_list(db, AUDIT_KEY, audit[-50:])
    return row
