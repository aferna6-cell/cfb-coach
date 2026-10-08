"""Live snaps read the newest book, then optionally lean on a VOD beater.

``play`` does not ask Aidan to confirm a suggested formation swap. The book
written into ``active_playbook_json`` (or the current CFB revision) is the
book the next snap calls. A session that already cached the previous book
passes that cache in and still gets the database copy.

Custom Adjustment pairs and call-sheet packages are filtered to that same
book. A package that names a formation the book no longer has is dropped and
the sheet is rebuilt from the formations that remain.
"""

from __future__ import annotations

import json
from typing import Any

from cfb_coach.vod_model.adapter import VodCell
from cfb_coach.vod_model.flags import prior_enabled
from cfb_coach.vod_model.loader import load_model
from cfb_coach.vod_model.prior import (
    cell_tier,
    cells_for_hint,
    clamp_norm,
    expected_coverage,
    select_cell,
)


def play_candidates(db: Any, side: str = "offense") -> dict[str, list[str]]:
    """Formations the next Madden snap may call. Re-read on every snap."""
    if db is None:
        return {}
    from cfb_coach.madden.playbook import active_books, eligible

    try:
        raw = active_books(db, (side,))
    except Exception:  # noqa: BLE001 — NoActivePlaybook and friends
        return {}
    return dict((eligible(raw).get(side) or {}))


def book_for_next_snap(
    db: Any,
    side: str = "offense",
    cached: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """The book a running session must use on the next play call.

    ``cached`` is the snapshot taken when the session started. It is not the
    book. Prep, or any other writer of ``active_playbook_json``, is visible
    here immediately.
    """
    del cached
    return play_candidates(db, side)


def _call_packages(db: Any, opponent_id: str) -> dict[str, list[dict[str, Any]]]:
    if db is None:
        return {}
    from cfb_coach.madden.macros import ACTIVE_META_KEY

    raw = db.get_meta(ACTIVE_META_KEY.format(opp=opponent_id))
    if not raw:
        return {}
    try:
        rec = json.loads(raw)
    except ValueError:
        return {}
    packages = rec.get("call_packages") or {}
    return packages if isinstance(packages, dict) else {}


def _rebuild_packages(book: dict[str, list[str]], db: Any, opponent_id: str) -> list[dict[str, Any]]:
    from cfb_coach.madden.gameplan_macros import build_gameplan

    built = build_gameplan(
        offense_book=book, defense_book={}, db=db, opponent_id=opponent_id, offense_only=True,
    )
    return list(built.get("offense") or [])


def macro_candidates(db: Any, opponent_id: str) -> list[dict[str, Any]]:
    """Call-sheet packages and Custom Adjustment pairs inside the newest book.

    Every row's formation is in that book. A formation the last edit removed
    cannot appear. Plays of a formation the edit added can.
    """
    book = play_candidates(db, "offense")
    if not book:
        return []
    allowed = set(book)
    stored = list((_call_packages(db, opponent_id).get("offense") or []))
    fresh = [
        pkg for pkg in stored
        if pkg.get("formation") in allowed and pkg.get("play") in (book.get(pkg.get("formation")) or [])
    ]
    if stored and len(fresh) != len(stored):
        fresh = _rebuild_packages(book, db, opponent_id)
        fresh = [
            pkg for pkg in fresh
            if pkg.get("formation") in allowed and pkg.get("play") in (book.get(pkg.get("formation")) or [])
        ]
    elif not stored:
        fresh = _rebuild_packages(book, db, opponent_id)
    out: list[dict[str, Any]] = []
    for pkg in fresh:
        out.append({
            "kind": "package",
            "id": pkg.get("id") or "",
            "formation": pkg.get("formation"),
            "play": pkg.get("play"),
        })
    from cfb_coach.madden.macros import load_selection
    from cfb_coach.madden.offense_macros import pairs_in_book

    selection = load_selection(db, opponent_id) or {}
    for mid in selection.get("offense") or []:
        if not pairs_in_book(mid, book, cap=1):
            continue
        for item in pairs_in_book(mid, book, cap=500):
            play, _, rest = item.partition(" (")
            formation = rest[:-1] if rest.endswith(")") else ""
            if formation not in allowed or play not in (book.get(formation) or []):
                continue
            out.append({"kind": "macro", "macro": mid, "formation": formation, "play": play})
    return out


def _own_norm(db: Any, opponent_id: str, formation: str, play: str) -> float:
    if db is None:
        return 0.0
    try:
        from cfb_coach.learning import LearnedWeights

        learned = LearnedWeights.load(db, opponent_id, side="offense")
        key = f"play::{formation}::{play}"
        if learned.n(key):
            return clamp_norm(float(learned.norm(key)))
    except Exception:  # noqa: BLE001
        return 0.0
    return 0.0


def _pair_in_book(forms: dict[str, list[str]], call_name: str) -> tuple[str, str] | None:
    from cfb_coach.madden.catalog import norm

    key = norm(call_name)
    hits = [(formation, play) for formation, plays in forms.items() for play in plays if norm(play) == key]
    return hits[0] if hits else None


def apply_madden_call(
    call: Any,
    sit: Any,
    opponent_id: str,
    db: Any,
    books: dict[str, dict[str, list[str]]],
) -> Any:
    """Switch the called play only when a med/high in-book beater wins. Thin data is a no-op."""
    if not prior_enabled() or call is None or getattr(call, "side", "") != "offense":
        return call
    model = load_model()
    if model is None:
        return call
    from cfb_coach.opponents import is_cpu_opponent

    opponent_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
    hint = getattr(sit, "coverage_hint", None) or expected_coverage(db, opponent_id)
    if not hint:
        return call
    forms = dict(books.get("offense") or {})
    cells = list(cells_for_hint(model, "madden27", opponent_type, hint))
    if not cells:
        return call
    in_book: list[VodCell] = []
    pairs: dict[int, tuple[str, str]] = {}
    for cell in cells:
        pair = _pair_in_book(forms, cell.call)
        if pair is None:
            continue
        in_book.append(cell)
        pairs[id(cell)] = pair
    chosen = select_cell(in_book, lambda c: _own_norm(db, opponent_id, *pairs[id(c)]), tentative_n=model.tentative_n)
    if chosen is None:
        return call
    formation, play = pairs[id(chosen)]
    tier = cell_tier(chosen, tentative_n=model.tentative_n)
    if (call.formation, call.play) == (formation, play):
        return call
    call.formation = formation
    call.play = play
    if getattr(call, "macro", None):
        from cfb_coach.madden.offense_macros import pairs_in_book

        paired = pairs_in_book(call.macro, forms, cap=500)
        if not any(play.lower() == item.split(" (", 1)[0].lower() for item in paired):
            call.macro = None
            call.macro_info = None
            if hasattr(call, "adj_or_macro"):
                call.adj_or_macro = "No adj"
    call.rationale = (
        f"{call.rationale} | VOD {hint} → {play} ({formation}) "
        f"n={chosen.n} shrunk={chosen.shrunk_success:.2f} lb={chosen.lower_bound:.2f} tier={tier}"
    ).strip(" |")
    return call


def apply_cfb_call(call: Any, sit: Any, opponent_id: str, db: Any, *, dynasty: str | None) -> Any:
    if not prior_enabled() or call is None or getattr(call, "side", "") != "offense":
        return call
    model = load_model()
    if model is None or db is None:
        return call
    from cfb_coach.opponents import is_cpu_opponent
    from cfb_coach.playcaller import _load_book

    book = _load_book(db, dynasty)
    forms = (book or {}).get("formations") or {}
    if not forms:
        return call
    opponent_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
    hint = getattr(sit, "coverage_hint", None) or expected_coverage(db, opponent_id)
    if not hint:
        return call
    cells = list(cells_for_hint(model, "cfb27", opponent_type, hint))
    in_book: list[VodCell] = []
    pairs: dict[int, tuple[str, str]] = {}
    for cell in cells:
        pair = _pair_in_book(forms, cell.call)
        if pair is None:
            continue
        in_book.append(cell)
        pairs[id(cell)] = pair
    chosen = select_cell(in_book, lambda c: _own_norm(db, opponent_id, *pairs[id(c)]), tentative_n=model.tentative_n)
    if chosen is None:
        return call
    formation, play = pairs[id(chosen)]
    if (call.formation, call.play) == (formation, play):
        return call
    tier = cell_tier(chosen, tentative_n=model.tentative_n)
    call.formation = formation
    call.play = play
    call.rationale = (
        f"{call.rationale} | VOD {hint} → {play} ({formation}) "
        f"n={chosen.n} shrunk={chosen.shrunk_success:.2f} lb={chosen.lower_bound:.2f} tier={tier}"
    ).strip(" |")
    return call
