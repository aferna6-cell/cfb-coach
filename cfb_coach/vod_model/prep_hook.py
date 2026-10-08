"""Attach VOD beaters to a prep plan and, when the sample bar is cleared, edit the book.

The edit is applied before macros and the call sheet are built, so both are
generated from the book live play will actually use.
"""

from __future__ import annotations

from typing import Any

from cfb_coach.vod_model.book import consider_cfb, consider_madden
from cfb_coach.vod_model.flags import book_frozen, prior_enabled
from cfb_coach.vod_model.loader import load_model
from cfb_coach.vod_model.prior import (
    beater_lines,
    cells_for_hint,
    expected_coverage,
    log_nudge,
    quality_weight,
    sheet_bonus,
)


def _mapping_bits(game: str) -> tuple[Any, Any]:
    from cfb_coach.vod_model.mappings import load_mappings

    index = load_mappings()

    def label_of(cell: Any) -> str:
        if index is None:
            return cell.call
        hit = index.lookup(cell.game or game, cell.call)
        return hit.display(cell.call) if hit else cell.call

    return index, label_of


def _report(model: Any, game: str, opponent_type: str, changes: list[dict[str, Any]]) -> dict[str, Any]:
    index, label_of = _mapping_bits(game)
    report = {
        "version": model.version,
        "game": game,
        "opponent_type": opponent_type,
        "quality_weight": quality_weight(),
        "log_nudge": log_nudge(),
        "beater_lines": beater_lines(model, game, opponent_type, label_of=label_of if index else None),
        "changes": changes,
        "frozen": book_frozen(),
    }
    if index is not None:
        report["mapping_version"] = index.version_label
    return report


def _bonus(model: Any, game: str, opponent_type: str, look: str | None, forms: dict[str, list[str]]) -> dict[tuple[str, str], float]:
    if not look or not forms:
        return {}
    from cfb_coach.madden.catalog import norm
    from cfb_coach.vod_model.mappings import load_mappings, match_pair

    index = {norm(play): (formation, play) for formation, plays in forms.items() for play in plays}
    mappings = load_mappings()
    bonus: dict[tuple[str, str], float] = {}
    for cell in cells_for_hint(model, game, opponent_type, look):
        hit = mappings.lookup(game, cell.call) if mappings else None
        if hit is not None:
            pair = match_pair(forms, hit.play_name, hit.formation)
        else:
            pair = index.get(norm(cell.call))
        if pair is None:
            continue
        extra = sheet_bonus(cell, tentative_n=model.tentative_n)
        if extra:
            bonus[pair] = extra
    return bonus


def integrate_madden(
    db: Any,
    opponent_id: str,
    books: dict[str, Any],
    *,
    persist: bool,
    offense_only: bool,
) -> dict[str, Any] | None:
    """Mutate the offense record in ``books`` when an edit clears the gate.

    Returns a plan report, or None when the model is missing or the prior is off.
    A thin model still returns a report and does not write audit or history.
    """
    del offense_only  # defense is never edited; CPU simply has no defense calls
    if not prior_enabled():
        return None
    model = load_model()
    if model is None:
        return None
    from cfb_coach.opponents import is_cpu_opponent

    opponent_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
    record = ((books.get("offense") or {}).get("record") or None)
    changes: list[dict[str, Any]] = []
    if isinstance(record, dict):
        changes = consider_madden(db, opponent_id, record, model, persist=persist)
    look = expected_coverage(db, opponent_id)
    forms = dict((record or {}).get("formations") or {})
    report = _report(model, "madden27", opponent_type, changes)
    report["sheet_bonus"] = {
        f"{formation}::{play}": value
        for (formation, play), value in _bonus(model, "madden27", opponent_type, look, forms).items()
    }
    return report


def sheet_bonus_from_report(report: dict[str, Any] | None) -> dict[tuple[str, str], float] | None:
    raw = (report or {}).get("sheet_bonus") or {}
    if not raw:
        return None
    out: dict[tuple[str, str], float] = {}
    for key, value in raw.items():
        formation, _, play = str(key).partition("::")
        if formation and play:
            out[(formation, play)] = float(value)
    return out or None


def integrate_cfb(db: Any, opponent_id: str, *, dynasty: str, persist: bool) -> dict[str, Any] | None:
    if not prior_enabled():
        return None
    model = load_model()
    if model is None:
        return None
    from cfb_coach.opponents import is_cpu_opponent

    opponent_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
    changes = consider_cfb(db, opponent_id, model, dynasty=dynasty, persist=persist) if db is not None else []
    return _report(model, "cfb27", opponent_type, changes)
