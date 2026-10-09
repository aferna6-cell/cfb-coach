"""Madden 27 offense Custom Adjustments for user games.

A macro here is a Custom Adjustment (Create & Share → Custom Adjustments → Offense),
not a play call. Settings are Aidan's confirmed rows in ``macro_catalog.json``
(route assignments, protection, blocking). Anything he did not confirm stays Default.
A macro is used only when at least one play in the trimmed custom book matches the
concept he wrote it for.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from cfb_coach.macros import (
    aidan_settings_gaps,
    catalog_offense_rows,
    get_macro,
    load_offense_settings,
    macro_key_settings,
)
from cfb_coach.madden.catalog import is_deep, is_run

# Stable preference when the opponent has no film. Later names fill in only when a
# persona or a logged look lifts them (pressure pulls ZERO / HEAT / PROT up).
_ORDER = ["MATCH", "C2", "O-RUN", "C3", "MAN", "RZ", "ZERO", "SHOT", "O-HEAT", "PROT", "O-RPO"]
_PERSONA = {
    "pressure_heavy": ["ZERO", "O-HEAT", "PROT", "MAN"],
    "split_field_zone": ["MATCH", "C2", "O-RUN"],
    "c2_c3_mixer": ["C2", "C3", "MATCH", "O-RUN"],
    "two_high_money_downs": ["MATCH", "C2", "O-RUN", "SHOT"],
}
# coverage_seen_family → macros that answer that look
_CLASS = {
    "single_high": ["C3", "SHOT", "MAN"],
    "cover2": ["C2", "O-RUN"],
    "two_high": ["MATCH", "C2", "O-RUN"],
    "man": ["MAN", "ZERO"],
    "pressure": ["ZERO", "O-HEAT", "PROT"],
}
_W_BASE = 0.30
_W_STEP = 0.02
_W_PERSONA = 0.30
_W_LOG = 0.08
_W_PREV = 0.05
EDITOR_PATH = ["Create & Share", "Custom Adjustments", "Offense", "Create Adjustment"]
_RPO = re.compile(r"rpo", re.I)


def known(mid: str | None) -> bool:
    m = get_macro(mid)
    return bool(m) and (m.get("side") or "") == "offense" and bool(catalog_offense_rows(str(mid)))


def clean_ids(ids: list[Any]) -> list[str]:
    out: list[str] = []
    for x in ids or []:
        k = str(x).upper()
        if known(k) and k not in out:
            out.append(k)
    return out


def xbox_name(mid: str) -> str:
    m = get_macro(mid) or {}
    return str(m.get("xbox_name") or m.get("name") or mid)


def activate_buttons(mid: str) -> str:
    """LB opens Custom Adjustments (research DB, confirmed). The name is the one to pick."""
    from cfb_coach.madden import research_db as rdb

    raw = rdb.buttons("offense", "custom_adjustments")
    name = xbox_name(mid)
    if "pick the adjustment" in raw:
        return raw.replace("pick the adjustment", name)
    return f"LB → {name}"


def _kind_ok(fire: dict[str, Any], play: str) -> bool:
    rpo = bool(_RPO.search(play or ""))
    if fire.get("rpo"):
        return rpo
    if fire.get("run"):
        return is_run(play) and not rpo
    if fire.get("pass"):
        return not is_run(play) and not rpo
    return True


def pairs_in_book(mid: str, book: dict[str, list[str]] | None, *, cap: int = 6) -> list[str]:
    """'Play (Formation)' for plays in *this* book the macro's concept matches.

    CFB pair names that are not in the trimmed book are never listed."""
    spec = (load_offense_settings().get("macros") or {}).get(mid) or {}
    fire = spec.get("fire") or {}
    rx = spec.get("play_re") or ""
    if not book or not rx:
        return []
    out: list[str] = []
    for formation, plays in book.items():
        for play in plays:
            if not re.search(rx, play, re.I) or not _kind_ok(fire, play):
                continue
            if fire.get("deep_only") and not is_deep(play):
                continue
            out.append(f"{play} ({formation})")
            if len(out) >= cap:
                return out
    return out


def _logged_classes(db: Any, opponent_id: str) -> Counter:
    """Coverage families seen while we were on offense (their defense)."""
    out: Counter = Counter()
    if db is None:
        return out
    from cfb_coach.madden.defense_select import coverage_seen_family

    try:
        snaps = db.get_recent_snaps(opponent_id, side="offense", limit=40)
    except Exception:  # noqa: BLE001
        return out
    for snap in snaps:
        fam = coverage_seen_family(snap["coverage_seen"] if hasattr(snap, "keys") else "")
        if fam:
            out[fam] += 1
    return out


def rank_offense(
    *,
    db: Any = None,
    opponent_id: str = "",
    archetype: str = "",
    book: dict[str, list[str]] | None = None,
    previous: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Every offense Custom Adjustment that has Aidan-confirmed settings, best first.

    A row with ``fits`` false has no play in the trimmed book and is not activated."""
    from cfb_coach.madden.macros import LEARNED_SUPPRESS

    persona = _PERSONA.get((archetype or "").lower(), [])
    logged = _logged_classes(db, opponent_id)
    prev = {str(x).upper() for x in (previous or [])}
    rows = []
    for i, mid in enumerate(_ORDER):
        if not known(mid):
            continue
        pairs = pairs_in_book(mid, book, cap=6) if book is not None else ["(book not passed)"]
        fits = book is None or bool(pairs_in_book(mid, book, cap=1))
        parts: dict[str, float] = {f"book order {_ORDER.index(mid) + 1}": round(_W_BASE - _W_STEP * i, 3)}
        if mid in persona:
            parts[f"persona {archetype}"] = _W_PERSONA
        hits = sum(logged[fam] for fam, mids in _CLASS.items() if mid in mids)
        if hits:
            parts["logged coverage"] = min(5, hits) * _W_LOG
        w = (weights or {}).get(mid)
        if w is not None:
            parts["learned weight"] = -1.0 if w <= LEARNED_SUPPRESS else max(-0.5, min(0.5, w))
        if mid in prev:
            parts["last prep"] = _W_PREV
        if not fits:
            parts["no play in the trimmed book"] = -2.0
        score = round(sum(parts.values()), 4)
        rows.append({
            "id": mid,
            "side": "offense",
            "score": score,
            "fits": fits,
            "parts": parts,
            "why": "; ".join(f"{k} {v:+.2f}" for k, v in parts.items()),
        })
    rows.sort(key=lambda r: -r["score"])
    return rows


def select_offense(
    opponent_id: str,
    *,
    db: Any = None,
    archetype: str = "",
    book: dict[str, list[str]] | None = None,
    previous: list[str] | None = None,
    weights: dict[str, float] | None = None,
    n: int = 8,
    allow_experimental: bool | None = None,
    swaps: set[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Top fits. A suppressed or un-opted experimental macro leaves its slot empty.

    The next name is not pulled in to fill that hole. ``allow_experimental`` /
    ``swaps`` opt a lab macro in (config flag or an explicit ``--swap-macro``).
    """
    from cfb_coach.madden.macro_select import apply_slots

    rows = rank_offense(db=db, opponent_id=opponent_id, archetype=archetype, book=book,
                        previous=previous, weights=weights)
    picked = apply_slots(
        rows, n, weights, require_fit=True, allow=allow_experimental, swaps=swaps,
    )
    return picked, rows


def offense_detail(mid: str, book: dict[str, list[str]] | None = None) -> dict[str, Any]:
    meta = get_macro(mid) or {}
    spec = (load_offense_settings().get("macros") or {}).get(mid) or {}
    rows = catalog_offense_rows(mid)
    for row in rows:
        row["source"] = "aidan_notes"
        row["research"] = ""
    gaps = aidan_settings_gaps(mid)
    name = xbox_name(mid)
    buttons = activate_buttons(mid)
    return {
        "id": mid,
        "side": "offense",
        "xbox_name": name,
        "editor_path": list(EDITOR_PATH),
        "in_game": f"At the line: {buttons}",
        "buttons": buttons,
        "settings": rows,
        "has_settings": bool(rows),
        "needs_settings": not rows,
        "n_researched": len(rows),
        "n_fields": len(rows),
        "gaps": gaps,
        "settings_source": "Aidan's offense macro notes (macro_catalog.json, status confirmed). "
                           "Fields he did not write stay Default. Approx notes are not entered.",
        "fire_when": spec.get("fire_when") or meta.get("when_to_arm") or "",
        "pairs_with": pairs_in_book(mid, book),
        "pair_note": "" if book is None or pairs_in_book(mid, book, cap=1)
        else "No play in the trimmed custom book matches this macro — it is not activated.",
        "key": macro_key_settings(mid),
        "purpose": meta.get("purpose") or "",
        "sources": [{
            "id": "aidan_notes",
            "title": "Aidan's Dynasty offense macro notes (macro_catalog.json, confirmed)",
            "url": "",
            "supports": "Route assignments, protection, and blocking, verbatim",
        }],
        "hot_route_menu_source": "",
        "limits": "Madden activation is LB → this name (research DB). The CFB 26 hot-route "
                  "button chart is not a Madden citation — do not copy those buttons. "
                  "Per-route menu buttons in Madden need verification.",
    }


def copy_block(detail: dict[str, Any]) -> str:
    name = detail.get("xbox_name") or detail.get("id")
    lines = [
        f"MACRO: {name} (offense) — Custom Adjustment",
        "Path: " + " > ".join(detail.get("editor_path") or []),
        f"[ ] Name: {name}",
    ]
    cur = None
    for row in detail.get("settings") or []:
        if row["setting"] == row["section"]:
            lines.append(f"[ ] {row['section']}: {row['value']}")
            cur = None
            continue
        if row["section"] != cur:
            cur = row["section"]
            lines.append(cur)
        lines.append(f"  [ ] {row['setting']}: {row['value']}")
    if not detail.get("settings"):
        lines.append("[ ] (no confirmed settings on file — do not invent any)")
    lines.append("[ ] Everything else: Default")
    lines.append("[ ] Save → set Active")
    lines.append(f"In game: {detail.get('buttons')}")
    if detail.get("fire_when"):
        lines.append(f"Fire when: {detail['fire_when']}")
    if detail.get("pairs_with"):
        lines.append("Pairs with (trimmed book): " + ", ".join(detail["pairs_with"]))
    elif detail.get("pair_note"):
        lines.append(detail["pair_note"])
    if detail.get("gaps"):
        lines.append("Needs verification (not entered): " + "; ".join(detail["gaps"]))
    lines.append(detail.get("limits") or "")
    return "\n".join(line for line in lines if line)


def attach_detail(card: dict[str, Any], book: dict[str, list[str]] | None = None) -> dict[str, Any]:
    det = offense_detail(str(card.get("id") or ""), book)
    card["ingame"] = det
    card["name"] = det["xbox_name"]
    card["purpose"] = det.get("purpose") or card.get("purpose") or ""
    card["copy_block"] = copy_block(det)
    card["book_plays"] = det["pairs_with"]
    card["missing_settings"] = det["needs_settings"]
    return card


def _coverage_hit(fire: dict[str, Any], cls: set[str], cov_ok: bool, have_heat: bool) -> bool:
    want = set(fire.get("coverages") or [])
    if not want:
        return True
    hit = cov_ok and bool(cls & want)
    if not hit and fire.get("fallback_pressure") and not have_heat:
        hit = cov_ok and "pressure" in cls
    return hit


def _avoided(fire: dict[str, Any], cls: set[str], play: str) -> bool:
    return any(
        a.get("coverage") in cls and re.search(a.get("play_re") or "$^", play, re.I)
        for a in fire.get("avoid") or []
    )


def _pair_key(item: str) -> tuple[str, str]:
    play, _, rest = item.partition(" (")
    form = rest[:-1] if rest.endswith(")") else ""
    return form, play


def _ready(
    mid: str,
    act: list[str],
    data: dict[str, Any],
    weights: dict[str, float] | None,
    cooled: set[str] | None,
    score_phase: str | None,
) -> tuple[float | None, dict[str, Any]] | None:
    """None when this macro cannot arm. Otherwise (learned weight, fire block)."""
    from cfb_coach.madden.macros import LEARNED_SUPPRESS

    if mid not in act or mid not in data:
        return None
    if cooled and mid in cooled:
        return None
    if mid == "SHOT" and score_phase in ("protect", "prevent"):
        return None
    w = (weights or {}).get(mid)
    if w is not None and w <= LEARNED_SUPPRESS:
        return None
    return w, data[mid].get("fire") or {}


def _choose_weighted(cands: list[tuple[str, float | None]]) -> tuple[str, float | None] | None:
    """Highest learned weight. A missing weight is neutral (0). Ties keep the old priority."""
    if not cands:
        return None
    from cfb_coach.macros import LIVE_MACRO_PRIORITY

    def key(item: tuple[str, float | None]) -> tuple[float, int]:
        mid, w = item
        weight = 0.0 if w is None else float(w)
        prio = LIVE_MACRO_PRIORITY.index(mid) if mid in LIVE_MACRO_PRIORITY else 99
        return (-weight, prio)

    return min(cands, key=key)


def situation_macro(
    *,
    zone: str,
    coverage: str | None,
    coverage_source: str,
    active: list[str],
    down: int | None = None,
    repeated: bool = False,
    book: dict[str, list[str]] | None = None,
    weights: dict[str, float] | None = None,
    score_phase: str | None = None,
    pool: list[tuple[str, str]] | None = None,
    cooled: set[str] | None = None,
) -> dict[str, Any] | None:
    """The stored offense Custom Adjustment this snap should call, before a play is sampled.

    Every macro whose fire rules match this look is a candidate. The highest learned
    weight wins (a missing weight is neutral). Ties keep ``LIVE_MACRO_PRIORITY``, which
    is what CFB still walks on its own. A blank snap with no look does not invent a
    coverage from old logs. Returns None unless at least one in-book pair is also in
    ``pool``. Protecting a lead skips SHOT. ``cooled`` macros stay off for the half."""
    from cfb_coach.macros import LIVE_MACRO_PRIORITY, classify_coverage

    act = clean_ids(active)
    if not act or not book:
        return None
    data = load_offense_settings().get("macros") or {}
    cls = classify_coverage(coverage)
    cov_ok = bool(cls) and (coverage_source == "live" or repeated)
    have_heat = any(a in act for a in ("O-HEAT", "PROT"))
    allowed = set(pool) if pool is not None else None
    found: list[tuple[str, float | None, list[tuple[str, str]], dict[str, Any]]] = []
    for mid in LIVE_MACRO_PRIORITY:
        ready = _ready(mid, act, data, weights, cooled, score_phase)
        if ready is None:
            continue
        w, fire = ready
        if zone not in (fire.get("zones") or ["open", "rz", "gl"]):
            continue
        if fire.get("downs") and down not in fire["downs"]:
            continue
        if not _coverage_hit(fire, cls, cov_ok, have_heat):
            continue
        keys: list[tuple[str, str]] = []
        for item in pairs_in_book(mid, book, cap=1 + sum(map(len, book.values()))):
            form, play = _pair_key(item)
            if _avoided(fire, cls, play):
                continue
            if allowed is not None and (form, play) not in allowed:
                continue
            keys.append((form, play))
        if not keys:
            continue
        found.append((mid, w, keys, fire))
    chosen = _choose_weighted([(mid, w) for mid, w, _keys, _fire in found])
    if chosen is None:
        return None
    mid, w = chosen
    _keys = next(keys for name, _w, keys, _fire in found if name == mid)
    fire = next(block for name, _w, _keys, block in found if name == mid)
    det = offense_detail(mid, book)
    want = set(fire.get("coverages") or [])
    trig = f"{coverage_source} {coverage}" if want else ("red zone" if zone in ("rz", "gl") else "run")
    return {
        "id": mid,
        "name": det["xbox_name"],
        "side": "offense",
        "why": trig,
        "key": det["key"],
        "buttons": det["buttons"],
        "settings": [r for r in det["settings"]],
        "fire_when": det["fire_when"],
        "learned_weight": w,
        "pairs": _keys,
        "rpo": bool(fire.get("rpo")),
    }


def suggest_for_snap(
    *,
    zone: str,
    play: str,
    coverage: str | None,
    coverage_source: str,
    active: list[str],
    down: int | None = None,
    repeated: bool = False,
    book: dict[str, list[str]] | None = None,
    weights: dict[str, float] | None = None,
    score_phase: str | None = None,
    cooled: set[str] | None = None,
) -> dict[str, Any] | None:
    """One offense Custom Adjustment for this snap, or None.

    Only ids in ``active``. The called play has to be one of the macro's pairs in the
    trimmed book. When more than one macro fits that play, the higher learned weight
    wins. A coverage macro needs a live look (or a look repeated this game)."""
    from cfb_coach.macros import LIVE_MACRO_PRIORITY, classify_coverage

    act = clean_ids(active)
    if not act or not play or not book:
        return None
    data = load_offense_settings().get("macros") or {}
    cls = classify_coverage(coverage)
    cov_ok = bool(cls) and (coverage_source == "live" or repeated)
    have_heat = any(a in act for a in ("O-HEAT", "PROT"))
    found: list[tuple[str, float | None, dict[str, Any]]] = []
    for mid in LIVE_MACRO_PRIORITY:
        ready = _ready(mid, act, data, weights, cooled, score_phase)
        if ready is None:
            continue
        w, fire = ready
        if zone not in (fire.get("zones") or ["open", "rz", "gl"]):
            continue
        pairs = pairs_in_book(mid, book, cap=1 + sum(map(len, book.values())))
        if not any(play.lower() == item.split(" (", 1)[0].lower() for item in pairs):
            continue
        if fire.get("downs") and down not in fire["downs"]:
            continue
        if not _coverage_hit(fire, cls, cov_ok, have_heat):
            continue
        if _avoided(fire, cls, play):
            continue
        found.append((mid, w, fire))
    chosen = _choose_weighted([(mid, w) for mid, w, _fire in found])
    if chosen is None:
        return None
    mid, w = chosen
    fire = next(block for name, _w, block in found if name == mid)
    det = offense_detail(mid, book)
    want = set(fire.get("coverages") or [])
    trig = f"{coverage_source} {coverage}" if want else ("red zone" if zone in ("rz", "gl") else "run")
    return {
        "id": mid,
        "name": det["xbox_name"],
        "side": "offense",
        "kind": "situation" if not want else "look",
        "why": f"{trig} on {play}",
        "key": det["key"],
        "buttons": det["buttons"],
        "settings": [r for r in det["settings"]],
        "fire_when": det["fire_when"],
        "learned_weight": w,
    }


__all__ = [
    "activate_buttons", "attach_detail", "clean_ids", "copy_block", "known", "offense_detail",
    "pairs_in_book", "rank_offense", "select_offense", "situation_macro", "suggest_for_snap",
    "xbox_name",
]
