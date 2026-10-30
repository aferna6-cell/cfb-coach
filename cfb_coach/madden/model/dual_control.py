"""Per-side labels, macro gate, and fallback source for the Wild Card dual pilot.

Offense and defense each keep their own mode. A displayed call is EXPERIMENTAL
only when that side's model produced it. A timeout, exception, illegal play,
missing artifact, or invalid adjustment stays HEURISTIC and names the
heuristic caller as ``fallback_source``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

OFFENSE_FALLBACK_SOURCE = "heuristic_offense"
DEFENSE_FALLBACK_SOURCE = "select_defense"
LABELS = ("EXPERIMENTAL", "HEURISTIC")


def displayed_label(call: Any) -> str:
    """EXPERIMENTAL only when this call is the model's own pick."""
    info = getattr(call, "ml_experimental", None)
    if isinstance(info, dict):
        if info.get("fell_back"):
            return "HEURISTIC"
        if info.get("ml_play"):
            return "EXPERIMENTAL"
        return "HEURISTIC"
    decision = getattr(call, "_pending_ml_decision", None)
    if decision is not None and getattr(decision, "fell_back", False) is False:
        mode = getattr(getattr(decision, "effective_mode", None), "value", None)
        source = getattr(getattr(decision, "policy_source", None), "value", None)
        if mode == "experimental" and source == "model":
            return "EXPERIMENTAL"
    note = getattr(call, "ml_defense_fallback", None)
    if isinstance(note, dict) and note.get("fell_back"):
        return "HEURISTIC"
    return "HEURISTIC"


def stamp_displayed_label(call: Any) -> str:
    label = displayed_label(call)
    try:
        call.call_label = label
    except Exception:  # noqa: BLE001 — a label must not change the play
        pass
    return label


def macro_is_live(
    name: str,
    *,
    side: str,
    armed_ids: Iterable[str],
    db: Any = None,
    opponent_id: str | None = None,
) -> bool:
    """True when the macro is installed, armed, and executable.

    Catalog loadout ids use the preflight screen. A defensive lab sheet counts
    only after it has been verified and armed.
    """
    from cfb_coach.madden.preflight import screen_adjustment_id

    token = str(name or "").strip()
    if not token:
        return False
    armed = {str(item).strip().upper() for item in armed_ids}
    if screen_adjustment_id(token, side=side, armed_ids=armed) == "executable":
        return True
    if side != "defense" or db is None:
        return False
    try:
        from cfb_coach.madden.model.defense_macro_lab import approved_macros

        for row in approved_macros(db, opponent_id or ""):
            if str(row.get("name") or "") == token and row.get("verified_armed"):
                return True
    except Exception:  # noqa: BLE001 — an unreadable lab is not executable
        return False
    return False


def _catalog_macro(name: str, side: str) -> bool:
    token = str(name or "").strip()
    if not token:
        return False
    if side == "offense":
        from cfb_coach.madden.offense_macros import known

        return bool(known(token))
    from cfb_coach.madden.macro_pool import pool_macro

    return bool(pool_macro(token))


def _macro_token(text: str) -> str:
    head = str(text or "").split("—", 1)[0].split(" - ", 1)[0]
    return head.split(" [", 1)[0].strip()


def enforce_shown_adjustments(
    call: Any,
    *,
    side: str,
    armed: Iterable[str],
    db: Any = None,
    opponent_id: str | None = None,
) -> list[str]:
    """Drop a shown macro that is not armed and installed for this opponent.

    The play stays. The macro slot becomes none and the reason is logged.
    A researched non-macro sheet (hot route, contain) is not a loadout id.
    """
    import logging

    reasons: list[str] = []

    def _drop(name: str) -> None:
        token = str(name or "").strip()
        if not token:
            return
        reason = f"dropped_unarmed_macro:{token}"
        if reason not in reasons:
            reasons.append(reason)

    macro = getattr(call, "macro", None)
    if macro and not macro_is_live(
        str(macro), side=side, armed_ids=armed, db=db, opponent_id=opponent_id,
    ):
        _drop(str(macro))
        call.macro = None
        call.macro_info = None
        label = str(getattr(call, "adj_or_macro", "") or "")
        if side == "defense" or _macro_token(label).upper() == str(macro).strip().upper():
            call.adj_or_macro = "none" if side == "defense" else "No adj"
    suggest = getattr(call, "suggest_macro", None)
    if suggest:
        token = _macro_token(str(suggest))
        if (
            token
            and _catalog_macro(token, side)
            and not macro_is_live(
                token, side=side, armed_ids=armed, db=db, opponent_id=opponent_id,
            )
        ):
            _drop(token)
            call.suggest_macro = None
    adjustment = getattr(call, "adjustment", None)
    if isinstance(adjustment, dict) and adjustment.get("id"):
        ident = str(adjustment["id"])
        if _catalog_macro(ident, side) and not macro_is_live(
            ident, side=side, armed_ids=armed, db=db, opponent_id=opponent_id,
        ):
            _drop(ident)
            call.adjustment = None
            if side == "offense":
                call.adj_or_macro = "No adj"
    if not reasons:
        return []
    prior = str(getattr(call, "adjustment_dropped", "") or "")
    fresh = [reason for reason in reasons if reason not in prior]
    if not fresh:
        return []
    logged = "; ".join(fresh)
    call.adjustment_dropped = f"{prior}; {logged}".strip("; ") if prior else logged
    rationale = str(getattr(call, "rationale", "") or "")
    call.rationale = f"{rationale} | {logged}".strip(" |")
    logging.getLogger("cfb_coach.madden.playcaller").info(logged)
    return fresh


def adjustment_block_reason(
    call: Any,
    *,
    side: str,
    armed: Iterable[str],
    db: Any = None,
    opponent_id: str | None = None,
) -> str | None:
    """None when the call's macro and adjustment may be shown.

    An empty adjustment slot is legal. A draft, invented, unarmed, or
    unverified sheet is not.
    """
    macro = getattr(call, "macro", None)
    if macro and not macro_is_live(
        str(macro), side=side, armed_ids=armed, db=db, opponent_id=opponent_id,
    ):
        return f"invalid_adjustment:macro:{macro}"
    adjustment = getattr(call, "adjustment", None)
    if not isinstance(adjustment, dict) or not adjustment.get("id"):
        return None
    if adjustment.get("executable") is False:
        return f"invalid_adjustment:not_executable:{adjustment.get('id')}"
    buttons = str(adjustment.get("buttons") or "")
    if not buttons.strip() or "VERIFY" in buttons.upper():
        return f"invalid_adjustment:unexecutable:{adjustment.get('id')}"
    return None


def artifact_fingerprint(path: str | None) -> dict[str, Any]:
    """Path, sha256, and training-row count for one artifact file."""
    from pathlib import Path

    if not path:
        return {"path": None, "sha256": None, "training_rows": 0, "exists": False}
    dest = Path(path)
    if not dest.is_file():
        return {"path": str(dest), "sha256": None, "training_rows": 0, "exists": False}
    raw = dest.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows = 0
    try:
        import json

        payload = json.loads(raw.decode("utf-8"))
        if "n_supervised" in payload:
            rows = int(payload.get("n_supervised") or 0)
        elif "supervised_defensive_snaps" in payload:
            rows = int(payload.get("supervised_defensive_snaps") or 0)
    except (OSError, ValueError, TypeError):
        rows = 0
    return {"path": str(dest), "sha256": digest, "training_rows": rows, "exists": True}


def defense_artifact_report(db: Any) -> dict[str, Any]:
    """Read the installed defense artifact. Does not train one."""
    from cfb_coach.madden.model.defense_coordinator import MODEL_META, load_model

    raw_path = None if db is None else db.get_meta(MODEL_META)
    model = load_model(raw_path or None)
    finger = artifact_fingerprint(str(raw_path) if raw_path else None)
    snaps = int((model or {}).get("supervised_defensive_snaps") or 0)
    quality = (model or {}).get("evidence_quality")
    valid = bool(model) and snaps > 0 and finger["exists"]
    return {
        "path": finger["path"],
        "sha256": finger["sha256"],
        "training_rows": snaps,
        "evidence_quality": quality,
        "valid_real_training": valid,
        "activated": False,
    }


def install_primary_lions_books(db: Any) -> dict[str, Any]:
    """Lock the trimmed Lions books prep uses for the Detroit primary."""
    from cfb_coach.madden.playbook import load_books, lock_books, make_trimmed_record

    plans: dict[str, Any] = {}
    for side in ("offense", "defense"):
        record = make_trimmed_record(
            side, "Lions", reason="isolated Detroit Lions primary book",
        )
        plans[side] = {"record": record, "change": "new", "untouched": False}
    lock_books(db, plans, applied=True)
    applied = load_books(db)
    return {
        "offense": list((applied.get("offense") or {}).get("formations") or {}),
        "defense": list((applied.get("defense") or {}).get("formations") or {}),
        "source_book": "Lions",
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    cursor = (len(ordered) - 1) * fraction
    low = int(cursor)
    high = min(low + 1, len(ordered) - 1)
    weight = cursor - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


def latency_summary(values: Iterable[float]) -> dict[str, float]:
    rows = [float(item) for item in values]
    return {
        "p50": round(percentile(rows, 0.50), 3),
        "p95": round(percentile(rows, 0.95), 3),
        "max": round(max(rows) if rows else 0.0, 3),
        "n": float(len(rows)),
    }


def book_contains(book: Mapping[str, Iterable[str]], formation: str, play: str) -> bool:
    plays = list(book.get(formation) or [])
    return play in plays
