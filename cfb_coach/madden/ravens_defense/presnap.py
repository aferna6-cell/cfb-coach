"""Pre-snap adjustment text shown with every live defensive call.

Offense already prints a VARIATION / ADJUSTMENTS block in the compact live UI.
Defense used to show only formation + play there, so 6-1 setup steps (shells,
man-ups, zone strategy, blitz adjustments) and nickel guide checks never
reached Aidan. This module builds a sourced checklist for the card and that
UI. Nothing here is invented: 6-1 lines come from the transcribed PDF
``setup_steps`` already on each even61 call; nickel / match lines come from
existing catalog and guide data and are labeled when they are unverified or
reference-only. Calls with no sourced adjustment use the same stock-play
default the offense catalog already prints.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from cfb_coach.madden.ravens_defense import catalog as rc

STOCK_DEFAULT = "no sourced adjustments listed for this call: stock play"
STOCK_DEFAULT_LINE = f"(no sourced adjustments listed for this call: stock play)"
PDF_SOURCE = "Civil.gg 6-1 Defense PDF / catalog setup_steps"
MATCH_SOURCE = "Civil.gg Match Coverage Guide (not verified in Nickel 2-4 Wide)"
NICKEL_GUIDE_SOURCE = "Civil.gg Nickel 2-4 Wide guide (reference only; not verified for stock)"
VIDEO_RE = re.compile(r"watch.{0,40}video", re.I)
MACRO_RE = re.compile(r"select macro|^call the play\b", re.I)
USER_RE = re.compile(r"^user\b", re.I)

# Nickel-stock base play -> guide play name used in catalog examples / civil entries.
_NICKEL_GUIDE_PLAY = {
    "nickel_cloud": "Cover 3 Cloud",
    "nickel_cover2": "Cover 2 Match",
    "nickel_tampa2": "Tampa 2 Contain",
    "nickel_cover9": "Cover 9 Show 2",
}


def _shape(decision: Mapping[str, Any]) -> str | None:
    user = decision.get("user") or {}
    return user.get("offense_shape") or None


def _is_video(step: str) -> bool:
    return bool(VIDEO_RE.search(step or ""))


def _even61_lines(raw: Mapping[str, Any]) -> tuple[list[str], str]:
    """Verbatim PDF setup steps that are pre-snap adjustments (not video pointers)."""
    lines: list[str] = []
    for step in raw.get("setup_steps") or []:
        text = str(step).strip()
        if not text or _is_video(text):
            continue
        lines.append(text)
    if raw.get("card_note"):
        note = str(raw["card_note"]).strip()
        if note and note not in lines:
            lines.append("Note: " + note)
    return lines, PDF_SOURCE


def _match_lines(cat: rc.RavensCatalog, call_id: str, shape: str | None,
                 zone_it: bool) -> tuple[list[str], str | None]:
    """Look-specific Match-guide checks, labeled unverified. Zone It drops match checks."""
    if zone_it:
        return (["Zone It: match rules off; play the spot-drop coverage from this alignment "
                 "(dataset macro_catalog; USER is not carried over)"], "catalog Zone It macro")
    rules = [r for r in (cat.raw.get("user_assignment_rules") or []) if r.get("call_id") == call_id]
    if not rules:
        return [], None
    lines: list[str] = []
    if shape:
        hits = [r for r in rules if r.get("opponent_shape_key") == shape]
        if hits:
            r = hits[0]
            if r.get("guide_user"):
                lines.append(f"Match guide user vs {shape} (not verified in {rc.NICKEL}): {r['guide_user']}")
            for chk in r.get("guide_checks") or []:
                lines.append(f"Match check vs {shape} (not verified in {rc.NICKEL}): {chk}")
        else:
            lines.append(f"no sourced Match-guide check for this look ({shape})")
    else:
        lines.append("offensive alignment not entered; showing every sourced Match-guide check by look")
        for r in rules:
            sh = r.get("opponent_shape_key")
            checks = ", ".join(r.get("guide_checks") or []) or "none listed"
            user = r.get("guide_user") or "none listed"
            lines.append(f"if {sh}: user {user}; check {checks} (not verified in {rc.NICKEL})")
    return lines, MATCH_SOURCE


def _nickel_guide_lines(cat: rc.RavensCatalog, call_id: str, shape: str | None) -> tuple[list[str], str | None]:
    """Shape-specific extra setup from the Nickel guide, labeled reference-only."""
    play = _NICKEL_GUIDE_PLAY.get(call_id)
    if not play:
        return [], None
    examples = [e for e in (cat.raw.get("nickel_guide_examples_reference_only") or [])
                if e.get("play") == play]
    if not examples:
        return [], None
    lines: list[str] = []

    def _fmt(ex: Mapping[str, Any], prefix: str) -> None:
        if ex.get("guide_user"):
            lines.append(f"{prefix}guide user: {ex['guide_user']}")
        for step in ex.get("extra_setup_required") or []:
            lines.append(f"{prefix}{step}")

    if shape:
        # inside_10 is the red-zone Tampa 2 example's shape key
        keys = {shape}
        if shape in {"inside_10", "red_zone"}:
            keys.update({"inside_10", "red_zone"})
        hits = [e for e in examples if e.get("offense_shape") in keys]
        if hits:
            _fmt(hits[0], "Nickel guide (reference only; not verified for stock): ")
        else:
            # No invented default for an unmatched look; list the sourced looks instead.
            lines.append(f"no Nickel-guide extra setup sourced for this look ({shape}); "
                         "guide examples by look (reference only)")
            for ex in examples:
                steps = "; ".join(ex.get("extra_setup_required") or []) or "none listed"
                lines.append(f"if {ex.get('offense_shape')}: {steps}")
    else:
        lines.append("offensive alignment not entered; Nickel-guide extra setup by look "
                     "(reference only; not verified for stock)")
        for ex in examples:
            steps = "; ".join(ex.get("extra_setup_required") or []) or "none listed"
            user = f" user {ex['guide_user']};" if ex.get("guide_user") else ""
            lines.append(f"if {ex.get('offense_shape')}:{user} {steps}")
    return lines, NICKEL_GUIDE_SOURCE


def collect(decision: Mapping[str, Any], cat: rc.RavensCatalog | None = None) -> dict[str, Any]:
    """Sourced pre-snap adjustment payload for a defensive CALL (or empty on no-call)."""
    if decision.get("status") != "CALL":
        return {"lines": [], "source": None, "kind": "none", "shown": False, "flagged": False}
    c = decision.get("candidate") or {}
    raw = decision.get("candidate_raw") or {}
    pkg = c.get("package")
    call_id = c.get("call_id") or ""
    zone_it = bool(c.get("optional_macro") == "ZONE_IT" or decision.get("zone_it_selected"))
    shape = _shape(decision)
    if cat is None:
        try:
            cat = rc.load_61_defense()
        except Exception:  # noqa: BLE001 — display must not change the call
            cat = None

    kind = "stock_default"
    source: str | None = None
    lines: list[str] = []
    flagged = False
    if pkg == "even_61":
        lines, source = _even61_lines(raw)
        kind = "even61_pdf"
    elif pkg == "match_coverage" and cat is not None:
        lines, source = _match_lines(cat, call_id, shape, zone_it)
        kind = "match_guide"
    elif pkg == "nickel_stock" and cat is not None:
        lines, source = _nickel_guide_lines(cat, call_id, shape)
        kind = "nickel_guide_reference"
    elif pkg == "spot_drop_fallback":
        kind = "stock_default"
    elif pkg == "nickel_single_mug":
        # Not live, but if a rollback catalog is loaded, keep the PDF setup_steps.
        lines, source = _even61_lines(raw) if raw.get("setup_steps") else ([], None)
        kind = "nsm_pdf" if lines else "stock_default"
    elif zone_it:
        lines, source = (["Zone It: match rules off; play the spot-drop coverage from this alignment "
                          "(dataset macro_catalog)"], "catalog Zone It macro")
        kind = "zone_it"

    if not lines:
        lines = [STOCK_DEFAULT_LINE]
        source = source or "generic stock-play default (same wording as the offense catalog)"
        kind = "stock_default"
        flagged = pkg not in {"spot_drop_fallback", "dime_blitz"} and not (
            pkg == "nickel_stock" and call_id not in _NICKEL_GUIDE_PLAY
        )
        # Cover 4 Zone and any other call with no guide extras are expected stock defaults, not gaps.
        if pkg == "spot_drop_fallback" or call_id in {"stock_cover4_zone", "dime_dbfire2"}:
            flagged = False

    return {
        "lines": lines,
        "source": source,
        "kind": kind,
        "shown": True,
        "flagged": flagged,
        "call_id": call_id,
        "package": pkg,
    }


def card_lines(payload: Mapping[str, Any]) -> list[str]:
    """Lines appended to the live PLAY / USER / MACRO card."""
    if not payload.get("shown"):
        return []
    out = ["ADJUSTMENTS:"]
    for i, line in enumerate(payload.get("lines") or [], 1):
        out.append(f"  {i}. {line}")
    src = payload.get("source")
    if src:
        out.append("ADJ SOURCE: " + str(src))
    if payload.get("flagged"):
        out.append("ADJ FLAG: no sourced pre-snap adjustment for this call")
    return out


def variation_label(decision: Mapping[str, Any]) -> str | None:
    """Short label next to formation — play in the compact live UI."""
    if decision.get("status") != "CALL":
        return None
    raw = decision.get("candidate_raw") or {}
    if raw.get("situation_tab"):
        return str(raw["situation_tab"])
    c = decision.get("candidate") or {}
    if c.get("package") == "even_61":
        return "4-3 Even 6-1"
    if c.get("optional_macro") == "ZONE_IT":
        return "Zone It"
    return c.get("package") or "civil defense"


def variation_block(decision: Mapping[str, Any], payload: Mapping[str, Any] | None = None) -> str:
    """Compact live-UI block: USER, MACRO, situation, and the adjustment checklist."""
    if decision.get("status") != "CALL":
        return ""
    payload = payload or decision.get("presnap") or collect(decision)
    c = decision.get("candidate") or {}
    raw = decision.get("candidate_raw") or {}
    user = decision.get("user") or {}
    lines: list[str] = []
    user_txt = user.get("user") if user.get("status") == "LIVE_VALID" else "UNVERIFIED"
    if raw.get("suggested_user") and user.get("status") != "LIVE_VALID":
        user_txt = f"{user_txt} (guide, unverified: {raw['suggested_user']})"
    lines.append("USER: " + str(user_txt))
    if decision.get("macro_display"):
        lines.append("MACRO: " + str(decision["macro_display"]))
    elif raw.get("titled_macros"):
        names = raw["titled_macros"]
        if isinstance(names, (list, tuple)):
            lines.append("MACRO: " + " or ".join(str(n) for n in names) + " (OR choice, from the PDF)")
    if c.get("package") == "even_61":
        lines.append("PACKAGE: 4-3 Even 6-1")
        if raw.get("situation_tab"):
            lines.append("SITUATION: " + str(raw["situation_tab"]))
    lines.append("ADJUSTMENTS:")
    for i, line in enumerate(payload.get("lines") or [STOCK_DEFAULT_LINE], 1):
        lines.append(f"  {i}. {line}")
    if payload.get("source"):
        lines.append("ADJ SOURCE: " + str(payload["source"]))
    if payload.get("flagged"):
        lines.append("ADJ FLAG: no sourced pre-snap adjustment for this call")
    return "\n".join(lines)


def attach(decision: dict[str, Any], cat: rc.RavensCatalog | None = None) -> dict[str, Any]:
    """Add presnap / adjustment_lines onto a decide() result (in place) and return it."""
    payload = collect(decision, cat)
    decision["presnap"] = payload
    decision["adjustment_lines"] = list(payload.get("lines") or [])
    decision["adjustment_source"] = payload.get("source")
    decision["adjustments_shown"] = bool(payload.get("shown") and payload.get("lines"))
    return decision
