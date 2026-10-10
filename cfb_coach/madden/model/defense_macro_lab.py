"""Research-grounded, model-generated Madden defensive Custom Adjustment drafts.

Composes settings from TWO independent existing researched macro recipes, fills
all editor fields with explicit values/Default, checks field conflicts and
playbook compatibility. NEVER invents editor rows, executable button combos,
or automatically arms Madden's Custom Adjustments.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from cfb_coach.madden import playbook, research_db

DRAFT_KEY = "ml_defense_generated_macro_drafts.v1"
APPROVED_KEY = "ml_defense_approved_macros.v1:{opponent}"
SCHEMA = "madden.defense.macro_blueprint.v1"


def _fields() -> list[tuple[str, str]]:
    return [
        (section, name)
        for section, names in (research_db.editor_fields().get("defense") or {}).items()
        for name in names
    ]


def _source_settings(recipe: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    fields = {(s.casefold(), n.casefold()): (s, n) for s, n in _fields()}
    aliases = research_db.editor_fields().get("aliases") or {}
    known_sources = set(research_db.sources())
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in recipe.get("settings") or []:
        section = str(raw.get("section") or "")
        setting = str(aliases.get(raw.get("setting"), raw.get("setting") or ""))
        value = str(raw.get("value") or "")
        source = str(raw.get("source") or "")
        key = (section.casefold(), setting.casefold())
        if key not in fields or not value or value == "Default" or source not in known_sources:
            continue
        canonical = fields[key]
        entry = {
            "section": canonical[0], "setting": canonical[1],
            "value": value, "source": source,
        }
        if key in chosen and chosen[key]["value"] != value:
            return {}
        chosen[key] = entry
    return chosen


def _installed_pairs(
    recipe: Mapping[str, Any], formations: Mapping[str, Sequence[str]]
) -> list[dict[str, str]]:
    base = recipe.get("base") or {}
    base_play = str(base.get("play") or "")
    # An "any"/generic base play is NOT proof the editor supports a chosen call.
    if not base_play or base_play.casefold() == "any" or "/" in base_play:
        return []
    return [
        {"formation": f, "play": base_play}
        for f, plays in formations.items() if base_play in plays
    ]


def compose_defensive_macros(
    formations: Mapping[str, Sequence[str]],
    *,
    recipes: Sequence[Mapping[str, Any]] | None = None,
    max_drafts: int = 12,
) -> list[dict[str, Any]]:
    """Build source-backed original variants; no active macro changes."""
    if max_drafts < 0:
        raise ValueError("max_drafts must be nonnegative")
    data = list(recipes if recipes is not None else research_db.defense_macros())
    fields = _fields()
    output: list[dict[str, Any]] = []
    seen = set()
    for lead in data:
        pairs = _installed_pairs(lead, formations)
        left = _source_settings(lead)
        if not pairs or not left:
            continue
        for donor in data:
            if donor.get("id") == lead.get("id"):
                continue
            right = _source_settings(donor)
            if not right:
                continue
            shared = sorted(set(lead.get("answers") or []) & set(donor.get("answers") or []))
            if not shared:
                continue
            # Conflicting field values must NOT be silently overwritten.
            if any(key in left and left[key]["value"] != value["value"]
                   for key, value in right.items()):
                continue
            added = {key: value for key, value in right.items() if key not in left}
            if not added:
                continue
            merged = dict(left)
            merged.update(added)
            name_seed = "|".join([
                str(lead.get("id")), str(donor.get("id")),
                ",".join(f"{key}:{x['value']}" for key, x in sorted(merged.items())),
            ])
            digest = hashlib.sha256(name_seed.encode()).hexdigest()[:7].upper()
            name = "ML-D-" + digest
            if name in seen:
                continue
            seen.add(name)
            settings = [
                dict(merged.get((s.casefold(), n.casefold()), {
                    "section": s, "setting": n, "value": "Default", "source": "default",
                }))
                for s, n in fields
            ]
            output.append({
                "schema": SCHEMA, "name": name,
                "state": "draft", "verified_armed": False,
                "parent_recipes": [str(lead["id"]), str(donor["id"])],
                "coverage": shared[0], "answers": shared,
                "base_pairs": pairs, "settings": settings,
                "n_resourced_settings": len(merged),
                "source_ids": sorted({r["source"] for r in merged.values()}),
                "compatibility": "UNVERIFIED: valid source fields do not prove combined gameplay effects",
                "status": "DRAFT_NEEDS_IN_GAME_VERIFICATION",
                "user_steps": (
                    "Create in Madden Custom Adjustments > Defense, confirm EVERY "
                    "editor row, test with the stated base play, arm one of the eight "
                    "defense slots, then explicitly verify in CLI"
                ),
                "live_eligible": False,
            })
            if len(output) >= max_drafts:
                return output
    return output


def _get_json(db: Any, key: str, fallback: Any) -> Any:
    try:
        return json.loads(db.get_meta(key) or "")
    except (ValueError, TypeError):
        return fallback


def stage_drafts(db: Any, *, max_drafts: int = 12) -> dict[str, Any]:
    book = (playbook.load_books(db).get("defense") or {}).get("formations") or {}
    if not book:
        raise ValueError("Install/confirm a defensive playbook before drafting macros")
    rows = compose_defensive_macros(book, max_drafts=max_drafts)
    data = {
        "schema": SCHEMA, "status": "draft_only",
        "book_fingerprint": hashlib.sha256(json.dumps(book, sort_keys=True).encode()).hexdigest(),
        "drafts": rows, "live_activation": False,
    }
    db.set_meta(DRAFT_KEY, json.dumps(data, sort_keys=True))
    return {"staged": len(rows), "names": [r["name"] for r in rows],
            "armed": False, "explicit_user_verification_required": True}


def draft_report(db: Any) -> dict[str, Any]:
    return _get_json(db, DRAFT_KEY, {"schema": SCHEMA, "drafts": [], "status": "none"})


def approved_macros(db: Any, opponent_id: str) -> list[dict[str, Any]]:
    return list(_get_json(db, APPROVED_KEY.format(opponent=opponent_id), {"macros": []}).get("macros") or [])


def verify_macro(
    db: Any, opponent_id: str, name: str, attestation: str, *,
    retire_existing: str | None = None,
) -> dict[str, Any]:
    """Verify a physically created and slotted macro; never silently arm."""
    from cfb_coach.madden import macros
    if len((attestation or "").strip()) < 30:
        raise ValueError("Explicit verification of every editor row and the armed game slot required")
    pending = draft_report(db)
    book = (playbook.load_books(db).get("defense") or {}).get("formations") or {}
    actual_hash = hashlib.sha256(json.dumps(book, sort_keys=True).encode()).hexdigest()
    if pending.get("book_fingerprint") != actual_hash:
        raise ValueError("Confirmed defensive book changed; re-stage the drafts")
    row = next((r for r in pending.get("drafts") or [] if r.get("name") == name), None)
    if row is None:
        raise ValueError("No draft under this exact name")
    if not any(p.get("play") in book.get(p.get("formation"), [])
               for p in row["base_pairs"]):
        raise ValueError("Draft base play is no longer installed")
    if len(row.get("settings") or []) != len(_fields()):
        raise ValueError("Incomplete Custom Adjustment editor settings")
    current = approved_macros(db, opponent_id)
    if any(m.get("name") == name for m in current):
        return {"name": name, "already_verified": True}
    existing = macros.load_selection(db, opponent_id) or {"offense": [], "defense": []}
    armed = list(existing.get("defense") or [])
    if retire_existing and retire_existing not in armed:
        raise ValueError("Retired macro must be in the existing defensive loadout")
    if len(current) + len(armed) - int(bool(retire_existing)) >= macros.LOADOUT_N:
        raise ValueError("No verified empty defensive macro slot; explicitly retire an existing slot")
    # Never manipulate actual Madden game controls or assume activation from code.
    approved = dict(row, verified_armed=True, state="verified",
                    opponent_id=opponent_id, attestation=attestation.strip(),
                    verified_ts=datetime.now(timezone.utc).isoformat())
    if retire_existing:
        from cfb_coach.madden.macros import ACTIVE_META_KEY
        key = ACTIVE_META_KEY.format(opp=opponent_id)
        raw = _get_json(db, key, {"schema": 2, "offense": [], "defense": []})
        raw["defense"] = [x for x in raw.get("defense") or [] if x != retire_existing]
        db.set_meta(key, json.dumps(raw, sort_keys=True))
    current.append(approved)
    db.set_meta(APPROVED_KEY.format(opponent=opponent_id),
                json.dumps({"macros": current, "schema": SCHEMA}, sort_keys=True))
    return {"name": name, "verified_armed": True, "source_ids": approved["source_ids"]}


def unverify_macro(db: Any, opponent_id: str, name: str) -> dict[str, Any]:
    current = approved_macros(db, opponent_id)
    remaining = [m for m in current if m.get("name") != name]
    db.set_meta(APPROVED_KEY.format(opponent=opponent_id),
                json.dumps({"macros": remaining, "schema": SCHEMA}, sort_keys=True))
    return {"removed": len(remaining) != len(current), "name": name}


def compatible_verified_macros(
    db: Any, opponent_id: str, formation: str, play: str, observed_concept_family: str
) -> list[dict[str, Any]]:
    book = (playbook.load_books(db).get("defense") or {}).get("formations") or {}
    if play not in book.get(formation, []):
        return []
    return [
        r for r in approved_macros(db, opponent_id)
        if r.get("verified_armed") is True
        and observed_concept_family in (r.get("answers") or [])
        and any(p.get("formation") == formation and p.get("play") == play
                for p in r.get("base_pairs") or [])
        and len(r.get("settings") or []) == len(_fields())
    ]
