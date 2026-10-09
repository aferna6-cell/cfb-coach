"""Read-only Madden offensive action readiness across the confirmed playbook.

Selected loadout is not proof a macro was physically armed in the game;
generated macros must be verified separately. This report makes missing
settings and missing compatible formation/plays visible before kickoff.
"""
from __future__ import annotations

import json
from typing import Any

from cfb_coach.madden import playbook, research_db
from cfb_coach.madden.macros import ACTIVE_META_KEY
from cfb_coach.madden.offense_macros import (
    clean_ids, offense_detail, pairs_in_book,
)
from cfb_coach.madden.model.offense_designer import (
    META_BLUEPRINTS, verified_created_macros, verified_macro_configurations,
)


def offense_actions_report(db: Any, opponent_id: str) -> dict[str, Any]:
    installed = playbook.load_books(db).get("offense") or {}
    book = installed.get("formations") or {}
    raw = db.get_meta(ACTIVE_META_KEY.format(opp=opponent_id)) or "{}"
    try:
        stored = json.loads(raw)
    except (TypeError, ValueError):
        stored = {}
    armed_prep_ids = clean_ids(
        (stored.get("offense") or []) if stored.get("schema") == 2 else []
    )
    macros = []
    for mid in armed_prep_ids:
        detail = offense_detail(mid, book)
        compatible = pairs_in_book(
            mid, book, cap=1 + sum(len(plays) for plays in book.values())
        )
        ready = bool(
            installed and compatible and detail.get("settings")
            and not detail.get("needs_settings")
            and not detail.get("gaps")
        )
        macros.append({
            "name": mid,
            "selected_in_saved_loadout": True,
            "editor_confirmed_armed": False,
            "source_type": "saved_user_notes",
            "supported_pairs": len(compatible),
            "settings_count": len(detail.get("settings") or []),
            "settings_source": detail.get("settings_source") or "",
            "settings_gaps": detail.get("gaps") or [],
            "settings_complete": bool(detail.get("settings")) and not detail.get("gaps"),
            "eligible_if_editor_armed_and_triggered": ready,
            "blockers": (
                (["no installed compatible play"] if not compatible else [])
                + (["missing editor settings"] if detail.get("gaps") or not detail.get("settings") else [])
            ),
            "trigger": detail.get("fire_when") or "",
        })
    generated = verified_created_macros(db, opponent_id)
    configurations = verified_macro_configurations(db)
    approved = []
    for item in generated:
        compatible = [
            pair for pair in item.get("base_pairs") or []
            if pair.get("play") in book.get(pair.get("formation"), [])
        ]
        approved.append({
            "name": item["name"], "editor_confirmed_armed": True,
            "source_type": "model_created_verified",
            "supported_pairs": len(compatible),
            "source_ids": item.get("source_ids") or [],
            "settings_count": len(item.get("settings") or []),
            "eligible_if_triggered": bool(compatible and item.get("settings")),
            "trigger": item.get("fire_when") or "",
        })
    try:
        blueprints = json.loads(db.get_meta(META_BLUEPRINTS) or "{}")
    except (TypeError, ValueError):
        blueprints = {}
    drafts = [str(x.get("name")) for x in blueprints.get("macro_blueprints") or []
              if x.get("name")
              and x.get("name") not in {p["name"] for p in approved}
              and x.get("name") not in {p["name"] for p in configurations}]
    return {
        "opponent_id": opponent_id,
        "applied_book": installed.get("name") or "none",
        "playbook_installed": bool(installed),
        "installed_formations": len(book),
        "installed_plays": sum(len(p) for p in book.values()),
        "saved_loadout_macros": len(macros),
        "ready_in_saved_loadout": sum(int(m["eligible_if_editor_armed_and_triggered"]) for m in macros),
        "verified_model_created_macros": len(approved),
        "verified_unarmed_macro_configurations": len(configurations),
        "generated_draft_macros": len(drafts),
        "existing_macros": macros,
        "verified_generated_macros": approved,
        "verified_unarmed_configurations": configurations,
        "unverified_generated_drafts": drafts,
        "no_action_available": True,
        "readiness_note": (
            "A saved loadout is not proof a macro was armed in Madden. "
            "A macro or hot route is offered live only when its formation/play "
            "and situational trigger match, controls/settings are sufficiently "
            "grounded, and its action score beats making no adjustment."
        ),
        "source_count": len(research_db.sources()),
    }
