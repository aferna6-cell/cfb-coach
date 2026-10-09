"""Offline research-grounded custom macro *variants* for model-selected formations.

This lab creates new *play-concept-specific* adjustment templates from sourced
primitives. It does not invent Madden editor controls, assert compatibility
of untested combinations or arm any in-game macro. Multi-action compositions
are DRAFT-only until independent UI verification and conflict testing.

Lifecycle states:
  DRAFT   — model-generated idea; not executable
  VERIFIED — user checked supported configuration (via offense-design --verify-macro)
  ARMED   — user confirmed in-game slot occupancy
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from cfb_coach.madden import catalog, playbook, research_db
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model.offense_designer import META_BLUEPRINTS

VERSION = "madden.offense.macro_lab.v2"
COMPATIBLE_COMPOSITION_TYPES = frozenset({("hot_route", "pass_protection"), ("pass_protection", "hot_route")})


def _installed_book(db: Any) -> dict[str, Any]:
    active = playbook.load_books(db).get("offense") or {}
    formations = active.get("formations") or {}
    if not formations or not active.get("locked_ts"):
        raise ValueError("A physically installed and confirmed offense is required")
    return active


def _concept_groups(active: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    formations = active.get("formations") or {}
    sources = active.get("formation_sources") or {}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for formation, plays in formations.items():
        for play in plays:
            if catalog.is_run(play) or "rpo" in play.lower():
                continue
            groups[experimental_model._play_concept(play)].append({
                "formation": formation, "play": play,
                "catalog_book": sources.get(formation) or active.get("source_book") or "",
            })
    return groups


def _existing_names(db: Any) -> set[str]:
    try:
        recorded = json.loads(db.get_meta(META_BLUEPRINTS) or "{}")
        return {m.get("name") for m in recorded.get("macro_blueprints") or []}
    except (TypeError, ValueError):
        return set()


def _settings_conflict(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    """Return a human-readable conflict reason, or None if compatible."""
    lt, rt = left.get("type"), right.get("type")
    if (lt, rt) not in COMPATIBLE_COMPOSITION_TYPES and lt == rt == "hot_route":
        # Two hot routes may target the same receiver — reject same target.
        if (left.get("target") or "").upper() == (right.get("target") or "").upper():
            return f"conflicting hot-route target {left.get('target')}"
    if lt == rt == "pass_protection":
        return "cannot stack two pass protections"
    if lt == "hot_route" and rt == "hot_route":
        if (left.get("target") or "").upper() == (right.get("target") or "").upper():
            return f"duplicate hot-route target {left.get('target')}"
    return None


def propose_variants(
    db: Any, *, limit: int = 6, include_compositions: bool = False,
) -> dict[str, Any]:
    if limit < 1 or limit > 16:
        raise ValueError("limit must be 1..16")
    active = _installed_book(db)
    groups = _concept_groups(active)
    existing = _existing_names(db)
    variants: list[dict[str, Any]] = []

    researched = [
        a for a in research_db.offense_adjustments()
        if a.get("type") in ("hot_route", "pass_protection")
        and a.get("sources") and a.get("target") and a.get("route")
    ]

    for action in researched:
        for cover in action.get("vs") or []:
            for concept, pairs in sorted(groups.items(), key=lambda v: (-len(v[1]), v[0])):
                if len(pairs) < 1:
                    continue
                signature = f"{action['id']}|{cover}|{concept}|{active.get('rev')}"
                digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:5].upper()
                name = "ML-" + digest + "-" + concept.upper().replace("_", "-")[:6]
                variants.append({
                    "name": name,
                    "kind": action["type"], "coverage": cover,
                    "source_action_id": action["id"],
                    "source_ids": list(action["sources"]),
                    "source_book_constraint": "exact installed formation/play and play concept",
                    "concept": concept,
                    "status": "DRAFT_NEEDS_IN_GAME_VERIFICATION",
                    "lifecycle": "DRAFT" if name not in existing else "DRAFT_ALREADY_STAGED",
                    "already_staged": name in existing,
                    "fire_when": (
                        f"credible live {cover} look, only with one of the listed "
                        f"{concept} passing plays"
                    ),
                    "base_pairs": pairs,
                    "settings": [{
                        "section": ("Route assignments" if action["type"] == "hot_route"
                                    else "Protection"),
                        "setting": (action["target"] if action["type"] == "hot_route"
                                    else "Protection"),
                        "value": action["route"],
                        "sources": list(action["sources"]),
                        "verification": "research_only",
                    }],
                    "composition": [{
                        "type": action["type"], "id": action["id"],
                        "target": action["target"], "route": action["route"],
                    }],
                    "other_editor_settings": (
                        "ALL unspecified Madden editor fields and exact compatibility "
                        "must be verified in game. This is one researched primitive, "
                        "not proof a custom macro can encode the combination."
                    ),
                    "activation": "NOT_ARMED: needs manual editing, slot and user attestation",
                    "evidence_label": "source_grounded_unvalidated_new_variant",
                })

    compositions: list[dict[str, Any]] = []
    if include_compositions:
        # Pair one hot route with one protection when covers overlap and targets differ.
        hot = [a for a in researched if a["type"] == "hot_route"]
        prot = [a for a in researched if a["type"] == "pass_protection"]
        for h in hot[:6]:
            for p in prot[:4]:
                conflict = _settings_conflict(
                    {"type": "hot_route", "target": h.get("target")},
                    {"type": "pass_protection", "target": p.get("target")},
                )
                if conflict:
                    continue
                covers = sorted(set(h.get("vs") or []) & set(p.get("vs") or []))
                if not covers:
                    # Allow composition when either covers pressure / man broadly.
                    covers = sorted(set(h.get("vs") or []) | set(p.get("vs") or []))[:1]
                if not covers:
                    continue
                cover = covers[0]
                for concept, pairs in sorted(groups.items(), key=lambda v: (-len(v[1]), v[0]))[:4]:
                    signature = f"compose|{h['id']}|{p['id']}|{cover}|{concept}|{active.get('rev')}"
                    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:5].upper()
                    name = "ML-CX-" + digest + "-" + concept.upper().replace("_", "-")[:5]
                    if any(v["name"] == name for v in compositions):
                        continue
                    compositions.append({
                        "name": name,
                        "kind": "composition",
                        "coverage": cover,
                        "source_action_id": f"{h['id']}+{p['id']}",
                        "source_ids": list(dict.fromkeys(list(h["sources"]) + list(p["sources"]))),
                        "source_book_constraint": "exact installed formation/play and play concept",
                        "concept": concept,
                        "status": "DRAFT_NEEDS_IN_GAME_VERIFICATION",
                        "lifecycle": "DRAFT" if name not in existing else "DRAFT_ALREADY_STAGED",
                        "already_staged": name in existing,
                        "fire_when": (
                            f"credible live {cover} look on {concept} concepts; "
                            f"requires verifying BOTH primitives in one Custom Adjustment"
                        ),
                        "base_pairs": pairs,
                        "settings": [
                            {
                                "section": "Route assignments",
                                "setting": h["target"], "value": h["route"],
                                "sources": list(h["sources"]), "verification": "research_only",
                            },
                            {
                                "section": "Protection",
                                "setting": "Protection", "value": p["route"],
                                "sources": list(p["sources"]), "verification": "research_only",
                            },
                        ],
                        "composition": [
                            {"type": "hot_route", "id": h["id"],
                             "target": h["target"], "route": h["route"]},
                            {"type": "pass_protection", "id": p["id"],
                             "target": p["target"], "route": p["route"]},
                        ],
                        "conflict_check": "passed_static_target_and_protection_rules",
                        "other_editor_settings": (
                            "Multi-action composition is UNVERIFIED in the Madden editor. "
                            "User must confirm both settings coexist and the slot arms. "
                            "This is not an on-the-fly manual sequence."
                        ),
                        "activation": "NOT_ARMED: draft composition only",
                        "evidence_label": "source_grounded_unvalidated_composition",
                    })

    proposals = variants + compositions
    return {
        "schema": VERSION, "installed_book": active.get("name"),
        "installed_revision": active.get("rev"),
        "proposals": proposals[:limit],
        "total_candidates": len(proposals),
        "n_single_primitive": len(variants),
        "n_compositions": len(compositions),
        "status": "DRAFT_ONLY",
        "lifecycle_note": (
            "DRAFT until offense-design --verify-macro; ARMED only after in-game "
            "slot attestation. Never confuse a manual executable sequence with a "
            "saved Custom Adjustment macro."
        ),
        "note": (
            "These are newly assembled, concept-targeted candidates built from "
            "individually sourced Madden adjustment primitives; no novel "
            "interaction is claimed tested, and nothing is callable yet."
        ),
    }


def stage_variants(
    db: Any, *, limit: int = 6, include_compositions: bool = False,
) -> dict[str, Any]:
    plan = propose_variants(db, limit=limit, include_compositions=include_compositions)
    raw = db.get_meta(META_BLUEPRINTS)
    try:
        registry = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        registry = {}
    current = list(registry.get("macro_blueprints") or [])
    existing = {m.get("name") for m in current}
    additions = [p for p in plan["proposals"] if p["name"] not in existing]
    if not additions:
        return {**plan, "added": 0, "already_staged": len(current)}
    # No approved or armed loadout state is changed. The existing explicit
    # verify-created-macro process continues to require in-game attestation.
    registry["macro_blueprints"] = current + additions
    registry["macro_lab_schema"] = VERSION
    registry["macro_lab_staged_ts"] = datetime.now(timezone.utc).isoformat()
    db.set_meta(META_BLUEPRINTS, json.dumps(registry, sort_keys=True))
    return {**plan, "added": len(additions), "already_staged": len(current)}


def blueprint_lifecycle(db: Any, opponent_id: str, name: str) -> str:
    """Return DRAFT | VERIFIED | ARMED for a named blueprint."""
    from cfb_coach.madden.model.offense_designer import verified_created_macros

    existing = _existing_names(db)
    if name not in existing:
        # May still be a designer-generated blueprint in the same registry.
        try:
            recorded = json.loads(db.get_meta(META_BLUEPRINTS) or "{}")
            names = {m.get("name") for m in recorded.get("macro_blueprints") or []}
            if name not in names:
                return "UNKNOWN"
        except (TypeError, ValueError):
            return "UNKNOWN"
    verified = {m.get("name") for m in verified_created_macros(db, opponent_id)}
    if name in verified:
        return "ARMED"  # verify_created_macro implies armed-slot attestation
    return "DRAFT"
