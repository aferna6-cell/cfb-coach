"""Offline research-grounded custom macro *variants* for model-selected formations.

This lab creates new *play-concept-specific* adjustment templates from sourced
primitives. It does not invent Madden editor controls, assert compatibility
of untested combinations or arm any in-game macro. Future multi-action
compositions require independent UI verification (and conflict testing).
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

VERSION = "madden.offense.macro_lab.v1"


def propose_variants(db: Any, *, limit: int = 6) -> dict[str, Any]:
    if limit < 1 or limit > 16:
        raise ValueError("limit must be 1..16")
    active = playbook.load_books(db).get("offense") or {}
    formations = active.get("formations") or {}
    sources = active.get("formation_sources") or {}
    if not formations or not active.get("locked_ts"):
        raise ValueError("A physically installed and confirmed offense is required")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for formation, plays in formations.items():
        for play in plays:
            if catalog.is_run(play) or "rpo" in play.lower():
                continue
            groups[experimental_model._play_concept(play)].append({
                "formation": formation, "play": play,
                "catalog_book": sources.get(formation) or active.get("source_book") or "",
            })

    existing = set()
    try:
        recorded = json.loads(db.get_meta(META_BLUEPRINTS) or "{}")
        existing = {m.get("name") for m in recorded.get("macro_blueprints") or []}
    except (TypeError, ValueError):
        pass
    variants: list[dict[str, Any]] = []
    for researched in research_db.offense_adjustments():
        if researched.get("type") not in ("hot_route", "pass_protection"):
            continue
        if not researched.get("sources") or not researched.get("target") or not researched.get("route"):
            continue
        for cover in researched.get("vs") or []:
            for concept, pairs in sorted(groups.items(), key=lambda v: (-len(v[1]), v[0])):
                if len(pairs) < 1:
                    continue
                signature = f"{researched['id']}|{cover}|{concept}|{active.get('rev')}"
                digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:5].upper()
                name = "ML-" + digest + "-" + concept.upper().replace("_", "-")[:6]
                if name in existing:
                    continue
                variants.append({
                    "name": name,
                    "kind": researched["type"], "coverage": cover,
                    "source_action_id": researched["id"],
                    "source_ids": list(researched["sources"]),
                    "source_book_constraint": "exact installed formation/play and play concept",
                    "concept": concept, "status": "DRAFT_NEEDS_IN_GAME_VERIFICATION",
                    "fire_when": (
                        f"credible live {cover} look, only with one of the listed "
                        f"{concept} passing plays"
                    ),
                    "base_pairs": pairs,
                    "settings": [{
                        "section": ("Route assignments" if researched["type"] == "hot_route"
                                    else "Protection"),
                        "setting": (researched["target"] if researched["type"] == "hot_route"
                                    else "Protection"),
                        "value": researched["route"],
                        "sources": list(researched["sources"]),
                        "verification": "research_only",
                    }],
                    "other_editor_settings": (
                        "ALL unspecified Madden editor fields and exact compatibility "
                        "must be verified in game. This is one researched primitive, "
                        "not proof a custom macro can encode the combination."
                    ),
                    "activation": "NOT_ARMED: needs manual editing, slot and user attestation",
                    "evidence_label": "source_grounded_unvalidated_new_variant",
                })
    return {
        "schema": VERSION, "installed_book": active.get("name"),
        "installed_revision": active.get("rev"),
        "proposals": variants[:limit],
        "total_candidates": len(variants),
        "status": "DRAFT_ONLY",
        "note": (
            "These are newly assembled, concept-targeted candidates built from "
            "individually sourced Madden adjustment primitives; no novel "
            "interaction is claimed tested, and nothing is callable yet."
        ),
    }


def stage_variants(db: Any, *, limit: int = 6) -> dict[str, Any]:
    plan = propose_variants(db, limit=limit)
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
