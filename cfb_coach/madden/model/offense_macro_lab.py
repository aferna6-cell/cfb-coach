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


def conflicts(parts: list[dict[str, Any]]) -> str | None:
    """Return a reason when researched primitives cannot be one macro.

    Audibles change the play, so they are not composed with hot routes or
    protections. Two protections, or two hot routes on the same target, are
    rejected. Max protect keeps the back in, so it conflicts with an HB hot
    route. These are compatibility rules over sourced primitives, not new
    editor settings.
    """
    if len(parts) < 2:
        return None
    kinds = [str(p.get("type") or p.get("kind") or "") for p in parts]
    if kinds.count("pass_protection") > 1:
        return "two pass protections cannot share one snap"
    if "audible" in kinds:
        return "an audible replaces the play and is not composed with other adjustments"
    targets: list[str] = []
    for part in parts:
        if str(part.get("type") or part.get("kind")) != "hot_route":
            continue
        target = str(part.get("target") or "").strip().upper()
        if not target:
            return "hot route is missing its researched target"
        if target in targets:
            return f"two hot routes assign {target}"
        targets.append(target)
    for part in parts:
        text = f"{part.get('route') or ''} {part.get('label') or ''}".lower()
        if str(part.get("type") or part.get("kind")) == "pass_protection" and "max protect" in text:
            if "HB" in targets:
                return "max protect conflicts with an HB hot route"
    looks = [set(part.get("vs") or []) for part in parts]
    if looks and not set.intersection(*looks):
        return "primitives do not share a defensive look"
    return None


def compose_primitives(
    primitives: list[dict[str, Any]],
    *,
    formations: dict[str, list[str]] | None = None,
    sources: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build draft multi-action blueprints from primitives that pass conflicts().

    Every result is DRAFT. Nothing here is armed or executable.
    """
    from cfb_coach.madden.catalog import is_run

    usable = [
        p for p in primitives
        if p.get("type") in ("hot_route", "pass_protection")
        and p.get("sources") and p.get("id")
        and (p.get("target") or p.get("route"))
    ]
    drafts: list[dict[str, Any]] = []
    pairs = []
    for formation, plays in (formations or {}).items():
        for play in plays:
            if is_run(play) or "rpo" in play.lower():
                continue
            pairs.append({
                "formation": formation, "play": play,
                "catalog_book": (sources or {}).get(formation, ""),
            })
    for i, left in enumerate(usable):
        for right in usable[i + 1:]:
            parts = [left, right]
            reason = conflicts(parts)
            if reason:
                continue
            shared = sorted(set(left.get("vs") or []) & set(right.get("vs") or []))
            if not shared:
                continue
            signature = "|".join(sorted(str(p["id"]) for p in parts)) + "|" + ",".join(shared)
            digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:5].upper()
            name = "ML-COMP-" + digest
            settings = []
            source_ids: list[str] = []
            for part in parts:
                kind = part["type"]
                settings.append({
                    "section": "Route assignments" if kind == "hot_route" else "Protection",
                    "setting": part.get("target") if kind == "hot_route" else "Protection",
                    "value": part.get("route"),
                    "sources": list(part["sources"]),
                    "verification": "research_only",
                })
                for source_id in part["sources"]:
                    if source_id not in source_ids:
                        source_ids.append(source_id)
            drafts.append({
                "name": name,
                "kind": "composition",
                "coverage": shared[0],
                "source_action_id": left["id"],
                "source_action_ids": [left["id"], right["id"]],
                "source_ids": source_ids,
                "source_book_constraint": "exact installed formation/play; both primitives must be editor-verified together",
                "status": "DRAFT_NEEDS_IN_GAME_VERIFICATION",
                "state": "draft",
                "fire_when": (
                    f"credible live {shared[0]} look, and only after both researched "
                    "actions are confirmed compatible in the Madden editor"
                ),
                "base_pairs": pairs,
                "settings": settings,
                "conflict_check": "passed",
                "other_editor_settings": (
                    "Combination is not a saved Custom Adjustment until you verify "
                    "every editor row. On-the-fly button sequences are not a macro slot."
                ),
                "activation": "NOT_ARMED: draft composition; needs manual editing, slot and user attestation",
                "evidence_label": "source_grounded_unvalidated_composition",
            })
    return drafts


def compose_drafts(
    formations: dict[str, list[str]],
    sources: dict[str, str],
    *,
    primitives: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Compositions from the research DB, or from an explicit primitive list."""
    rows = list(primitives) if primitives is not None else research_db.offense_adjustments()
    return compose_primitives(rows, formations=formations, sources=sources)


def recommend_blueprint_fate(
    blueprint: Mapping[str, Any],
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Retain, revise or retire a blueprint from verified evidence only.

    A draft is never retired into an armed slot, and one bad game is not
    enough to discard a concept. This recommendation does not edit Madden.
    """
    state = "armed" if blueprint.get("verified_armed") else (
        "verified" if blueprint.get("status") == "VERIFIED" else "draft"
    )
    if state == "draft":
        return {"name": blueprint.get("name"), "state": "draft", "recommendation": "retain",
                "reason": "still a draft; install and verify before judging results"}
    if not artifact or artifact.get("mode") != "bounded_active":
        return {"name": blueprint.get("name"), "state": state, "recommendation": "retain",
                "reason": "no promoted multi-game comparison yet"}
    keys = [
        row for row in (artifact.get("comparisons") or {}).values()
        if str(blueprint.get("name")) in str(row.get("action") or "")
        and row.get("ready_for_bounded_adjustment")
    ]
    if not keys:
        return {"name": blueprint.get("name"), "state": state, "recommendation": "retain",
                "reason": "verified sample does not yet meet the action-learning gate"}
    shift = float(keys[0].get("bounded_selection_shift") or 0.0)
    if shift < 0:
        return {"name": blueprint.get("name"), "state": state, "recommendation": "retire",
                "reason": "promoted observational association is negative; confounding remains"}
    if shift == 0:
        return {"name": blueprint.get("name"), "state": state, "recommendation": "revise",
                "reason": "no positive association after shrinkage; consider a different sourced primitive"}
    return {"name": blueprint.get("name"), "state": state, "recommendation": "retain",
            "reason": "promoted association is non-negative; still not a causal claim"}


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



def composed_macro_report(db: Any, *, limit: int = 10) -> dict[str, Any]:
    """Propose NEW sourced two-primitive offensive custom macro combinations."""
    if not 1 <= limit <= 30:
        raise ValueError("limit must be 1..30")
    book = playbook.load_books(db).get("offense") or {}
    formations = book.get("formations") or {}
    if not formations or not book.get("locked_ts"):
        raise ValueError("A confirmed installed offensive playbook is required")
    drafts = compose_drafts(
        formations, book.get("formation_sources") or {},
    )
    return {
        "schema": "madden.offense.macro_composition.v2",
        "status": "DRAFT_ONLY",
        "total_candidates": len(drafts),
        "proposals": drafts[:limit],
        "installed": False,
        "note": (
            "Original researched two-action compositions; editor support and "
            "action effects unverified. Do not activate before physical testing."
        ),
    }


def stage_compositions(db: Any, *, limit: int = 10) -> dict[str, Any]:
    """Save drafts without altering current loadout or verified approvals."""
    report = composed_macro_report(db, limit=limit)
    try:
        state = json.loads(db.get_meta(META_BLUEPRINTS) or "{}")
    except (TypeError, ValueError):
        state = {}
    existing = list(state.get("macro_blueprints") or [])
    names = {m.get("name") for m in existing}
    additions = [m for m in report["proposals"] if m["name"] not in names]
    if additions:
        state["macro_blueprints"] = existing + additions
        state["macro_composition_schema"] = report["schema"]
        db.set_meta(META_BLUEPRINTS, json.dumps(state, sort_keys=True))
    return {
        **report, "added": len(additions), "already_staged": len(existing),
        "live_activation": False,
    }
