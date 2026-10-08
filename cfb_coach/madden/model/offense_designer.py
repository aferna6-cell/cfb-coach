"""Offline ML offensive playbook and Custom Adjustment designer.

The model may replace formations and individual plays from catalogued Madden
books, and invent *draft configurations* from sourced adjustment primitives.
No recommendation is ever callable until the user builds/validates the custom
playbook and explicitly confirms its proposal ID. All macros remain drafts.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from cfb_coach.madden import catalog, playbook, research_db
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model.experimental_live import resolve_artifact_path

META_PENDING = "ml_offense_design_pending.v1"
META_HISTORY = "ml_offense_design_history.v1"
META_BLUEPRINTS = "ml_offense_created_blueprints.v1"
META_APPROVED = "ml_offense_verified_macros.v1:{opponent}"
MAX_FORMATIONS = 5
MAX_PLAYS_PER_FORMATION = 10


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:16]


def _identity(proposal: Mapping[str, Any]) -> str:
    return _digest({
        "expected_applied_hash": proposal["expected_applied_hash"],
        "book": proposal["book"],
        "macro_blueprints": proposal["macro_blueprints"],
        "opponent_id": proposal["opponent_id"],
    })


def _load_artifact(db: Any) -> experimental_model.ExperimentalArtifact:
    path = resolve_artifact_path(db)
    if path:
        return experimental_model.load_artifact(path)
    # A preview can still be drawn before training, but is clearly labeled as
    # prior-only, not an automatically trained/supervised model.
    return experimental_model.ExperimentalArtifact(
        evidence_quality="prior_driven", note="No installed artifact; prior-only preview"
    )


def _rate(art: Any, form: str, play: str, opponent: str) -> float:
    """Use model success estimates, not the heuristic's book/play ordering."""
    from cfb_coach.opponents import is_cpu_opponent

    # Multiple generic pre-snap situations; no post-snap coverage is assumed.
    checks = ((1, 10, 35), (2, 6, 50), (3, 7, 35))
    vals = [
        experimental_model.predict_success(
            art, formation=form, play=play, down=down, distance=distance,
            yardline=yardline, opponent_id=opponent,
            opponent_type="cpu" if is_cpu_opponent(opponent) else "human",
            coverage_hint=None, coverage_source="none",
            heuristic_bonus=0.0,
        )["probability"]
        for down, distance, yardline in checks
    ]
    return sum(vals) / len(vals)


def _play_rankings(
    art: Any, book: str, form: str, plays: list[str],
    opponent: str, current: Mapping[str, Any],
) -> list[dict[str, Any]]:
    old = (current.get("formations") or {}).get(form) or []
    ranked: list[dict[str, Any]] = []
    for play in dict.fromkeys(plays):
        if not catalog.zone_fit(play, "open"):
            continue
        is_run = catalog.is_run(play)
        concept = experimental_model._play_concept(play)
        score = _rate(art, form, play, opponent)
        # A small continuity preference, never a hard lock; the model may swap.
        if play in old:
            score += 0.012
        ranked.append({
            "formation": form, "play": play, "score": round(score, 6),
            "concept": concept, "is_run": is_run, "catalog_book": book,
        })
    ranked.sort(key=lambda x: (-x["score"], x["play"]))
    return ranked


def _trim_plays(rows: list[dict[str, Any]], cap: int) -> list[str]:
    """Retain multiple concepts and a run/quick option rather than ten verts."""
    chosen: list[dict[str, Any]] = []
    concepts: set[str] = set()
    for r in rows:
        if len(chosen) >= cap:
            break
        if r["concept"] not in concepts:
            chosen.append(r)
            concepts.add(r["concept"])
    for r in rows:
        if len(chosen) >= cap:
            break
        if r not in chosen:
            chosen.append(r)
    if chosen and not any(r["is_run"] for r in chosen):
        run = next((r for r in rows if r["is_run"]), None)
        if run and run not in chosen:
            chosen[-1] = run
    return [r["play"] for r in chosen]


def _macro_drafts(formations: Mapping[str, list[str]], sources: Mapping[str, str]) -> list[dict[str, Any]]:
    """Create reviewable templates from cited Madden 27 actions, not invented inputs."""
    from cfb_coach.madden.catalog import is_run

    drafts: list[dict[str, Any]] = []
    pass_pairs = [(f, p) for f, ps in formations.items() for p in ps if not is_run(p) and "rpo" not in p.lower()]
    if not pass_pairs:
        return []
    for action in research_db.offense_adjustments():
        kind = action.get("type")
        if kind not in ("hot_route", "pass_protection") or not action.get("sources"):
            continue
        target = str(action.get("target") or "").strip()
        route = str(action.get("route") or "").strip()
        if not target or not route:
            continue
        for coverage in (action.get("vs") or []):
            name = "ML-" + re.sub(r"[^A-Z0-9]+", "-", str(coverage).upper()).strip("-")[:9]
            name += "-" + ("HR" if kind == "hot_route" else "PROT")
            if any(d["name"] == name for d in drafts):
                continue
            # These are blueprints, never active editor settings by inference.
            section = "Route assignments" if kind == "hot_route" else "Protection"
            setting = target if kind == "hot_route" else "Protection"
            drafts.append({
                "name": name,
                "kind": kind, "coverage": str(coverage),
                "source_book_constraint": "exact verified formation/play pair",
                "status": "DRAFT_NEEDS_IN_GAME_VERIFICATION",
                "fire_when": f"live pre-snap {coverage}; only with matching passing play",
                "base_pairs": [
                    {"formation": f, "play": p, "catalog_book": sources[f]}
                    for f, p in pass_pairs[:6]
                ],
                "settings": [{
                    "section": section, "setting": setting, "value": route,
                    "sources": list(action["sources"]), "verification": "research_only",
                }],
                "other_editor_settings": "Unspecified; verify each in Madden editor; do not presume full inventory",
                "source_action_id": action["id"],
                "source_ids": list(action["sources"]),
                "activation": "NOT_ARMED: requires manual editor verification and an explicit future authorization",
            })
    return drafts[:8]


def design_offense(
    db: Any,
    *,
    opponent_id: str = "cpu",
    max_formations: int = MAX_FORMATIONS,
    max_plays: int = MAX_PLAYS_PER_FORMATION,
    artifact: Any = None,
    catalogue: Mapping[str, dict[str, list[str]]] | None = None,
) -> dict[str, Any]:
    """Deterministic offline design. No DB or file writes."""
    if not 1 <= max_formations <= MAX_FORMATIONS or not 2 <= max_plays <= MAX_PLAYS_PER_FORMATION:
        raise ValueError("max_formations must be 1..5 and max_plays 2..10")
    active = playbook.load_books(db).get("offense") or {}
    all_books = dict(catalogue) if catalogue is not None else {
        b: catalog.book_formations("offense", b) for b in catalog.book_names("offense")
    }
    art = artifact if artifact is not None else _load_artifact(db)
    if not all_books:
        raise ValueError("No Madden offensive formation catalog is available")

    best_by_form: dict[str, dict[str, Any]] = {}
    for source, formations in all_books.items():
        for form, ps in formations.items():
            if not ps or "hail mary" in form.lower() or "goal line" in form.lower():
                continue
            ranking = _play_rankings(art, source, form, ps, opponent_id, active)
            if not ranking:
                continue
            picks = _trim_plays(ranking, max_plays)
            if not picks:
                continue
            scores = [r["score"] for r in ranking if r["play"] in picks]
            is_current = form in (active.get("formations") or {})
            run = any(catalog.is_run(p) for p in picks)
            # Score comes from the model and coverage of concepts. Continuity
            # is a small tie-breaker, not a constraint or a heuristic pick.
            rank = sum(scores) / len(scores)
            rank += 0.014 if is_current else 0.0
            rank += 0.008 if run else 0.0
            rank += 0.002 * len({r["concept"] for r in ranking if r["play"] in picks})
            entry = {
                "formation": form, "source_book": source, "plays": picks,
                "score": round(rank, 6), "has_run": run,
                "top_play": ranking[0]["play"],
                "model_top_probability": ranking[0]["score"],
            }
            if form not in best_by_form or (
                entry["score"], source
            ) > (best_by_form[form]["score"], best_by_form[form]["source_book"]):
                best_by_form[form] = entry
    ranks = sorted(best_by_form.values(), key=lambda r: (-r["score"], r["formation"]))
    chosen = ranks[:max_formations]
    # Ensure the final custom book can call a run and a pass. This constraint is
    # football eligibility, not a return to heuristic play selection.
    if chosen and not any(r["has_run"] for r in chosen):
        runner = next((r for r in ranks[max_formations:] if r["has_run"]), None)
        if runner:
            chosen[-1] = runner
    if chosen and all(all(catalog.is_run(p) for p in r["plays"]) for r in chosen):
        passer = next((r for r in ranks[max_formations:]
                       if any(not catalog.is_run(p) for p in r["plays"])), None)
        if passer:
            chosen[-1] = passer
    if not chosen:
        raise ValueError("No eligible catalogued offensive plays were found")

    formations = {r["formation"]: r["plays"] for r in chosen}
    sources = {r["formation"]: r["source_book"] for r in chosen}
    all_pairs = sum(len(p) for p in formations.values())
    old = active.get("formations") or {}
    changes: list[dict[str, Any]] = []
    for f in sorted(set(old) | set(formations)):
        before = set(old.get(f) or [])
        after = set(formations.get(f) or [])
        if f not in old:
            changes.append({"action": "ADD_FORMATION", "formation": f, "source_book": sources[f]})
        elif f not in formations:
            changes.append({"action": "REMOVE_FORMATION", "formation": f})
        for p in sorted(after - before):
            changes.append({"action": "ADD_PLAY", "formation": f, "play": p,
                            "source_book": sources[f]})
        for p in sorted(before - after):
            changes.append({"action": "REMOVE_PLAY", "formation": f, "play": p})

    expected = _digest(active)
    proposal = {
        "schema": "ml_offense_design.v1", "opponent_id": opponent_id,
        "expected_applied_revision": int(active.get("rev") or 0),
        "expected_applied_hash": expected,
        "model_version": art.model_version,
        "evidence_quality": art.evidence_quality,
        "supervised_examples": art.n_supervised,
        "knowledge_version": art.knowledge_version,
        "book": {
            "side": "offense", "mode": "custom", "name": "ML Designed Offense (custom)",
            "source_book": None, "trimmed": True, "formations": formations,
            "formation_sources": sources,
            "audibles": {}, "core": list(formations),
            "rev": int(active.get("rev") or 0) + 1,
            "reason": "Model-designed individual formations and plays; explicitly installed in Madden",
        },
        "changes": changes,
        "candidate_formations": len(ranks), "ranked_formations": ranks[:12],
        "macro_blueprints": _macro_drafts(formations, sources),
        "n_plays": all_pairs,
        "editor_requires_confirmation": True,
        "notes": [
            "Proposal only; live offensive book is not changed.",
            "All pairs originate in the catalogued Madden 27 source book shown.",
            "Custom-editor import and availability must be checked in Madden before confirmation.",
            "Macro blueprints are NOT active. Editor fields and button sequences require validation.",
            "Predicted success is observational/shrinkage, not proof new plays win more.",
        ],
    }
    proposal["proposal_id"] = _identity(proposal)
    return proposal


def staged_design(db: Any) -> dict[str, Any] | None:
    raw = db.get_meta(META_PENDING)
    try:
        return json.loads(raw) if raw else None
    except (ValueError, TypeError):
        return None


def stage_design(db: Any, proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Only stage a proposal; never modify the active applied book or macro loadout."""
    if proposal.get("proposal_id") != _identity(proposal):
        raise ValueError("Proposal contents do not match its signed identifier")
    existing = staged_design(db)
    if existing and existing.get("proposal_id") == proposal.get("proposal_id"):
        return existing
    active = playbook.load_books(db).get("offense") or {}
    if _digest(active) != proposal["expected_applied_hash"]:
        raise ValueError("Applied book changed since proposal; re-run design")
    staged = dict(proposal)
    staged["staged_ts"] = datetime.now(timezone.utc).isoformat()
    db.set_meta(META_PENDING, _canonical(staged))
    return staged


def confirm_installed(db: Any, *, proposal_id: str, attestation: str) -> dict[str, Any]:
    """Explicit install confirmation; check revision/hashes and preserve rollback."""
    pending = staged_design(db)
    if not pending or pending.get("proposal_id") != proposal_id:
        raise ValueError("No staged proposal with this ID")
    if _identity(pending) != proposal_id:
        raise ValueError("Staged proposal contents changed; refuse installation")
    if len((attestation or "").strip()) < 18:
        raise ValueError("Explicit Madden in-game installation attestation required")
    state = playbook._load_state(db)
    current = state["applied"].get("offense") or {}
    if _digest(current) != pending.get("expected_applied_hash"):
        raise ValueError("Applied book changed since staging; refuse stale confirmation")
    new = dict(pending["book"])
    # Every pair must still be grounded in the indicated stock book.
    for form, ps in new["formations"].items():
        src = new["formation_sources"].get(form)
        available = set(catalog.book_formations("offense", src or "").get(form) or [])
        if any(p not in available for p in ps):
            raise ValueError(f"Catalog provenance invalid for {form}; refuse installation")
    history_raw = db.get_meta(META_HISTORY)
    try:
        history = json.loads(history_raw) if history_raw else []
    except ValueError:
        history = []
    history.append({
        "proposal_id": proposal_id, "old_book": current, "new_book": new,
        "attestation": attestation.strip(), "ts": datetime.now(timezone.utc).isoformat(),
    })
    # Guard the mutation: the user has confirmed the actual Madden editor state.
    new["locked_ts"] = datetime.now(timezone.utc).isoformat()
    state["applied"]["offense"] = new
    state["pending"].pop("offense", None)
    playbook._save_state(db, state)
    db.set_meta(META_HISTORY, _canonical(history[-30:]))
    # Carry blueprint proposals forward for separate manual validation. Never
    # activate a generated macro simply because its playbook was installed.
    db.set_meta(META_BLUEPRINTS, _canonical({
        "proposal_id": proposal_id, "macro_blueprints": pending.get("macro_blueprints") or [],
    }))
    db.set_meta(META_PENDING, "")
    return {
        "confirmed": True, "proposal_id": proposal_id, "revision": new["rev"],
        "formations": list(new["formations"]), "plays": sum(len(p) for p in new["formations"].values()),
        "macro_blueprints_activated": 0,
    }


def rollback_design(db: Any, *, proposal_id: str, attestation: str) -> dict[str, Any]:
    """Restore the prior plan only after the user has reinstalled it in Madden."""
    if len((attestation or "").strip()) < 25:
        raise ValueError("Explicit Madden in-game restoration attestation required")
    raw = db.get_meta(META_HISTORY)
    try:
        history = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        history = []
    match = next(
        (item for item in reversed(history)
         if item.get("proposal_id") == proposal_id and item.get("old_book")),
        None,
    )
    if not match:
        raise ValueError("Unknown proposal or no previous installed offense to restore")
    state = playbook._load_state(db)
    current = state["applied"].get("offense") or {}
    target = match["new_book"]
    if (current.get("formations") != target.get("formations")
            or current.get("rev") != target.get("rev")):
        raise ValueError("Applied offense changed since this proposal; refuse stale rollback")
    restored = dict(match["old_book"])
    restored["rev"] = int(current.get("rev") or 0) + 1
    restored["locked_ts"] = datetime.now(timezone.utc).isoformat()
    restored["reason"] = f"Manually restored prior installed offense from {proposal_id}"
    state["applied"]["offense"] = restored
    playbook._save_state(db, state)
    history.append({
        "rollback_of": proposal_id, "old_book": current, "new_book": restored,
        "attestation": attestation.strip(), "ts": datetime.now(timezone.utc).isoformat(),
    })
    db.set_meta(META_HISTORY, _canonical(history[-30:]))
    # The associated generated macros are no longer presumed installed or armed.
    db.set_meta(META_BLUEPRINTS, "")
    # Disable generated-macro recommendations after reverting the custom
    # playbook; require separate re-verification for another installed build.
    try:
        keys = db.conn.execute(
            "SELECT key FROM meta WHERE key LIKE 'ml_offense_verified_macros.v1:%'"
        ).fetchall()
        for row in keys:
            db.set_meta(row[0], "")
    except Exception:  # noqa: BLE001
        pass
    return {"rolled_back": True, "proposal_id": proposal_id,
            "revision": restored["rev"], "formations": list(restored["formations"])}


def verified_created_macros(db: Any, opponent_id: str) -> list[dict[str, Any]]:
    """Only editor-verified, explicitly armed user-created macros."""
    raw = db.get_meta(META_APPROVED.format(opponent=opponent_id)) if db else None
    try:
        obj = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return []
    return [dict(m) for m in obj.get("macros") or [] if m.get("name") and m.get("verified_armed")]


def verify_created_macro(
    db: Any, *, name: str, opponent_id: str,
    attestation: str, retire_existing: str | None = None,
) -> dict[str, Any]:
    """Approve exactly one generated macro already created and armed in Madden.

    Requires a user statement verifying the editor rows and active game slot.
    Cannot silently replace another macro or exceed eight active slots.
    """
    from cfb_coach.madden import macros as stored

    if len((attestation or "").strip()) < 25:
        raise ValueError("In-game macro creation, settings and armed slot attestation required")
    raw = db.get_meta(META_BLUEPRINTS)
    try:
        designs = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        designs = {}
    rows = designs.get("macro_blueprints") or []
    blueprint = next((m for m in rows if m.get("name") == name), None)
    if not blueprint:
        raise ValueError("Name is not a generated macro blueprint from a confirmed installed design")
    active_book = playbook.load_books(db).get("offense") or {}
    pairs = [
        item for item in blueprint["base_pairs"]
        if item.get("play") in (active_book.get("formations") or {}).get(item.get("formation"), [])
    ]
    if not pairs or not blueprint.get("settings") or not blueprint.get("source_ids"):
        raise ValueError("Macro not grounded in installed playbook and researched settings")
    source = next(
        (a for a in research_db.offense_adjustments()
         if a.get("id") == blueprint.get("source_action_id")), None
    )
    if not source or set(source.get("sources") or []) != set(blueprint.get("source_ids") or []):
        raise ValueError("Macro's researched source and settings have changed; re-design")
    existing = verified_created_macros(db, opponent_id)
    if any(m["name"] == name for m in existing):
        return {"name": name, "already_verified": True}
    current = stored.load_selection(db, opponent_id) or {"offense": [], "defense": []}
    current_o = list(current.get("offense") or [])
    replace = (retire_existing or "").upper().strip()
    if replace and replace not in current_o:
        raise ValueError("The existing macro to retire is not in this opponent's loadout")
    if len(current_o) + len(existing) - (1 if replace else 0) >= stored.LOADOUT_N:
        raise ValueError("All eight slots occupied; explicitly retire an existing macro after swapping it in Madden")
    approved = dict(blueprint)
    approved.update({
        "verified_armed": True, "opponent_id": opponent_id,
        "verified_ts": datetime.now(timezone.utc).isoformat(),
        "evidence": attestation.strip(), "base_pairs": pairs,
    })
    # Preserve existing call packages / defense macro state verbatim.
    if replace:
        key = stored.ACTIVE_META_KEY.format(opp=opponent_id)
        stored_raw = db.get_meta(key)
        data = json.loads(stored_raw) if stored_raw else {"schema": 2, "offense": [], "defense": []}
        data["offense"] = [k for k in (data.get("offense") or []) if str(k).upper() != replace]
        db.set_meta(key, _canonical(data))
    db.set_meta(
        META_APPROVED.format(opponent=opponent_id),
        _canonical({"schema": 1, "macros": existing + [approved]}),
    )
    return {
        "name": name, "verified_armed": True, "retired_existing": replace or None,
        "source_actions": approved["source_ids"], "base_pairs": pairs,
    }
