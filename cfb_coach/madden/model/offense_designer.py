"""Offline ML offensive playbook and Custom Adjustment designer.

The model may replace complete formations (containing all their plays) from catalogued Madden
books, and invent *draft configurations* from sourced adjustment primitives.
No recommendation is ever callable until the user builds/validates the custom
playbook and explicitly confirms its proposal ID. All macros remain drafts.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

from cfb_coach.madden import catalog, playbook, research_db
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model.experimental_live import resolve_artifact_path

META_PENDING = "ml_offense_design_pending.v1"
META_HISTORY = "ml_offense_design_history.v1"
META_BLUEPRINTS = "ml_offense_created_blueprints.v1"
META_CONFIG_VERIFIED = "ml_offense_verified_blueprints.v1"
META_APPROVED = "ml_offense_verified_macros.v1:{opponent}"
MAX_FORMATIONS = 12  # user-adjustable formation limit; all plays always included

SITUATION_MATRIX: tuple[dict[str, Any], ...] = (
    {"id": "normal_down", "down": 1, "distance": 10, "yardline": 35},
    {"id": "short_yardage", "down": 3, "distance": 2, "yardline": 55},
    {"id": "third_long", "down": 3, "distance": 10, "yardline": 55},
    {"id": "red_zone", "down": 2, "distance": 6, "yardline": 85, "red_zone": True},
    {"id": "goal_line", "down": 2, "distance": 2, "yardline": 98, "red_zone": True, "goal_line": True},
    {"id": "backed_up", "down": 1, "distance": 10, "yardline": 5},
    {"id": "two_minute_trailing", "down": 2, "distance": 7, "yardline": 60,
     "two_minute": True, "score_us": 17, "score_them": 24},
    {"id": "clock_management_lead", "down": 2, "distance": 6, "yardline": 60,
     "two_minute": True, "score_us": 24, "score_them": 17},
)


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


def _revoke_generated_macro_approvals(db: Any) -> None:
    """A newly installed/restored playbook requires re-arming macro slots."""
    try:
        keys = db.conn.execute(
            "SELECT key FROM meta WHERE key LIKE 'ml_offense_verified_macros.v1:%'"
        ).fetchall()
        for row in keys:
            db.set_meta(row[0], "")
    except Exception:  # noqa: BLE001 — no legacy table means none active
        pass


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


def _verified_roster_fit(roster: Any, *, has_run: bool, has_pass: bool) -> tuple[float, dict[str, Any]]:
    """Tiny transparent fit term from explicitly present roster features."""
    if roster is None:
        return 0.0, {"status": "unavailable", "features_used": []}
    values: list[tuple[str, float]] = []
    for feature in getattr(roster, "features", ()) or ():
        value = getattr(feature, "value_number", None)
        if getattr(feature, "missing", True) or value is None:
            continue
        key = str(getattr(feature, "key", "")).lower()
        relevant = (
            has_run and any(x in key for x in ("run_block", "halfback", "running_back"))
        ) or (
            has_pass and any(x in key for x in (
                "pass_block", "quarterback", "receiver", "tight_end", "speed"
            ))
        )
        if relevant:
            values.append((key, float(value)))
    if not values:
        return 0.0, {"status": "no_relevant_verified_features", "features_used": []}
    # Ratings may be 0..1 or 0..100. Normalize and cap the portfolio influence.
    normalized = [value / 100.0 if value > 1.0 else value for _, value in values]
    fit = max(-0.02, min(0.02, (sum(normalized) / len(normalized) - 0.5) * 0.04))
    return fit, {
        "status": "verified_features_applied",
        "features_used": [name for name, _ in values],
        "bounded_score_term": round(fit, 6),
    }


def _opponent_prior(opponent_context: Any) -> dict[str, Any]:
    counts: dict[str, int] = {}
    if opponent_context is not None:
        for tendency in getattr(opponent_context, "tendencies", ()) or ():
            key = str(getattr(tendency, "key", None) or getattr(tendency, "bucket", "")).strip()
            n = int(
                getattr(tendency, "sample_size", None)
                if getattr(tendency, "sample_size", None) is not None
                else getattr(tendency, "count", 0) or 0
            )
            if key and n > 0:
                counts[key] = counts.get(key, 0) + n
    n = sum(counts.values())
    dominant = max(counts, key=lambda k: (counts[k], k)) if counts else None
    return {
        "sample_size": n,
        "confidence": round(n / (n + 20.0), 4) if n else 0.0,
        "distribution": counts,
        "dominant_tendency": dominant,
        "source": "verified_opponent_context" if counts else "unknown",
    }


def _formation_situation_scores(
    art: Any,
    *,
    formation: str,
    plays: list[str],
    opponent: str,
    opponent_type: str,
    opponent_prior: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Value the formation in every required game state using concept breadth."""
    from cfb_coach.madden.model.football_situation import (
        evaluate_situation, score_play_suitability,
    )

    situation_scores: dict[str, float] = {}
    situation_answers: dict[str, list[str]] = {}
    tendency = opponent_prior.get("dominant_tendency")
    tendency_conf = float(opponent_prior.get("confidence") or 0.0)
    for scenario in SITUATION_MATRIX:
        sit = SimpleNamespace(
            **scenario,
            coverage_hint=None,
            coverage_source="none",
            extras={"quarter": 4 if scenario.get("two_minute") else 2},
        )
        state = evaluate_situation(
            sit,
            opponent_evidence={
                "dominant_coverage": tendency,
                "confidence": tendency_conf,
                "sample_size": int(opponent_prior.get("sample_size") or 0),
            },
        )
        by_concept: dict[str, tuple[float, str]] = {}
        for play in plays:
            prediction = experimental_model.predict_success(
                art, formation=formation, play=play,
                down=scenario["down"], distance=scenario["distance"],
                yardline=scenario["yardline"], opponent_id=opponent,
                opponent_type=opponent_type,
                coverage_hint=None, coverage_source="none",
                heuristic_bonus=0.0,
            )
            football = score_play_suitability(play, state)
            value = float(prediction["probability"]) + float(football["proxy_score"])
            concept = experimental_model._play_concept(play)
            if concept not in by_concept or value > by_concept[concept][0]:
                by_concept[concept] = (value, play)
        leaders = sorted(by_concept.values(), reverse=True)
        top = leaders[:min(3, len(leaders))]
        situation_scores[scenario["id"]] = round(
            sum(value for value, _ in top) / len(top), 6
        ) if top else 0.0
        situation_answers[scenario["id"]] = [play for _, play in top]
    return situation_scores, situation_answers


def _suggest_audibles(plays: list[str], rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suggest review candidates only; never claim the audible is installed."""
    ordered = [str(row["play"]) for row in rankings]
    buckets = (
        ("run", lambda play: catalog.is_run(play)),
        ("quick_pass", lambda play: not catalog.is_run(play) and any(
            token in play.lower() for token in ("slant", "mesh", "stick", "spacing", "flat")
        )),
        ("shot", lambda play: not catalog.is_run(play) and any(
            token in play.lower() for token in ("vert", "post", "fade", "wheel")
        )),
    )
    suggestions = []
    for role, predicate in buckets:
        play = next((candidate for candidate in ordered if predicate(candidate)), None)
        if play:
            suggestions.append({
                "role": role, "play": play,
                "status": "SUGGESTED_NEEDS_IN_GAME_VERIFICATION",
            })
    return suggestions


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
                    for f, p in pass_pairs
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
    artifact: Any = None,
    catalogue: Mapping[str, dict[str, list[str]]] | None = None,
    roster_snapshot: Any = None,
    opponent_context: Any = None,
) -> dict[str, Any]:
    """Deterministic formation-level design. Installs ALL catalogued plays per chosen formation.

    No individual play edits. The selected stock source is recorded because
    Madden's catalogued play list is stock-book-specific, not a guaranteed
    cross-book union in every version of the Madden custom editor.
    """
    if not 1 <= max_formations <= MAX_FORMATIONS:
        raise ValueError(f"max_formations must be 1..{MAX_FORMATIONS}")
    active = playbook.load_books(db).get("offense") or {}
    all_books = dict(catalogue) if catalogue is not None else {
        b: catalog.book_formations("offense", b) for b in catalog.book_names("offense")
    }
    art = artifact if artifact is not None else _load_artifact(db)
    from cfb_coach.opponents import is_cpu_opponent

    opponent_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
    opponent_prior = _opponent_prior(opponent_context)
    if not all_books:
        raise ValueError("No Madden offensive formation catalog is available")

    best_by_form: dict[str, dict[str, Any]] = {}
    for source, formations in all_books.items():
        for form, ps in formations.items():
            if not ps:
                continue
            ranking = _play_rankings(art, source, form, ps, opponent_id, active)
            if not ranking:
                continue
            # The formation is the indivisible installation unit. Every play
            # from that stock-book's formation stays eligible to the live model.
            picks = list(dict.fromkeys(ps))
            is_current = form in (active.get("formations") or {})
            run = any(catalog.is_run(p) for p in picks)
            has_pass = any(not catalog.is_run(p) for p in picks)
            # Score formations using the spread of concepts (not just one
            # anomalously high-ranked screen). A single screen cannot dictate
            # the score of an entire formation.
            best_by_concept: dict[str, float] = {}
            for row in ranking:
                concept = row["concept"]
                if concept not in best_by_concept:
                    best_by_concept[concept] = row["score"]
            situation_scores, situation_answers = _formation_situation_scores(
                art,
                formation=form,
                plays=picks,
                opponent=opponent_id,
                opponent_type=opponent_type,
                opponent_prior=opponent_prior,
            )
            balanced = sorted(situation_scores.values())
            # Mean across all required states plus a floor term prevents one
            # spectacular play/state from carrying an otherwise narrow formation.
            rank = sum(balanced) / len(balanced)
            rank += 0.10 * min(balanced)
            rank += 0.014 if is_current else 0.0
            rank += 0.008 if run else 0.0
            rank += 0.002 * len(best_by_concept)
            roster_term, roster_fit = _verified_roster_fit(
                roster_snapshot, has_run=run, has_pass=has_pass
            )
            rank += roster_term
            strongest_states = sorted(
                situation_scores, key=lambda key: (-situation_scores[key], key)
            )[:3]
            entry = {
                "formation": form, "source_book": source, "plays": picks,
                "score": round(rank, 6), "has_run": run,
                "top_play": ranking[0]["play"],
                "model_top_probability": ranking[0]["score"],
                "situation_scores": situation_scores,
                "situation_answers": situation_answers,
                "addresses": strongest_states,
                "selection_reason": (
                    "multi-situation formation value, concept breadth, portfolio fit, "
                    "and bounded verified roster/opponent context"
                ),
                "roster_fit": roster_fit,
                "opponent_prior": opponent_prior,
                "suggested_audibles": _suggest_audibles(picks, ranking),
                "uncertainty": (
                    "formation value is a situation-aware proxy; no causal win-rate "
                    "claim and current-snap coverage is not presumed"
                ),
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
    from cfb_coach.madden.model.offense_inventory import (
        inventory_fingerprint, pairs_in_inventory,
    )

    all_pairs = len(pairs_in_inventory(formations))
    inv_id = inventory_fingerprint(formations, sources)
    old = active.get("formations") or {}
    old_sources = active.get("formation_sources") or {}
    changes: list[dict[str, Any]] = []
    for f in sorted(set(old) | set(formations)):
        if f not in old:
            changes.append({
                "action": "ADD_FORMATION", "formation": f,
                "source_book": sources[f], "plays_included": len(formations[f]),
            })
        elif f not in formations:
            changes.append({"action": "REMOVE_FORMATION", "formation": f})
        elif (
            old_sources.get(f) != sources[f]
            or list(old[f]) != list(formations[f])
        ):
            changes.append({
                "action": "REINSTALL_FULL_FORMATION", "formation": f,
                "source_book": sources[f], "plays_included": len(formations[f]),
                "note": "Install the complete formation from this source; no individual play editing",
            })
    removed = [item["formation"] for item in changes if item["action"] == "REMOVE_FORMATION"]
    added = [item["formation"] for item in changes if item["action"] == "ADD_FORMATION"]
    replacements = [
        {"remove": old_form, "add": new_form}
        for old_form, new_form in zip(removed, added)
    ]
    situation_coverage = {
        scenario["id"]: [
            {
                "formation": row["formation"],
                "score": row["situation_scores"][scenario["id"]],
                "answer_plays": row["situation_answers"][scenario["id"]],
            }
            for row in sorted(
                chosen,
                key=lambda item: -item["situation_scores"][scenario["id"]],
            )[:3]
        ]
        for scenario in SITUATION_MATRIX
    }

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
            "formation_sources": sources, "inventory_id": inv_id,
            "audibles": {}, "core": list(formations),
            "rev": int(active.get("rev") or 0) + 1,
            "reason": "Model-designed whole formations with every catalogued play from selected source",
        },
        "changes": changes,
        "replacements": replacements,
        "candidate_formations": len(ranks),
        "catalog_formations_evaluated": len(ranks),
        "ranked_formations": ranks[:max_formations],
        "formation_evaluations": ranks,
        "chosen_formation_details": chosen,
        "situation_coverage": situation_coverage,
        "suggested_audibles": {
            row["formation"]: row["suggested_audibles"] for row in chosen
        },
        "macro_blueprints": _macro_drafts(formations, sources),
        "n_plays": all_pairs,
        "inventory_id": inv_id,
        "complete_inventory_fingerprint": inv_id,
        "provenance": {
            "model_version": art.model_version,
            "knowledge_version": art.knowledge_version,
            "evidence_quality": art.evidence_quality,
            "supervised_examples": art.n_supervised,
            "situation_knowledge": "madden.football_situation.v1",
            "opponent": opponent_prior,
            "roster": (
                "verified_snapshot_fields_only"
                if roster_snapshot is not None else "unavailable"
            ),
            "research_adjustments": "cfb_coach.madden.research_db",
        },
        "uncertainty": {
            "current_defensive_coverage": "unknown before each game snap",
            "formation_scores": "situation-aware proxy, not causal win probability",
            "roster_data": (
                "verified fields applied with bounded influence"
                if roster_snapshot is not None else "not provided"
            ),
            "opponent_sample_size": opponent_prior["sample_size"],
            "opponent_confidence": opponent_prior["confidence"],
        },
        "editor_requires_confirmation": True,
        "notes": [
            "Proposal only; live offensive book is not changed.",
            "Each included formation contains ALL catalogued plays from its selected stock source.",
            "Madden stock-book play lists differ; cross-book union is not assumed installed.",
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
        if len(ps) != len(set(ps)) or set(ps) != available or len(ps) != len(available):
            raise ValueError(
                f"Formation {form} is incomplete or diverged from sourced stock-book plays; "
                "cannot confirm partial formation installation"
            )
    from cfb_coach.madden.model.offense_inventory import inventory_fingerprint
    expected_inventory = inventory_fingerprint(
        new["formations"], new["formation_sources"]
    )
    if new.get("inventory_id") != expected_inventory:
        raise ValueError("Inventory changed since staged design; refuse installation")
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
    _revoke_generated_macro_approvals(db)
    db.set_meta(META_BLUEPRINTS, _canonical({
        "proposal_id": proposal_id, "macro_blueprints": pending.get("macro_blueprints") or [],
    }))
    db.set_meta(META_PENDING, "")
    return {
        "confirmed": True, "proposal_id": proposal_id, "revision": new["rev"],
        "formations": list(new["formations"]), "plays": sum(len(p) for p in new["formations"].values()),
        "inventory_id": new["inventory_id"], "macro_blueprints_activated": 0,
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
    _revoke_generated_macro_approvals(db)
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


def verified_macro_configurations(db: Any) -> list[dict[str, Any]]:
    """Editor-verified blueprint configurations that are not necessarily armed."""
    try:
        obj = json.loads(db.get_meta(META_CONFIG_VERIFIED) or "{}")
    except (TypeError, ValueError):
        return []
    return [
        dict(item) for item in obj.get("macros") or []
        if item.get("name") and item.get("configuration_verified")
    ]


def _blueprint_for_name(db: Any, name: str) -> dict[str, Any]:
    try:
        designs = json.loads(db.get_meta(META_BLUEPRINTS) or "{}")
    except (TypeError, ValueError):
        designs = {}
    blueprint = next(
        (dict(item) for item in designs.get("macro_blueprints") or []
         if item.get("name") == name),
        None,
    )
    if not blueprint:
        raise ValueError("Name is not a generated macro blueprint from a confirmed installed design")
    return blueprint


def _validate_blueprint_sources(db: Any, blueprint: Mapping[str, Any]) -> list[dict[str, Any]]:
    active_book = playbook.load_books(db).get("offense") or {}
    pairs = [
        item for item in blueprint.get("base_pairs") or []
        if item.get("play") in (active_book.get("formations") or {}).get(
            item.get("formation"), []
        )
    ]
    action_ids = list(blueprint.get("source_action_ids") or [])
    if not action_ids and blueprint.get("source_action_id"):
        action_ids = [blueprint["source_action_id"]]
    sources = {
        str(item.get("id")): item for item in research_db.offense_adjustments()
        if item.get("id")
    }
    actions = [sources[action_id] for action_id in action_ids if action_id in sources]
    expected_citations = {
        source for action in actions for source in (action.get("sources") or [])
    }
    if (
        not pairs
        or not blueprint.get("settings")
        or len(actions) != len(action_ids)
        or expected_citations != set(blueprint.get("source_ids") or [])
    ):
        raise ValueError("Macro is not grounded in the installed book and current researched settings")
    if len(blueprint.get("settings") or []) != len(actions):
        raise ValueError("Macro setting count no longer matches its researched primitives")
    return pairs


def verify_created_macro_configuration(
    db: Any, *, name: str, attestation: str,
) -> dict[str, Any]:
    """Move a draft to VERIFIED without claiming an occupied in-game slot."""
    if len((attestation or "").strip()) < 25:
        raise ValueError("Explicit in-game editor configuration attestation required")
    blueprint = _blueprint_for_name(db, name)
    pairs = _validate_blueprint_sources(db, blueprint)
    existing = verified_macro_configurations(db)
    if any(item["name"] == name for item in existing):
        return {"name": name, "configuration_verified": True, "armed": False}
    verified = dict(blueprint)
    verified.update({
        "status": "VERIFIED_NOT_ARMED",
        "configuration_verified": True,
        "verified_armed": False,
        "configuration_verified_ts": datetime.now(timezone.utc).isoformat(),
        "configuration_evidence": attestation.strip(),
        "base_pairs": pairs,
    })
    db.set_meta(
        META_CONFIG_VERIFIED,
        _canonical({"schema": 1, "macros": existing + [verified]}),
    )
    return {"name": name, "configuration_verified": True, "armed": False}


def unverify_created_macro(db: Any, *, name: str, opponent_id: str) -> dict[str, Any]:
    """Immediately stop live ML from offering a removed/unarmed custom macro."""
    existing = verified_created_macros(db, opponent_id)
    remaining = [m for m in existing if m["name"] != name]
    if len(remaining) == len(existing):
        return {"name": name, "opponent_id": opponent_id, "was_verified": False}
    db.set_meta(
        META_APPROVED.format(opponent=opponent_id),
        _canonical({"schema": 1, "macros": remaining}),
    )
    return {
        "name": name, "opponent_id": opponent_id, "was_verified": True,
        "now_callable": False,
    }


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
    blueprint = _blueprint_for_name(db, name)
    pairs = _validate_blueprint_sources(db, blueprint)
    # Preserve a separately inspectable VERIFIED state even when the legacy
    # one-step command explicitly verifies and arms in the same transaction.
    verify_created_macro_configuration(
        db, name=name, attestation=attestation,
    )
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
        "status": "ARMED", "configuration_verified": True,
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
