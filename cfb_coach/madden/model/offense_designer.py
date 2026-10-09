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
from typing import Any, Mapping

from cfb_coach.madden import catalog, playbook, research_db
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model.experimental_live import resolve_artifact_path

META_PENDING = "ml_offense_design_pending.v1"
META_HISTORY = "ml_offense_design_history.v1"
META_BLUEPRINTS = "ml_offense_created_blueprints.v1"
META_APPROVED = "ml_offense_verified_macros.v1:{opponent}"
MAX_FORMATIONS = 12  # user-adjustable formation limit; all plays always included


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


def _pregame_grid() -> list[dict[str, Any]]:
    """Grid cells with precomputed structured assessments (computed once)."""
    from cfb_coach.madden.model.football_situations import (
        PREGAME_SITUATION_GRID, grid_cell_assessment,
    )

    return [
        {"cell": dict(cell), "assessment": grid_cell_assessment(cell)}
        for cell in PREGAME_SITUATION_GRID
    ]


def _cell_scores(
    art: Any, form: str, play: str, opponent: str, grid: list[dict[str, Any]],
) -> dict[str, float]:
    """Model success per pregame situation, adjusted by named football priors.

    Uses multiple generic pre-snap situations; no post-snap coverage is ever
    assumed. A play contributes only to cells whose field zone it fits.
    """
    from cfb_coach.madden.model.football_situations import (
        play_fits_grid_zone, play_situation_fit,
    )
    from cfb_coach.opponents import is_cpu_opponent

    opponent_type = "cpu" if is_cpu_opponent(opponent) else "human"
    out: dict[str, float] = {}
    for item in grid:
        cell = item["cell"]
        if not play_fits_grid_zone(play, cell):
            continue
        prob = experimental_model.predict_success(
            art, formation=form, play=play,
            down=cell["down"], distance=cell["distance"],
            yardline=cell["yardline"], opponent_id=opponent,
            opponent_type=opponent_type,
            coverage_hint=None, coverage_source="none",
            heuristic_bonus=0.0,
        )["probability"]
        fit, _reasons = play_situation_fit(play, item["assessment"])
        out[str(cell["name"])] = round(float(prob) + fit, 6)
    return out


def _play_rankings(
    art: Any, book: str, form: str, plays: list[str],
    opponent: str, current: Mapping[str, Any],
    grid: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    old = (current.get("formations") or {}).get(form) or []
    grid = grid if grid is not None else _pregame_grid()
    weights = {str(g["cell"]["name"]): float(g["cell"]["weight"]) for g in grid}
    ranked: list[dict[str, Any]] = []
    for play in dict.fromkeys(plays):
        if not catalog.zone_fit(play, "open") and not catalog.zone_fit(play, "gl"):
            continue
        cells = _cell_scores(art, form, play, opponent, grid)
        if not cells:
            continue
        total_w = sum(weights[name] for name in cells)
        score = sum(cells[name] * weights[name] for name in cells) / total_w
        # A small continuity preference, never a hard lock; the model may swap.
        if play in old:
            score += 0.012
        ranked.append({
            "formation": form, "play": play, "score": round(score, 6),
            "concept": experimental_model._play_concept(play),
            "is_run": catalog.is_run(play), "catalog_book": book,
            "cell_scores": cells,
        })
    ranked.sort(key=lambda x: (-x["score"], x["play"]))
    return ranked


# Concept families that are standard, explicitly labeled football counters to
# a credibly observed defensive tendency. These are priors (basis documented),
# not learned effects, and only apply with enough live observations.
TENDENCY_COUNTER_FAMILIES: dict[str, frozenset[str]] = {
    "pressure": frozenset({"cross", "screen", "rpo"}),
    "man": frozenset({"cross", "stack"}),
    "cover2": frozenset({"flood", "vert"}),
    "two_high": frozenset({"run", "cross"}),
    "single_high": frozenset({"vert", "flood"}),
}
TENDENCY_MIN_OBSERVATIONS = 5
TENDENCY_FORMATION_BONUS = 0.012


def _play_counter_family(play: str) -> str | None:
    from cfb_coach.madden.model.experimental_model import _play_concept, _play_family
    from cfb_coach.madden.situation import concept_family

    fam = _play_family(play)
    if fam in ("screen", "run", "rpo"):
        return fam
    return concept_family(_play_concept(play))


def _tendency_evidence(db: Any, opponent_id: str) -> dict[str, Any]:
    """Observed opponent coverage tendencies with sample sizes and confidence."""
    from cfb_coach.madden.playcaller import coverage_class

    counts: dict[str, int] = {}
    total = 0
    try:
        for row in db.get_tendencies(opponent_id, "their_coverage"):
            key = str(row["key"])
            if key.endswith("_seed"):
                continue
            cls = coverage_class(key)
            if not cls:
                continue
            n = int(row["count"] or 0)
            counts[cls] = counts.get(cls, 0) + n
            total += n
    except Exception:  # noqa: BLE001 — tendency store is optional
        counts, total = {}, 0
    dominant = None
    if total >= TENDENCY_MIN_OBSERVATIONS and counts:
        top_class, top_n = max(counts.items(), key=lambda kv: kv[1])
        if top_n >= TENDENCY_MIN_OBSERVATIONS and top_n / total >= 0.4:
            dominant = top_class
    confidence = (
        "none" if total == 0 else
        "low" if total < TENDENCY_MIN_OBSERVATIONS else
        "medium" if total < 12 else "high"
    )
    return {
        "observed_coverage_classes": counts,
        "total_live_observations": total,
        "confidence": confidence,
        "dominant_class": dominant,
        "counter_families": sorted(TENDENCY_COUNTER_FAMILIES.get(dominant, ()))
        if dominant else [],
        "basis": "live in-game tendency counts; seed priors excluded",
        "note": (
            "Tendencies are inferred from past observed snaps, never proof of "
            "a future coverage. Bonus applies only with sufficient samples."
        ),
    }


def _suggested_audibles(ranking: list[dict[str, Any]]) -> list[str]:
    """Up to four in-formation plays spanning distinct families by model score."""
    from cfb_coach.madden.model.experimental_model import _play_family

    picks: list[str] = []
    seen_fams: set[str] = set()
    for row in ranking:
        fam = _play_family(row["play"])
        if fam in seen_fams:
            continue
        seen_fams.add(fam)
        picks.append(row["play"])
        if len(picks) >= 4:
            break
    return picks


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
    if not all_books:
        raise ValueError("No Madden offensive formation catalog is available")

    grid = _pregame_grid()
    weights = {str(g["cell"]["name"]): float(g["cell"]["weight"]) for g in grid}
    total_weight = sum(weights.values())
    tendencies = _tendency_evidence(db, opponent_id)
    counter_fams = set(tendencies.get("counter_families") or [])
    excluded_formations: list[dict[str, str]] = []
    best_by_form: dict[str, dict[str, Any]] = {}
    for source, formations in all_books.items():
        for form, ps in formations.items():
            if not ps:
                continue
            # The whole supported catalog is evaluated; only formation types
            # the custom editor cannot sensibly install stay out, visibly.
            if "hail mary" in form.lower() or "goal line" in form.lower():
                excluded_formations.append({
                    "formation": form, "source_book": source,
                    "reason": "special-teams/jumbo package not supported as a designed formation unit",
                })
                continue
            ranking = _play_rankings(
                art, source, form, ps, opponent_id, active, grid=grid
            )
            if not ranking:
                continue
            # The formation is the indivisible installation unit. Every play
            # from that stock-book's formation stays eligible to the live model.
            picks = list(dict.fromkeys(ps))
            is_current = form in (active.get("formations") or {})
            run = any(catalog.is_run(p) for p in picks)
            # Score the formation by its value ACROSS the pregame situation
            # grid: each cell's value is the mean of the top-2 play scores
            # there, so one anomalously high-rated play cannot dictate the
            # score of an entire formation, and a formation that only works
            # in one situation ranks below a versatile one.
            cell_values: dict[str, float] = {}
            cell_best: dict[str, dict[str, Any]] = {}
            for g in grid:
                name = str(g["cell"]["name"])
                scored = sorted(
                    ((r["cell_scores"][name], r) for r in ranking
                     if name in r["cell_scores"]),
                    key=lambda t: -t[0],
                )
                if not scored:
                    continue
                top = [s for s, _r in scored[:2]]
                cell_values[name] = sum(top) / len(top)
                cell_best[name] = {
                    "play": scored[0][1]["play"],
                    "score": round(float(scored[0][0]), 6),
                }
            if not cell_values:
                continue
            covered_w = sum(weights[n] for n in cell_values)
            rank = sum(cell_values[n] * weights[n] for n in cell_values) / covered_w
            # Situations the formation cannot address at all cost coverage.
            rank -= 0.05 * (1.0 - covered_w / total_weight)
            concepts = {r["concept"] for r in ranking}
            rank += 0.014 if is_current else 0.0
            rank += 0.008 if run else 0.0
            rank += 0.002 * len(concepts)
            counter_plays = [
                r["play"] for r in ranking
                if _play_counter_family(r["play"]) in counter_fams
            ]
            tendency_bonus = (
                TENDENCY_FORMATION_BONUS if len(counter_plays) >= 2 else 0.0
            )
            rank += tendency_bonus
            entry = {
                "formation": form, "source_book": source, "plays": picks,
                "score": round(rank, 6), "has_run": run,
                "top_play": ranking[0]["play"],
                "model_top_probability": ranking[0]["score"],
                "concept_count": len(concepts),
                "situations_addressed": [
                    {"situation": n, "weight": weights[n],
                     "best_play": cell_best[n]["play"],
                     "score": cell_best[n]["score"]}
                    for n in sorted(cell_best, key=lambda n: -weights[n])
                ],
                "cell_best": cell_best,
                "tendency_counter_plays": counter_plays[:4],
                "tendency_bonus": tendency_bonus,
                "suggested_audibles": _suggested_audibles(ranking),
                "why_selected": (
                    f"weighted value across {len(cell_values)}/{len(grid)} pregame situations; "
                    f"{len(concepts)} concepts; "
                    + ("includes run game; " if run else "")
                    + ("addresses observed "
                       f"{tendencies.get('dominant_class')} tendency; " if tendency_bonus else "")
                    + ("continuity with installed book" if is_current else "new installation")
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

    # Portfolio coverage repair: a plan must address every important game
    # state (weight >= 0.5) that ANY candidate formation can address. Swap in
    # the best-covering alternative for the weakest safely removable choice.
    def _covered(entries: list[dict[str, Any]]) -> set[str]:
        return {n for e in entries for n in e["cell_best"]}

    important = [str(g["cell"]["name"]) for g in grid
                 if float(g["cell"]["weight"]) >= 0.5]
    for cell_name in important:
        if cell_name in _covered(chosen):
            continue
        fixer = next(
            (r for r in ranks if r not in chosen and cell_name in r["cell_best"]),
            None,
        )
        if fixer is None:
            continue  # no catalogued formation addresses it; reported below
        for victim in reversed(chosen):
            without = [e for e in chosen if e is not victim]
            trial = without + [fixer]
            if not any(e["has_run"] for e in trial):
                continue
            if all(all(catalog.is_run(p) for p in e["plays"]) for e in trial):
                continue
            lost = {
                n for n in important
                if n in _covered(chosen) and n not in _covered(trial)
            }
            if lost:
                continue
            chosen = trial
            break

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

    # Plan-level situation coverage over the chosen portfolio.
    situation_coverage: dict[str, Any] = {}
    for g in grid:
        name = str(g["cell"]["name"])
        best = max(
            ((e["cell_best"][name]["score"], e["formation"],
              e["cell_best"][name]["play"])
             for e in chosen if name in e["cell_best"]),
            default=None,
        )
        situation_coverage[name] = {
            "weight": weights[name],
            "covered": best is not None,
            "best_formation": best[1] if best else None,
            "best_play": best[2] if best else None,
            "score": best[0] if best else None,
        }

    def _public(entry: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in entry.items() if k != "cell_best"}

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
        "candidate_formations": len(ranks),
        "ranked_formations": [_public(r) for r in ranks[:12]],
        "chosen_formations": [_public(r) for r in chosen],
        "situation_grid": [
            {"name": str(g["cell"]["name"]), "weight": float(g["cell"]["weight"]),
             "down": g["cell"].get("down"), "distance": g["cell"].get("distance"),
             "yardline": g["cell"].get("yardline")}
            for g in grid
        ],
        "situation_coverage": situation_coverage,
        "tendency_evidence": tendencies,
        "suggested_audibles": {
            e["formation"]: e.get("suggested_audibles") or [] for e in chosen
        },
        "excluded_unsupported_formations": excluded_formations[:20],
        "macro_blueprints": _macro_drafts(formations, sources),
        "n_plays": all_pairs,
        "inventory_id": inv_id,
        "editor_requires_confirmation": True,
        "provenance": {
            "model_version": art.model_version,
            "evidence_quality": art.evidence_quality,
            "supervised_examples": art.n_supervised,
            "scoring_basis": (
                "verified-data model probabilities across the pregame situation "
                "grid, adjusted by named football priors "
                "(football_situations.SITUATION_PRIORS)"
            ),
            "uncertainty": (
                "prior-driven and sparse-data formations carry high uncertainty; "
                "scores are situation-aware proxies, not win-probability estimates"
            ),
        },
        "notes": [
            "Proposal only; live offensive book is not changed.",
            "Each included formation contains ALL catalogued plays from its selected stock source.",
            "Madden stock-book play lists differ; cross-book union is not assumed installed.",
            "Custom-editor import and availability must be checked in Madden before confirmation.",
            "Macro blueprints are NOT active. Editor fields and button sequences require validation.",
            "Predicted success is observational/shrinkage, not proof new plays win more.",
            "Suggested audibles are in-book plays; configuring them in Madden is manual.",
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
