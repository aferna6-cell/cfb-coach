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
from typing import Any, Mapping, Sequence

from cfb_coach.madden import catalog, playbook, research_db
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model.experimental_live import resolve_artifact_path

META_PENDING = "ml_offense_design_pending.v1"
META_HISTORY = "ml_offense_design_history.v1"
META_BLUEPRINTS = "ml_offense_created_blueprints.v1"
META_APPROVED = "ml_offense_verified_macros.v1:{opponent}"
MAX_FORMATIONS = 15  # maximum custom-book formations; fewer is allowed
# Shared by offense-coordinator, offense-design, and offense-design --stage.
# 15 is the cap, not a quota.
DEFAULT_MAX_FORMATIONS = 15


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


def _rate(
    art: Any,
    form: str,
    play: str,
    opponent: str,
    *,
    priors_on: bool = False,
    coverage_mix: Mapping[str, float] | None = None,
    learned: Mapping | None = None,
    structure: Any | None = None,
) -> float:
    """Average shared pregame scores across generic probes; no coverage hint."""
    from types import SimpleNamespace

    from cfb_coach.madden.model.offense_pregame_scoring import score_pair

    # Multiple generic pre-snap situations; no post-snap coverage is assumed.
    checks = ((1, 10, 35), (2, 6, 50), (3, 7, 35))
    vals = [
        score_pair(
            art,
            formation=form,
            play=play,
            probe=SimpleNamespace(down=down, distance=distance, yardline=yardline),
            mix=coverage_mix,
            opponent_id=opponent,
            priors_on=priors_on,
            learned=learned,
            structure=structure,
        ).total_probability
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


SWAP_MIN_SCORE_GAP = 0.02   # an optional swap is only suggested when the evidence gap is at least this large


def additive_book(*, active: Mapping[str, Any], chosen_formations: Mapping[str, list[str]], chosen_sources: Mapping[str, str],
                  ranked: Sequence[Mapping[str, Any]], catalogue: Mapping[str, Mapping[str, list[str]]],
                  max_formations: int = MAX_FORMATIONS, must_add: Sequence[str] = ()
                  ) -> tuple[dict[str, list[str]], dict[str, str], list[dict[str, Any]], list[str]]:
    """Retain every installed formation; add up to the cap; at the cap offer optional, evidence-backed swaps only."""
    old = {f: list(p) for f, p in (active.get("formations") or {}).items()}
    old_sources = dict(active.get("formation_sources") or {})
    formations: dict[str, list[str]] = {}
    sources: dict[str, str] = {}
    notes: list[str] = []
    for f, plays in old.items():                                         # every installed formation stays; never removed
        src = old_sources.get(f)
        if not (src and f in (catalogue.get(src) or {})):
            src = chosen_sources.get(f) if f in (catalogue.get(chosen_sources.get(f) or "") or {}) else next((b for b, fm in catalogue.items() if f in fm), None)
        if src and list(catalogue[src][f]) != plays:                     # diverged/partial install -> complete the SAME formation (REINSTALL), not drop it
            plays = list(catalogue[src][f])
        formations[f], sources[f] = plays, src or old_sources.get(f) or "installed"
    score = {r["formation"]: float(r.get("score") or 0.0) for r in ranked}
    room = max_formations - len(formations)
    # owner-requested additions first (need a catalogued source), then the portfolio's own picks in rank order
    wanted: list[tuple[str, list[str], str]] = []
    for name in must_add:
        if name in formations:
            notes.append(f"{name} is already installed")
            continue
        src = next((b for b, fm in catalogue.items() if name in fm), None)
        if src is None:
            notes.append(f"{name}: no catalogued source found; not added")
            continue
        src = chosen_sources.get(name) or src
        wanted.append((name, list(catalogue[src][name]) if src in catalogue and name in catalogue[src] else list(chosen_formations.get(name, [])), src))
    for name, plays in chosen_formations.items():
        if name not in formations and name not in {w[0] for w in wanted}:
            wanted.append((name, list(plays), chosen_sources[name]))
    for name, plays, src in wanted:
        if room <= 0:
            break
        formations[name], sources[name] = plays, src
        room -= 1
    swaps: list[dict[str, Any]] = []
    if len(old) >= max_formations:                                       # only when the book was ALREADY full before this proposal
        notes.append(f"book is at the {max_formations}-formation cap: removals are only ever OPTIONAL swaps for the owner to decide")
        installed_scored = sorted(((score.get(f, 0.0), f) for f in formations))
        outside = sorted(((score.get(r["formation"], 0.0), r["formation"], r["source_book"]) for r in ranked if r["formation"] not in formations), reverse=True)
        for (s_out, f_out), (s_in, f_in, src) in zip(installed_scored, outside):
            if s_in - s_out >= SWAP_MIN_SCORE_GAP:
                swaps.append({"action": "OPTIONAL_SWAP", "remove": f_out, "add": f_in, "add_source_book": src,
                              "evidence": {"installed_score": round(s_out, 4), "candidate_score": round(s_in, 4), "gap": round(s_in - s_out, 4)},
                              "requires_owner_approval": True})
    else:
        notes.append(f"{len(formations)} of {max_formations} formations: nothing is removed; unused formations may stay installed")
    if len(formations) >= max_formations and len(old) < max_formations:
        notes.append(f"this proposal fills the {max_formations}-formation cap; swaps can only be proposed from the next proposal on")
    return formations, sources, swaps, notes


def design_offense(
    db: Any,
    *,
    opponent_id: str = "cpu",
    max_formations: int = MAX_FORMATIONS,
    artifact: Any = None,
    catalogue: Mapping[str, dict[str, list[str]]] | None = None,
    must_add: Sequence[str] = (),
    priors_on: bool = False,
    coverage_mix: Mapping[str, float] | None = None,
    adjustment_pairs: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Deterministic formation-level design.

    ADDITIVE POLICY (owner, #120): every currently installed formation is retained, verbatim, until the book holds
    ``max_formations`` (15). The model never proposes a removal merely because a formation scores poorly: it can simply
    stop calling it. At the cap it may list OPTIONAL swaps (evidence attached) for the owner to accept or ignore.
    ``must_add`` names owner-requested formations that are added first when a catalogued source exists.

    Previously: Installs ALL catalogued plays per chosen formation.

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

    # Every catalogued formation is scored across the situation probes.
    # Goal-line and Hail Mary packages stay in the catalog; a single favorite
    # list is not used. The helper keeps the run/pass portfolio constraint.
    from cfb_coach.madden.model.offense_portfolio import select_formation_portfolio

    # The shared scorer/portfolio interface lands independently. Never silently
    # claim a football-informed design on a stack that cannot consume its mix.
    from inspect import signature
    supported = signature(select_formation_portfolio).parameters
    scoring_kwargs = {}
    if "priors_on" in supported and "coverage_mix" in supported:
        scoring_kwargs = {"priors_on": priors_on, "coverage_mix": coverage_mix}
    elif priors_on or coverage_mix:
        raise ValueError("Shared pregame portfolio scoring is not integrated yet")
    portfolio = select_formation_portfolio(
        art=art, all_books=all_books, active=active, opponent_id=opponent_id,
        max_formations=max_formations, db=db,
        **scoring_kwargs,
    )
    chosen = portfolio["chosen"]
    ranks = portfolio["ranked"]
    if not chosen:
        raise ValueError("No eligible catalogued offensive plays were found")

    chosen_f = {r["formation"]: r["plays"] for r in chosen}
    chosen_s = {r["formation"]: r["source_book"] for r in chosen}
    formations, sources, optional_swaps, policy_notes = additive_book(
        active=active, chosen_formations=chosen_f, chosen_sources=chosen_s, ranked=ranks,
        catalogue=all_books, max_formations=max_formations, must_add=must_add,
    )
    from cfb_coach.madden.model.offense_macro_lab import compose_drafts

    macro_blueprints = _macro_drafts(formations, sources)
    seen_names = {row["name"] for row in macro_blueprints}
    for draft in compose_drafts(formations, sources):
        if draft["name"] in seen_names:
            continue
        macro_blueprints.append(draft)
        seen_names.add(draft["name"])
    macro_blueprints = macro_blueprints[:12]
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
        "optional_swaps": optional_swaps,
        "formation_policy": {"mode": "additive_until_cap", "cap": max_formations, "installed_before": len(old),
                             "installed_after": len(formations), "removals": [c["formation"] for c in changes if c["action"] == "REMOVE_FORMATION"],
                             "notes": policy_notes},
        "candidate_formations": len(ranks), "ranked_formations": [
            {
                "formation": row["formation"], "source_book": row["source_book"],
                "score": row["score"], "plays": len(row["plays"]),
            }
            for row in ranks[:12]
        ],
        "formation_rationales": portfolio["rationales"],
        "situation_coverage": portfolio["situation_coverage"],
        "provenance": {
            "model_version": art.model_version,
            "evidence_quality": art.evidence_quality,
            "supervised_examples": art.n_supervised,
            "knowledge_version": art.knowledge_version,
            "football_knowledge": portfolio.get("football_plan"),
            "roster": portfolio["roster"],
            "opponent_defense": portfolio["opponent_defense"],
            "uncertainty": portfolio["uncertainty"],
            "research_sources": len(research_db.sources()),
            "situation_probes": portfolio["probes"],
        },
        "suggested_audibles": [
            {
                "formation": row["formation"],
                "play": row["suggested_audible"],
                "status": row["audible_status"],
            }
            for row in portfolio["rationales"] if row.get("suggested_audible")
        ],
        "macro_blueprints": macro_blueprints,
        "n_plays": all_pairs,
        "inventory_id": inv_id,
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
    try:                                                                   # informational only: how much is actually KNOWN about each formation
        from cfb_coach.madden.model import football_universe as fu
        uni = fu.load()
        proposal["universe_evidence"] = {f: fu.formation_evidence("offense", f, uni) for f in formations}
        proposal["universe_digest"] = uni["digest"]
    except (OSError, KeyError, ValueError):
        proposal["universe_evidence"] = {}
    proposal["proposal_id"] = _identity(proposal)
    proposal["max_formations"] = max_formations
    proposal["pregame_diagnostic"] = pregame_input_diagnostic(proposal.get("provenance"))
    proposal["football_plan"] = portfolio.get("football_plan")
    proposal["preparation_guide"] = preparation_guide(proposal, active=active)
    if adjustment_pairs:
        from types import SimpleNamespace
        from cfb_coach.madden.model.offense_adjustment_designer import design_adjustments
        proposal["adjustment_designs"] = []
        for form, play in dict.fromkeys(adjustment_pairs):
            if play not in formations.get(form, []):
                raise ValueError("Adjustment design must use a selected stock play")
            proposal["adjustment_designs"].append(design_adjustments(
                art, formation=form, play=play, source_book=sources[form],
                probe=SimpleNamespace(down=1, distance=10, yardline=35),
                opponent_id=opponent_id, priors_on=priors_on, coverage_mix=coverage_mix,
            ))
    return proposal


def preparation_guide(proposal: Mapping[str, Any], *, active: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Manual checklist limited to the proposed book, never an executable loadout."""
    book = proposal.get("book") or {}
    installed = active or {}
    rows = []
    for form, plays in (book.get("formations") or {}).items():
        source = (book.get("formation_sources") or {}).get(form)
        confirmed = (list((installed.get("formations") or {}).get(form) or []) == list(plays)
                     and (installed.get("formation_sources") or {}).get(form) == source)
        rows.append({"formation": form, "source_book": source, "stock_plays": list(plays),
                     "status": "CONFIRMED_INSTALLED" if confirmed else "MANUAL_INSTALL_AND_CONFIRM_REQUIRED"})
    return {
        "formations": rows,
        "audibles": [{**a, "status": "DRAFT_NEEDS_IN_GAME_VERIFICATION", "arm_slot": None}
                     for a in proposal.get("suggested_audibles") or []],
        "macros": [{"name": m["name"], "base_pairs": m.get("base_pairs") or [],
                    "settings": m.get("settings") or [], "source_ids": m.get("source_ids") or [],
                    "status": "DRAFT_NEEDS_IN_GAME_VERIFICATION", "activation": "NOT_ARMED",
                    "arm_slot": None, "controls": "UNKNOWN: verify in Madden"}
                   for m in proposal.get("macro_blueprints") or []],
        "installation": "Install every listed stock play from its exact source, then stage and confirm_installed with the proposal ID.",
        "arming": "Separately verify audible/controller mappings and macro settings in Madden; attest installed active slots before arming. No slot is assigned by this proposal.",
    }


def pregame_input_diagnostic(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the portfolio actually knows before kickoff.

    Live snap fields are missing here on purpose. A historical defensive
    tendency is not the coverage that will be on the field.
    """
    prov = provenance or {}
    roster = prov.get("roster") or {}
    defense = prov.get("opponent_defense") or {}
    roster_state = str(roster.get("state") or "unknown")
    defense_state = str(defense.get("state") or "unknown")
    known: list[str] = []
    inferred: list[str] = []
    missing: list[str] = []
    if roster_state == "verified":
        known.append("roster")
    else:
        missing.append("roster")
    if defense_state == "inferred":
        inferred.append("opponent_tendency")
    elif defense_state == "inferred_low_sample":
        inferred.append("opponent_tendency_low_sample")
    else:
        missing.append("opponent_tendency")
    missing.extend([
        "quarter", "game_clock", "score", "down", "distance", "yardline",
        "timeouts", "red_zone", "current_coverage",
    ])
    return {
        "stage": "pregame",
        "known": known,
        "inferred": inferred,
        "missing": missing,
        "current_coverage": "missing",
        "roster_state": roster_state,
        "opponent_defense": {
            "state": defense_state,
            "sample_size": defense.get("sample_size"),
            "inferred_look": defense.get("inferred_look") if defense_state == "inferred" else None,
            "note": defense.get("note"),
        },
        "note": (
            "Pregame has no live snap. Historical defensive tendency is not "
            "current coverage, and it is not copied forward as this snap's shell."
        ),
    }


def plan_inspection(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Complete pre-install view. The fingerprint is the confirm-time check."""
    book = proposal.get("book") or {}
    formations = book.get("formations") or {}
    sources = book.get("formation_sources") or {}
    inventory_id = proposal.get("inventory_id") or book.get("inventory_id")
    limit = proposal.get("max_formations")
    opponent = proposal.get("opponent_id") or "cpu"
    proposal_id = proposal.get("proposal_id")
    listed = [
        {
            "formation": form,
            "source_book": sources.get(form),
            "plays": list(plays),
        }
        for form, plays in formations.items()
    ]
    listed.sort(key=lambda row: row["formation"])
    n_plays = sum(len(row["plays"]) for row in listed)
    limit_text = str(limit if limit is not None else DEFAULT_MAX_FORMATIONS)
    return {
        "status": "proposal_only_not_installed",
        "proposal_id": proposal_id,
        "inventory_id": inventory_id,
        "inventory_fingerprint": inventory_id,
        "max_formations": limit,
        "formations_selected": len(listed),
        "n_plays": proposal.get("n_plays", n_plays),
        "formations": listed,
        "changes": list(proposal.get("changes") or []),
        "formation_rationales": list(proposal.get("formation_rationales") or []),
        "situation_coverage": proposal.get("situation_coverage"),
        "football_plan": proposal.get("football_plan"),
        "pregame_diagnostic": proposal.get("pregame_diagnostic"),
        "preparation_guide": proposal.get("preparation_guide"),
        "macro_blueprints": [
            {
                "name": macro.get("name"),
                "status": macro.get("status"),
                "activation": macro.get("activation"),
                "kind": macro.get("kind"),
            }
            for macro in proposal.get("macro_blueprints") or []
        ],
        "confirm_after_physical_install": {
            "stage": (
                "python3 -m cfb_coach ml offense-design "
                f"-o {opponent} --max-formations {limit_text} --stage --text"
            ),
            "confirm": (
                "python3 -m cfb_coach ml offense-design "
                f"--confirm-installed {proposal_id} --attest "
                "\"I installed every listed formation and all of its plays in the Madden editor.\""
            ),
            "fingerprint_must_match": inventory_id,
            "physical_installation_required": True,
            "attestation_required": True,
        },
        "notes": list(proposal.get("notes") or [
            "Proposal only. The live book changes only after physical installation and confirm.",
        ]),
    }


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
    action_ids = [
        str(item) for item in (blueprint.get("source_action_ids") or []) if item
    ]
    single = blueprint.get("source_action_id")
    if single and single not in action_ids:
        action_ids.insert(0, str(single))
    researched = research_db.offense_adjustments()
    matched = []
    for action_id in action_ids:
        source = next((a for a in researched if a.get("id") == action_id), None)
        if source is None:
            raise ValueError("Macro's researched source and settings have changed; re-design")
        matched.append(source)
    expected_sources: set[str] = set()
    for source in matched:
        expected_sources.update(str(item) for item in (source.get("sources") or []))
    if not matched or expected_sources != set(blueprint.get("source_ids") or []):
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
