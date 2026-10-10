"""Opt-in, evidence-aware Madden defensive coordinator (Sprint 16 pilot).

Never fabricates a verified defensive result. Selection is allowed ONLY from the
installed defensive book; no research-only macro can enter a live call.
This is an assistant: it does not operate Madden or Xbox controls.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden import catalog, playbook, research_db
from cfb_coach.madden.defense_select import (
    COUNTERS, FAMILIES, FAMILY_USER_JOB, MIX, call_family, eligible_formations, mix_key,
)
from cfb_coach.madden.situation import Situation, concept_family
from cfb_coach.opponents import is_cpu_opponent

SCHEMA = "madden.defense.coordinator.v1"
MODE_META = "ml_defense_coordinator_mode.v1"
MODEL_META = "ml_defense_coordinator_artifact.v1"
MAX_FORMATIONS = 15
LATENCY_MS = 150.0
PRIOR_STRENGTH = 8.0


def _data_path() -> Path:
    from cfb_coach.games import data_dir
    return Path(data_dir()) / "madden_ml_experimental" / "defense_coordinator.json"


def _bucket(down: Any, distance: Any, *, red_zone: bool = False, goal_line: bool = False,
            two_minute: bool = False) -> str:
    if goal_line:
        return "goal_line"
    if red_zone:
        return "red_zone"
    if two_minute:
        return "two_minute"
    try:
        down, distance = int(down), int(distance)
    except (ValueError, TypeError):
        return "early"
    if down in (2, 3, 4) and distance <= 3:
        return "short"
    if (down in (3, 4) and distance >= 7) or (down == 2 and distance >= 12):
        return "long"
    if down in (3, 4):
        return "medium"
    return "early"


def train_defense(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit shrinkage stop rates, exclusively from verified defensive executions."""
    from cfb_coach.madden.model.dataset import supervised_rows
    stats: dict[str, list[int]] = {}
    context: dict[str, list[int]] = {}
    plays: dict[str, list[int]] = {}
    n = 0
    for row in supervised_rows(rows):
        if str(row.get("side") or "").lower() != "defense":
            continue
        play = str(row.get("executed_play") or row.get("action_play") or "")
        fam = call_family(play)
        if fam not in FAMILIES or str(row.get("success") or "").lower() not in ("true", "false"):
            continue
        ok = int(str(row["success"]).lower() == "true")
        key = _bucket(row.get("down"), row.get("distance"),
                      red_zone=bool(row.get("red_zone")), goal_line=bool(row.get("goal_line")))
        for dest, tag in ((stats, fam), (context, key + "|" + fam), (plays, play)):
            count = dest.setdefault(tag, [0, 0])
            count[0] += ok
            count[1] += 1
        n += 1
    return {
        "schema": SCHEMA, "type": "defense_stop_model",
        "supervised_defensive_snaps": n,
        "evidence_quality": "empirical" if n >= 40 else ("limited" if n >= 10 else "prior_driven"),
        "families": stats, "contexts": context, "plays": plays,
        "observational_only": True,
        "note": "Stop outcomes are observations, not proven counterfactual effects.",
    }


def save_model(model: Mapping[str, Any], path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dict(model), sort_keys=True, indent=2), encoding="utf-8")
    return dest


def load_model(path: str | Path | None = None) -> dict[str, Any] | None:
    dest = Path(path) if path else _data_path()
    if not dest.is_file():
        return None
    raw = json.loads(dest.read_text(encoding="utf-8"))
    return raw if raw.get("schema") == SCHEMA and raw.get("type") == "defense_stop_model" else None


def _rate(count: Sequence[int] | None) -> tuple[float, int]:
    if not count:
        return 0.5, 0
    won, total = int(count[0]), int(count[1])
    return (won + PRIOR_STRENGTH * 0.5) / (total + PRIOR_STRENGTH), total


def _score(
    formation: str, play: str, sit: Situation, model: Mapping[str, Any] | None,
    *,
    recent: Sequence[tuple[str, str]] = (),
    eligible: set[str] | None = None,
) -> dict[str, Any]:
    family = call_family(play)
    ctx = mix_key(sit)
    from cfb_coach.game_score import shift_defense_mix
    # Score/clock-aware mix adjusts risk, while the defensive trained model
    # still provides the primary empirical stop-rate evidence.
    mix = shift_defense_mix(dict(MIX[ctx]), sit)
    baseline = mix.get(family or "", 0.025)
    learned, n_family = _rate((model or {}).get("families", {}).get(family or ""))
    specific, n_ctx = _rate((model or {}).get("contexts", {}).get(ctx + "|" + str(family)))
    play_rate, n_play = _rate((model or {}).get("plays", {}).get(play))
    # Conservative prior; more verified executions can overcome it.
    score = 0.20 * math.log(max(0.02, baseline)) + 1.10 * (learned - 0.5)
    score += 0.75 * (specific - 0.5) + 0.24 * (play_rate - 0.5)
    if eligible and formation not in eligible:
        score -= 0.18
    if sit.goal_line and any(x in formation.lower() for x in ("dime", "dollar", "quarter")):
        score -= 0.38
    if ctx == "long" and any(x in formation.lower() for x in ("goal line", "4-4")):
        score -= 0.34
    recent_four = list(recent)[-4:]
    score -= 0.13 * recent_four.count((formation, play))
    score -= 0.035 * sum(form == formation for form, _ in recent_four)
    if recent_four and call_family(recent_four[-1][1]) == family:
        score -= 0.04
    # A previous snap is NOT a current offensive concept. Only use a current
    # pre-snap live, explicitly observed concept (rare for defensive player).
    concept = concept_family(getattr(sit, "concept_hint", None))
    current = getattr(sit, "concept_source", None) == "live"
    if current and concept in COUNTERS and family:
        score += 0.25 * COUNTERS[concept].get(family, 0.0)
    return {
        "formation": formation, "play": play, "family": family or "unknown",
        "score": round(score, 6),
        "prior_share": baseline, "verified_family_n": n_family,
        "verified_context_n": n_ctx, "verified_play_n": n_play,
        "current_offensive_concept_used": concept if current else None,
    }


def rank_defense(
    sit: Situation, book: Mapping[str, Sequence[str]], model: Mapping[str, Any] | None = None,
    *, recent: Sequence[tuple[str, str]] = (),
) -> list[dict[str, Any]]:
    """Score the ENTIRE confirmed menu, not a small fixed set."""
    eligible = set(eligible_formations(sit, dict(book)))
    ranks = [
        _score(form, play, sit, model, recent=recent, eligible=eligible)
        for form, plays in book.items() for play in plays
    ]
    ranks.sort(key=lambda r: (-r["score"], r["formation"], r["play"]))
    return ranks


def _stable_choice(rows: list[dict[str, Any]], session_id: str, snap_seq: int) -> dict[str, Any]:
    top = rows[0]["score"]
    near = [r for r in rows if top - r["score"] <= 0.035][:10]
    if len(near) == 1:
        return near[0]
    digest = hashlib.sha256(f"{session_id}:{snap_seq}:defense-v1".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    weights = [math.exp((r["score"] - top) / 0.025) for r in near]
    return rng.choices(near, weights=weights, k=1)[0]


def design_defense(*, max_formations: int = MAX_FORMATIONS,
                   catalogue: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
                   model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Propose whole formations from source stock books (no automatic installation)."""
    if not 1 <= max_formations <= MAX_FORMATIONS:
        raise ValueError("max_formations must be 1..15")
    books = catalogue if catalogue is not None else {
        book: catalog.book_formations("defense", book)
        for book in catalog.book_names("defense")
    }
    # Consider distinct formation/source pairs, and all of their actual plays.
    candidates = []
    for source, forms in books.items():
        for formation, ps in forms.items():
            calls = list(dict.fromkeys(str(p) for p in ps))
            families = {call_family(p) for p in calls} & set(FAMILIES)
            if not calls or not families:
                continue
            # Estimate coverage across short / normal / passing situations.
            # Call source and legality are checked again on confirmation.
            score = 0.0
            for context, weight in (
                ("early", 1.0), ("short", 0.7), ("long", 1.0),
                ("red_zone", 0.65), ("goal_line", 0.55),
                ("two_minute", 0.65),
            ):
                best_family = max(
                    families,
                    key=lambda f: (
                        MIX[context].get(f, 0.0) +
                        0.50 * (_rate((model or {}).get("contexts", {}).get(context + "|" + f))[0] - 0.5)
                    ),
                )
                evidence, n = _rate(
                    (model or {}).get("contexts", {}).get(context + "|" + best_family)
                )
                weight_conf = n / (n + PRIOR_STRENGTH)
                score += weight * (
                    MIX[context].get(best_family, 0.0)
                    + 0.30 * weight_conf * (evidence - 0.5)
                )
            candidates.append({
                "formation": formation, "source_book": source,
                "plays": calls, "families": sorted(families), "base_score": round(score, 5),
            })
    candidates.sort(key=lambda r: (-r["base_score"], r["formation"], r["source_book"]))
    chosen = []
    used = set()
    covered: dict[str, int] = {}
    while len(chosen) < max_formations:
        available = [r for r in candidates if r["formation"] not in used]
        if not available:
            break
        def gain(r: Mapping[str, Any]) -> float:
            return float(r["base_score"]) + sum(
                0.32 / (1 + covered.get(f, 0)) for f in r["families"]
            )
        winner = max(available, key=lambda r: (gain(r), r["base_score"], r["formation"]))
        chosen.append(winner)
        used.add(winner["formation"])
        for fam in winner["families"]:
            covered[fam] = covered.get(fam, 0) + 1
    if not chosen:
        raise ValueError("No catalogued defensive formations with recognized calls")
    forms = {r["formation"]: r["plays"] for r in chosen}
    sources = {r["formation"]: r["source_book"] for r in chosen}
    return {
        "schema": SCHEMA, "mode": "proposal_only",
        "formations": forms, "formation_sources": sources,
        "n_formations": len(forms), "n_plays": sum(map(len, forms.values())),
        "family_coverage": covered,
        "choices": [{k:v for k,v in r.items() if k != "plays"} for r in chosen],
        "model_evidence_quality": (model or {}).get("evidence_quality", "prior_driven"),
        "requires_physical_editor_installation": True,
        "note": "No installed defensive book was modified. Verify all source-book plays in Madden.",
    }


def mode(db: Any) -> str:
    return str(db.get_meta(MODE_META) or "off") if db is not None else "off"


def set_mode(db: Any, new_mode: str) -> None:
    if new_mode not in ("off", "experimental"):
        raise ValueError("Expected off or experimental")
    db.set_meta(MODE_META, new_mode)


def live_pick(
    sit: Situation, db: Any, opponent_id: str, book: Mapping[str, Sequence[str]], *,
    session_id: str = "", snap_seq: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not book:
        raise ValueError("Missing confirmed defensive book")
    art_path = db.get_meta(MODEL_META) if db is not None else None
    model = load_model(art_path or None)
    from cfb_coach import ingame
    records = ingame.session_records(db, session_id, "defense") if db is not None and session_id else []
    recent = [(r.formation, r.play) for r in records[-20:]]
    ranks = rank_defense(sit, book, model, recent=recent)
    if not ranks:
        raise ValueError("No recognizable legal defensive calls")
    chosen = _stable_choice(ranks, session_id, snap_seq)
    if (time.perf_counter() - started) * 1000.0 > LATENCY_MS:
        raise TimeoutError("Defensive decision exceeded 150 ms budget")
    return {
        "selected": chosen, "top": ranks[:8],
        "n_candidates": len(ranks),
        "model_quality": (model or {}).get("evidence_quality", "prior_driven"),
        "mode": "experimental", "model_primary": True,
        "confidence_note": "Prior-driven unless sufficient verified defensive outcomes exist",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def maybe_apply_defense(
    call: Any, sit: Situation, opponent_id: str, db: Any,
    book: Mapping[str, Sequence[str]],
) -> Any:
    """Opt-in for human-user defense only; timeout/error falls back to old caller."""
    if db is None or mode(db) != "experimental" or is_cpu_opponent(opponent_id):
        return call
    # Existing researched active macro/adjustment triggers are not yet jointly
    # calibrated against this new defensive model. Keep the established
    # caller's already-armed action instead of silently discarding it.
    if getattr(call, "macro", None) or getattr(call, "adjustment", None):
        return call
    try:
        from cfb_coach.madden.playcaller import MaddenCall
        from cfb_coach.madden.data import user_job_for
        from cfb_coach.madden.model.defense_macro_lab import compatible_verified_macros
        extras = getattr(sit, "extras", None) or {}
        session = str(extras.get("session_id") or "defense-pilot")
        seq = int(extras.get("snap_seq") or 0)
        ranked = live_pick(sit, db, opponent_id, book, session_id=session, snap_seq=seq)
        winner = ranked["selected"]
        macro_info = None
        macro_name = None
        if extras.get("live_macros", True) is not False:
            # Generated macros are callable only with explicit attestation,
            # matching installed play and a genuinely observed live concept.
            concept = concept_family(getattr(sit, "concept_hint", None))
            if getattr(sit, "concept_source", None) == "live" and concept:
                for macro in compatible_verified_macros(db, opponent_id, winner["formation"], winner["play"], concept):
                    macro_name = macro["name"]
                    macro_info = {
                        "name": macro_name, "buttons": f"LB → {macro_name}",
                        "key": " · ".join(f'{s["setting"]}: {s["value"]}' for s in macro["settings"] if s["value"] != "Default")[:135],
                        "why": f"Verified custom defense macro vs currently observed {concept}",
                    }
                    break
        rationale = (
            f"DEFENSE ML [{ranked['model_quality']}] {winner['family']} "
            f"({ranked['n_candidates']} legal calls, {ranked['elapsed_ms']} ms); "
            "context, formation fit, verified stop rates and recent exposure"
        )
        return MaddenCall(
            "defense", winner["formation"], winner["play"],
            macro_name or "No adj", user_job_for(winner["play"]) or FAMILY_USER_JOB.get(winner["family"], "User hook"),
            rationale, macro=macro_name, macro_info=macro_info,
        )
    except Exception:  # fail closed: legacy safe caller stays unchanged
        return call


# Defensive custom-playbook proposals use the same explicit installed/confirmed
# boundary as the offensive designer. Nothing is installed on "stage".
PENDING_KEY = "ml_defense_design_pending.v1"
HISTORY_KEY = "ml_defense_design_history.v1"


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True).encode()).hexdigest()


def stage_design(db: Any, *, max_formations: int = MAX_FORMATIONS) -> dict[str, Any]:
    artifact = load_model(db.get_meta(MODEL_META) or None)
    proposal = design_defense(max_formations=max_formations, model=artifact)
    applied = playbook.load_books(db).get("defense") or {}
    proposal["expected_applied_hash"] = _identity(applied)
    proposal["expected_applied_revision"] = int(applied.get("rev") or 0)
    proposal["proposal_id"] = _identity(proposal)
    db.set_meta(PENDING_KEY, json.dumps(proposal, sort_keys=True))
    return {
        "proposal_id": proposal["proposal_id"],
        "n_formations": proposal["n_formations"], "n_plays": proposal["n_plays"],
        "status": "staged_only",
        "requires_physical_editor_installation": True,
    }


def pending_design(db: Any) -> dict[str, Any] | None:
    try:
        return json.loads(db.get_meta(PENDING_KEY) or "") or None
    except (TypeError, ValueError):
        return None


def confirm_design(db: Any, proposal_id: str, attestation: str) -> dict[str, Any]:
    if len((attestation or "").strip()) < 30:
        raise ValueError("Explicit actual Madden playbook installation attestation required")
    staged = pending_design(db)
    if staged is None or staged.get("proposal_id") != proposal_id:
        raise ValueError("No staged defensive design with that ID")
    unsigned = {k: v for k, v in staged.items() if k != "proposal_id"}
    if _identity(unsigned) != proposal_id:
        raise ValueError("Defensive proposal has been modified since staging")
    state = playbook._load_state(db)
    current = (state.get("applied") or {}).get("defense") or {}
    if _identity(current) != staged["expected_applied_hash"]:
        raise ValueError("Installed defensive book changed; re-design")
    for formation, plays in staged["formations"].items():
        source = staged["formation_sources"].get(formation)
        available = catalog.book_formations("defense", source or "").get(formation) or []
        if len(set(plays)) != len(plays) or set(plays) != set(available):
            raise ValueError(f"Unverified source-book play list for formation: {formation}")
    old_history = []
    try:
        old_history = json.loads(db.get_meta(HISTORY_KEY) or "[]")
    except (TypeError, ValueError):
        pass
    new_book = {
        "side": "defense", "mode": "custom",
        "name": "ML Designed Defense (custom)", "source_book": None,
        "trimmed": True, "formations": staged["formations"],
        "formation_sources": staged["formation_sources"],
        "audibles": {}, "core": list(staged["formations"]),
        "rev": int(current.get("rev") or 0) + 1,
        "reason": "User installed complete model-ranked defensive formations",
        "locked_ts": datetime_now_iso(),
    }
    old_history.append({"id": proposal_id, "book": current, "new_book": new_book})
    state["applied"]["defense"] = new_book
    state["pending"].pop("defense", None)
    playbook._save_state(db, state)
    db.set_meta(HISTORY_KEY, json.dumps(old_history[-15:], sort_keys=True))
    db.set_meta(PENDING_KEY, "")
    # Changing a book requires physically rearming and re-verifying custom macros.
    try:
        keys = db.conn.execute(
            "SELECT key FROM meta WHERE key LIKE 'ml_defense_approved_macros.v1:%'"
        ).fetchall()
        for item in keys:
            db.set_meta(str(item[0]), "")
    except Exception:
        pass
    return {
        "confirmed": True, "proposal_id": proposal_id,
        "installed_formations": len(new_book["formations"]),
        "installed_plays": sum(map(len, new_book["formations"].values())),
        "generated_macros_armed": 0,
    }


def datetime_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
