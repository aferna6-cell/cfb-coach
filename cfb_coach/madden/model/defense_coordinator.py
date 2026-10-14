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
    """Default artifact file beside ``db.default_db_path()``.

    The directory comes from that function, which honors ``CFB_COACH_DB``.
    """
    from cfb_coach.db import default_db_path

    return default_db_path().parent / "madden_ml_experimental" / "defense_coordinator.json"


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
    from cfb_coach.madden.model.defense_audit import audit_defensive_rows
    from cfb_coach.madden.model.defense_intelligence import (
        verified_human_rows, fit_opponent_offense, fit_opponent_outcome_risks,
    )
    from cfb_coach.madden.model.defense_knowledge import fit_opponent_profiles
    from cfb_coach.madden.model.defense_shrinkage import fit_shrinkage

    audit = audit_defensive_rows(rows)
    shrinkage = fit_shrinkage(rows)
    verified = [r for r in verified_human_rows(rows) if str(r.get("side") or "") == "defense"]
    n = int(shrinkage.get("supervised_defensive_snaps") or 0)
    return {
        "schema": SCHEMA, "type": "defense_stop_model",
        "verification_policy": "strict_human_verified_execution.v1",
        "supervised_defensive_snaps": n,
        "evidence_quality": shrinkage.get("evidence_quality", "prior_driven"),
        "families": shrinkage.get("families") or {},
        "contexts": shrinkage.get("contexts") or {},
        "plays": shrinkage.get("plays") or {},
        "family_utility": shrinkage.get("family_utility") or {},
        "context_utility": shrinkage.get("context_utility") or {},
        "play_utility": shrinkage.get("play_utility") or {},
        "personnel_utility": shrinkage.get("personnel_utility") or {},
        "opp_formation_utility": shrinkage.get("opp_formation_utility") or {},
        "opponent_family_utility": shrinkage.get("opponent_family_utility") or {},
        "supervised_by_game": shrinkage.get("supervised_by_game") or {},
        "model_kind": shrinkage.get("model_kind"),
        "objective": shrinkage.get("objective"),
        "artifact_version": shrinkage.get("artifact_version"),
        "utility_labeled_snaps": shrinkage.get("utility_labeled_snaps") or 0,
        "explosive_allowed": shrinkage.get("explosive_allowed") or 0,
        "touchdowns_allowed": shrinkage.get("touchdowns_allowed") or 0,
        "training_audit": {
            "candidate_defensive_snaps": audit["candidate_defensive_snaps"],
            "supervised_play_specific": audit["supervised_play_specific"],
            "excluded": audit["excluded"],
            "by_game": audit["by_game"],
            "john_supervised": audit["john_supervised"],
            "non_john_supervised": audit["non_john_supervised"],
        },
        "opponent_tendencies": fit_opponent_offense(verified),
        "opponent_outcome_risks": fit_opponent_outcome_risks(rows),
        "opponent_profiles": fit_opponent_profiles(verified),
        "verified_human_games": len({str(r["game_id"]) for r in verified}),
        "observational_only": True,
        "independent_of_offense_artifact": True,
        "post_snap_not_used_as_live": True,
        "note": shrinkage.get("note") or (
            "Stop outcomes are observations, not proven counterfactual effects."
        ),
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
    return raw if (
        raw.get("schema") == SCHEMA
        and raw.get("type") == "defense_stop_model"
        and raw.get("verification_policy") == "strict_human_verified_execution.v1"
    ) else None


def _rate(count: Sequence[int] | None) -> tuple[float, int]:
    if not count:
        return 0.5, 0
    won, total = int(count[0]), int(count[1])
    return (won + PRIOR_STRENGTH * 0.5) / (total + PRIOR_STRENGTH), total


# One-hot tokens are football personnel and set labels. They are not editor
# values and they are not verified Madden mechanics.
_PERSONNEL_TOKENS = ("10", "11", "12", "13", "20", "21", "22")
# ``empty`` is the empty-backfield set: a recorded package, not a missing value.
_SET_TOKENS = ("bunch", "stack", "trips", "tight", "spread", "heavy", "condensed", "empty")
MIN_HYBRID_SNAPS = 40


def _missing_token() -> dict[str, Any]:
    return {"missing": 1, "empty": 0, "unknown": 0, "one_hot": {}}


def _encode_token(value: Any, tokens: tuple[str, ...]) -> tuple[dict[str, Any], float]:
    """One-hot a personnel or set token.

    ``None``, ``""``, and whitespace-only are missing: ``missing`` is 1 and
    there is no one-hot. The literal token ``empty`` is the empty-backfield
    set and keeps its own one-hot with ``missing`` 0 when that token is in
    ``tokens``.
    """
    if value is None:
        return _missing_token(), 0.0
    text = str(value).strip()
    if text == "":
        return _missing_token(), 0.0
    known = {token.casefold(): token for token in tokens}
    folded = text.casefold()
    if folded in known:
        name = known[folded]
        index = tokens.index(name)
        return {"missing": 0, "empty": 0, "unknown": 0, "one_hot": {name: 1}}, 0.001 * (index + 1)
    return {"missing": 0, "empty": 0, "unknown": 1, "one_hot": {}}, -0.001


def personnel_features(sit: Any) -> dict[str, Any]:
    """One-hot ``opp_personnel`` and ``opp_set`` plus a missing flag.

    Attributes are read with ``getattr`` because a situation may not have
    them. Both null contributes exactly 0. A blank or whitespace-only value
    is missing, the same as null. The literal set token ``empty`` is a
    one-hot, not a missing value. The encoding is never verified knowledge.
    """
    personnel = getattr(sit, "opp_personnel", None)
    opp_set = getattr(sit, "opp_set", None)
    if personnel is None and opp_set is None:
        blank = {"missing": 1, "empty": 0, "unknown": 0, "one_hot": {}}
        return {
            "delta": 0.0,
            "verified": False,
            "personnel": blank,
            "set": blank,
        }
    personnel_bits, personnel_delta = _encode_token(personnel, _PERSONNEL_TOKENS)
    set_bits, set_delta = _encode_token(opp_set, _SET_TOKENS)
    return {
        "delta": personnel_delta + set_delta,
        "verified": False,
        "personnel": personnel_bits,
        "set": set_bits,
    }


def _score(
    formation: str, play: str, sit: Situation, model: Mapping[str, Any] | None,
    *,
    recent: Sequence[tuple[str, str]] = (),
    eligible: set[str] | None = None,
    opponent_id: str = "",
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
    from cfb_coach.madden.model.defense_intelligence import score_defensive_knowledge
    from cfb_coach.madden.model.defense_knowledge import explain_call, live_verified_adjustment
    from cfb_coach.madden.model.defense_shrinkage import utility_delta
    intel = score_defensive_knowledge(
        family=family, sit=sit, opponent_id=opponent_id,
        tendencies=(model or {}).get("opponent_tendencies"),
        risks=(model or {}).get("opponent_outcome_risks"),
    )
    verified = live_verified_adjustment(sit, play)
    personnel = personnel_features(sit)
    utility = utility_delta(model, play, sit, opponent_id=opponent_id)
    score += intel["delta"] + float(verified["delta"]) + personnel["delta"]
    score += float(utility.get("delta") or 0.0)
    profiles = ((model or {}).get("opponent_profiles") or {}).get("opponents") or {}
    profile = profiles.get(opponent_id) or {
        "ready": False, "current_snap_known": False, "live_score_delta": 0.0,
    }
    return {
        "formation": formation, "play": play, "family": family or "unknown",
        "score": round(score, 6),
        "prior_share": baseline, "verified_family_n": n_family,
        "verified_context_n": n_ctx, "verified_play_n": n_play,
        "current_offensive_concept_used": intel["live_concept"],
        "football_intelligence": intel,
        "verified_knowledge": verified,
        "opponent_profile": profile,
        "personnel": personnel,
        "utility": utility,
        "why": explain_call(sit, formation, play, family, verified),
    }


def rank_defense(
    sit: Situation, book: Mapping[str, Sequence[str]], model: Mapping[str, Any] | None = None,
    *, recent: Sequence[tuple[str, str]] = (), opponent_id: str = "",
) -> list[dict[str, Any]]:
    """Score the ENTIRE confirmed menu, not a small fixed set."""
    eligible = set(eligible_formations(sit, dict(book)))
    ranks = [
        _score(form, play, sit, model, recent=recent, eligible=eligible,
               opponent_id=opponent_id)
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


DECISION_ATTR = "_pending_ml_decision"
# Same near-tie window and temperature as ``_stable_choice``. Logging reads
# this after the draw; it does not draw again.
_NEAR_WINDOW = 0.035
_NEAR_TEMPERATURE = 0.025


def _sampling_facts(
    rows: Sequence[Mapping[str, Any]], chosen: Mapping[str, Any],
) -> tuple[Any, float | None, Any]:
    """Propensity of ``chosen`` under the draw ``_stable_choice`` already made.

    A one-row window is deterministic with probability 1. A wider window is
    the exact softmax weight of the shown formation and play. The function
    does not construct a ``random.Random`` and does not change the pick.
    """
    from cfb_coach.madden.model.schema import PropensityMethod, PropensityOverrideReason

    top = rows[0]["score"]
    near = [r for r in rows if top - r["score"] <= _NEAR_WINDOW][:10]
    if len(near) <= 1:
        return PropensityMethod.DETERMINISTIC, 1.0, None
    weights = [math.exp((r["score"] - top) / _NEAR_TEMPERATURE) for r in near]
    total = sum(weights)
    matched = [
        weight for row, weight in zip(near, weights)
        if row["formation"] == chosen["formation"] and row["play"] == chosen["play"]
    ]
    if not matched or total <= 0:
        return PropensityMethod.OVERRIDE, None, PropensityOverrideReason.UNEXPRESSIBLE
    return PropensityMethod.EXACT, sum(matched) / total, None


def _book_candidate(book: Mapping[str, Sequence[str]], formation: Any, play: Any) -> Any:
    from cfb_coach.madden.model.schema import Possession, candidate_in_book

    try:
        return candidate_in_book(
            book, side=Possession.DEFENSE, formation=str(formation), play=str(play),
        )
    except (TypeError, ValueError):
        return None


def _stamp_decision(call: Any, decision: Any, db: Any, extras: Mapping[str, Any] | None) -> None:
    """Attach a decision. A logging failure must not replace the call."""
    try:
        setattr(call, DECISION_ATTR, decision)
    except Exception:
        return
    snap_id = None if not isinstance(extras, Mapping) else extras.get("snap_id")
    if db is None or not snap_id or decision is None:
        return
    try:
        db.log_ml_decision(decision, agree=None)
    except Exception:
        return


def _model_decision(
    *,
    book: Mapping[str, Sequence[str]],
    formation: str,
    play: str,
    method: Any,
    propensity: float | None,
    reason: Any,
    session_id: str,
    snap_seq: int,
    elapsed_ms: float | None,
    heuristic: Any = None,
    macro_name: str | None = None,
    snap_id: str | None = None,
    game_id: str | None = None,
) -> Any:
    from cfb_coach.madden.model.schema import (
        AdjustmentRef,
        CoachingDecision,
        CoachingMode,
        MLStatus,
        ML_LATENCY_BUDGET_MS,
        PolicySource,
        Possession,
        PropensityMethod,
    )

    final = _book_candidate(book, formation, play)
    override = method is PropensityMethod.OVERRIDE
    macro = None
    if macro_name:
        macro = AdjustmentRef(side=Possession.DEFENSE, macro_id=macro_name)
    return CoachingDecision(
        latency_ms=elapsed_ms,
        latency_ms_model=elapsed_ms,
        inference_budget_ms=ML_LATENCY_BUDGET_MS,
        mode=CoachingMode.EXPERIMENTAL,
        effective_mode=CoachingMode.EXPERIMENTAL,
        game_id=game_id,
        snap_id=snap_id,
        session_id=session_id or None,
        snap_seq=snap_seq,
        policy_version=SCHEMA,
        policy_source=PolicySource.MODEL,
        heuristic_pick=heuristic,
        final_pick=final,
        shadow_pick=final,
        macro=macro,
        behavior_propensity=propensity,
        propensity_method=method,
        propensity_override=override,
        propensity_override_reason=reason if override else None,
        shadow_status=MLStatus.OK,
        fell_back=False,
        ml_status_detail="defense experimental pick; sampling seed is sha256(session:snap:defense-v1)",
    )


def _fallback_decision(
    *,
    book: Mapping[str, Sequence[str]],
    call: Any,
    status: Any,
    elapsed_ms: float | None,
    session_id: str | None,
    snap_seq: int | None,
    snap_id: str | None,
    game_id: str | None,
    detail: str,
    coaching_mode: Any = None,
) -> Any:
    from cfb_coach.madden.model.schema import (
        CoachingDecision,
        CoachingMode,
        ML_LATENCY_BUDGET_MS,
        PolicySource,
        PropensityMethod,
    )

    heur = _book_candidate(book, getattr(call, "formation", None), getattr(call, "play", None))
    selected_mode = coaching_mode or CoachingMode.EXPERIMENTAL
    logged = detail if "fallback_source=" in detail else f"{detail}; fallback_source=select_defense"
    return CoachingDecision(
        latency_ms=elapsed_ms,
        latency_ms_model=elapsed_ms,
        inference_budget_ms=ML_LATENCY_BUDGET_MS,
        mode=selected_mode,
        effective_mode=CoachingMode.HEURISTIC,
        mode_downgrade_reason=logged,
        game_id=game_id,
        snap_id=snap_id,
        session_id=session_id or None,
        snap_seq=snap_seq,
        policy_version=SCHEMA,
        policy_source=PolicySource.HEURISTIC,
        heuristic_pick=heur,
        final_pick=heur,
        propensity_method=PropensityMethod.UNKNOWN,
        shadow_status=status,
        fell_back=True,
        fallback_reason=status,
        ml_status_detail=logged,
    )


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
    from cfb_coach.madden.model.defense_knowledge import (
        context_coverage, formation_packages, verified_tags,
    )
    forms = {r["formation"]: r["plays"] for r in chosen}
    sources = {r["formation"]: r["source_book"] for r in chosen}
    packages = formation_packages(chosen)
    return {
        "schema": SCHEMA, "mode": "proposal_only",
        "formations": forms, "formation_sources": sources,
        "n_formations": len(forms), "n_plays": sum(map(len, forms.values())),
        "family_coverage": covered,
        "choices": [{k:v for k,v in r.items() if k != "plays"} for r in chosen],
        "model_evidence_quality": (model or {}).get("evidence_quality", "prior_driven"),
        "requires_physical_editor_installation": True,
        "situation_packages": packages,
        "context_coverage": context_coverage(packages, proposal_formations=list(forms)),
        "knowledge_tags": [
            {
                "id": tag["id"], "label": tag["label"], "confidence": tag["confidence"],
                "sources": list(tag["sources"]),
            }
            for tag in verified_tags()
        ],
        "hypothesis_tags_scored": [],
        "note": "No installed defensive book was modified. Verify all source-book plays in Madden.",
    }


def mode(db: Any) -> str:
    return str(db.get_meta(MODE_META) or "off") if db is not None else "off"


def set_mode(db: Any, new_mode: str) -> None:
    if new_mode not in ("off", "experimental", "shadow", "hybrid"):
        raise ValueError("Expected off, experimental, shadow, or hybrid")
    db.set_meta(MODE_META, new_mode)


def live_pick(
    sit: Situation, db: Any, opponent_id: str, book: Mapping[str, Sequence[str]], *,
    session_id: str = "", snap_seq: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not book:
        raise ValueError("Missing confirmed defensive book")
    art_path = db.get_meta(MODEL_META) if db is not None else None
    from cfb_coach.madden.model.inference import cached_defense_model

    model = cached_defense_model(
        session_id,
        lambda: load_model(art_path or None),
        path=str(art_path) if art_path else "",
    )
    from cfb_coach import ingame
    records = ingame.session_records(db, session_id, "defense") if db is not None and session_id else []
    recent = [(r.formation, r.play) for r in records[-20:]]
    ranks = rank_defense(sit, book, model, recent=recent,
                         opponent_id=opponent_id)
    if not ranks:
        raise ValueError("No recognizable legal defensive calls")
    chosen = _stable_choice(ranks, session_id, snap_seq)
    if (time.perf_counter() - started) * 1000.0 > LATENCY_MS:
        raise TimeoutError("Defensive decision exceeded 150 ms budget")
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    method, propensity, reason = _sampling_facts(ranks, chosen)
    try:
        decision = _model_decision(
            book=book, formation=chosen["formation"], play=chosen["play"],
            method=method, propensity=propensity, reason=reason,
            session_id=session_id, snap_seq=snap_seq, elapsed_ms=elapsed,
        )
    except Exception:
        decision = None
    return {
        "selected": chosen, "top": ranks[:8],
        "tendencies": (model or {}).get("opponent_tendencies") or {},
        "n_candidates": len(ranks),
        "model_quality": (model or {}).get("evidence_quality", "prior_driven"),
        "mode": "experimental", "model_primary": True,
        "confidence_note": "Prior-driven unless sufficient verified defensive outcomes exist",
        "elapsed_ms": elapsed,
        "decision": decision,
    }


def count_verified_defensive_snaps(rows: Sequence[Mapping[str, Any]] | None) -> int:
    from cfb_coach.madden.model.defense_intelligence import is_counted_defensive_snap
    return sum(1 for row in rows or () if is_counted_defensive_snap(row))


def defense_hybrid_gate(entry: Any, rows: Sequence[Mapping[str, Any]], *, seed: int) -> Any:
    """Refuse hybrid promotion below 40 verified defensive snaps.

    The refusal happens before any metric is computed. At 40 or more, the
    standard held-out gate in ``evaluate`` runs.
    """
    from dataclasses import replace

    from cfb_coach.madden.model.evaluate import evaluate
    from cfb_coach.madden.model.schema import GateResult

    if count_verified_defensive_snaps(rows) < MIN_HYBRID_SNAPS:
        return replace(
            entry,
            gate=GateResult.INSUFFICIENT,
            gate_passed=False,
            metrics=(),
            seed=seed,
        )
    return evaluate(entry, rows, seed=seed)


def _installed_book(book: Mapping[str, Sequence[str]]) -> dict[str, dict[str, list[str]]]:
    return {
        "offense": {},
        "defense": {str(formation): [str(play) for play in plays] for formation, plays in book.items()},
    }


def _load_stored_records(db: Any) -> list[Any]:
    """Read designer drafts already stored on this db. Does not generate any."""
    from cfb_coach.madden.adjustments.candidates import candidate_from_json
    from cfb_coach.madden.adjustments.synthesis import DRAFT_META_KEY, DesignRecord

    if db is None:
        return []
    try:
        payload = json.loads(db.get_meta(DRAFT_META_KEY) or "")
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    records = []
    for raw in payload.get("candidates") or []:
        try:
            candidate = candidate_from_json(raw)
        except (TypeError, ValueError):
            continue
        if not candidate.design_record_id:
            continue
        records.append(DesignRecord(
            design_record_id=candidate.design_record_id,
            candidate=candidate,
            downside=None,
            uncertainty_note=None,
            catalog_comparison=(),
            install_instructions=(),
        ))
    return records


def _baseline_candidate(formation: str, play: str) -> Any:
    """No-adjustment option. Live legality is the designer's baseline check."""
    from cfb_coach.madden.adjustments.synthesis import design_record_id_for
    from cfb_coach.madden.model.schema import (
        MacroCandidate, MacroJointFeatures, MacroSide, MacroState,
    )

    macro_id = "NO_ADJUSTMENT"
    return MacroCandidate(
        macro_id=macro_id,
        design_record_id=design_record_id_for(macro_id),
        version="1",
        side=MacroSide.DEF,
        state=MacroState.ELIGIBLE,
        formation=formation,
        play=play,
        adjustment_primitives=(),
        editor_fields=None,
        tactical_purpose="Call the installed play with no pre-snap adjustment.",
        situations=(),
        conflicts=(),
        joint=MacroJointFeatures(compatibility=1.0, action_count=0),
        is_no_adjustment=True,
    )


def _action_options(
    records: Sequence[Any],
    formation: str,
    play: str,
    installed: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[dict[str, Any]]:
    """ARMED+ macros for this play, plus the no-adjustment baseline.

    Legality is ``require_live`` / ``is_baseline_live_legal`` from the
    adjustment designer. Editor-verified and draft sheets do not compete.
    """
    from cfb_coach.madden.adjustments.candidates import MacroSide, is_live_selectable
    from cfb_coach.madden.adjustments.synthesis import (
        NotLiveSelectable, is_baseline_live_legal, require_arm_cap, require_live,
    )

    require_arm_cap(list(records), MacroSide.DEF)
    options: list[dict[str, Any]] = []
    for row in records:
        candidate = row.candidate
        if candidate.formation != formation or candidate.play != play:
            continue
        if candidate.side is not MacroSide.DEF:
            continue
        try:
            require_live(candidate, installed=installed)
        except NotLiveSelectable:
            continue
        if candidate.is_no_adjustment:
            if is_baseline_live_legal(candidate, installed):
                options.append({"id": "NO_ADJUSTMENT", "kind": "none", "candidate": candidate})
            continue
        if is_live_selectable(candidate.state):
            options.append({
                "id": candidate.macro_id or "",
                "kind": "macro",
                "candidate": candidate,
            })
    if not any(option["kind"] == "none" for option in options):
        baseline = _baseline_candidate(formation, play)
        if is_baseline_live_legal(baseline, installed):
            require_live(baseline, installed=installed)
            options.append({"id": "NO_ADJUSTMENT", "kind": "none", "candidate": baseline})
    options.sort(key=lambda option: (0 if option["kind"] == "none" else 1, option["id"]))
    return options


def _combine_stages(
    play_method: Any,
    play_p: float | None,
    action_method: Any,
    action_p: float | None,
) -> dict[str, Any]:
    from cfb_coach.madden.model.schema import PropensityMethod

    stages = (
        {"name": "play", "method": play_method, "propensity": play_p},
        {"name": "action", "method": action_method, "propensity": action_p},
    )
    methods = (play_method, action_method)
    if any(method is PropensityMethod.OVERRIDE for method in methods):
        overall = PropensityMethod.OVERRIDE
    elif all(method is PropensityMethod.EXACT for method in methods):
        overall = PropensityMethod.EXACT
    else:
        overall = PropensityMethod.DETERMINISTIC
    if (
        play_p is None or action_p is None
        or play_p <= 0 or action_p <= 0
    ):
        product = None
        log_product = None
    else:
        product = float(play_p) * float(action_p)
        log_product = math.log(product)
    return {
        "stages": stages,
        "behavior_propensity": product,
        "log_product": log_product,
        "method": overall,
    }


def joint_rank(
    sit: Situation,
    book: Mapping[str, Sequence[str]],
    records: Sequence[Any] = (),
    *,
    session_id: str = "",
    snap_seq: int = 0,
    opponent_id: str = "",
    model: Mapping[str, Any] | None = None,
    recent: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Two-stage play-then-action rank for shadow and hybrid.

    Stage one is the existing near-tie draw. A one-row window is
    deterministic with propensity 1. Stage two always includes no adjustment
    and any armed-or-later macro on the chosen play. The logged probability
    is the product. It is exact only when every stage is exact.
    """
    from cfb_coach.madden.model.schema import PropensityMethod, PropensityOverrideReason

    installed = _installed_book(book)
    ranks = rank_defense(sit, book, model, recent=recent, opponent_id=opponent_id)
    if not ranks:
        raise ValueError("No recognizable legal defensive calls")
    chosen = _stable_choice(ranks, session_id, snap_seq)
    play_method, play_p, play_reason = _sampling_facts(ranks, chosen)
    options = _action_options(records, chosen["formation"], chosen["play"], installed)
    if not options:
        raise ValueError("no live defensive action for the chosen play")
    if len(options) == 1:
        action = options[0]
        action_method, action_p, action_reason = PropensityMethod.DETERMINISTIC, 1.0, None
    else:
        digest = hashlib.sha256(
            f"{session_id}:{snap_seq}:defense-action-v1:{chosen['formation']}:{chosen['play']}".encode()
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        action = rng.choices(options, weights=[1.0] * len(options), k=1)[0]
        action_method, action_p, action_reason = PropensityMethod.EXACT, 1.0 / len(options), None
    combined = _combine_stages(play_method, play_p, action_method, action_p)
    reason = play_reason or action_reason
    if combined["method"] is PropensityMethod.OVERRIDE and reason is None:
        reason = PropensityOverrideReason.UNEXPRESSIBLE
    winner = {
        "formation": chosen["formation"],
        "play": chosen["play"],
        "family": chosen.get("family"),
        "macro_id": None if action["kind"] == "none" else action["id"],
        "kind": action["kind"],
        "id": action["id"],
    }
    return {
        **combined,
        "reason": reason,
        "winner": winner,
        "actions": [{"id": option["id"], "kind": option["kind"]} for option in options],
        "play": {"formation": chosen["formation"], "play": chosen["play"]},
    }


def _joint_decision(
    *,
    book: Mapping[str, Sequence[str]],
    call: Any,
    ranking: Mapping[str, Any],
    coaching_mode: Any,
    policy_source: Any,
    session_id: str | None,
    snap_seq: int | None,
    snap_id: str | None,
    game_id: str | None,
    elapsed_ms: float | None,
    shown_formation: str,
    shown_play: str,
    shown_macro: str | None,
) -> Any:
    from cfb_coach.madden.model.schema import (
        AdjustmentRef,
        CoachingDecision,
        ML_LATENCY_BUDGET_MS,
        MLStatus,
        Possession,
        PropensityMethod,
    )

    winner = ranking["winner"]
    method = ranking["method"]
    override = method is PropensityMethod.OVERRIDE
    propensity = None if override else ranking["behavior_propensity"]
    macro = None
    if shown_macro:
        macro = AdjustmentRef(side=Possession.DEFENSE, macro_id=shown_macro)
    return CoachingDecision(
        latency_ms=elapsed_ms,
        latency_ms_model=elapsed_ms,
        inference_budget_ms=ML_LATENCY_BUDGET_MS,
        mode=coaching_mode,
        effective_mode=coaching_mode,
        game_id=game_id,
        snap_id=snap_id,
        session_id=session_id or None,
        snap_seq=snap_seq,
        policy_version=SCHEMA,
        policy_source=policy_source,
        heuristic_pick=_book_candidate(
            book, getattr(call, "formation", None), getattr(call, "play", None),
        ),
        final_pick=_book_candidate(book, shown_formation, shown_play),
        shadow_pick=_book_candidate(book, winner["formation"], winner["play"]),
        macro=macro,
        behavior_propensity=propensity if policy_source.value == "model" else None,
        propensity_method=method if policy_source.value == "model" else PropensityMethod.UNKNOWN,
        propensity_override=override if policy_source.value == "model" else False,
        propensity_override_reason=ranking.get("reason") if override and policy_source.value == "model" else None,
        shadow_status=MLStatus.OK,
        fell_back=False,
        ml_status_detail="defense joint rank; stages are on joint_ranking",
    )


def _apply_logged_defense(
    call: Any,
    sit: Situation,
    opponent_id: str,
    db: Any,
    book: Mapping[str, Sequence[str]],
    current_mode: str,
) -> Any:
    """Shadow logs the joint call. Hybrid shows it only at 40 verified snaps."""
    from cfb_coach.madden.model.schema import CoachingMode, MLStatus, PolicySource

    try:
        extras = getattr(sit, "extras", None) or {}
        session = str(extras.get("session_id") or "defense-pilot")
        seq = int(extras.get("snap_seq") or 0)
        snap_id = extras.get("snap_id")
        game_id = extras.get("game_id")
    except Exception:
        return call
    coaching_mode = CoachingMode.SHADOW if current_mode == "shadow" else CoachingMode.HYBRID
    if current_mode == "hybrid":
        supplied = extras.get("verified_defensive_snaps")
        if supplied is None:
            artifact = load_model(db.get_meta(MODEL_META) or None) if db is not None else None
            snaps = int((artifact or {}).get("supervised_defensive_snaps") or 0)
        else:
            snaps = count_verified_defensive_snaps(supplied)
        if snaps < MIN_HYBRID_SNAPS:
            try:
                _stamp_decision(call, _fallback_decision(
                    book=book, call=call, status=MLStatus.LOW_CONFIDENCE, elapsed_ms=None,
                    session_id=session, snap_seq=seq, snap_id=snap_id, game_id=game_id,
                    detail="fewer than 40 verified defensive snaps",
                    coaching_mode=CoachingMode.HYBRID,
                ), db, extras)
            except Exception:
                pass
            return call
    started = time.perf_counter()
    try:
        records = list(extras.get("design_records") or []) or _load_stored_records(db)
        ranking = joint_rank(
            sit, book, records,
            session_id=session, snap_seq=seq, opponent_id=opponent_id,
        )
        winner = ranking["winner"]
        if current_mode == "shadow":
            decision = _joint_decision(
                book=book, call=call, ranking=ranking, coaching_mode=coaching_mode,
                policy_source=PolicySource.HEURISTIC, session_id=session, snap_seq=seq,
                snap_id=snap_id, game_id=game_id,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                shown_formation=str(getattr(call, "formation", "") or ""),
                shown_play=str(getattr(call, "play", "") or ""),
                shown_macro=getattr(call, "macro", None),
            )
            _stamp_decision(call, decision, db, extras)
            try:
                setattr(call, "joint_ranking", ranking)
            except Exception:
                pass
            return call
        from cfb_coach.madden.data import user_job_for
        from cfb_coach.madden.playcaller import MaddenCall

        macro_name = winner["macro_id"]
        label = macro_name or "No adj"
        result = MaddenCall(
            "defense", winner["formation"], winner["play"], label,
            user_job_for(winner["play"]) or "User hook",
            f"DEFENSE HYBRID {winner['formation']} / {winner['play']} / {label}",
            macro=macro_name,
        )
        result.joint_ranking = ranking
        decision = _joint_decision(
            book=book, call=call, ranking=ranking, coaching_mode=CoachingMode.HYBRID,
            policy_source=PolicySource.MODEL, session_id=session, snap_seq=seq,
            snap_id=snap_id, game_id=game_id,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            shown_formation=winner["formation"], shown_play=winner["play"],
            shown_macro=macro_name,
        )
        _stamp_decision(result, decision, db, extras)
        return result
    except Exception:
        if current_mode == "hybrid":
            try:
                _stamp_decision(call, _fallback_decision(
                    book=book, call=call, status=MLStatus.EXCEPTION,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    session_id=session, snap_seq=seq, snap_id=snap_id, game_id=game_id,
                    detail="defense hybrid joint rank raised",
                    coaching_mode=CoachingMode.HYBRID,
                ), db, extras)
            except Exception:
                pass
        return call


def maybe_apply_defense(
    call: Any, sit: Situation, opponent_id: str, db: Any,
    book: Mapping[str, Sequence[str]],
) -> Any:
    """Opt-in for human-user defense only; timeout/error falls back to old caller."""
    if db is None or is_cpu_opponent(opponent_id):
        return call
    current_mode = mode(db)
    if current_mode in ("shadow", "hybrid"):
        return _apply_logged_defense(call, sit, opponent_id, db, book, current_mode)
    if current_mode != "experimental":
        return call
    try:
        extras = getattr(sit, "extras", None) or {}
        session = str(extras.get("session_id") or "defense-pilot")
        seq = int(extras.get("snap_seq") or 0)
        snap_id = extras.get("snap_id")
        game_id = extras.get("game_id")
    except Exception:
        return call
    # Experimental mode does not go live on an empty or prior-only file.
    # Do not fit a model here. The established caller is the legal play.
    # Unset MODEL_META is missing even if a default JSON happens to exist.
    art_path = db.get_meta(MODEL_META) if db is not None else None
    model = load_model(art_path) if art_path else None
    snaps = int((model or {}).get("supervised_defensive_snaps") or 0)
    if model is None or snaps < 1:
        try:
            from cfb_coach.madden.model.schema import MLStatus
            _stamp_decision(call, _fallback_decision(
                book=book, call=call, status=MLStatus.MODEL_MISSING, elapsed_ms=0.0,
                session_id=session, snap_seq=seq, snap_id=snap_id, game_id=game_id,
                detail="missing defense artifact or zero verified snaps",
            ), db, extras)
        except Exception:
            pass
        return call
    # Existing researched active macro/adjustment triggers are not yet jointly
    # calibrated against this new defensive model. Keep the established
    # caller's already-armed action instead of silently discarding it.
    if getattr(call, "macro", None) or getattr(call, "adjustment", None):
        from cfb_coach.madden.model.schema import MLStatus
        try:
            _stamp_decision(call, _fallback_decision(
                book=book, call=call, status=MLStatus.CIRCUIT_OPEN, elapsed_ms=None,
                session_id=session, snap_seq=seq, snap_id=snap_id, game_id=game_id,
                detail="kept existing armed macro or adjustment",
            ), db, extras)
        except Exception:
            pass
        return call
    started = time.perf_counter()
    try:
        from cfb_coach.madden.playcaller import MaddenCall
        from cfb_coach.madden.data import user_job_for
        from cfb_coach.madden.model.schema import MLStatus, PropensityMethod
        ranked = live_pick(sit, db, opponent_id, book, session_id=session, snap_seq=seq)
        from cfb_coach.madden.model.defense_action_policy import select_defensive_action
        # Evaluate the whole playbook for calls and a bounded top set for
        # executable changes, comparing every action with doing nothing.
        options = []
        for candidate in ranked["top"]:
            plan = select_defensive_action(
                sit, db, opponent_id,
                candidate["formation"], candidate["play"],
                tendencies=ranked.get("tendencies"),
                allowed_macros=extras.get("live_macros", True) is not False,
                installed_book=book,
            )
            options.append((candidate["score"] + plan["influence"], candidate, plan))
        if not options:
            _stamp_decision(call, _fallback_decision(
                book=book, call=call, status=MLStatus.INVALID_OUTPUT,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                session_id=session, snap_seq=seq, snap_id=snap_id, game_id=game_id,
                detail="defense experimental produced no call",
            ), db, extras)
            return call
        # Near-tie defense sampling stays available without forcing a rotation.
        # The shown call is this deterministic joint sort, not the near-tie draw
        # stored on live_pick["selected"].
        options.sort(key=lambda item: (-item[0], item[1]["formation"], item[1]["play"]))
        best_total, winner, plan = options[0]
        macro_name = plan["id"] if plan["kind"] == "macro" else None
        macro_info = None
        if macro_name:
            macro = plan["macro"]
            macro_info = {
                "name": macro_name, "buttons": f"LB → {macro_name}",
                "key": " · ".join(
                    f'{row["setting"]}: {row["value"]}'
                    for row in macro["settings"] if row["value"] != "Default"
                )[:135],
                "why": plan["why"],
            }
        adjustment = None
        if plan["kind"] == "adjustment":
            adjustment = {
                "id": plan["id"], "side": "defense", "kind": "adjustment",
                "label": plan["label"], "buttons": plan["buttons"],
                "why": plan["why"], "sources": plan.get("research_sources") or [],
                "executable": plan.get("executable", True),
            }
        plays = [str(item) for item in (book.get(winner["formation"]) or [])]
        if str(winner["play"]) not in plays:
            _stamp_decision(call, _fallback_decision(
                book=book, call=call, status=MLStatus.INVALID_OUTPUT,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                session_id=session, snap_seq=seq, snap_id=snap_id, game_id=game_id,
                detail="illegal play not in installed book",
            ), db, extras)
            return call
        from types import SimpleNamespace

        from cfb_coach.madden.model.dual_control import adjustment_block_reason

        blocked = None
        if extras.get("require_trained_defense_artifact"):
            blocked = adjustment_block_reason(
                SimpleNamespace(macro=macro_name, adjustment=adjustment),
                side="defense",
                armed=list(extras.get("defense_armed_ids") or []),
                db=db,
                opponent_id=opponent_id,
            )
        if blocked:
            _stamp_decision(call, _fallback_decision(
                book=book, call=call, status=MLStatus.INVALID_OUTPUT,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                session_id=session, snap_seq=seq, snap_id=snap_id, game_id=game_id,
                detail=blocked,
            ), db, extras)
            return call
        if (time.perf_counter() - started) * 1000.0 > LATENCY_MS:
            _stamp_decision(call, _fallback_decision(
                book=book, call=call, status=MLStatus.TIMEOUT,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                session_id=session, snap_seq=seq, snap_id=snap_id, game_id=game_id,
                detail="defense experimental exceeded 150 ms",
            ), db, extras)
            return call
        intelligence = winner.get("football_intelligence") or {}
        observation = intelligence.get("live_concept")
        tendency_state = intelligence.get("opponent_tendency") or {}
        verified = winner.get("verified_knowledge") or {}
        verified_note = ""
        if verified.get("scored_ids"):
            verified_note = " " + str(verified.get("why") or "")
        note = (
            f"DEFENSE ML [{ranked['model_quality']}] {winner['family']} "
            f"({ranked['n_candidates']} installed plays; action {plan['id']}; "
            f"concept={observation or 'unknown'}; "
            f"trend={tendency_state.get('family') if tendency_state.get('ready') else 'uncertain'}); "
            f"knowledge={intelligence.get('principle') or 'unknown'}"
            f"{verified_note}"
        )
        result = MaddenCall(
            "defense", winner["formation"], winner["play"],
            macro_name or (adjustment["label"] if adjustment else "No adj"),
            user_job_for(winner["play"]) or FAMILY_USER_JOB.get(winner["family"], "User hook"),
            note, macro=macro_name, macro_info=macro_info, adjustment=adjustment,
        )
        heur = _book_candidate(book, getattr(call, "formation", None), getattr(call, "play", None))
        try:
            decision = _model_decision(
                book=book, formation=winner["formation"], play=winner["play"],
                method=PropensityMethod.DETERMINISTIC, propensity=1.0, reason=None,
                session_id=session, snap_seq=seq,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                heuristic=heur, macro_name=macro_name, snap_id=snap_id, game_id=game_id,
            )
            _stamp_decision(result, decision, db, extras)
        except Exception:
            pass
        return result
    except TimeoutError:
        try:
            from cfb_coach.madden.model.schema import MLStatus
            _stamp_decision(call, _fallback_decision(
                book=book, call=call, status=MLStatus.TIMEOUT,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                session_id=session, snap_seq=seq, snap_id=snap_id, game_id=game_id,
                detail="defense experimental exceeded 150 ms",
            ), db, extras)
        except Exception:
            pass
        return call
    except Exception:  # fail closed: legacy safe caller stays unchanged
        try:
            from cfb_coach.madden.model.schema import MLStatus
            _stamp_decision(call, _fallback_decision(
                book=book, call=call, status=MLStatus.EXCEPTION,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                session_id=session, snap_seq=seq, snap_id=snap_id, game_id=game_id,
                detail="defense experimental raised",
            ), db, extras)
        except Exception:
            pass
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


def _validated_design(db: Any, proposal_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    return staged, current, new_book


def _install_instructions(new_book: Mapping[str, Any]) -> list[str]:
    lines = [
        "This command did not write the defensive book.",
        "In Madden, open the defensive playbook editor and install each formation below with every listed play.",
    ]
    for formation, plays in (new_book.get("formations") or {}).items():
        lines.append(f"Install {formation}: {', '.join(plays)}.")
    lines.append(
        "After the book is installed in Madden, applying it here is a separate Aidan-only step and requires an attestation."
    )
    return lines


def confirm_design(db: Any, proposal_id: str, attestation: str = "") -> dict[str, Any]:
    """Return a staged diff and install instructions. Does not write the book.

    ``attestation`` is accepted so existing callers keep working. A short
    attestation does not raise and does not install anything.
    """
    del attestation
    _staged, current, new_book = _validated_design(db, proposal_id)
    return {
        "confirmed": False,
        "wrote_book": False,
        "proposal_id": proposal_id,
        "diff": {"from": current, "to": new_book},
        "install_instructions": _install_instructions(new_book),
        "generated_macros_armed": 0,
        "installed_formations": 0,
        "installed_plays": 0,
    }


def apply_design(
    db: Any,
    proposal_id: str,
    attestation: str,
    *,
    operator: str,
) -> dict[str, Any]:
    """Write a previously staged design. Only Aidan may apply it."""
    if operator != "Aidan":
        raise ValueError("Applying a defensive design is limited to Aidan")
    if len((attestation or "").strip()) < 30:
        raise ValueError("Explicit actual Madden playbook installation attestation required")
    _staged, current, new_book = _validated_design(db, proposal_id)
    old_history = []
    try:
        old_history = json.loads(db.get_meta(HISTORY_KEY) or "[]")
    except (TypeError, ValueError):
        pass
    old_history.append({"id": proposal_id, "book": current, "new_book": new_book})
    state = playbook._load_state(db)
    state.setdefault("applied", {})["defense"] = new_book
    state.setdefault("pending", {}).pop("defense", None)
    playbook._save_state(db, state)
    db.set_meta(HISTORY_KEY, json.dumps(old_history[-15:], sort_keys=True))
    db.set_meta(PENDING_KEY, "")
    try:
        keys = db.conn.execute(
            "SELECT key FROM meta WHERE key LIKE 'ml_defense_approved_macros.v1:%'"
        ).fetchall()
        for item in keys:
            db.set_meta(str(item[0]), "")
    except Exception:
        pass
    return {
        "confirmed": True,
        "wrote_book": True,
        "proposal_id": proposal_id,
        "operator": operator,
        "installed_formations": len(new_book["formations"]),
        "installed_plays": sum(map(len, new_book["formations"].values())),
        "generated_macros_armed": 0,
    }


def datetime_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
