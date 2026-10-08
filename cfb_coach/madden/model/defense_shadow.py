"""Stage 3 defensive shadow advisor — recommend, never control.

Produces a defensive formation/play + optional armed Custom Adjustment
alongside the heuristic call. The live displayed call is never changed.

Activation of live control is an explicit future opt-in (see docs); this module
refuses to take authority.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from cfb_coach.madden.defense_select import COUNTERS, call_family, mix_key
from cfb_coach.madden.model import defense_ca, defense_opponent
from cfb_coach.madden.model.schema import (
    ML_LATENCY_BUDGET_MS,
    CoachingDecision,
    CoachingMode,
    MLStatus,
    PolicySource,
    Possession,
    PropensityMethod,
    Tri,
)
from cfb_coach.opponents import is_cpu_opponent

META_CONTROL = "ml_defense_shadow_control"  # must stay "off" until Stage 3 activation
META_LOG = "ml_defense_shadow_log"  # "on" (default) | "off"
PENDING_ATTR = "_pending_defense_shadow"
INFO_ATTR = "ml_defense_shadow"

# Soft scores — hierarchical shrinkage style, not a deep model.
PRIOR_STRENGTH = 6.0
HEURISTIC_BLEND = 0.2
NEAR_TIE = 0.03


@dataclass
class DefenseShadowRecommendation:
    formation: str | None
    play: str | None
    coverage_family: str | None
    adjustment: str | None
    adjustment_reason: str | None
    reason: str
    confidence: float
    evidence_quality: str
    heuristic_formation: str | None
    heuristic_play: str | None
    heuristic_adjustment: str | None
    incompatibility: str | None = None
    missing: list[str] = field(default_factory=list)
    opponent_summary: dict[str, Any] = field(default_factory=dict)
    rankings: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float | None = None
    controlled: bool = False  # always False in Stage 3 shadow

    def to_dict(self) -> dict[str, Any]:
        return {
            "formation": self.formation,
            "play": self.play,
            "coverage_family": self.coverage_family,
            "adjustment": self.adjustment,
            "adjustment_reason": self.adjustment_reason,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "evidence_quality": self.evidence_quality,
            "heuristic_formation": self.heuristic_formation,
            "heuristic_play": self.heuristic_play,
            "heuristic_adjustment": self.heuristic_adjustment,
            "incompatibility": self.incompatibility,
            "missing": list(self.missing),
            "opponent_summary": dict(self.opponent_summary),
            "rankings": list(self.rankings[:8]),
            "latency_ms": self.latency_ms,
            "controlled": False,
            "format_line": self.format_line(),
        }

    def format_line(self) -> str:
        """Fast live-call convention: Formation — Play | Adj (only if essential) | User/key."""
        form = self.formation or "?"
        play = self.play or "?"
        line = f"{form} — {play}"
        if self.adjustment:
            line += f" | {self.adjustment}"
        # User/key hint from coverage family.
        from cfb_coach.madden.defense_select import FAMILY_USER_JOB

        user = FAMILY_USER_JOB.get(self.coverage_family or "", "User the landmark")
        line += f" | {user}"
        return line


def control_enabled(db: Any) -> bool:
    """Live control must stay off until Stage 3 activation criteria are met."""
    if db is None:
        return False
    try:
        return str(db.get_meta(META_CONTROL) or "off").lower() in ("on", "1", "true", "control")
    except Exception:  # noqa: BLE001
        return False


def logging_enabled(db: Any) -> bool:
    if db is None:
        return True
    try:
        raw = db.get_meta(META_LOG)
        if raw is None or raw == "":
            return True
        return str(raw).lower() not in ("off", "0", "false")
    except Exception:  # noqa: BLE001
        return True


def refuse_activation(db: Any | None = None) -> dict[str, Any]:
    """Explicit Stage 3 activation gate — always refuses in this sprint."""
    return {
        "ok": False,
        "activated": False,
        "control": "off",
        "reason": (
            "Stage 3 defensive ML control is not enabled. Shadow advisor only. "
            "See docs/madden_ml_defense_shadow.md for readiness criteria."
        ),
        "current_meta": {
            META_CONTROL: (db.get_meta(META_CONTROL) if db is not None else None),
            META_LOG: (db.get_meta(META_LOG) if db is not None else None),
        },
    }


def _book_candidates(book: Mapping[str, Sequence[str]]) -> list[tuple[str, str, str | None]]:
    out: list[tuple[str, str, str | None]] = []
    for form, plays in (book or {}).items():
        for play in plays or []:
            out.append((form, play, call_family(play)))
    return out


def _situation_family_prior(sit: Any) -> dict[str, float]:
    key = mix_key(sit) if sit is not None else "early"
    from cfb_coach.madden.defense_select import MIX

    return dict(MIX.get(key) or MIX["early"])


def rank_defense_candidates(
    *,
    book: Mapping[str, Sequence[str]],
    sit: Any,
    opponent_model: defense_opponent.OpponentOffenseModel,
    concept_weights: Mapping[str, float] | None = None,
    heuristic: tuple[str, str] | None = None,
    research_family_boost: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank in-book defensive (formation, play) pairs for the current situation."""
    sit_prior = _situation_family_prior(sit)
    bucket = None
    try:
        from cfb_coach.scouting import down_bucket

        bucket = down_bucket(
            getattr(sit, "down", None),
            getattr(sit, "distance", None),
            getattr(sit, "yardline", None),
        )
    except Exception:  # noqa: BLE001
        bucket = None
    counters = opponent_model.expected_counters(bucket=bucket)
    concepts = dict(concept_weights or {})
    # Blend concept→counter with situational mix.
    family_score: dict[str, float] = {f: float(sit_prior.get(f, 0.0)) for f in sit_prior}
    for cov, w in counters.items():
        family_score[cov] = family_score.get(cov, 0.0) + 0.9 * float(w)
    for fam, w in concepts.items():
        base = fam if fam in COUNTERS else (
            "run" if fam in ("inside_zone", "outside_zone", "power") else fam
        )
        for cov, cw in (COUNTERS.get(base) or {}).items():
            family_score[cov] = family_score.get(cov, 0.0) + 0.5 * float(w) * float(cw)
    for cov, boost in (research_family_boost or {}).items():
        family_score[cov] = family_score.get(cov, 0.0) + float(boost)

    scored: list[dict[str, Any]] = []
    for form, play, fam in _book_candidates(book):
        base = float(family_score.get(fam or "", 0.15))
        # Shrink toward situational prior when opponent evidence is thin.
        unc = 0.7 if opponent_model.evidence_quality == "prior_driven" else 0.35
        sit_p = float(sit_prior.get(fam or "", 0.15))
        score = (1.0 - unc) * base + unc * sit_p
        if heuristic and (form, play) == heuristic:
            score += HEURISTIC_BLEND * 0.15
        scored.append(
            {
                "formation": form,
                "play": play,
                "coverage_family": fam,
                "score": score,
                "components": {
                    "situation_prior": sit_p,
                    "opponent_counter": float(counters.get(fam or "", 0.0)),
                    "raw_family": base,
                    "uncertainty": unc,
                },
            }
        )
    scored.sort(key=lambda r: (-r["score"], r["formation"], r["play"]))
    if heuristic and scored:
        top = scored[0]["score"]
        for row in scored:
            if (row["formation"], row["play"]) == heuristic and (top - row["score"]) <= NEAR_TIE:
                scored.remove(row)
                scored.insert(0, row)
                row["near_tie_kept_heuristic"] = True
                break
    for i, row in enumerate(scored, start=1):
        row["rank"] = i
        row["probability"] = 1.0 / (1.0 + pow(2.718281828, -row["score"]))
    return scored


def _research_defense_boosts() -> dict[str, float]:
    """Soft coverage-family boosts from Madden research — capped, never fabricated."""
    boosts: dict[str, float] = {}
    try:
        from cfb_coach import ai_research

        loaded = ai_research.load_research("madden27", offline=True)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            return boosts
        doc, _origin = loaded
        if not isinstance(doc, dict):
            return boosts
        for line in doc.get("meta_defense") or []:
            text = str(line).lower()
            if "quarters" in text or "cover 4" in text or "two-high" in text:
                boosts["two_high"] = min(0.08, boosts.get("two_high", 0.0) + 0.03)
            if "tampa" in text or "cover 2" in text:
                boosts["cover2"] = min(0.08, boosts.get("cover2", 0.0) + 0.03)
            if "cover 3" in text or "match" in text:
                boosts["single_high"] = min(0.08, boosts.get("single_high", 0.0) + 0.02)
            if "spy" in text or "scram" in text:
                boosts["man"] = min(0.08, boosts.get("man", 0.0) + 0.02)
            if "blitz" in text or "pressure" in text or "sim" in text:
                boosts["pressure"] = min(0.08, boosts.get("pressure", 0.0) + 0.02)
    except Exception:  # noqa: BLE001
        return boosts
    return boosts


def _fallback_rec(
    *,
    heur_form: str | None,
    heur_play: str | None,
    heur_adj: str | None,
    reason: str,
    adjustment_reason: str,
    evidence_quality: str,
    missing: list[str],
    opponent_summary: dict[str, Any] | None,
    latency_ms: float,
    incompatibility: str | None = None,
) -> DefenseShadowRecommendation:
    return DefenseShadowRecommendation(
        formation=heur_form,
        play=heur_play,
        coverage_family=call_family(heur_play),
        adjustment=None,
        adjustment_reason=adjustment_reason,
        reason=reason,
        confidence=0.0,
        evidence_quality=evidence_quality,
        heuristic_formation=heur_form,
        heuristic_play=heur_play,
        heuristic_adjustment=heur_adj,
        incompatibility=incompatibility,
        missing=missing,
        opponent_summary=dict(opponent_summary or {}),
        rankings=[],
        latency_ms=latency_ms,
        controlled=False,
    )


def build_shadow_recommendation(
    *,
    sit: Any,
    heuristic_call: Any,
    book: Mapping[str, Sequence[str]],
    armed_macros: Sequence[str],
    opponent_id: str,
    db: Any = None,
    budget_ms: float | None = None,
) -> DefenseShadowRecommendation:
    """Compute a shadow recommendation. Never mutates the heuristic call display.

    The full path (opponent model + ranking + CA validation) must finish inside
    ``budget_ms`` (default 150). On timeout or exception the heuristic call is
    preserved with an accurate fallback reason.
    """
    started = time.perf_counter()
    budget = float(ML_LATENCY_BUDGET_MS if budget_ms is None else budget_ms)

    def _elapsed() -> float:
        return (time.perf_counter() - started) * 1000.0

    heur_form = getattr(heuristic_call, "formation", None)
    heur_play = getattr(heuristic_call, "play", None)
    heur_adj = getattr(heuristic_call, "macro", None) or getattr(heuristic_call, "adj_or_macro", None)
    if heur_adj in (None, "", "No adj", "—"):
        heur_adj = None
    missing: list[str] = []
    if not book:
        missing.append("defensive playbook")
    if is_cpu_opponent(opponent_id):
        return _fallback_rec(
            heur_form=heur_form,
            heur_play=heur_play,
            heur_adj=heur_adj,
            reason="CPU opponent: defense calls are not used",
            adjustment_reason="CPU offense-only — no defense shadow",
            evidence_quality="n/a",
            missing=["cpu_offense_only"],
            opponent_summary=None,
            latency_ms=_elapsed(),
        )

    try:
        if _elapsed() > budget:
            return _fallback_rec(
                heur_form=heur_form,
                heur_play=heur_play,
                heur_adj=heur_adj,
                reason="shadow fallback to heuristic (timeout before opponent model)",
                adjustment_reason="timeout — keep heuristic",
                evidence_quality="unknown",
                missing=missing + ["timeout"],
                opponent_summary=None,
                latency_ms=_elapsed(),
            )

        model = defense_opponent.build_opponent_offense_model(db, opponent_id)
        if _elapsed() > budget:
            return _fallback_rec(
                heur_form=heur_form,
                heur_play=heur_play,
                heur_adj=heur_adj,
                reason="shadow fallback to heuristic (timeout during opponent model)",
                adjustment_reason="timeout — keep heuristic",
                evidence_quality=model.evidence_quality,
                missing=missing + ["timeout"],
                opponent_summary=model.to_dict(),
                latency_ms=_elapsed(),
            )

        concepts = defense_opponent.situation_concept_prior(sit, model)
        if not concepts:
            missing.append("opponent concept observations")

        research = _research_defense_boosts()
        heur_pair = (heur_form, heur_play) if heur_form and heur_play else None
        ranked = rank_defense_candidates(
            book=book,
            sit=sit,
            opponent_model=model,
            concept_weights=concepts,
            heuristic=heur_pair,
            research_family_boost=research,
        )
        if _elapsed() > budget:
            return _fallback_rec(
                heur_form=heur_form,
                heur_play=heur_play,
                heur_adj=heur_adj,
                reason="shadow fallback to heuristic (timeout during ranking)",
                adjustment_reason="timeout — keep heuristic",
                evidence_quality=model.evidence_quality,
                missing=missing + ["timeout"],
                opponent_summary=model.to_dict(),
                latency_ms=_elapsed(),
            )
        if not ranked:
            return _fallback_rec(
                heur_form=heur_form,
                heur_play=heur_play,
                heur_adj=heur_adj,
                reason="shadow fallback to heuristic (no candidates)",
                adjustment_reason="empty candidate pool — keep heuristic",
                evidence_quality=model.evidence_quality,
                missing=missing + ["no_candidates"],
                opponent_summary=model.to_dict(),
                latency_ms=_elapsed(),
            )

        top = ranked[0]
        # Guard: selected play must be in the applied book.
        top_form, top_play = top["formation"], top["play"]
        book_plays = list((book or {}).get(top_form) or [])
        if top_play not in book_plays:
            return _fallback_rec(
                heur_form=heur_form,
                heur_play=heur_play,
                heur_adj=heur_adj,
                reason="shadow fallback to heuristic (illegal book play)",
                adjustment_reason="selected play not in applied book",
                evidence_quality=model.evidence_quality,
                missing=missing + ["illegal_book_play"],
                opponent_summary=model.to_dict(),
                latency_ms=_elapsed(),
            )

        repeated = any(
            defense_opponent.is_repeated_concept(model, c) for c in model.repeated_concepts
        )
        strong = any(w >= 0.25 for w in concepts.values())
        if _elapsed() > budget:
            # Ranked play OK but no time for CA validation — play without adj.
            adj: dict[str, Any] = {
                "macro": None,
                "reason": "timeout before CA validation — play only",
                "incompatibility": None,
            }
        else:
            adj = defense_ca.recommend_adjustment(
                list(armed_macros or []),
                concept_families=concepts,
                heuristic_macro=str(heur_adj) if heur_adj else None,
                essential_only=not (
                    repeated
                    or strong
                    or getattr(sit, "red_zone", False)
                    or getattr(sit, "two_minute", False)
                ),
                formation=top_form,
                play=top_play,
                book=book,
            )
        # Final budget check after CA validation.
        if _elapsed() > budget and adj.get("macro"):
            adj = {
                "macro": None,
                "reason": "timeout after CA validation — dropped adjustment",
                "incompatibility": adj.get("incompatibility"),
                "candidates": adj.get("candidates") or [],
            }

        adj_id = (adj or {}).get("macro")
        margin = 0.0
        if len(ranked) > 1:
            margin = float(top["score"] - ranked[1]["score"])
        conf = max(
            0.05,
            min(
                0.85,
                0.35
                + 0.3 * (1.0 if model.evidence_quality != "prior_driven" else 0.0)
                + 0.2 * margin,
            ),
        )
        reason_bits = [
            f"answers expected {max(concepts, key=concepts.get) if concepts else 'situational mix'}",
            f"coverage={top.get('coverage_family')}",
            f"evidence={model.evidence_quality}",
        ]
        if top.get("near_tie_kept_heuristic"):
            reason_bits.append("near-tie kept heuristic play")
        if adj_id is None and (adj or {}).get("incompatibility"):
            reason_bits.append("adj withheld (incompatible/incomplete)")
        return DefenseShadowRecommendation(
            formation=top_form,
            play=top_play,
            coverage_family=top.get("coverage_family"),
            adjustment=adj_id,
            adjustment_reason=(adj or {}).get("reason"),
            reason="; ".join(reason_bits),
            confidence=conf,
            evidence_quality=model.evidence_quality,
            heuristic_formation=heur_form,
            heuristic_play=heur_play,
            heuristic_adjustment=heur_adj,
            incompatibility=(adj or {}).get("incompatibility"),
            missing=missing,
            opponent_summary=model.to_dict(),
            rankings=ranked[:8],
            latency_ms=_elapsed(),
            controlled=False,
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback_rec(
            heur_form=heur_form,
            heur_play=heur_play,
            heur_adj=heur_adj,
            reason=f"shadow fallback to heuristic (exception: {type(exc).__name__})",
            adjustment_reason="exception — keep heuristic",
            evidence_quality="unknown",
            missing=missing + ["exception"],
            opponent_summary=None,
            latency_ms=_elapsed(),
            incompatibility=str(exc)[:120],
        )


def maybe_attach_defense_shadow(
    *,
    call: Any,
    sit: Any,
    opponent_id: str,
    db: Any,
    book: dict[str, list[str]] | None = None,
    armed: Sequence[str] | None = None,
) -> Any:
    """Attach a shadow recommendation to a defense call. Never changes the call.

    No-op for CPU, offense, or when logging is disabled. Never enables control.
    """
    try:
        if control_enabled(db):
            # Hard refuse — Stage 3 not ready. Leave call unchanged and annotate.
            refuse = refuse_activation(db)
            try:
                call.ml_defense_shadow = {  # type: ignore[attr-defined]
                    "controlled": False,
                    "refused_activation": refuse,
                    "reason": refuse["reason"],
                }
            except Exception:  # noqa: BLE001
                pass
            return call
        if not logging_enabled(db):
            return call
        if is_cpu_opponent(opponent_id):
            return call
        if not str(getattr(call, "side", "") or "").startswith("d"):
            return call
        if book is None:
            from cfb_coach.madden.playbook import active_books, eligible

            raw = active_books(db, ("defense",))
            book = eligible(raw).get("defense") or {}
        if armed is None and db is not None:
            from cfb_coach.madden.macros import as_selection, load_selection

            sel = as_selection(load_selection(db, opponent_id) or {})
            armed = list(sel.get("defense") or [])
        rec = build_shadow_recommendation(
            sit=sit,
            heuristic_call=call,
            book=book or {},
            armed_macros=list(armed or []),
            opponent_id=opponent_id,
            db=db,
        )
        info = rec.to_dict()
        try:
            setattr(call, INFO_ATTR, info)
            setattr(call, PENDING_ATTR, rec)
        except Exception:  # noqa: BLE001
            pass
        return call
    except Exception:  # noqa: BLE001
        traceback.format_exc()
        return call


def commit_defense_shadow(
    db: Any,
    call: Any,
    *,
    game_id: str,
    snap_id: str,
    snap_seq: int,
    session_id: str | None = None,
) -> int | None:
    """Persist shadow recommendation with sealed snap identity. Does not change final_pick."""
    if not logging_enabled(db):
        return None
    rec = getattr(call, PENDING_ATTR, None)
    if rec is None or not isinstance(rec, DefenseShadowRecommendation):
        return None
    if control_enabled(db):
        return None
    heur_cand = None
    shadow_cand = None
    try:
        from cfb_coach.madden.model.schema import AppliedPlaybookRef, CandidatePlay, Tri

        def _cand(form: str | None, play: str | None) -> CandidatePlay | None:
            if not form or not play:
                return None
            return CandidatePlay(
                playbook=AppliedPlaybookRef(side=Possession.DEFENSE),
                formation=form,
                play=play,
                in_applied_book=Tri.TRUE,
            )

        heur_cand = _cand(rec.heuristic_formation, rec.heuristic_play)
        shadow_cand = _cand(rec.formation, rec.play)
    except Exception:  # noqa: BLE001
        pass

    decision_ts = datetime.now(timezone.utc).isoformat()
    dec = CoachingDecision(
        decision_ts=decision_ts,
        latency_ms=rec.latency_ms,
        latency_ms_model=rec.latency_ms,
        inference_budget_ms=ML_LATENCY_BUDGET_MS,
        mode=CoachingMode.SHADOW,
        effective_mode=CoachingMode.HEURISTIC,
        mode_downgrade_reason="defense_shadow_advisor_only",
        game_id=game_id,
        snap_id=snap_id,
        session_id=session_id or game_id,
        snap_seq=snap_seq,
        policy_source=PolicySource.HEURISTIC,
        heuristic_pick=heur_cand,
        final_pick=heur_cand,  # live call unchanged
        shadow_pick=shadow_cand,
        shadow_status=MLStatus.OK,
        fell_back=False,
        propensity_method=PropensityMethod.UNKNOWN,
        accepted=Tri.UNKNOWN,
    )
    # Merge shadow payload into decision_json after the standard log write.
    try:
        payload = rec.to_dict()
        payload["kind"] = "defense_shadow"
        payload["controlled"] = False
        agree = None
        if heur_cand and shadow_cand:
            agree = int(
                heur_cand.formation == shadow_cand.formation and heur_cand.play == shadow_cand.play
            )
        row_id = _log_defense_shadow(db, dec, payload, agree=agree)
        info = getattr(call, INFO_ATTR, None)
        if isinstance(info, dict):
            info = dict(info)
            info["snap_id"] = snap_id
            info["game_id"] = game_id
            info["committed"] = True
            setattr(call, INFO_ATTR, info)
        return row_id
    except Exception:  # noqa: BLE001
        return None


def _log_defense_shadow(
    db: Any, dec: CoachingDecision, payload: dict[str, Any], *, agree: int | None
) -> int | None:
    """Write ml_decisions with defense shadow payload merged into decision_json."""
    row_id = db.log_ml_decision(dec, agree=agree)
    if row_id is None:
        return None
    try:
        row = db.conn.execute(
            "SELECT decision_json FROM ml_decisions WHERE id=?", (row_id,)
        ).fetchone()
        raw = {}
        if row is not None:
            raw = json.loads(row["decision_json"] if hasattr(row, "keys") else row[0] or "{}")
        raw["defense_shadow"] = payload
        db.conn.execute(
            "UPDATE ml_decisions SET decision_json=? WHERE id=?",
            (json.dumps(raw), row_id),
        )
        db.conn.commit()
    except Exception:  # noqa: BLE001
        pass
    return int(row_id)


def _outcome_verified(outc: Any) -> bool:
    """True only when execution status is identified and verification is verified."""
    if outc is None:
        return False
    status = str(outc["executed_status"] if hasattr(outc, "keys") else "").lower()
    ver = ""
    try:
        ver = str(outc["executed_verification"] or "").lower()
    except Exception:  # noqa: BLE001
        ver = ""
    if not ver:
        try:
            raw = outc["outcome_json"] if hasattr(outc, "keys") else None
            if raw:
                payload = json.loads(raw) if isinstance(raw, str) else raw
                ver = str((payload or {}).get("executed_verification") or "").lower()
        except Exception:  # noqa: BLE001
            ver = ""
    return status == "identified" and ver == "verified"


def postgame_defense_shadow_report(
    db: Any, *, game_id: str | None = None, limit: int = 500
) -> dict[str, Any]:
    """Compare heuristic vs shadow defense recommendations. No counterfactual credit.

    ``verified_executions_linked`` counts only outcomes with explicit verified
    execution status — not merely the presence of an outcome row.
    """
    params: list[Any] = []
    sql = (
        "SELECT * FROM ml_decisions WHERE mode='shadow' "
        "AND decision_json LIKE '%defense_shadow%'"
    )
    if game_id:
        sql += " AND (game_id = ? OR session_id = ?)"
        params.extend([game_id, game_id])
    sql += " ORDER BY id ASC LIMIT ?"
    params.append(int(limit))
    rows = list(db.conn.execute(sql, params))
    n = len(rows)
    agree = 0
    used_shown_play = 0
    verified_adj_usage = 0
    linked_outcomes = 0
    verified_executions = 0
    unknown_executions = 0
    by_game: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for r in rows:
        gid = r["game_id"] or r["session_id"] or "?"
        by_game[gid] = by_game.get(gid, 0) + 1
        if r["agree"] == 1:
            agree += 1
        try:
            payload = json.loads(r["decision_json"] or "{}").get("defense_shadow") or {}
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        snap_id = r["snap_id"]
        outc = None
        if snap_id:
            outc = db.conn.execute(
                "SELECT * FROM ml_outcomes WHERE snap_id=? ORDER BY id DESC LIMIT 1",
                (snap_id,),
            ).fetchone()
        if outc is not None:
            linked_outcomes += 1
            if _outcome_verified(outc):
                verified_executions += 1
                if (outc["executed_play"] or "") == (r["final_play"] or ""):
                    used_shown_play += 1
                suggested_adj = payload.get("adjustment")
                if suggested_adj:
                    executed_macro = None
                    try:
                        raw = outc["outcome_json"]
                        if raw:
                            oj = json.loads(raw) if isinstance(raw, str) else raw
                            executed_macro = (oj or {}).get("executed_macro")
                    except Exception:  # noqa: BLE001
                        executed_macro = None
                    try:
                        if not executed_macro:
                            executed_macro = outc["executed_macro"]
                    except Exception:  # noqa: BLE001
                        pass
                    if executed_macro and str(executed_macro).upper() == str(suggested_adj).upper():
                        verified_adj_usage += 1
            else:
                unknown_executions += 1
        if len(examples) < 8:
            examples.append(
                {
                    "snap_id": snap_id,
                    "heuristic": f"{r['heuristic_formation']}/{r['heuristic_play']}",
                    "shadow": f"{r['shadow_formation']}/{r['shadow_play']}",
                    "final_shown": f"{r['final_formation']}/{r['final_play']}",
                    "adjustment": payload.get("adjustment"),
                    "reason": payload.get("reason"),
                    "evidence_quality": payload.get("evidence_quality"),
                    "executed": (outc["executed_play"] if outc is not None else None),
                    "executed_status": (
                        outc["executed_status"] if outc is not None else None
                    ),
                    "execution_verified": _outcome_verified(outc),
                    "outcome_attributed_to_shadow": False,
                }
            )
    return {
        "n_shadow_defense_calls": n,
        "agree_with_heuristic": agree,
        "agree_rate": (agree / n) if n else None,
        "linked_outcomes": linked_outcomes,
        "verified_executions_linked": verified_executions,
        "unknown_executions": unknown_executions,
        "user_used_shown_play": used_shown_play,
        "verified_adjustment_usage": verified_adj_usage,
        "game_counts": by_game,
        "anonymous_unlinked_rows_ignored": 0,
        "examples": examples,
        "note": (
            "verified_executions_linked requires executed_status=identified AND "
            "executed_verification=verified. Outcomes are never credited to an "
            "unexecuted shadow alternative. Shadow did not control the live call."
        ),
    }
