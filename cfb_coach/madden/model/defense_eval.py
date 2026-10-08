"""Stage 3 defensive shadow evaluation across verified snaps and complete games.

Never assigns outcomes to unexecuted shadow recommendations. Verified means
``executed_status=identified`` and ``executed_verification=verified``.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from cfb_coach.madden.model.defense_shadow import _outcome_verified, postgame_defense_shadow_report


def _payload(row: Any) -> dict[str, Any]:
    try:
        return dict(json.loads(row["decision_json"] or "{}").get("defense_shadow") or {})
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _executed_macro(outc: Any) -> str | None:
    if outc is None:
        return None
    try:
        raw = outc["outcome_json"]
        if raw:
            oj = json.loads(raw) if isinstance(raw, str) else raw
            mac = (oj or {}).get("executed_macro")
            if mac:
                return str(mac)
    except Exception:  # noqa: BLE001
        pass
    try:
        mac = outc["executed_macro"]
        return str(mac) if mac else None
    except Exception:  # noqa: BLE001
        return None


def _concept_seen(db: Any, snap_id: str | None) -> str | None:
    """Best-effort concept lookup; snaps rows are not keyed by ml snap_id."""
    if not snap_id:
        return None
    try:
        # Prefer outcome_json / decision join payloads when present.
        outc = db.conn.execute(
            "SELECT outcome_json FROM ml_outcomes WHERE snap_id=? ORDER BY id DESC LIMIT 1",
            (snap_id,),
        ).fetchone()
        if outc is not None:
            raw = outc["outcome_json"] if hasattr(outc, "keys") else outc[0]
            if raw:
                oj = json.loads(raw) if isinstance(raw, str) else raw
                concept = (oj or {}).get("concept_seen") or (oj or {}).get("concept")
                if concept:
                    return str(concept)
    except Exception:  # noqa: BLE001
        pass
    try:
        row = db.conn.execute(
            "SELECT concept_seen FROM snaps WHERE concept_seen IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        val = row["concept_seen"] if hasattr(row, "keys") else row[0]
        return str(val) if val else None
    except Exception:  # noqa: BLE001
        return None


def _outcome_yards(outc: Any) -> float | None:
    if outc is None:
        return None
    for key in ("yards_gained", "result_yards"):
        try:
            val = outc[key]
            if val is not None:
                return float(val)
        except Exception:  # noqa: BLE001
            continue
    try:
        raw = outc["outcome_json"]
        if raw:
            oj = json.loads(raw) if isinstance(raw, str) else raw
            for key in ("yards_gained", "result_yards", "yards"):
                if (oj or {}).get(key) is not None:
                    return float(oj[key])
    except Exception:  # noqa: BLE001
        pass
    return None


def stage3_evaluation_report(
    db: Any,
    *,
    game_id: str | None = None,
    limit: int = 2000,
) -> dict[str, Any]:
    """Build a Stage 3 evaluation report across verified defensive snaps/games.

    Outcomes are attributed only to the shown (heuristic) call when execution
    is verified. Shadow alternatives never receive counterfactual credit.
    """
    base = postgame_defense_shadow_report(db, game_id=game_id, limit=limit)
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

    by_opponent: dict[str, dict[str, Any]] = {}
    by_situation: Counter[str] = Counter()
    coverage_recs: Counter[str] = Counter()
    adj_recs: Counter[str] = Counter()
    verified_plays: Counter[str] = Counter()
    verified_adjs: Counter[str] = Counter()
    concepts: Counter[str] = Counter()
    latencies: list[float] = []
    fallback_n = 0
    missing_evidence: Counter[str] = Counter()
    agree_n = 0
    disagree_examples: list[dict[str, Any]] = []
    verified_outcomes: list[dict[str, Any]] = []
    games: set[str] = set()
    opponents: set[str] = set()

    for r in rows:
        gid = r["game_id"] or r["session_id"] or "?"
        games.add(str(gid))
        payload = _payload(r)
        opp = str(
            (payload.get("opponent_summary") or {}).get("opponent_id")
            or r["session_id"]
            or "?"
        )
        opponents.add(opp)
        bucket = by_opponent.setdefault(
            opp,
            {
                "n_shadow": 0,
                "agree": 0,
                "verified": 0,
                "fallback": 0,
                "game_ids": set(),
            },
        )
        bucket["n_shadow"] += 1
        bucket["game_ids"].add(str(gid))

        reason = str(payload.get("reason") or "")
        is_fallback = "fallback" in reason.lower() or "timeout" in reason.lower()
        if is_fallback:
            fallback_n += 1
            bucket["fallback"] += 1
        for m in payload.get("missing") or []:
            missing_evidence[str(m)] += 1

        lat = payload.get("latency_ms")
        if lat is not None:
            try:
                latencies.append(float(lat))
            except (TypeError, ValueError):
                pass

        cov = payload.get("coverage_family") or "?"
        coverage_recs[str(cov)] += 1
        if payload.get("adjustment"):
            adj_recs[str(payload["adjustment"])] += 1

        heur = f"{r['heuristic_formation']}/{r['heuristic_play']}"
        shadow = f"{r['shadow_formation']}/{r['shadow_play']}"
        agreed = r["agree"] == 1 or heur == shadow
        if agreed:
            agree_n += 1
            bucket["agree"] += 1
        elif len(disagree_examples) < 12:
            disagree_examples.append(
                {
                    "snap_id": r["snap_id"],
                    "game_id": gid,
                    "heuristic": heur,
                    "shadow": shadow,
                    "shadow_adjustment": payload.get("adjustment"),
                    "reason": payload.get("reason"),
                    "outcome_attributed_to_shadow": False,
                }
            )

        # Situation key from rankings / opponent summary when present.
        sit_key = "unknown"
        opp_sum = payload.get("opponent_summary") or {}
        if opp_sum.get("evidence_quality"):
            sit_key = str(opp_sum.get("evidence_quality"))
        by_situation[sit_key] += 1

        snap_id = r["snap_id"]
        outc = None
        if snap_id:
            outc = db.conn.execute(
                "SELECT * FROM ml_outcomes WHERE snap_id=? ORDER BY id DESC LIMIT 1",
                (snap_id,),
            ).fetchone()
        if outc is not None and _outcome_verified(outc):
            bucket["verified"] += 1
            exec_play = f"{outc['executed_formation'] or '?'}/{outc['executed_play'] or '?'}"
            verified_plays[exec_play] += 1
            mac = _executed_macro(outc)
            if mac:
                verified_adjs[mac.upper()] += 1
            concept = _concept_seen(db, snap_id)
            if concept:
                concepts[str(concept)] += 1
            verified_outcomes.append(
                {
                    "snap_id": snap_id,
                    "game_id": gid,
                    "executed_play": exec_play,
                    "executed_adjustment": mac,
                    "concept_seen": concept,
                    "yards_gained": _outcome_yards(outc),
                    "shown_call": f"{r['final_formation']}/{r['final_play']}",
                    "shadow_call": shadow,
                    "shadow_agreed_with_shown": agreed,
                    "outcome_attributed_to_shadow": False,
                    "note": "Outcome linked to shown heuristic call only when verified.",
                }
            )

    # Serialize opponent game_ids sets.
    opp_out = {}
    for oid, bucket in by_opponent.items():
        opp_out[oid] = {
            "n_shadow": bucket["n_shadow"],
            "agree": bucket["agree"],
            "agree_rate": (bucket["agree"] / bucket["n_shadow"]) if bucket["n_shadow"] else None,
            "verified": bucket["verified"],
            "fallback": bucket["fallback"],
            "n_games": len(bucket["game_ids"]),
            "game_ids": sorted(bucket["game_ids"])[:20],
        }

    n = len(rows)
    lat_summary = None
    if latencies:
        ordered = sorted(latencies)
        lat_summary = {
            "n": len(ordered),
            "mean_ms": round(sum(ordered) / len(ordered), 2),
            "p50_ms": ordered[len(ordered) // 2],
            "p95_ms": ordered[max(0, int(len(ordered) * 0.95) - 1)],
            "max_ms": ordered[-1],
            "over_budget_150ms": sum(1 for x in ordered if x > 150.0),
        }

    # Predictive quality / calibration only when verified outcomes support it.
    calibration = None
    if len(verified_outcomes) >= 8:
        # Agreement with shown play among verified: crude reliability of shadow
        # as an advisor (not causal claim).
        agree_v = sum(1 for o in verified_outcomes if o["shadow_agreed_with_shown"])
        calibration = {
            "n_verified": len(verified_outcomes),
            "shadow_agree_with_shown_among_verified": agree_v,
            "shadow_agree_rate_among_verified": round(agree_v / len(verified_outcomes), 3),
            "note": (
                "Not a causal win-rate. Measures how often the shadow pick matched "
                "the shown heuristic among verified executions only."
            ),
        }

    return {
        "kind": "stage3_defense_evaluation",
        "n_shadow_defense_calls": n,
        "n_games": len(games),
        "n_opponents": len(opponents),
        "game_ids": sorted(games)[:40],
        "opponent_breakdown": opp_out,
        "situation_evidence_breakdown": dict(by_situation),
        "heuristic_vs_shadow": {
            "agree": agree_n,
            "disagree": n - agree_n,
            "agree_rate": (agree_n / n) if n else None,
            "disagree_examples": disagree_examples,
        },
        "coverage_family_recommendations": dict(coverage_recs.most_common(20)),
        "adjustment_recommendations": dict(adj_recs.most_common(20)),
        "verified_executed_plays": dict(verified_plays.most_common(30)),
        "verified_executed_adjustments": dict(verified_adjs.most_common(20)),
        "opponent_concepts_observed": dict(concepts.most_common(20)),
        "verified_outcome_examples": verified_outcomes[:16],
        "latency": lat_summary,
        "fallback_rate": (fallback_n / n) if n else None,
        "fallback_count": fallback_n,
        "missing_evidence": dict(missing_evidence.most_common(20)),
        "calibration": calibration,
        "verified_summary": {
            "linked_outcomes": base.get("linked_outcomes"),
            "verified_executions_linked": base.get("verified_executions_linked"),
            "unknown_executions": base.get("unknown_executions"),
            "user_used_shown_play": base.get("user_used_shown_play"),
            "verified_adjustment_usage": base.get("verified_adjustment_usage"),
        },
        "note": (
            "Outcomes are never credited to an unexecuted shadow alternative. "
            "Shadow did not control the live call. Calibration is descriptive only."
        ),
    }
