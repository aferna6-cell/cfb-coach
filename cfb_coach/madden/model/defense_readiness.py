"""Stage 3 readiness dashboard — experimental targets, not competitive proof.

Criteria are observation / control-prep gates. Passing them does not prove
defensive ML is stronger than the heuristic.
"""

from __future__ import annotations

from typing import Any

from cfb_coach.madden.model import defense_ca, defense_control, defense_eval, defense_shadow
from cfb_coach.madden.model.schema import ML_LATENCY_BUDGET_MS

# Suggested Stage 3 observation→control prep targets (experimental, not superiority).
TARGET_VERIFIED_SNAPS = 40
TARGET_HUMAN_GAMES = 3
TARGET_FALLBACK_RATE_MAX = 0.15
TARGET_P95_LATENCY_MS = float(ML_LATENCY_BUDGET_MS)


def _check(name: str, ok: bool, detail: str, *, current: Any = None, target: Any = None) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "detail": detail,
        "current": current,
        "target": target,
    }


def readiness_dashboard(
    db: Any,
    *,
    game_id: str | None = None,
    armed_macros: list[str] | None = None,
) -> dict[str, Any]:
    """Explicit progress report toward Stage 3 readiness criteria."""
    eval_report = defense_eval.stage3_evaluation_report(db, game_id=game_id)
    base = defense_shadow.postgame_defense_shadow_report(db, game_id=game_id)

    verified = int(base.get("verified_executions_linked") or 0)
    # Distinct human-opponent games: prefer opponent breakdown game counts.
    human_games = 0
    for oid, bucket in (eval_report.get("opponent_breakdown") or {}).items():
        if str(oid).lower() in ("cpu", "cpu_dynasty", "notre_dame", "?"):
            continue
        human_games += int(bucket.get("n_games") or 0)
    if human_games == 0:
        human_games = int(eval_report.get("n_games") or 0)

    lat = eval_report.get("latency") or {}
    fallback_rate = eval_report.get("fallback_rate")
    p95 = lat.get("p95_ms")
    over_budget = int(lat.get("over_budget_150ms") or 0)

    # Valid play↔adjustment combinations among recommended / default loadout.
    adj_recs = eval_report.get("adjustment_recommendations") or {}
    default_loadout = [
        "TAMPA MABLE",
        "SAFE DEEP",
        "PRESS SHADE IN",
        "LOOP MAN 0",
        "RZ COVER 2",
        "TEX 4 MAN",
        "TEX2 L CONT",
        "TEX2 R CONT",
    ]
    armed = list(armed_macros or adj_recs.keys() or default_loadout)
    if not armed:
        armed = list(default_loadout)
    valid_adj_n = 0
    control_elig_n = 0
    provenance_notes: list[str] = []
    for mid in armed[:16]:
        view = defense_ca.inspect_armed_macro(str(mid))
        if view.valid:
            valid_adj_n += 1
        if view.control_eligible:
            control_elig_n += 1
        cov = view.source_coverage or {}
        provenance_notes.append(
            f"{view.mid}: researched={cov.get('researched')} "
            f"synthesized_default={cov.get('synthesized_default')} "
            f"control_eligible={view.control_eligible}"
        )

    outcome_attr_ok = (
        verified > 0
        and all(
            not ex.get("outcome_attributed_to_shadow")
            for ex in (base.get("examples") or [])
        )
    )

    # Situation-specific suites: treated as "passing" when the eval module and
    # shadow demo path are importable and acceptance tests are the CI gate.
    # Here we surface whether we have multi-situation verified coverage.
    sit_bins = eval_report.get("situation_evidence_breakdown") or {}
    situation_tests_ok = len(sit_bins) >= 1 and verified >= 1

    checks = [
        _check(
            "verified_defensive_snaps",
            verified >= TARGET_VERIFIED_SNAPS,
            f"{verified} verified defensive snaps (identified+verified)",
            current=verified,
            target=TARGET_VERIFIED_SNAPS,
        ),
        _check(
            "distinct_human_opponent_games",
            human_games >= TARGET_HUMAN_GAMES,
            f"{human_games} distinct human-opponent game(s) with shadow logs",
            current=human_games,
            target=TARGET_HUMAN_GAMES,
        ),
        _check(
            "valid_play_adjustment_combinations",
            valid_adj_n > 0,
            f"{valid_adj_n} armed macros currently valid/complete; "
            f"{control_elig_n} provenance-ok for future control",
            current={"valid": valid_adj_n, "control_eligible": control_elig_n},
            target="≥1 valid armed CA; control requires provenance+compat",
        ),
        _check(
            "reliable_outcome_attribution",
            outcome_attr_ok or verified == 0,
            (
                "Verified outcomes attributed to shown call only; "
                "never to unexecuted shadow alternatives"
                if verified
                else "No verified outcomes yet — attribution gate pending evidence"
            ),
            current=base.get("verified_executions_linked"),
            target="verified joins; no shadow counterfactual credit",
        ),
        _check(
            "situation_specific_tests",
            situation_tests_ok or verified == 0,
            (
                f"Evidence quality bins observed: {sorted(sit_bins)}"
                if sit_bins
                else "CI situation suites (crossers/flood/RPO/scramble/…) are the test gate"
            ),
            current=list(sit_bins.keys()),
            target="passing situation suites in CI + multi-situation verified snaps",
        ),
        _check(
            "latency_and_fallback",
            (
                (fallback_rate is None or float(fallback_rate) <= TARGET_FALLBACK_RATE_MAX)
                and (p95 is None or float(p95) <= TARGET_P95_LATENCY_MS + 1e-6)
                and over_budget == 0
            )
            if lat
            else verified == 0,
            (
                f"fallback_rate={fallback_rate}, p95_ms={p95}, over_budget={over_budget}"
                if lat
                else "No latency samples yet"
            ),
            current={"fallback_rate": fallback_rate, "p95_ms": p95, "over_budget_150ms": over_budget},
            target={
                "fallback_rate_max": TARGET_FALLBACK_RATE_MAX,
                "p95_ms_max": TARGET_P95_LATENCY_MS,
            },
        ),
        _check(
            "control_path_disabled",
            not defense_shadow.control_enabled(db)
            and not defense_control.pilot_enabled(db),
            "Defensive ML control and pilot remain off; activation refuses",
            current={
                "shadow_control": defense_shadow.control_enabled(db),
                "pilot": defense_control.pilot_enabled(db),
            },
            target=False,
        ),
    ]

    passed = sum(1 for c in checks if c["ok"])
    # Observation GO when core safety + logging path is healthy; control stays NO-GO.
    observation_ready = all(
        c["ok"]
        for c in checks
        if c["name"]
        in (
            "valid_play_adjustment_combinations",
            "reliable_outcome_attribution",
            "control_path_disabled",
        )
    )
    control_ready = all(c["ok"] for c in checks) and verified >= TARGET_VERIFIED_SNAPS

    remaining = [c for c in checks if not c["ok"]]
    recommendation = _next_game_recommendation(
        verified=verified,
        human_games=human_games,
        remaining=remaining,
        observation_ready=observation_ready,
        control_ready=control_ready,
    )

    return {
        "kind": "stage3_readiness_dashboard",
        "targets": {
            "verified_defensive_snaps": TARGET_VERIFIED_SNAPS,
            "distinct_human_opponent_games": TARGET_HUMAN_GAMES,
            "fallback_rate_max": TARGET_FALLBACK_RATE_MAX,
            "p95_latency_ms_max": TARGET_P95_LATENCY_MS,
        },
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "observation_ready": observation_ready,
        "control_ready": False,  # Sprint 5.2: never true via this dashboard alone
        "control_ready_raw": control_ready,
        "control_activation": defense_control.refuse_pilot_activation(db),
        "provenance_notes": provenance_notes[:12],
        "remaining_evidence": [
            {"name": c["name"], "detail": c["detail"], "current": c["current"], "target": c["target"]}
            for c in remaining
        ],
        "next_human_opponent_recommendation": recommendation,
        "note": (
            "These are experimental readiness targets, not proof of competitive "
            "superiority. Do not merge or activate defensive control from this report."
        ),
    }


def _next_game_recommendation(
    *,
    verified: int,
    human_games: int,
    remaining: list[dict[str, Any]],
    observation_ready: bool,
    control_ready: bool,
) -> dict[str, Any]:
    del control_ready  # always refused this sprint
    need_snaps = max(0, TARGET_VERIFIED_SNAPS - verified)
    need_games = max(0, TARGET_HUMAN_GAMES - human_games)
    if not observation_ready:
        action = "fix_observation_gates"
        why = "Finish safety/provenance observation gates before another scored game."
    elif need_games > 0 or need_snaps > 0:
        action = "play_shadow_observation_game"
        why = (
            f"Need ~{need_snaps} more verified defensive snaps across "
            f"~{need_games} additional distinct human-opponent game(s). "
            "Keep ml_defense_shadow_control=off; verify execution after each snap."
        )
    elif remaining:
        action = "close_remaining_checks"
        why = "Volume targets met; close remaining readiness checks: " + ", ".join(
            c["name"] for c in remaining
        )
    else:
        action = "continue_shadow_observation"
        why = (
            "Observation targets met on paper; keep shadow-only logging for another "
            "human game to stabilize latency/fallback and situation coverage. "
            "Do not activate defensive ML control."
        )
    return {
        "action": action,
        "why": why,
        "mode": "shadow_observation_only",
        "control": "off",
        "verify_execution": True,
        "suggested_focus": [
            "human opponent (not CPU)",
            "verify executed play + adjustment when used",
            "cover 3rd-long, red zone, scramble, RPO/run looks",
            "confirm 150 ms full heuristic fallback never leaves a partial shadow pick",
        ],
    }


def format_readiness_text(report: dict[str, Any]) -> str:
    """Human-readable CLI dashboard."""
    lines = [
        "=== Stage 3 Defense Readiness Dashboard ===",
        f"Checks passed: {report.get('passed')}/{report.get('total')}",
        f"Observation ready: {report.get('observation_ready')}",
        f"Control ready: NO (sprint refuses activation)",
        "",
        "Criteria:",
    ]
    for c in report.get("checks") or []:
        mark = "PASS" if c.get("ok") else "NEED"
        lines.append(f"  [{mark}] {c['name']}: {c['detail']}")
    rem = report.get("remaining_evidence") or []
    if rem:
        lines.append("")
        lines.append("Remaining evidence:")
        for r in rem:
            lines.append(f"  - {r['name']}: current={r.get('current')} target={r.get('target')}")
    rec = report.get("next_human_opponent_recommendation") or {}
    lines.append("")
    lines.append(f"Next game: {rec.get('action')} — {rec.get('why')}")
    lines.append(f"Mode: {rec.get('mode')} | control={rec.get('control')}")
    lines.append(str(report.get("note") or ""))
    return "\n".join(lines)
