"""Human-mode preflight for a Madden playoff game.

The check is read-mostly. It registers a missing cold-start profile and
does not change coaching mode, defense mode, or the locked book. Offense
model calls stay behind ``--ml-human-control``. Defense stays on the
established caller for a cold-start human. Adjustments that are invented
or not armed are rejected. ``--no-vod-prior`` is required.
"""

from __future__ import annotations

import random
from typing import Any

from cfb_coach.madden.human_opponents import (
    ensure_madden_opponent,
    is_cold_start_opponent,
    resolve_madden_opponent,
)
from cfb_coach.madden.model.schema import CoachingMode, ML_LATENCY_BUDGET_MS, Tri

# In-memory book for the smoke. It is not written to the database.
_SMOKE_BOOK = {
    "offense": {"Gun Bunch": ["Inside Zone"]},
    "defense": {"Nickel Over": ["Cover 4 Quarters", "Cover 3 Sky"]},
}
_SMOKE_ACTIVE = {
    "offense": ["MATCH", "C2", "O-RUN", "C3", "MAN", "RZ", "ZERO", "PROT"],
    "defense": [],
}


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def screen_adjustment_id(mid: str, *, side: str, armed_ids: set[str]) -> str:
    """``executable``, ``invented``, or ``unarmed``.

    A catalog id is executable only when it is in the armed loadout. An
    unknown name is invented. ``macro_weights.armed`` is not this gate.
    """
    token = str(mid or "").strip().upper()
    if side == "offense":
        from cfb_coach.madden.offense_macros import known

        legal = known(token)
    else:
        from cfb_coach.madden.macro_pool import pool_macro

        legal = bool(pool_macro(token))
    if not legal:
        return "invented"
    if token not in {str(item).strip().upper() for item in armed_ids}:
        return "unarmed"
    return "executable"


def screen_design_candidate(candidate: Any) -> str:
    """``executable`` only when the designer says the sheet is live.

    Installed-but-unarmed, draft, and conflicting sheets are rejected.
    """
    from cfb_coach.madden.adjustments.synthesis import NotLiveSelectable, require_live

    try:
        require_live(candidate)
    except NotLiveSelectable:
        return "rejected"
    return "executable"


def adjustment_probe() -> dict[str, str]:
    """Fixed probe. Invented and unarmed names fail. An armed catalog id passes."""
    from cfb_coach.madden.model.schema import MacroCandidate, MacroSide, MacroState

    armed = {"O-RUN"}
    draft = MacroCandidate(
        macro_id="FAKE-SHEET",
        side=MacroSide.OFF,
        state=MacroState.DRAFT,
        conflicts=(),
    )
    installed = MacroCandidate(
        macro_id="FAKE-SHEET",
        side=MacroSide.OFF,
        state=MacroState.INSTALLED,
        conflicts=(),
    )
    armed_sheet = MacroCandidate(
        macro_id="O-RUN",
        side=MacroSide.OFF,
        state=MacroState.ARMED,
        conflicts=(),
    )
    return {
        "invented": screen_adjustment_id("NOT-A-REAL-MACRO", side="offense", armed_ids=armed),
        "unarmed": screen_adjustment_id("O-RUN", side="offense", armed_ids=set()),
        "executable": screen_adjustment_id("O-RUN", side="offense", armed_ids=armed),
        "draft": screen_design_candidate(draft),
        "installed_unarmed": screen_design_candidate(installed),
        "armed_sheet": screen_design_candidate(armed_sheet),
    }


def _stored_loadout_ok(db: Any, oid: str) -> tuple[bool, str]:
    from cfb_coach.madden.macros import load_selection

    stored = load_selection(db, oid) or {}
    problems: list[str] = []
    for side in ("offense", "defense"):
        ids = [str(item).upper() for item in (stored.get(side) or [])]
        armed = set(ids)
        for mid in ids:
            kind = screen_adjustment_id(mid, side=side, armed_ids=armed)
            if kind != "executable":
                problems.append(f"{side}:{mid}:{kind}")
    if problems:
        return False, "rejected " + ", ".join(problems)
    if not stored:
        return True, "no stored loadout yet; probe rejects invented and unarmed ids"
    return True, (
        f"stored loadout executable "
        f"O {len(stored.get('offense') or [])} D {len(stored.get('defense') or [])}"
    )


def _smoke_offense(db: Any, oid: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Without the flag the call stays heuristic. With it, acceptance stays unknown."""
    from cfb_coach.madden.model import inference as ml_inference
    from cfb_coach.madden.model.call_outcome_log import acceptance_for_log
    from cfb_coach.madden.playcaller import make_call
    from cfb_coach.madden.situation import parse_madden_situation

    def _sit():
        return parse_madden_situation("1&10 my 25", default_side="offense")

    plain = make_call(
        _sit(), oid, db, rng=random.Random(0), playbook=_SMOKE_BOOK,
        active_macros=_SMOKE_ACTIVE, allow_human_ml=False,
    )
    plain_model = getattr(plain, "ml_experimental", None)
    without = _check(
        "offense_without_ml_human_control",
        plain_model is None,
        "heuristic only" if plain_model is None else "experimental call leaked without the flag",
    )
    opted = make_call(
        _sit(), oid, db, rng=random.Random(0), playbook=_SMOKE_BOOK,
        active_macros=_SMOKE_ACTIVE, allow_human_ml=True,
    )
    decision = getattr(opted, "_pending_ml_decision", None)
    accepted = None if decision is None else decision.accepted
    untouched = acceptance_for_log(touched=False)
    confirmed = acceptance_for_log(touched=True, explicit="true")
    mode = ml_inference.resolve_mode(db)
    decision_unknown = decision is None or accepted is Tri.UNKNOWN
    contract = untouched == "unknown" and confirmed == "true" and decision_unknown
    if mode is not CoachingMode.EXPERIMENTAL:
        detail = (
            "confirmation contract holds (accepted stays unknown until verify); "
            "play --ml-human-control refuses until `ml experimental`"
        )
    elif decision is None:
        detail = "flag is on but no model decision was sealed; acceptance stays unknown until verify"
    else:
        detail = "model call sealed with accepted=unknown until verify is touched"
    with_flag = _check("offense_with_ml_human_control", contract, detail)
    return without, with_flag


def _smoke_defense(db: Any, oid: str) -> dict[str, Any]:
    from cfb_coach.madden.model import defense_coordinator as defense
    from cfb_coach.madden.playcaller import make_call
    from cfb_coach.madden.situation import parse_madden_situation

    sit = parse_madden_situation("d 2&7", default_side="defense")
    call = make_call(
        sit, oid, db, rng=random.Random(1), playbook=_SMOKE_BOOK,
        active_macros=_SMOKE_ACTIVE,
    )
    blocked = is_cold_start_opponent(oid, db)
    leaked = "DEFENSE ML" in (getattr(call, "rationale", "") or "")
    mode = defense.mode(db)
    ok = blocked and not leaked and bool(call.formation) and bool(call.play)
    detail = (
        f"established caller {call.formation} — {call.play}; "
        f"persistent defense mode {mode}; experimental defense "
        + ("blocked for this cold start" if blocked and not leaked else "ACTIVATED")
    )
    return _check("defense_established_only", ok, detail)


def run_human_preflight(db: Any, raw_opponent: str) -> dict[str, Any]:
    """Run the human-mode checks. Returns a report. Does not change mode."""
    from cfb_coach.madden.model import defense_coordinator as defense
    from cfb_coach.madden.model import inference as ml_inference
    from cfb_coach.madden.model.call_outcome_log import current_run_flags
    from cfb_coach.madden.model.schema import CoachingDecision

    flags = current_run_flags()
    oid = resolve_madden_opponent(raw_opponent, db)
    checks: list[dict[str, Any]] = []
    if oid is None:
        checks.append(_check("opponent", False, f"unknown opponent {raw_opponent!r}"))
        return {"ok": False, "opponent_id": None, "checks": checks, "flags": flags}
    profile = ensure_madden_opponent(db, oid) or (db.get_opponent(oid) if db is not None else None) or {}
    arch = (profile.get("archetype") or "").lower()
    tend_rows = []
    if db is not None:
        tend_rows = list(db.get_tendencies(oid))
    seed_rows = [row for row in tend_rows if str(row["bucket"]).endswith("_seed")]
    neutral = (
        is_cold_start_opponent(oid, db)
        and arch == "unknown"
        and (profile.get("confidence") or "") == "none"
        and not (profile.get("offense") or {})
        and not ((profile.get("defense_vs_us") or {}).get("coverages") or [])
        and not seed_rows
        and (profile.get("persona_source") or "").startswith("madden cold-start")
    )
    checks.append(_check(
        "neutral_priors",
        neutral,
        "archetype unknown, no seed tendencies, no borrowed film"
        if neutral else "profile is not a neutral cold start",
    ))
    vod_ok = flags.get("no_vod_prior") is True and flags.get("freeze_vod_book") is True
    checks.append(_check(
        "no_vod_prior",
        vod_ok,
        "no_vod_prior and freeze are in effect" if vod_ok else f"flags {flags}",
    ))
    default_decision = CoachingDecision()
    checks.append(_check(
        "heuristic_default",
        default_decision.mode is CoachingMode.HEURISTIC and ML_LATENCY_BUDGET_MS == 150,
        f"default mode {default_decision.mode.value}; budget {ML_LATENCY_BUDGET_MS} ms; "
        f"db mode {ml_inference.resolve_mode(db).value}",
    ))
    without, with_flag = _smoke_offense(db, oid)
    checks.append(without)
    checks.append(with_flag)
    checks.append(_smoke_defense(db, oid))
    defense_off = defense.mode(db) == "off"
    checks.append(_check(
        "defense_mode_off",
        defense_off,
        "defense experimental is off"
        if defense_off else
        "defense experimental is on; run `ml defense-experimental --disable --db <absolute>`. "
        "John still will not use it.",
    ))
    probe = adjustment_probe()
    probe_ok = (
        probe["invented"] == "invented"
        and probe["unarmed"] == "unarmed"
        and probe["executable"] == "executable"
        and probe["draft"] == "rejected"
        and probe["installed_unarmed"] == "rejected"
        and probe["armed_sheet"] == "executable"
    )
    loadout_ok, loadout_detail = _stored_loadout_ok(db, oid)
    checks.append(_check(
        "adjustments",
        probe_ok and loadout_ok,
        f"probe {probe}; {loadout_detail}",
    ))
    return {
        "ok": all(item["ok"] for item in checks),
        "opponent_id": oid,
        "checks": checks,
        "flags": flags,
        "budget_ms": ML_LATENCY_BUDGET_MS,
    }


def format_preflight(report: dict[str, Any]) -> str:
    lines = [
        f"WILD CARD PREFLIGHT — human vs {report.get('opponent_id') or '?'}",
        f"budget_ms: {report.get('budget_ms')}",
        f"flags: {report.get('flags')}",
    ]
    for item in report.get("checks") or []:
        mark = "PASS" if item.get("ok") else "FAIL"
        lines.append(f"  {mark}  {item.get('name')}: {item.get('detail')}")
    lines.append("verdict: PASS" if report.get("ok") else "verdict: FAIL")
    return "\n".join(lines)
