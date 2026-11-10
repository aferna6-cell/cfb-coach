"""Learned, individual Madden defensive adjustments: fail closed to NO ADJUSTMENT.

Only physically verified Xbox actions and verified prior executions may become
live candidates. No macros or unsupported controller paths. A historical
association is not a causal benefit; conservative shrinkage and support gates
prevent synthetic football priors from masquerading as learned action gains.
"""
from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

SCHEMA = "madden.defense.v2.adjustments.v1"
CONFIRM_SCHEMA = "madden.defense.v2.xbox_confirmations.v1"
MIN_ACTION_SNAPS = 12
MIN_CONSERVATIVE_GAIN = 0.03
SHRINK_N = 24.0


def catalog() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "data" / "madden27" / "executable_actions.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("schema") != "madden27.executable_actions.v1":
        raise ValueError("unknown Madden 27 executable action catalog")
    return content


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def fit_verified_adjustment_effects(conn: sqlite3.Connection, *, franchise: str) -> dict[str, Any]:
    """Train ONLY from verified own defensive executions with confirmed action IDs.

    The exact executed_adjustment_id string is a composite ID for multi-action
    packages if that is what the owner actually verified; never infer a combo
    from separately logged constituent actions.
    """
    required = {"session_id", "side", "result", "executed_status",
                "executed_verification", "executed_play", "executed_formation",
                "executed_adjustment_id", "down", "distance"}
    cols = _columns(conn, "snaps")
    if not required <= cols:
        return {"schema": SCHEMA, "counts": {}, "n": 0,
                "reason": "required executed-adjustment columns absent"}
    from cfb_coach.madden.model.defense_v1 import offense_won
    from cfb_coach.madden.model.defense_v2_pilot import context_key, _key
    counts: dict[str, list[int]] = {}
    n = 0
    sql = ("SELECT s.executed_adjustment_id,s.result,s.down,s.distance,"
           "s.executed_formation,s.executed_play FROM snaps s"
           " JOIN game_sessions g ON g.session_id=s.session_id"
           " WHERE g.dynasty=? AND s.side='defense'"
           " AND s.executed_status='identified' AND s.executed_verification='verified'"
           " AND NULLIF(TRIM(s.executed_adjustment_id),'') IS NOT NULL"
           " AND NULLIF(TRIM(s.executed_formation),'') IS NOT NULL"
           " AND NULLIF(TRIM(s.executed_play),'') IS NOT NULL")
    for raw, result, down, dist, form, play in conn.execute(sql, (franchise,)):
        success = offense_won(result, down, dist)
        if success is None:
            continue
        action_id = str(raw).strip()
        if not action_id or action_id == "NO_ADJUSTMENT":
            continue
        key = _key(context_key(down, dist), str(form), str(play), action_id)
        stats = counts.setdefault(key, [0, 0])
        stats[0] += int(bool(success))
        stats[1] += 1
        n += 1
    return {"schema": SCHEMA, "counts": counts, "n": n,
            "evidence": "verified observational executed actions; not causal"}


def _action_registry() -> dict[str, dict[str, Any]]:
    return {str(a["id"]): a for a in catalog()["actions"]
            if a.get("side") == "defense" and a.get("kind") == "individual_presnap"
            and not a.get("blocked_reason") and a.get("inputs")}


def read_confirmed_options(path: str | Path | None) -> list[dict[str, Any]]:
    """Owner-created physical controller confirmations; never inferred from EA entry map."""
    if not path:
        return []
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if payload.get("schema") != CONFIRM_SCHEMA:
        raise ValueError("invalid Xbox physical confirmation file")
    registry = _action_registry()
    options = []
    unique = set()
    for raw in payload.get("confirmed_actions") or []:
        if not isinstance(raw, dict):
            continue
        aid = str(raw.get("action_id") or "")
        instance = str(raw.get("instance_id") or "")
        item = registry.get(aid)
        # User-entered sequence is a physical verification assertion; the
        # catalog checks that the adjustment entry point exists at all.
        if not (item and raw.get("physically_verified") is True
                and raw.get("console") == "Xbox"
                and str(raw.get("verified_buttons") or "").strip()
                and str(raw.get("verified_on") or "").strip()
                and instance and instance not in unique):
            continue
        if aid == "def.coverage_audible":  # selects another stock play, not a defensive adjustment
            continue
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            continue
        valid = True
        for name, spec in (item.get("params") or {}).items():
            val = params.get(name)
            opts = spec.get("options")
            if not isinstance(val, str) or not val.strip():
                valid = False
            if isinstance(opts, list) and val not in opts:
                valid = False
        req = item.get("requires") or {}
        if req.get("target") and not str(raw.get("target") or "").strip():
            valid = False
        if not valid:
            continue
        unique.add(instance)
        options.append({
            "id": instance, "catalog_action_id": aid,
            "label": str(raw.get("label") or item["name"]),
            "buttons": str(raw["verified_buttons"]).strip(),
            "params": params, "target": raw.get("target"),
            "exclusive_groups": list(item.get("exclusive_groups") or []),
            "touches": list(item.get("touches") or []),
            "estimated_seconds": int(item.get("est_seconds") or 0),
            "verified_on": raw["verified_on"],
            "evidence": "physically verified controller sequence",
        })
    return options


def _expanded(values: Sequence[str], action: Mapping[str, Any]) -> set[str]:
    result = set()
    for val in values:
        target = str(action.get("target") or "UNKNOWN")
        group = str(action.get("params", {}).get("option") or "")
        result.add(str(val).replace("{target}", target).replace("{option_group}", group))
    return result


def compatible(actions: Sequence[Mapping[str, Any]]) -> bool:
    groups: set[str] = set()
    touches: set[str] = set()
    for action in actions:
        g = _expanded(action.get("exclusive_groups") or [], action)
        t = _expanded(action.get("touches") or [], action)
        if groups & g or touches & t:
            return False
        groups.update(g)
        touches.update(t)
    return True


def choose_adjustments(
    model: Mapping[str, Any], *, context: str, formation: str, play: str,
    stock_offense_success: float, confirmed: Sequence[Mapping[str, Any]],
    max_seconds: int = 9, max_actions: int = 3,
) -> dict[str, Any]:
    """Conservatively compare exact trained combinations against stock.

    The model's executed_adjustments counts must have been trained before the
    CURRENT game; live calls never use the current snap's adjustments.
    """
    from cfb_coach.madden.model.defense_v2_pilot import _key
    none = {"id": "NO_ADJUSTMENT", "actions": [], "effect_support": 0,
            "predicted_offense_success": float(stock_offense_success),
            "estimated_seconds": 0, "learned": False}
    counts = (model.get("executed_adjustments") or {}).get("counts") or {}
    if not counts or not confirmed:
        return {**none, "options_considered": 1}
    options = [none]
    safe = list(confirmed)
    for n in range(1, min(len(safe), max_actions) + 1):
        for group in combinations(safe, n):
            if not compatible(group):
                continue
            seconds = sum(int(x.get("estimated_seconds") or 0) for x in group)
            if seconds > max_seconds:
                continue
            # The action ID is explicit even for combinations; only a labeled
            # execution of that exact package qualifies as training support.
            ids = sorted(str(x["id"]) for x in group)
            combo = "+".join(ids)
            n_bad, n_seen = counts.get(_key(context, formation, play, combo), [0, 0])
            n_seen = int(n_seen)
            if n_seen < MIN_ACTION_SNAPS:
                continue
            mu = (float(n_bad) + SHRINK_N * stock_offense_success) / (n_seen + SHRINK_N)
            sd = math.sqrt(max(0.0, mu * (1.0 - mu) / (n_seen + SHRINK_N + 1.0)))
            benefit_lower = stock_offense_success - (mu + 1.64 * sd)
            if benefit_lower <= MIN_CONSERVATIVE_GAIN:
                continue
            options.append({
                "id": combo, "actions": list(group),
                "effect_support": n_seen, "predicted_offense_success": mu,
                "conservative_benefit": benefit_lower,
                "estimated_seconds": seconds, "learned": True,
            })
    winner = min(options, key=lambda x: (x["predicted_offense_success"],
                                         -x["effect_support"], x["id"]))
    return {**winner, "options_considered": len(options),
            "effect_is_causal": False, "no_adjustment_compared": True}
