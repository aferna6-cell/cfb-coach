"""Baseline META gameplan + learned opponent overlays.

Architecture:
  1. Every game starts from generic META baseline (meta_baseline.json).
  2. As snaps/games are logged, adjust gameplan + macro weights.
  3. Opponent-specific overlays stack ON TOP of baseline; thin film ≈ baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from cfb_coach.db import CoachDB
from cfb_coach.format_call import format_defense, format_offense


def _baseline_path() -> Path:
    try:
        ref = resources.files("cfb_coach").joinpath("data/meta_baseline.json")
        with resources.as_file(ref) as p:
            return Path(p)
    except Exception:
        return Path(__file__).resolve().parent / "data" / "meta_baseline.json"


@lru_cache(maxsize=1)
def load_baseline() -> dict[str, Any]:
    path = _baseline_path()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Overlay / effective merge
# ---------------------------------------------------------------------------

GLOBAL_BUCKET = "global"
CPU_BUCKET = "cpu"


@dataclass
class Overlay:
    """Learned deltas stacked on baseline for one opponent (or global/cpu)."""

    opponent_id: str
    gameplan: dict[str, float] = field(default_factory=dict)  # side:key → delta
    macros: dict[str, float] = field(default_factory=dict)  # macro → delta
    snap_count: int = 0

    @property
    def depth(self) -> str:
        if self.snap_count <= 0 and not self.gameplan and not self.macros:
            return "none (thin — stay near baseline)"
        if self.snap_count < 5 and abs(sum(self.gameplan.values())) < 0.5:
            return "thin"
        if self.snap_count < 15:
            return "light"
        return "developed"


def _load_overlay(db: CoachDB | None, opponent_id: str) -> Overlay:
    ov = Overlay(opponent_id=opponent_id)
    if not db:
        return ov
    ov.snap_count = db.count_snaps(opponent_id)
    for row in db.get_gameplan_weights(opponent_id):
        key = f"{row['side']}:{row['key']}"
        ov.gameplan[key] = float(row["weight"])
    for row in db.get_macro_weights(opponent_id):
        ov.macros[row["macro"]] = float(row["weight"])
    return ov


def _merge_overlays(*overlays: Overlay) -> Overlay:
    """Merge global → cpu/opponent; later overlays win on keys they set."""
    merged = Overlay(opponent_id=overlays[-1].opponent_id if overlays else "")
    for ov in overlays:
        merged.snap_count = max(merged.snap_count, ov.snap_count)
        for k, v in ov.gameplan.items():
            merged.gameplan[k] = merged.gameplan.get(k, 0.0) + v
        for k, v in ov.macros.items():
            merged.macros[k] = merged.macros.get(k, 0.0) + v
    return merged


def get_opponent_overlay(db: CoachDB | None, opponent_id: str) -> Overlay:
    """Stack global + (cpu if cpu else opponent) learned weights."""
    global_ov = _load_overlay(db, GLOBAL_BUCKET)
    if opponent_id == CPU_BUCKET:
        return _merge_overlays(global_ov, _load_overlay(db, CPU_BUCKET))
    # Always include global; opponent-specific on top. CPU bucket as soft prior
    # for human opponents only when they have almost no film.
    opp_ov = _load_overlay(db, opponent_id)
    if opp_ov.snap_count < 3:
        cpu_ov = _load_overlay(db, CPU_BUCKET)
        # dilute cpu prior so thin humans stay near baseline
        diluted = Overlay(opponent_id=opponent_id, snap_count=opp_ov.snap_count)
        for k, v in cpu_ov.gameplan.items():
            diluted.gameplan[k] = v * 0.25
        for k, v in cpu_ov.macros.items():
            diluted.macros[k] = v * 0.25
        return _merge_overlays(global_ov, diluted, opp_ov)
    return _merge_overlays(global_ov, opp_ov)


@dataclass
class EffectiveGameplan:
    version: str
    baseline: dict[str, Any]
    overlay: Overlay
    offense_emphasis: dict[str, float]
    defense_emphasis: dict[str, float]
    macro_weights: dict[str, float]
    macros_keep: list[str]
    macros_bench: list[str]
    opening_menu: list[dict[str, Any]]
    defense_home: dict[str, Any]
    meta_notes: list[str]


def effective_gameplan(
    opponent_id: str, db: CoachDB | None = None
) -> EffectiveGameplan:
    """baseline + learned overlays from SQLite (global + opponent/cpu)."""
    bl = load_baseline()
    ov = get_opponent_overlay(db, opponent_id)

    o_emph = dict(bl["offense_gameplan"].get("emphasis_keys") or {})
    d_emph = dict(bl["defense_gameplan"].get("emphasis_keys") or {})
    for key, delta in ov.gameplan.items():
        side, _, name = key.partition(":")
        if "::" in name:
            # v1.12 structured keys (play::, vs_look::, zone::) feed the live
            # ranker / prep zone plan, not the family emphasis table
            continue
        if side == "offense" and name in o_emph:
            o_emph[name] = max(0.05, o_emph[name] + delta)
        elif side == "offense":
            o_emph[name] = max(0.05, 1.0 + delta)
        elif side == "defense" and name in d_emph:
            d_emph[name] = max(0.05, d_emph[name] + delta)
        elif side == "defense":
            d_emph[name] = max(0.05, 1.0 + delta)

    macro_w = dict(bl["macros_baseline"].get("weights") or {})
    for m, delta in ov.macros.items():
        macro_w[m] = max(0.0, macro_w.get(m, 0.0) + delta)

    keep = list(bl["macros_baseline"]["keep"])
    bench = list(bl["macros_baseline"]["bench"])
    # Never auto-unbench from thin overlay; only elevate keep-set weights
    return EffectiveGameplan(
        version=bl.get("version", "?"),
        baseline=bl,
        overlay=ov,
        offense_emphasis=o_emph,
        defense_emphasis=d_emph,
        macro_weights=macro_w,
        macros_keep=keep,
        macros_bench=bench,
        opening_menu=list(bl["offense_gameplan"]["opening_menu"]),
        defense_home={
            "package": bl["defense_gameplan"]["home_package"],
            "calls": list(bl["defense_gameplan"]["home_calls"]),
            "rotation": dict(bl["defense_gameplan"]["home_rotation"]),
        },
        meta_notes=list(bl.get("meta_notes") or []),
    )


# ---------------------------------------------------------------------------
# Prep formatting: BASELINE → OVERLAY → EFFECTIVE
# ---------------------------------------------------------------------------

def _fmt_menu_o(items: list[dict[str, Any]]) -> list[str]:
    lines = []
    for it in items:
        lines.append(
            f"    [{it['label']}] "
            + format_offense(
                it["formation"],
                it["play"],
                it.get("adj", "No adj"),
                it.get("reads", "Primary → Checkdown"),
            )
        )
    return lines


def format_baseline_section(bl: dict[str, Any] | None = None) -> str:
    bl = bl or load_baseline()
    og = bl["offense_gameplan"]
    dg = bl["defense_gameplan"]
    mb = bl["macros_baseline"]
    oc = bl.get("offense_concepts") or {}
    dc = bl.get("defense_concepts") or {}
    lines = [
        f"## BASELINE META (CFB27 / {bl.get('version', '?')})",
        f"  Game: {bl.get('game', 'CFB 27')}  |  Patch: {bl.get('patch', 'n/a')}",
        "  Generic starting point for every user/CPU game.",
        "",
        "  Offense home: " + ", ".join(og["home_formations"]),
        f"  Identity: {oc.get('core_identity', 'Bunch meta')}",
        f"  Two-high rule: {og['two_high_rule']}",
        "  Pressure answers: "
        + ", ".join(p["play"] for p in og["pressure_answers"]),
        f"  Red zone: {og['red_zone']['priority']} — "
        + ", ".join(
            f"{x['formation']}/{x['play']}" for x in og["red_zone"]["pass"][:3]
        ),
        f"  Anti-repeat: {og['anti_repeat']}",
        "",
        "  Coverage beater matrix (sample):",
    ]
    matrix = (oc.get("coverage_beater_matrix") or {})
    for cov in ("Cover 6", "Cover 9", "Cover 3 Sky", "Cover 1 / Pressure / Cover 0"):
        row = matrix.get(cov)
        if row:
            lines.append(
                f"    {cov}: prefer {', '.join(row['prefer'][:3])} — {row.get('note', '')}"
            )
    pairs = oc.get("constraint_pairs") or []
    if pairs:
        lines.append("  Constraint pairs:")
        for p in pairs[:4]:
            lines.append(f"    {p['a']} ↔ {p['b']} ({p['why']})")
    lines += [
        "",
        "  Opening menu:",
        *_fmt_menu_o(og["opening_menu"]),
        "",
        f"  Defense home: {dg['home_package']} — "
        + " / ".join(dg["home_calls"]),
        f"  Identity: {dc.get('core_identity', 'Nickel Over home')}",
        f"  C6/C9 awareness: {dc.get('cover_6_9_awareness', '')}",
        f"  Situational: Even 6-1 short/GL; "
        f"selective pressure ({', '.join(dg['selective_pressure']['packages'])})",
        f"  Default macro: {dg['default_macro']}",
        "",
        f"  Macros KEEP: {', '.join(mb['keep'])}",
        f"  Macros BENCH: {', '.join(mb['bench'])} — "
        + "; ".join(f"{m}: {mb['reasons'][m]}" for m in mb["bench"]),
    ]
    create = mb.get("create_candidates") or []
    if create:
        lines.append(f"  Macros CREATE candidates: {', '.join(create)}")
    # Recipe PURPOSE lines for keep set
    recipes = mb.get("recipes") or {}
    if recipes:
        lines.append("  Macro recipes (PURPOSE / when-to-arm):")
        for m in mb["keep"]:
            rec = recipes.get(m) or {}
            lines.append(
                f"    {m}: {rec.get('purpose', '')} | WHEN {rec.get('when_to_arm', mb.get('when_to_arm', {}).get(m, ''))}"
            )
    lines += [
        f"  Arm doctrine: {mb['arm_doctrine']}",
        "",
        "  Soft META notes:",
    ]
    for note in bl.get("meta_notes") or []:
        lines.append(f"    - {note}")
    return "\n".join(lines)


def format_overlay_section(ov: Overlay) -> str:
    lines = [
        f"## OPPONENT OVERLAY — {ov.opponent_id} (depth: {ov.depth})",
        f"  Logged snaps (this id): {ov.snap_count}",
    ]
    if not ov.gameplan and not ov.macros:
        lines.append(
            "  (no learned deltas — effective ≈ baseline; thin film stays META)"
        )
        return "\n".join(lines)

    if ov.gameplan:
        lines.append("  Gameplan weight deltas:")
        for k, v in sorted(ov.gameplan.items(), key=lambda kv: -abs(kv[1])):
            if abs(v) < 0.01:
                continue
            sign = "+" if v >= 0 else ""
            lines.append(f"    {k}: {sign}{v:.2f}")
    if ov.macros:
        lines.append("  Macro weight deltas:")
        for k, v in sorted(ov.macros.items(), key=lambda kv: -abs(kv[1])):
            if abs(v) < 0.01:
                continue
            sign = "+" if v >= 0 else ""
            lines.append(f"    {k}: {sign}{v:.2f}")
    return "\n".join(lines)


def format_effective_macros(eg: EffectiveGameplan) -> str:
    bl = eg.baseline
    mb = bl["macros_baseline"]
    lines = [
        "## EFFECTIVE MACROS (baseline ⊕ overlay)",
        f"  KEEP: {', '.join(eg.macros_keep)}",
        f"  BENCH: {', '.join(eg.macros_bench)}",
        "  Weights (higher = more ready to arm after repeated tendency):",
    ]
    for m in eg.macros_keep:
        w = eg.macro_weights.get(m, 1.0)
        when = mb["when_to_arm"].get(m, "")
        lines.append(f"    {m}: {w:.2f}  — {when}")
    for m in eg.macros_bench:
        w = eg.macro_weights.get(m, 0.0)
        lines.append(f"    {m} [BENCHED]: {w:.2f}  — {mb['reasons'].get(m, '')}")
    lines.append(f"  Doctrine: {mb['arm_doctrine']}")
    return "\n".join(lines)


def format_effective_emphasis(eg: EffectiveGameplan) -> str:
    lines = [
        "## EFFECTIVE EMPHASIS (mix rates)",
        "  Offense:",
    ]
    for k, v in sorted(eg.offense_emphasis.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {k}: {v:.2f}")
    lines.append("  Defense:")
    for k, v in sorted(eg.defense_emphasis.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {k}: {v:.2f}")
    lines.append(
        "  Free reign preserved — weights bias mix; never remove legal calls."
    )
    return "\n".join(lines)


def format_gameplan_block(
    opponent_id: str, db: CoachDB | None = None
) -> str:
    """BASELINE then OPPONENT OVERLAY then EFFECTIVE macros/emphasis."""
    eg = effective_gameplan(opponent_id, db)
    parts = [
        format_baseline_section(eg.baseline),
        "",
        format_overlay_section(eg.overlay),
        "",
        format_effective_macros(eg),
        "",
        format_effective_emphasis(eg),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Learning — postgame / after snaps
# ---------------------------------------------------------------------------

_SUCCESS_WORDS = (
    "td",
    "+",
    "good",
    "convert",
    "stop",
    "sack",
    "int",
    "hold",
    "stuff",
    "punt",
    "turnover",
)
_FAIL_WORDS = (
    "int",
    "sack",
    "stuff",
    "loss",
    "fail",
    "incomp",
    "turnover",
    "pick",
    "-",
)


def _result_success(result: str | None, side: str) -> bool | None:
    """True/False/None (unknown). Offense: yards/td good; Defense: stop/sack good."""
    from cfb_coach.outcome import outcome_success

    return outcome_success(result, side)


def _play_emphasis_key(formation: str | None, play: str | None) -> str | None:
    if not play:
        return None
    p = play.lower()
    f = (formation or "").lower()
    if any(x in p for x in ("inside zone", "hb base", "counter", "duo", "dive", "outside zone", "stretch")):
        return "run_first"
    if "mesh" in p or "drive" in p:
        return "mesh_family"
    if "cluster" in f or "z spot" in p:
        return "cluster_changeup"
    if "flood" in p or "cross post" in p or "vertical" in p:
        return "flood_family"
    if "whip" in p:
        return "whip_hot"
    if "empty" in f or "quads" in f:
        return "empty_stress"
    if "deuce" in f:
        return "deuce_bully"
    return None


def _def_emphasis_key(formation: str | None, play: str | None) -> str | None:
    if not play and not formation:
        return None
    f = (formation or "").lower()
    p = (play or "").lower()
    if "even 6-1" in f or "6-1" in f:
        return "even_61_short"
    if "dime" in f:
        return "dime_long"
    if "tampa" in p:
        return "tampa_2"
    if "quarters" in p or "cover 4" in p:
        return "c4_quarters"
    if "cover 3" in p or "sky" in p:
        return "c3_sky"
    if "nickel over" in f:
        return "nickel_over_home"
    if "cub" in f or "mug" in f or "blitz" in p:
        return "selective_heat"
    return "nickel_over_home"


def _macro_snapshot(db: CoachDB) -> dict[tuple[str, str], float]:
    try:
        return {
            (r[0], r[1]): float(r[2])
            for r in db.conn.execute("SELECT opponent_id, macro, weight FROM macro_weights").fetchall()
        }
    except Exception:  # noqa: BLE001
        return {}


def learn_from_snaps(
    db: CoachDB,
    opponent_id: str,
    *,
    since_id: int | None = None,
    also_global: bool = True,
    bump: float = 0.15,
    fail_bump: float = 0.10,
) -> dict[str, Any]:
    """Retrain (v1.12): rebuild ALL weights from every logged snap + game result.

    The old incremental bumps (``bump`` / ``fail_bump``) grew without bound; the
    v2 rules in ``cfb_coach.learning`` are situation-aware, leverage-weighted,
    capped and W/L-aware, and are recomputed from scratch each time (idempotent).
    ``since_id`` only scopes the grades / ``snaps_considered`` narrative.
    Returns a changelog of deltas (``<bucket>/<side>:<key>`` and ``<bucket>/macro:<M>``).
    """
    from cfb_coach.learning import rebuild_all
    from cfb_coach.retrain import grade_play_vs_look

    _ = (also_global, bump, fail_bump)  # kept for API compatibility
    snaps = db.get_recent_snaps(opponent_id, since_id=since_id, limit=400)
    m_before = _macro_snapshot(db)
    res = rebuild_all(db)
    m_after = _macro_snapshot(db)
    buckets = {opponent_id, GLOBAL_BUCKET}
    changes: dict[str, float] = {}
    for k in set(res["before"]) | set(res["after"]):
        if k[0] not in buckets:
            continue
        d = res["after"].get(k, 0.0) - res["before"].get(k, 0.0)
        if abs(d) >= 0.001:
            changes[f"{k[0]}/{k[1]}:{k[2]}"] = d
    for k in set(m_before) | set(m_after):
        if k[0] not in buckets:
            continue
        d = m_after.get(k, 0.0) - m_before.get(k, 0.0)
        if abs(d) >= 0.001:
            changes[f"{k[0]}/macro:{k[1]}"] = d
    vs_changes = {k: v for k, v in changes.items() if ":vs_look::" in k}
    return {
        "opponent_id": opponent_id,
        "snaps_considered": len(snaps),
        "changes": {k: round(v, 3) for k, v in sorted(changes.items())},
        "grades": grade_play_vs_look(list(snaps)),
        "vs_look_changes": {k: round(v, 3) for k, v in vs_changes.items()},
        "rebuild": res,
    }


def evaluate_ohio_state_promotions(
    changes: dict[str, float],
    *,
    opponent_id: str,
    threshold: float | None = None,
) -> list[dict]:
    """If experimental strategies work in ohio_state, propose Alabama promotions."""
    from cfb_coach.dynasty import PROMOTE_WEIGHT_THRESHOLD, OHIO_STATE
    from cfb_coach.macros import META_GROUNDED, UNVALIDATED, validation_status, get_macro

    thr = PROMOTE_WEIGHT_THRESHOLD if threshold is None else threshold
    promotions: list[dict] = []
    for k, v in (changes or {}).items():
        if v < thr:
            continue
        if "/macro:" in k:
            macro = k.split("/macro:", 1)[-1].upper()
            st = validation_status(macro)
            meta = get_macro(macro) or {}
            # Promote experimental / meta_grounded successes into Alabama
            if st in (META_GROUNDED, UNVALIDATED) or not meta.get("active", True):
                promotions.append(
                    {
                        "kind": "macro_loadout",
                        "target": macro,
                        "macro": macro,
                        "delta": round(v, 3),
                        "source_opponent": opponent_id,
                        "source_dynasty": OHIO_STATE,
                        "status": "pending",
                        "note": (
                            f"ohio_state lab success ({v:+.2f}) — consider swapping "
                            f"into Alabama Active-8 for {macro}"
                        ),
                    }
                )
        elif "/offense:" in k:
            key = k.split("/offense:", 1)[-1]
            promotions.append(
                {
                    "kind": "gameplan_overlay",
                    "target": key,
                    "key": key,
                    "side": "offense",
                    "delta": round(v, 3),
                    "source_opponent": opponent_id,
                    "source_dynasty": OHIO_STATE,
                    "status": "pending",
                    "note": (
                        f"ohio_state lab success ({v:+.2f}) — elevate Alabama "
                        f"offense overlay '{key}'"
                    ),
                }
            )
        elif "/defense:" in k:
            # Defense overlays only promote for user (non-CPU) lab games
            if (opponent_id or "").lower() == "cpu":
                continue
            key = k.split("/defense:", 1)[-1]
            promotions.append(
                {
                    "kind": "gameplan_overlay",
                    "target": key,
                    "key": key,
                    "side": "defense",
                    "delta": round(v, 3),
                    "source_opponent": opponent_id,
                    "source_dynasty": OHIO_STATE,
                    "status": "pending",
                    "note": (
                        f"ohio_state lab success ({v:+.2f}) — elevate Alabama "
                        f"defense overlay '{key}'"
                    ),
                }
            )
    return promotions


def postgame_summary(db: CoachDB, opponent_id: str, dynasty: str | None = None) -> str:
    """Game over -> retrain (v1.12 rules) and explain it in plain language.

    Rebuilds every weight from all logged snaps + W/L (``cfb_coach.learning``),
    then reports THIS game's red-zone / goal-line findings, turnovers, the W/L
    adjustment and the top risers / fallers. ohio_state lab successes still
    produce Alabama promotion notes.
    """
    from cfb_coach.dynasty import (
        DEFAULT_DYNASTY,
        OHIO_STATE,
        dynasty_config,
        doctrine_line,
        format_promotions,
        normalize_dynasty,
        record_promotions,
    )
    from cfb_coach.learning import RULES_VERSION, format_postgame_v2

    dcfg = dynasty_config(normalize_dynasty(dynasty or DEFAULT_DYNASTY))
    meta_key = f"last_postgame_snap_id:{opponent_id}"
    row = db.conn.execute(
        "SELECT value FROM meta WHERE key = ?", (meta_key,)
    ).fetchone()
    since_id = int(row["value"]) if row else None

    before = get_opponent_overlay(db, opponent_id)
    result = learn_from_snaps(db, opponent_id, since_id=since_id)
    after = get_opponent_overlay(db, opponent_id)

    # Which game is this? latest session with new snaps for this opponent
    session_id = None
    try:
        q = "SELECT session_id FROM snaps WHERE opponent_id = ? AND session_id IS NOT NULL"
        params: tuple = (opponent_id,)
        if since_id is not None:
            q += " AND id > ?"
            params = (opponent_id, since_id)
        r = db.conn.execute(q + " ORDER BY id DESC LIMIT 1", params).fetchone()
        session_id = r[0] if r else None
    except Exception:  # noqa: BLE001
        session_id = None

    # Advance cursor to latest snap for this opponent
    latest = db.conn.execute(
        "SELECT MAX(id) AS m FROM snaps WHERE opponent_id = ?",
        (opponent_id,),
    ).fetchone()
    if latest and latest["m"] is not None:
        db.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (meta_key, str(int(latest["m"]))),
        )
        db.conn.commit()

    n_all = db.count_snaps(opponent_id)
    lines = [
        f"# POSTGAME — vs {opponent_id}",
        f"New snaps learned: {result['snaps_considered']} (all {n_all} logged snaps re-scored under rules {RULES_VERSION})",
        f"Overlay depth: {before.depth} → {after.depth}",
        "",
    ]
    changes = result["changes"]
    if not result["snaps_considered"] and not changes:
        lines.append(
            "  (no scored results to learn from — log snaps with "
            "result +N / stop / td / etc.)"
        )
    lines.extend(format_postgame_v2(db, opponent_id, result["rebuild"], session_id=session_id))
    grades = result.get("grades") or []
    if grades:
        from cfb_coach.retrain import format_grades_summary

        lines.append("")
        lines.append("## Play vs coverage/look (this game)")
        lines.extend(format_grades_summary(grades))
    lines.append("")
    lines.append("## Effective macros (after)")
    eg = effective_gameplan(opponent_id, db)
    for m in eg.macros_keep:
        lines.append(f"  {m}: {eg.macro_weights.get(m, 1.0):.2f}")
    lines.append("")
    lines.append(
        "Free reign unchanged — learning only reweights mix / macro readiness."
    )
    lines.append("")
    lines.append(f"Dynasty: {dcfg.get('label')} ({dcfg.get('mode')})")
    lines.append(doctrine_line())

    # ohio_state lab → Alabama promotion notes (family / macro level only)
    # Only keys that ROSE this game AND are net-positive after the rebuild
    after_w = (result.get("rebuild") or {}).get("after") or {}

    def _after_positive(k: str) -> bool:
        if "/macro:" in k:
            return True
        bucket, _, rest = k.partition("/")
        side, _, key = rest.partition(":")
        return after_w.get((bucket, side, key), 0.0) > 0

    promo_changes = {
        k: float(v)
        for k, v in changes.items()
        if "::" not in k.split(":", 1)[-1] and _after_positive(k)
    }
    if dcfg.get("id") == OHIO_STATE and promo_changes:
        promos = evaluate_ohio_state_promotions(
            promo_changes,
            opponent_id=opponent_id,
        )
        if promos:
            record_promotions(db, promos)
            lines.append("")
            lines.append("## Promotion note (ohio_state → Alabama)")
            for pr in promos:
                lines.append(f"  - {pr.get('note')}")
            lines.append(
                "  Pending promotions stored — "
                "`cfb_coach promote` to review / `--accept-all` to accept."
            )
        else:
            lines.append("")
            lines.append(
                "## Promotion note: no strong ohio_state successes this batch "
                "(threshold not met)."
            )
    elif dcfg.get("id") != OHIO_STATE:
        pending_txt = format_promotions(db=db)
        if "No Alabama promotions" not in pending_txt:
            lines.append("")
            lines.append(pending_txt)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mid-game pivot stub + anti-repeat helpers
# ---------------------------------------------------------------------------

def recent_play_keys(db: CoachDB, opponent_id: str, *, side: str, n: int = 4) -> list[str]:
    snaps = db.get_recent_snaps(opponent_id, side=side, limit=n)
    return [f"{s['formation']}::{s['play']}" for s in snaps]


def anti_repeat_penalty(
    db: CoachDB | None, opponent_id: str, formation: str, play: str, *, side: str = "offense"
) -> float:
    """Return penalty weight if this play was called too often recently."""
    if not db:
        return 0.0
    keys = recent_play_keys(db, opponent_id, side=side, n=4)
    target = f"{formation}::{play}"
    repeats = sum(1 for k in keys if k == target)
    if repeats >= 2:
        return 0.5 * repeats
    return 0.0


@dataclass
class PivotSuggestion:
    kind: str  # "offense_pivot" | "defense_macro"
    message: str
    family: str | None = None
    macro: str | None = None


def _side_fail_streak(
    db: CoachDB, opponent_id: str, side: str, n: int = 3
) -> int:
    snaps = db.get_recent_snaps(opponent_id, side=side, limit=n)
    if len(snaps) < n:
        return 0
    return sum(
        1 for s in snaps if _result_success(s["result"], side) is False
    )


def active_pivot(
    db: CoachDB | None, opponent_id: str, side: str
) -> PivotSuggestion | None:
    """
    Stronger mid-game pivot: last 3 snaps fail on a side → PIVOT tag +
    switch family/macro plan. No single-snap whiplash (needs 3 fails).

    Milestone 2: also surfaces live tendency explosive/scramble pivots.
    """
    live_tip = live_tendency_pivot_tip(side)
    if not db:
        return live_tip
    from cfb_coach.tendency import is_user_opponent

    user = is_user_opponent(opponent_id)
    fails3 = _side_fail_streak(db, opponent_id, side, n=3)
    fails2 = _side_fail_streak(db, opponent_id, side, n=2)
    # Hard PIVOT at 3 for everyone; user soft PIVOT at 2 (this-game adaptation)
    if fails3 >= 3:
        hard = True
    elif user and fails2 >= 2:
        hard = False
    else:
        return live_tip
    if side == "offense":
        if hard:
            msg = (
                "PIVOT: last 3 O snaps failed — switch family now "
                "(run ↔ Mesh Spot easy ↔ Gun Cluster changeup). "
                "No hero-shot whiplash."
            )
        else:
            msg = (
                "PIVOT (user game): last 2 O snaps failed — start switching family "
                "(run ↔ Mesh Spot easy ↔ Gun Cluster). Still no single-snap whiplash."
            )
        return PivotSuggestion(
            kind="offense_pivot",
            family="constraint",
            message=msg,
        )
    if hard:
        msg = (
            "PIVOT: last 3 D snaps failed — reset to Nickel Over base "
            "(C3 Sky / C4 Quarters / Tampa), clear chase macros, "
            "re-arm only on repeated tendency. No single-snap whiplash."
        )
    else:
        msg = (
            "PIVOT (user game): last 2 D snaps failed — lean back to Nickel Over base, "
            "drop chase macros until 2+ tells THIS game. No single-snap whiplash."
        )
    return PivotSuggestion(
        kind="defense_pivot",
        family="shell_reset",
        macro="none",
        message=msg,
    )


def midgame_pivot_stub(
    db: CoachDB | None,
    opponent_id: str,
) -> list[PivotSuggestion]:
    """
    If last 3 snaps fail on a side → PIVOT (switch family/macro plan).
    If same D concept hits twice → allow macro consideration (no single-snap whiplash).
    """
    out: list[PivotSuggestion] = []
    if not db:
        return out

    for side in ("offense", "defense"):
        tip = active_pivot(db, opponent_id, side)
        if tip:
            out.append(tip)

    d_snaps = db.get_recent_snaps(opponent_id, side="defense", limit=6)
    concepts: dict[str, int] = {}
    for s in d_snaps:
        c = s["concept_seen"]
        if c:
            concepts[c.lower()] = concepts.get(c.lower(), 0) + 1
    for concept, n in concepts.items():
        if n >= 2:
            macro = None
            cl = concept
            if "vert" in cl or "seam" in cl or "four" in cl:
                macro = "VERT"
            elif "cross" in cl or "wheel" in cl:
                macro = "CROSS"
            elif "mesh" in cl or "bunch" in cl:
                macro = "BUNCH"
            elif "rpo" in cl or "bubble" in cl:
                macro = "RPO"
            elif "scram" in cl:
                macro = "SCRAM"
            if macro:
                out.append(
                    PivotSuggestion(
                        kind="defense_macro",
                        family=concept,
                        macro=macro,
                        message=(
                            f"MACRO CONSIDERATION: {concept} hit {n}× recently — "
                            f"{macro} now legal to arm (still situational; "
                            "no single-snap whiplash)."
                        ),
                    )
                )
    return out


def format_pivot_hints(
    db: CoachDB | None, opponent_id: str
) -> str | None:
    tips = midgame_pivot_stub(db, opponent_id)
    if not tips:
        return None
    lines = ["## Mid-game pivot / macro consideration"]
    for t in tips:
        lines.append(f"  - {t.message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Milestone 2 — live tendency pivots (explosive / counter validation)
# ---------------------------------------------------------------------------

_LIVE_PIVOT_NOTE: dict[str, Any] = {}


def note_live_tendency_pivot(tendency: Any) -> None:
    """Record strongest live tendency for active_pivot extension (process-local)."""
    try:
        _LIVE_PIVOT_NOTE["signal"] = getattr(tendency, "signal", str(tendency))
        _LIVE_PIVOT_NOTE["sample_size"] = int(getattr(tendency, "sample_size", 0))
        _LIVE_PIVOT_NOTE["confidence"] = str(getattr(tendency, "confidence", ""))
        _LIVE_PIVOT_NOTE["hit_rate"] = float(getattr(tendency, "hit_rate", 0.0))
    except Exception:
        pass


def clear_live_tendency_pivot() -> None:
    _LIVE_PIVOT_NOTE.clear()


def live_tendency_pivot_tip(side: str) -> PivotSuggestion | None:
    """If live engine shows strong explosive / confirmed tendency → soft pivot tip."""
    sig = str(_LIVE_PIVOT_NOTE.get("signal") or "")
    n = int(_LIVE_PIVOT_NOTE.get("sample_size") or 0)
    conf = str(_LIVE_PIVOT_NOTE.get("confidence") or "")
    if n < 3 or conf not in ("actionable", "strong"):
        return None
    if sig == "EXPLOSIVE" or "EXPLOSIVE" in sig:
        return PivotSuggestion(
            kind="offense_pivot" if side == "offense" else "defense_pivot",
            family="explosive",
            message=(
                f"PIVOT (live): EXPLOSIVE trend n={n} — "
                "constrain hero shots / tighten leverage. No single-snap whiplash."
            ),
        )
    if "SCRAMBLE" in sig:
        return PivotSuggestion(
            kind="defense_macro",
            family="scramble",
            macro="SCRAM",
            message=f"PIVOT (live): QB escape n={n} — SCRAM/contain now legal.",
        )
    return None



def postgame_live_report(engine: Any | None) -> str | None:
    """Concise live-tendency summary for `postgame --report`."""
    if engine is None:
        return None
    try:
        return engine.summary_report()
    except Exception:
        return None
