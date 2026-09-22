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
    if not result:
        return None
    r = result.lower().strip()
    # crude: leading +N = offense success; stop/sack = defense success
    if side == "offense":
        if r.startswith("+") or "td" in r or "convert" in r or "good" in r:
            return True
        if any(w in r for w in ("int", "sack", "loss", "fail", "incomp", "pick", "stuff")):
            return False
        if r.startswith("-"):
            return False
        return None
    # defense
    if any(w in r for w in ("stop", "sack", "stuff", "hold", "punt", "int", "turnover")):
        return True
    if r.startswith("+") or "td" in r or "convert" in r:
        return False
    return None


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


def learn_from_snaps(
    db: CoachDB,
    opponent_id: str,
    *,
    since_id: int | None = None,
    also_global: bool = True,
    bump: float = 0.15,
    fail_bump: float = 0.10,
) -> dict[str, Any]:
    """
    Adjust gameplan_weights / macro_weights from recent snaps.
    Never removes free-reign ability — only reweights mix.
    Returns a changelog of deltas applied.
    """
    snaps = db.get_recent_snaps(opponent_id, since_id=since_id, limit=200)
    changes: dict[str, float] = {}
    buckets = [opponent_id]
    if also_global:
        buckets.append(GLOBAL_BUCKET)
    if opponent_id != CPU_BUCKET:
        # human games also lightly feed global only (already listed)
        pass
    else:
        if CPU_BUCKET not in buckets:
            buckets.append(CPU_BUCKET)

    # Anti-repeat: count play frequency in this batch
    play_counts: dict[str, int] = {}
    for s in snaps:
        key = f"{s['formation'] or ''}::{s['play'] or ''}"
        play_counts[key] = play_counts.get(key, 0) + 1

    for s in snaps:
        side = s["side"]
        success = _result_success(s["result"], side)
        if success is None:
            continue
        delta = bump if success else -fail_bump

        if side == "offense":
            ek = _play_emphasis_key(s["formation"], s["play"])
            if ek:
                for b in buckets:
                    scale = 1.0 if b == opponent_id else 0.35
                    db.bump_gameplan_weight(b, "offense", ek, delta * scale)
                    ck = f"{b}/offense:{ek}"
                    changes[ck] = changes.get(ck, 0.0) + delta * scale
            # Penalize over-repeat even on success (mix discipline)
            pk = f"{s['formation'] or ''}::{s['play'] or ''}"
            if play_counts.get(pk, 0) >= 3 and s["play"]:
                for b in buckets:
                    scale = 1.0 if b == opponent_id else 0.35
                    db.bump_gameplan_weight(
                        b, "offense", "anti_repeat_penalty", -0.05 * scale
                    )
                    # nudge changeup up
                    db.bump_gameplan_weight(
                        b, "offense", "cluster_changeup", 0.05 * scale
                    )
                    changes[f"{b}/offense:cluster_changeup"] = (
                        changes.get(f"{b}/offense:cluster_changeup", 0.0)
                        + 0.05 * scale
                    )
        else:
            ek = _def_emphasis_key(s["formation"], s["play"])
            if ek:
                for b in buckets:
                    scale = 1.0 if b == opponent_id else 0.35
                    db.bump_gameplan_weight(b, "defense", ek, delta * scale)
                    ck = f"{b}/defense:{ek}"
                    changes[ck] = changes.get(ck, 0.0) + delta * scale
            macro = s["macro"]
            if macro and macro.lower() not in ("none", "", "null"):
                m = macro.upper()
                for b in buckets:
                    scale = 1.0 if b == opponent_id else 0.35
                    db.bump_macro_weight(b, m, delta * scale)
                    ck = f"{b}/macro:{m}"
                    changes[ck] = changes.get(ck, 0.0) + delta * scale

    return {
        "opponent_id": opponent_id,
        "snaps_considered": len(snaps),
        "changes": {k: round(v, 3) for k, v in sorted(changes.items())},
    }


def postgame_summary(db: CoachDB, opponent_id: str) -> str:
    """Run learning on snaps since last postgame; print what changed."""
    meta_key = f"last_postgame_snap_id:{opponent_id}"
    row = db.conn.execute(
        "SELECT value FROM meta WHERE key = ?", (meta_key,)
    ).fetchone()
    since_id = int(row["value"]) if row else None

    before = get_opponent_overlay(db, opponent_id)
    result = learn_from_snaps(db, opponent_id, since_id=since_id)
    after = get_opponent_overlay(db, opponent_id)

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

    lines = [
        f"# POSTGAME — vs {opponent_id}",
        f"Snaps considered: {result['snaps_considered']}",
        f"Overlay depth: {before.depth} → {after.depth}",
        "",
        "## Weight changes",
    ]
    changes = result["changes"]
    if not changes:
        lines.append(
            "  (no scored results to learn from — log snaps with "
            "result +N / stop / td / etc.)"
        )
    else:
        for k, v in changes.items():
            sign = "+" if v >= 0 else ""
            lines.append(f"  {k}: {sign}{v:.3f}")
    lines.append("")
    lines.append("## Effective macros (after)")
    eg = effective_gameplan(opponent_id, db)
    for m in eg.macros_keep:
        lines.append(f"  {m}: {eg.macro_weights.get(m, 1.0):.2f}")
    lines.append("")
    lines.append(
        "Free reign unchanged — learning only reweights mix / macro readiness."
    )
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
    """
    if not db:
        return None
    fails = _side_fail_streak(db, opponent_id, side, n=3)
    if fails < 3:
        return None
    if side == "offense":
        return PivotSuggestion(
            kind="offense_pivot",
            family="constraint",
            message=(
                "PIVOT: last 3 O snaps failed — switch family now "
                "(run ↔ Mesh Spot easy ↔ Gun Cluster changeup). "
                "No hero-shot whiplash."
            ),
        )
    return PivotSuggestion(
        kind="defense_pivot",
        family="shell_reset",
        macro="none",
        message=(
            "PIVOT: last 3 D snaps failed — reset to Nickel Over base "
            "(C3 Sky / C4 Quarters / Tampa), clear chase macros, "
            "re-arm only on repeated tendency. No single-snap whiplash."
        ),
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
