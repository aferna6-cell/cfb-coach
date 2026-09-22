"""INSTALL SHEET — prepper has full permission to CREATE/EDIT/BENCH playbooks + macros.

Emits concrete Xbox custom-book steps (offense/defense books + custom adjustments).
Persists per-opponent install diffs in SQLite so the next prep remembers prior choices.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from cfb_coach.db import CoachDB
from cfb_coach.gameplan import effective_gameplan, load_baseline


# ---------------------------------------------------------------------------
# Step builders
# ---------------------------------------------------------------------------

def _step(action: str, target: str, detail: str, *, why: str = "") -> dict[str, str]:
    return {
        "action": action.upper(),  # CREATE | ADD | EDIT | BENCH
        "target": target,
        "detail": detail,
        "why": why,
    }


def _baseline_macro_steps(bl: dict[str, Any]) -> list[dict[str, str]]:
    mb = bl["macros_baseline"]
    recipes = mb.get("recipes") or {}
    steps: list[dict[str, str]] = []

    steps.append(
        _step(
            "CREATE",
            "Custom Defense book",
            f"Ensure book exists: BAMA META D (or rename to CFB27 META D). Home package Nickel Over.",
            why="CFB27 defense home",
        )
    )
    steps.append(
        _step(
            "CREATE",
            "Custom Offense book",
            f"Ensure book exists: BAMA META O (or rename to CFB27 META O). Home: Gun Bunch X Nasty + Gun Cluster.",
            why="CFB27 Bunch meta",
        )
    )

    for m in mb["keep"]:
        rec = recipes.get(m) or {}
        purpose = rec.get("purpose") or mb["reasons"].get(m, "")
        when = rec.get("when_to_arm") or mb.get("when_to_arm", {}).get(m, "")
        shell = rec.get("shell_pair", "")
        user = rec.get("user_job", "")
        steps.append(
            _step(
                "ADD",
                f"D macro {m}",
                f"Arm recipe: shell={shell or 'Nickel Over'}; user={user or 'one job'}; "
                f"PURPOSE={purpose}. WHEN={when}",
                why="baseline KEEP",
            )
        )

    for m in mb["bench"]:
        reason = mb["reasons"].get(m, "benched until stable")
        steps.append(
            _step(
                "BENCH",
                f"D macro {m}",
                f"Leave OFF / do not bind until online-stable. Reason: {reason}",
                why="baseline BENCH",
            )
        )

    # CREATE candidates (invented names with clear purpose)
    for m in mb.get("create_candidates") or []:
        rec = recipes.get(m) or {}
        steps.append(
            _step(
                "CREATE",
                f"Custom adj / macro {m}",
                f"PURPOSE={rec.get('purpose', m)}. WHEN={rec.get('when_to_arm', 'situational')}. "
                f"Shell={rec.get('shell_pair', 'n/a')}; User={rec.get('user_job', 'n/a')}",
                why="CFB27 create-candidate",
            )
        )

    # Offense playbook edits
    og = bl["offense_gameplan"]
    steps.append(
        _step(
            "EDIT",
            "O book — Gun Bunch X Nasty",
            "Pin core: Inside Zone, HB Base, Counter Y, Mesh Spot, Mesh Traffic, Drive HB Under, "
            "Deep Flood, Return Whip Trail, Mtn Cross Post, Mtn RPO Zone Alert, RZ PA X Whip, Z Spot GoalLine",
            why="Bunch is CFB27 passing meta",
        )
    )
    steps.append(
        _step(
            "EDIT",
            "O book — Gun Cluster",
            "Pin changeup: Outside Zone, HB Counter, Z Spot Shake, Mesh Post, Verticals",
            why="Cluster when Bunch overplayed",
        )
    )
    steps.append(
        _step(
            "ADD",
            "O book — changeups",
            "Keep available: Pistol U Off Trips (HB Stretch, RPO Alert TE Flat); "
            "Singleback Deuce Close (Mtn Duo, HB Dive); Gun Empty Quads (Out Double Under)",
            why="constraint formations",
        )
    )
    steps.append(
        _step(
            "EDIT",
            "O custom adj — Protection / Hot",
            "Save 'Protection first' and 'Hot ready' as favorite adjs for Mesh Spot / Whip / HB Base",
            why="pressure answers",
        )
    )
    steps.append(
        _step(
            "EDIT",
            "D book — Nickel Over",
            "Home rotation: Cover 3 Sky / Cover 4 Quarters / Tampa 2. "
            "Add Tampa 2 + hard flats lean vs mesh/cross.",
            why="Nickel Over zones home",
        )
    )
    steps.append(
        _step(
            "ADD",
            "D book — situational packages",
            "4-3 Even 6-1 (short/GL); Nickel 3-3 Cub + Double Mug (selective HEAT); "
            "Dime Normal (distance >= 12)",
            why="situational D",
        )
    )
    steps.append(
        _step(
            "EDIT",
            "D custom adj — Contain",
            "Patch 1.012 contain custom adj fix — save CONTAIN-SCRAM recipe (contain + spy lean) "
            "for dual-threat / scramble games",
            why="patch 1.012",
        )
    )

    # Opening menu pins
    for item in og.get("opening_menu") or []:
        steps.append(
            _step(
                "ADD",
                f"O favorite — {item['label']}",
                f"{item['formation']} — {item['play']} | {item.get('adj', 'No adj')}",
                why="opening menu pin",
            )
        )

    return steps


def _opponent_overlay_steps(opponent_id: str, opp: dict[str, Any]) -> list[dict[str, str]]:
    """Opponent-specific CREATE/EDIT/BENCH lean — prepper free reign."""
    oid = (opponent_id or "").lower()
    dvs = opp.get("defense_vs_us") or {}
    arch = (dvs.get("archetype") or "").lower()
    offense = opp.get("offense") or {}
    steps: list[dict[str, str]] = []

    if oid == "gavin" or ("split_field" in arch) or ("cover_6" in arch and oid != "cpu"):
        steps.append(
            _step(
                "EDIT",
                "O book lean — run-vs-C6",
                "Elevate Inside Zone / HB Base / Counter Y on 1st&10 favorites. "
                "Demote Mesh Post & Deep Flood from early-down favorites vs Cover 6/9.",
                why="Gavin/split-field — users live in C6/C9; RUN FIRST",
            )
        )
        steps.append(
            _step(
                "ADD",
                "O favorite — vs C6/C9",
                "Gun Bunch X Nasty — HB Base | No adj  AND  Gun Bunch X Nasty — Inside Zone | No adj",
                why="run-vs-C6 install lean",
            )
        )
        steps.append(
            _step(
                "ADD",
                "O favorite — easy vs two-high",
                "Gun Bunch X Nasty — Mesh Spot | No adj; Gun Cluster — Z Spot Shake | No adj",
                why="easy completion + Cluster changeup",
            )
        )
        steps.append(
            _step(
                "BENCH",
                "O early-down Mesh Post",
                "Do not pin Mesh Post as opening vs this opponent (INT film into Cover 6)",
                why="avoid C6 INT",
            )
        )
        steps.append(
            _step(
                "EDIT",
                "D macros vs Gavin",
                "KEEP base zones; VERT only after repeated 4-verts; SCRAM ready; "
                "RUN-IN only high confidence — he will PA/scramble the sellout",
                why="split-field patience on D",
            )
        )
        steps.append(
            _step(
                "CREATE",
                "Custom adj CONTAIN-SCRAM",
                "Bind contain custom adj (patch 1.012 fix) — PURPOSE=stop scramble when coverage carries",
                why="Gavin escape threat",
            )
        )

    if oid == "quen" or "pressure" in arch:
        steps.append(
            _step(
                "EDIT",
                "O book lean — protection macros",
                "Pin Mesh Spot (Hot ready), Return Whip Trail (Hot ready), HB Base (Protection first) "
                "as top-3 pressure answers. Elevate PROT cue.",
                why="Quen/pressure — protection + hot",
            )
        )
        steps.append(
            _step(
                "CREATE",
                "Custom adj PROT",
                "PURPOSE=protection/hot reminder vs Sam/Will/WS blitz. "
                "WHEN=any obvious pressure look. Pair with Mesh Spot / Whip / HB Base.",
                why="protection macro lean",
            )
        )
        steps.append(
            _step(
                "ADD",
                "O favorite — vs pressure",
                "Gun Bunch X Nasty — Return Whip Trail | Hot ready; "
                "Gun Bunch X Nasty — Mesh Spot | Hot ready; "
                "Gun Bunch X Nasty — HB Base | Protection first",
                why="quen protection install",
            )
        )
        steps.append(
            _step(
                "EDIT",
                "D macros vs Quen",
                "Elevate CROSS / RPO / SCRAM readiness; HEAT selective on 3rd-short (Cub/Mug), not every snap",
                why="Cross Wheels + bubbles + scramble",
            )
        )
        steps.append(
            _step(
                "CREATE",
                "Custom adj GLASS",
                "PURPOSE=soft zone glass vs their mesh if they flip to Bunch. WHEN=repeated mesh wins.",
                why="optional mesh counter",
            )
        )

    if oid == "tiano" or "c2_c3" in arch:
        steps.append(
            _step(
                "EDIT",
                "O RZ lean",
                "Pin Mesh Spot / Z Spot Shake / Deuce Duo near scoring — tag whip by leverage, not auto",
                why="C2-heavy near scoring",
            )
        )
        steps.append(
            _step(
                "EDIT",
                "D GL",
                "Even 6-1 OK; do NOT sell out run (Z Smash lesson). BUNCH/CROSS situational only",
                why="Temple/Tiano GL lesson",
            )
        )

    if oid == "cpu" or "two_high_money" in arch:
        steps.append(
            _step(
                "EDIT",
                "O vs CPU money",
                "Pin Mesh Spot / Drive HB Under / Inside Zone on 3rd-long two-high — no forced Mesh Post/Whip",
                why="CPU sticks coverage",
            )
        )
        steps.append(
            _step(
                "BENCH",
                "D macros default",
                "Mostly no macro vs CPU — Quarters/Tampa on money; force underneath",
                why="CPU patience",
            )
        )

    # Thin / generic
    if not steps:
        steps.append(
            _step(
                "EDIT",
                "O thin-film lean",
                "Establish run + Mesh Spot easy + Cluster mix. Two-high → run; pressure → quick; C3 → flood",
                why="thin film ≈ baseline",
            )
        )

    # Film-driven adds from their offense lists
    blob = " ".join(
        str(v)
        for k, v in offense.items()
        if isinstance(v, (str, list))
        for v in (v if isinstance(v, list) else [v])
    ).lower()
    if "vertical" in blob or "four vert" in blob:
        steps.append(
            _step(
                "ADD",
                "D macro VERT readiness",
                "After 2+ live vertical tells — Quarters + VERT | User #3 seam",
                why="film verticals",
            )
        )
    if "cross" in blob or "wheel" in blob:
        steps.append(
            _step(
                "ADD",
                "D macro CROSS readiness",
                "After 2+ live crosser/wheel — Tampa/Quarters + CROSS | User inside cross",
                why="film crossers",
            )
        )
    if "rpo" in blob or "bubble" in blob:
        steps.append(
            _step(
                "ADD",
                "D macro RPO readiness",
                "After repeated bubble — C3 Sky + RPO | User flat",
                why="film RPO/bubble",
            )
        )
    if "scram" in blob or "escape" in (offense.get("escape") or "").lower():
        steps.append(
            _step(
                "CREATE",
                "Custom adj CONTAIN-SCRAM",
                "PURPOSE=contain integrity vs escape artist. Patch 1.012 contain adj fix.",
                why="escape film",
            )
        )

    return steps


def _diff_against_saved(
    current: list[dict[str, str]], saved: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Return steps in current that were not in the last saved install (by action+target+detail)."""
    saved_keys = {
        (s.get("action", ""), s.get("target", ""), s.get("detail", "")) for s in saved
    }
    return [
        s
        for s in current
        if (s.get("action", ""), s.get("target", ""), s.get("detail", "")) not in saved_keys
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_install_steps(
    opponent_id: str,
    opp: dict[str, Any] | None = None,
    *,
    db: CoachDB | None = None,
) -> list[dict[str, str]]:
    bl = load_baseline()
    steps = _baseline_macro_steps(bl)
    steps.extend(_opponent_overlay_steps(opponent_id, opp or {}))
    return steps


def save_install_diff(
    db: CoachDB,
    opponent_id: str,
    steps: list[dict[str, str]],
) -> dict[str, Any]:
    """Persist full sheet + compute/store new diffs vs previous sheet."""
    prev = db.get_install_sheet(opponent_id)
    prev_steps = (prev or {}).get("steps") or []
    new_diffs = _diff_against_saved(steps, prev_steps)
    payload = {
        "version": load_baseline().get("version", "?"),
        "ts": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "new_since_last": new_diffs,
    }
    db.save_install_sheet(opponent_id, payload)
    # Also store each new diff row for history
    for d in new_diffs:
        db.log_install_diff(
            opponent_id=opponent_id,
            action=d["action"],
            target=d["target"],
            detail=d["detail"],
            why=d.get("why", ""),
        )
    return payload


def format_install_sheet(
    opponent_id: str,
    opp: dict[str, Any] | None = None,
    *,
    db: CoachDB | None = None,
    persist: bool = True,
) -> str:
    """Format INSTALL SHEET section; optionally persist diffs to SQLite."""
    bl = load_baseline()
    eg = effective_gameplan(opponent_id, db)
    steps = build_install_steps(opponent_id, opp, db=db)

    remembered: list[dict[str, str]] = []
    new_diffs: list[dict[str, str]] = []
    if db is not None:
        prev = db.get_install_sheet(opponent_id)
        if prev:
            remembered = prev.get("steps") or []
            new_diffs = _diff_against_saved(steps, remembered)
        else:
            new_diffs = list(steps)  # first prep = all new
        if persist:
            save_install_diff(db, opponent_id, steps)
            # re-read new_since from save
            saved = db.get_install_sheet(opponent_id) or {}
            new_diffs = saved.get("new_since_last") or new_diffs

    lines = [
        f"## INSTALL SHEET — Xbox custom books (CFB27 / {bl.get('version', '?')})",
        "  Prepper: FULL permission to CREATE / ADD / EDIT / BENCH playbooks + macros + custom adjs.",
        f"  Books: BAMA META O / BAMA META D  |  Patch: {bl.get('patch', 'n/a')}",
        f"  Opponent: {opponent_id}  |  Overlay depth: {eg.overlay.depth}",
        "",
        "  ### Steps (execute on Xbox)",
    ]
    for i, s in enumerate(steps, 1):
        why = f"  [{s['why']}]" if s.get("why") else ""
        lines.append(f"  {i}. [{s['action']}] {s['target']}")
        lines.append(f"       {s['detail']}{why}")

    if remembered and new_diffs:
        lines.append("")
        lines.append("  ### NEW since last prep (remembered diffs)")
        for d in new_diffs[:12]:
            lines.append(f"  - [{d['action']}] {d['target']}: {d['detail'][:100]}")
    elif remembered and not new_diffs:
        lines.append("")
        lines.append("  ### Remembered: install unchanged since last prep (SQLite)")
    elif db is not None:
        lines.append("")
        lines.append("  ### First install for this opponent — full sheet saved to SQLite")

    lines.append("")
    lines.append(
        "  Doctrine: names only (no invented button sequences). "
        "Default macro = none until repeated tendency."
    )
    return "\n".join(lines)
