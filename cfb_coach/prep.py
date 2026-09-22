"""Pregame prep sheets — threat, D-vs-us, macros, emphasis, opening menus."""

from __future__ import annotations

from typing import Any

from cfb_coach.db import CoachDB
from cfb_coach.format_call import format_defense, format_offense
from cfb_coach.playcaller import _ACTIVE_MACROS, _BENCHED_MACROS
from cfb_coach.seed import load_seed


def _opp(seed: dict, oid: str) -> dict[str, Any]:
    opp = dict(seed["opponents"].get(oid) or {})
    opp["_id"] = oid
    return opp


def _bullet(items: list[str] | None, indent: str = "  - ") -> str:
    if not items:
        return f"{indent}(thin film — lean meta priors)"
    return "\n".join(f"{indent}{x}" for x in items)


def threat_sheet(opp: dict[str, Any]) -> str:
    oid = opp.get("_id", "")
    lines = [
        f"## Threat sheet — {opp.get('display_name', oid)} ({opp.get('team_now', '?')})",
        f"Confidence: {opp.get('confidence', '?')}  |  Skill: {opp.get('skill') or 'n/a'}",
    ]
    if opp.get("notes"):
        lines.append(f"Notes: {opp['notes']}")
    if opp.get("prev"):
        lines.append(f"Prev team: {opp['prev']}")

    offense = opp.get("offense") or {}
    if not offense:
        lines.append("Offense film: thin — use meta priors + live log.")
    else:
        lines.append("Their offense:")
        for key in (
            "early_down",
            "explosives",
            "intermediate",
            "core",
            "money",
            "stress",
            "runs",
            "rpo",
            "goal_line",
        ):
            vals = offense.get(key)
            if vals:
                label = key.replace("_", " ")
                lines.append(f"  {label}: {', '.join(vals) if isinstance(vals, list) else vals}")
        for key in ("escape", "qb", "tempo", "result"):
            if offense.get(key):
                lines.append(f"  {key}: {offense[key]}")

    snaps = opp.get("sample_snaps") or []
    if snaps:
        lines.append("Sample snaps (seed):")
        for s in snaps[:6]:
            lines.append(
                f"  - {s.get('sit')}: cov={s.get('cov')} | our={s.get('our')} | {s.get('note')}"
            )
    return "\n".join(lines)


def defense_vs_us(opp: dict[str, Any]) -> str:
    dvs = opp.get("defense_vs_us") or {}
    lines = ["## Defense vs us"]
    if not dvs:
        lines.append("  Thin film — expect generic two-high / zone mix; run more.")
        return "\n".join(lines)
    lines.append(f"  Archetype: {dvs.get('archetype', '?')}")
    if dvs.get("coverages"):
        lines.append(f"  Coverages: {', '.join(dvs['coverages'])}")
    if dvs.get("pressures"):
        lines.append(f"  Pressures: {', '.join(dvs['pressures'])}")
    if dvs.get("example_nd"):
        lines.append(f"  Money-down examples: {', '.join(dvs['example_nd'])}")
    if dvs.get("man_blitz"):
        lines.append(f"  Man/blitz: {dvs['man_blitz']}")
    if dvs.get("vs_bunch_cluster"):
        lines.append(f"  vs Bunch/Cluster: {dvs['vs_bunch_cluster']}")
    if dvs.get("lesson"):
        lines.append(f"  Lesson: {dvs['lesson']}")
    return "\n".join(lines)


def macro_plan(opp: dict[str, Any], seed: dict) -> str:
    baseline = seed["league"]["online_baseline"]
    keep = list(baseline.get("defensive_macros_active") or _ACTIVE_MACROS)
    bench = list(baseline.get("defensive_macros_benched") or _BENCHED_MACROS)
    add: list[str] = []
    suggest: list[str] = []

    oid = opp.get("_id", "")
    offense = opp.get("offense") or {}
    blob = " ".join(
        str(v)
        for k, v in offense.items()
        if isinstance(v, (str, list))
        for v in (v if isinstance(v, list) else [v])
    ).lower()
    dvs = opp.get("defense_vs_us") or {}
    arch = (dvs.get("archetype") or "").lower()

    # Opponent-specific emphasis (still within active set unless SUGGEST)
    emphasis: list[str] = []
    if oid == "gavin" or "split_field" in arch or "two_high" in arch:
        emphasis.append("KEEP base zones; VERT only after repeated 4-verts; avoid chase macros")
        emphasis.append("SCRAM ready (QB scramble when coverage carries)")
        emphasis.append("RUN-IN only high confidence — he will PA/scramble the sellout")
    if oid == "quen" or "pressure" in arch:
        emphasis.append("CROSS / RPO / SCRAM elevated — Cross Wheels + bubbles + scramble")
        emphasis.append("HEAT selective on 3rd-short, not every snap")
    if oid == "tiano":
        emphasis.append("BUNCH / CROSS situational; GL don't sell out run (Z Smash lesson)")
    if oid == "cpu" or "two_high_money" in arch:
        emphasis.append("Mostly no macro — Quarters/Tampa on money; force underneath")
    if "vertical" in blob or "four vert" in blob:
        emphasis.append("VERT candidate once tendency confirms")
    if "cross" in blob or "wheel" in blob:
        emphasis.append("CROSS candidate once tendency confirms")
    if "rpo" in blob or "bubble" in blob:
        emphasis.append("RPO candidate vs bubble tendency")

    # Suggest reactivation only as SUGGEST, never invent PS buttons
    if oid == "quen":
        suggest.append("SCREEN (benched) — only if online stability re-tested and bubbles dominate")
    if oid == "gavin":
        suggest.append("FLOOD (benched) — only if flood/levels become primary; keep benched for now")

    lines = [
        "## Macro plan (Xbox — names only, no invented button sequences)",
        f"  KEEP active: {', '.join(keep)}",
        f"  BENCH: {', '.join(bench)}",
    ]
    if add:
        lines.append(f"  ADD (still designed): {', '.join(add)}")
    if emphasis:
        lines.append("  Emphasis this opponent:")
        lines.extend(f"    - {e}" for e in emphasis)
    if suggest:
        lines.append("  SUGGEST (optional / not auto-armed):")
        lines.extend(f"    - {s}" for s in suggest)
    lines.append("  Doctrine: macros situational, not every snap. Default = no macro.")
    return "\n".join(lines)


def playbook_emphasis(opp: dict[str, Any], seed: dict) -> str:
    oid = opp.get("_id", "")
    dvs = opp.get("defense_vs_us") or {}
    arch = (dvs.get("archetype") or "").lower()
    lines = ["## Playbook emphasis (free reign — any call legal)"]
    lines.append("  Home: Gun Bunch X Nasty (preserve) + Gun Cluster counterpunch")

    if "split_field" in arch or "two_high" in arch or oid == "gavin":
        lines.append("  vs them: RUN FIRST on early downs (IZ / HB Base / Counter)")
        lines.append("  Pass: Mesh Spot easy; Z Spot Shake / Cluster when Bunch overplayed")
        lines.append("  AVOID: spam Mesh Post into Cover 6 (INT film)")
        lines.append("  Cover 2 Invert early: Mesh Spot / underneath OR Inside Zone — not Deep Flood")
    elif "pressure" in arch or oid == "quen":
        lines.append("  vs them: protection + hot — Mesh Spot, Return Whip Trail, HB Base")
        lines.append("  Load answers to Sam/Will/WS blitz; don't hero into pressure")
    elif oid == "tiano" or "c2_c3" in arch:
        lines.append("  vs them: more C2 near scoring — Z Spot Shake / run; tag whip by leverage")
        lines.append("  RZ: possession > hero (film: whip into C2 Invert HF can INT)")
    elif oid == "cpu":
        lines.append("  vs CPU: take free underneath; don't force Mesh Post/Whip into sticks coverage")
        lines.append("  Two-high money downs → run / Mesh Spot / Drive HB Under")
    else:
        lines.append("  Thin film: establish run, Mesh Spot easy completions, mix Cluster")
        lines.append("  Meta soft prior: two-high → run; pressure → quick game; C3 → flood/crossers")

    doctrine = seed.get("doctrine_priors") or {}
    if doctrine.get("offense"):
        lines.append(f"  Doctrine: {doctrine['offense']}")
    return "\n".join(lines)


def opening_menus(opp: dict[str, Any]) -> str:
    oid = opp.get("_id", "")
    lines = ["## Opening menus"]

    # Offense
    lines.append("  Offense:")
    o_menu: list[tuple[str, str, str, str, str]] = [
        ("1st&10 open", "Gun Bunch X Nasty", "Inside Zone", "No adj", "Front → Cutback"),
        ("Easy pass", "Gun Bunch X Nasty", "Mesh Spot", "No adj", "Spot → Drag"),
    ]
    if oid == "gavin":
        o_menu.append(
            ("vs two-high", "Gun Bunch X Nasty", "HB Base", "No adj", "Front → Bounce")
        )
        o_menu.append(
            ("C2 Invert early", "Gun Bunch X Nasty", "Mesh Spot", "No adj", "Spot → Drag")
        )
        o_menu.append(
            ("Cluster changeup", "Gun Cluster", "Z Spot Shake", "No adj", "Shake → Spot")
        )
    elif oid == "quen":
        o_menu.append(
            ("vs pressure", "Gun Bunch X Nasty", "Return Whip Trail", "Hot ready", "Whip → Trail")
        )
        o_menu.append(
            ("3rd&short", "Gun Bunch X Nasty", "HB Base", "Protection first", "Front → Bounce")
        )
    elif oid == "cpu":
        o_menu.append(
            ("3rd-long two-high", "Gun Bunch X Nasty", "Mesh Spot", "No adj", "Spot → Drag")
        )
        o_menu.append(
            ("Underneath", "Gun Bunch X Nasty", "Drive HB Under", "No adj", "Drive → HB Under")
        )
    else:
        o_menu.append(
            ("Single-high", "Gun Bunch X Nasty", "Deep Flood", "No adj", "Flat → Corner")
        )
        o_menu.append(
            ("Cluster", "Gun Cluster", "Outside Zone", "No adj", "Reach → Cut")
        )

    for label, form, play, adj, reads in o_menu:
        lines.append(f"    [{label}] {format_offense(form, play, adj, reads)}")

    # Defense
    lines.append("  Defense (default prior: Nickel Over):")
    d_menu: list[tuple[str, str, str, str, str]] = [
        ("1st/2nd default", "Nickel Over", "Cover 3 Sky", "none", "User hook/HB"),
        ("Protect explosive", "Nickel Over", "Cover 4 Quarters", "none", "User seam"),
        ("Mesh changeup", "Nickel Over", "Tampa 2", "none", "User middle"),
    ]
    if oid == "gavin":
        d_menu.append(
            ("vs 4-verts tell", "Nickel Over", "Cover 4 Quarters", "VERT", "User #3 seam")
        )
        d_menu.append(
            ("early IZ fit", "Nickel Over", "Cover 3 Sky", "none", "User HB")
        )
    elif oid == "quen":
        d_menu.append(
            ("crosser tendency", "Nickel Over", "Tampa 2", "CROSS", "User inside cross")
        )
        d_menu.append(
            ("bubble tendency", "Nickel Over", "Cover 3 Sky", "RPO", "User flat/bubble")
        )
    elif oid == "cpu":
        d_menu.append(
            ("money down", "Nickel Over", "Cover 4 Quarters", "none", "User seam")
        )
    else:
        d_menu.append(
            ("short yardage", "4-3 Even 6-1", "Cover 3 Buzz", "none", "User cutback")
        )

    for label, form, play, macro, user in d_menu:
        lines.append(f"    [{label}] {format_defense(form, play, macro, user)}")

    return "\n".join(lines)


def build_prep(opponent_id: str, db: CoachDB | None = None) -> str:
    from cfb_coach.gameplan import format_gameplan_block, format_pivot_hints

    seed = load_seed()
    opp = _opp(seed, opponent_id)
    if db:
        profile = db.get_opponent(opponent_id)
        if profile:
            opp = dict(profile)
            opp["_id"] = opponent_id

    league = seed["league"]
    header = [
        f"# PREP — vs {opp.get('display_name', opponent_id)} ({opp.get('team_now', '?')})",
        f"League: {league['name']} | Platform: {league['platform']}",
        f"Timing: O≈{league['timing']['offense_seconds']}s  "
        f"D≈{league['timing']['defense_seconds']}s  "
        f"(D call target {league['timing']['defense_call_target_seconds']}s)  pause={league['timing']['pause']}",
        f"Books: {league['online_baseline']['custom_offense']} / "
        f"{league['online_baseline']['custom_defense']}",
        "Architecture: BASELINE → LEARN → ADJUST (opponent overlay stacks on META)",
        "",
    ]
    sections = [
        format_gameplan_block(opponent_id, db),
        "",
        threat_sheet(opp),
        "",
        defense_vs_us(opp),
        "",
        macro_plan(opp, seed),
        "",
        playbook_emphasis(opp, seed),
        "",
        opening_menus(opp),
    ]
    pivot = format_pivot_hints(db, opponent_id)
    if pivot:
        sections.extend(["", pivot])
    return "\n".join(header + sections)
