"""Madden 27 Franchise prep plan — inventory (scheme pack) vs opponent deltas.

Same UX contract as CFB prep: the scheme pack is assumed already stocked
(custom playbook), so prep shows only ADD / EDIT / BENCH-style deltas that are
at least meta_grounded, plus the Active-8 loadout (post-swap) and call tips.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from cfb_coach.install_sheet import filter_new_deltas, get_applied_deltas
from cfb_coach.madden.data import (
    META_VERSION,
    archetype_lean,
    get_macro,
    load_macro_catalog,
    load_meta_baseline,
    load_seed,
)
from cfb_coach.madden.franchise import (
    LAB,
    doctrine_line,
    get_session_profile,
    normalize_profile,
    profile_config,
)
from cfb_coach.madden.macros import USER_ACTIVE_CAP, loadout_cards, macro_side, swap_plan
from cfb_coach.opponents import is_cpu_opponent


def load_profile(opponent_id: str, db: Any = None) -> dict[str, Any]:
    opp = dict((load_seed().get("opponents") or {}).get(opponent_id) or {})
    if db is not None:
        prof = db.get_opponent(opponent_id)
        if prof:
            opp = dict(prof)
    opp["_id"] = opponent_id
    return opp


def build_inventory(seed: dict[str, Any] | None = None) -> dict[str, Any]:
    seed = seed or load_seed()
    pb = seed["playbooks"]
    base = seed["league"]["online_baseline"]
    offense = {
        name: {
            "role": meta.get("role", ""),
            "book": meta.get("book", ""),
            "plays": list(meta.get("core") or []),
            "audibles": list(meta.get("audibles") or []),
        }
        for name, meta in pb["offense_formations"].items()
    }
    defense = {
        name: {"role": meta.get("role", ""), "book": meta.get("book", ""), "calls": list(meta.get("calls") or [])}
        for name, meta in pb["defense_packages"].items()
    }
    return {
        "offense_book": base["custom_offense"],
        "defense_book": base["custom_defense"],
        "offense": offense,
        "defense": defense,
        "macros_active": list(base["defensive_macros_active"]),
        "offensive_macros": list(base["offensive_macros_active"]),
        "macros_benched": list(base["macros_benched"]),
    }


def _delta(action: str, target: str, detail: str, *, kind: str = "playbook", field: str = "",
           before: str = "", after: str = "", why: str = "", side: str = "") -> dict[str, Any]:
    return {
        "action": action.upper(),
        "kind": kind,
        "target": target,
        "field": field,
        "before": before,
        "after": after,
        "detail": detail,
        "why": why,
        "side": side,
        "validated_status": "meta_grounded",
    }


def propose_deltas(
    opponent_id: str,
    opp: dict[str, Any],
    *,
    profile: str,
) -> list[dict[str, Any]]:
    """Macro (Custom Adjustment) deltas only — formations go through the playbook of record."""
    arch = (opp.get("archetype") or "unknown").lower()
    traits = opp.get("traits") or {}
    cpu = is_cpu_opponent(opponent_id)
    out: list[dict[str, Any]] = []

    if arch == "split_field_zone" and traits.get("escape") and not cpu:
        m = get_macro("SPY") or {}
        out.append(_delta(
            "EDIT", "SPY", "Pre-load QB spy plan for this persona's escapes",
            kind="macro", field="When to arm", before=m.get("when_to_arm", ""),
            after="After 2+ scrambles this game — 3rd down first",
            why=f"Persona trait: {traits['escape']}", side="defense",
        ))
    elif arch == "pressure_heavy":
        m = get_macro("O-PROT") or {}
        out.append(_delta(
            "EDIT", "O-PROT", "Arm protection earlier vs this persona",
            kind="macro", field="When to arm", before=m.get("when_to_arm", ""),
            after="From snap 1 on passing downs (pressure persona)",
            why="Pressure-heavy archetype — O-PROT is the persona answer, not a one-tell chase.",
            side="offense",
        ))
    elif arch == "c2_c3_mixer" and not cpu:
        m = get_macro("STACK") or {}
        out.append(_delta(
            "EDIT", "STACK", "Compressed/GL persona — STACK is the likely first macro",
            kind="macro", field="When to arm", before=m.get("when_to_arm", ""),
            after="After 2+ stack/bunch wins, incl. RZ/GL",
            why="c2_c3_mixer persona lives in compressed sets near scoring.",
            side="defense",
        ))

    # Lab = freer: bring a benched meta_grounded macro in (with swap at cap)
    if normalize_profile(profile) == LAB:
        exp = "O-RPO" if cpu else "HEAT"
        m = get_macro(exp) or {}
        out.append(_delta(
            "ADD", exp, f"Lab experiment: {m.get('purpose', '')}",
            kind="macro", field="Active", after=f"{exp} Active",
            why="Franchise lab — test benched meta_grounded macro; promotes to primary if it holds.",
            side=macro_side(exp),
        ))
    return out


_AUDIBLE_TIPS = {
    "split_field_zone": ("Gun Doubles Clamp Stack", "Mtn Shuffle Verts Smash", "Same Side Zone",
                         "two-high persona — zone run on the audible, don't force verts into safeties"),
    "two_high_money_downs": ("Gun Trips X Nasty", "Switch HB Wheel", "Hi Lo Cross",
                             "CPU two-high money downs — take free underneath"),
}


def audible_tips(opp: dict[str, Any], book: dict[str, list[str]]) -> list[str]:
    """Optional audible-slot tweaks (tips only, never install steps); only for in-book plays."""
    tip = _AUDIBLE_TIPS.get((opp.get("archetype") or "").lower())
    if not tip:
        return []
    form, old, new, why = tip
    if form in book and old in book[form] and new in book[form]:
        return [f"Optional audible: {form} — swap {old} → {new} ({why})"]
    return []


def resolve_loadout(
    active: list[str],
    deltas: list[dict[str, Any]],
    archetype: str | None,
    *,
    offense_only: bool,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Apply macro ADDs (with swap at cap) → (active after, swap plans, replacing lines)."""
    after = list(active)
    swaps: list[dict[str, Any]] = []
    replacing: list[str] = []
    for d in deltas:
        if d.get("kind") != "macro" or d.get("action") != "ADD" or d["target"] in after:
            continue
        counted = [m for m in after if not offense_only or macro_side(m) == "offense"]
        if len(counted) >= USER_ACTIVE_CAP:
            sp = swap_plan(d["target"], after, archetype)
            d["swap_plan"] = sp
            swaps.append(sp)
            after.remove(sp["bench"])
            replacing.append(f"replacing {sp['bench']} with {d['target']}")
        after.append(d["target"])
    return after, swaps, replacing


def call_tips(opp: dict[str, Any], *, offense_only: bool, bl: dict[str, Any]) -> list[str]:
    lean = archetype_lean(opp.get("archetype"))
    tips: list[str] = []
    if offense_only:
        tips.append("CPU game = offense-only coaching (no D calls / no D macros)")
    tips.append(f"O: {lean.get('offense', '')}")
    if not offense_only:
        tips.append(f"D: {lean.get('defense', '')}")
    tips.append(f"Run game: {bl['offense']['run_preference']}")
    if not offense_only:
        tips.append(f"D pressure: {bl['defense']['pressure']}")
    tips.append("One tell = mild bump; macros only on REPEATED tendency (2+) this game.")
    return [t for t in tips if t.split(":", 1)[-1].strip()]


def build_prep_plan(
    opponent_id: str,
    opp: dict[str, Any] | None = None,
    *,
    db: Any = None,
    persist: bool = True,
    profile: str | None = None,
    offline: bool = False,
    refresh_meta: bool = False,
    o_book: str | None = None,
    d_book: str | None = None,
    apply_books: bool = False,
) -> dict[str, Any]:
    from cfb_coach.madden.playbook import lock_books, plan_books

    opp = opp or load_profile(opponent_id, db)
    if profile is None:
        profile = get_session_profile(db)
    pcfg = profile_config(profile)
    bl = load_meta_baseline()
    inv = build_inventory()
    offense_only = is_cpu_opponent(opponent_id)
    arch = (opp.get("archetype") or "unknown").lower()

    books = plan_books(db, opp=opp, team=pcfg["team"], offense_only=offense_only,
                       o_book=o_book, d_book=d_book)
    book_deltas = [d for side in ("offense", "defense") for d in books[side]["deltas"]]
    proposed = propose_deltas(opponent_id, opp, profile=pcfg["id"])
    applied = get_applied_deltas(db, opponent_id)
    shown = book_deltas + filter_new_deltas(proposed, applied)

    active = list(pcfg["default_active"] or inv["macros_active"] + inv["offensive_macros"])
    active_after, swaps, replacing = resolve_loadout(active, shown, arch, offense_only=offense_only)
    cards, loadout = loadout_cards(active_after, db, offense_only=offense_only)
    budget = {
        "meter": loadout["meter"],
        "total": loadout["total"],
        "cap": USER_ACTIVE_CAP,
        "at_cap": loadout["total"] >= USER_ACTIVE_CAP,
    }

    tips = call_tips(opp, offense_only=offense_only, bl=bl)
    tips.extend(audible_tips(opp, books["offense"]["record"]["formations"]))
    if not pcfg["team"]:
        tips.append("Primary team TBD — books picked from the verified meta catalog "
                    "(set later: config --game madden27 --primary-team <NFL team>).")

    scout_dict: dict[str, Any]
    try:
        from cfb_coach.madden.meta_scout import apply_scout, run_madden_scout

        scout = run_madden_scout(offline=offline, refresh=refresh_meta)
        s_tips, _ = apply_scout(scout, profile=pcfg["id"], offense_only=offense_only, active=active_after)
        tips.extend(s_tips)
        scout_dict = scout.to_dict()
    except Exception as exc:  # noqa: BLE001 — never break prep
        scout_dict = {
            "available": False,
            "offline": offline,
            "message": f"Scout unavailable — using cached/baseline {META_VERSION} ({type(exc).__name__})",
            "baseline_fallback": META_VERSION,
            "confidence": "low",
        }

    plan = {
        "game_id": "madden27",
        "opponent_id": opponent_id,
        "display_name": opp.get("display_name", opponent_id),
        "team": opp.get("nfl_team") or opp.get("team_now") or "persona",
        "archetype": arch,
        "persona_confidence": opp.get("persona_confidence"),
        "film_confidence": opp.get("confidence"),
        "version": bl["version"],
        "game": "Madden 27 Franchise",
        "patch": bl.get("patch", ""),
        "patch_notes": list(bl.get("patch_notes") or []),
        "playbook": books,
        "proposed_deltas": proposed,
        "shown_deltas": shown,
        "applied_count": len(applied),
        "tips": tips,
        "ts": datetime.now(timezone.utc).isoformat(),
        "active_cap": USER_ACTIVE_CAP,
        "slot_budget": budget,
        "macro_cards": cards,
        "loadout": loadout,
        "active_after": active_after,
        "replacing_lines": replacing,
        "swap_banners": swaps,
        "offense_only": offense_only,
        "macro_catalog_version": load_macro_catalog().get("version", "?"),
        "profile": pcfg["id"],
        "profile_config": pcfg,
        "primary_team": profile_config("primary")["team"],
        "doctrine": doctrine_line(pcfg["team"]),
        "meta_scout": scout_dict,
    }
    if db is not None and persist:
        lock_books(db, books, applied=apply_books)
        save_prep(db, opponent_id, proposed, shown)
    return plan


def save_prep(db: Any, opponent_id: str, proposed: list[dict[str, Any]], shown: list[dict[str, Any]]) -> None:
    prev = db.get_install_sheet(opponent_id) or {}
    db.save_install_sheet(opponent_id, {
        "version": META_VERSION,
        "ts": datetime.now(timezone.utc).isoformat(),
        "proposed_deltas": proposed,
        "shown_deltas": shown,
        "applied_deltas": list(prev.get("applied_deltas") or []),
    })
    for d in shown:
        db.log_install_diff(opponent_id=opponent_id, action=d["action"], target=d["target"],
                            detail=d["detail"], why=d.get("why", ""))


def mark_applied(db: Any, opponent_id: str, deltas: list[dict[str, Any]]) -> None:
    sheet = db.get_install_sheet(opponent_id) or {}
    sheet["applied_deltas"] = list(deltas)
    sheet["applied_ts"] = datetime.now(timezone.utc).isoformat()
    sheet["version"] = META_VERSION
    db.save_install_sheet(opponent_id, sheet)


def _status_label(bp: dict[str, Any], oid: str) -> str:
    if bp.get("status") == "pending":
        return f"PENDING (build it, then prep --game madden27 -o {oid} --mark-applied; live calls use the last locked book)"
    return "LOCKED for live calls"


def format_delta_text(plan: dict[str, Any]) -> str:
    pcfg = plan["profile_config"]
    lines = [
        f"# PREP — vs {plan['display_name']} (persona: {plan['archetype']})  |  "
        f"{plan['game']} / {plan['version']}",
        f"Franchise profile: {pcfg['label']} ({pcfg['mode']}) — team: {pcfg['team_label']}",
    ]
    from cfb_coach.madden.playbook import format_book

    lines.append("## Playbook of record (locked by this prep)")
    for side in ("offense",) if plan["offense_only"] else ("offense", "defense"):
        bp = plan["playbook"][side]
        lines.append(f"  {side.title()}: {bp['record']['name']} [{bp['record']['mode']}] "
                     f"{_status_label(bp, plan['opponent_id'])} — {bp['reason']}")
        if bp["checklist"]:
            lines.append(f"  BUILD CUSTOM {side.upper()} BOOK — install exactly these formations:")
            for item in bp["checklist"]:
                lines.append(f"    [ ] {item['formation']} ({item['books']}): {', '.join(item['plays'])}")
    shown = plan["shown_deltas"]
    if not shown:
        lines.append("No playbook changes — keep the locked book as-is.")
    for d in shown:
        bit = f"  [{d['action']}] {d['target']}"
        if d.get("field"):
            bit += f" · {d['field']}"
        bit += f" [{d.get('validated_status')}] — {d['detail']}"
        lines.append(bit)
        if d.get("before") or d.get("after"):
            lines.append(f"      {d.get('before') or '—'} → {d.get('after') or '—'}")
        if d.get("why"):
            lines.append(f"      why: {d['why']}")
    for r in plan.get("replacing_lines") or []:
        lines.append(f"  Loadout: {r}")
    lines.append(f"## Active loadout — {plan['loadout']['meter']}")
    if plan["offense_only"]:
        lines.append("  D macros: N/A — offense only (CPU)")
    for c in plan["macro_cards"]:
        lines.append(f"  - {c['id']} ({c.get('side')}) [{c.get('validated_status')}] — {c.get('purpose', '')}")
    lines.append("## Full playbook (show)")
    for side in ("offense",) if plan["offense_only"] else ("offense", "defense"):
        lines.extend("  " + ln for ln in format_book(plan["playbook"][side]["record"]).splitlines())
    lines.append("## Call emphasis")
    lines.extend(f"  - {t}" for t in plan["tips"])
    scout = plan.get("meta_scout") or {}
    if not scout.get("available"):
        lines.append(f"## Live meta scout\n  {scout.get('message') or 'Scout unavailable'}")
    return "\n".join(lines)
