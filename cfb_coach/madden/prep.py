"""Madden 27 Franchise prep plan — playbook of record + macro deltas.

Every prep chooses a stock or custom playbook of record per side (see
`cfb_coach.madden.playbook`). First custom / switch-to-custom emits a full
formation checklist; successive custom preps show formation ADD/REMOVE only.
Macro (Custom Adjustments) deltas stay ADD/EDIT/BENCH and meta_grounded;
audible-slot swaps are tips, never install steps. The macro loadout (v1.17: 10 offense +
10 defense per opponent, settings shared with CFB) + call tips ride along. Live `play` is
hard-locked to the applied book.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from cfb_coach.install_sheet import filter_new_deltas, get_applied_deltas
from cfb_coach.madden.data import (
    META_VERSION,
    archetype_lean,
    load_macro_catalog,
    load_meta_baseline,
    load_seed,
)
from cfb_coach.madden.franchise import doctrine_line, get_session_profile, profile_config
from cfb_coach.madden.macros import PER_SIDE
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
    """Macro deltas: none since v1.17 — the defense macros and their settings are rebuilt from the
    research DB every prep (``madden/research_db.py``), so there is nothing to EDIT / ADD by hand.
    Formations go through the playbook of record."""
    del opponent_id, opp, profile
    return []


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
    offense_only = is_cpu_opponent(opponent_id)
    arch = (opp.get("archetype") or "unknown").lower()

    # 1) Fresh research every prep (web + YouTube transcripts, like CFB) — feeds the book pick
    scout_dict: dict[str, Any]
    scout = None
    research: dict[str, Any] = {}
    try:
        from cfb_coach.madden.meta_scout import research_from_scout, run_madden_scout

        scout = run_madden_scout(offline=offline, refresh=refresh_meta)
        research = research_from_scout(scout)
        scout_dict = scout.to_dict()
    except Exception as exc:  # noqa: BLE001 — never break prep
        scout_dict = {
            "available": False,
            "offline": offline,
            "message": f"Scout unavailable — using cached/baseline {META_VERSION} ({type(exc).__name__})",
            "baseline_fallback": META_VERSION,
            "confidence": "low",
            "mode": "seed",
            "research_status": "failed",
        }

    # 2) Books of record: research recommends the O + D book (start: Buccaneers O / 49ers D)
    if o_book is None or d_book is None:
        from cfb_coach.madden.franchise import book_choice, load_config

        cfg = load_config()
        o_book = o_book if o_book is not None else book_choice("offense", cfg)
        d_book = d_book if d_book is not None else book_choice("defense", cfg)
    books = plan_books(db, opp=opp, team=pcfg["team"], offense_only=offense_only,
                       o_book=o_book, d_book=d_book, research=research, opponent_id=opponent_id)
    book_deltas = [d for side in ("offense", "defense") for d in books[side]["deltas"]]
    proposed = propose_deltas(opponent_id, opp, profile=pcfg["id"])
    applied = get_applied_deltas(db, opponent_id)
    shown = book_deltas + filter_new_deltas(proposed, applied)

    # 3) Defense: 10 research-built macros (research DB pulled this prep + live scout +
    #    tendencies + learned weights). Offense: no macros — researched adjustments with buttons.
    from cfb_coach.madden import research_db as rdb
    from cfb_coach.madden.adjustments import controls_table, offense_plan
    from cfb_coach.madden.macro_pool import pool_macro
    from cfb_coach.madden.macro_select import select_loadout
    from cfb_coach.madden.macros import (
        attach_detail,
        copy_checklist,
        legacy_picks,
        load_selection,
        macro_status,
        missing_settings_report,
    )

    rdb.load(pull=not offline)
    prev = (load_selection(db, opponent_id) or {}).get("defense") or []
    pick = select_loadout(opponent_id, db=db, archetype=arch, scout=scout, offense_only=offense_only,
                          previous=prev + legacy_picks(db, opponent_id))
    selection = {"offense": [], "defense": pick["defense"]}
    rows = {r["id"]: r for r in pick["ranked"]}
    d_book = books.get("defense", {}).get("record", {}).get("formations")
    cards = []
    for rank, mid in enumerate(selection["defense"], 1):
        meta = pool_macro(mid) or {}
        card = {"id": mid, "name": meta.get("xbox_name") or mid, "side": "defense", "slot": "active",
                "rank": rank, "score": rows.get(mid, {}).get("score"), "why": rows.get(mid, {}).get("why", ""),
                "purpose": meta.get("purpose") or "", "when_to_arm": meta.get("when_to_arm") or "",
                "validated_status": macro_status(mid, db)}
        cards.append(attach_detail(card, d_book))
    active_after = list(selection["defense"])
    missing = missing_settings_report(active_after)
    n_d = len(selection["defense"])
    meter = "Offense: adjustments (CPU — offense only)" if offense_only else f"Defense {n_d} macros · offense: adjustments"
    loadout = {"meter": meter, "total": n_d, "offense": [], "defense": selection["defense"], "per_side": PER_SIDE}
    budget = {"meter": meter, "total": n_d, "cap": PER_SIDE, "at_cap": False}
    research_db_status = rdb.status()

    tips = call_tips(opp, offense_only=offense_only, bl=bl)
    tips.extend(audible_tips(opp, books["offense"]["record"]["formations"]))
    if not pcfg["team"]:
        tips.append("Primary team not set — books picked from research + the verified catalog "
                    "(config --game madden27 --primary-team \"Detroit Lions\").")
    if scout is not None:
        try:
            from cfb_coach.madden.meta_scout import apply_scout

            s_tips, _ = apply_scout(scout, profile=pcfg["id"], offense_only=offense_only, active=active_after)
            tips.extend(s_tips)
            scout_dict = scout.to_dict()
        except Exception:  # noqa: BLE001
            pass

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
        "active_cap": PER_SIDE,
        "slot_budget": budget,
        "macro_cards": cards,
        "loadout": loadout,
        "macro_selection": selection,
        "macro_ranking": [{k: r[k] for k in ("id", "score", "why")} for r in pick["ranked"]],
        "offense_adjustments": offense_plan(),
        "controls": controls_table(),
        "research_db": research_db_status,
        "copy_checklist": copy_checklist(selection),
        "active_after": active_after,
        "replacing_lines": [],
        "swap_banners": [],
        "offense_only": offense_only,
        "macro_catalog_version": load_macro_catalog().get("version", "?"),
        "profile": pcfg["id"],
        "profile_config": pcfg,
        "primary_team": profile_config("primary")["team"],
        "doctrine": doctrine_line(pcfg["team"]),
        "meta_scout": scout_dict,
        "research": {"mode": research.get("mode") or scout_dict.get("mode") or "",
                     "status": scout_dict.get("research_status") or "",
                     "books": research.get("books") or {}},
        "missing_settings": missing,
    }
    if db is not None and persist:
        lock_books(db, books, applied=apply_books)
        save_prep(db, opponent_id, proposed, shown)
        from cfb_coach.madden.macros import store_selection

        store_selection(db, opponent_id, selection)
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

    rs = plan.get("research") or {}
    scout = plan.get("meta_scout") or {}
    lines.append(f"## Research — {scout.get('message') or rs.get('mode') or 'n/a'}")
    for side in ("offense",) if plan["offense_only"] else ("offense", "defense"):
        top = list(((rs.get("books") or {}).get(side) or {}).items())[:4]
        if top:
            lines.append(f"  {side} books named: " + ", ".join(f"{b} ({v.get('docs', 0)} src)" for b, v in top))
        rec = plan["playbook"][side].get("recommendation") or {}
        if rec.get("rows"):
            lines.append(f"  {side} book scores: " + ", ".join(f"{r['book']} {r['score']:.2f}" for r in rec["rows"][:4]))
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
    lines.append(f"## {(plan.get('research_db') or {}).get('line', '')}")
    lines.append("## Offense adjustments (no macros) — called live only when the look calls for it")
    for a in plan.get("offense_adjustments") or []:
        lines.append(f"  - vs {a['vs']}: {a['label']} — {a['buttons']}  ({a['why']})")
    if plan["offense_only"]:
        lines.append("## Defense macros: N/A — offense only (CPU)")
    else:
        lines.append(f"## Defense macros — {plan['loadout']['meter']} (research-built settings)")
        for c in plan["macro_cards"]:
            ing = c.get("ingame") or {}
            lines.append(f"   {c.get('rank', 0):>2}. {c['id']} [{c.get('validated_status')}] — {c.get('purpose', '')}")
            lines.append(f"       fire: {ing.get('buttons')} · {ing.get('n_researched', 0)}/{ing.get('n_fields', 0)} "
                         f"fields researched: {ing.get('key')}")
        lines.append("## Copy checklist")
        lines.extend("  " + ln for ln in (plan.get("copy_checklist") or "").splitlines())
    lines.append("## Full playbook (show)")
    for side in ("offense",) if plan["offense_only"] else ("offense", "defense"):
        lines.extend("  " + ln for ln in format_book(plan["playbook"][side]["record"]).splitlines())
    lines.append("## Call emphasis")
    lines.extend(f"  - {t}" for t in plan["tips"])
    if not scout.get("available"):
        lines.append(f"## Live meta scout\n  {scout.get('message') or 'Scout unavailable'}")
    return "\n".join(lines)
