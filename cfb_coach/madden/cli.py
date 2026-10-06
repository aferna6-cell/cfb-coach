"""`--game madden27` command handlers. CFB handlers in cfb_coach.cli stay untouched."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.games import GAMES, MADDEN27, madden_db_path
from cfb_coach.madden.data import load_seed
from cfb_coach.madden.franchise import (
    doctrine_line,
    get_session_profile,
    load_config,
    profile_config,
    save_config,
    set_session_profile,
    team_label,
)
from cfb_coach.madden.playcaller import active_pivot, make_call
from cfb_coach.madden.situation import format_heard, parse_madden_situation
from cfb_coach.opponents import is_cpu_opponent, resolve_opponent

PROFILE = GAMES[MADDEN27]


def open_db() -> CoachDB:
    db = CoachDB(madden_db_path(), seed=load_seed())
    try:  # v1.17: stored Active 8 → 10 offense + 10 defense (nothing dropped; idempotent)
        from cfb_coach.madden.macros import migrate_all_selections

        migrate_all_selections(db)
    except Exception:  # noqa: BLE001 — never block opening the DB
        pass
    return db


def _require_opponent(raw: str) -> str:
    oid = resolve_opponent(raw)
    if not oid:
        print(f"Unknown opponent: {raw!r}\n\n{format_opponents()}", file=sys.stderr)
        raise SystemExit(2)
    return oid


def _profile_arg(args: argparse.Namespace, db: CoachDB) -> str:
    return set_session_profile(db, getattr(args, "franchise", None) or get_session_profile(db))


def _profile_header(pid: str) -> str:
    cfg = profile_config(pid)
    exp = " [experimental]" if cfg["experimental_badge"] else ""
    return f"Madden 27 Franchise — {cfg['label']} ({cfg['mode']}){exp} — team: {cfg['team_label']}"


def _active_after_prep(db: CoachDB, oid: str, pid: str) -> dict[str, list[str]]:
    """Custom Adjustments from the last prep. Without one, rank defense only (no offense book yet)."""
    from cfb_coach.madden.macro_select import select_loadout
    from cfb_coach.madden.macros import legacy_picks, load_selection

    del pid
    stored = load_selection(db, oid)
    if stored:
        return stored
    prof = db.get_opponent(oid) or {}
    pick = select_loadout(oid, db=db, archetype=prof.get("archetype") or "", offense_only=is_cpu_opponent(oid),
                          previous=legacy_picks(db, oid))
    return {"offense": list(pick["offense"]), "defense": list(pick["defense"])}


def _active_line(active: dict[str, list[str]], cpu: bool) -> str:
    if cpu:
        return "Active macros (from last prep): none — CPU = offense only; offense calls carry adjustments"
    o = ", ".join(active.get("offense") or []) or "none"
    d = ", ".join(active.get("defense") or []) or "none"
    return (f"Active macros (from last prep): O {len(active.get('offense') or [])}: {o} | "
            f"D {len(active.get('defense') or [])}: {d}")


# ---------------------------------------------------------------------------
# opponents / config
# ---------------------------------------------------------------------------

def format_opponents() -> str:
    seed = load_seed()
    lines = [
        "Madden 27 Franchise — shared personas (same cast as CFB)",
        "ID          Display        Archetype              Persona  Madden film",
        "-" * 74,
    ]
    for oid, o in seed["opponents"].items():
        lines.append(
            f"{oid:<12}{o['display_name']:<15}{o['archetype']:<23}"
            f"{o['persona_confidence']:<9}{o['confidence']}"
        )
    lines.append("")
    lines.append("CPU = offense-only (adjustments). User personas = O + D (10 defense macros; offense adjustments).")
    return "\n".join(lines)


def cmd_opponents(_args: argparse.Namespace) -> int:
    print(format_opponents())
    return 0


def _session_live_macros(args: argparse.Namespace) -> bool:
    """``--no-macros`` forces this session off. Otherwise the config decides."""
    from cfb_coach.madden.macro_policy import resolve_live_macros

    if getattr(args, "no_macros", False):
        return False
    return resolve_live_macros(None)


def cmd_config(args: argparse.Namespace) -> int:
    if getattr(args, "no_macros", False) and getattr(args, "macros", False):
        raise SystemExit("Pass only one of --no-macros or --macros.")
    live_macros = None
    if getattr(args, "no_macros", False):
        live_macros = False
    elif getattr(args, "macros", False):
        live_macros = True
    changing = any(
        getattr(args, k, None)
        for k in ("primary_team", "lab_team", "clear_primary", "clear_lab", "o_book", "d_book")
    ) or live_macros is not None
    if changing:
        try:
            _, warnings = save_config(
                primary_team=args.primary_team,
                lab_team=args.lab_team,
                clear_primary=bool(args.clear_primary),
                clear_lab=bool(args.clear_lab),
                offense_book=getattr(args, "o_book", None),
                defense_book=getattr(args, "d_book", None),
                live_macros=live_macros,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        for w in warnings:
            print(f"warning: {w}")
    cfg = load_config()
    from cfb_coach.madden.franchise import config_path

    print(f"Madden 27 Franchise config → {config_path()}")
    print(f"  primary team (franchise_primary): {team_label(cfg.get('primary_team'))}")
    print(f"  lab team     (franchise_lab):     {team_label(cfg.get('lab_team'))}")
    from cfb_coach.madden.franchise import DEFAULT_START_BOOK

    for side, key in (("offense", "offense_book"), ("defense", "defense_book")):
        val = cfg.get(key) or "auto"
        note = (f" (research picks each prep; starts on stock {DEFAULT_START_BOOK[side]})" if val == "auto" else "")
        print(f"  {side} book: {val}{note}")
    macros_on = bool(cfg.get("live_macros", True))
    print("  live macros: " + ("on" if macros_on else "off")
          + (" — play --game madden27 --no-macros turns off one session;"
             " config --macros turns them back on" if not macros_on else
             " — only a confirmed live look, red zone, or clock/score; config --no-macros turns them off"))
    if not cfg.get("primary_team"):
        print("  Set: config --game madden27 --primary-team \"Detroit Lions\"")
    return 0


def cmd_macro_settings(args: argparse.Namespace) -> int:
    """Show the research-built defense macros (every editor field + source) and the Xbox buttons.
    Madden macro settings come from the research DB (daily routine) — not typed in by hand."""
    from cfb_coach.madden import research_db as rdb
    from cfb_coach.madden.adjustments import controls_table
    from cfb_coach.madden.macro_pool import pool_ids
    from cfb_coach.madden.macros import copy_block, macro_detail

    if args.settings or args.clear or args.xbox_name:
        raise SystemExit("Madden 27 macro settings are research-built (research DB, refreshed daily) — they "
                         "aren't entered by hand. CFB 27 settings: macro-settings --game cfb27 NAME --set ...")
    rdb.load(pull=not getattr(args, "offline", False))
    print(rdb.status()["line"])
    if args.macro:
        det = macro_detail(args.macro.upper())
        if not det["settings"]:
            raise SystemExit(f"{args.macro}: not in the research DB (known: {', '.join(pool_ids('defense'))})")
        print(copy_block(det))
        print("Sources:")
        for s_ in det["sources"]:
            print(f"  - {s_['title']} — {s_['url']}")
        return 0
    print("Defense macros (research-built; fields no source names = Default):")
    for mid in pool_ids("defense"):
        det = macro_detail(mid)
        print(f"  {mid:<16} {det['n_researched']:>2}/{det['n_fields']} fields researched — fire: {det['buttons']}")
    print("Xbox pre-snap controls:")
    for c in controls_table():
        print(f"  {c['side']:<8} {c['action']:<20} {c['buttons']}  [{c['confidence']}]")
    return 0


# ---------------------------------------------------------------------------
# prep / call / postgame / promote
# ---------------------------------------------------------------------------

def cmd_prep(args: argparse.Namespace) -> int:
    from cfb_coach.madden.prep import build_prep_plan, format_delta_text, mark_applied
    from cfb_coach.madden.prep_browser import generate_and_open

    from cfb_coach.madden.playbook import parse_choice

    try:
        parse_choice("offense", args.o_book)
        parse_choice("defense", args.d_book)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    oid = _require_opponent(args.opponent)
    db = open_db()
    try:
        pid = _profile_arg(args, db)
        if getattr(args, "text", False):
            plan = build_prep_plan(
                oid, db=db, persist=True, profile=pid,
                offline=args.offline, refresh_meta=args.refresh_meta,
                o_book=args.o_book, d_book=args.d_book, apply_books=args.mark_applied,
                opp_team=getattr(args, "opp_team", None),
                n_gameplan=getattr(args, "gameplan", 8),
            )
            if args.mark_applied:
                mark_applied(db, oid, plan["proposed_deltas"])
                print(f"Marked {len(plan['proposed_deltas'])} deltas applied for {oid}.")
            print(_profile_header(pid))
            print(doctrine_line(profile_config(pid)["team"]))
            print(format_delta_text(plan))
            return 0
        path, plan = generate_and_open(
            oid, db=db, persist=True, open_browser=not args.no_open,
            mark=args.mark_applied, profile=pid,
            offline=args.offline, refresh_meta=args.refresh_meta,
            o_book=args.o_book, d_book=args.d_book,
            opp_team=getattr(args, "opp_team", None),
            n_gameplan=getattr(args, "gameplan", 8),
        )
        n = len(plan.get("shown_deltas") or [])
        print(f"Prep (Madden 27 Franchise) vs {plan['display_name']} → {path}")
        for side in ("offense",) if plan["offense_only"] else ("offense", "defense"):
            bp = plan["playbook"][side]
            state = "LOCKED" if bp.get("status") != "pending" else "PENDING until --mark-applied"
            print(f"{side.title()} book: {bp['record']['name']} [{bp['record']['mode']}] {state} — {bp['reason']}")
            if bp["checklist"]:
                print(f"  BUILD CUSTOM {side.upper()} BOOK — {len(bp['checklist'])} formations (see browser / playbook --game madden27)")
        print(_profile_header(pid))
        print(doctrine_line(profile_config(pid)["team"]))
        sel = plan.get("macro_selection") or {}
        if plan["offense_only"]:
            print("Custom Adjustments: none (CPU — offense only).")
        else:
            print(
                f"Custom Adjustments: {len(sel.get('offense') or [])} offense + "
                f"{len(sel.get('defense') or [])} defense (Create & Share → Custom Adjustments, LB in game)."
            )
        for warning in plan.get("playbook_warnings") or []:
            print(warning if str(warning).startswith("WARNING:") else f"WARNING: {warning}")
        if plan["offense_only"]:
            print("CPU opponent — OFFENSE-ONLY prep (no D macros).")
        if args.mark_applied:
            print(f"Marked proposed deltas applied for {oid}.")
        elif n == 0:
            print("No playbook changes — keep the active book (tips in browser).")
        else:
            print(f"{n} adjustment(s) shown (deltas only).")
        scout = plan.get("meta_scout") or {}
        if scout.get("available"):
            tag = "cached" if scout.get("from_cache") else "live"
            print(f"Meta scout ({tag}, conf={scout.get('confidence', '?')}): "
                  f"{len(scout.get('suggestions') or [])} book suggestion(s).")
        else:
            print(scout.get("message") or "Scout unavailable — using baseline madden27-2026-09")
    finally:
        db.close()
    return 0


def cmd_call(args: argparse.Namespace) -> int:
    oid = _require_opponent(args.opponent)
    db = open_db()
    try:
        pid = get_session_profile(db)
        sit = parse_madden_situation(args.situation, default_side=args.side or "offense")
        if args.side:
            sit.side = args.side
        from cfb_coach.game_score import absorb_and_stamp, context_from_args

        absorb_and_stamp(sit, context_from_args(args))
        from cfb_coach.madden.playbook import NoActivePlaybook

        try:
            call = make_call(
                sit, oid, db, active_macros=_active_after_prep(db, oid, pid),
                live_macros=_session_live_macros(args),
            )
        except NoActivePlaybook as exc:
            print(str(exc).replace("<opp>", oid), file=sys.stderr)
            return 2
        print(call.format())
        if args.why:
            print(f"  ({call.rationale})")
    finally:
        db.close()
    return 0


def cmd_playbook(args: argparse.Namespace) -> int:
    from cfb_coach.madden.playbook import format_book, load_books, load_pending

    db = open_db()
    try:
        locked = load_books(db)
        pending = load_pending(db)
        sides = (args.side,) if args.side else ("offense", "defense")
        print("Madden 27 playbook of record (live calls are locked to these formations/plays)")
        for side in sides:
            if side in locked:
                print("LOCKED " + format_book(locked[side]))
            else:
                print(f"{side.title()}: no book locked — run `prep --game madden27 -o <opp>` first")
            if side in pending:
                print("PENDING (build it, then prep --mark-applied) " + format_book(pending[side]))
    finally:
        db.close()
    return 0


def cmd_postgame(args: argparse.Namespace) -> int:
    from cfb_coach.madden.postgame import summary

    oid = _require_opponent(args.opponent)
    db = open_db()
    try:
        pid = _profile_arg(args, db)
        print(summary(db, oid, pid))
    finally:
        db.close()
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    from cfb_coach.madden.postgame import format_promotions, promote

    db = open_db()
    try:
        if args.accept_all or args.target:
            res = promote(db, target=args.target, kind=args.kind, accept_all=bool(args.accept_all))
            acc = res.get("accepted") or []
            print(f"Accepted {len(acc)} promotion(s) for the primary Franchise profile."
                  if acc else "No matching pending promotions to accept.")
        print(format_promotions(db))
    finally:
        db.close()
    return 0


# ---------------------------------------------------------------------------
# play (typed live + overlay)
# ---------------------------------------------------------------------------

def _overlay_path(args: argparse.Namespace) -> Path | None:
    from cfb_coach.copilot import default_overlay_path

    if getattr(args, "no_overlay", False):
        return None
    if getattr(args, "overlay", None):
        return Path(args.overlay)
    return default_overlay_path(PROFILE.overlay_filename)


def _write_overlay(path: Path | None, text: str, short: str) -> None:
    if path is None:
        return
    from cfb_coach.copilot import write_overlay_html

    write_overlay_html(
        str(path), None, [], call_text=text, short_line=short, mode="play",
        brand=PROFILE.brand, play_cmd="cfb-coach play --game madden27",
    )


def cmd_play(args: argparse.Namespace) -> int:
    oid = _require_opponent(args.opponent)
    db = open_db()
    pid = _profile_arg(args, db)
    active = _active_after_prep(db, oid, pid)
    cpu = is_cpu_opponent(oid)
    overlay = _overlay_path(args)

    print(f"LIVE PLAY — Madden 27 Franchise vs {oid}  (db: {db.path})")
    print(_profile_header(pid))
    print(doctrine_line(profile_config(pid)["team"]))
    from cfb_coach.madden.playbook import NoActivePlaybook, active_books, load_pending

    try:
        books = active_books(db, ("offense",) if cpu else ("offense", "defense"))
    except NoActivePlaybook as exc:
        print(str(exc).replace("<opp>", oid), file=sys.stderr)
        db.close()
        return 2
    print(_active_line(active, cpu))
    print(f"Locked book: O = {books['offense']['name']} [{books['offense']['mode']}]"
          + ("" if cpu else f" · D = {books['defense']['name']} [{books['defense']['mode']}]")
          + " — calls stay inside it (playbook --game madden27 to list)")
    for side, rec in load_pending(db).items():
        if side == "offense" or not cpu:
            print(f"  note: {side} custom book rev {rec.get('rev')} is PENDING — build it, then "
                  f"`prep --game madden27 -o {oid} --mark-applied` to switch live calls to it")
    if cpu:
        print("CPU opponent — OFFENSE-ONLY coaching (no defense calls / no D macros).")
        print("Commands: result <text> | why | quit  (side d disabled)")
    else:
        print("User game — O + D. Prefix 'd ' for defense. Commands: side o|d | result <text> | why | quit")
    print("  Type D&D (+ yl) + previous play/coverage name, e.g. '2&7 my 35 stick wheel' | '1&10 cover 3 match'")
    print("  Live look only with: showing / live / pre-snap / aligned (e.g. 'showing cover 2 man')")
    print("  A play or result with no down (mesh, 4 verts, cover 2, +7, td, int) is the last snap, not a new call. undo removes it.")
    print("  Score (optional, us-them): --score 21-14, or `score 21-14` / `score clear` mid-game.")
    print("  Quarter: --quarter 4, `quarter 4`, or `q4` on the sit line. Close early games stay neutral.")
    macros_on = _session_live_macros(args)
    if macros_on:
        print("  Live macros: on. A call stays plain unless the look is on the field and already confirmed,")
        print("  or the snap is red zone / two-minute / protecting a lead. --no-macros turns them off.")
    else:
        print("  Live macros: OFF. Plays still come out; no Custom Adjustment is suggested.")

    from cfb_coach.game_score import absorb_and_stamp, context_from_args, interpret_live_command, sit_prompt

    live_ctx = context_from_args(args)
    if live_ctx.describe():
        print(f"  Game situation: {live_ctx.describe()} (sticks until you update it)")

    if getattr(args, "once", None):
        sit = parse_madden_situation(args.once, default_side="offense")
        absorb_and_stamp(sit, live_ctx)
        heard = format_heard(sit)
        print(heard)
        call = make_call(sit, oid, db, active_macros=active, live_macros=macros_on)
        print(call.headline())
        print(call.format())
        if args.why:
            print(f"  ({call.rationale})")
        _write_overlay(overlay, call.headline() + "\n" + call.format(), heard)
        if overlay is not None:
            print(f"  overlay → {overlay}")
        db.close()
        return 0

    use_html = not bool(getattr(args, "terminal", False) or getattr(args, "no_html", False))
    if use_html:
        from cfb_coach.live_server import LivePlayController, run_live_server
        from cfb_coach.madden.playbook import live_apply, live_book_info

        def _make(sit, **kwargs):
            kwargs.setdefault("live_macros", macros_on)
            return make_call(sit, oid, db, active_macros=active, **kwargs)

        def _learn():
            from cfb_coach.madden.postgame import summary

            return summary(db, oid, profile=pid)

        ctrl = LivePlayController(
            db=db,
            opponent_id=oid,
            make_call=_make,
            parse_situation=parse_madden_situation,
            learn_summary=_learn,
            brand="Madden 27 Franchise",
            play_cmd="cfb-coach play --game madden27",
            dynasty=pid,
            dynasty_label=profile_config(pid)["label"],
            cpu_only=cpu,
            book_info=lambda: live_book_info(db, cpu=cpu, profile_label=profile_config(pid)["label"]),
            book_apply=lambda rev: live_apply(db, rev),
            live_score=live_ctx.score,
            quarter=live_ctx.quarter,
        )
        print("HTML live input ON (default). Use --terminal / --no-html for classic sit> loop.")
        try:
            return run_live_server(
                ctrl,
                port=getattr(args, "html_port", None),
                open_browser=True,
            )
        finally:
            db.close()

    if overlay is not None:
        from cfb_coach.prep_browser import open_prep_html

        _write_overlay(overlay, "waiting for sit> …", f"Madden 27 vs {oid}")
        open_prep_html(overlay, open_browser=True)
        print(f"Overlay ON → {overlay}  (--no-overlay to disable)")
    else:
        print("Overlay OFF (--no-overlay)")
    print("-" * 60)

    side = "offense"
    last_call = last_sit = None
    last_cov: str | None = None
    last_concept: str | None = None
    from cfb_coach.last_snap import LastSnapBook

    snap_book = LastSnapBook(db, oid, parse_madden_situation)
    if live_ctx.score is not None:
        snap_book.spot.score_us = live_ctx.score.us
        snap_book.spot.score_them = live_ctx.score.them
    try:
        while True:
            try:
                raw = input(sit_prompt(side, live_ctx)).strip()
            except EOFError:
                print()
                break
            if not raw:
                continue
            low = raw.lower()
            if low in ("q", "quit", "exit"):
                break
            if low in ("o", "side o", "offense"):
                side = "offense"
                print("  side → offense")
                continue
            if low in ("d", "side d", "defense"):
                if cpu:
                    print("  CPU = offense-only — defense calls disabled")
                else:
                    side = "defense"
                    print("  side → defense")
                continue
            if low == "why" and last_call:
                print(f"  ({last_call.rationale})")
                continue
            if low in ("undo", "undo last"):
                print(snap_book.undo())
                if snap_book.spot.score_us is not None:
                    from cfb_coach.game_score import GameScore

                    live_ctx.score = GameScore(snap_book.spot.score_us, snap_book.spot.score_them or 0)
                continue
            score_msg = interpret_live_command(raw, live_ctx)
            if score_msg is not None:
                print(score_msg)
                if live_ctx.score is not None:
                    snap_book.spot.score_us = live_ctx.score.us
                    snap_book.spot.score_them = live_ctx.score.them
                continue
            note_raw = raw.split(" ", 1)[1].strip() if low.startswith(("result ", "log ")) else raw
            noted = snap_book.handle(note_raw)
            if noted is not None:
                print(noted)
                if snap_book.concept and last_call and last_call.side == "defense":
                    last_concept = snap_book.concept
                if snap_book.coverage and last_call and last_call.side == "offense":
                    last_cov = snap_book.coverage
                if snap_book.spot.score_us is not None:
                    from cfb_coach.game_score import GameScore

                    live_ctx.score = GameScore(snap_book.spot.score_us, snap_book.spot.score_them or 0)
                if snap_book.wrote:
                    for s in ("offense", "defense"):
                        tip = active_pivot(db, oid, s)
                        if tip and (s == "offense" or not cpu):
                            print(f"  {tip}")
                continue

            sit = parse_madden_situation(raw, default_side=side)
            snap_book.stamp_situation(sit)
            absorb_and_stamp(sit, live_ctx)
            heard = format_heard(sit)
            print(heard)
            call = make_call(
                sit, oid, db, active_macros=active, live_macros=macros_on,
                last_coverage=last_cov if sit.side == "offense" else None,
                last_concept=last_concept if sit.side == "defense" else None,
            )
            print(call.headline())
            print(call.format())
            _write_overlay(overlay, call.headline() + "\n" + call.format(), heard)
            last_call, last_sit = call, sit
            snap_book.remember_call(call, sit)
            if sit.coverage_hint and sit.side == "offense":
                last_cov = sit.coverage_hint
            if sit.concept_hint and sit.side == "defense":
                last_concept = sit.concept_hint
    finally:
        db.close()
    return 0
