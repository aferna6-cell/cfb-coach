"""`--game madden27` command handlers. CFB handlers in cfb_coach.cli stay untouched."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
from cfb_coach.tendency import mild_bump_concept, mild_bump_coverage

PROFILE = GAMES[MADDEN27]


def open_db() -> CoachDB:
    return CoachDB(madden_db_path(), seed=load_seed())


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


def _active_after_prep(db: CoachDB, oid: str, pid: str) -> list[str]:
    """Active-8 for live calls = profile default + ADD deltas already marked applied."""
    from cfb_coach.install_sheet import get_applied_deltas
    from cfb_coach.madden.prep import resolve_loadout

    active = profile_config(pid)["default_active"]
    applied = [d for d in get_applied_deltas(db, oid) if d.get("kind") == "macro"]
    prof = db.get_opponent(oid) or {}
    after, _, _ = resolve_loadout(active, applied, prof.get("archetype"), offense_only=is_cpu_opponent(oid))
    return after


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
    lines.append("CPU = offense-only. User personas = O + D (8-macro cap).")
    return "\n".join(lines)


def cmd_opponents(_args: argparse.Namespace) -> int:
    print(format_opponents())
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    changing = any(
        getattr(args, k, None)
        for k in ("primary_team", "lab_team", "clear_primary", "clear_lab")
    )
    if changing:
        _, warnings = save_config(
            primary_team=args.primary_team,
            lab_team=args.lab_team,
            clear_primary=bool(args.clear_primary),
            clear_lab=bool(args.clear_lab),
        )
        for w in warnings:
            print(f"warning: {w}")
    cfg = load_config()
    from cfb_coach.madden.franchise import config_path

    print(f"Madden 27 Franchise config → {config_path()}")
    print(f"  primary team (franchise_primary): {team_label(cfg.get('primary_team'))}")
    print(f"  lab team     (franchise_lab):     {team_label(cfg.get('lab_team'))}")
    if not cfg.get("primary_team"):
        print("  Set later: config --game madden27 --primary-team \"<NFL team>\"")
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
                o_book=args.o_book, d_book=args.d_book,
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
        )
        n = len(plan.get("shown_deltas") or [])
        print(f"Prep (Madden 27 Franchise) vs {plan['display_name']} → {path}")
        for side in ("offense",) if plan["offense_only"] else ("offense", "defense"):
            bp = plan["playbook"][side]
            print(f"{side.title()} book: {bp['record']['name']} [{bp['record']['mode']}] — {bp['reason']}")
            if bp["checklist"]:
                print(f"  BUILD CUSTOM {side.upper()} BOOK — {len(bp['checklist'])} formations (see browser / playbook --game madden27)")
        print(_profile_header(pid))
        print(doctrine_line(profile_config(pid)["team"]))
        if plan["offense_only"]:
            print("CPU opponent — OFFENSE-ONLY prep (no D macros).")
        if args.mark_applied:
            print(f"Marked proposed deltas applied for {oid}.")
        elif n == 0:
            print("No playbook changes — run scheme pack as-is (tips in browser).")
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
        call = make_call(sit, oid, db, active_macros=_active_after_prep(db, oid, pid))
        print(call.format())
        if args.why:
            print(f"  ({call.rationale})")
    finally:
        db.close()
    return 0


def cmd_playbook(args: argparse.Namespace) -> int:
    from cfb_coach.madden.playbook import active_books, format_book, load_books

    db = open_db()
    try:
        locked = load_books(db)
        books = active_books(db)
        sides = (args.side,) if args.side else ("offense", "defense")
        print("Madden 27 playbook of record (live calls are locked to these formations/plays)")
        for side in sides:
            if side not in locked:
                print(f"  ({side}: no prep yet — default stock book)")
            print(format_book(books[side]))
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


def _log_result(db: CoachDB, oid: str, call: Any, sit: Any, result: str) -> tuple[str | None, str | None]:
    res_sit = parse_madden_situation(result, default_side=sit.side)
    cov = res_sit.coverage_hint or sit.coverage_hint
    concept = res_sit.concept_hint or sit.concept_hint
    db.log_snap(
        opponent_id=oid, side=call.side, situation_raw=sit.raw,
        our_call=call.format().split("\n")[0], formation=call.formation, play=call.play,
        macro=call.macro, down=sit.down, distance=sit.distance, yardline=sit.yardline,
        result=result, coverage_seen=cov, concept_seen=concept,
    )
    success = any(w in result.lower() for w in ("td", "+", "good", "convert", "stop", "sack", "int"))
    if concept and call.side == "defense":
        mild_bump_concept(db, oid, concept, sit, success=success)
        print(f"  logged: {result} | mild bump concept={concept} (no hard-counter next snap)")
        return None, concept
    if cov and call.side == "offense":
        mild_bump_coverage(db, oid, cov, sit)
        print(f"  logged: {result} | mild bump coverage={cov} (no hard-counter next snap)")
        return cov, None
    print(f"  logged: {result}")
    return None, None


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
    from cfb_coach.madden.playbook import active_books

    books = active_books(db)
    print(f"Locked book: O = {books['offense']['name']} [{books['offense']['mode']}]"
          + ("" if cpu else f" · D = {books['defense']['name']} [{books['defense']['mode']}]")
          + " — calls stay inside it (playbook --game madden27 to list)")
    if cpu:
        print("CPU opponent — OFFENSE-ONLY coaching (no defense calls / no D macros).")
        print("Commands: result <text> | why | quit  (side d disabled)")
    else:
        print("User game — O + D. Prefix 'd ' for defense. Commands: side o|d | result <text> | why | quit")
    print("  Type D&D (+ yl) + previous play/coverage name, e.g. '2&7 my 35 stick wheel' | '1&10 cover 3 match'")
    print("  Live look only with: showing / live / pre-snap / aligned (e.g. 'showing cover 2 man')")

    if getattr(args, "once", None):
        sit = parse_madden_situation(args.once, default_side="offense")
        heard = format_heard(sit)
        print(heard)
        call = make_call(sit, oid, db, active_macros=active)
        print(call.format())
        if args.why:
            print(f"  ({call.rationale})")
        _write_overlay(overlay, call.format(), heard)
        if overlay is not None:
            print(f"  overlay → {overlay}")
        db.close()
        return 0

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
    try:
        while True:
            try:
                raw = input(f"[{side[0].upper()}] sit> ").strip()
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
            if low.startswith(("result ", "log ")):
                if not last_call:
                    print("  No call to log yet.")
                    continue
                cov, concept = _log_result(db, oid, last_call, last_sit, raw.split(" ", 1)[1].strip())
                last_cov = cov or last_cov
                last_concept = concept or last_concept
                for s in ("offense", "defense"):
                    tip = active_pivot(db, oid, s)
                    if tip and (s == "offense" or not cpu):
                        print(f"  {tip}")
                continue

            sit = parse_madden_situation(raw, default_side=side)
            heard = format_heard(sit)
            print(heard)
            call = make_call(
                sit, oid, db, active_macros=active,
                last_coverage=last_cov if sit.side == "offense" else None,
                last_concept=last_concept if sit.side == "defense" else None,
            )
            print(call.format())
            _write_overlay(overlay, call.format(), heard)
            last_call, last_sit = call, sit
            if sit.coverage_hint and sit.side == "offense":
                last_cov = sit.coverage_hint
            if sit.concept_hint and sit.side == "defense":
                last_concept = sit.concept_hint
    finally:
        db.close()
    return 0
