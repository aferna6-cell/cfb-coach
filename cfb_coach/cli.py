"""CLI: prep / play / postgame / promote / opponents / call / config / watch.

`--game cfb27` (default) runs the CFB 27 coach unchanged; `--game madden27`
dispatches to cfb_coach.madden.cli (Madden 27 Franchise).
"""

from __future__ import annotations

import argparse
import os
import sys

from cfb_coach.db import CoachDB, resolve_db_path_from_env
from cfb_coach.opponents import format_opponent_list, resolve_opponent
from cfb_coach.playcaller import make_call
from cfb_coach.situation import parse_situation, format_heard
from cfb_coach.tendency import mild_bump_concept, mild_bump_coverage
from cfb_coach.dynasty import (
    DEFAULT_DYNASTY,
    dynasty_config,
    doctrine_line,
    normalize_dynasty,
    set_session_dynasty,
)


def _db() -> CoachDB:
    return CoachDB(resolve_db_path_from_env())


def _local_ts(ts: str) -> str:
    from cfb_coach.meta_scout import local_ts

    return local_ts(ts)


def _ensure_rules(db: CoachDB) -> None:
    """One-time rebuild of learned weights when the retrain rules version changed.

    Writes a timestamped backup of the DB first and prints a one-line notice.
    Never blocks prep/play: failures are reported and ignored.
    """
    try:
        from cfb_coach.learning import ensure_rules_current

        ensure_rules_current(db)
    except Exception as exc:  # noqa: BLE001
        print(f"(retrain rules rebuild skipped: {exc})", file=sys.stderr)


def _require_opponent(raw: str) -> str:
    oid = resolve_opponent(raw)
    if not oid:
        print(
            f"Unknown opponent: {raw!r}\n\n{format_opponent_list()}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return oid


def _madden_handler(args: argparse.Namespace, name: str):
    """Return the Madden 27 handler when --game madden27, else None (CFB path)."""
    from cfb_coach.games import is_madden

    madden = is_madden(getattr(args, "game", None))
    if madden and getattr(args, "dynasty", None):
        raise SystemExit("--dynasty is CFB-only; Madden 27 uses --franchise primary|lab")
    if not madden and getattr(args, "franchise", None):
        raise SystemExit("--franchise is Madden-only; add --game madden27")
    if not madden and (getattr(args, "o_book", None) or getattr(args, "d_book", None)):
        raise SystemExit("--o-book / --d-book are Madden-only; add --game madden27")
    if not madden:
        return None
    from cfb_coach.madden import cli as madden_cli

    return getattr(madden_cli, name)


def cmd_opponents(args: argparse.Namespace) -> int:
    handler = _madden_handler(args, "cmd_opponents")
    if handler:
        return handler(args)
    print(format_opponent_list())
    return 0


def cmd_playbook(args: argparse.Namespace) -> int:
    from cfb_coach.games import is_madden

    if not is_madden(args.game):
        raise SystemExit("playbook currently applies to --game madden27 (playbook of record)")
    from cfb_coach.madden import cli as madden_cli

    return madden_cli.cmd_playbook(args)


def _book_live_info(db, dynasty):
    from cfb_coach.cfb_playbook import live_book_info

    return live_book_info(db, dynasty)


def _book_live_apply(db, dynasty, rev):
    from cfb_coach.cfb_playbook import apply_pending

    return apply_pending(db, dynasty, rev)


def _book_status_line(db, dynasty) -> str:
    try:
        from cfb_coach.cfb_playbook import live_status_line

        return live_status_line(db, dynasty)
    except Exception as exc:  # noqa: BLE001
        return f"Playbook: unavailable ({type(exc).__name__})"


def cmd_book(args: argparse.Namespace) -> int:
    """CFB 27 custom playbook of record: show | apply | history | rollback | diff."""
    from cfb_coach import cfb_playbook as cp
    from cfb_coach.dynasty import normalize_dynasty

    db = _db()
    try:
        dyn = normalize_dynasty(args.dynasty or db.get_meta("dynasty_mode") or DEFAULT_DYNASTY)
        action = args.action
        if action == "apply":
            pend = cp.pending_rev(db, dyn)
            if not pend:
                print(f"No pending playbook edits for {dyn} — nothing to apply.")
                return 0
            if args.rev is not None and int(args.rev) != int(pend["rev"]):
                print(f"Pending revision is rev {pend['rev']}, not rev {args.rev} (a newer prep replaced it). "
                      f"Re-check the prep page, then run: book apply --dynasty {dyn} --rev {pend['rev']}")
                return 1
            rec = cp.apply_pending(db, dyn)
            print(f"Applied rev {rec['rev']} for {dyn}: {len(rec['edits'])} edit(s). Live calls now use this book.")
            print(cp.format_book_text(rec["book"].get("formations") or {}, audibles=rec["book"].get("audibles"),
                                      name=rec["book"].get("name", ""), rev=rec["rev"]))
            return 0
        if action == "history":
            print(cp.format_history(db, dyn))
            return 0
        if action == "rollback":
            if args.to is None:
                raise SystemExit("book rollback needs --to REV (see `book history`)")
            rec = cp.rollback(db, dyn, int(args.to))
            print(f"Rolled back {dyn} to rev {args.to} as new rev {rec['rev']} (current). Undo these in-game:")
            print(cp.format_edit_list(rec["edits"]) or "  (no in-game changes)")
            return 0
        cur = cp.current_rev(db, dyn)
        pend = cp.pending_rev(db, dyn)
        if action == "diff":
            if not pend:
                print(f"No pending edits for {dyn}.")
                return 0
            print(f"Pending rev {pend['rev']} ({pend['created_ts'][:16]} UTC) — {pend.get('summary') or ''}")
            print(cp.format_edit_list(pend["edits"], first_build=cur is None, book_name=pend["book"].get("name", "")))
            return 0
        if not cur and not pend:
            print(f"No custom playbook yet for {dyn} — run prep first.")
            return 0
        for label, rec in (("APPLIED (live calls use this)", cur), ("PENDING (make these edits, then `book apply`)", pend)):
            if not rec:
                continue
            print(f"== {label}: rev {rec['rev']} ==")
            print(cp.format_book_text(rec["book"].get("formations") or {}, audibles=rec["book"].get("audibles"),
                                      name=rec["book"].get("name", ""), rev=rec["rev"]))
            for f, fl in (rec["book"].get("formation_flags") or {}).items():
                print(f"  [{fl.get('flag')}] {f}: {fl.get('why', '')[:160]}")
            if rec is pend:
                print(cp.format_edit_list(pend["edits"], first_build=cur is None, book_name=pend["book"].get("name", "")))
        return 0
    finally:
        db.close()


def cmd_config(args: argparse.Namespace) -> int:
    from cfb_coach.games import is_madden

    if not is_madden(args.game):
        raise SystemExit("config currently applies to --game madden27 (primary/lab Franchise team)")
    from cfb_coach.madden import cli as madden_cli

    return madden_cli.cmd_config(args)


def cmd_macro_settings(args: argparse.Namespace) -> int:
    from cfb_coach.games import is_madden

    if not is_madden(args.game):  # v1.17: same shared store as Madden (settings are per macro name)
        from cfb_coach.macro_settings import cfb_cli

        return cfb_cli(args)
    from cfb_coach.madden import cli as madden_cli

    return madden_cli.cmd_macro_settings(args)


def cmd_prep(args: argparse.Namespace) -> int:
    if getattr(args, "live_scout", False):
        os.environ["CFB_COACH_LIVE_SCOUT"] = "1"
    handler = _madden_handler(args, "cmd_prep")
    if handler:
        return handler(args)
    from cfb_coach.install_sheet import format_delta_text, mark_prep_applied
    from cfb_coach.prep import load_opponent_profile
    from cfb_coach.prep_browser import generate_and_open

    oid = _require_opponent(args.opponent)
    db = _db()
    try:
        _ensure_rules(db)
        from cfb_coach.dynasty import resolve_dynasty

        # explicit --dynasty → the latest prep for this opponent → configured default
        dyn_resolved, dyn_src = resolve_dynasty(db, oid, getattr(args, "dynasty", None))
        dynasty = set_session_dynasty(db, dyn_resolved)
        dcfg = dynasty_config(dynasty)
        print(f"Prep dynasty: {dcfg['label']} ({dynasty}) — {dyn_src}")
        opp = load_opponent_profile(oid, db)
        if getattr(args, "mark_applied", False):
            # "I made the edits the last prep showed": confirm the pending custom book first
            from cfb_coach.cfb_playbook import apply_pending

            rec = apply_pending(db, dynasty)
            if rec:
                print(f"Custom playbook rev {rec['rev']} marked applied for {dynasty} ({len(rec['edits'])} edit(s)).")
        if getattr(args, "text", False):
            from cfb_coach.install_sheet import build_prep_plan

            plan = build_prep_plan(
                oid,
                opp,
                db=db,
                persist=True,
                dynasty=dynasty,
                offline=getattr(args, "offline", False),
                refresh_meta=getattr(args, "refresh_meta", False),
            )
            if getattr(args, "mark_applied", False):
                mark_prep_applied(db, oid, plan["proposed_deltas"])
                print(f"Marked {len(plan['proposed_deltas'])} deltas applied for {oid}.")
            exp = " [experimental]" if dcfg.get("experimental_badge") else ""
            print(
                f"Dynasty mode: {dcfg['label']} ({dcfg['mode']}){exp}"
            )
            if getattr(args, "details", False):
                print(doctrine_line())
                print(format_delta_text(plan))
            else:
                from cfb_coach.install_sheet import format_prep_minimal_text

                print(format_prep_minimal_text(plan))
            return 0

        path, plan = generate_and_open(
            oid,
            opp,
            db=db,
            persist=True,
            open_browser=not getattr(args, "no_open", False),
            mark_applied=getattr(args, "mark_applied", False),
            dynasty=dynasty,
            offline=getattr(args, "offline", False),
            refresh_meta=getattr(args, "refresh_meta", False),
        )
        n = len(plan.get("shown_deltas") or [])
        print(f"Prep vs {plan.get('display_name', oid)} → {path}")
        details = plan.get("details_path") or ""
        if details:
            print(f"Details (reasons, research, sources, history) → {details}")
            if getattr(args, "details", False) and not getattr(args, "no_open", False):
                from pathlib import Path as _P

                from cfb_coach.prep_browser import open_prep_html

                open_prep_html(_P(details), open_browser=True)
        exp = " [experimental]" if dcfg.get("experimental_badge") else ""
        print(
            f"Dynasty mode: {dcfg['label']} ({dcfg['mode']}){exp} "
            f"— stored for play/postgame"
        )
        print(doctrine_line())
        if getattr(args, "mark_applied", False):
            print(f"Marked proposed deltas applied for {oid}.")
        elif n == 0:
            print("No playbook changes — run baseline as-is (tips in browser).")
        else:
            print(f"{n} adjustment(s) shown (deltas only).")
        scout = plan.get("meta_scout") or {}
        n_ok = sum(1 for s_ in (scout.get("sources") or []) if s_.get("fetched"))
        n_src = len(scout.get("sources") or [])
        print(
            f"Meta: {scout.get('mode') or '?'} (fetched {_local_ts(scout.get('fetched_at') or '')}; "
            f"{n_ok}/{n_src} sources ok; {len(scout.get('changes_since_last') or [])} change note(s)) — "
            f"{scout.get('message') or ''}"
        )
        za = plan.get("zone_alignment") or {}
        if za.get("conflicts"):
            print(f"Meta vs your data: {len(za['conflicts'])} conflict(s) flagged on the prep page.")
        bk = plan.get("cfb_book") or {}
        if bk.get("error"):
            print(f"Custom playbook: unavailable ({bk['error']})")
        elif bk:
            if bk.get("seeded_now"):
                print(f"Custom playbook: {bk.get('seed_summary')}.")
            forms = ", ".join(f"{r['formation']}" + ("" if r["status"] == "applied" else f" [{r['status'].upper()}]")
                              for r in bk.get("formation_list") or [])
            print(f"Formations: {forms}")
            if bk.get("pending"):
                print(f"Custom playbook: {len(bk.get('edits') or [])} formation change(s) pending (rev {bk['pending'].get('rev')}) — "
                      f"make them in CFB 27, then: PYTHONPATH=. python3 -m cfb_coach book apply --dynasty {dynasty}")
            else:
                print(f"Custom playbook: no changes — rev {(bk.get('current') or {}).get('rev')} stands.")
        yt = scout.get("youtube") or {}
        if yt:
            print(f"YouTube: {yt.get('found', 0)} CFB 27 videos, {yt.get('transcripts', 0)} transcript(s) used"
                  + (f", {len(yt.get('blocked') or [])} blocked" if yt.get("blocked") else ""))
        if scout.get("mode") != "live":
            print(f"!! LIVE RESEARCH DID NOT RUN — {scout.get('message') or ''}")
        print(f"Text summary → {path.with_suffix('.txt')}")
    finally:
        db.close()
    return 0


def cmd_postgame(args: argparse.Namespace) -> int:
    handler = _madden_handler(args, "cmd_postgame")
    if handler:
        return handler(args)
    from cfb_coach.gameplan import postgame_summary

    oid = _require_opponent(args.opponent)
    db = _db()
    try:
        from cfb_coach.dynasty import resolve_dynasty

        dynasty = set_session_dynasty(db, resolve_dynasty(db, oid, getattr(args, "dynasty", None))[0])
        dcfg = dynasty_config(dynasty)
        exp = " [experimental]" if dcfg.get("experimental_badge") else ""
        print(f"Dynasty: {dcfg['label']} ({dcfg['mode']}){exp}")
        print(postgame_summary(db, oid, dynasty=dynasty))
        if getattr(args, "report", False):
            # Concise live-tendency summary from play_records if present
            from cfb_coach.live_tendency import LiveTendencyEngine
            from cfb_coach.vision.play_record import PlayRecord

            eng = LiveTendencyEngine()
            # Prefer latest session for opponent
            rows = list(
                db.conn.execute(
                    "SELECT session_id FROM game_sessions WHERE opponent_id = ? "
                    "ORDER BY started_ts DESC LIMIT 1",
                    (oid,),
                )
            )
            if rows:
                sid = rows[0]["session_id"]
                for pr in db.get_session_plays(sid):
                    try:
                        payload = __import__("json").loads(pr["payload_json"])
                        eng.add_play(PlayRecord.from_dict(payload))
                    except Exception:
                        pass
                print()
                print(eng.summary_report())
            else:
                print()
                print("# LIVE TENDENCY REPORT\n  (no game_sessions / play_records yet)")
    finally:
        db.close()
    return 0


def cmd_call(args: argparse.Namespace) -> int:
    handler = _madden_handler(args, "cmd_call")
    if handler:
        return handler(args)
    oid = _require_opponent(args.opponent)
    db = _db()
    try:
        sit = parse_situation(args.situation, default_side=args.side or "offense")
        if args.side:
            sit.side = args.side
        from cfb_coach.game_score import absorb_and_stamp, context_from_args

        absorb_and_stamp(sit, context_from_args(args))
        from cfb_coach.dynasty import resolve_dynasty

        call = make_call(sit, oid, db, dynasty=resolve_dynasty(db, oid, getattr(args, "dynasty", None))[0])
        print(call.format())
        if args.why:
            print(f"  ({call.rationale})")
    finally:
        db.close()
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    handler = _madden_handler(args, "cmd_play")
    if handler:
        return handler(args)
    oid = _require_opponent(args.opponent)
    db = _db()
    _ensure_rules(db)
    from cfb_coach.dynasty import resolve_dynasty

    # v1.15.1: play the dynasty the latest prep for this opponent used (explicit --dynasty
    # overrides; configured default only when there is no prep). Everything below — book,
    # Active 8 macros, live window, book apply button, retrain — uses this one value.
    dyn_resolved, dyn_src = resolve_dynasty(db, oid, getattr(args, "dynasty", None))
    dynasty = set_session_dynasty(db, dyn_resolved)
    dcfg = dynasty_config(dynasty)
    from cfb_coach.opponents import is_cpu_opponent
    from cfb_coach.copilot import default_overlay_path, write_overlay_html
    from cfb_coach.prep_browser import open_prep_html

    cpu_only = is_cpu_opponent(oid)
    print(f"LIVE PLAY — vs {oid}  (db: {db.path})")
    exp = " [experimental]" if dcfg.get("experimental_badge") else ""
    print(f"Dynasty: {dcfg['label']} ({dcfg['mode']}){exp} — {dyn_src}")
    print(_book_status_line(db, dynasty))
    print(doctrine_line())
    if cpu_only:
        print("CPU opponent — OFFENSE-ONLY coaching (no defense calls / no D macros).")
        print("Shorthand: 1&10 | 2&7 | 3&8 | rz 3&2 | my 35 | opp 40")
        print("Commands: result <text> | why | book | book apply | quit  (side d disabled)")
    else:
        print("Side defaults to offense. Prefix with 'd ' for defense.")
        print("Shorthand: 1&10 | 2&7 | 3&8 d | rz 3&2 | d 1&10 | my 35 | opp 40")
        print("Commands: side o|d | result <text> | why | book | book apply | quit")
    print("Doctrine: one tell = log/mild bump; hard-counter only on REPEATED tendency.")
    print("  Aidan UX: type D&D (+ yl) + previous play/coverage name — no need to say 'last'.")
    print("  Examples: '1&10 my 35 mesh spot' | '2&7 deep flood' | '1&10 cover 2'")
    print("  Live look only with: showing / live / pre-snap / aligned (e.g. 'showing cover 2')")
    print("  Score (optional, us-them): --score 21-14, or `score 21-14` / `score clear` mid-game.")
    print("  Quarter: --quarter 4, `quarter 4`, or `q4` on the sit line. Close early games stay neutral.")

    from cfb_coach.game_score import absorb_and_stamp, context_from_args, interpret_live_command, sit_prompt, snap_notes_for

    live_ctx = context_from_args(args)
    if live_ctx.describe():
        print(f"  Game situation: {live_ctx.describe()} (sticks until you update it)")

    # Overlay: default ON for interactive play; --no-overlay disables; --once skips browser
    no_overlay = bool(getattr(args, "no_overlay", False))
    overlay_arg = getattr(args, "overlay", None)
    if no_overlay:
        overlay_path = None
    elif overlay_arg:
        from pathlib import Path as _P

        overlay_path = _P(overlay_arg)
    else:
        overlay_path = default_overlay_path()  # ~/.cfb-coach/copilot_overlay.html

    if getattr(args, "once", None):
        # Non-interactive: still refresh overlay file if enabled, but do not auto-open
        default_side = "offense"
        sit = parse_situation(args.once, default_side=default_side)
        absorb_and_stamp(sit, live_ctx)
        heard = format_heard(sit)
        print(heard)
        call = make_call(sit, oid, db, dynasty=dynasty)
        formatted = call.format()
        print(formatted)
        if args.why:
            print(f"  ({call.rationale})")
        if overlay_path is not None:
            write_overlay_html(
                str(overlay_path),
                None,
                [],
                call_text=formatted,
                short_line=heard,
                mode="play",
            )
            print(f"  overlay → {overlay_path}")
        db.close()
        return 0

    use_html = not bool(getattr(args, "terminal", False) or getattr(args, "no_html", False))
    if use_html:
        from cfb_coach.live_server import LivePlayController, run_live_server

        def _make(sit, **kwargs):
            return make_call(sit, oid, db, dynasty=dynasty, **kwargs)

        def _learn():
            from cfb_coach.gameplan import postgame_summary

            return postgame_summary(db, oid, dynasty=dynasty)

        ctrl = LivePlayController(
            db=db,
            opponent_id=oid,
            make_call=_make,
            parse_situation=parse_situation,
            learn_summary=_learn,
            brand="CFB Coach",
            play_cmd="cfb-coach play",
            dynasty=dynasty,
            dynasty_label=f"{dcfg['label']} ({dcfg['mode']})",
            dynasty_source=dyn_src,
            cpu_only=cpu_only,
            book_info=lambda: _book_live_info(db, dynasty),
            book_apply=lambda rev: _book_live_apply(db, dynasty, rev),
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

    if overlay_path is not None:
        write_overlay_html(
            str(overlay_path),
            None,
            [],
            call_text="waiting for sit> …",
            short_line=f"vs {oid}",
            mode="play",
        )
        open_prep_html(overlay_path, open_browser=True)
        print(f"Overlay ON → {overlay_path}  (auto-opens browser; --no-overlay to disable)")
    else:
        print("Overlay OFF (--no-overlay)")
    print("-" * 60)

    default_side = "offense"
    last_call = None
    last_sit = None
    # Previous-snap observations — mild context only, never auto hard-counter
    last_coverage: str | None = None
    last_concept: str | None = None

    def _refresh_overlay(call_obj, sit_obj, heard: str = "") -> None:
        if overlay_path is None:
            return
        write_overlay_html(
            str(overlay_path),
            None,
            [],
            call_text=call_obj.format(),
            short_line=heard or (sit_obj.label if sit_obj else ""),
            mode="play",
        )

    try:
        while True:
            try:
                raw = input(sit_prompt(default_side, live_ctx)).strip()
            except EOFError:
                print()
                break
            if not raw:
                continue
            low = raw.lower()
            if low in ("q", "quit", "exit"):
                break
            if low in ("o", "side o", "offense"):
                default_side = "offense"
                print("  side → offense")
                continue
            if low in ("d", "side d", "defense"):
                if cpu_only:
                    print("  CPU = offense-only — defense calls disabled")
                    default_side = "offense"
                else:
                    default_side = "defense"
                    print("  side → defense")
                continue
            if low in ("book", "book show"):
                print("  " + _book_status_line(db, dynasty))
                continue
            if low == "book apply":
                rec = _book_live_apply(db, dynasty, None)
                print(f"  Applied playbook rev {rec['rev']} — new plays are callable now." if rec
                      else "  No pending playbook edits.")
                continue
            if low == "why" and last_call:
                print(f"  ({last_call.rationale})")
                continue
            if low.startswith("result ") or low.startswith("log "):
                if not last_call or not last_sit:
                    print("  No call to log yet.")
                    continue
                result = raw.split(" ", 1)[1].strip()
                # Optional inline coverage/concept in result: "result +4 cov c2 invert"
                res_sit = parse_situation(result, default_side=last_sit.side)
                cov_seen = res_sit.coverage_hint or last_sit.coverage_hint
                concept_seen = res_sit.concept_hint or last_sit.concept_hint

                db.log_snap(
                    opponent_id=oid,
                    side=last_call.side,
                    situation_raw=last_sit.raw,
                    our_call=last_call.format().split("\n")[0],
                    formation=last_call.formation,
                    play=last_call.play,
                    macro=last_call.adj_or_macro if last_call.side == "defense" else getattr(last_call, "macro", None),
                    down=last_sit.down,
                    distance=last_sit.distance,
                    yardline=last_sit.yardline,
                    quarter=(getattr(last_sit, "extras", None) or {}).get("quarter"),
                    notes=snap_notes_for(last_sit),
                    result=result,
                    coverage_seen=cov_seen,
                    concept_seen=concept_seen,
                )
                success = any(
                    w in result.lower()
                    for w in ("td", "+", "good", "convert", "stop", "sack", "int")
                )
                # ONE tell = mild bump only (symmetric O/D)
                if concept_seen and last_call.side == "defense":
                    mild_bump_concept(
                        db, oid, concept_seen, last_sit, success=success
                    )
                    last_concept = concept_seen
                    print(
                        f"  logged: {result} | mild bump concept={concept_seen} "
                        "(no hard-counter next snap)"
                    )
                elif cov_seen and last_call.side == "offense":
                    mild_bump_coverage(db, oid, cov_seen, last_sit)
                    last_coverage = cov_seen
                    print(
                        f"  logged: {result} | mild bump coverage={cov_seen} "
                        "(no hard-counter next snap)"
                    )
                else:
                    print(f"  logged: {result}")
                from cfb_coach.gameplan import format_pivot_hints

                tip = format_pivot_hints(db, oid)
                if tip:
                    for line in tip.splitlines()[1:]:
                        if line.strip():
                            print(f"  {line.strip()}")
                continue
            score_msg = interpret_live_command(raw, live_ctx)
            if score_msg is not None:
                print(score_msg)
                continue

            sit = parse_situation(raw, default_side=default_side)
            absorb_and_stamp(sit, live_ctx)
            heard = format_heard(sit)
            print(heard)
            # Pass previous-snap signals as last-only context (not hard-counters)
            call = make_call(
                sit,
                oid,
                db,
                dynasty=dynasty,
                last_coverage=last_coverage if sit.side == "offense" else None,
                last_concept=last_concept if sit.side == "defense" else None,
            )
            print(call.format())
            _refresh_overlay(call, sit, heard)
            last_call, last_sit = call, sit
            # If this sit itself named a live/last coverage or concept, remember for NEXT snap
            if sit.coverage_hint and sit.side == "offense":
                last_coverage = sit.coverage_hint
            if sit.concept_hint and sit.side == "defense":
                last_concept = sit.concept_hint
    finally:
        db.close()
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Recompute ALL learned weights from every logged snap + game result (v2 rules)."""
    from cfb_coach.games import is_madden

    if is_madden(getattr(args, "game", None)):
        raise SystemExit("rebuild applies to CFB 27 (--game cfb27); Madden 27 keeps its own learning")
    from cfb_coach.learning import (
        RULES_VERSION,
        LearnedWeights,
        backup_db_file,
        rebuild_all,
        weight_diff,
        zone_leaderboard,
    )
    from cfb_coach.learning import _fmt_key  # noqa: PLC2701

    db = _db()
    try:
        oid = _require_opponent(args.opponent) if getattr(args, "opponent", None) else "cpu"
        n_snaps = int(db.conn.execute("SELECT COUNT(*) FROM snaps").fetchone()[0])
        bpath = None
        if not getattr(args, "no_backup", False) and n_snaps:
            bpath = backup_db_file(db)
            if bpath:
                db.set_meta("learn_rules_backup_path", str(bpath))
        res = rebuild_all(db)
        n_games = len([r for r in res["results"].values() if r.get("result_wl")])
        print(
            f"Rebuilt learned weights under rules {RULES_VERSION} from {n_snaps} snaps / {n_games} game results"
            + (f" (backup: {bpath})" if bpath else " (no backup)")
        )
        print("Snaps and game history untouched.")
        diff = weight_diff(res["before"], res["after"], opponent_id=oid)
        top = int(getattr(args, "top", 15) or 15)
        print(f"\n## Biggest changes vs {oid} (before → after)")
        for k, b, a, _d in diff[:top]:
            print(f"  {_fmt_key(k)}: {b:+.2f} → {a:+.2f}")
        lw = LearnedWeights.load(db, oid)
        for zone, label in (("general", "general"), ("rz", "red zone"), ("gl", "goal line / goal-to-go")):
            lb = zone_leaderboard(lw, zone, top=5)
            print(f"\n## {label}: best")
            for name, v, n in lb["best"]:
                print(f"  {v:+.2f}  {name}  (n={n})")
            print(f"## {label}: worst")
            for name, v, n in lb["worst"]:
                print(f"  {v:+.2f}  {name}  (n={n})")
    finally:
        db.close()
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """List / accept ohio_state → Alabama promotions."""
    handler = _madden_handler(args, "cmd_promote")
    if handler:
        return handler(args)
    from cfb_coach.dynasty import format_promotions, promote

    db = _db()
    try:
        if getattr(args, "accept_all", False) or getattr(args, "target", None):
            result = promote(
                db,
                target=getattr(args, "target", None),
                kind=getattr(args, "kind", None),
                accept_all=bool(getattr(args, "accept_all", False)),
            )
            accepted = result.get("accepted") or []
            if not accepted:
                print("No matching pending promotions to accept.")
            else:
                print(f"Accepted {len(accepted)} promotion(s) for Alabama:")
                for item in accepted:
                    tgt = item.get("target") or item.get("macro") or item.get("key")
                    print(f"  [{item.get('kind')}] {tgt} — {item.get('note', '')}")
            print()
        print(format_promotions(db=db))
    finally:
        db.close()
    return 0



def cmd_watch(args: argparse.Namespace) -> int:
    """Screen co-pilot: DefenseLook → tips (demo/hotkeys/image/live vision)."""
    from cfb_coach.watch import run_watch

    return run_watch(args)


def _add_game_args(p: argparse.ArgumentParser, *, franchise: bool = True) -> None:
    from cfb_coach.games import GAME_CHOICES

    p.add_argument(
        "--game",
        choices=GAME_CHOICES,
        default="cfb27",
        help="cfb27 (default) | madden27 (alias madden) — Madden 27 Franchise typed coach",
    )
    if franchise:
        p.add_argument(
            "--franchise",
            choices=("primary", "lab"),
            default=None,
            help="Madden only: primary (serious Franchise, team TBD) | lab (experimental; promotes to primary)",
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cfb_coach",
        description="Xbox CFB 27 dynasty + Madden 27 Franchise play-caller (heuristics + seed + log learning)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser(
        "prep",
        help="Pregame: browser deltas + Active loadout (<=8). CPU=offense-only. No benched FLOOD/SCREEN dump.",
    )
    p_prep.add_argument("--opponent", "-o", required=True)
    p_prep.add_argument(
        "--text",
        action="store_true",
        help="Print compact terminal delta dump instead of opening browser",
    )
    p_prep.add_argument(
        "--no-open",
        action="store_true",
        help="Write HTML but do not open a browser",
    )
    p_prep.add_argument(
        "--details",
        action="store_true",
        help="CFB: also open the details page (book reasons/flags/history, research, sources, zone plan, constants); "
             "with --text print the full details dump instead of the minimal view",
    )
    p_prep.add_argument(
        "--dynasty",
        choices=("alabama", "ohio_state"),
        default=None,
        help="Dynasty: alabama=serious USER (default) | ohio_state=experimental lab (promotes to Alabama on success).",
    )
    p_prep.add_argument(
        "--mark-applied",
        action="store_true",
        help="Mark current proposed deltas as applied AND confirm pending custom-playbook edits (same as `book apply`)",
    )
    p_prep.add_argument(
        "--offline",
        action="store_true",
        help="No network: use the last downloaded daily research (else the copy in this checkout / seed research)",
    )
    p_prep.add_argument(
        "--refresh-meta",
        action="store_true",
        help="(Kept for compatibility) prep always loads the newest daily AI research",
    )
    p_prep.add_argument(
        "--live-scout",
        action="store_true",
        help="Scrape the web during prep (old keyword scout) instead of using the daily AI research",
    )
    _add_game_args(p_prep)
    p_prep.add_argument(
        "--o-book",
        dest="o_book",
        default=None,
        metavar="auto|custom|stock:NAME",
        help="Madden only: offensive playbook of record (default auto; e.g. stock:Buccaneers, stock:'Shotgun Classic', custom)",
    )
    p_prep.add_argument(
        "--d-book",
        dest="d_book",
        default=None,
        metavar="auto|custom|stock:NAME",
        help="Madden only: defensive playbook of record (default auto → stock:49ers)",
    )
    p_prep.set_defaults(func=cmd_prep)

    p_play = sub.add_parser(
        "play",
        help="Live play: HTML input window (default) or --terminal sit> loop (CPU = offense-only)",
    )
    p_play.add_argument("--opponent", "-o", required=True)
    p_play.add_argument(
        "--once",
        help="Non-interactive: one situation string, print one call, exit",
    )
    p_play.add_argument("--why", action="store_true", help="Show rationale")
    p_play.add_argument(
        "--score",
        default=None,
        metavar="US-THEM",
        help="Current score, us-them (e.g. 21-14). Optional; persists for the session. "
             "Update mid-game with `score 17-21` or `score clear`.",
    )
    p_play.add_argument(
        "--quarter",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
        help="Current quarter (5 = OT). Optional. Also `quarter 4` or `q4` on the sit line.",
    )
    p_play.add_argument(
        "--dynasty",
        choices=("alabama", "ohio_state"),
        default=None,
        help="Override/store dynasty: alabama (serious) | ohio_state (lab)",
    )
    p_play.add_argument(
        "--overlay",
        metavar="PATH",
        default=None,
        help="HTML overlay path (default ON: ~/.cfb-coach/copilot_overlay.html)",
    )
    p_play.add_argument(
        "--no-overlay",
        dest="no_overlay",
        action="store_true",
        help="Disable HTML overlay (terminal mode only; interactive play defaults overlay ON)",
    )
    p_play.add_argument(
        "--terminal",
        "--no-html",
        dest="terminal",
        action="store_true",
        help="Classic terminal sit> loop instead of the HTML live window (default: HTML ON)",
    )
    p_play.add_argument(
        "--html-port",
        type=int,
        default=None,
        metavar="PORT",
        help="Localhost port for HTML live play (default 8765 or next free)",
    )
    _add_game_args(p_play)
    p_play.set_defaults(func=cmd_play)

    p_post = sub.add_parser(
        "postgame",
        help="Learn from snaps; ohio_state strong results -> Alabama promotion notes",
    )
    p_post.add_argument("--opponent", "-o", required=True)
    p_post.add_argument(
        "--dynasty",
        choices=("alabama", "ohio_state"),
        default=None,
        help="Override/store dynasty: alabama (serious) | ohio_state (lab)",
    )
    p_post.add_argument(
        "--report",
        action="store_true",
        help="Also print concise this-game live tendency summary from play_records",
    )
    _add_game_args(p_post)
    p_post.set_defaults(func=cmd_postgame)

    p_reb = sub.add_parser(
        "rebuild",
        help="Recompute ALL learned weights from every logged snap + W/L under the current retrain rules (backs up the DB first)",
    )
    p_reb.add_argument("--opponent", "-o", default=None, help="Which opponent's changes to print (default cpu)")
    p_reb.add_argument("--no-backup", action="store_true", help="Skip the timestamped DB backup")
    p_reb.add_argument("--top", type=int, default=15, help="How many before/after changes to print")
    p_reb.add_argument("--game", default="cfb27", help=argparse.SUPPRESS)
    p_reb.set_defaults(func=cmd_rebuild)

    p_ops = sub.add_parser("opponents", help="List opponents + aliases")
    _add_game_args(p_ops, franchise=False)
    p_ops.set_defaults(func=cmd_opponents)

    p_call = sub.add_parser("call", help="One-shot call (non-interactive)")
    p_call.add_argument("--opponent", "-o", required=True)
    p_call.add_argument("--situation", "-s", required=True)
    p_call.add_argument("--side", choices=("offense", "defense"), default=None)
    p_call.add_argument("--why", action="store_true")
    p_call.add_argument(
        "--score",
        default=None,
        metavar="US-THEM",
        help="Current score, us-them (e.g. 21-14). Optional. Same flag for CFB and --game madden27.",
    )
    p_call.add_argument(
        "--quarter",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
        help="Current quarter (5 = OT). Optional; score stays neutral early and close without it.",
    )
    _add_game_args(p_call, franchise=False)
    p_call.set_defaults(func=cmd_call)

    p_prom = sub.add_parser(
        "promote",
        help="List/accept ohio_state lab -> Alabama promotions (macro loadout / gameplan overlay)",
    )
    p_prom.add_argument(
        "--accept-all",
        action="store_true",
        help="Accept all pending Alabama promotions",
    )
    p_prom.add_argument(
        "--target",
        help="Accept one promotion by macro/overlay target name",
    )
    p_prom.add_argument(
        "--kind",
        choices=("macro_loadout", "gameplan_overlay"),
        default=None,
        help="Optional kind filter with --target",
    )
    _add_game_args(p_prom, franchise=False)
    p_prom.set_defaults(func=cmd_promote)

    p_cfg = sub.add_parser(
        "config",
        help="Madden 27 Franchise config: primary/lab team (TBD until set)",
    )
    _add_game_args(p_cfg, franchise=False)
    p_cfg.set_defaults(game="madden27")
    p_cfg.add_argument("--primary-team", default=None, help='Primary Franchise NFL team, e.g. "Buccaneers" or TB')
    p_cfg.add_argument("--lab-team", default=None, help="Optional lab Franchise team")
    p_cfg.add_argument("--clear-primary", action="store_true", help="Reset primary team to TBD")
    p_cfg.add_argument("--clear-lab", action="store_true", help="Reset lab team to TBD")
    p_cfg.add_argument("--o-book", dest="o_book", default=None, metavar="auto|custom|stock:NAME",
                       help="Default offense book for every prep (auto = research picks; start stock:Buccaneers)")
    p_cfg.add_argument("--d-book", dest="d_book", default=None, metavar="auto|custom|stock:NAME",
                       help="Default defense book for every prep (auto = research picks; start stock:49ers)")
    p_cfg.set_defaults(func=cmd_config)

    p_ms = sub.add_parser(
        "macro-settings",
        help="Enter / show YOUR exact Custom Adjustment settings — one store shared by CFB 27 + Madden 27 "
             "(used verbatim; macros with none are flagged)",
    )
    _add_game_args(p_ms, franchise=False)
    p_ms.set_defaults(game="madden27")
    p_ms.add_argument("macro", nargs="?", default=None, help="Macro id, e.g. MATCH-4, O-PROT (omit to list all)")
    p_ms.add_argument("--set", dest="settings", action="append", default=[], metavar='"Section: Setting = value"',
                      help='One exact setting as it reads in game; repeat --set for each')
    p_ms.add_argument("--replace", action="store_true", help="Replace all stored settings for this macro")
    p_ms.add_argument("--xbox-name", default=None, help="The name you saved it under in game (if different)")
    p_ms.add_argument("--clear", action="store_true", help="Delete your stored settings for this macro")
    p_ms.add_argument("--this-game-only", dest="this_game_only", action="store_true",
                      help="Store as an override for this game only (default: shared by CFB 27 + Madden 27)")
    p_ms.add_argument("--side", choices=("offense", "defense"), default=None,
                      help="CFB only: which side, when the name exists on both (e.g. HEAT)")
    p_ms.set_defaults(func=cmd_macro_settings)

    p_book = sub.add_parser(
        "playbook",
        help="Madden 27: show the full playbook of record locked by the latest prep",
    )
    _add_game_args(p_book, franchise=False)
    p_book.set_defaults(game="madden27")
    p_book.add_argument("--side", choices=("offense", "defense"), default=None)
    p_book.set_defaults(func=cmd_playbook)

    p_cbook = sub.add_parser(
        "book",
        help="CFB 27 custom playbook of record (managed each prep): show | apply | history | rollback --to N | diff",
    )
    p_cbook.add_argument("action", nargs="?", default="show", choices=("show", "apply", "history", "rollback", "diff"))
    p_cbook.add_argument("--dynasty", choices=("alabama", "ohio_state"), default=None,
                         help="Which dynasty's book (default: last prep/play dynasty)")
    p_cbook.add_argument("--rev", type=int, default=None, help="apply: only if this is still the pending revision")
    p_cbook.add_argument("--to", type=int, default=None, help="rollback: revision to restore")
    p_cbook.set_defaults(func=cmd_book)

    p_watch = sub.add_parser(
        "watch",
        aliases=("copilot",),
        help="Screen co-pilot: pre-snap tips from defense look / live Remote Play vision",
    )
    p_watch.add_argument(
        "--demo",
        action="store_true",
        help="Simulate DefenseLooks and print sample pre-snap tips (no Xbox/capture needed)",
    )
    p_watch.add_argument(
        "--once",
        action="store_true",
        help="With --demo/--image: emit one tip block and exit (non-interactive)",
    )
    p_watch.add_argument(
        "--image",
        metavar="PATH",
        help="PNG/JPG path: run naive ROI heuristic stub → tips",
    )
    p_watch.add_argument(
        "--overlay",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="HTML overlay path (default ON for --window/--screen-region: ~/.cfb-coach/copilot_overlay.html)",
    )
    p_watch.add_argument(
        "--no-overlay",
        dest="no_overlay",
        action="store_true",
        help="Disable HTML overlay (live modes default overlay ON)",
    )
    p_watch.add_argument(
        "--opponent",
        "-o",
        default=None,
        help="Optional opponent id for light prior lean (does not change prep/play)",
    )
    p_watch.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Seconds between --demo looks when non-interactive (default 1.5)",
    )
    p_watch.add_argument(
        "--setup",
        action="store_true",
        help="Print Xbox Remote Play / capture setup notes and exit",
    )
    p_watch.add_argument(
        "--window",
        metavar="TITLE",
        default=None,
        help='Capture window by title substring (e.g. "Xbox") — Windows dxcam/mss; case-insensitive',
    )
    p_watch.add_argument(
        "--list-windows",
        dest="list_windows",
        action="store_true",
        help="Print visible window titles (Windows) so you can pick --window",
    )
    p_watch.add_argument(
        "--screen-region",
        dest="screen_region",
        nargs="?",
        const="calib",
        default=None,
        metavar="L,T,W,H",
        help="Screen crop left,top,width,height via mss. "
        "With no args: use crop from ~/.cfb-coach/vision_calib.json",
    )
    p_watch.add_argument(
        "--calibrate",
        action="store_true",
        help="Interactive calibrate → ~/.cfb-coach/vision_calib.json",
    )
    p_watch.add_argument(
        "--video",
        metavar="PATH",
        default=None,
        help="Offline video file for vision pipeline (needs opencv)",
    )
    p_watch.add_argument(
        "--debug",
        action="store_true",
        help="OpenCV debug view (FPS, crop, ROIs, state) + pipeline on --image",
    )
    p_watch.add_argument(
        "--tts",
        action="store_true",
        help="Optional Windows SAPI TTS for tip lines",
    )
    p_watch.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="N",
        help="Future: OpenCV capture-card device index",
    )
    p_watch.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Target analyzed FPS for live/video pipeline (default 8)",
    )
    p_watch.add_argument(
        "--dynasty",
        choices=("alabama", "ohio_state"),
        default=None,
        help="Dynasty for watch session logging (alabama|ohio_state)",
    )
    p_watch.add_argument(
        "--record-plays",
        dest="record_plays",
        action="store_true",
        help="Optional: save short clips under ~/.cfb-coach/games/<id>/plays/ when low-conf/explosive",
    )
    p_watch.add_argument(
        "--no-post-play-line",
        dest="post_play_line",
        action="store_false",
        default=True,
        help="Disable post-play one-liner on play end",
    )
    p_watch.set_defaults(func=cmd_watch)


    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
