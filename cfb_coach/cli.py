"""CLI: prep / play / postgame / promote / opponents / call / watch."""

from __future__ import annotations

import argparse
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


def _require_opponent(raw: str) -> str:
    oid = resolve_opponent(raw)
    if not oid:
        print(
            f"Unknown opponent: {raw!r}\n\n{format_opponent_list()}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return oid


def cmd_opponents(_args: argparse.Namespace) -> int:
    print(format_opponent_list())
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    from cfb_coach.install_sheet import format_delta_text, mark_prep_applied
    from cfb_coach.prep import load_opponent_profile
    from cfb_coach.prep_browser import generate_and_open

    oid = _require_opponent(args.opponent)
    db = _db()
    try:
        dynasty = set_session_dynasty(
            db, getattr(args, "dynasty", None) or DEFAULT_DYNASTY
        )
        dcfg = dynasty_config(dynasty)
        opp = load_opponent_profile(oid, db)
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
            print(doctrine_line())
            print(format_delta_text(plan))
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
        if scout.get("available"):
            tag = "cached" if scout.get("from_cache") else "live"
            print(
                f"Meta scout ({tag}, conf={scout.get('confidence', '?')}): "
                f"{len(scout.get('patch_notes') or [])} patch note(s), "
                f"{len(scout.get('suggestions') or [])} book suggestion(s)."
            )
        else:
            print(
                scout.get("message")
                or "Scout unavailable — using cached/baseline cfb27-2026-09"
            )
    finally:
        db.close()
    return 0


def cmd_postgame(args: argparse.Namespace) -> int:
    from cfb_coach.gameplan import postgame_summary

    oid = _require_opponent(args.opponent)
    db = _db()
    try:
        dynasty = set_session_dynasty(
            db, getattr(args, "dynasty", None) or db.get_meta("dynasty_mode") or DEFAULT_DYNASTY
        )
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
    oid = _require_opponent(args.opponent)
    db = _db()
    try:
        sit = parse_situation(args.situation, default_side=args.side or "offense")
        if args.side:
            sit.side = args.side
        call = make_call(sit, oid, db)
        print(call.format())
        if args.why:
            print(f"  ({call.rationale})")
    finally:
        db.close()
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    oid = _require_opponent(args.opponent)
    db = _db()
    dynasty = set_session_dynasty(
        db, getattr(args, "dynasty", None) or db.get_meta("dynasty_mode") or DEFAULT_DYNASTY
    )
    dcfg = dynasty_config(dynasty)
    from cfb_coach.opponents import is_cpu_opponent
    from cfb_coach.copilot import default_overlay_path, write_overlay_html
    from cfb_coach.prep_browser import open_prep_html

    cpu_only = is_cpu_opponent(oid)
    print(f"LIVE PLAY — vs {oid}  (db: {db.path})")
    exp = " [experimental]" if dcfg.get("experimental_badge") else ""
    print(f"Dynasty: {dcfg['label']} ({dcfg['mode']}){exp}")
    print(doctrine_line())
    if cpu_only:
        print("CPU opponent — OFFENSE-ONLY coaching (no defense calls / no D macros).")
        print("Shorthand: 1&10 | 2&7 | 3&8 | rz 3&2 | my 35 | opp 40")
        print("Commands: result <text> | why | quit  (side d disabled)")
    else:
        print("Side defaults to offense. Prefix with 'd ' for defense.")
        print("Shorthand: 1&10 | 2&7 | 3&8 d | rz 3&2 | d 1&10 | my 35 | opp 40")
        print("Commands: side o|d | result <text> | why | quit")
    print("Doctrine: one tell = log/mild bump; hard-counter only on REPEATED tendency.")
    print("  Aidan UX: type D&D (+ yl) + previous play/coverage name — no need to say 'last'.")
    print("  Examples: '1&10 my 35 mesh spot' | '2&7 deep flood' | '1&10 cover 2'")
    print("  Live look only with: showing / live / pre-snap / aligned (e.g. 'showing cover 2')")

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
        heard = format_heard(sit)
        print(heard)
        call = make_call(sit, oid, db)
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
                raw = input(f"[{default_side[0].upper()}] sit> ").strip()
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
                    macro=last_call.adj_or_macro if last_call.side == "defense" else None,
                    down=last_sit.down,
                    distance=last_sit.distance,
                    yardline=last_sit.yardline,
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

            sit = parse_situation(raw, default_side=default_side)
            heard = format_heard(sit)
            print(heard)
            # Pass previous-snap signals as last-only context (not hard-counters)
            call = make_call(
                sit,
                oid,
                db,
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


def cmd_promote(args: argparse.Namespace) -> int:
    """List / accept ohio_state → Alabama promotions."""
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cfb_coach",
        description="Xbox CFB dynasty play-caller (heuristics + seed + log learning)",
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
        "--dynasty",
        choices=("alabama", "ohio_state"),
        default=None,
        help="Dynasty: alabama=serious USER (default) | ohio_state=experimental lab (promotes to Alabama on success).",
    )
    p_prep.add_argument(
        "--mark-applied",
        action="store_true",
        help="Mark current proposed deltas as applied (next prep shows only NEW)",
    )
    p_prep.add_argument(
        "--offline",
        action="store_true",
        help="Skip live meta scout network fetch (use cache/baseline cfb27-2026-09)",
    )
    p_prep.add_argument(
        "--refresh-meta",
        action="store_true",
        help="Force refetch meta scout (ignore <6h cache)",
    )
    p_prep.set_defaults(func=cmd_prep)

    p_play = sub.add_parser(
        "play",
        help="Interactive typed live call loop + browser overlay (CPU = offense-only)",
    )
    p_play.add_argument("--opponent", "-o", required=True)
    p_play.add_argument(
        "--once",
        help="Non-interactive: one situation string, print one call, exit",
    )
    p_play.add_argument("--why", action="store_true", help="Show rationale")
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
        help="Disable HTML overlay (interactive play defaults overlay ON)",
    )
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
    p_post.set_defaults(func=cmd_postgame)

    p_ops = sub.add_parser("opponents", help="List opponents + aliases")
    p_ops.set_defaults(func=cmd_opponents)

    p_call = sub.add_parser("call", help="One-shot call (non-interactive)")
    p_call.add_argument("--opponent", "-o", required=True)
    p_call.add_argument("--situation", "-s", required=True)
    p_call.add_argument("--side", choices=("offense", "defense"), default=None)
    p_call.add_argument("--why", action="store_true")
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
    p_prom.set_defaults(func=cmd_promote)

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
