"""CLI: prep / play / postgame / opponents / call."""

from __future__ import annotations

import argparse
import sys

from cfb_coach.db import CoachDB, resolve_db_path_from_env
from cfb_coach.opponents import format_opponent_list, resolve_opponent
from cfb_coach.playcaller import make_call
from cfb_coach.situation import parse_situation
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
                oid, opp, db=db, persist=True, dynasty=dynasty
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
    print(f"LIVE PLAY — vs {oid}  (db: {db.path})")
    exp = " [experimental]" if dcfg.get("experimental_badge") else ""
    print(f"Dynasty: {dcfg['label']} ({dcfg['mode']}){exp}")
    print(doctrine_line())
    print("Side defaults to offense. Prefix with 'd ' for defense.")
    print("Shorthand: 1&10 | 2&7 | 3&8 d | rz 3&2 | d 1&10")
    print("Doctrine: one tell = log/mild bump; hard-counter only on REPEATED tendency.")
    print("  last c2 invert / last cross wheels → does NOT auto-counter next snap")
    print("Commands: side o|d | result <text> | why | quit")
    print("-" * 60)

    default_side = "offense"
    last_call = None
    last_sit = None
    # Previous-snap observations — mild context only, never auto hard-counter
    last_coverage: str | None = None
    last_concept: str | None = None

    if getattr(args, "once", None):
        sit = parse_situation(args.once, default_side=default_side)
        call = make_call(sit, oid, db)
        print(call.format())
        if args.why:
            print(f"  ({call.rationale})")
        db.close()
        return 0

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
            # Pass previous-snap signals as last-only context (not hard-counters)
            call = make_call(
                sit,
                oid,
                db,
                last_coverage=last_coverage if sit.side == "offense" else None,
                last_concept=last_concept if sit.side == "defense" else None,
            )
            print(call.format())
            last_call, last_sit = call, sit
            # If this sit itself named a live/last coverage or concept, remember for NEXT snap
            if sit.coverage_hint and sit.side == "offense":
                last_coverage = sit.coverage_hint
            if sit.concept_hint and sit.side == "defense":
                last_concept = sit.concept_hint
    finally:
        db.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cfb_coach",
        description="Xbox CFB dynasty play-caller (heuristics + seed + log learning)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser(
        "prep",
        help="Pregame: open browser with playbook/macro deltas only (not full recreate)",
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
        help="Dynasty mode: alabama=serious (default) | ohio_state=experimental. Stored for play/postgame.",
    )
    p_prep.add_argument(
        "--mark-applied",
        action="store_true",
        help="Mark current proposed deltas as applied (next prep shows only NEW)",
    )
    p_prep.set_defaults(func=cmd_prep)

    p_play = sub.add_parser("play", help="Interactive live call loop")
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
        help="Override/store dynasty mode (default: session or alabama)",
    )
    p_play.set_defaults(func=cmd_play)

    p_post = sub.add_parser(
        "postgame",
        help="Learn from recent snaps — adjust gameplan/macro weights vs opponent",
    )
    p_post.add_argument("--opponent", "-o", required=True)
    p_post.add_argument(
        "--dynasty",
        choices=("alabama", "ohio_state"),
        default=None,
        help="Override/store dynasty mode (default: session or alabama)",
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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
