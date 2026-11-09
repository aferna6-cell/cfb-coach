"""Command line for the unified coordinator.

  python -m cfb_coach.madden.unified audit | game-prep | confirm-install | preflight | live | postgame | smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from cfb_coach.madden.unified import guard, knowledge as K, postgame as PG, pregame as PRE, preflight as PF
from cfb_coach.madden.unified.coordinator import UnifiedCoordinator, inventory_fp
from cfb_coach.madden.unified.store import UnifiedStore
from cfb_coach.madden.unified.types import (DEF, EXEC_CONFIRMED, EXEC_MODIFIED, EXEC_NOT_RUN, EXEC_UNKNOWN, OFF, GameState,
                                            OpponentAction, Outcome)
from cfb_coach.madden.unified.world_model import WorldModel


def _write(path: str | None, obj: Any) -> None:
    txt = json.dumps(obj, indent=2, default=str)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(txt)
    else:
        print(txt)


def cmd_audit(a: argparse.Namespace) -> int:
    out = {"knowledge_coverage": K.knowledge_coverage_report(),
           "protected_assets": guard.snapshot(guard.protected_paths(a.protect or [], PF.REPO))}
    if a.db:
        out["database"] = PF.audit_db(a.db)
    _write(a.out, out)
    return 0


def cmd_game_prep(a: argparse.Namespace) -> int:
    sides = (OFF, DEF) if a.side == "both" else (a.side,)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wm = WorldModel.from_dict(json.loads(Path(a.model).read_text())) if a.model else None
    for s in sides:
        prop = PRE.build_proposal(s, opponent_id=a.opponent, opponent_class=a.opponent_class, model=wm, max_formations=a.max_formations)
        (out / f"proposal_{s}.json").write_text(json.dumps(prop, indent=2))
        lines = [f"# {s.upper()} proposal for {a.opponent}  (NOT INSTALLED)", f"proposed fingerprint: `{prop['proposed_fingerprint']}`",
                 f"{prop['n_formations']} formations, {prop['n_plays']} plays; unmet requirements: {prop['unmet_requirements'] or 'none'}", ""]
        for f in prop["formations"]:
            lines += [f"## {f['formation']}  <- {f['source_playbook']} ({f['n_plays']} plays)", f"why: {f['why']}", f"install: {f['install']}",
                      "missing verification: " + "; ".join(f["verification_missing"]), ""]
        (out / f"install_{s}.md").write_text("\n".join(lines))
        print(f"{s}: {prop['n_formations']} formations -> {out / f'proposal_{s}.json'}  fingerprint {prop['proposed_fingerprint']}")
    print("After physically installing AND checking each formation in Madden, run `confirm-install`. Nothing is marked installed automatically.")
    return 0


def cmd_confirm(a: argparse.Namespace) -> int:
    prop = json.loads(Path(a.proposal).read_text())
    rec = PRE.confirm_install(prop, a.confirmed_books, attested_by=a.attested_by)
    print(f"recorded owner attestation for {rec['side']} fingerprint {rec['fingerprint']} by {rec['attested_by']}")
    return 0


def cmd_preflight(a: argparse.Namespace) -> int:
    rep = PF.preflight(db=a.db, confirmed_books=a.confirmed_books, model=a.model, calls=a.calls, extra_protected=a.protect or [])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(PF.render_report(rep))
        Path(a.out).with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"VERDICT: {rep['verdict']}  blocking: {rep['blocking'] or 'none'}")
    for g in rep["gates"]:
        print(f"  [{g['status']:7s}] {g['gate']}")
    return 0 if rep["verdict"] != "NO-GO" else 2


def _ask(prompt: str, default: str = "") -> str:
    try:
        v = input(prompt).strip()
    except EOFError:
        return "q"
    return v or default


def cmd_live(a: argparse.Namespace) -> int:
    confirmed = PF.load_confirmed_books(a.confirmed_books)
    if a.side not in confirmed:
        print(f"REFUSING: no owner-confirmed installed {a.side} book in {a.confirmed_books}. Install, verify, then run confirm-install.", file=sys.stderr)
        return 2
    rec = confirmed[a.side]
    wm = WorldModel.from_dict(json.loads(Path(a.model).read_text())) if a.model else None
    store = UnifiedStore(a.store)
    art = f"{Path(a.model).name}:{guard.sha256_file(Path(a.model))[:10]}" if a.model else "cold_start_prior_v1"
    co = UnifiedCoordinator(a.side, rec["inventory"], confirmed_fingerprint=rec["fingerprint"], model=wm, store=store, artifact=art,
                            roles_file=a.verified_roles, planner=_planner(a))
    print("history:", co.ingest_history(store.load_games()))
    game_id = a.game_id or f"{a.opponent}-{len(store.load_games()) + 1:03d}"
    seq = 1
    print(f"UNIFIED {a.side.upper()} vs {a.opponent} | model={art} | q to quit")
    while True:
        raw = _ask(f"[{seq}] down distance yardline (e.g. '3 7 45'): ")
        if raw in ("q", "quit"):
            break
        try:
            d, dist, yl = (int(x) for x in raw.split())
        except ValueError:
            print("need three integers")
            continue
        st = GameState(game_id, seq, a.side, a.opponent, d, dist, yl, opponent_class=a.opponent_class, playbook_fingerprint=rec["fingerprint"])
        card = co.recommend(st)
        print("\n" + card.render() + "\n")
        ex = _ask("executed as called? y / n(not run) / u(unknown) [u]: ", "u")
        rec_act = co.snaps[st.snap_id].recommended
        co.record_execution(st, rec_act if ex == "y" else None, EXEC_CONFIRMED if ex == "y" else EXEC_NOT_RUN if ex == "n" else EXEC_UNKNOWN)
        opp = _ask("opponent call revealed 'formation | play' (blank=unknown): ")
        if "|" in opp:
            f, p = (x.strip() for x in opp.split("|", 1))
            mk = K.defense_play if a.side == OFF else K.offense_play
            co.record_opponent(st, OpponentAction(DEF if a.side == OFF else OFF, f, p, mk(f, p).signature, "typed_exact", 1.0))
        y = _ask("yards (blank=skip outcome, q=quit): ")
        if y in ("q", ""):
            if y == "q":
                break
        else:
            flags = _ask("flags: td / to / sack / fd (space-separated, blank none): ").split()
            yards = float(y)
            co.close_snap(st, Outcome(yards, "fd" in flags or "td" in flags or yards >= dist, "td" in flags, "to" in flags, "sack" in flags, yards >= 15))
        seq += 1
    co.end_game(game_id)
    store.close()
    return 0


def _planner(a: argparse.Namespace):
    from cfb_coach.madden.unified.adjustments import AdjustmentPlanner
    conf = json.loads(Path(a.adjustment_confirmations).read_text()) if getattr(a, "adjustment_confirmations", None) else {}
    return AdjustmentPlanner(conf)


def cmd_postgame(a: argparse.Namespace) -> int:
    store = UnifiedStore(a.store)
    games = store.load_games()
    rows, excluded = PG.validate_labels(games)
    out = Path(a.out)
    man = PG.write_snapshot(rows, excluded, out / "snapshot")
    print("snapshot:", man)
    rep = PG.train_challenger(PG.read_snapshot(out / "snapshot" / "snapshot.jsonl"), out / "challenger", champion_path=a.champion)
    print(json.dumps({k: rep[k] for k in ("rows", "games", "gate", "promotion", "champion_modified")}, indent=1, default=str))
    return 0


def cmd_smoke(a: argparse.Namespace) -> int:
    rep = PF.preflight(calls=a.calls)
    print(PF.render_report(rep)[:1500])
    return 0 if rep["verdict"] != "NO-GO" else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("unified", description="Unified football coordinator (unified_v1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audit"); s.add_argument("--db"); s.add_argument("--out"); s.add_argument("--protect", nargs="*"); s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("game-prep"); s.add_argument("--side", choices=("offense", "defense", "both"), default="both")
    s.add_argument("--opponent", default="cpu"); s.add_argument("--opponent-class", default="cpu"); s.add_argument("--model")
    s.add_argument("--max-formations", type=int, default=15); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_game_prep)
    s = sub.add_parser("confirm-install"); s.add_argument("--proposal", required=True); s.add_argument("--attested-by", required=True)
    s.add_argument("--confirmed-books", required=True); s.set_defaults(fn=cmd_confirm)
    s = sub.add_parser("preflight"); s.add_argument("--db"); s.add_argument("--confirmed-books"); s.add_argument("--model")
    s.add_argument("--calls", type=int, default=100); s.add_argument("--out"); s.add_argument("--protect", nargs="*"); s.set_defaults(fn=cmd_preflight)
    s = sub.add_parser("live"); s.add_argument("--side", choices=(OFF, DEF), required=True); s.add_argument("--opponent", required=True)
    s.add_argument("--opponent-class", default="cpu"); s.add_argument("--confirmed-books", required=True); s.add_argument("--store", required=True)
    s.add_argument("--model"); s.add_argument("--game-id"); s.add_argument("--verified-roles"); s.add_argument("--adjustment-confirmations"); s.set_defaults(fn=cmd_live)
    s = sub.add_parser("postgame"); s.add_argument("--store", required=True); s.add_argument("--out", required=True); s.add_argument("--champion"); s.set_defaults(fn=cmd_postgame)
    s = sub.add_parser("smoke"); s.add_argument("--calls", type=int, default=100); s.set_defaults(fn=cmd_smoke)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)
