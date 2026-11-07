#!/usr/bin/env python3
"""Offense structure pilot: train an artifact from a COPY of the game DB, validate it, run shadow / CPU pilot.

Nothing here edits the installed playbook, the live caller or the live DB. Everything is written under --workdir
(default ~/.cfb-coach-pilot, deliberately NOT ~/.cfb-coach). `pilot` refuses until you run `approve`.

  config-init   write an UNAPPROVED config template
  train         fit the artifact from a copy of the DB (read-only)
  validate      check artifact + config (digests, fingerprint, paths); exit 1 on any problem
  shadow        one typed situation -> card + logged recommendation (no approval needed)
  pilot         same, but only if the config is approved and mode=pilot
  approve       YOU record approval (flips approved=true, mode=pilot)
  confirm       record what you actually executed for a recommendation
  report        summarize the log (recommended vs confirmed executed)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from urllib.parse import quote
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cfb_coach.madden.adjustments import executable as ex  # noqa: E402
from cfb_coach.madden.model import heldout_eval as he  # noqa: E402
from cfb_coach.madden.model import offense_decision as od  # noqa: E402
from cfb_coach.madden.model import offense_pilot as pilot  # noqa: E402
from cfb_coach.madden.model import offense_presnap_advice as advice  # noqa: E402
from cfb_coach.madden.model import offense_structural_model as sm  # noqa: E402

DEFAULT_WORKDIR = Path.home() / ".cfb-coach-pilot"


def _paths(a: argparse.Namespace) -> tuple[Path, Path, Path]:
    w = Path(a.workdir).expanduser()
    return w, w / "pilot_config.json", w / "offense_structural_v1.json"


def _session(a: argparse.Namespace, want_mode: str | None) -> pilot.PilotSession:
    w, cfg_path, _ = _paths(a)
    cfg = pilot.PilotConfig.load(cfg_path)
    if want_mode == "pilot" and not (cfg.mode == "pilot" and cfg.approved):
        raise pilot.PilotRefused("pilot not approved: run `approve` first (shadow needs no approval)")
    if want_mode == "shadow":
        cfg.mode = "shadow"
    loadout = ex.Loadout.load(cfg.loadout_path) if cfg.loadout_path else None
    conf = json.loads(Path(cfg.confirmations_path).read_text()) if cfg.confirmations_path else None
    return pilot.PilotSession(cfg, loadout=loadout, confirmations=conf)


def cmd_config_init(a: argparse.Namespace) -> int:
    w, cfg_path, art = _paths(a)
    w.mkdir(parents=True, exist_ok=True)
    if cfg_path.exists() and not a.force:
        print(f"{cfg_path} exists (use --force to overwrite)")
        return 1
    cfg = pilot.PilotConfig(artifact_path=str(art), log_path=str(w / "pilot_log.jsonl"))
    cfg.save(cfg_path)
    print(f"wrote UNAPPROVED template {cfg_path}")
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    w, cfg_path, art_path = _paths(a)
    w.mkdir(parents=True, exist_ok=True)
    data = he.load_eval_data(a.db)
    art = sm.train(data.rows)
    sm.save_artifact(art, art_path)
    print(f"artifact -> {art_path}\n{art.notes}\nverified rows: {art.n_train}, games: {art.trained_games}, residual_active={art.residual_active}")
    if cfg_path.exists():
        cfg = pilot.PilotConfig.load(cfg_path)
        cfg.artifact_path, cfg.artifact_digest = str(art_path), sm.artifact_digest(art)
        cfg.assignment_digest = art.assignment_digest
        cfg.save(cfg_path)
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    _, cfg_path, _ = _paths(a)
    problems = pilot.validate_config(pilot.PilotConfig.load(cfg_path))
    print("OK: artifact and config valid" if not problems else "PROBLEMS:\n- " + "\n- ".join(problems))
    return 0 if not problems else 1


def _situation(a: argparse.Namespace) -> dict:
    s = {"down": a.down, "distance": a.distance, "yardline": a.yardline, "opponent_id": a.opponent,
         "opponent_type": a.opponent_type, "play_clock_remaining": a.play_clock}
    return {k: v for k, v in s.items() if v is not None}


def _run(a: argparse.Namespace, mode: str) -> int:
    sess = _session(a, mode)
    using_history = bool(a.history_db or a.session_id or a.next_snap_seq is not None)
    if using_history:
        if not (a.history_db and a.session_id and a.next_snap_seq is not None):
            raise pilot.PilotRefused("coverage feedback requires --history-db, --session-id, and --next-snap-seq")
        path = Path(a.history_db).expanduser().resolve()
        # Opening a live Madden DB read-only is safe; this does not write to it.
        with sqlite3.connect("file:" + quote(str(path), safe="/") + "?mode=ro", uri=True) as conn:
            rec = sess.recommend_from_logged_game(
                conn, _situation(a), session_id=a.session_id,
                next_snap_seq=a.next_snap_seq, game_id=a.game or a.session_id)
    else:
        rec = sess.recommend(_situation(a), game_id=a.game or "")
    if rec.get("coverage_history"):
        ctx = rec["coverage_history"]
        print(f"Previous-play coverage samples: {ctx['named_n']} named / "
              f"{ctx['pressure_n']} pressure-only / {ctx['unknown_n']} unknown")
    print(advice.format_card(rec["card"]) if "PRIMARY READ" in rec["card"] else json.dumps(rec["card"], indent=1))
    print(f"\nrec_id={rec['rec_id']}  latency={rec['latency_ms']}ms  baseline_agrees={rec['baseline_agrees']}"
          + (f"  FALLBACK={rec['fallback_reason']}" if rec["fallback_reason"] else ""))
    print("After the snap:  confirm --rec-id", rec["rec_id"], "--formation ... --play ... [--adjustment ...]")
    return 0


def cmd_approve(a: argparse.Namespace) -> int:
    _, cfg_path, _ = _paths(a)
    cfg = pilot.PilotConfig.load(cfg_path)
    cfg.mode = "pilot"
    pilot.approve(cfg, a.by)
    problems = pilot.validate_config(cfg)
    if problems:
        print("cannot approve:\n- " + "\n- ".join(problems))
        return 1
    cfg.save(cfg_path)
    print(f"approved by {cfg.approved_by} at {cfg.approved_at}; mode=pilot (CPU only)")
    return 0


def cmd_confirm(a: argparse.Namespace) -> int:
    sess = _session(a, None)
    print(json.dumps(sess.confirm_execution(a.rec_id, formation=a.formation, play=a.play,
                                            adjustment=a.adjustment, status=a.status), indent=1))
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    _, cfg_path, _ = _paths(a)
    print(json.dumps(pilot.summarize_log(pilot.PilotConfig.load(cfg_path).log_path), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("config-init"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_config_init)
    p = sub.add_parser("train"); p.add_argument("--db", required=True, help="COPY of the game DB"); p.set_defaults(fn=cmd_train)
    p = sub.add_parser("validate"); p.set_defaults(fn=cmd_validate)
    for name in ("shadow", "pilot"):
        p = sub.add_parser(name)
        p.add_argument("--down", type=int, required=True); p.add_argument("--distance", type=int, required=True)
        p.add_argument("--yardline", type=int, required=True, help="yards from own goal line, 1-99")
        p.add_argument("--opponent", default="cpu"); p.add_argument("--opponent-type", default="cpu")
        p.add_argument("--play-clock", type=float); p.add_argument("--game")
        p.add_argument("--history-db", help="Existing Madden SQLite DB, read-only; no extra logger")
        p.add_argument("--session-id", help="Current game session ID from existing UI")
        p.add_argument("--next-snap-seq", type=int, help="Sequence number of the NEXT unplayed snap")
        p.set_defaults(fn=lambda a, n=name: _run(a, n))
    p = sub.add_parser("approve"); p.add_argument("--by", required=True); p.set_defaults(fn=cmd_approve)
    p = sub.add_parser("confirm"); p.add_argument("--rec-id", required=True); p.add_argument("--formation", required=True)
    p.add_argument("--play", required=True); p.add_argument("--adjustment", default=od.NO_ADJUSTMENT)
    p.add_argument("--status", default="executed", choices=["executed", "not_executed", "unknown"]); p.set_defaults(fn=cmd_confirm)
    p = sub.add_parser("report"); p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except pilot.PilotRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
