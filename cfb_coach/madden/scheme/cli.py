"""``python3 -m cfb_coach ml civil-scheme <action>`` — the civil_scheme_only operator surface.

Writes to the Madden DB happen ONLY in enable / disable / verify / prepare-book --stage /
report-adjustments, each after a consistent SQLite backup. train / evaluate work on a backup COPY
opened read-only and prove (sha256) the copy is unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import mode as md


def _db_path(args) -> Path:
    if getattr(args, "db", None):
        return Path(args.db).expanduser()
    from cfb_coach.madden.model.cli import _resolved_madden_db_path
    return _resolved_madden_db_path()


def backups_root() -> Path:
    return Path.home() / ".cfb-coach" / "backups" / "civil_scheme"


def backup(path: Path, tag: str) -> Path:
    from cfb_coach.db import CoachDB
    dest = backups_root() / f"{path.stem}.{tag}.{time.strftime('%Y%m%d-%H%M%S')}.db"
    src = CoachDB.open_read_only(path)
    try:
        src.backup_to(dest)
    finally:
        src.close()
    print(f"backup: {dest}")
    return dest


def _open_rw(path: Path):
    from cfb_coach.madden.model.cli import open_explicit_madden_db
    return open_explicit_madden_db(path, read_only=False)


def _open_ro(path: Path):
    from cfb_coach.db import CoachDB
    return CoachDB.open_read_only(path)


def _latest_artifact() -> Path | None:
    p = md.models_root() / "latest"
    return Path(p.read_text().strip()) if p.is_file() else None


# ------------------------------------------------------------------ actions

def cmd_status(args) -> int:
    from cfb_coach.madden.scheme import legality as lg
    from cfb_coach.madden.scheme import model as sm
    path = _db_path(args)
    print(f"Using DB: {path}")
    db = _open_ro(path) if path.is_file() else None
    cat = sc.load()
    print(f"catalog: {cat.catalog_id} version {cat.version()[:12]} ({len(cat.variants)} variants, "
          f"contract errors: {sc.contract_errors(cat) or 'none'})")
    mode = md.offense_mode(db)
    st = md.load_state(db)
    print(f"offense mode: {mode}  (meta: {json.dumps(st) if st else 'unset'}; env {md.ENV_MODE}="
          f"{__import__('os').environ.get(md.ENV_MODE, '') or 'unset'})")
    art = md.artifact_dir(db)
    print(f"artifact: {art or 'none'}")
    if art and (Path(art) / "model.json").is_file():
        try:
            m = sm.load(art)
            ok = (not st.get("model_sha256")) or st.get("model_sha256") == m.meta.get("model_sha256")
            print(f"  loads: yes  sha256 {m.meta.get('model_sha256', '')[:16]}  matches enabled record: {ok}")
            print(f"  architecture: {m.meta.get('architecture')}  rows {m.meta.get('training_rows')} "
                  f"games {m.meta.get('games')}  trained {m.meta.get('trained_at')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  loads: NO — {exc}")
    legal, verdicts = lg.legal_set(cat, db=db, require_verified=True)
    print(f"live-legal variants: {len(legal)} / {len(cat.variants)}")
    if db is not None:
        db.close()
    return 0


def cmd_checklist(args) -> int:
    from cfb_coach.madden.scheme import adjustments as adj
    from cfb_coach.madden.scheme import legality as lg
    path = _db_path(args)
    db = _open_ro(path) if path.is_file() else None
    cat = sc.load()
    _legal, verdicts = lg.legal_set(cat, db=db, require_verified=True)
    led = lg.load_verified(db)
    print("PREGAME CHECKLIST — civil_scheme_only (no controller buttons are given; set each adjustment by hand)")
    print("Restriction: the 26 variants are ALL PASSES. No run, QB sneak or red-zone package can be called in this mode;")
    print("  short yardage and goal line are covered only by these passes (see the evaluation report).")
    by_form: dict[str, list] = {}
    for v in cat.variants:
        by_form.setdefault(v.formation, []).append(v)
    for f, vs in by_form.items():
        print(f"\n== {f} ==")
        for v in vs:
            vd = verdicts[v.variant_id]
            status = ("LIVE-LEGAL (adjustments owner-confirmed 2026-11-10, t333u)" if vd.legal else
                      "BLOCKED: " + "; ".join(vd.reasons))
            print(f"[ ] {v.variant_name}  [{v.variant_id}]  — {status}")
            print(f"      base play: {v.base_play}")
            for line in adj.checklist_lines(v):
                print(f"      {line}")
            if v.variant_id in led:
                print(f"      verified_at {led[v.variant_id].get('verified_at')}")
    print("\nAll catalog adjustments are owner-confirmed executable (2026-11-10, t333u). Formations must be in the")
    print("applied book: ml civil-scheme prepare-book --stage, install, then prep --mark-applied.")
    if db is not None:
        db.close()
    return 0


def cmd_prepare_book(args) -> int:
    from cfb_coach.madden import playbook
    from cfb_coach.madden import catalog as mc
    path = _db_path(args)
    print(f"Using DB: {path}")
    cat = sc.load()
    db = _open_ro(path)
    try:
        applied = playbook.load_books(db).get("offense")
    finally:
        db.close()
    stock = mc.book_formations("offense", "Buccaneers")
    need = []
    for v in cat.variants:
        if v.formation not in need:
            need.append(v.formation)
    have = dict((applied or {}).get("formations") or {})
    missing = [f for f in need if f not in have]
    print(f"applied offense book: {(applied or {}).get('name', 'none')} ({len(have)} formations)")
    for f in need:
        plays = sorted({v.base_play for v in cat.variants if v.formation == f})
        print(f"  {f}: {'installed' if f in have else 'MISSING from applied book'}; base plays: {', '.join(plays)}")
    if not missing:
        print("All 4 scheme formations are in the applied book.")
        return 0
    if not args.stage:
        print("Dry run. Re-run with --stage to stage a pending merged book (current applied + missing formations).")
        return 0
    if not applied:
        print("error: no applied offense book to merge into; run prep first", file=sys.stderr)
        return 2
    backup(path, "pre-prepare-book")
    db = _open_rw(path)
    try:
        state = playbook._load_state(db)
        rec = json.loads(json.dumps(applied))
        for f in missing:
            if f in stock:
                rec["formations"][f] = list(stock[f])
        rec["rev"] = int(rec.get("rev") or 1) + 1
        rec["reason"] = "civil_scheme_only: add scheme formations"
        rec["core"] = list(rec["formations"])
        state["pending"]["offense"] = rec
        playbook._save_state(db, state)
    finally:
        db.close()
    print(f"staged pending offense book with {len(missing)} added formation(s): {', '.join(missing)}")
    print("Install it in Madden, then: python3 -m cfb_coach prep --game madden27 -o cpu --mark-applied")
    return 0


def _snapshot(path: Path) -> Path:
    snap = backup(path, "train-snapshot")
    snap.chmod(0o444)
    return snap


def cmd_evaluate(args, *, train_after: bool = False) -> int:
    from cfb_coach.madden.scheme import evaluate as evm
    from cfb_coach.madden.scheme import model as sm
    path = _db_path(args)
    print(f"Using DB: {path} (training/evaluation read a backup copy, read-only)")
    src = Path(args.db_copy) if getattr(args, "db_copy", None) else _snapshot(path)
    rep = evm.run(src, n_members=args.members, log=lambda s: print(s, flush=True))
    out_dir = Path(args.out) if getattr(args, "out", None) else md.models_root() / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rp = out_dir / f"civil_scheme_eval_{stamp}.json"
    rp.write_text(json.dumps(rep, indent=1, default=str))
    print(f"report: {rp}")
    print(f"chosen heads: {rep['chosen']}")
    print(f"DB copy unchanged: {rep['db_integrity']['unchanged']}")
    if not train_after:
        return 0
    model = evm.train_final(src, rep["chosen"], n_members=5, eval_report=rep)
    version = time.strftime("v%Y%m%d-%H%M%S")
    model.meta["version"] = version
    model.meta["source_db_original"] = str(path)
    art = sm.save(model, md.models_root() / version)
    (art / "eval_report.json").write_text(json.dumps(rep, indent=1, default=str))
    (md.models_root() / "latest").write_text(str(art))
    m2 = sm.load(art)
    print(f"artifact: {art}  sha256 {m2.meta['model_sha256'][:16]}  (not enabled; run `ml civil-scheme enable`)")
    return 0


def cmd_train(args) -> int:
    return cmd_evaluate(args, train_after=True)


def cmd_enable(args) -> int:
    from cfb_coach.madden.scheme import coordinator as co
    from cfb_coach.madden.scheme import model as sm
    path = _db_path(args)
    print(f"Using DB: {path}")
    art = Path(args.artifact) if args.artifact else _latest_artifact()
    if art is None or not (art / "model.json").is_file():
        print("error: no trained artifact; run `ml civil-scheme train` first", file=sys.stderr)
        return 2
    m = sm.load(art)                          # verifies sha256
    co.load_scorer(art)                       # proves the live path can build the scorer
    backup(path, "pre-enable")
    db = _open_rw(path)
    try:
        prev = md.load_state(db)
        st = md.set_state(db, enabled=True, artifact=str(art), model_sha256=m.meta["model_sha256"])
        st["previous"] = {k: prev.get(k) for k in ("artifact", "model_sha256", "enabled")}
        db.set_meta(md.META_KEY, json.dumps(st, sort_keys=True))
    finally:
        db.close()
    print(f"civil_scheme_only ENABLED with {art} (sha256 {m.meta['model_sha256'][:16]})")
    print("Unrestricted coordinator remains available: play ... --unrestricted-offense, or `ml civil-scheme disable`.")
    return 0


def cmd_disable(args) -> int:
    path = _db_path(args)
    print(f"Using DB: {path}")
    backup(path, "pre-disable")
    db = _open_rw(path)
    try:
        st = md.load_state(db)
        if args.revert_artifact:
            prev = (st.get("previous") or {})
            if not prev.get("artifact"):
                print("no previous artifact recorded; mode disabled only")
            else:
                md.set_state(db, enabled=bool(prev.get("enabled")), artifact=prev["artifact"],
                             model_sha256=prev.get("model_sha256"))
                print(f"reverted to previous artifact {prev['artifact']} (enabled={bool(prev.get('enabled'))})")
                return 0
        md.set_state(db, enabled=False, artifact=None, model_sha256=None)
    finally:
        db.close()
    print("civil_scheme_only DISABLED — offense uses the existing unrestricted coordinator.")
    return 0


def cmd_verify(args) -> int:
    from cfb_coach.madden.scheme import legality as lg
    path = _db_path(args)
    print(f"Using DB: {path}")
    cat = sc.load()
    backup(path, "pre-verify")
    db = _open_rw(path)
    try:
        lg.record_verified(db, cat, args.variant, note=args.note or "", revoke=args.revoke,
                           book_play=args.book_play)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        db.close()
    print(("revoked " if args.revoke else "verified executable: ") + args.variant)
    return 0


def cmd_shadow(args) -> int:
    from cfb_coach.madden.scheme import coordinator as co
    from cfb_coach.situation import Situation
    path = _db_path(args)
    print(f"Using DB: {path} (read-only shadow; nothing is logged)")
    db = _open_ro(path)
    try:
        cat = sc.load()
        art = Path(args.artifact) if args.artifact else (md.artifact_dir(db) or _latest_artifact())
        if art is None:
            print("error: no artifact", file=sys.stderr)
            return 2
        scorer = co.load_scorer(art, cat)
        sit = Situation(raw="shadow", down=args.down, distance=args.distance, yardline=args.yardline,
                        coverage_hint=args.look, coverage_source="live" if args.look else "none")
        sit.extras = {"session_id": args.session} if args.session else {}
        from cfb_coach.madden import playbook
        book = playbook.load_books(db).get("offense")
        call, ranked = co.select(scorer, cat, db.conn, db=db, sit=sit, opponent_id=args.opponent, book=book,
                                 shadow=not args.live_legal_only)
        print("\n".join(co.format_lines(call)))
        if ranked:
            print("\nRANKING (shadow):")
            for i, s in enumerate(ranked[: args.top]):
                print(f"{i + 1:2d}. {s.variant.variant_name:55s} util {s.utility:.3f}  P(succ) {s.p_success:.2f}"
                      f"  beater {s.parts['beater']:+.2f}  adj {s.parts['adjustment_logit_delta']:+.2f}")
    finally:
        db.close()
    return 0


def _parse_idx(s: str | None) -> list[int]:
    return [int(x) - 1 for x in (s or "").replace(" ", "").split(",") if x]


def cmd_report_adjustments(args) -> int:
    from cfb_coach.madden.scheme import evidence as ev
    path = _db_path(args)
    print(f"Using DB: {path}")
    cat = sc.load()
    changed = {}
    for item in args.changed or []:
        k, _, v = item.partition("=")
        changed[int(k) - 1] = v
    backup(path, "pre-report")
    db = _open_rw(path)
    try:
        rep = ev.report_execution(db.conn, cat, snap_id=args.snap, applied=_parse_idx(args.applied),
                                  skipped=_parse_idx(args.skipped), changed=changed, note=args.note or "")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        db.close()
    print(json.dumps(rep, indent=1))
    return 0


def cmd_log(args) -> int:
    from cfb_coach.madden.scheme import evidence as ev
    path = _db_path(args)
    db = _open_ro(path)
    try:
        rows = ev.variant_log(db.conn, sc.load())
    finally:
        db.close()
    print(json.dumps(rows, indent=1, default=str))
    return 0


def register(ml_sub: Any) -> None:
    p = ml_sub.add_parser("civil-scheme", help="civil_scheme_only offense: train/evaluate/enable/disable/verify/shadow")
    s = p.add_subparsers(dest="civil_cmd", required=True)

    def add(name, func, help_):
        q = s.add_parser(name, help=help_)
        q.add_argument("--db", default=None, help="Madden DB path (default: the active Madden DB)")
        q.set_defaults(func=func)
        return q

    add("status", cmd_status, "mode, artifact, load proof, live-legal count")
    add("checklist", cmd_checklist, "pregame checklist: catalog-approved vs verified executable")
    q = add("prepare-book", cmd_prepare_book, "check / stage the 4 scheme formations in the applied book")
    q.add_argument("--stage", action="store_true", help="back up, then stage a PENDING merged book")
    for name, func in (("evaluate", cmd_evaluate), ("train", cmd_train)):
        q = add(name, func, f"{name} on a read-only backup copy")
        q.add_argument("--db-copy", default=None, help="use this existing copy instead of making a backup")
        q.add_argument("--out", default=None)
        q.add_argument("--members", type=int, default=3, help="GBT bootstrap members during evaluation")
    q = add("enable", cmd_enable, "turn civil_scheme_only on with a trained artifact (backs up first)")
    q.add_argument("--artifact", default=None)
    q = add("disable", cmd_disable, "turn it off (existing unrestricted coordinator)")
    q.add_argument("--revert-artifact", action="store_true", help="restore the previously enabled artifact instead")
    q = add("verify", cmd_verify, "owner: mark ONE exact variant executable after checking it in Madden")
    q.add_argument("--variant", required=True)
    q.add_argument("--note", default="")
    q.add_argument("--revoke", action="store_true")
    q.add_argument("--book-play", default=None,
                   help="owner confirmation that the guide's base play is named differently in the in-game book "
                        "(e.g. Z Spot -> 'L Spot'); never inferred")
    q = add("shadow", cmd_shadow, "offline ranking for a situation (unverified variants allowed, labeled SHADOW)")
    q.add_argument("--opponent", "-o", default="cpu")
    q.add_argument("--down", type=int, required=True)
    q.add_argument("--distance", type=int, required=True)
    q.add_argument("--yardline", type=int, required=True, help="yards from our goal line")
    q.add_argument("--look", default=None, help="live pre-snap coverage look, e.g. 'cover 3'")
    q.add_argument("--session", default=None, help="session id to include its closed snaps")
    q.add_argument("--artifact", default=None)
    q.add_argument("--top", type=int, default=10)
    q.add_argument("--live-legal-only", action="store_true", help="apply the live (owner-verified) filter")
    q = add("report-adjustments", cmd_report_adjustments, "record which adjustments were really run on a snap")
    q.add_argument("--snap", type=int, required=True)
    q.add_argument("--applied", default="", help="1-based, e.g. 1,3")
    q.add_argument("--skipped", default="", help="1-based")
    q.add_argument("--changed", action="append", help="N=what was run instead (kept, not trained)")
    q.add_argument("--note", default="")
    add("log", cmd_log, "variant-level log (decision + outcome + what was executed), JSON")
