"""``python3 -m cfb_coach ml ravens-defense <action>`` — the ravens_defense_ml operator surface.

Read-only: audit, train, evaluate, verify-book, shadow, postgame, status, coaching-audit, users.
  train/evaluate/postgame open ``--db`` with SQLite mode=ro + query_only and prove (sha256) it is unchanged;
  they refuse mode-444 official copies (use a mode-644 backup copy).
DB writes happen ONLY in enable / disable, each after a consistent SQLite backup. Nothing here promotes.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense import mode as md


def _db_path(args) -> Path:
    if getattr(args, "db", None):
        return Path(args.db).expanduser()
    from cfb_coach.madden.model.cli import _resolved_madden_db_path
    return _resolved_madden_db_path()


def _require_backup_copy(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"error: {path} does not exist")
    mode = stat.S_IMODE(path.stat().st_mode)
    if not mode & stat.S_IWUSR:
        raise SystemExit(f"error: {path} is mode {oct(mode)} (an official read-only copy); pass a mode-644 "
                         "backup copy as --db")


def _installed(path: Path) -> dict[str, Any] | None:
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import playbook
    db = CoachDB.open_read_only(path)
    try:
        return playbook.load_books(db).get("defense")
    except Exception:  # noqa: BLE001 — no applied book = nothing installed
        return None
    finally:
        db.close()


def _snaps(path: Path, conf):
    from cfb_coach.madden.ravens_defense import data
    conn = data.open_ro(path)
    return conn, data.read_snaps(conn, standing_macro_attestation=conf.standing_macro_attestation)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=str)


def backup(path: Path, tag: str) -> Path:
    from cfb_coach.db import CoachDB
    dest = Path.home() / ".cfb-coach" / "backups" / "ravens_defense" / f"{path.stem}.{tag}.{time.strftime('%Y%m%d-%H%M%S')}.db"
    src = CoachDB.open_read_only(path)
    try:
        src.backup_to(dest)
    finally:
        src.close()
    print(f"backup: {dest}")
    return dest


# ------------------------------------------------------------------ read-only actions

def cmd_audit(args) -> int:
    from cfb_coach.madden.ravens_defense import confirmations as cf, data
    path = _db_path(args)
    _require_backup_copy(path)
    conf = cf.load(args.confirmations)
    before = data.sha256_file(path)
    conn, snaps = _snaps(path, conf)
    rep = {"db": str(path), "db_sha256": before, **data.audit(conn, snaps)}
    conn.close()
    rep["db_unchanged"] = data.sha256_file(path) == before
    if not args.per_game:
        rep.pop("per_game")
    print(_json(rep))
    return 0


def evaluation(path: Path, conf) -> dict[str, Any]:
    from cfb_coach.madden.ravens_defense import data, evaluate as ev, opponent as op
    conn, snaps = _snaps(path, conf)
    try:
        cat = rc.load()
        belief = op.evaluate_forward([s for s in snaps if s.concept is not None and s.closed])
        succ = ev.presnap_success_eval(snaps)
        cand = ev.candidate_outcome_eval(snaps)
        v2 = ev.v2_outcome_eval(conn)
        rep = ev.replay(snaps)
        audit = data.audit(conn, snaps)
    finally:
        conn.close()
    gate = "PASS" if (cand.get("status") == "EVALUATED" and (cand.get("trained_vs_prior_gain_ci") or [0])[0] > 0
                      and cand.get("heldout_snaps", 0) >= 50) else "NO_GO"
    return {"opponent_belief": belief, "presnap_success": succ, "candidate_outcome": cand,
            "defensive_v2_baseline": v2, "shadow_replay": rep, "audit": audit, "catalog": cat.version(),
            "candidate_outcome_gate": gate,
            "latency_gate": "PASS" if (rep["decision_latency_ms"]["p95"] or 1e9) < 150 else "FAIL"}


def cmd_evaluate(args) -> int:
    from cfb_coach.madden.ravens_defense import confirmations as cf, data
    path = _db_path(args)
    _require_backup_copy(path)
    before = data.sha256_file(path)
    out = evaluation(path, cf.load(args.confirmations))
    out["db"], out["db_sha256_before"] = str(path), before
    out["db_sha256_after"] = data.sha256_file(path)
    out["db_unchanged"] = out["db_sha256_after"] == before
    if args.out:
        Path(args.out).expanduser().write_text(_json(out), encoding="utf-8")
        print(f"wrote {args.out}")
    for k in ("opponent_belief", "presnap_success"):
        out[k].pop("per_game", None)
    out["shadow_replay"].pop("cards", None)
    out["audit"].pop("per_game", None)
    print(_json({k: out[k] for k in ("opponent_belief", "presnap_success", "candidate_outcome",
                                     "candidate_outcome_gate", "latency_gate", "db_unchanged")}))
    return 0


def cmd_train(args) -> int:
    from cfb_coach.madden.ravens_defense import confirmations as cf, data, model as rm
    path = _db_path(args)
    _require_backup_copy(path)
    conf = cf.load(args.confirmations)
    before = data.sha256_file(path)
    ev = evaluation(path, conf)
    conn, snaps = _snaps(path, conf)
    conn.close()
    cat = rc.load()
    installed = _installed(path)
    m = rm.train(snaps, cat, belief_eval=ev["opponent_belief"], success_eval=ev["presnap_success"],
                 source={"db": str(path), "db_sha256": before})
    version = time.strftime("%Y%m%d-%H%M%S") + "-" + m.outcome.status.lower()
    out = Path(args.out).expanduser() / version if args.out else rm.models_root() / version
    from cfb_coach.madden.ravens_defense import legality as lg
    book = lg.book_report(cat, installed=rc.installed_two_formation_book(installed), conf=conf)
    after = data.sha256_file(path)
    if after != before:
        raise SystemExit("error: the --db file changed during training; artifact not written")
    extra = {
        "artifact_version": version,
        "held_out_evaluation": {"candidate_outcome": ev["candidate_outcome"], "candidate_outcome_gate": ev["candidate_outcome_gate"],
                                "opponent_belief": {k: v for k, v in ev["opponent_belief"].items() if k != "per_game"},
                                "presnap_success": {k: v for k, v in ev["presnap_success"].items() if k != "per_game"},
                                "latency_gate": ev["latency_gate"]},
        "installed_playbook": {"applied_defense_book": (installed or {}).get("name"),
                               "two_formation_fingerprint": book["installed_two_formation_fingerprint"],
                               "owner_confirmed_fingerprint": book["owner_confirmed_fingerprint"],
                               "live_legal_count": book["live_legal_count"]},
        "macro_status": {mid: conf.macro_confirmed(mid) for mid in cat.raw["macro_catalog"]},
        "user_status": {"verified_assignments": len(conf.user_verifications), "rules": len(cat.raw["user_assignment_rules"])},
        "coaching_profile": {c: conf.coaching_status(c) for c in cat.calls},
        "db_sha256_before": before, "db_sha256_after": after, "db_unchanged": True,
        "promotion": "NONE (artifacts never enable themselves; see `ml ravens-defense enable`)",
    }
    rm.save(m, out, extra)
    (out / "eval.json").write_text(_json(ev), encoding="utf-8")
    print(f"artifact: {out}")
    print(f"outcome model: {m.outcome.status}  verified examples: {m.meta['verified_defensive_execution_examples']}"
          f"  live belief: {m.belief_method}  success base: {m.meta['architecture']['success_base']}")
    print(f"db unchanged: True ({before[:16]})")
    return 0


def cmd_verify_book(args) -> int:
    from cfb_coach.madden.ravens_defense import confirmations as cf, legality as lg
    path = _db_path(args)
    cat = rc.load()
    installed = _installed(path) if path.is_file() else None
    conf = cf.load(args.confirmations)
    rep = lg.book_report(cat, installed=rc.installed_two_formation_book(installed), conf=conf)
    rep["applied_defense_book"] = (installed or {}).get("name")
    rep["applied_defense_formations"] = sorted(((installed or {}).get("formations") or {}).keys())
    print(_json(rep) if args.json else "\n".join(
        [f"applied defense book: {rep['applied_defense_book']}  formations: {rep['applied_defense_formations']}",
         f"two-formation fingerprint: {rep['installed_two_formation_fingerprint']}  owner-confirmed: "
         f"{rep['owner_confirmed_fingerprint']}",
         f"live-legal candidates: {rep['live_legal_count']} / {len(rep['candidates'])}"] +
        [f"  [{'OK' if c['live_legal'] else 'BLOCKED'}] {c['candidate_id']}: {c['formation']} — {c['base_play']}"
         + (f"  ({'; '.join(c['reasons'])})" if c['reasons'] else "") for c in rep["candidates"]]))
    return 0 if rep["live_legal_count"] else 1


def cmd_shadow(args) -> int:
    from cfb_coach.madden.ravens_defense import confirmations as cf, coordinator as co, evaluate as ev
    from cfb_coach.madden.ravens_defense import legality as lg, model as rm, opponent as op
    path = _db_path(args)
    _require_backup_copy(path)
    conf = cf.load(args.confirmations)
    print("SHADOW — not a live call" + ("; install/macro/coaching setup ASSUMED for display" if args.assume_setup else ""))
    conn, snaps = _snaps(path, conf)
    conn.close()
    if args.session:
        rep = ev.replay(snaps, sessions=[args.session], conf=conf, assume_setup=args.assume_setup, keep_cards=True)
        for c in rep["cards"] or []:
            print(f"\n[{c['situation']}] belief {c['belief_top']}  observed after snap: {c['observed_concept_after_snap']}")
            print("  " + "\n  ".join(c["card"]))
            print(f"  top3: {c['top3']}")
        rep.pop("cards")
        print(_json(rep))
        return 0
    art = args.artifact or md.artifact_dir(None)
    if not art:
        raise SystemExit("error: --artifact required (no latest artifact)")
    model = rm.load(art)
    cat = rc.load()
    installed = rc.installed_two_formation_book(_installed(path))
    legal, _ = lg.legal_set(cat, installed=installed, conf=conf, assume_setup=args.assume_setup)
    prior = [co.SnapCtx("shadow", args.opponent, i, None, None, None, closed=True, concept=c.strip())
             for i, c in enumerate((args.prior_concepts or "").split(",")) if c.strip()]
    snap = co.SnapCtx("shadow", args.opponent, len(prior), args.down, args.distance, args.yardline, args.shape)
    d = co.decide(model, legal=legal, snap=snap, same_game=prior, history=model.history, conf=conf, cat=cat,
                  installed=installed, assume_setup=args.assume_setup)
    print("\n".join(co.card_lines(d)))
    if d["status"] == "CALL":
        print(f"belief ({d['belief_method']}): {d['belief']}")
        for s in d["ranked"][: args.top]:
            print(f"  {s['candidate_id']:<24} loss {s['defensive_loss']:.4f}  P(off success) "
                  f"{s['predicted_offense_success']:.3f} ±{s['success_sd']:.3f}")
        print(f"latency {d['latency_ms']} ms  outcome model {d['outcome_model_status']}")
    else:
        print(_json(d.get("reasons")))
    return 0


def cmd_postgame(args) -> int:
    """Champion (an artifact) vs a challenger refit with the new game: report only, never promotes."""
    from cfb_coach.madden.ravens_defense import confirmations as cf, data, evaluate as ev, model as rm
    path = _db_path(args)
    _require_backup_copy(path)
    conf = cf.load(args.confirmations)
    before = data.sha256_file(path)
    champion = rm.load(args.artifact) if args.artifact else None
    out = evaluation(path, conf)
    out["shadow_replay"].pop("cards", None)
    out["champion"] = {"artifact": args.artifact, "outcome_model_status": champion.outcome.status if champion else None,
                       "trained_sessions": len(champion.meta.get("trained_sessions") or []) if champion else None}
    out["challenger"] = {"outcome_model_status": out["candidate_outcome"].get("status"),
                         "verified_ravens_executions": out["audit"]["verified_ravens_executions_with_outcome"]}
    out["promotion"] = "NONE — postgame never promotes; enable a new artifact by hand after the gates pass"
    out["db_unchanged"] = data.sha256_file(path) == before
    if args.out:
        Path(args.out).expanduser().write_text(_json(out), encoding="utf-8")
    print(_json({k: out[k] for k in ("champion", "challenger", "candidate_outcome_gate", "promotion", "db_unchanged")}))
    return 0


def cmd_status(args) -> int:
    from cfb_coach.db import CoachDB
    from cfb_coach.madden.ravens_defense import model as rm
    path = _db_path(args)
    db = CoachDB.open_read_only(path) if path.is_file() else None
    print(f"defense mode: {md.defense_mode(db)}  (meta: {json.dumps(md.load_state(db)) or 'unset'}; env "
          f"{md.ENV_MODE}={os.environ.get(md.ENV_MODE, '') or 'unset'})")
    art = md.artifact_dir(db)
    print(f"artifact: {art or 'none'}")
    if art and (Path(art) / "model.json").is_file():
        m = rm.load(art)
        print(f"  outcome model: {m.outcome.status}  belief: {m.belief_method}")
    if db is not None:
        db.close()
    return 0


def cmd_coaching_audit(args) -> int:
    from cfb_coach.madden.ravens_defense import confirmations as cf
    print(_json(cf.coaching_audit(rc.load(), cf.load(args.confirmations))))
    return 0


def cmd_users(args) -> int:
    from cfb_coach.madden.ravens_defense import confirmations as cf, users as us
    for u in us.audit(rc.load(), cf.load(args.confirmations)):
        print(_json(u) if args.json else f"[{u['status']}] {u['assignment_id']}: {u['call_id']} / {u['shape']} — guide USER (not displayed live): {u['guide_user']}")
    return 0


# ------------------------------------------------------------------ writes (backup first)

def release_gates(art: Path, db_path: Path, conf) -> dict[str, Any]:
    """All must PASS for enable. Fails closed on any error."""
    from cfb_coach.madden.ravens_defense import legality as lg, model as rm
    g: dict[str, Any] = {}
    try:
        m = rm.load(art)
        man = rm.manifest(art)
        import hashlib
        g["artifact_sha256"] = "PASS" if hashlib.sha256((art / "model.json").read_bytes()).hexdigest() == man.get("model_sha256") else "FAIL"
        g["outcome_model_trained"] = "PASS" if m.outcome.status == "TRAINED" else f"FAIL ({m.outcome.status})"
        ho = man.get("held_out_evaluation") or {}
        g["held_out_candidate_outcome"] = "PASS" if ho.get("candidate_outcome_gate") == "PASS" else "FAIL"
        g["latency"] = "PASS" if ho.get("latency_gate") == "PASS" else "FAIL"
        cat = rc.load()
        rep = lg.book_report(cat, installed=rc.installed_two_formation_book(_installed(db_path)), conf=conf)
        blocked = [c["candidate_id"] for c in rep["candidates"] if not c["live_legal"]
                   and not all(r.startswith("DATASET_BLOCK") for r in c["reasons"])]
        g["installed_book_macros_coaching"] = "PASS" if not blocked else "FAIL (" + ", ".join(blocked) + ")"
        g["owner_confirmations_valid"] = "PASS" if conf.present else "FAIL"
    except Exception as exc:  # noqa: BLE001
        g["error"] = f"FAIL ({type(exc).__name__}: {exc})"
    g["all_pass"] = bool(g) and all(str(v).startswith("PASS") for k, v in g.items() if k != "all_pass")
    return g


def cmd_enable(args) -> int:
    from cfb_coach.madden.model.cli import open_explicit_madden_db
    from cfb_coach.madden.ravens_defense import confirmations as cf, model as rm
    path = _db_path(args)
    print(f"Using DB: {path}")
    art = Path(args.artifact).expanduser()
    gates = release_gates(art, path, cf.load(args.confirmations))
    print(_json(gates))
    if not gates["all_pass"]:
        print("REFUSED: release gates did not pass; defense mode unchanged", file=sys.stderr)
        return 2
    sha = rm.manifest(art)["model_sha256"]
    backup(path, "pre-enable")
    db = open_explicit_madden_db(path, read_only=False)
    try:
        md.set_state(db, enabled=True, artifact=str(art), model_sha256=sha)
    finally:
        db.close()
    print(f"ravens_defense_ml ENABLED with {art}. Revert: `ml ravens-defense disable` or play --previous-defense")
    return 0


def cmd_disable(args) -> int:
    from cfb_coach.madden.model.cli import open_explicit_madden_db
    path = _db_path(args)
    print(f"Using DB: {path}")
    backup(path, "pre-disable")
    db = open_explicit_madden_db(path, read_only=False)
    try:
        st = md.load_state(db)
        prev = st.get("previous") or {}
        if args.revert_artifact and prev.get("artifact"):
            md.set_state(db, enabled=bool(prev.get("enabled")), artifact=prev["artifact"], model_sha256=prev.get("model_sha256"))
            print(f"reverted to previous artifact {prev['artifact']} (enabled={bool(prev.get('enabled'))})")
            return 0
        md.set_state(db, enabled=False, artifact=None, model_sha256=None)
    finally:
        db.close()
    print("ravens_defense_ml DISABLED — defense uses the previous defensive caller.")
    return 0


def register(ml_sub: Any) -> None:
    p = ml_sub.add_parser("ravens-defense", help="Ravens two-formation defensive ML coordinator (Nickel 2-4 Wide / Dime 2-3 Odd)")
    s = p.add_subparsers(dest="ravens_cmd", required=True)

    def add(name, func, help_):
        q = s.add_parser(name, help=help_)
        q.add_argument("--db", default=None, help="Madden DB path; read-only actions need a mode-644 backup copy")
        q.add_argument("--confirmations", default=None, help="owner confirmation file (default ~/.cfb-coach/ravens_defense/)")
        q.set_defaults(func=func)
        return q

    q = add("audit", cmd_audit, "defensive data audit (read-only)")
    q.add_argument("--per-game", action="store_true")
    q = add("train", cmd_train, "train a versioned artifact on a read-only backup copy (never enables)")
    q.add_argument("--out", default=None, help="artifact root (default ~/.cfb-coach/models/ravens_defense)")
    q = add("evaluate", cmd_evaluate, "forward-only whole-game held-out evaluation (read-only)")
    q.add_argument("--out", default=None, help="write the full evaluation JSON here")
    q = add("verify-book", cmd_verify_book, "validate the 14 candidates against the installed book (fail closed)")
    q.add_argument("--json", action="store_true")
    q = add("shadow", cmd_shadow, "offline SHADOW call(s): one situation, or replay a session")
    q.add_argument("--artifact", default=None)
    q.add_argument("--session", default=None, help="replay this historical session (model refit on earlier games only)")
    q.add_argument("--opponent", "-o", default="shadow")
    q.add_argument("--down", type=int)
    q.add_argument("--distance", type=int)
    q.add_argument("--yardline", type=int, help="offense's yards to our goal line")
    q.add_argument("--shape", default=None, help="offense shape: trips/bunch/2x2/stacks/5wr_tight/4x1")
    q.add_argument("--prior-concepts", default=None, help="comma list of this game's earlier concepts")
    q.add_argument("--assume-setup", action="store_true", help="assume install/macros/coaching (display only)")
    q.add_argument("--top", type=int, default=14)
    q = add("postgame", cmd_postgame, "postgame challenger vs champion report (never promotes)")
    q.add_argument("--artifact", default=None, help="the champion artifact")
    q.add_argument("--out", default=None)
    add("status", cmd_status, "defense mode, artifact")
    add("coaching-audit", cmd_coaching_audit, "coaching-adjustment compatibility audit")
    q = add("users", cmd_users, "the 18 USER assignment rules and their verification status")
    q.add_argument("--json", action="store_true")
    q = add("enable", cmd_enable, "turn ravens_defense_ml on — refuses unless every release gate passes (backs up first)")
    q.add_argument("--artifact", required=True)
    q = add("disable", cmd_disable, "revert: previous defensive caller (backs up first)")
    q.add_argument("--revert-artifact", action="store_true", help="restore the previously enabled artifact instead")
