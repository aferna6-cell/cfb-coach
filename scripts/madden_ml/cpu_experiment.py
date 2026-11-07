#!/usr/bin/env python3
"""Explicit, reversible CPU-only experiment driver for the route-aware offense (PR #113 + #115 + #117).

  prepare   back up your live DB, work on a COPY, run the readiness evaluation + defensive-knowledge held-out comparison
  enable    gate-check, then make route_aware the primary offensive ranker (explicit; records how to roll back)
  status    what is active, the artifact, playbook coverage of the 93 scored plays, fallbacks/latency/adjustments so far
  rollback  restore the previous ranker / ML mode (instant, next call)
  report    after the game: selection, efficiency, diversity, adaptation, man/zone/blitz splits, defense, latency, fallbacks,
            and (from the shadow log) route-aware vs existing-coordinator picks

Nothing here edits your gameplay form. `prepare` never writes to your live DB. `enable` writes only the two ranker/mode meta
values (and a rollback record). The existing coordinator remains the automatic fallback on any technical failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LIVE_DIR = Path.home() / ".cfb-coach"
ROLLBACK_META = "cpu_experiment_rollback"


def _live_db(arg: str | None) -> Path:
    p = Path(arg or os.environ.get("CFB_COACH_MADDEN_DB") or LIVE_DIR / "madden27.db").expanduser()
    if not p.is_file():
        raise SystemExit(f"database not found: {p}\n(pass --db PATH; `python -m cfb_coach ml find-db` prints the one the app uses)")
    return p


def _open_db(path: Path, *, read_only: bool = False):
    from cfb_coach.madden.model.cli import open_explicit_madden_db  # same opener the `ml` commands use
    return open_explicit_madden_db(path, read_only=read_only)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------- prepare
def cmd_prepare(a) -> int:
    from cfb_coach.madden.model.data_layer import backup_database
    src = _live_db(a.db)
    work = Path(a.work).expanduser().resolve()
    if work == LIVE_DIR.resolve() or LIVE_DIR.resolve() in work.parents:
        raise SystemExit("--work must be outside ~/.cfb-coach")
    (work / "backup").mkdir(parents=True, exist_ok=True)
    backup = backup_database(src, work / "backup")                      # sqlite backup API; source opened read-only
    copy = work / "experiment_copy.db"
    con_b = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    con_c = sqlite3.connect(copy)
    con_b.backup(con_c)
    con_b.close()
    con_c.close()
    print(f"live DB (not modified): {src}\nbackup: {backup}  sha256={_sha(backup)[:16]}…\nexperiment copy: {copy}")
    if a.backup_only:
        return 0
    env = {**os.environ, "HOME": str(work / "home")}                    # keep every eval write away from ~/.cfb-coach
    (work / "home").mkdir(exist_ok=True)
    out = work / "readiness"
    rc = subprocess.call([sys.executable, str(ROOT / "scripts/madden_ml/offense_route_aware_readiness.py"), "--db", str(copy),
                          "--out", str(out), "--run-tests"], env=env)
    dk = work / "defense_knowledge"
    rc2 = subprocess.call([sys.executable, str(ROOT / "scripts/madden_ml/defense_knowledge_eval.py"), "heldout", "--db", str(copy),
                           "--out", str(dk)], env=env)
    print(f"\nreadiness exit={rc}, defensive-knowledge held-out exit={rc2}\nartifact for `enable`: {out / 'offense_route_aware.json'}")
    print("next: python scripts/madden_ml/cpu_experiment.py summary --work", work)
    return 0


def cmd_summary(a) -> int:
    work = Path(a.work).expanduser()
    rep = json.loads((work / "readiness" / "readiness_report.json").read_text())
    r = rep["readiness"]
    ev = rep.get("evaluation") or {}
    print(f"VERDICT: {r['verdict']} - {r['reason']}")
    for k, v in r["gates"].items():
        print(f"  [{v['status']:>13}] {k}: {v['detail']}")
    d = rep.get("data_audit", {})
    print(f"verified offense games: {(ev.get('data') or {}).get('verified_offense_games')}  usable snaps: {d.get('usable_for_supervised_training')}")
    art = rep["artifact"]
    print(f"artifact: {art['path']}  valid={art['valid']} games={art['trained_games']} rows={art['verified_rows']} "
          f"residual_active={art['residual_active']} features={art['n_features']}")
    cov = json.loads((ROOT / "cfb_coach/data/madden27/offense_play_art_features.json").read_text())["coverage"]
    print(f"play-art records model-eligible: {cov['model_eligible']}/{cov['total']}  (all must be 93)")
    sr = rep.get("shadow_replay") or {}
    if sr:
        print(f"shadow replay: {sr.get('records')} decisions, illegal={sr.get('illegal')}, errors/fallbacks={sr.get('errors_or_fallbacks')}, "
              f"max latency {sr.get('latency_ms_max')} ms")
    dkj = work / "defense_knowledge" / "heldout.json"
    if dkj.is_file():
        print("defensive-knowledge held-out:", json.dumps(json.load(open(dkj)).get("verdict", "see file"))[:300])
    return 0


# --------------------------------------------------------------------------------------------------------------- common
def _playbook_check(db) -> dict:
    """Which of the 93 scored plays does the CURRENTLY APPLIED offense playbook contain?"""
    from cfb_coach.madden import playbook
    from cfb_coach.madden.model import offense_assignments as oa
    snap = json.loads(oa.SNAPSHOT_PATH.read_text(encoding="utf-8"))["sides"]["offense"]["formations"]
    scored = {(f, p) for f, plays in snap.items() for p in plays}
    try:
        books = playbook.active_books(db, ("offense",))["offense"]
        book = books.get("formations") if isinstance(books, dict) else None
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "scored": len(scored), "present": 0, "missing": sorted(scored)}
    have = {(f, p) for f, plays in (book or {}).items() for p in plays}
    return {"scored": len(scored), "present": len(scored & have), "missing": sorted(scored - have),
            "extra_in_book_not_scored": sorted(have - scored), "book_name": books.get("name") if isinstance(books, dict) else None}


def _shadow_summary(since: float | None = None) -> dict:
    from cfb_coach.madden.model import offense_route_ranker as rr
    p = rr.shadow_log_path()
    recs = []
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if since is None or r.get("ts", "") >= time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(since)):
                recs.append(r)
    lat = sorted(r.get("latency_ms") or 0 for r in recs)
    adj = [r for r in recs if r.get("adjustment") not in (None, "", "NO ADJUSTMENT", {})]
    return {"path": str(p), "records": len(recs), "applied": sum(bool(r.get("applied")) for r in recs),
            "fallbacks": [{"snap_seq": r.get("snap_seq"), "reason": r.get("fallback_reason")} for r in recs if r.get("fallback_reason")],
            "latency_ms_p50": lat[len(lat) // 2] if lat else None, "latency_ms_p95": lat[int(len(lat) * .95) - 1] if len(lat) > 1 else (lat[0] if lat else None),
            "latency_ms_max": lat[-1] if lat else None, "over_150ms": sum(x > 150 for x in lat),
            "agree_with_existing_coordinator": sum(bool(r.get("agree")) for r in recs),
            "custom_adjustments": [{"snap_seq": r.get("snap_seq"), "adjustment": r.get("adjustment")} for r in adj], "records_raw": recs}


def cmd_status(a) -> int:
    from cfb_coach.madden.model import inference
    from cfb_coach.madden.model import offense_route_ranker as rr
    db = _open_db(_live_db(a.db), read_only=True)
    try:
        env = os.environ.get(rr.ENV_MODE)
        path = rr.resolve_artifact_path(db)
        print(f"ML mode (must be EXPERIMENTAL for route_aware): {inference.resolve_mode(db) if hasattr(inference, 'resolve_mode') else '?'}")
        print(f"offense ranker stored: {db.get_meta(rr.META_MODE) or 'experimental'}   effective in this shell: {rr.resolve_mode(db)}"
              + (f"   [env {rr.ENV_MODE}={env} overrides]" if env else ""))
        print(f"route-aware artifact: {path}")
        if path:
            art = rr.load_artifact(path)                                  # raises if stale / invalid
            from cfb_coach.madden.model import offense_structural_model as sm
            print(f"  loads OK: games={art.trained_games} rows={art.n_train} residual_active={art.residual_active} "
                  f"features={len(art.names)} digest={sm.artifact_digest(art)[:12]} knowledge={art.uses_defense_knowledge}")
        pb = _playbook_check(db)
        print(f"applied offense playbook contains {pb['present']}/{pb['scored']} of the plays the model scores"
              + (f"  [{pb.get('book_name')}]" if pb.get("book_name") else ""))
        if pb.get("missing"):
            print("  MISSING (the model can never pick these here):", ", ".join(f"{f} / {p}" for f, p in pb["missing"][:12]), "…" if len(pb["missing"]) > 12 else "")
        sh = _shadow_summary()
        print(f"shadow log {sh['path']}: {sh['records']} decisions, applied={sh['applied']}, fallbacks={len(sh['fallbacks'])}, "
              f"p95={sh['latency_ms_p95']} ms, custom adjustments={len(sh['custom_adjustments'])}")
        for f in sh["fallbacks"][-5:]:
            print("  fallback:", f)
    finally:
        db.close()
    return 0


# --------------------------------------------------------------------------------------------------------- enable / rollback
def cmd_enable(a) -> int:
    from cfb_coach.madden.model import experimental_live as xl
    from cfb_coach.madden.model import inference
    from cfb_coach.madden.model import offense_route_ranker as rr
    from cfb_coach.madden.model import offense_structural_model as sm
    from cfb_coach.madden.model.schema import CoachingMode
    if not a.i_approve_cpu_experiment:
        raise SystemExit("refusing: pass --i-approve-cpu-experiment (this makes the route-aware model the primary offensive caller, CPU opponents only)")
    art_path = Path(a.artifact).expanduser()
    rep_path = Path(a.report).expanduser() if a.report else art_path.parent / "readiness_report.json"
    if not rep_path.is_file():
        raise SystemExit(f"no readiness report beside the artifact ({rep_path}); run `prepare` first")
    rep = json.loads(rep_path.read_text())
    if not rep.get("real_data"):
        raise SystemExit("that readiness report is a synthetic self-check, not your data")
    bad = []
    gates = rep["readiness"]["gates"]
    for k, v in gates.items():
        if v["status"] == "FAIL" or (k == "model_quality_vs_existing_coordinator" and v["status"] == "FAIL"):
            bad.append(f"{k}: {v['detail']}")
    if bad:
        raise SystemExit("gate failure, not enabling:\n  " + "\n  ".join(bad))
    art = rr.load_artifact(art_path)                                       # refuses stale / invalid
    if art.uses_defense_knowledge:
        raise SystemExit("this is a defense-knowledge artifact; use the default route-aware artifact for the first game")
    db = _open_db(_live_db(a.db))
    try:
        if xl.resolve_artifact_path(db) is None:
            raise SystemExit("no experimental offense artifact installed. Run once: python -m cfb_coach ml experimental --retrain")
        pb = _playbook_check(db)
        if pb.get("error") or pb["present"] < pb["scored"] and not a.allow_partial_playbook:
            raise SystemExit(f"applied offense playbook has {pb.get('present')}/{pb['scored']} scored plays ({pb.get('error') or 'missing plays'}); "
                             "fix it or pass --allow-partial-playbook (the model never selects a play outside the applied book)")
        prev = {"ranker": db.get_meta(rr.META_MODE) or "experimental", "artifact": db.get_meta(rr.META_ARTIFACT),
                "ml_mode": str(inference.resolve_mode(db)) if hasattr(inference, "resolve_mode") else None,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        db.set_meta(ROLLBACK_META, json.dumps(prev))
        db.set_meta(rr.META_ARTIFACT, str(art_path))
        inference.set_mode(db, CoachingMode.EXPERIMENTAL)
        rr.set_mode(db, "route_aware_preview" if a.preview else "route_aware")
        print(f"ENABLED: offense ranker = {'route_aware_preview' if a.preview else 'route_aware'} (CPU opponents only); artifact {art_path}")
        print("previous settings saved; roll back with:  python scripts/madden_ml/cpu_experiment.py rollback")
    finally:
        db.close()
    return 0


def cmd_rollback(a) -> int:
    from cfb_coach.madden.model import inference
    from cfb_coach.madden.model import offense_route_ranker as rr
    from cfb_coach.madden.model.schema import CoachingMode
    db = _open_db(_live_db(a.db))
    try:
        raw = db.get_meta(ROLLBACK_META)
        prev = json.loads(raw) if raw else {"ranker": "experimental", "ml_mode": None}
        rr.set_mode(db, "experimental")                                   # the instant, mandatory part
        if prev.get("ml_mode") and "HEURISTIC" in str(prev["ml_mode"]).upper():
            inference.set_mode(db, CoachingMode.HEURISTIC)
        print(f"rolled back: offense ranker = experimental; ML mode restored to {prev.get('ml_mode') or '(unchanged)'}")
        print("(env override, if you used one: unset CFB_COACH_OFFENSE_RANKER)")
    finally:
        db.close()
    return 0


# ------------------------------------------------------------------------------------------------------------------ report
def _cov_class(label: str | None) -> str:
    s = (label or "").lower()
    if not s:
        return "unreported"
    if any(k in s for k in ("blitz", "pressure", "fire", "zero", "cover 0", "sim", "dog")):
        return "blitz/pressure"
    if "man" in s or "cover 1" in s or "cover 0" in s:
        return "man"
    return "zone"


def cmd_report(a) -> int:
    db = _open_db(_live_db(a.db), read_only=True)
    try:
        db.conn.row_factory = sqlite3.Row
        sess = a.session or (db.conn.execute("SELECT session_id FROM snaps WHERE session_id IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone() or [None])[0]
        if not sess:
            raise SystemExit("no logged session found")
        rows = [dict(r) for r in db.conn.execute("SELECT * FROM snaps WHERE session_id=? ORDER BY COALESCE(snap_seq, id), id", (sess,))]
        off = [r for r in rows if r["side"] == "offense" and r.get("result")]
        dfn = [r for r in rows if r["side"] == "defense" and r.get("result")]

        def yards(r):
            try:
                return float(r.get("yards")) if r.get("yards") is not None else float(str(r["result"]).replace("+", ""))
            except (TypeError, ValueError):
                return None

        def eff(rs):
            ys = [y for y in map(yards, rs) if y is not None]
            return {"n": len(rs), "yards_per_play": round(sum(ys) / len(ys), 2) if ys else None,
                    "success_rate": round(sum(bool(r.get("success")) if "success" in r else (y or 0) >= 4 for r, y in zip(rs, map(yards, rs))) / len(rs), 3) if rs else None}
        plays = Counter((r.get("formation"), r.get("play")) for r in off)
        by_cov = defaultdict(list)
        for r in off:
            by_cov[_cov_class(r.get("coverage_typed") or r.get("coverage_seen"))].append(r)
        sh = _shadow_summary()
        illegal = []
        try:
            from cfb_coach.madden.model import offense_assignments as oa
            snap = json.loads(oa.SNAPSHOT_PATH.read_text(encoding="utf-8"))["sides"]["offense"]["formations"]
            legal = {(f, p) for f, ps in snap.items() for p in ps}
            illegal = [(r["formation"], r["play"]) for r in off if (r["formation"], r["play"]) not in legal]
        except Exception:  # noqa: BLE001
            pass
        out = {
            "session": sess, "offense_snaps": len(off), "defense_snaps": len(dfn),
            "offense_efficiency": eff(off), "distinct_plays": len(plays), "diversity_ratio": round(len(plays) / max(1, len(off)), 3),
            "top_repeats": plays.most_common(5),
            "by_reported_coverage_class": {k: eff(v) for k, v in by_cov.items()},
            "exact_defensive_calls_logged": Counter(r.get("coverage_typed") for r in off if r.get("coverage_typed")).most_common(),
            "exact_offensive_calls_logged_on_defense": Counter(r.get("concept_typed") for r in dfn if r.get("concept_typed")).most_common(),
            "defense_results": {"snaps": len(dfn), "yards_allowed_per_play": eff(dfn)["yards_per_play"]},
            "executed_attested": Counter(r.get("executed_verification") for r in off),
            "illegal_recommendations": illegal,
            "model": {k: v for k, v in sh.items() if k != "records_raw"},
            "route_aware_vs_existing_coordinator": {
                "decisions": sh["records"], "same_top_pick": sh["agree_with_existing_coordinator"],
                "differences_logged": [{"snap_seq": r.get("snap_seq"), "route_aware": (r.get("route_aware_top") or [None])[0],
                                        "existing": (r.get("current_ranker_top") or [None])[0]}
                                       for r in sh["records_raw"] if not r.get("agree")][:25]},
            "adaptation": [{"snap_seq": r.get("snap_seq"), "expected_coverage_top": (r.get("coverage") or {}).get("top"),
                            "named_n": (r.get("coverage") or {}).get("named_n")} for r in sh["records_raw"]],
            "caveat": "one game: descriptive only. Differences between route-aware and existing picks are logged decisions, not outcomes of unplayed calls.",
        }
        text = json.dumps(out, indent=1, default=str)
        dest = Path(a.out).expanduser() if a.out else Path.cwd() / f"cpu_game_report_{sess}.json"
        dest.write_text(text, encoding="utf-8")
        print(text)
        print(f"\nwritten: {dest}")
    finally:
        db.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("prepare", cmd_prepare), ("summary", cmd_summary), ("status", cmd_status), ("enable", cmd_enable),
                     ("rollback", cmd_rollback), ("report", cmd_report)):
        p = sub.add_parser(name)
        p.add_argument("--db", help="live Madden DB (default ~/.cfb-coach/madden27.db or $CFB_COACH_MADDEN_DB)")
        p.set_defaults(fn=fn)
        if name in ("prepare", "summary"):
            p.add_argument("--work", default="~/cfb_cpu_experiment")
            if name == "prepare":
                p.add_argument("--backup-only", action="store_true", help="make the backup + copy and stop")
        if name == "enable":
            p.add_argument("--artifact", required=True)
            p.add_argument("--report")
            p.add_argument("--preview", action="store_true", help="route_aware_preview: show the model's pick but keep the existing coordinator calling")
            p.add_argument("--allow-partial-playbook", action="store_true")
            p.add_argument("--i-approve-cpu-experiment", action="store_true")
        if name == "report":
            p.add_argument("--session")
            p.add_argument("--out")
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
