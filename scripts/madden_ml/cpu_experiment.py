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
    import shutil
    env = {**os.environ, "HOME": str(work / "home")}                    # keep every eval write away from ~/.cfb-coach
    (work / "home").mkdir(exist_ok=True)
    out, dk = work / "readiness", work / "defense_knowledge"
    for stale in (out, dk, work / "prepare_manifest.json"):             # never reuse a previous run's report or artifact
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
    rc = subprocess.call([sys.executable, str(ROOT / "scripts/madden_ml/offense_route_aware_readiness.py"), "--db", str(copy),
                          "--out", str(out), "--run-tests"], env=env)
    rc2 = subprocess.call([sys.executable, str(ROOT / "scripts/madden_ml/defense_knowledge_eval.py"), "heldout", "--db", str(copy),
                           "--out", str(dk)], env=env)
    rc3 = subprocess.call([sys.executable, str(ROOT / "scripts/madden_ml/football_priors_eval.py"), "--db", str(copy), "--out", str(work / "football_priors")], env=env)
    print(f"football-priors held-out comparison exit={rc3} (informational; result in {work / 'football_priors' / 'football_priors_eval.json'})")
    rep_p, art_p = out / "readiness_report.json", out / "offense_route_aware.json"
    problems = []
    if rc != 0:
        problems.append(f"readiness evaluation exited {rc}")
    if rc2 != 0:
        problems.append(f"defensive-knowledge held-out comparison exited {rc2}")
    if not rep_p.is_file() or not art_p.is_file():
        problems.append("readiness report or artifact was not produced")
    if problems:
        print("\nPREPARE FAILED (nothing is enabled, no manifest written):\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    rep = json.loads(rep_p.read_text())
    if not rep.get("real_data") or str(copy) not in str(rep.get("source", "")):
        print("PREPARE FAILED: the report was not produced from this experiment copy", file=sys.stderr)
        return 1
    manifest = {"schema": "cpu_experiment.prepare.v1", "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_db": str(src),
                "backup": str(backup), "backup_sha256": _sha(backup), "copy": str(copy), "copy_sha256": _sha(copy),
                "report": str(rep_p), "report_sha256": _sha(rep_p), "artifact": str(art_p), "artifact_sha256": _sha(art_p),
                "artifact_digest": rep["artifact"]["digest"], "art_bank": _art_bank(), "readiness_verdict": rep["readiness"]["verdict"]}
    (work / "prepare_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    gate_errs, _q = gate_errors(rep)
    if gate_errs:
        print("\nPREPARE INCOMPLETE: the evaluation ran, but the pilot cannot be enabled from it:\n  " + "\n  ".join(gate_errs), file=sys.stderr)
        return 2
    print(f"\nreadiness exit={rc}, defensive-knowledge held-out exit={rc2}; manifest: {work / 'prepare_manifest.json'}")
    print("next: python scripts/madden_ml/cpu_experiment.py summary --work", work)
    return 0


def _art_bank() -> dict:
    d = json.loads((ROOT / "cfb_coach/data/madden27/offense_play_art_features.json").read_text())
    return {"fingerprint": d.get("inventory_fingerprint"), **d["coverage"], "file_sha256": _sha(ROOT / "cfb_coach/data/madden27/offense_play_art_features.json")}


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
        print(f"football priors in the live score: {'ON' if rr.football_priors_on(db) else 'off'}")
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
REQUIRED_GATES = ("coverage_feedback_connected", "no_cross_game_or_future_leakage", "no_invalid_formations_plays_or_adjustments",
                  "adjustment_gate", "latency_within_150ms", "interface_displays_recommendation", "shadow_replay_clean",
                  "regression_and_integration_tests")


def gate_errors(rep: dict) -> tuple[list[str], dict]:
    gates = rep["readiness"]["gates"]
    errs = []
    for g in REQUIRED_GATES:                                           # present AND PASS; "not FAIL" is not enough
        if g not in gates:
            errs.append(f"required gate missing: {g}")
        elif gates[g]["status"] != "PASS":
            errs.append(f"required gate {g} is {gates[g]['status']}: {gates[g]['detail']}")
    q = gates.get("model_quality_vs_existing_coordinator", {})
    if q.get("status") == "FAIL":
        errs.append(f"model is explicitly WORSE than the existing coordinator on held-out games: {q.get('detail')}")
    elif q.get("status") not in ("PASS", "NOT EVALUATED"):          # NOT EVALUATED / non-significant lift may proceed as an owner-approved experiment
        errs.append("model-quality gate missing")
    return errs, q


def verify_prepared(work: Path, artifact: Path, live_db: Path) -> dict:
    """Fail closed: the report + artifact must be the untampered product of ONE successful `prepare` against the intended DB."""
    mp = work / "prepare_manifest.json"
    if not mp.is_file():
        raise SystemExit(f"no prepare manifest at {mp}; run `prepare` (a failed or missing prepare cannot be enabled)")
    m = json.loads(mp.read_text())
    errs = []
    if Path(m["artifact"]).resolve() != artifact.resolve():
        errs.append("artifact is not the one `prepare` produced")
    for key, path in (("artifact_sha256", m["artifact"]), ("report_sha256", m["report"]), ("copy_sha256", m["copy"])):
        if not Path(path).is_file() or _sha(Path(path)) != m[key]:
            errs.append(f"{Path(path).name} changed or is missing since prepare ({key})")
    if Path(m["source_db"]).resolve() != live_db.resolve():
        errs.append(f"prepare was run against {m['source_db']}, not {live_db}")
    if m["art_bank"] != _art_bank() or m["art_bank"].get("model_eligible") != 93 or m["art_bank"].get("total") != 93:
        errs.append("the 93-play art bank changed since prepare (or is not 93/93 eligible)")
    if errs:
        raise SystemExit("not enabling:\n  " + "\n  ".join(errs))
    rep = json.loads(Path(m["report"]).read_text())
    if not rep.get("real_data") or str(m["copy"]) not in str(rep.get("source", "")):
        errs.append("report was not produced from the prepared copy of your database")
    if not rep["artifact"].get("valid") or rep["artifact"]["digest"] != m["artifact_digest"]:
        errs.append("artifact invalid or digest mismatch")
    gate_errs, q = gate_errors(rep)
    errs += gate_errs
    if errs:
        raise SystemExit("not enabling:\n  " + "\n  ".join(errs))
    return {"manifest": m, "quality": f"{q['status']}: {q['detail']}"}


def cmd_enable(a) -> int:
    from cfb_coach.madden.model import experimental_live as xl
    from cfb_coach.madden.model import inference
    from cfb_coach.madden.model import offense_route_ranker as rr
    from cfb_coach.madden.model.schema import CoachingMode
    if not a.i_approve_cpu_experiment:
        raise SystemExit("refusing: pass --i-approve-cpu-experiment (this makes the route-aware model the primary offensive caller, CPU opponents only)")
    art_path = Path(a.artifact).expanduser()
    live = _live_db(a.db)
    info = verify_prepared(Path(a.work).expanduser(), art_path, live)
    art = rr.load_artifact(art_path)                                       # refuses stale / invalid
    if art.uses_defense_knowledge:
        raise SystemExit("this is a defense-knowledge artifact; use the default route-aware artifact for the first game")
    db = _open_db(live)
    try:
        if xl.resolve_artifact_path(db) is None:
            raise SystemExit("no experimental offense artifact installed. Run once: python -m cfb_coach ml experimental --retrain")
        pb = _playbook_check(db)
        if pb.get("error") or pb["present"] < pb["scored"] and not a.allow_partial_playbook:
            raise SystemExit(f"applied offense playbook has {pb.get('present')}/{pb['scored']} scored plays ({pb.get('error') or 'missing plays'}); "
                             "fix it or pass --allow-partial-playbook (the model never selects a play outside the applied book)")
        if not db.get_meta(ROLLBACK_META) or str(rr.resolve_mode(db)) == "experimental":
            # exact raw values (None = unset) so rollback restores precisely this state; never overwrite the ORIGINAL record when re-enabling
            db.set_meta(ROLLBACK_META, json.dumps({"ranker": db.get_meta(rr.META_MODE), "artifact": db.get_meta(rr.META_ARTIFACT),
                                                   "ml_mode": db.get_meta("ml_mode"), "priors": db.get_meta(rr.META_PRIORS), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        db.set_meta(rr.META_ARTIFACT, str(art_path))
        inference.set_mode(db, CoachingMode.EXPERIMENTAL)
        rr.set_mode(db, "route_aware_preview" if a.preview else "route_aware")
        db.set_meta(rr.META_PRIORS, "1" if a.football_priors else (db.get_meta(rr.META_PRIORS) or "0"))
        print(f"ENABLED: offense ranker = {'route_aware_preview' if a.preview else 'route_aware'} (CPU opponents only); artifact {art_path}")
        print(f"model quality gate: {info['quality']}")
        print("roll back with:  python scripts/madden_ml/cpu_experiment.py rollback   (exact previous values; add --safe for ranker-only)")
    finally:
        db.close()
    return 0


def cmd_rollback(a) -> int:
    from cfb_coach.madden.model import offense_route_ranker as rr
    db = _open_db(_live_db(a.db))
    try:
        raw = db.get_meta(ROLLBACK_META)
        if a.safe or not raw:
            rr.set_mode(db, "experimental")                               # the instant, always-safe state
            db.set_meta(rr.META_PRIORS, "0")
            print("rolled back (safe): offense ranker = experimental; ML mode and artifact left as they are"
                  + ("" if raw or a.safe else "  [no saved record found]"))
            return 0
        prev = json.loads(raw)
        db.set_meta(rr.META_MODE, prev["ranker"] if prev.get("ranker") is not None else "experimental")   # unset == experimental
        db.set_meta(rr.META_ARTIFACT, prev["artifact"] if prev.get("artifact") is not None else "")        # "" is ignored by the resolver
        db.set_meta("ml_mode", prev["ml_mode"] if prev.get("ml_mode") is not None else "heuristic")        # unset == heuristic
        db.set_meta(rr.META_PRIORS, prev.get("priors") if prev.get("priors") is not None else "0")
        db.set_meta(ROLLBACK_META, "")
        print(f"rolled back to the exact previous state: ranker={prev.get('ranker') or '(unset=experimental)'} artifact={prev.get('artifact') or '(unset)'} "
              f"ml_mode={prev.get('ml_mode') or '(unset=heuristic)'}")
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
        if name == "rollback":
            p.add_argument("--safe", action="store_true", help="only force the ranker back to experimental; leave everything else")
        if name == "enable":
            p.add_argument("--artifact", required=True)
            p.add_argument("--work", default="~/cfb_cpu_experiment")
            p.add_argument("--football-priors", action="store_true", help="make the structured football priors + learned cells part of the live score (opt-in; rollback restores)")
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
