"""Owner-machine procedure (WSL/laptop). Every step is read-only toward the primary data.

  discover      list candidate Madden databases and model artifacts (read-only)
  local-prep    1 protected backup+checksums  2 isolated testing-franchise copy (sqlite backup API)  3 data/model audit
                4 historical import + chronological evaluation  5 both-side game-prep  -> STOPS for physical installation
  local-finish  6 live-path benchmark  7 two-sided smoke  8 final preflight on the owner's attested books
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from cfb_coach.madden.unified import guard, history as H, pregame as PRE, preflight as PF
from cfb_coach.madden.unified.types import DEF, OFF

WSL_USERS = Path("/mnt/c/Users")


def candidate_paths() -> list[Path]:
    out: list[Path] = []
    for env in ("CFB_COACH_MADDEN_DB", "CFB_COACH_DB"):
        if os.environ.get(env):
            out.append(Path(os.environ[env]))
    home = guard.DEFAULT_HOME
    out += sorted(home.glob("*.db")) if home.exists() else []
    if WSL_USERS.exists():
        for u in WSL_USERS.iterdir():
            d = u / ".cfb-coach"
            if d.exists():
                out += sorted(d.glob("*.db"))
    out += sorted(Path.home().glob("**/madden27*.db"))[:20] if Path.home().exists() else []
    seen, uniq = set(), []
    for p in out:
        rp = p.expanduser().resolve()
        if rp.exists() and rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def model_artifacts() -> list[Path]:
    dirs = [guard.DEFAULT_HOME / "models", guard.DEFAULT_HOME]
    if WSL_USERS.exists():
        dirs += [u / ".cfb-coach" / "models" for u in WSL_USERS.iterdir()]
    out = []
    for d in dirs:
        if d.exists():
            out += [p for p in d.glob("*.json") if p.is_file()]
    return sorted({p.resolve() for p in out})


def discover() -> dict[str, Any]:
    dbs = []
    home = guard.DEFAULT_HOME.resolve()
    for p in candidate_paths():
        a = PF.audit_db(p)
        dbs.append({"path": str(p), "size_mb": round(p.stat().st_size / 1e6, 2), "primary_data_dir": home in p.parents or p.parent == home,
                    **{k: a.get(k) for k in ("row_counts", "dynasties", "verified_executions", "ok", "error")}})
    return {"databases": dbs, "model_artifacts": [{"path": str(p), "sha256": guard.sha256_file(p)[:16]} for p in model_artifacts()],
            "hint": "pass the database you want COPIED as --source-db; it is only ever opened read-only"}


def safe_copy_db(src: Path, dst: Path) -> dict[str, Any]:
    """SQLite online-backup from a read-only source connection; verifies row counts afterwards."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"{dst} exists; refusing to overwrite")
    before = guard.sha256_file(src)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(str(dst))
    try:
        s.backup(d)
        tables = [r[0] for r in s.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'")]
        counts_s = {t: s.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables}
        counts_d = {t: d.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables}
    finally:
        d.close(); s.close()
    return {"source": str(src), "copy": str(dst), "source_sha256_before": before, "source_sha256_after": guard.sha256_file(src),
            "source_unchanged": before == guard.sha256_file(src), "row_counts_match": counts_s == counts_d, "tables": len(tables)}


def local_prep(source_db: str | Path, workdir: str | Path, *, opponent: str = "cpu", extra_protected: Iterable[str | Path] = ()) -> dict[str, Any]:
    src = Path(source_db).expanduser().resolve()
    work = guard.assert_isolated(workdir, protected=[src])
    work.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {"workdir": str(work), "steps": {}}
    # 1. protected-asset backup + checksums (db via sqlite backup API; model files byte-copied)
    prot = [src, *[Path(x) for x in extra_protected], *model_artifacts()]
    manifest = {str(p): guard.sha256_file(p) for p in prot if p.exists()}
    bk = work / "protected_backup"
    bk.mkdir(exist_ok=True)
    backup = safe_copy_db(src, bk / src.name)
    for p in prot:
        if p.exists() and p != src and p.suffix != ".db":
            shutil.copy2(p, bk / p.name)
    (work / "protected_manifest.json").write_text(json.dumps({"checksums_before": manifest, "backup": backup}, indent=2))
    rep["steps"]["1_protected_backup"] = {"files_checksummed": len(manifest), "db_backup": backup}
    # 2. isolated testing-franchise copy
    test_db = work / "testing_franchise" / "madden27_test.db"
    rep["steps"]["2_isolated_test_db"] = safe_copy_db(src, test_db)
    # 3. data / model audit (on the COPY)
    rep["steps"]["3_audit"] = {"database": PF.audit_db(test_db), "model_artifacts": [str(p) for p in model_artifacts()]}
    # 4. historical import + chronological evaluation (on the COPY)
    hist = H.train_from_history(test_db, work / "history")
    rep["steps"]["4_history"] = {k: hist.get(k) for k in ("rows_seen", "usable_rows", "usable_games", "usable_by_side", "skipped", "decision", "reason")}
    model = Path(hist["candidate_artifact"]) if hist["decision"] == "CANDIDATE_SUPPORTED" else work / "history" / "cold_start_model.json"
    (work / "MODEL_CHOICE.json").write_text(json.dumps({"model": str(model), "decision": hist["decision"], "reason": hist["reason"],
                                                         "mode": "learned weights" if hist["decision"] == "CANDIDATE_SUPPORTED" else "UNCALIBRATED COLD-START PRIOR"}, indent=2))
    # 5. both-side game prep with the chosen model
    from cfb_coach.madden.unified.world_model import WorldModel
    wm = WorldModel.from_dict(json.loads(model.read_text()))
    props = {}
    prep = work / "prep"
    prep.mkdir(exist_ok=True)
    for side in (OFF, DEF):
        props[side] = PRE.build_proposal(side, opponent_id=opponent, opponent_class="cpu" if opponent.startswith("cpu") else "human", model=wm)
        (prep / f"proposal_{side}.json").write_text(json.dumps(props[side], indent=2))
    (prep / "INSTALL_GUIDE.md").write_text(PRE.render_install_guide(props, opponent))
    rep["steps"]["5_game_prep"] = {s: {"formations": p["n_formations"], "plays": p["n_plays"], "proposed_fingerprint": p["proposed_fingerprint"]} for s, p in props.items()}
    rep["stop"] = ("STOPPED: books are PROPOSALS. Build them in Madden using " + str(prep / "INSTALL_GUIDE.md") + ", check them in the editor, then run "
                   "confirm-install for each side and `local-finish`.")
    (work / "local_prep_report.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


def local_finish(workdir: str | Path, confirmed_books: str | Path, *, calls: int = 100, adjustment_confirmations: str | None = None) -> dict[str, Any]:
    work = Path(workdir).expanduser().resolve()
    choice = json.loads((work / "MODEL_CHOICE.json").read_text())
    rep = PF.preflight(db=str(work / "testing_franchise" / "madden27_test.db"), confirmed_books=str(confirmed_books), model=choice["model"],
                       history_report=str(work / "history" / "history_report.json"), adjustment_confirmations=adjustment_confirmations, calls=calls,
                       extra_protected=[str(work / "protected_backup" / p.name) for p in (work / "protected_backup").glob("*")], workdir=str(work / "preflight_run"))
    (work / "final_preflight.json").write_text(json.dumps(rep, indent=2, default=str))
    (work / "final_preflight.md").write_text(PF.render_report(rep))
    store = work / "game" / "unified.db"
    rep["launch"] = (f"python -m cfb_coach.madden.unified game --opponent cpu --confirmed-books {confirmed_books} --store {store} --model {choice['model']}"
                     + (f" --adjustment-confirmations {adjustment_confirmations}" if adjustment_confirmations else ""))
    rep["postgame"] = f"python -m cfb_coach.madden.unified postgame --store {store} --out {work / 'postgame_001'} --champion {choice['model']}"
    return rep
