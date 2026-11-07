#!/usr/bin/env python3
"""Evaluate the Madden 27 defensive knowledge features. Read-only; never touches a live database or an installed model.

    python scripts/madden_ml/defense_knowledge_eval.py audit     --out OUT_DIR
    python scripts/madden_ml/defense_knowledge_eval.py selfcheck --out OUT_DIR [--games 16 --snaps 40 --boot 200]
    python scripts/madden_ml/defense_knowledge_eval.py heldout   --db COPY_OF_MADDEN.db --out OUT_DIR
    python scripts/madden_ml/defense_knowledge_eval.py latency   --out OUT_DIR
    python scripts/madden_ml/defense_knowledge_eval.py train     --db COPY_OF_MADDEN.db --out OUT_DIR
    python scripts/madden_ml/defense_knowledge_eval.py defense-offline --db COPY_OF_MADDEN.db --current LIVE_DEFENSE_ARTIFACT.json --out OUT_DIR

* ``audit``      deterministic name-recognition table (old recognizer vs catalog) and the verification summary.
* ``selfcheck``  SYNTHETIC planted-signal / null / label-permutation worlds. Proves plumbing and the absence of invented signal. It is
                 NOT evidence about real Madden games.
* ``heldout``    whole-game chronological comparison on a COPY of the game database: the offense structural model with vs without the
                 knowledge block, paired game-clustered bootstrap, explicit GO / NO-GO. Without a database this step is NOT EVALUATED.
* ``train``      fits the knowledge-block offense artifact (and the default one, for comparison) on a COPY of the database and writes them
                 to OUT_DIR. It never installs or enables anything.
* ``defense-offline``  our OWN defensive calls: baseline logistic vs the same model plus catalog candidate features, same leave-one-game-out
                 protocol, stricter gate. An offline research artifact whose schema the live defensive caller rejects.
* ``latency``    wall time to score all 93 installed plays against one opponent mix (150 ms live budget).

Every command runs with HOME pointed at a temporary directory, so no training step can create or modify files under the real
~/.cfb-coach. ``heldout`` opens the database read-only (and refuses a path inside ~/.cfb-coach).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@contextlib.contextmanager
def isolated_home():
    old = os.environ.get("HOME")
    with tempfile.TemporaryDirectory(prefix="dk-eval-home-") as tmp:
        os.environ["HOME"] = tmp
        try:
            yield Path(tmp)
        finally:
            if old is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old


def _write(out: Path, name: str, obj) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return p


def cmd_audit(args) -> int:
    from cfb_coach.madden.model import defense_knowledge_eval as ke
    rep = ke.recognition_audit()
    _write(Path(args.out), "recognition_audit.json", rep)
    i, e = rep["installed"], rep["external"]
    print(f"installed: {i['entries']} entries; old recognizer named a coverage for {i['old_recognizer_named_coverage']}; "
          f"catalog resolved {i['catalog_resolved']} ({i['resolution_types']})")
    print(f"external : {e['entries']} entries; old recognizer {e['old_recognizer_named_coverage']}; catalog resolved {e['catalog_resolved']}")
    print(f"legacy disagreements: {len(rep['legacy_disagreements'])}; unresolved: {len(rep['unresolved'])}")
    print("verification:", rep["verification"])
    return 0 if not rep["legacy_disagreements"] and not rep["unresolved"] else 1


def cmd_selfcheck(args) -> int:
    from cfb_coach.madden.model import defense_knowledge_eval as ke
    mk = lambda **kw: ke.make_pressure_world_rows(n_games=args.games, snaps=args.snaps, seed=args.seed, strength=args.strength, **kw)
    worlds = {
        # outcome depends on the call the opponent REALLY ran; the knowledge model sees what the EXISTING form stored for it
        "A_planted_signal_TODAYS_CAPTURE": (mk(through_form=True), "SYNTHETIC planted pressure effect; labels pass through the real post-play form"),
        # the same world if the exact call name were stored: measures what lossless capture would add
        "B_planted_signal_LOSSLESS_CAPTURE_counterfactual": (mk(through_form=False), "SYNTHETIC planted effect; exact call names stored (counterfactual)"),
        "C_null_no_effect": (mk(through_form=True, null=True), "SYNTHETIC: outcomes independent of the defense"),
        "D_planted_labels_shuffled_globally_NULL_CONTROL": (
            ke.permute_coverage_labels(mk(through_form=True), seed=args.seed + 1, mode="global"),
            "SYNTHETIC planted effect, labels shuffled across all rows (valid null control)"),
        "E_planted_labels_shuffled_within_game_DIAGNOSTIC": (
            ke.permute_coverage_labels(mk(through_form=True), seed=args.seed + 1, mode="within_game"),
            "SYNTHETIC planted effect, shuffled inside each game (keeps each game's pressure tendency; NOT a null control)"),
    }
    out = {}
    only = {x.strip().upper() for x in (args.only or "").split(",") if x.strip()}
    for name, (rows, src) in worlds.items():
        if only and name[0].upper() not in only:
            continue
        out[name] = ke.evaluate_rows(rows, source=src, n_boot=args.boot)
        cmp = out[name].get("knowledge_vs_baseline", {})
        d, lo, hi = cmp.get("mean_diff"), cmp.get("lo"), cmp.get("hi")
        print(f"{name:52s} folds={out[name].get('folds')} diff={d if d is None else round(d, 4)} CI=({lo if lo is None else round(lo, 4)},{hi if hi is None else round(hi, 4)})"
              f" | {out[name].get('verdict')}", flush=True)
    _write(Path(args.out), "selfcheck.json", out)
    return 0


def cmd_heldout(args) -> int:
    db = Path(args.db).expanduser().resolve()
    if ".cfb-coach" in db.parts:
        print("REFUSED: pass a COPY of the database outside ~/.cfb-coach")
        return 2
    if not db.is_file():
        print(f"NOT EVALUATED: database copy not found: {db}")
        _write(Path(args.out), "heldout.json", {"verdict": "NOT EVALUATED: no database copy available", "db": str(db)})
        return 3
    from cfb_coach.madden.model import defense_knowledge_eval as ke
    from cfb_coach.madden.model import heldout_eval as he
    data = he.load_eval_data(db)
    rep = ke.evaluate_rows(data.rows, data.game_order, source=f"copy of {db.name}", n_boot=args.boot)
    _write(Path(args.out), "heldout.json", rep)
    print(rep.get("verdict"))
    return 0


def _copy_guard(path: str) -> Path | None:
    db = Path(path).expanduser().resolve()
    if ".cfb-coach" in db.parts:
        print("REFUSED: pass a COPY of the database outside ~/.cfb-coach")
        return None
    return db


def cmd_train(args) -> int:
    db = _copy_guard(args.db)
    if db is None:
        return 2
    if not db.is_file():
        print(f"NOT TRAINED: database copy not found: {db}")
        return 3
    from cfb_coach.madden.model import heldout_eval as he
    from cfb_coach.madden.model import offense_structural_model as sm
    data = he.load_eval_data(db)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, dk in (("offense_route_aware.json", False), ("offense_route_aware_defense_knowledge.json", True)):
        art = sm.train(data.rows, defense_knowledge=dk)
        sm.save_artifact(art, out / name)
        errs = sm.validate_artifact(art)
        summary[name] = {"residual_active": art.residual_active, "n_train": art.n_train, "games": art.trained_games, "valid": not errs,
                         "errors": errs, "selection": art.selection.get("decision"), "digest": sm.artifact_digest(art)}
        print(name, summary[name])
    _write(out, "train_summary.json", summary)
    print("Nothing was installed or enabled. Evaluate with the `heldout` command before any preview.")
    return 0


def cmd_defense_offline(args) -> int:
    db = _copy_guard(args.db)
    if db is None:
        return 2
    if not db.is_file() or not Path(args.current).is_file():
        print("NOT EVALUATED: database copy or current defense artifact not found")
        _write(Path(args.out), "defense_offline.json", {"verdict": "NOT EVALUATED: inputs not found"})
        return 3
    import sqlite3
    from cfb_coach.madden.model import defense_knowledge_train as kt
    from cfb_coach.madden.model import defense_logistic_train as dlt
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as conn:
        rows = dlt.collect_rows(conn)
    labels = dlt.label_calls(rows, current)
    admitted = [r for r in rows if r.admitted]
    rep = kt.compare_leave_one_game_out(admitted, labels, current)
    rep["rows_admitted"], rep["rows_total"] = len(admitted), len(rows)
    _write(Path(args.out), "defense_offline.json", rep)
    print(rep["gate"]["result"], rep["gate"]["checks"], rep["gate"]["interval"])
    return 0


def cmd_latency(args) -> int:
    from cfb_coach.madden.model import defense_knowledge_eval as ke
    from cfb_coach.madden.model import offense_assignments as oa
    from cfb_coach.madden.model import offense_structural_model as sm
    rows = ke.make_pressure_world_rows(n_games=10, snaps=25, seed=5)
    book = [(p["formation"], p["play"]) for p in oa.load_effective()["plays"]]
    mix = {"dk:cover_3_sky": 0.3, "dk:fire_zone_3": 0.2, "dk:cover_2_man": 0.2, "dk:zero_blitz": 0.1, "dk:cover_4_quarters": 0.2}
    res = {}
    for name, dk in (("default_artifact", False), ("defense_knowledge_artifact", True)):
        art = sm.train(rows, penalty_scale=1.0, defense_knowledge=dk)
        m = mix if dk else {"cover3": 0.5, "cover_2_man": 0.2, "cover_4_quarters": 0.3}
        res[name] = ke.latency_report(art, book, m)
        print(name, res[name])
    _write(Path(args.out), "latency.json", res)
    return 0 if all(v["within_budget"] for v in res.values()) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("audit", cmd_audit), ("selfcheck", cmd_selfcheck), ("heldout", cmd_heldout), ("latency", cmd_latency),
                     ("train", cmd_train), ("defense-offline", cmd_defense_offline)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
        if name == "selfcheck":
            p.add_argument("--games", type=int, default=16)
            p.add_argument("--snaps", type=int, default=40)
            p.add_argument("--seed", type=int, default=7)
            p.add_argument("--strength", type=float, default=2.0, help="planted effect size (probability change per unit of shallow-route share)")
            p.add_argument("--boot", type=int, default=200)
            p.add_argument("--only", default="", help="comma-separated world letters to run (A..E); default all")
        if name in ("heldout", "train", "defense-offline"):
            p.add_argument("--db", required=True)
        if name == "heldout":
            p.add_argument("--boot", type=int, default=500)
        if name == "defense-offline":
            p.add_argument("--current", required=True, help="the live defensive artifact JSON (read only; supplies the vocabulary)")
    args = ap.parse_args(argv)
    with isolated_home():
        return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
