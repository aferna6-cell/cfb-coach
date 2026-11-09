"""READ-ONLY import of the owner's verified historical Madden executions into unified observations.

* The database is opened with SQLite ``mode=ro`` through a tiny shim; nothing is migrated, created or written.
* Eligibility comes from the repo's existing ``dataset.build_rows`` rules: a row is a verified execution only when the
  owner attested ``executed_status == identified`` with ``verified``. Recommendations are NEVER promoted.
* Nothing is fabricated: a row needs a catalogued play we really ran, a classifiable opponent call from what was
  actually entered, a known situation and a known yardage. Anything else is counted and skipped, with the reason.
* Rows whose recommendation carried a Custom Adjustment/macro are skipped by default: whether that adjustment was
  actually performed is not verified, so attributing the result to the stock play would be a guess.
"""
from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cfb_coach.madden.model import dataset as DS
from cfb_coach.madden.unified import knowledge as K
from cfb_coach.madden.unified.inputs import InputError, _catalog_pairs, parse_opponent_call, parse_result
from cfb_coach.madden.unified.types import (DEF, EXEC_CONFIRMED, OFF, FootballAction, GameState, OpponentAction, SnapObservation)

IMPORT_VERSION = "unified_history_import.v1"


class _RO:
    """Minimal stand-in for CoachDB: just a read-only ``conn`` and a ``path``."""

    def __init__(self, path: str | Path) -> None:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        self.path = str(p)
        self.conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()


def _opp_action(row: dict[str, Any], opp_side: str) -> OpponentAction | None:
    keys = ("coverage_typed", "coverage_seen") if opp_side == DEF else ("concept_seen",)
    for k in keys:
        v = row.get(k)
        if v and str(v).strip():
            try:
                return parse_opponent_call(str(v), opp_side)
            except InputError:
                continue
    return None


def import_verified(db_path: str | Path, *, include_macro_rows: bool = False) -> tuple[list[list[SnapObservation]], dict[str, Any]]:
    db = _RO(db_path)
    try:
        rows = DS.build_rows(db=db)
    finally:
        db.close()
    skip: Counter = Counter()
    per_game: dict[str, list[tuple[int, SnapObservation]]] = defaultdict(list)
    elig = Counter(r.get("eligibility") for r in rows)
    verification = Counter()
    for idx, r in enumerate(rows):
        if r.get("eligibility") != DS.ELIGIBILITY_VERIFIED_EXECUTION:
            skip["not_a_verified_execution(" + str(r.get("eligibility")) + ")"] += 1
            continue
        side = OFF if r["side"] == "offense" else DEF
        if not r.get("game_id"):
            skip["no_game_identity"] += 1
            continue
        if not include_macro_rows and str(r.get("recommended_macro") or "none").lower() not in ("none", "", "no adjustment"):
            skip["adjustment_execution_unverified"] += 1
            continue
        f, p = r.get("action_formation"), r.get("action_play")
        if not f or not p or (f, p) not in _catalog_pairs(side):
            skip["executed_play_not_in_catalog"] += 1
            continue
        if None in (r.get("down"), r.get("distance"), r.get("yardline")) or r.get("yards") is None:
            skip["situation_or_yards_unknown"] += 1
            continue
        yl = int(r["yardline"]) if side == OFF else 100 - int(r["yardline"])
        try:
            st = GameState(str(r["game_id"]), 1, side, str(r.get("opponent_id") or "unknown"), int(r["down"]), int(r["distance"]),
                           max(1, min(99, yl)), opponent_class=str(r.get("opponent_type") or "UNKNOWN"))
        except ValueError:
            skip["invalid_situation"] += 1
            continue
        play = (K.offense_play if side == OFF else K.defense_play)(f, p)
        if side == DEF and play.signature == ("unknown", "unknown", "unknown"):
            skip["our_defensive_call_unclassifiable"] += 1
            continue
        opp = _opp_action(r, DEF if side == OFF else OFF)
        if opp is None:
            skip["opponent_call_not_recorded_or_unclassifiable"] += 1
            continue
        try:
            out = parse_result(str(r.get("result") or f"{int(r['yards']):+d}"), st)
        except InputError:
            skip["result_unparseable"] += 1
            continue
        verification[opp.verification] += 1
        act = FootballAction(play)
        obs = SnapObservation(st, recommended=act, executed=act, executed_status=EXEC_CONFIRMED, opponent_action=opp, outcome=out,
                              data_source="historical_db", verification="owner_confirmed", closed=True, closed_ts="historical")
        per_game[str(r["game_id"])].append((idx, obs))
    games: list[list[SnapObservation]] = []
    from dataclasses import replace
    for gid, lst in sorted(per_game.items(), key=lambda kv: min(i for i, _ in kv[1])):  # chronology = database row order
        lst.sort(key=lambda t: t[0])
        games.append([replace(o, state=replace(o.state, seq=n)) for n, (_, o) in enumerate(lst, 1)])
    usable = sum(len(g) for g in games)
    report = {"version": IMPORT_VERSION, "database": str(Path(db_path).expanduser().resolve()), "opened_read_only": True,
              "rows_seen": len(rows), "eligibility": dict(elig), "usable_rows": usable, "usable_games": len(games),
              "usable_by_side": dict(Counter(o.state.our_side for g in games for o in g)),
              "opponent_call_verification": dict(verification), "skipped": dict(skip),
              "note": "recommendations are never counted as executions; unverified adjustments, unclassifiable plays and "
                      "unrecorded opponent calls are skipped, not guessed"}
    return games, report


def train_from_history(db_path: str | Path, out_dir: str | Path, *, include_macro_rows: bool = False) -> dict[str, Any]:
    """Import (read-only) -> immutable snapshot -> candidate model -> whole-game chronological comparison.

    Writes ``cold_start_model.json`` (the labelled empty prior, so a real artifact file always exists) and, only if
    there is anything to learn from, ``candidate_model.json``. It never promotes: the report states which artifact the
    evidence supports and the owner chooses with ``--model``.
    """
    import json

    from cfb_coach.madden.unified import postgame as PG
    from cfb_coach.madden.unified.world_model import WorldModel

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    games, rep = import_verified(db_path, include_macro_rows=include_macro_rows)
    (out / "cold_start_model.json").write_text(json.dumps(WorldModel().to_dict()))
    rows, excluded = PG.validate_labels(games)
    rep["labels_excluded"] = dict(excluded)
    rep["training_rows"] = len(rows)
    rep["training_games"] = len({r["game_id"] for r in rows})
    if not rows:
        rep.update(decision="COLD_START_RETAINED", reason="no verified, classifiable historical executions were found",
                   comparison=None, candidate_artifact=None)
    else:
        PG.write_snapshot(rows, excluded, out / "snapshot")
        cand = PG.fit_on(rows)
        (out / "candidate_model.json").write_text(json.dumps(cand.to_dict()))
        cmp = PG.compare_models(rows) if rep["training_games"] >= 4 else None
        rep["comparison"] = cmp
        rep["candidate_artifact"] = str(out / "candidate_model.json")
        rep["candidate_provenance"] = dict(cand.n_support(), rows=cand.n_records)
        if cmp is None:
            rep.update(decision="COLD_START_RETAINED", reason=f"only {rep['training_games']} usable game(s); whole-game evaluation needs at least 4")
        elif cmp["learned_improves_prediction"] and cmp["enough_games"]:
            rep.update(decision="CANDIDATE_SUPPORTED", reason="learned estimates beat cold start and both baselines (game-clustered CI excludes 0)")
        else:
            rep.update(decision="COLD_START_RETAINED", reason="learned estimates did not clearly beat the best competitor on held-out games, or too few games")
    rep["promotion"] = "NOT PERFORMED: choose a model explicitly with --model; the candidate is advisory"
    (out / "history_report.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep
