"""Read-only historical import: eligibility, no fabrication, no writes, chronological comparison."""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.madden.unified import history as H, postgame as PG, sim
from cfb_coach.madden.unified.cli import main as cli_main

OFF_FORM = "Gun Trips X Nasty"
CALLS = [("Cover 3 Sky", "4-3 Over"), ("Cover 1 Hole", "4-3 Over"), ("Cover 6", "4-3 Over"), ("Cover 2 Man", "4-3 Over")]


def make_db(path, n_games=10, per_game=20, *, seed=4, extra=True):
    """Real CoachDB schema; verified offensive executions drawn from the synthetic truth (a known-answer test)."""
    import random
    from cfb_coach.madden.unified import knowledge as K
    rng = random.Random(seed)
    db = CoachDB(path)
    n = 0
    for g in range(n_games):
        sid = f"game{g:02d}"
        for i in range(1, per_game + 1):
            cov, form = rng.choice(CALLS)
            f, p = OFF_FORM, rng.choice([x[1] for x in sim.OFF_PLAYS])
            off, dfn = K.offense_play(f, p), K.defense_play(form, cov)
            yards = round(rng.gauss(sim.true_mean(off.concept, dfn.signature[1], 4.0 if off.play_type == "run" else 6.0), 5.0))
            n += 1
            db.log_snap(opponent_id="cpu_x", side="offense", situation_raw=f"2&6 my 40", our_call=f"{f} {p}", formation=f, play=p,
                        down=2, distance=6, yardline=40, result=f"{yards:+d}", yards=yards, coverage_typed=f"{form} | {cov}",
                        session_id=sid, ml_snap_id=f"{sid}-{i:04d}", snap_seq=i, executed_status="identified",
                        executed_formation=f, executed_play=p, executed_verification="verified")
    if extra:  # rows that must NOT be trained on
        db.log_snap(opponent_id="cpu_x", side="offense", situation_raw="x", our_call="x", formation=OFF_FORM, play="WR Smash", down=1, distance=10,
                    yardline=30, result="+20", yards=20, coverage_typed="4-3 Over | Cover 1 Hole", session_id="game00", ml_snap_id="game00-0900",
                    executed_status="shown", executed_verification="shown_call")                # recommendation only
        db.log_snap(opponent_id="cpu_x", side="offense", situation_raw="x", our_call="x", formation="Not A Formation", play="Nope", down=1, distance=10,
                    yardline=30, result="+20", yards=20, coverage_typed="4-3 Over | Cover 1 Hole", session_id="game00", ml_snap_id="game00-0901",
                    executed_status="identified", executed_formation="Not A Formation", executed_play="Nope", executed_verification="verified")
        db.log_snap(opponent_id="cpu_x", side="offense", situation_raw="x", our_call="x", formation=OFF_FORM, play="WR Smash", down=1, distance=10,
                    yardline=30, result="+9", yards=9, session_id="game00", ml_snap_id="game00-0902", executed_status="identified",
                    executed_formation=OFF_FORM, executed_play="WR Smash", executed_verification="verified")  # opponent call never recorded
        db.log_snap(opponent_id="cpu_x", side="offense", situation_raw="x", our_call="x", formation=OFF_FORM, play="WR Smash", macro="MAN", down=1,
                    distance=10, yardline=30, result="+9", yards=9, coverage_typed="4-3 Over | Cover 1 Hole", session_id="game00",
                    ml_snap_id="game00-0903", executed_status="identified", executed_formation=OFF_FORM, executed_play="WR Smash",
                    executed_verification="verified")                                              # adjustment execution unverified
    db.close()
    return str(path)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_importer_is_read_only_and_counts_every_exclusion(tmp_path):
    p = make_db(tmp_path / "hist.db")
    before = sha(p)
    games, rep = H.import_verified(p)
    assert sha(p) == before and rep["opened_read_only"]
    assert rep["usable_rows"] == 200 and rep["usable_games"] == 10 and rep["usable_by_side"] == {"offense": 200}
    sk = rep["skipped"]
    assert sk["executed_play_not_in_catalog"] == 1 and sk["opponent_call_not_recorded_or_unclassifiable"] == 1 and sk["adjustment_execution_unverified"] == 1
    assert any(k.startswith("not_a_verified_execution") for k in sk)  # the 'shown' recommendation was not promoted
    assert all(o.executed_status == "confirmed" and o.data_source == "historical_db" for g in games for o in g)
    assert [o.state.seq for o in games[0]] == list(range(1, len(games[0]) + 1))
    with pytest.raises(sqlite3.OperationalError):  # prove the connection really cannot write
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        c.execute("create table zz(a)")


def test_importer_missing_database_is_an_error_not_an_empty_success(tmp_path):
    with pytest.raises(FileNotFoundError):
        H.import_verified(tmp_path / "nope.db")


def test_history_training_compares_chronologically_and_never_promotes(tmp_path):
    p = make_db(tmp_path / "hist.db", n_games=24, per_game=30)
    before = sha(p)
    rep = H.train_from_history(p, tmp_path / "out")
    assert sha(p) == before
    cmp = rep["comparison"]
    assert set(cmp["models"]) == {"cold_start", "learned_challenger", "baseline_global_mean", "baseline_situational"}
    assert rep["training_games"] == 24 and cmp["test_games"] == 22  # whole-game folds, first 2 games only ever train
    assert (tmp_path / "out" / "cold_start_model.json").exists() and (tmp_path / "out" / "candidate_model.json").exists()
    assert "NOT PERFORMED" in rep["promotion"] and rep["decision"] in ("CANDIDATE_SUPPORTED", "COLD_START_RETAINED")
    # planted truth + 720 rows: the learned candidate must beat the global-mean baseline on held-out games
    assert cmp["models"]["learned_challenger"]["mae_yards"] < cmp["models"]["baseline_global_mean"]["mae_yards"]


def test_too_little_history_keeps_the_cold_start_and_says_so(tmp_path):
    p = make_db(tmp_path / "small.db", n_games=2, per_game=5, extra=False)
    rep = H.train_from_history(p, tmp_path / "out")
    assert rep["decision"] == "COLD_START_RETAINED" and "usable game" in rep["reason"]
    empty = CoachDB(tmp_path / "empty.db"); empty.close()
    rep2 = H.train_from_history(tmp_path / "empty.db", tmp_path / "out2")
    assert rep2["decision"] == "COLD_START_RETAINED" and rep2["usable_rows"] == 0 and rep2["candidate_artifact"] is None


def test_cli_history_command(tmp_path, capsys):
    p = make_db(tmp_path / "hist.db", n_games=6, per_game=10, extra=False)
    assert cli_main(["history", "--db", p, "--out", str(tmp_path / "o")]) == 0
    assert "COLD_START_RETAINED" in capsys.readouterr().out  # 6 games is below the evidence bar
    assert json.loads((tmp_path / "o" / "history_report.json").read_text())["usable_rows"] == 60
