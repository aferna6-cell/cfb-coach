"""Proof-of-plumbing tests (not performance claims)."""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cfb_coach.madden.model import historical_defense_baseline as hb


def test_distinguish_named_coverage_pressure_and_unknown():
    assert hb.normalize_label("Cover 2 Man")=="cover_2_man"
    assert hb.normalize_label("Cover 2") == "cover_2_unspecified"
    assert hb.normalize_label("Cover 3 Match") == "cover_3_match"
    assert hb.normalize_label("Cover 3 Sky") == "cover_3_sky"
    assert hb.normalize_label("Cover 4 Palms") == "cover_4_palms"
    assert hb.normalize_label("Cover 4 Quarters") == "cover_4_quarters"
    assert hb.normalize_label("Cover 1 Robber") == "cover_1_robber"
    assert hb.normalize_label("pressure") == hb.PRESSURE_UNKNOWN
    assert hb.normalize_label("blitz") != "cover_0"
    assert hb.normalize_label("2 high") == hb.UNRESOLVED


def test_rules_give_conditional_not_universal_routes():
    rules=hb.load_rules()
    assert len(rules["schemes"])>=19
    p=hb.explain_scheme("cover_4_palms",rules=rules)
    q=hb.explain_scheme("Cover 4 Quarters",rules=rules)
    assert "#2" in p["mechanics"] and p["mechanics"] != q["mechanics"]
    assert "runner" in hb.explain_scheme("Tampa 2",rules=rules)["mechanics"]
    assert "not independently" in hb.explain_scheme("Cover 2",rules=rules)["mechanics"]
    assert hb.explain_scheme("pressure",rules=rules)["scheme"]==hb.PRESSURE_UNKNOWN
    assert all(not r["exact_player_duties_known"] for r in rules["schemes"].values())


def test_denies_post_snap_fields_in_prediction():
    for extra in ("coverage_seen", "coverage", "result", "yards", "executed_play", "success", "postplay_call", "session_id"):
        with pytest.raises(ValueError):
            hb.predict_coverage([], {"down":3,"distance":8,extra:"secret"})


def test_distribution_is_normalized_and_no_pressure_alias():
    rows=[{"coverage":"cover_2_man","opponent_id":"cpu","down":1,"distance":10} for _ in range(20)]
    p=hb.predict_coverage(rows,{"opponent_id":"cpu","down":1,"distance":10})
    assert sum(p["probabilities"].values()) == pytest.approx(1)
    assert p["probabilities"]["cover_2_man"] > p["probabilities"]["cover_4_palms"]
    mix=hb.offense_model_mix(p)
    assert mix["cover_2_man"] == pytest.approx(p["probabilities"]["cover_2_man"])
    assert sum(mix.values())==pytest.approx(1)
    assert not any("pressure" in key for key in mix)


def test_generic_coverage_not_falsely_specialized():
    p={"probabilities":{"cover_2_unspecified":0.6,"cover_2_man":0.3,"cover_1_unspecified":0.1}}
    mix=hb.offense_model_mix(p)
    assert mix==pytest.approx({"cover2":0.6,"cover_2_man":0.3,"cover1":0.1})
    with pytest.raises(ValueError):hb.offense_model_mix({"probabilities":{"pressure_unspecified":1.0}})


def test_success_metric_40_60_100_and_turnovers():
    base={"down":1,"distance":10}
    assert hb.success_label({**base,"result":"+4"})==1
    assert hb.success_label({**base,"result":"+3"})==0
    assert hb.success_label({**base,"down":2,"result":"+6"})==1
    assert hb.success_label({**base,"down":3,"result":"+6"})==0
    assert hb.success_label({**base,"down":4,"result":"+10"})==1
    assert hb.success_label({**base,"result":"int"})==0
    assert hb.success_label({**base,"result":"incomplete"})==0
    assert hb.success_label({**base,"result":"td"})==1
    assert hb.success_label({**base,"result":"unknown"}) is None


def _make_game(j:int,n:int=12):
    rows=[]
    for i in range(n):
        r={"id":j*100+i,"session_id":f"game{j}","ts":f"2026-10-{j+1:02}T00:00:{i:02}Z",
           "coverage":"cover_3_match" if j%2 else "cover_2_man", "opponent_id":"cpu" if j%2 else "human_test",
           "down":i%3+1,"distance":2 if i%3==0 else 10,"result":"+8",
           "actual_formation":"Gun Test","actual_play":"Test Play","execution_verified":True}
        rows.append(r)
    return rows


def test_whole_game_chronological_heldout():
    rows=[r for j in range(7) for r in _make_game(j)]
    out=hb.evaluate_games(rows,min_train_games=5)
    assert out["games_total"]==7
    assert out["games_held_out"]==2
    assert out["snap_labels_tested"]==24
    assert out["no_same_game_leakage"]
    assert all(0 <= m["top1_accuracy"] <= 1 for m in out["results"].values())


def test_predicted_coverage_unaffected_by_test_game_labels():
    train=_make_game(0)+_make_game(1)
    live={"opponent_id":"cpu","down":1,"distance":10}
    a=hb.predict_coverage(train,live)
    test=_make_game(2)
    for r in test:r["coverage"]="cover_6"
    b=hb.predict_coverage(train,live)
    assert a==b


def test_matchup_requires_verified_executed_and_named_coverage():
    rows=_make_game(0,8)
    rows.append({**rows[0],"execution_verified":False})
    rows.append({**rows[0],"coverage":"pressure_unspecified"})
    out=hb.matchup_summary(rows,min_cell=5)
    assert out["usable_snaps"]==8
    assert len(out["matchup_cells_at_min_n"])==1
    assert len(out["formation_cells_at_min_n"])==1
    assert out["excluded"]["own_execution_unverified"]==1
    assert out["excluded"]["coverage_unspecified_or_pressure"]==1


def test_readonly_madden_sqlite(tmp_path:Path):
    db=tmp_path/"snap.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE snaps (id integer, session_id text, ts text, opponent_id text, side text,
                      down integer, distance integer, yardline integer, quarter integer,
                      formation text, play text, executed_formation text, executed_play text,
                      executed_status text, executed_verification text, coverage_seen text, result text,
                      yards integer, data_source text)""")
        conn.executemany("INSERT INTO snaps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",[
            (1,"a","2026-01-01","cpu","offense",1,10,20,1,"Gun X","Mesh",None,None,"identified","verified","Cover 4 Palms","+8",None,"live"),
            (2,"a","2026-01-02","cpu","defense",1,10,20,1,"Nickel","Cover 3",None,None,"identified","verified","Cover 2 Man","+8",None,"live")])
    raw=db.read_bytes()
    rows=hb.read_offense_snap_rows(db)
    assert len(rows)==1 and rows[0]["coverage"]=="cover_4_palms"
    assert rows[0]["execution_verified"]
    assert db.read_bytes()==raw
    assert not (tmp_path/"snap.db-journal").exists()


def test_audit_doesnt_claim_postplay_verification():
    out=hb.audit(_make_game(0))
    assert out["coverage_label_screen_verified"]==0
    assert out["own_executions_verified"]==12
    assert out["named_coverage_with_own_execution_verified"]==12


if __name__=="__main__":raise SystemExit(pytest.main([__file__]))
