"""NFL prior prototype on synthetic play-by-play (skips without research deps)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("torch")

from cfb_coach.nfl_prior import data as data_mod  # noqa: E402
from cfb_coach.nfl_prior import evaluate  # noqa: E402
from cfb_coach.nfl_prior.models import TableBaseline  # noqa: E402

TEAMS = [f"T{i:02d}" for i in range(12)]


def synthetic_pbp(seasons=(2019, 2020, 2021, 2022), plays_per_game=60, games_per_season=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for g in range(games_per_season):
            home, away = rng.choice(TEAMS, size=2, replace=False)
            gid = f"{season}_{g:02d}_{away}_{home}"
            for i in range(plays_per_game):
                down = int(rng.integers(1, 5))
                togo = int(rng.integers(1, 16))
                pos = home if i % 2 else away
                p_pass = 1 / (1 + np.exp(-(-0.6 + 0.5 * (down - 1) + 0.12 * (togo - 7))))
                is_pass = int(rng.random() < p_pass)
                epa = rng.normal(0.15 if is_pass else -0.05, 1.2)
                rows.append({
                    "game_id": gid, "season": season, "week": g + 1, "season_type": "REG",
                    "game_date": f"{season}-10-01", "play_id": float(i), "order_sequence": float(i),
                    "posteam": pos, "defteam": away if pos == home else home, "home_team": home,
                    "play_type": "pass" if is_pass else "run", "pass": float(is_pass), "down": float(down),
                    "ydstogo": float(togo), "yardline_100": float(rng.integers(1, 100)), "qtr": float(1 + i * 4 // plays_per_game),
                    "half_seconds_remaining": float(rng.integers(0, 1800)), "game_seconds_remaining": float(rng.integers(0, 3600)),
                    "score_differential": float(rng.integers(-21, 22)), "posteam_timeouts_remaining": 3.0,
                    "defteam_timeouts_remaining": 3.0, "goal_to_go": 0.0, "qb_kneel": 0.0, "qb_spike": 0.0,
                    "two_point_attempt": 0.0, "epa": epa, "success": float(epa > 0),
                })
    df = pd.DataFrame(rows)
    df.loc[0, "qb_kneel"] = 1.0
    df.loc[1, "two_point_attempt"] = 1.0
    df.loc[2, "play_type"] = "punt"
    return df


def quick_cfg(**kw):
    cfg = evaluate.default_config()
    cfg.update(train_seasons=[2019, 2020], val_season=2021, test_seasons=[2022], n_holdout_teams=3,
               bootstrap=20, threads=2, quick=True)
    cfg.update(kw)
    return cfg


def test_prepare_drops_non_plays_and_uses_only_pre_snap_features() -> None:
    raw = synthetic_pbp()
    df = data_mod.prepare(raw)
    assert len(df) == len(raw) - 3
    for feats in (data_mod.SITUATION_FEATURES, data_mod.OUTCOME_FEATURES):
        assert not set(feats) & data_mod.POST_SNAP_COLUMNS
    assert "is_pass" not in data_mod.SITUATION_FEATURES


def test_splits_are_season_separated_and_hold_out_teams() -> None:
    df = data_mod.prepare(synthetic_pbp())
    splits = evaluate.make_splits(df, quick_cfg())
    s = splits["season"]
    assert set(s["train"]["season"]) == {2019, 2020}
    assert set(s["val"]["season"]) == {2021}
    assert set(s["test"]["season"]) == {2022}
    th = splits["team_holdout"]
    held = set(th["holdout_teams"])
    assert len(held) == 3
    assert not held & set(th["train"]["posteam"]) and not held & set(th["val"]["posteam"])
    assert set(th["test"]["posteam"]) <= held


def test_table_baseline_learns_situation_rates() -> None:
    df = data_mod.prepare(synthetic_pbp())
    model = TableBaseline("binary", "is_pass", use_play_type=False).fit(df)
    probe = pd.concat([df[(df["down"] == 1) & (df["ydstogo"] <= 2)].head(1),
                       df[(df["down"] == 3) & (df["ydstogo"] >= 11)].head(1)])
    p = model.predict(probe)
    assert p[0] < p[1]


def test_run_writes_artifacts_and_metrics(tmp_path: Path) -> None:
    run_dir = evaluate.run(quick_cfg(), tmp_path / "runs", frame=synthetic_pbp(), log=lambda s: None)
    meta = json.loads((run_dir / "metadata.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert meta["config"]["seed"] == 20261106
    assert set(metrics) == {"run_pass", "success", "epa"}
    rp = metrics["run_pass"]["splits"]["season"]
    assert set(rp["models"]) == {"table_baseline", "xgboost", "mlp"}
    assert rp["models"]["xgboost"]["auc"] > 0.6
    assert rp["winner"] in rp["models"]
    assert "rmse" in metrics["epa"]["splits"]["team_holdout"]["models"]["mlp"]
    assert (run_dir / "models" / "run_pass" / "xgboost.json").exists()
    assert (run_dir / "models" / "epa" / "mlp.pt").exists()
    assert "Winner" in (run_dir / "report.md").read_text()


def test_run_refuses_live_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate, "FORBIDDEN_LIVE_ROOT", tmp_path / "live")
    with pytest.raises(ValueError):
        evaluate.run(quick_cfg(), tmp_path / "live" / "nfl", frame=synthetic_pbp())
    assert not (tmp_path / "live").exists()
