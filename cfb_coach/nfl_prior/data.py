"""Load, cache and featurise NFL play-by-play with pre-snap features only."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

DEFAULT_CACHE = Path("~/.cfb-research/nfl/pbp_cache")

RAW_COLUMNS = (
    "game_id", "season", "week", "season_type", "game_date", "play_id", "order_sequence",
    "posteam", "defteam", "home_team", "play_type", "pass", "down", "ydstogo", "yardline_100",
    "qtr", "half_seconds_remaining", "game_seconds_remaining", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "goal_to_go",
    "qb_kneel", "qb_spike", "two_point_attempt", "epa", "success",
)

# Everything a coach knows before the snap. Formation (shotgun / no-huddle) is
# deliberately excluded: for our own call it would leak the decision itself.
SITUATION_FEATURES = (
    "down_1", "down_2", "down_3", "down_4", "ydstogo", "log_ydstogo", "yardline_100",
    "goal_to_go", "qtr", "half_seconds_remaining", "game_seconds_remaining",
    "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "is_home", "two_minute",
)
OUTCOME_FEATURES = SITUATION_FEATURES + ("is_pass",)

# Known only after the snap; must never be a feature.
POST_SNAP_COLUMNS = frozenset({
    "epa", "success", "yards_gained", "wpa", "air_yards", "yards_after_catch", "complete_pass",
    "interception", "fumble_lost", "touchdown", "sack", "qb_scramble", "penalty", "first_down",
})


def load_seasons(seasons: Sequence[int], cache_dir: str | Path = DEFAULT_CACHE) -> Any:
    """Return a pandas frame of RAW_COLUMNS for ``seasons``, caching one parquet per season."""
    import pandas as pd

    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        path = cache / f"pbp_{int(season)}.parquet"
        if not path.exists():
            import nflreadpy

            raw = nflreadpy.load_pbp(seasons=[int(season)])
            raw.select(list(RAW_COLUMNS)).write_parquet(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def prepare(df: Any) -> Any:
    """Keep real offensive run/pass snaps and add pre-snap features and targets."""
    import numpy as np

    keep = (
        df["play_type"].isin(["run", "pass"])
        & (df["qb_kneel"].fillna(0) == 0)
        & (df["qb_spike"].fillna(0) == 0)
        & (df["two_point_attempt"].fillna(0) == 0)
        & df["down"].notna()
        & df["epa"].notna()
        & df["posteam"].notna()
    )
    out = df.loc[keep].copy()
    down = out["down"].astype(int)
    for d in (1, 2, 3, 4):
        out[f"down_{d}"] = (down == d).astype("float32")
    out["ydstogo"] = out["ydstogo"].clip(0, 30).astype("float32")
    out["log_ydstogo"] = np.log1p(out["ydstogo"]).astype("float32")
    out["yardline_100"] = out["yardline_100"].astype("float32")
    out["goal_to_go"] = out["goal_to_go"].fillna(0).astype("float32")
    out["qtr"] = out["qtr"].clip(1, 5).astype("float32")
    out["half_seconds_remaining"] = out["half_seconds_remaining"].fillna(0).astype("float32")
    out["game_seconds_remaining"] = out["game_seconds_remaining"].fillna(0).astype("float32")
    out["score_differential"] = out["score_differential"].fillna(0).clip(-35, 35).astype("float32")
    out["posteam_timeouts_remaining"] = out["posteam_timeouts_remaining"].fillna(3).astype("float32")
    out["defteam_timeouts_remaining"] = out["defteam_timeouts_remaining"].fillna(3).astype("float32")
    out["is_home"] = (out["posteam"] == out["home_team"]).astype("float32")
    out["two_minute"] = (out["half_seconds_remaining"] <= 120).astype("float32")
    out["is_pass"] = (out["pass"].fillna(0) > 0).astype("float32")
    out["success"] = (out["epa"] > 0).astype("float32")
    out["epa"] = out["epa"].astype("float32")
    return out.reset_index(drop=True)


def frame_fingerprint(df: Any, columns: Sequence[str]) -> str:
    import pandas as pd

    h = pd.util.hash_pandas_object(df[list(columns)], index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]
