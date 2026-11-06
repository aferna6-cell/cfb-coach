"""Three comparable model families: shrunk situation table, XGBoost, small MLP.

Every model exposes ``fit(train, val)``, ``predict(frame)`` and ``save(dir)``.
``task`` is ``"binary"`` (probability) or ``"regression"`` (expected value).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def distance_bucket(yds: Any) -> Any:
    import numpy as np

    return np.select([yds <= 2, yds <= 4, yds <= 7, yds <= 10], ["1-2", "3-4", "5-7", "8-10"], "11+")


def field_bucket(yl: Any) -> Any:
    import numpy as np

    return np.select([yl <= 5, yl <= 20, yl <= 40, yl <= 80], ["goal", "red", "plus", "mid"], "own")


class TableBaseline:
    """Nested partial pooling: global -> down -> distance -> field [-> run/pass].

    Each cell's mean is shrunk toward its parent with pseudo-count ``strength``.
    """

    name = "table_baseline"

    def __init__(self, task: str, target: str, use_play_type: bool, strength: float = 50.0):
        self.task = task
        self.target = target
        self.use_play_type = use_play_type
        self.strength = strength
        self.table: dict[str, float] = {}
        self.global_mean = 0.0

    def _keys(self, df: Any) -> list[Any]:
        import numpy as np

        parts = [
            df["down"].astype(int).astype(str).to_numpy(),
            distance_bucket(df["ydstogo"].to_numpy()),
            field_bucket(df["yardline_100"].to_numpy()),
        ]
        if self.use_play_type:
            parts.append(np.where(df["is_pass"].to_numpy() > 0, "pass", "run"))
        keys, acc = [], None
        for p in parts:
            p = np.asarray(p, dtype=str)
            acc = p if acc is None else np.char.add(np.char.add(acc, "|"), p)
            keys.append(acc)
        return keys

    def fit(self, train: Any, val: Any | None = None) -> "TableBaseline":
        import pandas as pd

        y = train[self.target].to_numpy(dtype="float64")
        self.global_mean = float(y.mean())
        parent = {"": self.global_mean}
        prev_keys = None
        for keys in self._keys(train):
            stats = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
            if prev_keys is None:
                parent_of = {k: "" for k in stats.index}
            else:
                parent_of = dict(zip(keys, prev_keys))
            for k, row in stats.iterrows():
                prior = parent.get(parent_of[k], self.global_mean)
                self.table[k] = (row["sum"] + self.strength * prior) / (row["count"] + self.strength)
            parent = self.table
            prev_keys = keys
        return self

    def predict(self, df: Any) -> Any:
        import numpy as np

        pred = np.full(len(df), self.global_mean, dtype="float64")
        for keys in self._keys(df):
            hit = np.array([self.table.get(k, np.nan) for k in keys])
            pred = np.where(np.isnan(hit), pred, hit)
        return pred

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{self.name}.json").write_text(
            json.dumps({"global_mean": self.global_mean, "strength": self.strength, "table": self.table}),
            encoding="utf-8",
        )


class XGBModel:
    name = "xgboost"

    def __init__(self, task: str, target: str, features: Sequence[str], seed: int, n_jobs: int = 8,
                 n_estimators: int = 2000, quick: bool = False):
        self.task = task
        self.target = target
        self.features = list(features)
        self.seed = seed
        self.n_jobs = n_jobs
        self.n_estimators = 60 if quick else n_estimators
        self.model: Any = None

    def fit(self, train: Any, val: Any) -> "XGBModel":
        import xgboost as xgb

        common = dict(
            n_estimators=self.n_estimators, learning_rate=0.05, max_depth=6, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=self.n_jobs,
            random_state=self.seed, early_stopping_rounds=50,
        )
        if self.task == "binary":
            self.model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
        else:
            self.model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
        self.model.fit(
            train[self.features], train[self.target],
            eval_set=[(val[self.features], val[self.target])], verbose=False,
        )
        return self

    def predict(self, df: Any) -> Any:
        if self.task == "binary":
            return self.model.predict_proba(df[self.features])[:, 1]
        return self.model.predict(df[self.features])

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(directory / f"{self.name}.json")


class MLPModel:
    """Two hidden layers (64, 32), early-stopped on the validation season."""

    name = "mlp"

    def __init__(self, task: str, target: str, features: Sequence[str], seed: int, threads: int = 8,
                 max_epochs: int = 40, patience: int = 4, quick: bool = False):
        self.task = task
        self.target = target
        self.features = list(features)
        self.seed = seed
        self.threads = threads
        self.max_epochs = 3 if quick else max_epochs
        self.patience = patience
        self.net: Any = None
        self.mean: Any = None
        self.std: Any = None
        self.epochs_run = 0

    def _x(self, df: Any) -> Any:
        import torch

        x = (df[self.features].to_numpy(dtype="float32") - self.mean) / self.std
        return torch.from_numpy(x)

    def fit(self, train: Any, val: Any) -> "MLPModel":
        import copy

        import numpy as np
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        rng = np.random.default_rng(self.seed)
        x = train[self.features].to_numpy(dtype="float32")
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        xt = self._x(train)
        yt = torch.from_numpy(train[self.target].to_numpy(dtype="float32", copy=True))
        xv = self._x(val)
        yv = torch.from_numpy(val[self.target].to_numpy(dtype="float32", copy=True))
        self.net = nn.Sequential(
            nn.Linear(len(self.features), 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )
        loss_fn = nn.BCEWithLogitsLoss() if self.task == "binary" else nn.MSELoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            self.net.train()
            order = torch.from_numpy(rng.permutation(len(xt)))
            for i in range(0, len(order), 1024):
                idx = order[i:i + 1024]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(loss_fn(self.net(xv).squeeze(1), yv))
            self.epochs_run = epoch + 1
            if v < best - 1e-5:
                best, best_state, bad = v, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict(self, df: Any) -> Any:
        import torch

        with torch.no_grad():
            out = self.net(self._x(df)).squeeze(1)
            if self.task == "binary":
                out = torch.sigmoid(out)
        return out.numpy().astype("float64")

    def save(self, directory: Path) -> None:
        import torch

        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.net.state_dict(), "mean": self.mean, "std": self.std,
             "features": self.features, "task": self.task},
            directory / f"{self.name}.pt",
        )
