"""Train a VOD play-call success model.

The package only adds modules. Prep and live play-calling do not import the
trainers, and nothing here writes ``coach.db``. CPU and human snaps stay in
separate strata. Unknown opponent rows are stored in the metrics and left out
of the fit.

``train_and_write`` writes the box v0.7 files into ``<models_dir>/v<label>/``
and updates ``<models_dir>/LAST_TRAINED.json``. It is loaded lazily so that
``cfb_coach.vod_success.shared_features`` (the stdlib-only feature module the
live path imports) does not pull in numpy through this package.
"""

from typing import Any

__all__ = ["train_and_write"]


def __getattr__(name: str) -> Any:
    if name == "train_and_write":
        from cfb_coach.vod_success.train import train_and_write

        return train_and_write
    raise AttributeError(name)
