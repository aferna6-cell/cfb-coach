"""Train a VOD play-call success model.

The package only adds modules. Prep and live play-calling do not import it,
and it does not write ``coach.db``. CPU and human snaps stay in separate
strata. Unknown opponent rows are stored in the metrics and left out of the fit.

``train_and_write`` writes the box v0.7 files into ``<models_dir>/v<label>/``
and updates ``<models_dir>/LAST_TRAINED.json``.
"""

from cfb_coach.vod_success.train import train_and_write

__all__ = ["train_and_write"]
