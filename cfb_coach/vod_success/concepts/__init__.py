"""Concept taxonomy and the concept-level VOD success model.

The coach does not import this package. ``python -m cfb_coach.vod_success
concepts build`` writes ``vod_concepts.v1``. ``concepts train`` writes
``<models_dir>/concept/c<N>/`` and ``concept/LAST_TRAINED.json``.
"""

from cfb_coach.vod_success.concepts.taxonomy import build_taxonomy, offense_concept, defense_map
from cfb_coach.vod_success.concepts.train import train_and_write

__all__ = ["build_taxonomy", "defense_map", "offense_concept", "train_and_write"]
