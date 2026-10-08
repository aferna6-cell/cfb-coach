"""Offline supervised training. Stub until ML Training implements it.

Keep optional learners (logistic regression, LightGBM, CatBoost) inside this
module's implementation so importing the package does not require them.
``seed`` has no default: a run must pass the seed it wants recorded.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.schema import ModelRegistryEntry


def train(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    out_dir: str,
) -> ModelRegistryEntry:
    """Fit candidate models and write a registry entry. Not implemented.

    The returned gate stays ``not_run`` or ``insufficient`` until
    :mod:`cfb_coach.madden.model.evaluate` has scored held-out rows. Do not
    fill metrics that were not computed.
    """
    raise NotImplementedError
