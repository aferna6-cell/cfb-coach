"""Held-out evaluation and the promotion gate. Stub until ML Training implements it.

Splits must keep whole games, opponents, and VOD sources together. In-sample
fit is not a passing gate.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.schema import ModelRegistryEntry


def evaluate(
    entry: ModelRegistryEntry,
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> ModelRegistryEntry:
    """Score ``entry`` on held-out rows and return a new registry entry.

    Not implemented. Must not mark the gate passed from training accuracy.
    """
    raise NotImplementedError
