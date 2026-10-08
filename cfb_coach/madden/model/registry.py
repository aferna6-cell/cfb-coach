"""Versioned model artifacts. Stub until the registry store is implemented.

Entries are :class:`~cfb_coach.madden.model.schema.ModelRegistryEntry` values.
Reading a directory that has no entry raises. It does not return a fake model.
"""

from __future__ import annotations

from cfb_coach.madden.model.schema import ModelRegistryEntry


def write_entry(entry: ModelRegistryEntry, directory: str) -> str:
    """Write ``entry`` under ``directory`` and return the path. Not implemented."""
    raise NotImplementedError


def read_entry(directory: str) -> ModelRegistryEntry:
    """Load the entry stored in ``directory``. Not implemented."""
    raise NotImplementedError


def promotion_allowed(entry: ModelRegistryEntry) -> bool:
    """Whether hybrid mode may select ``entry``. Not implemented.

    A later implementation may return true only when ``entry.gate`` is
    ``passed``. Until then callers must leave hybrid off.
    """
    raise NotImplementedError
