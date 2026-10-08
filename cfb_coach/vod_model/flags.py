"""VOD prior on/off, and the separate freeze-the-book switch.

Precedence, first match wins:

1. CLI context from ``--no-vod-prior`` / ``--freeze-vod-book`` (this process only).
2. ``CFB_COACH_NO_VOD_PRIOR`` / ``CFB_COACH_VOD_FREEZE_BOOK`` (``1``, ``true``, ``yes``).
3. Madden config ``vod_prior`` / ``vod_freeze_book``.
4. Prior on, book editable.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar

_NO_PRIOR: ContextVar[bool | None] = ContextVar("vod_no_prior", default=None)
_FREEZE: ContextVar[bool | None] = ContextVar("vod_freeze_book", default=None)


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _config() -> dict:
    try:
        from cfb_coach.madden.franchise import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def prior_enabled() -> bool:
    if _NO_PRIOR.get() is True:
        return False
    if _truthy("CFB_COACH_NO_VOD_PRIOR"):
        return False
    if _config().get("vod_prior") is False:
        return False
    return True


def book_frozen() -> bool:
    """True when VOD may rank in-book calls but must not edit the book."""
    if not prior_enabled():
        return True
    if _FREEZE.get() is True:
        return True
    if _truthy("CFB_COACH_VOD_FREEZE_BOOK"):
        return True
    return bool(_config().get("vod_freeze_book"))


@contextmanager
def use_cli_flags(*, no_vod_prior: bool = False, freeze_vod_book: bool = False):
    tok_off = _NO_PRIOR.set(True if no_vod_prior else None)
    tok_freeze = _FREEZE.set(True if freeze_vod_book else None)
    try:
        yield
    finally:
        _NO_PRIOR.reset(tok_off)
        _FREEZE.reset(tok_freeze)
