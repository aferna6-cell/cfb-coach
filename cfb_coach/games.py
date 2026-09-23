"""Game registry — `--game cfb27|madden27` selects seed, meta, DB and file names.

CFB 27 stays the default and keeps its original paths (coach.db,
prep_<opp>.html, copilot_overlay.html). Madden 27 Franchise gets its own
namespace next to it so the two never share tendencies or prep state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CFB27 = "cfb27"
MADDEN27 = "madden27"
DEFAULT_GAME = CFB27
GAME_CHOICES = ("cfb27", "cfb", "madden27", "madden")

_ALIASES = {
    "cfb27": CFB27,
    "cfb": CFB27,
    "cfb_27": CFB27,
    "ncaa": CFB27,
    "madden27": MADDEN27,
    "madden": MADDEN27,
    "madden_27": MADDEN27,
    "m27": MADDEN27,
    "nfl": MADDEN27,
}


@dataclass(frozen=True)
class GameProfile:
    id: str
    label: str
    mode: str
    meta_version: str
    db_filename: str
    prep_prefix: str
    overlay_filename: str
    brand: str


GAMES: dict[str, GameProfile] = {
    CFB27: GameProfile(
        id=CFB27,
        label="CFB 27",
        mode="dynasty",
        meta_version="cfb27-2026-09",
        db_filename="coach.db",
        prep_prefix="prep_",
        overlay_filename="copilot_overlay.html",
        brand="CFB Coach",
    ),
    MADDEN27: GameProfile(
        id=MADDEN27,
        label="Madden 27 Franchise",
        mode="franchise",
        meta_version="madden27-2026-09",
        db_filename="madden27.db",
        prep_prefix="prep_madden27_",
        overlay_filename="madden27_overlay.html",
        brand="Madden Coach",
    ),
}


def normalize_game(raw: str | None) -> str:
    key = (raw or DEFAULT_GAME).strip().lower().replace("-", "_").replace(" ", "_")
    gid = _ALIASES.get(key)
    if gid is None:
        raise ValueError(f"Unknown game {raw!r}. Use: cfb27 | madden27 (alias: madden).")
    return gid


def game_profile(raw: str | None = None) -> GameProfile:
    return GAMES[normalize_game(raw)]


def is_madden(raw: str | None) -> bool:
    return normalize_game(raw) == MADDEN27


def data_dir() -> Path:
    """Same data dir as the CFB DB: $CFB_COACH_DB's parent, else ~/.cfb-coach."""
    from cfb_coach.prep_browser import default_prep_dir

    return default_prep_dir()


def madden_db_path() -> Path:
    """$CFB_COACH_MADDEN_DB, else <data dir>/madden27.db (sibling of coach.db)."""
    raw = os.environ.get("CFB_COACH_MADDEN_DB")
    if raw:
        p = Path(raw).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return data_dir() / GAMES[MADDEN27].db_filename
