"""Madden Franchise profiles (primary / lab / online) + configurable teams.

primary = serious Franchise save — Detroit Lions by default (multi-user league).
lab     = optional practice save (mirrors CFB ohio_state → alabama promote).
online  = online Madden league, separate from the Franchise saves. User team is
          the Kansas City Chiefs by default (``online_team``); every game needs
          the opponent's NFL team (``--opponent-team``). Never written as the
          sticky DB profile, so a later ``play`` without ``--franchise`` stays on
          primary (Lions).
titans  = Tennessee Titans franchise, also never sticky. Same team-keyed logging
          and civil offense + Ravens civil defense defaults as online, but a
          distinct league id / opponent-id prefix / user team so games never
          mix with the Chiefs online league or the Lions franchise.
giants  = New York Giants franchise, also never sticky. Same team-keyed logging
          and civil defaults as titans, but offense calls from the Lions/CPU
          custom book and a 3-snap offensive play-family cooldown. Kept
          separate from Lions, Chiefs, and Titans.

Team config lives in <data dir>/madden27_config.json and can be overridden by
CFB_COACH_MADDEN_PRIMARY_TEAM / CFB_COACH_MADDEN_LAB_TEAM. With no config file the
defaults below apply (Lions; offense starts on stock Buccaneers, defense on stock
49ers, and prep's research may recommend a switch — `offense_book` / `defense_book`
= "auto"). Pin a book with `config --game madden27 --o-book stock:Buccaneers`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfb_coach.madden.data import _load_json, load_macro_catalog

PRIMARY = "primary"
LAB = "lab"
ONLINE = "online"
TITANS = "titans"
GIANTS = "giants"
DEFAULT_PROFILE = PRIMARY
PROFILE_CHOICES = ("primary", "lab", "online", "titans", "giants")
META_KEY = "franchise_profile"
CONFIG_FILENAME = "madden27_config.json"
DEFAULT_PRIMARY_TEAM = "Detroit Lions"
DEFAULT_ONLINE_TEAM = "Kansas City Chiefs"
DEFAULT_TITANS_TEAM = "Tennessee Titans"
DEFAULT_GIANTS_TEAM = "New York Giants"
DEFAULT_START_BOOK = {"offense": "Buccaneers", "defense": "49ers"}
DEFAULT_CONFIG: dict[str, Any] = {
    "primary_team": DEFAULT_PRIMARY_TEAM,
    "lab_team": None,
    "online_team": DEFAULT_ONLINE_TEAM,  # online league user team (separate from primary)
    "offense_book": "auto",  # auto = start on stock Buccaneers, research may recommend a switch
    "defense_book": "auto",  # auto = start on stock 49ers
    "live_macros": True,  # False = play shows no Custom Adjustment (config --no-macros)
    # Experimental macros (O-RPO, HEAT) stay out of the primary list unless one of these is set.
    "experimental_macros": False,
    "experimental_swaps": [],
    # VOD model prior. Models dir stays empty here — set CFB_COACH_VOD_MODELS or --vod-models.
    "vod_prior": True,
    "vod_freeze_book": False,
    "vod_models_dir": "",
}

_PROFILE_ALIASES = {
    "primary": PRIMARY,
    "franchise_primary": PRIMARY,
    "main": PRIMARY,
    "serious": PRIMARY,
    "lab": LAB,
    "franchise_lab": LAB,
    "practice": LAB,
    "experimental": LAB,
    "online": ONLINE,
    "online_league": ONLINE,
    "league": ONLINE,
    "titans": TITANS,
    "tennessee": TITANS,
    "tennessee_titans": TITANS,
    "titans_franchise": TITANS,
    "giants": GIANTS,
    "ny_giants": GIANTS,
    "new_york_giants": GIANTS,
    "giants_franchise": GIANTS,
}

# Team-keyed leagues log by opponent NFL team, never by gamertag. They are
# per-run only (never the sticky DB profile). Specs keep Chiefs and Titans data
# in separate buckets (league id + opponent_id prefix + user_team).
# Giants is a third team-keyed bucket, isolated from Chiefs and Titans.
_TEAM_KEYED: dict[str, dict[str, str]] = {
    ONLINE: {
        "league": ONLINE,
        "id_prefix": "online_",
        "user_team": DEFAULT_ONLINE_TEAM,
        "label": "Online league",
    },
    TITANS: {
        "league": TITANS,
        "id_prefix": "titans_",
        "user_team": DEFAULT_TITANS_TEAM,
        "label": "Titans franchise",
    },
    GIANTS: {
        "league": GIANTS,
        "id_prefix": "giants_",
        "user_team": DEFAULT_GIANTS_TEAM,
        "label": "Giants franchise",
    },
}

_TEAM_ABBR = {
    "ari": "Arizona Cardinals", "atl": "Atlanta Falcons", "bal": "Baltimore Ravens",
    "buf": "Buffalo Bills", "car": "Carolina Panthers", "chi": "Chicago Bears",
    "cin": "Cincinnati Bengals", "cle": "Cleveland Browns", "dal": "Dallas Cowboys",
    "den": "Denver Broncos", "det": "Detroit Lions", "gb": "Green Bay Packers",
    "hou": "Houston Texans", "ind": "Indianapolis Colts", "jax": "Jacksonville Jaguars",
    "kc": "Kansas City Chiefs", "lv": "Las Vegas Raiders", "lac": "Los Angeles Chargers",
    "lar": "Los Angeles Rams", "mia": "Miami Dolphins", "min": "Minnesota Vikings",
    "ne": "New England Patriots", "no": "New Orleans Saints", "nyg": "New York Giants",
    "nyj": "New York Jets", "phi": "Philadelphia Eagles", "pit": "Pittsburgh Steelers",
    "sf": "San Francisco 49ers", "sea": "Seattle Seahawks", "tb": "Tampa Bay Buccaneers",
    "ten": "Tennessee Titans", "was": "Washington Commanders", "wsh": "Washington Commanders",
    "bucs": "Tampa Bay Buccaneers", "niners": "San Francisco 49ers", "pats": "New England Patriots",
}


def normalize_profile(raw: str | None) -> str:
    key = (raw or DEFAULT_PROFILE).strip().lower().replace("-", "_").replace(" ", "_")
    out = _PROFILE_ALIASES.get(key)
    if out is None:
        raise ValueError(f"Unknown franchise profile {raw!r}. Use: primary | lab | online | titans | giants.")
    return out


def is_team_keyed_profile(profile: str | None) -> bool:
    """True for online / titans / giants: log by opponent NFL team, never sticky."""
    if not profile:
        return False
    try:
        return normalize_profile(profile) in _TEAM_KEYED
    except ValueError:
        return False


def team_keyed_spec(profile: str | None) -> dict[str, str]:
    """Return league / id_prefix / user_team / label for a team-keyed profile."""
    pid = normalize_profile(profile)
    spec = _TEAM_KEYED.get(pid)
    if spec is None:
        raise ValueError(f"{pid} is not a team-keyed franchise (online | titans | giants).")
    return dict(spec)


def nfl_teams() -> list[str]:
    return list(_load_json("seed.json").get("nfl_teams") or [])


def resolve_nfl_team(raw: str) -> tuple[str, bool]:
    """Return (team_name, is_known_nfl_team). Unknown names are kept as custom."""
    text = " ".join((raw or "").strip().split())
    if not text:
        raise ValueError("Team name is empty.")
    low = text.lower()
    if low in _TEAM_ABBR:
        return _TEAM_ABBR[low], True
    teams = nfl_teams()
    for t in teams:
        if low == t.lower():
            return t, True
    # Nickname match (e.g. "Buccaneers", "49ers") — unique by construction
    for t in teams:
        if low == t.lower().split()[-1]:
            return t, True
    # City match only when unambiguous (not LA / New York)
    hits = [t for t in teams if t.lower().startswith(low + " ")]
    if len(hits) == 1:
        return hits[0], True
    return text, False


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

def config_path() -> Path:
    from cfb_coach.games import data_dir

    return data_dir() / CONFIG_FILENAME


def load_config() -> dict[str, Any]:
    cfg: dict[str, Any] = dict(DEFAULT_CONFIG)
    path = config_path()
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg.update(raw)
        except (OSError, ValueError):
            pass
    env_p = os.environ.get("CFB_COACH_MADDEN_PRIMARY_TEAM")
    env_l = os.environ.get("CFB_COACH_MADDEN_LAB_TEAM")
    if env_p:
        cfg["primary_team"] = resolve_nfl_team(env_p)[0]
        cfg["primary_team_source"] = "env"
    if env_l:
        cfg["lab_team"] = resolve_nfl_team(env_l)[0]
        cfg["lab_team_source"] = "env"
    return cfg


def save_config(
    *,
    primary_team: str | None = None,
    lab_team: str | None = None,
    online_team: str | None = None,
    clear_primary: bool = False,
    clear_lab: bool = False,
    offense_book: str | None = None,
    defense_book: str | None = None,
    live_macros: bool | None = None,
    experimental_macros: bool | None = None,
    swap_macro: str | None = None,
    unswap_macro: str | None = None,
    vod_prior: bool | None = None,
    vod_freeze_book: bool | None = None,
    vod_models_dir: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Update the on-disk config. Returns (config, warnings)."""
    path = config_path()
    cfg: dict[str, Any] = dict(DEFAULT_CONFIG)
    if path.is_file():
        try:
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    warnings: list[str] = []
    for key, raw, clear in (
        ("primary_team", primary_team, clear_primary),
        ("lab_team", lab_team, clear_lab),
        ("online_team", online_team, False),
    ):
        if clear:
            cfg[key] = None
        elif raw:
            name, known = resolve_nfl_team(raw)
            if not known:
                warnings.append(
                    f"{name!r} is not a current NFL team name — kept as a custom/relocated team."
                )
            cfg[key] = name
    from cfb_coach.madden.playbook import parse_choice

    for key, side, raw in (("offense_book", "offense", offense_book), ("defense_book", "defense", defense_book)):
        if raw:
            mode, book = parse_choice(side, raw)  # raises ValueError on an unknown book
            cfg[key] = mode if mode != "stock" else f"stock:{book}"
    if live_macros is not None:
        cfg["live_macros"] = bool(live_macros)
    if experimental_macros is not None:
        cfg["experimental_macros"] = bool(experimental_macros)
    if vod_prior is not None:
        cfg["vod_prior"] = bool(vod_prior)
    if vod_freeze_book is not None:
        cfg["vod_freeze_book"] = bool(vod_freeze_book)
    if vod_models_dir is not None:
        cfg["vod_models_dir"] = str(vod_models_dir).strip()
    swaps = [str(x).upper() for x in (cfg.get("experimental_swaps") or []) if str(x).strip()]
    if swap_macro:
        key = str(swap_macro).upper()
        if key not in swaps:
            swaps.append(key)
    if unswap_macro:
        swaps = [s for s in swaps if s != str(unswap_macro).upper()]
    cfg["experimental_swaps"] = swaps
    cfg["updated_ts"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return load_config(), warnings


def team_for_profile(profile: str | None, cfg: dict[str, Any] | None = None) -> str | None:
    cfg = cfg if cfg is not None else load_config()
    pid = normalize_profile(profile)
    if pid == ONLINE:
        return cfg.get("online_team") or DEFAULT_ONLINE_TEAM
    if pid == TITANS:
        return DEFAULT_TITANS_TEAM
    if pid == GIANTS:
        return DEFAULT_GIANTS_TEAM
    key = "lab_team" if pid == LAB else "primary_team"
    return cfg.get(key) or None


def team_label(team: str | None) -> str:
    return team or "TBD"


def book_choice(side: str, cfg: dict[str, Any] | None = None) -> str:
    """Configured default --o-book / --d-book (``auto`` unless Aidan pinned one)."""
    cfg = cfg if cfg is not None else load_config()
    return str(cfg.get(f"{side}_book") or "auto")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def profile_config(profile: str | None = None) -> dict[str, Any]:
    pid = normalize_profile(profile)
    league = (_load_json("seed.json").get("league") or {}).get("franchise_profiles") or {}
    block = league.get(pid) or {}
    cats = load_macro_catalog().get("franchise_profiles") or {}
    # team-keyed leagues play the same serious doctrine as primary unless named.
    cat = cats.get(pid) or (cats.get(PRIMARY) if pid in _TEAM_KEYED else None) or {}
    experimental = bool(cat.get("allow_experimental_active", pid == LAB))
    team = team_for_profile(pid)
    default_label = {
        LAB: "Franchise lab",
        ONLINE: "Online league",
        TITANS: "Titans franchise",
        GIANTS: "Giants franchise",
    }.get(pid, "Franchise primary")
    default_profile_id = {
        ONLINE: "online_league",
        TITANS: "titans_franchise",
        GIANTS: "giants_franchise",
    }.get(pid, f"franchise_{pid}")
    return {
        "id": pid,
        "profile_id": block.get("id") or default_profile_id,
        "label": block.get("label") or default_label,
        "mode": block.get("mode") or ("experimental" if pid == LAB else "serious"),
        "description": block.get("description") or "",
        "team": team,
        "team_label": team_label(team),
        "default_active": list(cat.get("default_active") or []),
        "default_benched": list(cat.get("default_benched") or []),
        "allow_experimental_active": experimental,
        "experimental_badge": experimental and pid == LAB,
    }


def get_session_profile(db: Any) -> str:
    try:
        raw = db.get_meta(META_KEY) if db is not None else None
        pid = normalize_profile(raw) if raw else DEFAULT_PROFILE
        # team-keyed leagues are per-run only; a stale meta value must not
        # pull Lions games into them.
        return DEFAULT_PROFILE if pid in _TEAM_KEYED else pid
    except (ValueError, Exception):  # noqa: BLE001
        return DEFAULT_PROFILE


def set_session_profile(db: Any, profile: str | None) -> str:
    pid = normalize_profile(profile)
    if db is not None and pid not in _TEAM_KEYED:  # team-keyed are never the sticky default
        db.set_meta(META_KEY, pid)
    return pid


def doctrine_line(team: str | None = None) -> str:
    who = f"primary team {team}" if team else "primary team TBD"
    return (
        f"Doctrine (Madden 27 Franchise): meta-grounded scheme pack, {who}. "
        "One tell = log + mild bump; targeted macro on a REPEATED tendency this game, "
        "or when its when-to-fire matches the snap. config --no-macros turns live macros off. "
        "User games: 8 offense + 8 defense Custom Adjustments (LB); "
        "CPU = 8 offense macros, no defense macros. "
        "Most D snaps Nickel Over Cover 4 Quarters / Cover 3 Match / Tampa 2 with no macro (rush four, drop seven)."
    )
