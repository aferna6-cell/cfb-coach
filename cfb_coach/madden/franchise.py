"""Madden Franchise profiles (primary / lab) + configurable primary team.

primary = serious Franchise save (team TBD until Aidan sets it).
lab     = optional practice save (mirrors CFB ohio_state → alabama promote).

Team config lives in <data dir>/madden27_config.json and can be overridden by
CFB_COACH_MADDEN_PRIMARY_TEAM / CFB_COACH_MADDEN_LAB_TEAM.
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
DEFAULT_PROFILE = PRIMARY
PROFILE_CHOICES = ("primary", "lab")
META_KEY = "franchise_profile"
CONFIG_FILENAME = "madden27_config.json"

_PROFILE_ALIASES = {
    "primary": PRIMARY,
    "franchise_primary": PRIMARY,
    "main": PRIMARY,
    "serious": PRIMARY,
    "lab": LAB,
    "franchise_lab": LAB,
    "practice": LAB,
    "experimental": LAB,
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
        raise ValueError(f"Unknown franchise profile {raw!r}. Use: primary | lab.")
    return out


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
    cfg: dict[str, Any] = {"primary_team": None, "lab_team": None}
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
    clear_primary: bool = False,
    clear_lab: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Update the on-disk config. Returns (config, warnings)."""
    path = config_path()
    cfg: dict[str, Any] = {"primary_team": None, "lab_team": None}
    if path.is_file():
        try:
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    warnings: list[str] = []
    for key, raw, clear in (
        ("primary_team", primary_team, clear_primary),
        ("lab_team", lab_team, clear_lab),
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
    cfg["updated_ts"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return load_config(), warnings


def team_for_profile(profile: str | None, cfg: dict[str, Any] | None = None) -> str | None:
    cfg = cfg if cfg is not None else load_config()
    key = "lab_team" if normalize_profile(profile) == LAB else "primary_team"
    return cfg.get(key) or None


def team_label(team: str | None) -> str:
    return team or "TBD (scheme pack)"


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def profile_config(profile: str | None = None) -> dict[str, Any]:
    pid = normalize_profile(profile)
    league = (_load_json("seed.json").get("league") or {}).get("franchise_profiles") or {}
    block = league.get(pid) or {}
    cat = (load_macro_catalog().get("franchise_profiles") or {}).get(pid) or {}
    experimental = bool(cat.get("allow_experimental_active", pid == LAB))
    team = team_for_profile(pid)
    return {
        "id": pid,
        "profile_id": block.get("id") or f"franchise_{pid}",
        "label": block.get("label") or ("Franchise lab" if pid == LAB else "Franchise primary"),
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
        return normalize_profile(raw) if raw else DEFAULT_PROFILE
    except (ValueError, Exception):  # noqa: BLE001
        return DEFAULT_PROFILE


def set_session_profile(db: Any, profile: str | None) -> str:
    pid = normalize_profile(profile)
    if db is not None:
        db.set_meta(META_KEY, pid)
    return pid


def doctrine_line(team: str | None = None) -> str:
    who = f"primary team {team}" if team else "primary team TBD"
    return (
        f"Doctrine (Madden 27 Franchise): meta-grounded scheme pack, {who}. "
        "One tell = log + mild bump; targeted macro only on REPEATED tendency this game. "
        "User games O+D within 8-macro cap; CPU = offense-only. "
        "Most D snaps Nickel Mug Cover 4 Quarters / Cover 3 Match / Cover 2 Sink with no macro."
    )
