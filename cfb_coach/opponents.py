"""Opponent ID resolution — accept ids, display names, and team aliases."""

from __future__ import annotations

from cfb_coach.seed import load_seed

# Extra aliases beyond display_name / team_now
_EXTRA_ALIASES: dict[str, str] = {
    "auburn": "gavin",
    "houston": "quen",
    "smu": "tiano",
    "temple": "tiano",
    "tcu": "gio",
    "florida": "gio",
    "south carolina": "michael",
    "southcarolina": "michael",
    "sc": "michael",
    "duke": "harrison",
    "lsu": "harrison",
    "ucf": "jaxon",
    "kentucky": "jaxon",
    "nc state": "ryan",
    "ncstate": "ryan",
    "ncsu": "ryan",
    "arizona state": "james",
    "arizonastate": "james",
    "asu": "james",
    "cpu dynasty": "cpu",
    "notre dame": "cpu",
    "nd": "cpu",
}


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().replace("_", " ").replace("-", " ").split())


def build_alias_map() -> dict[str, str]:
    seed = load_seed()
    aliases: dict[str, str] = {}
    for oid, opp in seed["opponents"].items():
        aliases[_norm(oid)] = oid
        aliases[_norm(opp.get("display_name", ""))] = oid
        team = opp.get("team_now")
        if team:
            aliases[_norm(team)] = oid
        prev = opp.get("prev")
        if prev:
            aliases[_norm(str(prev))] = oid
    for alias, oid in _EXTRA_ALIASES.items():
        aliases[_norm(alias)] = oid
    # drop empty keys
    return {k: v for k, v in aliases.items() if k}


def resolve_opponent(raw: str) -> str | None:
    aliases = build_alias_map()
    key = _norm(raw)
    if key in aliases:
        return aliases[key]
    # prefix / containment soft match
    matches = [oid for a, oid in aliases.items() if key in a or a in key]
    uniq = sorted(set(matches))
    if len(uniq) == 1:
        return uniq[0]
    return None


def format_opponent_list() -> str:
    seed = load_seed()
    lines = ["ID          Display      Team              Confidence  Skill"]
    lines.append("-" * 72)
    for oid, opp in seed["opponents"].items():
        lines.append(
            f"{oid:<12}{opp.get('display_name', ''):<13}"
            f"{(opp.get('team_now') or ''):<18}"
            f"{(opp.get('confidence') or ''):<12}"
            f"{(opp.get('skill') or '')}"
        )
    lines.append("")
    lines.append(
        "Aliases: display names, current/previous teams "
        "(e.g. auburn→gavin, houston→quen, smu→tiano)."
    )
    return "\n".join(lines)
