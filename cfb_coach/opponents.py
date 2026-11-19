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


def _tokens(s: str) -> list[str]:
    return [tok for tok in _norm(s).split() if tok]


def _phrase_in(phrase: list[str], text: list[str]) -> bool:
    """True when ``phrase`` appears as contiguous whole tokens in ``text``."""
    n = len(phrase)
    if n == 0 or n > len(text):
        return False
    return any(text[i:i + n] == phrase for i in range(len(text) - n + 1))


def _is_known_nfl_team(raw: str) -> bool:
    """True for a current NFL full name, nickname, or abbreviation."""
    try:
        from cfb_coach.madden.franchise import resolve_nfl_team

        _name, known = resolve_nfl_team(raw)
        return bool(known)
    except ValueError:
        return False


def resolve_opponent(raw: str) -> str | None:
    aliases = build_alias_map()
    key = _norm(raw)
    if key in aliases:
        return aliases[key]
    # NFL team names must not soft-match short aliases ("nd" in Indianapolis /
    # Browns / Patriots / Commanders, "sc" in San Francisco) or college-team
    # aliases ("houston" in Houston Texans). Exact alias hits above still work
    # ("houston" alone -> quen, "nd" alone -> cpu).
    if _is_known_nfl_team(raw):
        return None
    # Soft match only on whole tokens or exact ids, never a substring inside a token.
    key_toks = _tokens(key)
    matches = [
        oid
        for alias, oid in aliases.items()
        if alias == key or _phrase_in(_tokens(alias), key_toks) or _phrase_in(key_toks, _tokens(alias))
    ]
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


def is_cpu_opponent(opponent_id: str | None) -> bool:
    """CPU / dynasty-CPU games — offense-only coaching (no D calls / D macros)."""
    return (opponent_id or "").strip().lower() in {"cpu", "cpu_dynasty", "notre_dame"}
