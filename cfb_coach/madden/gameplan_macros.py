"""Pregame call packages for Madden prep — 8 offense + 8 defense by default.

A macro here is a ready-to-call package (formation + play, when to call it, pre-snap
adjustment, read, counter). Every formation and play is taken from the custom playbook
passed in (the coach's trimmed plan). Nothing is invented: if a meta-menu play is not
in that book, it is not called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cfb_coach.madden.catalog import is_deep, is_run, zone_fit
from cfb_coach.madden.data import load_meta_baseline, reads_for, user_job_for

DEFAULT_COUNT = 8

_OFFENSE_SLOTS: tuple[dict[str, Any], ...] = (
    {
        "id": "O-OPENER", "situation": "opener", "label": "Opener",
        "when": "1st & 10, open field (between the 20s).",
        "score_situation": "Any score. Up 8+ in the 4th: stay here and bleed clock.",
        "want_run": True, "zone": "open", "prefer": ("early_down", "two_high"),
        "adjustments": (), "beats": ("two_high",),
        "counter": "Stacked box or a run blitz. Check to {audible} if it is in this formation.",
    },
    {
        "id": "O-3RD-SHORT", "situation": "3rd_short", "label": "3rd-and-short",
        "when": "3rd or 4th and 1–3, open field.",
        "score_situation": "Any score. Tied or trailing: take the sure yards; leading late: still run it.",
        "want_run": True, "zone": "open", "prefer": ("short_yardage",),
        "adjustments": (), "beats": ("run",),
        "counter": "If they sell out on the run, check to {audible}.",
    },
    {
        "id": "O-3RD-LONG", "situation": "3rd_long", "label": "3rd-and-long",
        "when": "3rd or 4th and 7+, open field.",
        "score_situation": "Any score. Trailing by 8+ in the 4th: this is a must-throw, not a draw.",
        "want_run": False, "zone": "open", "prefer": ("long_yardage", "two_high", "man"),
        "adjustments": ("hot_slant_vs_man",), "beats": ("two_high", "man"),
        "counter": "Two-deep safeties sitting on the shot. Take the underneath; don't force a hole shot.",
    },
    {
        "id": "O-RED-ZONE", "situation": "red_zone", "label": "Red zone",
        "when": "Inside the opponent 20, including goal-to-go.",
        "score_situation": "Any score. One-score game: points over style — no hero shot.",
        "want_run": None, "zone": "red_zone", "prefer": ("red_zone", "goal_line", "short_yardage"),
        "adjustments": (), "beats": ("man",),
        "counter": "Goal-line pressure. If the box is stacked, check to {audible}.",
    },
    {
        "id": "O-TWO-MINUTE", "situation": "two_minute", "label": "2-minute",
        "when": "Two-minute drill, hurry-up, or the last possession of a half.",
        "score_situation": "Need a score, or need a first down to ice the game. Trailing late: throw.",
        "want_run": False, "zone": "open", "prefer": ("two_minute", "pressure"),
        "adjustments": ("hot_hb_flat_vs_pressure",), "beats": ("pressure",),
        "counter": "A zero blitz with no outlet. Hot the back; don't hold the ball.",
    },
    {
        "id": "O-VS-BLITZ", "situation": "vs_blitz", "label": "Vs blitz / pressure",
        "when": "They show pressure, Cover 0, or a blitz you've seen twice.",
        "score_situation": "Any score. Protect first when trailing — a sack ends the drive.",
        "want_run": False, "zone": "open", "prefer": ("pressure", "short_yardage"),
        "adjustments": ("hot_hb_flat_vs_pressure", "max_protect_vs_pressure"), "beats": ("pressure",),
        "counter": "They drop out of the blitz look into coverage. Throw the hot, don't wait.",
    },
    {
        "id": "O-VS-RUN", "situation": "vs_run", "label": "Vs run-fit / light box",
        "when": "They are in a two-high or light-box run fit (playing the pass, not the run).",
        "score_situation": "Any score. Leading: this is the clock play. Trailing early: still take the yards.",
        "want_run": True, "zone": "open", "prefer": ("early_down", "two_high"),
        "adjustments": (), "beats": ("two_high", "run"),
        "counter": "They spin down to a stacked box. Check to {audible}.",
    },
    {
        "id": "O-VS-PASS", "situation": "vs_pass", "label": "Vs pass coverage (man / single-high)",
        "when": "They show man, Cover 1, or a single-high shell.",
        "score_situation": "Any score. Trailing: this is the man-beater, not a checkdown.",
        "want_run": False, "zone": "open", "prefer": ("man", "single_high", "long_yardage"),
        "adjustments": ("hot_slant_vs_man",), "beats": ("man", "single_high"),
        "counter": "They bail to two-high. Don't force the man-beater into safeties — check to {audible}.",
    },
)

_DEFENSE_SLOTS: tuple[dict[str, Any], ...] = (
    {
        "id": "D-OPENER", "situation": "opener", "label": "Opener",
        "when": "1st & 10, open field.",
        "score_situation": "Any score. Base call — rush four, drop seven unless the logs say otherwise.",
        "want_run": None, "zone": "open", "prefer": ("home",),
        "adjustments": (), "beats": ("vert", "two_high"),
        "counter": "A quick game or RPO under the two-high shell. Don't chase one tell.",
    },
    {
        "id": "D-3RD-SHORT", "situation": "3rd_short", "label": "3rd-and-short",
        "when": "3rd or 4th and 1–3.",
        "score_situation": "Any score. Leading late: fit the run and make them snap it.",
        "want_run": None, "zone": "open", "prefer": ("short_yardage",),
        "adjustments": ("pinch_dline",), "beats": ("run",),
        "counter": "Play-action off the short-yardage look. Don't crash both edges.",
    },
    {
        "id": "D-3RD-LONG", "situation": "3rd_long", "label": "3rd-and-long",
        "when": "3rd or 4th and 7+.",
        "score_situation": "Any score. Trailing late: no all-out blitz that gives up the sticks.",
        "want_run": None, "zone": "open", "prefer": ("long_yardage", "very_long", "home"),
        "adjustments": ("back_off_overtop",), "beats": ("vert",),
        "counter": "A screen or draw on the obvious pass down. Stay in your rush lanes.",
    },
    {
        "id": "D-RED-ZONE", "situation": "red_zone", "label": "Red zone",
        "when": "Inside your 20, including goal-line.",
        "score_situation": "One-score game: a stop matters more than a sack.",
        "want_run": None, "zone": "red_zone", "prefer": ("goal_line", "short_yardage"),
        "adjustments": ("pinch_dline",), "beats": ("run", "man"),
        "counter": "Play-action or a rub at the goal line. Don't sell out on the dive.",
    },
    {
        "id": "D-TWO-MINUTE", "situation": "two_minute", "label": "2-minute",
        "when": "Two-minute drill or the last possession of a half.",
        "score_situation": "Protect a lead: keep the ball in front. Trailing: you still need a stop, not a bomb given up.",
        "want_run": None, "zone": "open", "prefer": ("long_yardage", "home"),
        "adjustments": ("show_blitz_bail",), "beats": ("vert",),
        "counter": "Sideline comebacks and the checkdown. Don't jump a double move.",
    },
    {
        "id": "D-VS-BLITZ", "situation": "vs_blitz", "label": "Pressure package",
        "when": "3rd-and-medium, or after they have thrown hot into your base twice.",
        "score_situation": "Any score. Don't call this when trailing by two scores late.",
        "want_run": None, "zone": "open", "prefer": ("pressure_changeup", "disguise_changeup"),
        "adjustments": ("qb_contain",), "beats": ("pressure",),
        "counter": "Hot route to the back, or max protect. Contain the QB if he keeps.",
    },
    {
        "id": "D-VS-RUN", "situation": "vs_run", "label": "Vs run",
        "when": "Early downs, short yardage, or a logged run tendency.",
        "score_situation": "Any score. Leading late: this is the down to make them throw.",
        "want_run": None, "zone": "open", "prefer": ("short_yardage", "home"),
        "adjustments": ("pinch_dline",), "beats": ("run",),
        "counter": "Outside zone or a toss if you pinch inside. Set the edge.",
    },
    {
        "id": "D-VS-PASS", "situation": "vs_pass", "label": "Vs pass",
        "when": "Obvious pass, or a logged vertical / crosser tendency.",
        "score_situation": "Any score. Trailing: play the sticks, not a hero pick.",
        "want_run": None, "zone": "open", "prefer": ("home", "long_yardage"),
        "adjustments": ("back_off_overtop",), "beats": ("vert", "cross"),
        "counter": "A checkdown or a run draw once the safeties bail. Rally to the ball.",
    },
)

_FAMILY_CALL = {
    "vert": "two_high",
    "flood": "two_high",
    "cross": "man",
    "stack": "single_high",
    "run": "single_high",
    "scram": "man",
    "pressure": "pressure",
    "two_high": "two_high",
    "man": "man",
    "single_high": "single_high",
}


def _patch() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "research" / "madden27.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    patch = data.get("patch") or {}
    return patch if isinstance(patch, dict) else {}


def _in_book(book: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(f, p) for f, plays in (book or {}).items() for p in plays if p]


def _offense_prefers(baseline: dict[str, Any], keys: tuple[str, ...]) -> list[tuple[str, str]]:
    og = baseline.get("offense_gameplan") or {}
    out: list[tuple[str, str]] = []
    for key in keys:
        for entry in (og.get("situations") or {}).get(key) or []:
            out.append((entry.get("formation") or "", entry.get("play") or ""))
        for entry in (og.get("coverage_answers") or {}).get(key) or []:
            out.append((entry.get("formation") or "", entry.get("play") or ""))
    return [(f, p) for f, p in out if f and p]


def _defense_prefers(baseline: dict[str, Any], keys: tuple[str, ...]) -> list[tuple[str, str]]:
    dg = baseline.get("defense_gameplan") or {}
    sit = dg.get("situational") or {}
    out: list[tuple[str, str]] = []
    for key in keys:
        if key == "home":
            pkg = dg.get("home_package") or ""
            for call in dg.get("home_rotation") or {}:
                out.append((pkg, call))
            continue
        block = sit.get(key) or {}
        pkg = block.get("package") or ""
        for call in block.get("calls") or []:
            out.append((pkg, call))
    return [(f, p) for f, p in out if f and p]


def _audible(formation: str, play: str, audibles: dict[str, list[str]], book: dict[str, list[str]], *, run: bool | None) -> str | None:
    """Another play in this formation's audibles that is actually in the book."""
    allowed = set(book.get(formation) or [])
    for candidate in audibles.get(formation) or []:
        if candidate == play or candidate not in allowed:
            continue
        if run is True and not is_run(candidate):
            continue
        if run is False and is_run(candidate):
            continue
        return candidate
    for candidate in audibles.get(formation) or []:
        if candidate != play and candidate in allowed:
            return candidate
    return None


def _adjustment_text(side: str, ids: tuple[str, ...], *, audible_play: str | None) -> str:
    from cfb_coach.madden import research_db as rdb

    if side == "offense":
        catalog = {a.get("id"): a for a in rdb.offense_adjustments()}
        bits: list[str] = []
        for aid in ids:
            adj = catalog.get(aid)
            if not adj:
                continue
            kind = adj.get("type")
            if kind == "audible":
                if not audible_play:
                    continue
                label = f"Audible → {audible_play}"
                buttons = rdb.buttons("offense", "audible").replace("that audible", audible_play)
            elif kind == "hot_route":
                label = f"Hot route {adj.get('target')} → {adj.get('route')}"
                buttons = (rdb.buttons("offense", "hot_route")
                           .replace("the receiver's icon button", f"{adj.get('target')}'s icon button")
                           .replace("pick the route", f"pick {adj.get('route')}"))
            else:
                label = str(adj.get("route") or kind or "").strip() or aid
                buttons = rdb.buttons("offense", "pass_protection").replace("pick the protection", f"pick {adj.get('route')}")
            bits.append(f"{label} — {buttons}")
        return " · ".join(bits) if bits else "No extra pre-snap change — call it as drawn."
    catalog_d = {a.get("id"): a for a in rdb.defense_adjustments()}
    bits = []
    for aid in ids:
        adj = catalog_d.get(aid)
        if not adj:
            continue
        buttons = rdb.buttons("defense", adj["control"])
        if adj.get("extra_control"):
            buttons += " ; then " + rdb.buttons("defense", adj["extra_control"])
        bits.append(f"{adj.get('label')} — {buttons}")
    return " · ".join(bits) if bits else "No extra pre-snap change — rush four, drop seven."


def _team_books(team: str | None) -> dict[str, Any]:
    if not team:
        return {}
    from cfb_coach.madden import catalog
    from cfb_coach.madden.defense_select import call_family

    found: dict[str, Any] = {}
    for side in ("offense", "defense"):
        for name in catalog.book_names(side):
            if catalog.book_team(side, name) != team:
                continue
            forms = catalog.book_formations(side, name)
            calls = [p for plays in forms.values() for p in plays]
            families: dict[str, int] = {}
            if side == "defense":
                for call in calls:
                    fam = call_family(call)
                    if fam:
                        families[fam] = families.get(fam, 0) + 1
            found[side] = {"book": name, "n_formations": len(forms), "n_calls": len(calls), "families": families}
    return found


def _logged(db: Any, opponent_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {"concepts": [], "coverages": [], "families": []}
    if db is None:
        return out
    try:
        from cfb_coach.madden.defense_select import coverage_seen_family
        from cfb_coach.madden.situation import concept_family
        from cfb_coach.scouting import scout_opponent

        report = scout_opponent(db, opponent_id, family_of=concept_family, coverage_class_of=coverage_seen_family)
    except Exception:  # noqa: BLE001
        report = {}
    off = ((report.get("their_offense") or {}).get("overall") or {})
    de = ((report.get("their_defense") or {}).get("overall") or {})
    out["concepts"] = [c.get("concept") for c in (off.get("concepts") or []) if c.get("concept")]
    out["coverages"] = [c.get("coverage") for c in (de.get("coverages") or []) if c.get("coverage")]
    out["families"] = list((off.get("families") or {}).keys())
    try:
        for row in db.get_tendencies(opponent_id):
            key = str(row["key"])
            if int(row["count"]) >= 2 and key not in out["concepts"]:
                out["concepts"].append(key)
    except Exception:  # noqa: BLE001
        pass
    return out


def _learned(db: Any, opponent_id: str, side: str) -> Any:
    if db is None:
        return None
    try:
        from cfb_coach.learning import LearnedWeights

        return LearnedWeights.load(db, opponent_id, side=side)
    except Exception:  # noqa: BLE001
        return None


def _score_pair(
    formation: str,
    play: str,
    slot: dict[str, Any],
    *,
    prefers: list[tuple[str, str]],
    used: set[tuple[str, str]],
    lw: Any,
    favor_family: str | None,
) -> tuple[float, list[str]]:
    from cfb_coach.madden.defense_select import call_family

    why: list[str] = []
    score = 0.0
    if (formation, play) in prefers:
        rank = prefers.index((formation, play))
        score += max(0.4, 2.4 - 0.15 * rank)
        why.append("in the meta menu for this situation")
    want_run = slot.get("want_run")
    if want_run is True:
        score += 1.4 if is_run(play) else -1.6
    elif want_run is False:
        score += 1.1 if not is_run(play) else -1.6
    zone = slot.get("zone") or "open"
    if not zone_fit(play, zone):
        score -= 4.0
    elif zone == "red_zone" and not is_deep(play):
        score += 0.4
    if slot["side"] == "defense" and favor_family:
        if call_family(play) == favor_family:
            score += 1.3
            why.append(f"answers their {favor_family} tendency")
    if (formation, play) in used:
        score -= 5.0
    if lw is not None:
        key = f"play::{formation}::{play}"
        try:
            n = lw.n(key)
            if n:
                norm = lw.norm(key)
                score += 1.1 * norm
                why.append(f"your logs {norm:+.2f} over {n} snap(s)")
        except Exception:  # noqa: BLE001
            pass
    return score, why


def _pick(
    book: dict[str, list[str]],
    slot: dict[str, Any],
    *,
    prefers: list[tuple[str, str]],
    used: set[tuple[str, str]],
    lw: Any,
    favor_family: str | None,
) -> tuple[str, str, list[str]] | None:
    best: tuple[float, str, str, list[str]] | None = None
    for formation, play in _in_book(book):
        score, why = _score_pair(formation, play, slot, prefers=prefers, used=used, lw=lw, favor_family=favor_family)
        if best is None or score > best[0] or (score == best[0] and (formation, play) < (best[1], best[2])):
            best = (score, formation, play, why)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _favor_family(slot: dict[str, Any], *, their_def_families: dict[str, int], logged_families: list[str]) -> str | None:
    if slot["side"] == "defense":
        beats = set(slot.get("beats") or ())
        for fam in logged_families:
            mapped = _FAMILY_CALL.get(fam)
            if mapped and (fam in beats or mapped in beats):
                return mapped
        return None
    if not their_def_families:
        return None
    top = max(their_def_families, key=their_def_families.get)
    if top in (slot.get("beats") or ()):
        return top
    return None


def _slots(side: str, n: int) -> list[dict[str, Any]]:
    base = [dict(slot, side=side) for slot in (_OFFENSE_SLOTS if side == "offense" else _DEFENSE_SLOTS)]
    if n <= len(base):
        return base[: max(0, n)]
    out = list(base)
    for i in range(n - len(base)):
        out.append({
            "id": f"{side[0].upper()}-CHANGEUP-{i + 1}",
            "side": side,
            "situation": "changeup",
            "label": f"Changeup {i + 1}",
            "when": "A different formation/play from the same custom book when the first answer is stale.",
            "score_situation": "Any score.",
            "want_run": None,
            "zone": "open",
            "prefer": ("early_down", "home"),
            "adjustments": (),
            "beats": (),
            "counter": "If this look is the one they just adjusted to, go back to the opener.",
        })
    return out


def build_gameplan(
    *,
    offense_book: dict[str, list[str]] | None,
    defense_book: dict[str, list[str]] | None,
    audibles: dict[str, list[str]] | None = None,
    opp: dict[str, Any] | None = None,
    db: Any = None,
    opponent_id: str = "",
    n: int = DEFAULT_COUNT,
    offense_only: bool = False,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ``n`` offense macros and ``n`` defense macros from the given books."""
    baseline = baseline or load_meta_baseline()
    opp = opp or {}
    n = max(0, int(n))
    audibles = audibles or {}
    offense_book = dict(offense_book or {})
    defense_book = {} if offense_only else dict(defense_book or {})
    team = opp.get("nfl_team") or None
    books = _team_books(team if isinstance(team, str) else None)
    logged = _logged(db, opponent_id)
    patch = _patch()
    patch_bit = ""
    if str(patch.get("version") or "") == "1.007":
        patch_bit = "TU 1.007 improved run-action block targeting — zone and play-action still the base."

    their_def = (books.get("defense") or {}).get("families") or {}
    top_def = max(their_def, key=their_def.get) if their_def else None
    note_bits: list[str] = []
    if team:
        if books.get("defense"):
            dinfo = books["defense"]
            lean = f", lean {top_def}" if top_def else ""
            note_bits.append(
                f"{team} defense book in the catalog is {dinfo['book']} "
                f"({dinfo['n_formations']} formations, {dinfo['n_calls']} calls{lean}). "
                "No separate player-roster file — scheme comes from that book plus your logs."
            )
        elif books.get("offense"):
            note_bits.append(f"{team} is in the catalog on offense ({books['offense']['book']}).")
        else:
            note_bits.append(f"{team} has no catalogued stock book here — using persona, meta, and your logs.")
    else:
        note_bits.append("Opponent NFL team is not set. Pass --opp-team (for example \"Minnesota Vikings\") to tailor the plan.")
    if logged["concepts"]:
        note_bits.append("Logged concepts: " + ", ".join(logged["concepts"][:4]) + ".")
    if logged["coverages"]:
        note_bits.append("Logged coverages: " + ", ".join(logged["coverages"][:4]) + ".")
    if patch.get("version"):
        note_bits.append(f"Meta patch {patch.get('version')} ({patch.get('date') or 'dated in research'}).")

    warnings: list[str] = []
    used_o: set[tuple[str, str]] = set()
    used_d: set[tuple[str, str]] = set()
    offense = _build_side(
        "offense", offense_book, n=n, baseline=baseline, audibles=audibles, used=used_o,
        lw=_learned(db, opponent_id, "offense"), their_def=their_def, logged=logged,
        patch_bit=patch_bit, team=team if isinstance(team, str) else "",
    )
    defense: list[dict[str, Any]] = []
    if not offense_only:
        defense = _build_side(
            "defense", defense_book, n=n, baseline=baseline, audibles={}, used=used_d,
            lw=_learned(db, opponent_id, "defense"), their_def={}, logged=logged,
            patch_bit=patch_bit, team=team if isinstance(team, str) else "",
        )
    if n and len(offense) < n:
        warnings.append(
            f"Offense custom book only had enough plays for {len(offense)} of {n} macros."
        )
    if n and not offense_only and len(defense) < n:
        warnings.append(
            f"Defense custom book only had enough plays for {len(defense)} of {n} macros."
        )
    return {
        "per_side": n,
        "offense": offense,
        "defense": defense,
        "opponent_note": " ".join(note_bits),
        "warnings": warnings,
        "offense_only": offense_only,
        "patch": patch.get("version") or baseline.get("patch") or "",
    }


def _build_side(
    side: str,
    book: dict[str, list[str]],
    *,
    n: int,
    baseline: dict[str, Any],
    audibles: dict[str, list[str]],
    used: set[tuple[str, str]],
    lw: Any,
    their_def: dict[str, int],
    logged: dict[str, Any],
    patch_bit: str,
    team: str,
) -> list[dict[str, Any]]:
    if not book or n <= 0:
        return []
    out: list[dict[str, Any]] = []
    allowed = {(f, p) for f, p in _in_book(book)}
    for slot in _slots(side, n):
        prefers = _offense_prefers(baseline, slot["prefer"]) if side == "offense" else _defense_prefers(baseline, slot["prefer"])
        prefers = [pair for pair in prefers if pair in allowed]
        favor = _favor_family(slot, their_def_families=their_def, logged_families=list(logged.get("families") or []))
        picked = _pick(book, slot, prefers=prefers, used=used, lw=lw, favor_family=favor)
        if picked is None:
            break
        formation, play, why = picked
        if (formation, play) not in allowed:
            continue
        used.add((formation, play))
        check = _audible(
            formation, play, audibles, book,
            run=True if slot.get("want_run") is False else False if slot.get("want_run") is True else None,
        )
        counter = str(slot["counter"]).replace(
            "{audible}", check or "another play already in this formation",
        )
        if side == "offense" and is_run(play) and patch_bit and "TU 1.007" not in " ".join(why):
            why.append(patch_bit)
        if team and their_def and side == "offense":
            top = max(their_def, key=their_def.get)
            if top in (slot.get("beats") or ()):
                why.append(f"{team} defense book leans {top}")
        if logged.get("concepts") and side == "defense" and slot["situation"] in ("opener", "vs_pass", "vs_run"):
            why.append("your logs: " + ", ".join(logged["concepts"][:3]))
        if logged.get("coverages") and side == "offense" and slot["situation"] in ("vs_pass", "3rd_long", "opener"):
            why.append("coverages you've seen: " + ", ".join(logged["coverages"][:3]))
        read = reads_for(play) if side == "offense" else user_job_for(play)
        out.append({
            "id": slot["id"],
            "side": side,
            "situation": slot["situation"],
            "label": slot["label"],
            "formation": formation,
            "play": play,
            "when": slot["when"],
            "score_situation": slot["score_situation"],
            "adjustments": _adjustment_text(side, tuple(slot.get("adjustments") or ()), audible_play=check),
            "read": read,
            "counter": counter,
            "why": "; ".join(why),
        })
    return out


def format_gameplan(gameplan: dict[str, Any] | None) -> str:
    gp = gameplan or {}
    o = list(gp.get("offense") or [])
    d = list(gp.get("defense") or [])
    lines = [
        f"## Game plan — {len(o)} offense + {len(d)} defense (custom book only)",
        gp.get("opponent_note") or "",
    ]
    for warning in gp.get("warnings") or []:
        lines.append(f"WARNING: {warning}")
    lines.append("")
    lines.extend(_format_side("Offense", o))
    lines.append("")
    if not d and gp.get("offense_only"):
        lines.extend(["### Defense", "  N/A — offense only (CPU)"])
    else:
        lines.extend(_format_side("Defense", d))
    return "\n".join(line for line in lines if line is not None)


def _format_side(title: str, macros: list[dict[str, Any]]) -> list[str]:
    if not macros:
        return [f"### {title}", "  (none)"]
    lines = [f"### {title}"]
    for i, m in enumerate(macros, 1):
        lines.append(f"{i}. {m['id']} — {m['label']}")
        lines.append(f"   CALL: {m['play']} ({m['formation']})")
        lines.append(f"   WHEN: {m['when']} {m.get('score_situation') or ''}".rstrip())
        lines.append(f"   PRE-SNAP: {m.get('adjustments') or ''}")
        lines.append(f"   READ: {m.get('read') or ''}")
        lines.append(f"   COUNTER: {m.get('counter') or ''}")
        if m.get("why"):
            lines.append(f"   WHY: {m['why']}")
    return lines


__all__ = ["DEFAULT_COUNT", "build_gameplan", "format_gameplan"]
