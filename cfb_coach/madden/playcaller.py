"""Madden 27 Franchise live play-caller — data-driven from meta_baseline menus.

Same doctrine as CFB (symmetric O + D):
  - One tell = log + mild bump only; bare coverage/play name = previous snap.
  - Coverage-/concept-specific answers only on a REPEATED tendency (2+) this game.
  - Default = base situational call (D&D, field, persona archetype prior).
  - PIVOT after 2 fails (user soft) / 3 fails (hard) on a side.
CPU opponents are offense-only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from cfb_coach.format_call import format_defense, format_offense
from cfb_coach.madden.data import (
    load_meta_baseline,
    load_seed,
    reads_for,
    user_job_for,
)
from cfb_coach.madden.macro_pool import pool_macro as get_macro
from cfb_coach.madden.macros import LOADOUT_N, as_selection, best_for_family, tag_live
from cfb_coach.madden.situation import Situation, concept_family
from cfb_coach.opponents import is_cpu_opponent
from cfb_coach.tendency import describe_coverage_policy, is_repeated_coverage, is_user_opponent


@dataclass
class MaddenCall:
    side: str
    formation: str
    play: str
    adj_or_macro: str
    read_or_user: str
    rationale: str = ""
    suggest_macro: str | None = None
    macro: str | None = None  # untagged macro name armed this snap (for logging)
    macro_info: dict[str, Any] | None = None  # defense macro (of the D 10) + research-built settings + buttons
    adjustment: dict[str, Any] | None = None  # v1.17: pre-snap adjustment (hot route / audible / contain…) + buttons

    def macro_line(self) -> str:
        """'MACRO: TAMPA MABLE — press LB → TAMPA MABLE | <researched settings> · why: …' or ''."""
        if not self.macro:
            return ""
        mi = self.macro_info or {}
        name = mi.get("name") or self.macro
        return (f"MACRO: {name} — press {mi.get('buttons') or f'LB → {name}'} | {mi.get('key') or ''}"
                + (f"  · why: {mi['why']}" if mi.get("why") else ""))

    def adjustment_line(self) -> str:
        """'ADJ: Hot route WR1 → Slant — press Y → WR1's icon button → pick Slant … · why: …' or ''."""
        a = self.adjustment
        if not a:
            return ""
        return f"ADJ: {a['label']} — press {a['buttons']}" + (f"  · why: {a['why']}" if a.get("why") else "")

    def headline(self) -> str:
        """'PLAY: Mesh Post (Gun 5WR Tight) + ADJ: Hot route WR1 → Slant' (offense) or
        'PLAY: Tampa 2 (Nickel Over) + MACRO: TAMPA MABLE' (defense)."""
        head = f"PLAY: {self.play} ({self.formation})"
        if self.macro:
            head += f" + MACRO: {(self.macro_info or {}).get('name') or self.macro}"
        elif self.adjustment:
            head += f" + ADJ: {self.adjustment['label']}"
        return head

    def format(self) -> str:
        if self.side == "offense":
            line = format_offense(self.formation, self.play, self.adj_or_macro, self.read_or_user)
        else:
            line = format_defense(self.formation, self.play, self.adj_or_macro, self.read_or_user)
        if self.suggest_macro:
            line += f"\n  SUGGEST macro: {self.suggest_macro}"
        if self.macro_line():
            line += f"\n  {self.macro_line()}"
        if self.adjustment_line():
            line += f"\n  {self.adjustment_line()}"
        return line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def situation_key(sit: Situation) -> str:
    if sit.goal_line:
        return "goal_line"
    if sit.red_zone:
        return "red_zone"
    if sit.two_minute:
        return "two_minute"
    if sit.down in (3, 4) and sit.long_yardage:
        return "long_yardage"
    if sit.down in (2, 3, 4) and sit.distance is not None and sit.distance <= 2:
        return "short_yardage"
    if sit.down in (3, 4) and sit.short_yardage:
        return "short_yardage"
    return "early_down"


def coverage_class(cov: str | None) -> str | None:
    c = (cov or "").lower()
    if not c:
        return None
    if "pressure" in c or "cover 0" in c or "zero" in c:
        return "pressure"
    if "man" in c or "cover 1" in c:
        return "man"
    if any(k in c for k in ("cover 4", "cover 6", "cover 9", "quarters", "palms", "two-high", "drop")):
        return "two_high"
    if "cover 2" in c or "tampa" in c or "invert" in c or "sink" in c:
        return "cover2"
    if "cover 3" in c or "match" in c or "sky" in c:
        return "single_high"
    return None


_ARCH_PREF = {
    "split_field_zone": "vs_two_high",
    "two_high_money_downs": "vs_two_high",
    "pressure_heavy": "vs_pressure",
    "c2_c3_mixer": "vs_single_high",
}


def _arch(opp: dict[str, Any]) -> str:
    return (opp.get("archetype") or (opp.get("defense_vs_us") or {}).get("archetype") or "unknown").lower()


def _weighted_pick(
    entries: list[dict[str, Any]],
    rng: random.Random,
    *,
    prefer_tag: str | None = None,
    exclude_play: str | None = None,
) -> dict[str, Any]:
    pool = [e for e in entries if e.get("play") != exclude_play] or list(entries)
    weights = [2.0 if prefer_tag and prefer_tag in (e.get("tags") or []) else 1.0 for e in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def _in_book(e: dict[str, Any], book: dict[str, list[str]]) -> bool:
    return e.get("formation") in book and e.get("play") in book[e["formation"]]


def _book_menu(og: dict[str, Any], key: str, book: dict[str, list[str]]) -> tuple[list[dict[str, Any]], str]:
    """Situation menu filtered to the locked book; remenu (never soft-warn) when empty."""
    menu = [e for e in og["situations"][key] if _in_book(e, book)]
    if menu:
        return menu, ""
    menu = [e for e in og["situations"]["early_down"] if _in_book(e, book)]
    if menu:
        return menu, f"remenu: no {key.replace('_', ' ')} call in book → early-down menu"
    menu = [e for m in og["situations"].values() for e in m if _in_book(e, book)]
    if menu:
        return menu, "remenu: any in-book menu call"
    menu = [{"formation": f, "play": p, "tags": []} for f, plays in book.items() for p in plays]
    return menu, "remenu: raw locked-book plays"


# ---------------------------------------------------------------------------
# Offense
# ---------------------------------------------------------------------------

BOOK_MENU_BONUS = 0.12  # CFB parity: play is in the situational meta menu
SIT_BONUS = 0.12
PERSONA_BONUS = 0.06
LIVE_COVERAGE_BONUS = 0.10


def madden_priors(research: dict[str, Any] | None = None) -> Any:
    """CFB MetaPriors, Madden flavored: seed = the cited meta menus / coverage answers, live =
    this prep's research (named formations + plays, web + YouTube), Madden zone-fit."""
    from cfb_coach.madden.catalog import zone_fit
    from cfb_coach.meta_align import MetaPriors

    mp = MetaPriors(fit=zone_fit)
    try:
        og = load_meta_baseline()["offense_gameplan"]
        zmap = {"goal_line": "gl", "red_zone": "rz"}
        for key, menu in (og.get("situations") or {}).items():
            z = zmap.get(key, "open")
            for e in menu:
                rec = mp.seed.setdefault((e["formation"], e["play"]), {
                    "formation": e["formation"], "play": e["play"], "zones": {}, "refs": [],
                    "why": "Madden 27 meta menu (cited baseline)", "coverage": {}})
                rec["zones"][z] = max(float(rec["zones"].get(z, 0.0)), 0.15)
        for cls, answers in (og.get("coverage_answers") or {}).items():
            for e in answers:
                rec = mp.seed.setdefault((e["formation"], e["play"]), {
                    "formation": e["formation"], "play": e["play"], "zones": {}, "refs": [],
                    "why": "Madden 27 coverage answer", "coverage": {}})
                rec.setdefault("coverage", {})[{"single_high": "cover 3", "two_high": "cover 4", "cover2": "cover 2",
                                                "man": "man", "pressure": "pressure"}.get(cls, cls)] = 0.2
        if research is None:
            research = _cached_research()
        from cfb_coach.madden.playbook import side_named

        named = side_named(research, "offense")
        mp.live_mode = (research or {}).get("mode") or "none"
        if named:
            mp.add_named_boosts(named)
            mp.add_named_formations(named)
    except Exception:  # noqa: BLE001 — never break play calling
        pass
    return mp


def _cached_research() -> dict[str, Any]:
    try:
        from cfb_coach.madden.meta_scout import _load_cache, research_from_scout

        cached = _load_cache()
        return research_from_scout(cached) if cached else {}
    except Exception:  # noqa: BLE001
        return {}


class _MaddenRanker:
    """The CFB `_Ranker` (learned zone weights + decayed meta prior + softmax/explore) with
    Madden priors and learned weights from the Madden DB."""

    def __init__(self, db: Any, opponent_id: str, side: str = "offense") -> None:
        from cfb_coach.learning import LearnedWeights

        try:
            self.lw = LearnedWeights.load(db, opponent_id, side=side) if db is not None else LearnedWeights.empty()
        except Exception:  # noqa: BLE001
            self.lw = LearnedWeights.empty()
        self.priors = madden_priors()

    def rank(self, sit: Situation, pool: list[tuple[str, str]], bonus: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
        from cfb_coach.playcaller import _Ranker

        return _Ranker.rank(self, sit, pool, bonus)  # type: ignore[arg-type]


def _offense_pool(
    sit: Situation, book: dict[str, list[str]], og: dict[str, Any], key: str, arch_pref: str | None,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], float]]:
    """Every zone-fit play in the locked formations (CFB v1.14 pool); the situational meta menu
    and down-and-distance are bonuses, not filters."""
    import re

    from cfb_coach.madden.catalog import is_deep, is_run, zone_fit
    from cfb_coach.zones import zone_of_situation

    zone = zone_of_situation(sit)
    pairs = [(f, p) for f, ps in book.items() for p in ps]
    pool = [fp for fp in pairs if zone_fit(fp[1], zone)] or pairs
    short = bool(sit.short_yardage or (sit.down in (2, 3, 4) and (sit.distance or 10) <= 2))
    if zone == "open" and not short:
        pool = [fp for fp in pool if not re.search(r"\bsneak\b|goal\s*line", fp[1], re.I)] or pool
    menu = {(e["formation"], e["play"]): e for e in og["situations"].get(key, [])}
    any_menu = {(e["formation"], e["play"]) for m in og["situations"].values() for e in m}
    bonus: dict[tuple[str, str], float] = {}
    for f, p in pool:
        b = 0.0
        if (f, p) in menu:
            b += BOOK_MENU_BONUS
            if arch_pref and arch_pref in (menu[(f, p)].get("tags") or []):
                b += PERSONA_BONUS
        elif (f, p) in any_menu:
            b += 0.03
        run, deep = is_run(p), is_deep(p)
        dist = sit.distance or 10
        if short:
            b += SIT_BONUS if run else 0.0
            b -= 0.10 if deep else 0.0
        elif sit.down in (3, 4) and dist >= 7:
            b += SIT_BONUS if not run else -0.20
        if sit.two_minute:
            b += 0.08 if not run else -0.05
        from cfb_coach.game_score import offense_score_bonus

        b += offense_score_bonus(sit, p, run=run, deep=deep)
        if b:
            bonus[(f, p)] = round(b, 3)
    return pool, bonus


SCOUT_MIN_SHARE = 0.45


def _scouted_coverage_bonus(
    sit: Situation,
    oid: str,
    db: Any,
    og: dict[str, Any],
    pool: list[tuple[str, str]],
    bonus: dict[tuple[str, str], float],
) -> str:
    """No look typed: lean toward the cited answers to the coverage this opponent usually
    plays in this down bucket (your logs). Mutates ``bonus``; returns a rationale note."""
    if db is None:
        return ""
    try:
        from cfb_coach.scouting import expected_coverage, scout_opponent

        report = scout_opponent(db, oid, coverage_class_of=coverage_class)
        classes, scope, n = expected_coverage(report, sit)
    except Exception:  # noqa: BLE001
        return ""
    if not classes or n < 4:
        return ""
    cls, share = max(classes.items(), key=lambda kv: kv[1])
    if share < SCOUT_MIN_SHARE:
        return ""
    answers = {(e["formation"], e["play"]) for e in og["coverage_answers"].get(cls, [])}
    try:
        from cfb_coach.ai_research import opponent_counters

        named = {p for rc in opponent_counters("madden27", oid, "offense") if rc.get("vs_class") == cls for p in rc.get("plays") or []}
        answers |= {fp for fp in pool if fp[1] in named}
    except Exception:  # noqa: BLE001
        pass
    hit = 0
    for fp in pool:
        if fp in answers:
            bonus[fp] = round(bonus.get(fp, 0.0) + LIVE_COVERAGE_BONUS * share, 3)
            hit += 1
    return f"scout {scope}: they play {cls} {round(share * 100)}% (n={n})" + (f" — {cls} answers weighted" if hit else "")


def _pick_offense(
    sit: Situation,
    opp: dict[str, Any],
    bl: dict[str, Any],
    db: Any,
    rng: random.Random,
    active: list[str],
    book: dict[str, list[str]],
    audibles: dict[str, list[str]] | None = None,
) -> MaddenCall:
    """User games may arm one offense Custom Adjustment from ``active`` when the play and look match."""
    from cfb_coach.madden.catalog import is_run

    og = bl["offense_gameplan"]
    oid = opp["_id"]
    arch = _arch(opp)
    key = situation_key(sit)
    pref = _ARCH_PREF.get(arch)
    pool, bonus = _offense_pool(sit, book, og, key, pref)

    cov = sit.coverage_hint
    src = sit.coverage_source or "none"
    cls = coverage_class(cov) if cov else None
    repeated = bool(cov) and is_repeated_coverage(db, oid, cov, sit, threshold=2)
    policy = describe_coverage_policy(cov, live=src == "live", last_snap=src == "last", repeated=repeated) if cov else ""
    answered_repeat = False
    if cov and src == "live" and cls:
        # soft lean vs a live look: menu plays tagged for it + cited coverage answers in the book
        for e in og["situations"].get(key, []) + og["coverage_answers"].get(cls, []):
            fp = (e["formation"], e["play"])
            if fp in pool and (f"vs_{cls}" in (e.get("tags") or []) or e in og["coverage_answers"].get(cls, [])):
                bonus[fp] = round(bonus.get(fp, 0.0) + LIVE_COVERAGE_BONUS * (2.0 if repeated else 1.0), 3)
        answered_repeat = repeated and any(_in_book(a, book) for a in og["coverage_answers"].get(cls, []))
    scout_note = "" if cov else _scouted_coverage_bonus(sit, oid, db, og, pool, bonus)

    ranker = _MaddenRanker(db, oid, "offense")
    rows = ranker.rank(sit, pool, bonus)
    from cfb_coach import ingame
    from cfb_coach.playcaller import _sample

    bench = ingame.bench_for_situation(db, sit, "offense")
    rows = ingame.drop_benched(rows, bench)
    pick = _sample(rows, rng) if rows else {"formation": next(iter(book)), "play": book[next(iter(book))][0], "total": 0.0}
    form, play = pick["formation"], pick["play"]
    rationale = (f"{key.replace('_', ' ')}: ranked {len(rows)} in-book plays (learned {pick.get('learned', 0):+.2f}, "
                 f"meta {pick.get('meta', 0):+.2f}, sit {pick.get('sit', 0):+.2f})")
    if pref:
        rationale += f" | persona {arch} → {pref}"
    if cov:
        rationale += (f" | REPEATED {cov} — {cls} answer weighted" if answered_repeat else "") + f" | {policy}"
    if scout_note:
        rationale += f" | {scout_note}"
    if bench.benched:
        rationale += f" | {bench.note()}"

    # Anti-repeat: same play 2+ of last 4 → next best different play
    if db is not None:
        from cfb_coach.gameplan import anti_repeat_penalty

        pen = anti_repeat_penalty(db, oid, form, play, side="offense")
        if pen >= 1.0:
            alt = next((r for r in rows if r["play"] != play), None)
            if alt:
                rationale = f"anti-repeat → {alt['formation']}/{alt['play']} (pen={pen:.1f}) | {rationale}"
                form, play = alt["formation"], alt["play"]

    pivot = active_pivot(db, oid, "offense")
    if pivot and answered_repeat:
        rationale = f"{pivot} — REPEATED answer is the switch | {rationale}"
    elif pivot:
        was_run = is_run(play)
        options = [r for r in rows if is_run(r["play"]) != was_run and r["play"] != play] or [r for r in rows if r["play"] != play]
        if options:
            alt = options[0] if rng.random() < 0.6 else rng.choice(options[:4])
            form, play = alt["formation"], alt["play"]
        rationale = f"{pivot} → {form}/{play} | {rationale}"

    if not _in_book({"formation": form, "play": play}, book):  # hard guard — never leave the book
        form = next(iter(book))
        play = book[form][0]
        rationale += " | remenu: guard pulled call back into locked book"

    adj = "Hot ready" if cls == "pressure" and src == "live" else "No adj"
    adjustment = None
    macro = None
    info = None
    zone = "gl" if sit.goal_line else "rz" if sit.red_zone else "open"
    try:
        from cfb_coach.madden.offense_macros import suggest_for_snap

        weights = {}
        if db is not None:
            weights = {str(r["macro"]).upper(): float(r["weight"] or 0.0) for r in db.get_macro_weights(oid)}
        info = suggest_for_snap(zone=zone, play=play, coverage=cov, coverage_source=src, active=active,
                                down=sit.down, repeated=repeated, book=book, weights=weights)
    except Exception:  # noqa: BLE001 — never break a call
        info = None
    if info:
        macro = info["id"]
        adj = "No adj"
        rationale += f" | MACRO {info['name']}: {info['why']}"
    else:
        try:  # one researched pre-snap adjustment when no Custom Adjustment matches the look
            from cfb_coach.madden.adjustments import offense_adjustment

            adjustment = offense_adjustment(play=play, formation=form, coverage_class=cls, coverage_source=src,
                                            repeated=repeated, audibles=audibles)
        except Exception:  # noqa: BLE001 — never break a call
            adjustment = None
        if adjustment:
            adj = adjustment["label"]
            rationale += f" | adj {adjustment['label']}: {adjustment['why']}"
    return MaddenCall("offense", form, play, adj, reads_for(play), rationale, macro=macro, macro_info=info,
                      adjustment=adjustment)


# ---------------------------------------------------------------------------
# Defense
# ---------------------------------------------------------------------------

_SOFT_SHELL = {
    "vert": ("Cover 4 Quarters", "one vertical tell — Quarters soft lean"),
    "flood": ("Cover 3 Match", "one flood/sail tell — Match soft lean"),
    "cross": ("Tampa 2", "one mesh/crosser tell — Tampa 2 soft lean"),
    "stack": ("Cover 3 Match", "one stack/bunch tell — Match soft lean"),
    "scram": (None, "one scramble tell — contain job"),
    "run": (None, "one zone-run tell — fit from base shell"),
    "rpo": (None, "one RPO/bubble tell — base zone, flats honest"),
}


def _base_defense(sit: Situation, opp: dict[str, Any], bl: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    dg = bl["defense_gameplan"]
    st = dg["situational"]
    arch = _arch(opp)
    if sit.goal_line:
        return st["goal_line"]["package"], st["goal_line"]["calls"][0], "GL — 4-3 Over heavy; don't sell out vs stack/PA"
    if sit.short_yardage and sit.down in (3, 4) and (sit.distance or 9) <= 2:
        return st["short_yardage"]["package"], rng.choice(st["short_yardage"]["calls"]), "short yardage — 4-3 Over (no auto RUN-FIT on one tell)"
    if sit.distance and sit.distance >= 12 and sit.down in (2, 3, 4) and rng.random() < 0.35:
        return st["very_long"]["package"], st["very_long"]["calls"][0], "true long passing down — Dime Quarters, protect deep"
    if (sit.long_yardage and sit.down in (3, 4)) or (sit.down == 3 and (sit.distance or 0) >= 8):
        return st["long_yardage"]["package"], rng.choice(st["long_yardage"]["calls"]), "obvious pass D&D — Dime 3-2 Odd"
    if arch == "pressure_heavy" and sit.down == 3 and 3 <= (sit.distance or 0) <= 6 and rng.random() < 0.2:
        pc = st["pressure_changeup"]
        return pc["package"], rng.choice(pc["calls"]), "3rd-medium vs pressure persona — sim/stunt off the mug look (rare, not contain-four)"
    rot = dg["home_rotation"]
    play = rng.choices(list(rot), weights=list(rot.values()), k=1)[0]
    rationale = "Nickel Over home (C4 Quarters / C3 Match / Tampa 2 — rush four, drop seven)"
    if arch == "split_field_zone" and rng.random() < 0.5:
        play, rationale = "Cover 4 Quarters", "vs explosive-shot persona — Quarters prior"
    return dg["home_package"], play, rationale


def _family_repeat_count(db: Any, oid: str, family: str | None) -> int:
    """Live tells of this concept family (this-game window for users; career for CPU)."""
    if db is None or not family:
        return 0
    n = sum(
        1
        for s in db.get_recent_snaps(oid, limit=12)
        if concept_family(s["concept_seen"]) == family
    )
    if not is_user_opponent(oid):
        n = max(
            n,
            sum(
                int(r["count"])
                for r in db.get_tendencies(oid, "offense_concept")
                if concept_family(r["key"]) == family
            ),
        )
    return n


def _pkg_family(f: str) -> str:
    return (f.split()[0] if f else "").lower()


def _fit_defense(
    form: str,
    play: str,
    book: dict[str, list[str]],
    bl: dict[str, Any],
    rng: random.Random,
) -> tuple[str, str, str]:
    """Pull a (package, call) back inside the locked defensive book (remenu, never warn).
    Works for any recommended D book: same call in a same-family package first (Nickel/Dime/
    4-3/Goal Line), then any package with that call, then the same coverage family."""
    if form in book and play in book[form]:
        return form, play, ""
    fam = _pkg_family(form)
    holders = [f for f in book if play in book[f]]
    if holders:
        same = [f for f in holders if _pkg_family(f) == fam]
        home = bl["defense_gameplan"]["home_package"]
        f = home if home in holders else (same[0] if same else holders[0])
        return f, play, f"remenu: {form} not in book → {f}"
    pkgs = [f for f in book if _pkg_family(f) == fam] or [f for f in book if f == bl["defense_gameplan"]["home_package"]] or list(book)
    pkg = pkgs[0]
    cls = coverage_class(play)
    rot = [c for c in bl["defense_gameplan"]["home_rotation"] if c in book[pkg]]
    same_cov = [c for c in book[pkg] if cls and coverage_class(c) == cls]
    call = rng.choice(same_cov) if same_cov else (rng.choice(rot) if rot else book[pkg][0])
    return pkg, call, f"remenu: {form}/{play} not in book → {pkg}/{call}"


def _select_base_defense(
    sit: Situation,
    oid: str,
    db: Any,
    bl: dict[str, Any],
    rng: random.Random,
    book: dict[str, list[str]],
    *,
    exclude: set[str] | None = None,
) -> Any:
    """Whole-book defense pick (``defense_select``); None → legacy Nickel Over rotation."""
    try:
        from cfb_coach.ai_research import opponent_counters
        from cfb_coach.learning import LearnedWeights
        from cfb_coach.madden.defense_select import coverage_seen_family, select_defense
        from cfb_coach.scouting import scout_opponent

        report = scout_opponent(db, oid, family_of=concept_family, coverage_class_of=coverage_seen_family) if db is not None else None
        lw = LearnedWeights.load(db, oid, side="defense") if db is not None else None
        return select_defense(
            sit, oid, db, book, rng, baseline=bl, scouting=report,
            research_counters=opponent_counters("madden27", oid, "defense"),
            exclude_families=exclude, lw=lw,
        )
    except Exception:  # noqa: BLE001 — never break a live call
        return None


def _pick_defense(
    sit: Situation,
    opp: dict[str, Any],
    bl: dict[str, Any],
    db: Any,
    rng: random.Random,
    active: list[str],
    book: dict[str, list[str]],
) -> MaddenCall:
    """``active`` = the defense Custom Adjustments (prep rank order) — the only D macros this call may arm."""
    oid = opp["_id"]
    picked = _select_base_defense(sit, oid, db, bl, rng, book)
    if picked is not None:
        form, play, rationale, user = picked.formation, picked.play, picked.rationale, picked.user_job
    else:
        form, play, rationale = _base_defense(sit, opp, bl, rng)
        user = user_job_for(play)
    form, play, note = _fit_defense(form, play, book, bl, rng)
    if note:
        rationale += f" | {note}"
        user = user_job_for(play)
    macro: str | None = None
    suggest: str | None = None
    d_adj: dict[str, Any] | None = None

    concept = sit.concept_hint
    fam = concept_family(concept)
    weights: dict[str, float] = {}
    if db is not None:
        try:
            weights = {str(r["macro"]).upper(): float(r["weight"] or 0.0) for r in db.get_macro_weights(oid)}
        except Exception:  # noqa: BLE001
            weights = {}
    # The family's best macro among the defense loadout (prep rank order). When none of them
    # answers it, the research DB's best one is only named as a suggestion (re-prep to carry it).
    from cfb_coach.madden.macro_pool import pool_ids

    by_rank = sorted(pool_ids("defense"), key=lambda m: (get_macro(m) or {}).get("meta_rank", 99))
    fam_macro = best_for_family(fam, active, weights) or best_for_family(fam, by_rank, weights)
    if concept:
        src = sit.concept_source or "none"
        repeated = _family_repeat_count(db, oid, fam) >= 2
        if src == "last":
            # Previous-snap tell never arms a macro this snap (CFB parity)
            rationale += f" | last snap {concept} — mild bump only; stay base"
            if fam_macro and repeated:
                suggest = f"{fam_macro} — REPEATED {concept}; arm when they show it live again"
            elif fam_macro:
                suggest = f"{fam_macro} — only after repeated {concept} in similar D&D"
        elif not repeated:
            shell, note = _SOFT_SHELL.get(fam or "", (None, f"one {concept} tell — logged/soft only"))
            if shell and form == bl["defense_gameplan"]["home_package"]:
                play = shell
                user = user_job_for(play)
            if fam == "scram":
                user = "User QB spy"
            rationale += f" | {note}; no macro until repeated"
            if fam_macro:
                suggest = f"{fam_macro} — after one more {concept} tell"
        elif fam_macro:
            if fam_macro in active:
                base_play = str(((get_macro(fam_macro) or {}).get("base") or {}).get("play") or "")
                if base_play and base_play in {p for ps in book.values() for p in ps}:
                    holders = [f for f, ps in book.items() if base_play in ps]  # researched base play
                    home = bl["defense_gameplan"]["home_package"]
                    form, play = (home if home in holders else holders[0]), base_play
                else:
                    shell = (_SOFT_SHELL.get(fam or "") or (None, ""))[0]
                    if fam == "run" and sit.short_yardage:
                        form, play = "4-3 Over", "Cover 3 Sky"
                    elif shell:
                        form, play = "Nickel Over", shell
                macro = fam_macro
                user = (get_macro(fam_macro) or {}).get("user_job") or user_job_for(play)
                rationale = f"REPEATED {fam} tendency ({concept}) — {play} + {fam_macro}"
            else:
                suggest = f"{fam_macro} — repeated {concept} but not in your defense macros; re-prep to add it"
                rationale += f" | REPEATED {concept} — {fam_macro} not in the defense macros"
                d_adj = _d_adjustment(fam, concept)
        else:
            rationale += f" | REPEATED {concept} — no defense macro answers it"
            d_adj = _d_adjustment(fam, concept)

    pivot = active_pivot(db, oid, "defense")
    if pivot and macro:
        # Arming the REPEATED-tendency macro is the family switch — keep it
        rationale = f"{pivot} — keep {macro} (REPEATED answer is the switch) | {rationale}"
    elif pivot:
        from cfb_coach.madden.defense_select import call_family

        switched = _select_base_defense(sit, oid, db, bl, rng, book, exclude={call_family(play)} - {None})
        if switched is not None:
            form, play, user = switched.formation, switched.play, switched.user_job
        else:
            dg = bl["defense_gameplan"]
            form, play = dg["home_package"], rng.choice(list(dg["home_rotation"]))
            user = user_job_for(play)
        macro, d_adj = None, None
        rationale = f"{pivot} → {form}/{play} (new coverage family) | {rationale}"
        suggest = suggest or "macros cleared — re-arm only on repeated tendency"

    fitted = _fit_defense(form, play, book, bl, rng)
    if fitted[:2] != (form, play):
        form, play = fitted[0], fitted[1]
        if not macro:
            user = user_job_for(play)
        rationale += f" | {fitted[2]}"

    macro_out = tag_live(macro, db) if macro else "none"
    info = None
    if macro:
        from cfb_coach.madden.macros import macro_info

        info = macro_info(macro, f"REPEATED {fam} tendency ({concept}) — {play}")
    if suggest:
        head, _, rest = suggest.partition(" — ")
        if get_macro(head):
            suggest = tag_live(head, db) + (f" — {rest}" if rest else "")
    if d_adj and not macro:
        rationale += f" | adj {d_adj['label']}"
    return MaddenCall("defense", form, play, macro_out, user, rationale, suggest, macro=macro, macro_info=info,
                      adjustment=None if macro else d_adj)


def _d_adjustment(fam: str | None, concept: str | None) -> dict[str, Any] | None:
    try:
        from cfb_coach.madden.adjustments import defense_adjustment

        return defense_adjustment(fam, why=f"REPEATED {concept}")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Pivot (Madden wording; shared fail-streak logic)
# ---------------------------------------------------------------------------

def active_pivot(db: Any, opponent_id: str, side: str) -> str | None:
    if db is None:
        return None
    from cfb_coach.gameplan import _side_fail_streak

    hard = _side_fail_streak(db, opponent_id, side, n=3) >= 3
    soft = is_user_opponent(opponent_id) and _side_fail_streak(db, opponent_id, side, n=2) >= 2
    if not (hard or soft):
        return None
    tag = "PIVOT:" if hard else "PIVOT (user game):"
    n = 3 if hard else 2
    if side == "offense":
        return f"{tag} last {n} O snaps failed — switch family (zone run ↔ Mesh Post ↔ Mtn Stick Wheel), no hero shot"
    return f"{tag} last {n} D snaps failed — switch coverage family, clear chase macros"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _sit_stamp(sit: Situation) -> str:
    bits: list[str] = []
    if sit.down and sit.distance is not None:
        bits.append(f"{sit.down}&{sit.distance}")
    if sit.yardline is not None:
        bits.append(f"yl{sit.yardline}")
    if sit.goal_line:
        bits.append("GL")
    elif sit.red_zone:
        bits.append("RZ")
    return "sit " + " ".join(bits) if bits else ""


def make_call(
    sit: Situation,
    opponent_id: str,
    db: Any = None,
    *,
    seed: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
    rng: random.Random | None = None,
    last_coverage: str | None = None,
    last_concept: str | None = None,
    active_macros: list[str] | None = None,
    playbook: dict[str, dict[str, list[str]]] | None = None,
) -> MaddenCall:
    """One call. `playbook` = {side: {formation: [plays]}} locked by prep; defaults to the
    DB's locked (applied) books. Raises NoActivePlaybook when the needed side has no
    locked book — never calls freely. Every returned formation/play is in that book."""
    from cfb_coach.madden.playbook import active_books, eligible

    seed = seed or load_seed()
    bl = baseline or load_meta_baseline()
    opp = dict((seed.get("opponents") or {}).get(opponent_id) or {})
    if db is not None:
        prof = db.get_opponent(opponent_id)
        if prof:
            opp = dict(prof)
    opp["_id"] = opponent_id
    rng = rng or random.Random()
    # Each side only arms Custom Adjustments from its own list.
    if active_macros:
        sel = as_selection(active_macros)
    else:  # no prep yet: research DB's top defense macros. Offense waits for a prep.
        from cfb_coach.madden.macro_pool import pool_ids

        top = sorted(pool_ids("defense"), key=lambda m: (get_macro(m) or {}).get("meta_rank", 99))[:LOADOUT_N]
        sel = {"offense": [], "defense": top}

    if last_coverage and not sit.coverage_hint:
        sit.coverage_hint, sit.coverage_source = last_coverage, "last"
    if last_concept and not sit.concept_hint:
        sit.concept_hint, sit.concept_source = last_concept, "last"

    cpu = is_cpu_opponent(opponent_id)
    note = ""
    if cpu:
        sel["defense"] = []
        if sit.side == "defense":
            sit.side = "offense"
            note = "CPU = offense-only (no D calls) — switched to O | "

    audibles: dict[str, list[str]] | None = None
    if playbook is None:
        raw = active_books(db, (sit.side,))
        books = eligible(raw)
        audibles = (raw.get("offense") or {}).get("audibles")
    else:
        books = playbook
    if sit.side not in books:
        from cfb_coach.madden.playbook import NoActivePlaybook

        raise NoActivePlaybook(f"No {sit.side} playbook given — run `prep --game madden27` first.")
    if sit.side == "defense":
        call = _pick_defense(sit, opp, bl, db, rng, sel["defense"], books["defense"])
    else:
        call = _pick_offense(sit, opp, bl, db, rng, sel["offense"], books["offense"], audibles=audibles)
    stamp = _sit_stamp(sit)
    call.rationale = note + call.rationale + (f" | {stamp}" if stamp else "")
    from cfb_coach.game_score import score_call_note

    snote = score_call_note(sit)
    if snote:
        call.rationale += f" | {snote}"
    return call
