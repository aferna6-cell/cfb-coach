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
    get_macro,
    load_meta_baseline,
    load_seed,
    reads_for,
    user_job_for,
)
from cfb_coach.madden.macros import macro_side, tag_live
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

    def format(self) -> str:
        if self.side == "offense":
            line = format_offense(self.formation, self.play, self.adj_or_macro, self.read_or_user)
        else:
            line = format_defense(self.formation, self.play, self.adj_or_macro, self.read_or_user)
        if self.suggest_macro:
            line += f"\n  SUGGEST macro: {self.suggest_macro}"
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

def _pick_offense(
    sit: Situation,
    opp: dict[str, Any],
    bl: dict[str, Any],
    db: Any,
    rng: random.Random,
    active: list[str],
    book: dict[str, list[str]],
) -> MaddenCall:
    og = bl["offense_gameplan"]
    oid = opp["_id"]
    arch = _arch(opp)
    key = situation_key(sit)
    menu, remenu = _book_menu(og, key, book)
    pref = _ARCH_PREF.get(arch)
    entry = _weighted_pick(menu, rng, prefer_tag=pref)
    form, play, adj = entry["formation"], entry["play"], entry.get("adj") or "No adj"
    rationale = f"{key.replace('_', ' ')} menu" + (f" (persona {arch} → {pref})" if pref else "")
    if remenu:
        rationale += f" | {remenu}"
    o_macro: str | None = None
    answered_repeat = False

    # Pressure persona: protection macro on some passing downs (still soft)
    if arch == "pressure_heavy" and "pass" in (entry.get("tags") or []) and rng.random() < 0.4:
        if "O-PROT" in active:
            o_macro = "O-PROT"
            rationale += " | pressure persona — O-PROT"

    cov = sit.coverage_hint
    if cov:
        src = sit.coverage_source or "none"
        repeated = is_repeated_coverage(db, oid, cov, sit, threshold=2)
        live = src == "live"
        policy = describe_coverage_policy(cov, live=live, last_snap=src == "last", repeated=repeated)
        cls = coverage_class(cov)
        answers = [a for a in og["coverage_answers"].get(cls or "", []) if _in_book(a, book)]
        if live and repeated and answers:
            ans = rng.choice(answers)
            form, play = ans["formation"], ans["play"]
            adj = ans.get("adj") or adj
            if cls == "man" and "O-MAN" in active:
                o_macro = "O-MAN"
            if cls == "pressure" and adj == "O-PROT":
                o_macro = "O-PROT" if "O-PROT" in active else None
                adj = "Hot ready" if o_macro is None else adj
            rationale = f"REPEATED {cov} — {cls} answer | {policy}"
            answered_repeat = True
        elif live and cls:
            fits = [e for e in menu if f"vs_{cls}" in (e.get("tags") or [])]
            if fits:
                entry = _weighted_pick(fits, rng)
                form, play = entry["formation"], entry["play"]
                rationale = f"soft lean vs live {cov} ({cls}) | {policy}"
            else:
                rationale = f"{rationale} | {policy}"
            if cls == "pressure":
                adj = "Hot ready"
        else:
            rationale = f"{rationale} | {policy}"

    # Anti-repeat: same play 2+ of last 4 → rotate within the in-book menu
    if db is not None:
        from cfb_coach.gameplan import anti_repeat_penalty

        pen = anti_repeat_penalty(db, oid, form, play, side="offense")
        if pen >= 1.0:
            alt = _weighted_pick(menu, rng, exclude_play=play)
            rationale = f"anti-repeat → {alt['formation']}/{alt['play']} (pen={pen:.1f}) | {rationale}"
            form, play = alt["formation"], alt["play"]

    pivot = active_pivot(db, oid, "offense")
    if pivot and answered_repeat:
        rationale = f"{pivot} — REPEATED answer is the switch | {rationale}"
    elif pivot:
        fams = og["pivot_families"]
        was_run = play in fams["run"]["plays"] or play in ("HB Zone WK", "HB Dive", "HB Stretch", "Mid Zone")
        order = ["quick", "stick", "run"] if was_run else ["run", "stick", "quick"]
        if rng.random() < 0.35:
            order.insert(0, "stick")
        options: list[tuple[str, str]] = []
        for fam_name in order:
            fam = fams[fam_name]
            options = [(fam["formation"], p) for p in fam["plays"] if _in_book({"formation": fam["formation"], "play": p}, book)]
            if options:
                break
        if not options:  # family not in book → switch run ↔ pass within the in-book menu
            want = "pass" if was_run else "run"
            options = [(e["formation"], e["play"]) for e in menu if want in (e.get("tags") or [])] or [
                (e["formation"], e["play"]) for e in menu if e["play"] != play
            ] or [(form, play)]
        form, play = rng.choice(options)
        adj, o_macro = "No adj", None
        rationale = f"{pivot} → {form}/{play} | {rationale}"

    if not _in_book({"formation": form, "play": play}, book):  # hard guard — never leave the book
        entry = _weighted_pick(menu, rng)
        form, play = entry["formation"], entry["play"]
        rationale += " | remenu: guard pulled call back into locked book"
    if o_macro:
        adj = tag_live(o_macro, db)
    return MaddenCall("offense", form, play, adj, reads_for(play), rationale, macro=o_macro)


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


def _fit_defense(
    form: str,
    play: str,
    book: dict[str, list[str]],
    bl: dict[str, Any],
    rng: random.Random,
) -> tuple[str, str, str]:
    """Pull a (package, call) back inside the locked defensive book (remenu, never warn)."""
    if form in book and play in book[form]:
        return form, play, ""
    home = bl["defense_gameplan"]["home_package"]
    holders = [f for f in book if play in book[f]]
    if holders:
        f = home if home in holders else holders[0]
        return f, play, f"remenu: {form} not in book → {f}"
    pkg = form if form in book else (home if home in book else next(iter(book)))
    rot = [c for c in bl["defense_gameplan"]["home_rotation"] if c in book[pkg]]
    call = rng.choice(rot) if rot else book[pkg][0]
    return pkg, call, f"remenu: {form}/{play} not in book → {pkg}/{call}"


def _pick_defense(
    sit: Situation,
    opp: dict[str, Any],
    bl: dict[str, Any],
    db: Any,
    rng: random.Random,
    active: list[str],
    book: dict[str, list[str]],
) -> MaddenCall:
    oid = opp["_id"]
    form, play, rationale = _base_defense(sit, opp, bl, rng)
    form, play, note = _fit_defense(form, play, book, bl, rng)
    if note:
        rationale += f" | {note}"
    macro: str | None = None
    user = user_job_for(play)
    suggest: str | None = None

    concept = sit.concept_hint
    fam = concept_family(concept)
    fam_macro = (bl["macros_baseline"].get("concept_family_macro") or {}).get(fam or "")
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
                cat_pair = {
                    "MATCH-4": ("Nickel Over", "Cover 4 Quarters"),
                    "FLAT-CAP": ("Nickel Over", "Cover 3 Match"),
                    "MESH-RAT": ("Nickel Over", "Tampa 2"),
                    "STACK": ("Nickel Over", "Cover 3 Match"),
                    "SPY": ("Nickel Over", "Cover 4 Quarters"),
                    "RUN-FIT": ("4-3 Over", "Cover 3 Sky") if sit.short_yardage else (form, play),
                }
                form, play = cat_pair.get(fam_macro, (form, play))
                macro = fam_macro
                user = (get_macro(fam_macro) or {}).get("user_job") or user_job_for(play)
                rationale = f"REPEATED {fam} tendency ({concept}) — {play} + {fam_macro}"
            else:
                suggest = f"{fam_macro} — repeated {concept} but benched; swap in at next break"
                rationale += f" | REPEATED {concept} — {fam_macro} not Active"
        else:
            rationale += f" | REPEATED {concept} — stay sound base"

    pivot = active_pivot(db, oid, "defense")
    if pivot and macro:
        # Arming the REPEATED-tendency macro is the family switch — keep it
        rationale = f"{pivot} — keep {macro} (REPEATED answer is the switch) | {rationale}"
    elif pivot:
        dg = bl["defense_gameplan"]
        form, play, macro = dg["home_package"], rng.choice(list(dg["home_rotation"])), None
        user = user_job_for(play)
        rationale = f"{pivot} → {form}/{play} | {rationale}"
        suggest = suggest or "macros cleared — re-arm only on repeated tendency"

    fitted = _fit_defense(form, play, book, bl, rng)
    if fitted[:2] != (form, play):
        form, play = fitted[0], fitted[1]
        if not macro:
            user = user_job_for(play)
        rationale += f" | {fitted[2]}"

    macro_out = tag_live(macro, db) if macro else "none"
    if suggest:
        head, _, rest = suggest.partition(" — ")
        if get_macro(head):
            suggest = tag_live(head, db) + (f" — {rest}" if rest else "")
    return MaddenCall("defense", form, play, macro_out, user, rationale, suggest, macro=macro)


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
    return f"{tag} last {n} D snaps failed — reset Nickel Over base, clear chase macros"


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
    """One call. `playbook` = {side: {formation: [plays]}} locked by the latest prep;
    defaults to the DB's locked books (or the default stock books before any prep).
    Every returned formation/play pair is a member of that book."""
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
    active = list(active_macros or bl["macros_baseline"]["keep"])
    books = playbook or eligible(active_books(db))

    if last_coverage and not sit.coverage_hint:
        sit.coverage_hint, sit.coverage_source = last_coverage, "last"
    if last_concept and not sit.concept_hint:
        sit.concept_hint, sit.concept_source = last_concept, "last"

    cpu = is_cpu_opponent(opponent_id)
    note = ""
    if cpu:
        active = [m for m in active if macro_side(m) == "offense"]
        if sit.side == "defense":
            sit.side = "offense"
            note = "CPU = offense-only (no D calls) — switched to O | "

    if sit.side == "defense":
        call = _pick_defense(sit, opp, bl, db, rng, active, books["defense"])
    else:
        call = _pick_offense(sit, opp, bl, db, rng, active, books["offense"])
    stamp = _sit_stamp(sit)
    call.rationale = note + call.rationale + (f" | {stamp}" if stamp else "")
    return call
