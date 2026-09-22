"""Heuristic play-caller: seed + meta priors + dynasty log tendencies.

Tendency doctrine (symmetric O and D):
  - One tell = log + mild probability bump only.
  - Do NOT hard-counter the previous coverage/concept every snap.
  - Hard-counter / targeted macro only when the SAME signal is a REPEATED
    tendency in a similar D&D/field zone.
  - Default next snap = base situational call (D&D, field, opponent archetype).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from cfb_coach.db import CoachDB
from cfb_coach.seed import load_seed
from cfb_coach.situation import Situation
from cfb_coach.tendency import (
    describe_coverage_policy,
    is_repeated_concept,
    is_repeated_coverage,
)


@dataclass
class Call:
    side: str
    formation: str
    play: str
    adj_or_macro: str  # offense adj / defense macro
    read_or_user: str  # offense reads / defense user job
    rationale: str = ""
    suggest_macro: str | None = None  # optional SUGGEST for new macros

    def format(self) -> str:
        from cfb_coach.format_call import format_defense, format_offense

        if self.side == "offense":
            line = format_offense(
                self.formation, self.play, self.adj_or_macro, self.read_or_user
            )
        else:
            line = format_defense(
                self.formation, self.play, self.adj_or_macro, self.read_or_user
            )
        if self.suggest_macro:
            line += f"\n  SUGGEST macro: {self.suggest_macro}"
        return line


# --- Offense helpers ---------------------------------------------------------

_READS: dict[str, str] = {
    "Inside Zone": "Front → Cutback",
    "HB Base": "Front → Bounce",
    "Counter Y": "Pull → Cut",
    "QB Sweep": "Edge → Cut up",
    "Mtn RPO Zone Alert": "Bubble → Zone",
    "Mesh Spot": "Spot → Drag",
    "Mesh Traffic": "Traffic → Mesh",
    "Mesh Corner": "Corner → Mesh",
    "Drive HB Under": "Drive → HB Under",
    "Deep Flood": "Flat → Corner",
    "Return Whip Trail": "Whip → Trail",
    "Whip Double Spot": "Whip → Spot",
    "Mtn Cross Post": "Cross → Post",
    "Mtn HB Choice": "Choice → Check",
    "Mtn Speed Dig Under": "Dig → Under",
    "RZ PA X Whip": "Whip → Cross",
    "Z Spot GoalLine": "Spot → Flat",
    "Mesh": "Mesh → Sit",
    "Mesh Post": "Post → Mesh",
    "Verticals": "Seam → Outside",
    "Z Spot": "Spot → Flat",
    "Z Spot Shake": "Shake → Spot",
    "PA Read": "PA → Checkdown",
    "Outside Zone": "Reach → Cut",
    "HB Counter": "Pull → Cut",
    "HB Mid Draw": "Draw → Lane",
    "HB Slip Screen": "Screen → Blockers",
    "Spacing": "Spacing → Check",
    "Irish Mesh Whip": "Whip → Mesh",
    "518 Hook": "Hook → Flat",
    "DBL Post": "Post → Dig",
    "HB Slam": "Slam → Bounce",
    "HB Stretch": "Stretch → Cut",
    "HB Zone WK": "WK → Front",
    "Inside Zone Split": "Split → Front",
    "PA Boot": "Boot → Cross",
    "PA Boot Over": "Over → Flat",
    "PA Corner Dig": "Corner → Dig",
    "Quick Slants": "Slant → Hot",
    "RPO Alert TE Flat": "Flat → Zone",
    "Shock": "Shock → Check",
    "TE Cross": "Cross → Check",
    "Mtn Duo": "Duo → Cut",
    "HB Dive": "Dive → Bounce",
    "Motion Power": "Power → Cut",
    "PA RZ Crossers": "Crosser → Check",
    "PA Stretch Shot": "Shot → Check",
    "PA Mtn Spider Y Leak": "Leak → Check",
    "PA Z Cross": "Cross → Flat",
    "Bench": "Bench → Flat",
    "Spacing Switch": "Switch → Check",
    "Cross Z Post": "Post → Cross",
    "Curl Pivot Dig": "Curl → Dig",
    "All Go": "Go → Check",
    "Out Double Under": "Out → Under",
    "QB Blast": "Blast → Edge",
    "QB Draw": "Draw → Lane",
    "RPO Alert QB Draw": "Draw → Alert",
    "Vert Double Under": "Vert → Under",
    "WR Screen": "Screen → Blocks",
}


def _reads_for(play: str) -> str:
    return _READS.get(play, "Primary → Checkdown")


def _opp_id(opp: dict[str, Any]) -> str:
    return (opp.get("_id") or opp.get("id") or "").lower()


def _arch(opp: dict[str, Any]) -> str:
    return ((opp.get("defense_vs_us") or {}).get("archetype") or "").lower()


def _base_offense(
    sit: Situation, opp: dict[str, Any], rng: random.Random
) -> tuple[str, str, str, str]:
    """Situational base call from D&D / field / opponent archetype — ignores one-off coverage."""
    bunch, cluster = "Gun Bunch X Nasty", "Gun Cluster"
    deuce, pistol = "Singleback Deuce Close", "Pistol U Off Trips"
    oid = _opp_id(opp)
    arch = _arch(opp)
    adj = "No adj"

    if sit.goal_line or (sit.red_zone and sit.short_yardage):
        return deuce, rng.choice(["Mtn Duo", "HB Dive", "Inside Zone Split"]), adj, "GL/RZ short — bully run"

    if sit.red_zone:
        return (
            bunch,
            rng.choice(["RZ PA X Whip", "Z Spot GoalLine", "Inside Zone", "Mesh Spot"]),
            adj,
            "RZ — possession + whip/compressed / easy",
        )

    if sit.short_yardage:
        if oid == "quen":
            adj = "Protection first"
            return bunch, rng.choice(["HB Base", "Mesh Spot", "Return Whip Trail"]), adj, "3rd/short vs pressure arch — run/hot"
        return bunch, rng.choice(["HB Base", "Inside Zone", "Mesh Spot"]), adj, "3rd/short — run or quick man-beater"

    if sit.long_yardage and sit.down in (3, 4):
        if "two_high" in arch or "split_field" in arch or oid in ("gavin", "cpu"):
            return (
                bunch,
                rng.choice(["Mesh Spot", "Drive HB Under", "HB Mid Draw", "Spacing"]),
                adj,
                "3rd-long two-high arch — underneath to sticks (no force)",
            )
        return cluster, rng.choice(["Mesh Post", "Z Spot", "Verticals"]), adj, "3rd-long — sticks answer"

    if sit.two_minute:
        play = rng.choice(["Mesh Spot", "Deep Flood", "Quick Slants"])
        if play == "Quick Slants":
            return pistol, "Quick Slants", adj, "2-min — ball security + clock"
        return bunch, play, adj, "2-min — ball security + clock"

    # Early-down / normal — opponent archetype soft priors (seed dynasty), not last-snap coverage
    if oid == "gavin" or "split_field" in arch or "two_high" in arch:
        return (
            bunch,
            rng.choice(["Inside Zone", "HB Base", "Counter Y", "Mesh Spot", "Mtn RPO Zone Alert"]),
            adj,
            "base vs two-high/split-field arch — run / easy (not last-coverage chase)",
        )
    if oid == "quen" or "pressure" in arch:
        return (
            bunch,
            rng.choice(["Mesh Spot", "HB Base", "Inside Zone", "Return Whip Trail"]),
            "Protection first" if rng.random() < 0.4 else adj,
            "base vs pressure arch — protection + easy answers",
        )
    if oid == "cpu":
        return (
            bunch,
            rng.choice(["Inside Zone", "Mesh Spot", "Drive HB Under", "HB Base"]),
            adj,
            "base vs CPU — free underneath / run",
        )

    # Generic early-down mix — preserve Bunch + Cluster
    roll = rng.random()
    if roll < 0.45:
        return bunch, rng.choice(["Inside Zone", "HB Base", "Mesh Spot"]), adj, "early down — establish run / easy completion"
    if roll < 0.7:
        return bunch, rng.choice(["Deep Flood", "Mesh Traffic", "Mtn Cross Post"]), adj, "early pass mix"
    if roll < 0.85:
        return cluster, rng.choice(["Outside Zone", "Z Spot Shake", "Mesh Post"]), adj, "Cluster counterpunch"
    return pistol, rng.choice(["HB Stretch", "RPO Alert TE Flat", "Inside Zone Split"]), adj, "Pistol run/RPO changeup"


def _soft_coverage_lean(
    form: str,
    play: str,
    adj: str,
    rationale: str,
    sit: Situation,
    opp: dict[str, Any],
    db: CoachDB | None,
    rng: random.Random,
) -> tuple[str, str, str, str, str | None]:
    """
    Coverage handling:
      - last-snap only → no call change (policy note)
      - live single look → mild soft lean toward sound base answers (never Deep Flood force)
      - repeated in-situation → allow coverage-specific beater
    """
    cov = sit.coverage_hint
    if not cov:
        return form, play, adj, rationale, None

    oid = _opp_id(opp)
    src = getattr(sit, "coverage_source", "none") or "none"
    last_only = src == "last"
    live = src == "live"
    repeated = is_repeated_coverage(db, oid, cov, sit, threshold=2)
    policy = describe_coverage_policy(
        cov, live=live and not last_only, last_snap=last_only, repeated=repeated
    )

    # LAST SNAP: never hard-counter this snap — keep base situational call
    if last_only:
        return form, play, adj, f"{rationale} | {policy}", None

    cl = cov.lower()
    invert = "invert" in cl
    c2 = "cover 2" in cl or invert
    bunch, cluster = "Gun Bunch X Nasty", "Gun Cluster"

    # REPEATED live tendency + this snap shows it → coverage-specific answers OK
    if repeated and live:
        if invert or c2:
            if sit.red_zone:
                form, play = rng.choice(
                    [(bunch, "Mesh Spot"), (bunch, "Inside Zone"), (cluster, "Z Spot Shake")]
                )
                return form, play, adj, f"REPEATED {cov} RZ — Mesh Spot/IZ/Shake | {policy}", None
            form, play = bunch, rng.choice(["Mesh Spot", "Inside Zone", "HB Base", "Mtn Speed Dig Under"])
            return form, play, adj, f"REPEATED {cov} — Mesh Spot/IZ family | {policy}", None
        if any(x in cl for x in ("cover 4", "cover 6", "cover 9", "quarters", "two-high", "palms")):
            form, play = bunch, rng.choice(["Inside Zone", "HB Base", "Counter Y", "Mesh Spot"])
            return form, play, adj, f"REPEATED two-high {cov} — run/easy | {policy}", None
        if "cover 3" in cl:
            form, play = bunch, rng.choice(["Deep Flood", "Mtn Cross Post", "Drive HB Under"])
            return form, play, adj, f"REPEATED C3 — flood/crossers; confirm leverage | {policy}", None
        if "cover 1" in cl or "pressure" in cl or "cover 0" in cl:
            form, play = bunch, rng.choice(["Mesh Spot", "Return Whip Trail", "Mesh Traffic"])
            return form, play, "Hot ready", f"REPEATED pressure/man — quick mesh/whip | {policy}", None
        return form, play, adj, f"{rationale} | {policy}", None

    # LIVE single look — soft lean only (compatible base answers; never force Flood into Invert)
    if live:
        if invert and not sit.long_yardage:
            # Aidan UX: C2 Invert early → Mesh Spot / underneath OR IZ — not Deep Flood.
            # Soft lean only (one look ≠ hard-counter beater).
            form, play = bunch, rng.choice(
                ["Mesh Spot", "Mesh Spot", "Inside Zone", "HB Base"]
            )
            return (
                form,
                play,
                adj,
                f"soft lean vs live {cov} — Mesh Spot/IZ (not Flood) | {policy}",
                None,
            )
        if c2 and play == "Deep Flood":
            form, play = bunch, rng.choice(["Mesh Spot", "Inside Zone"])
            return form, play, adj, f"soft lean vs live C2 — no forced Flood | {policy}", None
        if "pressure" in cl or "cover 0" in cl:
            if play in ("Deep Flood", "Verticals", "Mtn Cross Post"):
                form, play = bunch, "Mesh Spot"
                return form, play, "Hot ready", f"soft lean vs live pressure — quick | {policy}", None
        return form, play, adj, f"{rationale} | {policy}", None

    return form, play, adj, f"{rationale} | {policy}", None


def _validate_play(form: str, play: str, pb: dict) -> tuple[str, str]:
    core = pb.get(form, {}).get("core", [])
    if play in core or not core:
        return form, play
    for f, meta in pb.items():
        if play in meta.get("core", []):
            return f, play
    return form, core[0]


def _pick_offense(
    sit: Situation,
    opp: dict[str, Any],
    seed: dict,
    db: CoachDB | None,
    rng: random.Random,
) -> Call:
    pb = seed["playbooks"]["offense_formations"]
    form, play, adj, rationale = _base_offense(sit, opp, rng)
    form, play, adj, rationale, _ = _soft_coverage_lean(
        form, play, adj, rationale, sit, opp, db, rng
    )
    # Rare empty changeup — still situational, not coverage-chase
    if sit.down == 3 and sit.long_yardage and rng.random() < 0.12:
        form, play = "Gun Empty Quads", rng.choice(
            ["Out Double Under", "Curl Pivot Dig", "QB Draw"]
        )
        rationale = "Empty changeup / QB stress | " + rationale
    form, play = _validate_play(form, play, pb)

    # Anti-repeat: if same play flooded recent snaps, rotate to constraint changeup
    if db is not None:
        try:
            from cfb_coach.gameplan import anti_repeat_penalty

            pen = anti_repeat_penalty(db, _opp_id(opp), form, play, side="offense")
            if pen >= 1.0:
                alt_form, alt_play = "Gun Cluster", rng.choice(
                    ["Z Spot Shake", "Outside Zone", "Mesh Post"]
                )
                alt_form, alt_play = _validate_play(alt_form, alt_play, pb)
                rationale = (
                    f"anti-repeat pivot → {alt_form}/{alt_play} "
                    f"(pen={pen:.1f}) | {rationale}"
                )
                form, play = alt_form, alt_play
        except Exception:
            pass

    # Mid-game PIVOT: last 3 O snaps failed → force constraint family switch
    if db is not None:
        try:
            from cfb_coach.gameplan import active_pivot

            tip = active_pivot(db, _opp_id(opp), "offense")
            if tip:
                alt_form, alt_play = "Gun Cluster", rng.choice(
                    ["Z Spot Shake", "Outside Zone", "Mesh Spot"]
                )
                # Prefer easy Mesh Spot from Bunch if Cluster just failed too
                if "Cluster" in (form or "") or rng.random() < 0.45:
                    alt_form, alt_play = "Gun Bunch X Nasty", rng.choice(
                        ["Inside Zone", "HB Base", "Mesh Spot"]
                    )
                alt_form, alt_play = _validate_play(alt_form, alt_play, pb)
                form, play = alt_form, alt_play
                adj = "No adj"
                rationale = f"{tip.message} → {form}/{play} | {rationale}"
        except Exception:
            pass

    return Call("offense", form, play, adj, _reads_for(play), rationale)


# --- Defense helpers ---------------------------------------------------------

_ACTIVE_MACROS = ["CROSS", "VERT", "BUNCH", "RPO", "SCRAM", "RUN-IN", "RUN-OUT", "HEAT"]
_BENCHED_MACROS = ["FLOOD", "SCREEN"]

_USER_FOR_PLAY = {
    "Cover 3 Sky": "User hook",
    "Cover 4 Quarters": "User seam",
    "Tampa 2": "User middle",
    "Cover 3 Buzz": "User cutback",
    "Field Sim 3": "User hook",
}


def _base_defense(sit: Situation, opp: dict[str, Any], rng: random.Random) -> tuple[str, str, str, str]:
    """Nickel Over home unless D&D/field clearly demands otherwise. No concept chase."""
    oid = _opp_id(opp)
    form, play, macro = "Nickel Over", "Cover 3 Sky", "none"
    user = "User hook"
    rationale = "base Nickel Over — situational (not last-concept chase)"

    # True short-yardage package — still no auto RUN-IN
    if sit.short_yardage and sit.down in (3, 4) and sit.distance is not None and sit.distance <= 2:
        form, play = "4-3 Even 6-1", "Cover 3 Buzz"
        user = "User cutback"
        rationale = "short yardage — Even 6-1 base (no auto RUN-IN on one IZ tell)"
        return form, play, macro, rationale

    if sit.goal_line:
        form, play = "4-3 Even 6-1", "Cover 3 Buzz"
        user = "User cutback"
        rationale = "GL — heavy front; don't sell out run (Temple Z Smash lesson)"
        return form, play, macro, rationale

    if sit.long_yardage or (sit.down == 3 and sit.distance and sit.distance >= 8):
        play = "Cover 4 Quarters"
        user = "User seam"
        rationale = "obvious pass D&D — Nickel Over Quarters"
        if sit.distance and sit.distance >= 12 and rng.random() < 0.3:
            form, play = "Dime Normal", "Cover 4 Quarters"
            rationale = "true long passing down — Dime Quarters"
        return form, play, macro, rationale

    # Default Nickel Over rotation (C3 heavy)
    play = rng.choice(["Cover 3 Sky", "Cover 3 Sky", "Cover 3 Sky", "Cover 4 Quarters", "Tampa 2"])
    user = _USER_FOR_PLAY.get(play, "User hook")
    rationale = "Nickel Over home (C3/C4/Tampa)"

    # Opponent archetype soft prior (seed), not a one-snap hard-counter
    if oid == "cpu":
        if sit.down in (3, 4) or sit.long_yardage:
            play, user = "Cover 4 Quarters", "User seam"
            rationale = "CPU money-down prior — Quarters"
    elif oid == "gavin" and sit.down in (1, 2, None) and not sit.long_yardage:
        play, user = "Cover 3 Sky", "User HB"
        rationale = "vs Gavin early-down prior — fit IZ from C3 Sky"
    elif oid == "quen" and sit.down == 3 and sit.short_yardage and rng.random() < 0.2:
        # Occasional pressure changeup — rare, purposeful, not every tell
        form = rng.choice(["Nickel 3-3 Cub", "Nickel Double Mug"])
        play = "Mike Blitz 0" if "Cub" in form else "Mid Blitz 0"
        macro = "HEAT"
        user = "User hot"
        rationale = "Quen 3rd-short — selective HEAT (archetype, not one-tell)"

    return form, play, macro, rationale


def _concept_family(concept: str) -> str | None:
    c = concept.lower()
    if "vert" in c or "four" in c or "seam" in c:
        return "vert"
    if "cross" in c or "wheel" in c:
        return "cross"
    if "mesh" in c or "bunch" in c or "cluster" in c:
        return "mesh"
    if "rpo" in c or "bubble" in c:
        return "rpo"
    if "scram" in c:
        return "scram"
    if "inside zone" in c or "duo" in c or c in ("iz", "zone"):
        return "run_in"
    if "stretch" in c or "outside zone" in c or "toss" in c:
        return "run_out"
    return None


def _soft_concept_lean(
    form: str,
    play: str,
    macro: str,
    user: str,
    rationale: str,
    sit: Situation,
    opp: dict[str, Any],
    db: CoachDB | None,
    rng: random.Random,
) -> tuple[str, str, str, str, str, str | None]:
    """
    Concept handling (symmetric with coverage doctrine):
      - one tell / last snap → stay base Nickel Over; SUGGEST only
      - repeated in-situation → targeted macro OK
    """
    suggest = None
    oid = _opp_id(opp)
    concept = sit.concept_hint
    if not concept:
        # Soft seed awareness without macros: nudge shell only when archetype-known
        # and still no macro on thin evidence
        return form, play, macro, user, rationale, None

    family = _concept_family(concept)
    # Treat bare concept in sit as a "tell" — need LIVE repeats for macro.
    raw = (sit.raw or "").lower()
    last_only = bool(
        __import__("re").search(r"\b(last|prev|previous|saw|showed|was)\b", raw)
    )
    repeated = is_repeated_concept(db, oid, concept, threshold=2)
    if not repeated and family and db:
        # Sum live tells for same family only (seed buckets excluded by helper)
        for row in db.get_tendencies(oid, "offense_concept"):
            if _concept_family(row["key"]) == family and int(row["count"]) >= 2:
                repeated = True
                break

    # Previous-snap concept: never auto-macro this snap
    if last_only:
        return (
            form,
            play,
            "none",
            user,
            f"{rationale} | last snap {concept} — mild bump only; stay base (no hard-counter)",
            f"{_macro_for_family(family)} — only after repeated {concept} in similar D&D",
        )

    if not repeated:
        # Single live tell: maybe soft shell nudge, NEVER auto macro / Even sellout
        if family == "vert" and form == "Nickel Over":
            play, user = "Cover 4 Quarters", "User seam"
            rationale = (
                f"{rationale} | one {concept} tell — Quarters soft lean; "
                "no VERT until repeated"
            )
            suggest = "VERT — after one more vertical tell in similar D&D"
        elif family == "cross" and form == "Nickel Over":
            play = rng.choice(["Tampa 2", "Cover 4 Quarters"])
            user = _USER_FOR_PLAY.get(play, "User middle")
            rationale = (
                f"{rationale} | one {concept} tell — Tampa/Quarters soft lean; "
                "no CROSS until repeated"
            )
            suggest = "CROSS — wait for repeated crosser/wheel in similar D&D"
        elif family == "mesh" and form == "Nickel Over":
            play, user = "Tampa 2", "User middle"
            rationale = f"{rationale} | one mesh tell — Tampa changeup; no BUNCH yet"
            suggest = "BUNCH — if compressed keeps winning"
        elif family == "rpo":
            rationale = f"{rationale} | one RPO/bubble tell — base zone + flats; no RPO macro yet"
            suggest = "RPO — if bubble keeps winning"
        elif family == "run_in":
            # Explicitly do NOT jump to Even 6-1 / RUN-IN on one IZ
            if form != "4-3 Even 6-1":
                play, user = "Cover 3 Sky", "User HB"
            rationale = (
                f"{rationale} | one IZ/Duo tell — fit from C3; "
                "no RUN-IN / Even sellout until repeated"
            )
            suggest = "RUN-IN — only after repeated downhill run in similar D&D"
        elif family == "scram":
            user = "User contain"
            rationale = f"{rationale} | one scramble tell — contain job; no SCRAM macro yet"
            suggest = "SCRAM — if escape keeps beating coverage"
        else:
            rationale = f"{rationale} | one {concept} tell — logged/soft only; stay base"
        return form, play, "none", user, rationale, suggest

    # REPEATED — targeted macro OK while staying mostly Nickel Over
    if family == "vert":
        form, play, macro, user = "Nickel Over", "Cover 4 Quarters", "VERT", "User #3 seam"
        rationale = f"REPEATED vertical tendency — Quarters + VERT"
    elif family == "cross":
        form = "Nickel Over"
        play = rng.choice(["Tampa 2", "Cover 4 Quarters"])
        macro, user = "CROSS", "User inside cross"
        rationale = f"REPEATED crosser/wheel — {play} + CROSS"
    elif family == "mesh":
        form, play, macro, user = "Nickel Over", "Tampa 2", "BUNCH", "User middle"
        rationale = "REPEATED mesh/compressed — Tampa + BUNCH"
    elif family == "rpo":
        form, play, macro, user = "Nickel Over", "Cover 3 Sky", "RPO", "User flat"
        rationale = "REPEATED RPO/bubble — RPO macro"
    elif family == "run_in":
        if sit.short_yardage:
            form, play, macro, user = "4-3 Even 6-1", "Cover 3 Buzz", "RUN-IN", "User cutback"
            rationale = "REPEATED downhill run + short — Even + RUN-IN"
        else:
            form, play, macro, user = "Nickel Over", "Cover 3 Sky", "none", "User HB"
            rationale = "REPEATED IZ early — fit C3; RUN-IN only if still short later"
            suggest = "RUN-IN — available if they keep hammering downhill"
    elif family == "run_out":
        form, play, macro, user = "Nickel Over", "Cover 3 Sky", "RUN-OUT", "User edge"
        rationale = "REPEATED perimeter run — RUN-OUT"
    elif family == "scram":
        form, play = "Nickel Over", play if form == "Nickel Over" else "Cover 3 Sky"
        macro, user = "SCRAM", "User contain"
        rationale = "REPEATED scramble — SCRAM + one contain job"
    else:
        rationale = f"REPEATED {concept} — stay sound base"
    return form, play, macro, user, rationale, suggest


def _macro_for_family(family: str | None) -> str:
    return {
        "vert": "VERT",
        "cross": "CROSS",
        "mesh": "BUNCH",
        "rpo": "RPO",
        "run_in": "RUN-IN",
        "run_out": "RUN-OUT",
        "scram": "SCRAM",
    }.get(family or "", "targeted macro")


def _pick_defense(
    sit: Situation,
    opp: dict[str, Any],
    seed: dict,
    db: CoachDB | None,
    rng: random.Random,
) -> Call:
    form, play, macro, rationale = _base_defense(sit, opp, rng)
    user = _USER_FOR_PLAY.get(play, "User hook")
    if "cutback" in rationale:
        user = "User cutback"
    if "HEAT" in rationale or macro == "HEAT":
        user = "User hot"

    form, play, macro, user, rationale, suggest = _soft_concept_lean(
        form, play, macro, user, rationale, sit, opp, db, rng
    )

    # Low-confidence opponents: never arm macros from thin film
    conf = (opp.get("confidence") or "low").lower()
    if conf in ("low", "none") and macro not in ("none", ""):
        suggest = suggest or f"{macro} — low film; keep base zone"
        macro = "none"
        rationale += " | low-confidence opponent — strip macro"

    # Mid-game PIVOT: last 3 D snaps failed → reset to Nickel Over base, clear chase macros
    if db is not None:
        try:
            from cfb_coach.gameplan import active_pivot

            tip = active_pivot(db, _opp_id(opp), "defense")
            if tip:
                form, play, macro = "Nickel Over", rng.choice(
                    ["Cover 3 Sky", "Cover 4 Quarters", "Tampa 2"]
                ), "none"
                user = _USER_FOR_PLAY.get(play, "User hook")
                rationale = f"{tip.message} → {form}/{play} | {rationale}"
                suggest = suggest or "macros cleared — re-arm only on repeated tendency"
        except Exception:
            pass

    return Call("defense", form, play, macro or "none", user, rationale, suggest_macro=suggest)


def make_call(
    sit: Situation,
    opponent_id: str,
    db: CoachDB | None = None,
    *,
    seed: dict | None = None,
    rng: random.Random | None = None,
    last_coverage: str | None = None,
    last_concept: str | None = None,
) -> Call:
    """
    Build one call. Optional last_coverage / last_concept are PREVIOUS-snap
    observations: they must NOT hard-counter this snap (mild context only).
    """
    seed = seed or load_seed()
    opp = (seed.get("opponents") or {}).get(opponent_id, {})
    if db:
        profile = db.get_opponent(opponent_id)
        if profile:
            opp = profile
    opp = dict(opp)
    opp["_id"] = opponent_id
    rng = rng or random.Random()

    # Inject last-snap signals as soft context without overwriting a live look
    if last_coverage and not sit.coverage_hint:
        sit.coverage_hint = last_coverage
        sit.coverage_source = "last"
    elif last_coverage and sit.coverage_source == "none":
        sit.coverage_source = "last"

    if last_concept and not sit.concept_hint:
        # Encode as last via raw tag so defense lean treats it as last-only
        sit.concept_hint = last_concept
        if "last" not in (sit.raw or "").lower():
            sit.raw = f"{sit.raw} last {last_concept}".strip()

    if sit.side == "defense":
        return _pick_defense(sit, opp, seed, db, rng)
    return _pick_offense(sit, opp, seed, db, rng)


def one_shot(
    opponent_id: str,
    situation_raw: str,
    *,
    side: str = "offense",
    db: CoachDB | None = None,
) -> Call:
    from cfb_coach.situation import parse_situation

    sit = parse_situation(situation_raw, default_side=side)
    return make_call(sit, opponent_id, db)
