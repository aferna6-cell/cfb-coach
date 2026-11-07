"""Madden 27 defensive knowledge LEXICON (data only; no logic).

Everything the catalog says about a play comes from one of these records, so every claim has a source id and an
evidence status. Nothing here asserts a Madden 27 eleven-defender assignment: no first-party (Xbox) play-art evidence
exists in this repository, so the per-defender ledger stays UNKNOWN and the weakest claims are labelled as such.

Evidence statuses (strongest first)
    xbox_verified          first-party play-art screenshot/hash (NO entry has this yet; the schema requires a hash)
    ea_documented          EA's official Madden 27 text states the general mechanic (not a per-play assignment)
    name_derived           the fact is only that the label CONTAINS a token / digit; says nothing about its meaning
    football_hypothesis    a standard-football meaning from a licensed or cited reference, or the author's convention;
                           whether Madden 27 implements this exact call that way is NOT verified
    third_party_unverified an educational/third-party Madden source states it; version or primary evidence unverified
    unknown                nothing usable; the catalog records exactly what is missing

Only ``feature_eligible`` claims may reach model features: name tokens, EA-documented mechanics, and CC BY-SA football
definitions. Text from sites whose terms restrict extraction/ML reuse (Madden Tools, 818 Madden, Huddle) is never ingested.
"""
from __future__ import annotations

ACCESSED = "2026-11-07"

# ------------------------------------------------------------------------------------------------ sources

SOURCES: dict[str, dict] = {
    "ea_deep_dive": {
        "kind": "official_documentation", "title": "EA SPORTS Madden NFL 27 - Gameplay Deep Dive",
        "url": "https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay", "published": "2026-06-04",
        "fetched": True, "quality": "ea_documented_general_mechanic", "license": "EA copyright; short excerpts for citation only",
        "trainable_text": False,
        "note": "States match checks, Smart Zones, plaster, the Zone Pressure style and the Double Team tool. It does NOT define "
                "Sky/Cloud/Buzz, Tampa 2, Robber/Hole, Cover 9, fire zone, creeper or spy. Excerpts were relayed through a "
                "summarizing fetch tool; re-verify wording against the page before quoting publicly."},
    "wiki_glossary": {
        "kind": "encyclopedia", "title": "Glossary of American football (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Glossary_of_American_football", "published": None, "fetched": True,
        "quality": "football_reference", "license": "CC BY-SA 4.0", "trainable_text": True,
        "note": "Only the first ~100k characters were read: Cover 0/1/2/3/4, Tampa 2 alias, nickel/dime back, blitz. Spy, robber, "
                "bracket, prevent, trap, pattern-match were NOT found in the portion read."},
    "wiki_zone_blitz": {
        "kind": "encyclopedia", "title": "Zone blitz (Wikipedia)", "url": "https://en.wikipedia.org/wiki/Zone_blitz",
        "published": None, "fetched": True, "quality": "football_reference", "license": "CC BY-SA 4.0", "trainable_text": True,
        "note": "Defines fire zone as rushing 5 with 3-3 coverage behind. Does not mention simulated pressure, creeper or dog."},
    "wiki_tampa_2": {
        "kind": "encyclopedia", "title": "Tampa 2 (Wikipedia)", "url": "https://en.wikipedia.org/wiki/Tampa_2", "published": None,
        "fetched": True, "quality": "football_reference", "license": "CC BY-SA 4.0", "trainable_text": True,
        "note": "Cover 2 except the middle linebacker drops into the deep middle on a read pass; tight ends are the typical target."},
    "wiki_prevent": {
        "kind": "encyclopedia", "title": "Prevent defense (Wikipedia)", "url": "https://en.wikipedia.org/wiki/Prevent_defense",
        "published": None, "fetched": True, "quality": "football_reference", "license": "CC BY-SA 4.0", "trainable_text": True,
        "note": "Five or more defensive backs, deep coverage; concedes short gains."},
    "wiki_46": {
        "kind": "encyclopedia", "title": "46 defense (Wikipedia)", "url": "https://en.wikipedia.org/wiki/46_defense", "published": None,
        "fetched": True, "quality": "football_reference", "license": "CC BY-SA 4.0", "trainable_text": True,
        "note": "Six on the line, two at linebacker depth, three defensive backs; risk is quick timed passes."},
    "wiki_nickel": {
        "kind": "encyclopedia", "title": "Nickel defense (Wikipedia)", "url": "https://en.wikipedia.org/wiki/Nickel_defense",
        "published": None, "fetched": True, "quality": "football_reference", "license": "CC BY-SA 4.0", "trainable_text": True,
        "note": "Nickel = five defensive backs. The page does NOT define dime."},
    "mutgg_cover1": {
        "kind": "educational_article", "title": "Ask Huddle #42 - Cover 1 Explained in Madden (MUT.GG)",
        "url": "https://www.mut.gg/news/ask-huddle-42-cover-1-explained-in-madden/", "published": "2026-01-09", "fetched": True,
        "quality": "third_party_education_version_unverified", "license": "third-party site; paraphrase/cite only",
        "trainable_text": False,
        "note": "Published 2026-01-09, BEFORE Madden 27's release, and the article text does not name a game version. Defines Hole "
                "(bonus defender drops to a hook), Robber (hook/hole defender comes from safety) and Contain (both edge rushers "
                "contain, bonus defender is a QB spy). Earlier notes that called this a Madden 27 source were not supportable."},
    "mutgg_cover3_naming": {
        "kind": "educational_article", "title": "MUT.GG Cover 3 naming rule (Buzz=backer, Sky=safety, Cloud=corner on the backside flat; Mable=match)",
        "url": "https://www.mut.gg/news/", "published": None, "fetched": False,
        "quality": "unverified_search_summary_only", "license": "n/a", "trainable_text": False,
        "note": "Seen ONLY in a web-search result summary. The article it was attributed to (Ask Huddle #7) was fetched and does "
                "not contain the rule; two independent pages that might (Big Cat Country 2013, Operation Sports forum) returned "
                "HTTP 403. Another summary said Madden School/Big Blue View describe Mable differently (zone+man mix). Treat as an "
                "unverified football hypothesis with conflicting descriptions."},
    "civil_zone_drop_hook": {
        "kind": "educational_article", "title": "Civil.GG Zone Drop Hook Defense Guide (CFB 27 & Madden 27)",
        "url": "https://www.civil.gg/tips/zone-drop-hook-defense-guide", "published": None, "fetched": True,
        "quality": "third_party_education_version_unverified", "license": "third-party site; cite only", "trainable_text": False,
        "note": "States Hard Flats play 0-5 yards from the line of scrimmage. Accessed 2026-09-29 (earlier research run)."},
    "prodigy_multiple_d": {
        "kind": "name_catalog", "title": "Madden Prodigy - Madden 27 Multiple D defense playbook",
        "url": "https://www.maddenprodigy.com/playbooks/madden-27-multiple-d-defense/", "published": None, "fetched": True,
        "quality": "name_only", "license": "no terms statement visible; names used for identification only", "trainable_text": False,
        "note": "Plays are listed by name and thumbnail only; no defender responsibilities. Only the first 100k of 228k characters "
                "were read, so a play absent from the visible portion is NOT proven absent from Madden 27."},
    "repo_snapshot": {
        "kind": "repository_record", "title": "installed_playbook_snapshot.json (DEF-F1CF-1862)",
        "url": "cfb_coach/data/madden27/installed_playbook_snapshot.json", "published": "2026-11-05", "fetched": True,
        "quality": "recorded_application_state", "license": "owner data", "trainable_text": True,
        "note": "Recorded application state of the owner's custom defensive book; not a fresh per-play Xbox verification."},
    "repo_audit_115": {
        "kind": "repository_record", "title": "DEFENSIVE_COVERAGE_GAP_AUDIT_20261107.md (PR #115)",
        "url": "docs/madden_ml/DEFENSIVE_COVERAGE_GAP_AUDIT_20261107.md", "published": "2026-11-07", "fetched": True,
        "quality": "name_based_classification", "license": "owner data", "trainable_text": True,
        "note": "Groups unrecognized names into coverage/pressure/bracket categories by NAME. Not verified behaviour."},
    "repo_third_party_annotations": {
        "kind": "repository_record", "title": "installed_play_annotations.json (public Madden Tools pages)",
        "url": "cfb_coach/data/madden27/installed_play_annotations.json", "published": "2026-11-05", "fetched": True,
        "quality": "third_party_unverified", "license": "Madden Tools terms restrict extraction / ML reuse", "trainable_text": False,
        "note": "Three defensive plays carry published player jobs. Shown for research only; never a model input."},
    "domain_convention": {
        "kind": "author_domain_knowledge", "title": "Standard football convention asserted by the catalog author",
        "url": None, "published": None, "fetched": False, "quality": "football_hypothesis", "license": "n/a",
        "trainable_text": True,
        "note": "Generic football usage with NO fetched citation (e.g. Cover 6 = quarters to one side, half to the other). Needs coach "
                "or owner review. It may never be reported as a verified Madden 27 mechanic."},
}

STATUS_ORDER = ("xbox_verified", "ea_documented", "football_hypothesis", "name_derived", "third_party_unverified", "unknown")
FEATURE_ELIGIBLE_STATUS = ("ea_documented", "name_derived", "football_hypothesis")
CONFIDENCE = ("high", "medium", "low")

# ------------------------------------------------------------------------------------------------ formations

# Personnel arithmetic (DL-LB-DB summing to 11) is a naming convention + Wikipedia definitions, i.e. football_hypothesis.
FORMATIONS: dict[str, dict] = {
    "4-3 Over": {"book": "49ers", "dl": 4, "lb": 3, "db": 4, "basis": "domain_convention",
                 "note": "'4-3' = four linemen, three linebackers; 'Over' shifts the front toward the strong side (exact shift unverified)."},
    "46 Normal": {"book": "Chargers", "dl": 6, "lb": 2, "db": 3, "basis": "wiki_46",
                  "note": "Six on the line, two linebackers, three defensive backs per the 46 definition; what 'Normal' changes is unverified."},
    "Dime 3-2 Odd": {"book": "49ers", "dl": 3, "lb": 2, "db": 6, "basis": "domain_convention",
                     "note": "Six DBs inferred from 3+2+6=11; Wikipedia's nickel page does not define dime. 'Odd' = odd front (unverified)."},
    "Nickel Double Mug": {"book": "49ers", "dl": None, "lb": None, "db": 5, "basis": "wiki_nickel",
                          "note": "Nickel = five DBs. 'Double Mug' name implies two mugged (walked-up) defenders; who and where is unverified."},
    "Nickel Over": {"book": "49ers", "dl": 4, "lb": 2, "db": 5, "basis": "wiki_nickel",
                    "note": "Nickel = five DBs; the usual 4-2-5 split is a convention. 'Over' shift unverified."},
    "Nickel Single Mug": {"book": "49ers", "dl": None, "lb": None, "db": 5, "basis": "wiki_nickel",
                          "note": "Nickel = five DBs. 'Single Mug' name implies one mugged defender; who and where is unverified."},
}

# ------------------------------------------------------------------------------------------------ bases and traits
# base: id -> structural axes. 'shell' = deep-defender structure; 'type' = how underneath defenders are tied to routes.

BASES: dict[str, dict] = {
    "c0": {"label": "Cover 0", "shell": 0, "type": "man", "family": "cover0", "src": "wiki_glossary", "status": "football_hypothesis",
           "text": "Glossary: strict man-to-man with no help from safeties."},
    "c1": {"label": "Cover 1", "shell": 1, "type": "man", "family": "cover1", "src": "wiki_glossary", "status": "football_hypothesis",
           "text": "Glossary: man coverage with at least one safety not assigned a player."},
    "c2": {"label": "Cover 2", "shell": 2, "type": "spot_zone", "family": "cover2", "src": "wiki_glossary", "status": "football_hypothesis",
           "text": "Glossary: zone with the safeties deep, each covering half the field."},
    "t2": {"label": "Tampa 2", "shell": 2, "type": "spot_zone", "family": "cover2", "src": "wiki_tampa_2", "status": "football_hypothesis",
           "text": "Cover 2 except the middle linebacker drops into the deep middle on a read pass."},
    "c3": {"label": "Cover 3", "shell": 3, "type": "spot_zone", "family": "cover3", "src": "wiki_glossary", "status": "football_hypothesis",
           "text": "Glossary: zone with extra help from the strong safety or a cornerback (three deep)."},
    "c4": {"label": "Cover 4", "shell": 4, "type": "unknown", "family": "cover4", "src": "wiki_glossary", "status": "football_hypothesis",
           "text": "Glossary: corners and safeties drop into deep coverage (four deep). Match vs spot depends on the variant."},
    "c6": {"label": "Cover 6", "shell": "split", "type": "match", "family": "cover6", "src": "ea_deep_dive", "status": "ea_documented",
           "text": "EA lists Cover 6 among the families that use match coverage checks. Which side plays quarters vs half is a convention "
                   "(domain_convention), not EA text."},
    "c9": {"label": "Cover 9", "shell": "split", "type": "unknown", "family": "cover9", "src": "domain_convention", "status": "football_hypothesis",
           "text": "Treated as a split-field coverage (as in the earlier research stack). EA does not define it; side/zones unknown."},
    "m3d": {"label": "Man 3 Deep", "shell": 3, "type": "man", "family": "man3deep", "src": "domain_convention", "status": "football_hypothesis",
            "text": "Name says man with three deep. Exact assignments unknown."},
    "prev": {"label": "Prevent", "shell": "unknown", "type": "spot_zone", "family": "prevent", "src": "wiki_prevent",
             "status": "football_hypothesis", "text": "Five or more DBs playing deep to prevent a long pass; concedes short gains."},
}

# variant tokens (coverage axis). 'meaning' is deliberately short; 'status' is the status OF THE MEANING, the token's presence
# in the label is always name_derived.
VARIANTS: dict[str, dict] = {
    "hole": {"meaning": "bonus defender drops into a hook/hole zone (Cover 1)", "status": "third_party_unverified", "src": "mutgg_cover1", "conf": "low"},
    "robber": {"meaning": "hook/hole defender comes from the safety position (Cover 1)", "status": "third_party_unverified", "src": "mutgg_cover1", "conf": "low"},
    "contain": {"meaning": "both edge rushers contain; the bonus defender is a QB spy (Cover 1)", "status": "third_party_unverified", "src": "mutgg_cover1", "conf": "low"},
    "spy": {"meaning": "a defender is assigned to mirror the quarterback", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "press": {"meaning": "corners align at the line; EA documents press as a corner-depth option", "status": "ea_documented", "src": "ea_deep_dive", "conf": "medium"},
    "sting": {"meaning": "unknown", "status": "unknown", "src": None, "conf": "low"},
    "thief": {"meaning": "unknown", "status": "unknown", "src": None, "conf": "low"},
    "double": {"meaning": "double coverage on the receiver named in the label", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "bracket": {"meaning": "two defenders bracket one receiver; EA documents bracketing only for the user Double Team tool and Triangle checks", "status": "football_hypothesis", "src": "ea_deep_dive", "conf": "low"},
    "switch": {"meaning": "defenders may switch assignments; exact trigger unknown", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "man": {"meaning": "underneath defenders are in man coverage", "status": "football_hypothesis", "src": "domain_convention", "conf": "medium"},
    "invert": {"meaning": "at least one defender's usual deep/flat role is swapped; WHICH defenders is unverified", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "hard_flat": {"meaning": "a defender squats the flat zone (0-5 yards per Civil.GG)", "status": "third_party_unverified", "src": "civil_zone_drop_hook", "conf": "low"},
    "roll": {"meaning": "coverage rotates (pre- or post-snap); EA's 'Roll Coverage' setting is a different feature", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "trap": {"meaning": "a flat defender sits to bait an outside throw (Cover 2 trap); details unverified", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "drop": {"meaning": "spot-drop zone rather than a match; exact drops unverified", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "cloud": {"meaning": "a corner takes the flat (Cover 3 naming convention)", "status": "football_hypothesis", "src": "mutgg_cover3_naming", "conf": "low"},
    "sky": {"meaning": "a safety rotates down to the flat (Cover 3 naming convention)", "status": "football_hypothesis", "src": "mutgg_cover3_naming", "conf": "low"},
    "buzz": {"meaning": "a linebacker takes the flat/curl (Cover 3 naming convention)", "status": "football_hypothesis", "src": "mutgg_cover3_naming", "conf": "low"},
    "match": {"meaning": "pattern-matching zone: defenders pass routes off by receiver releases; EA documents match checks", "status": "ea_documented", "src": "ea_deep_dive", "conf": "high"},
    "mable": {"meaning": "match version of Cover 3 per one unverified summary; another summary described a zone/man mix", "status": "football_hypothesis", "src": "mutgg_cover3_naming", "conf": "low"},
    "lock": {"meaning": "unknown for this call; EA's 'Lock' CHECK means defenders keep original assignments (not shown to apply here)", "status": "unknown", "src": "ea_deep_dive", "conf": "low"},
    "seam": {"meaning": "a defender has the seam zone; who is unverified", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "show2": {"meaning": "shows a two-high look before rotating; who rotates when is unverified", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "quarters": {"meaning": "four deep; EA documents match checks for Quarters", "status": "ea_documented", "src": "ea_deep_dive", "conf": "high"},
    "palms": {"meaning": "two-read quarters variant; EA documents match checks for Palms; assignments unverified", "status": "ea_documented", "src": "ea_deep_dive", "conf": "medium"},
    "willie": {"meaning": "unknown token (possibly Will-linebacker related)", "status": "unknown", "src": None, "conf": "low"},
    "sammie": {"meaning": "unknown token (possibly Sam-linebacker related)", "status": "unknown", "src": None, "conf": "low"},
    "str": {"meaning": "variant applied to the strong side", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "wk": {"meaning": "variant applied to the weak side", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "field": {"meaning": "variant applied to the field side", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "zone": {"meaning": "zone coverage (label says Zone)", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "coffee_house": {"meaning": "unknown token", "status": "unknown", "src": None, "conf": "low"},
}
VARIANT_KEYS = tuple(VARIANTS)

# pressure concepts. 'zone_blitz' includes fire zone; 'sim' = simulated pressure; 'stunt' = loop/Texas; 'unspecified' = a pressure
# NAME without a type; 'standard' = a coverage-only name (a normal rush is PRESUMED, not verified).
PRESSURE: dict[str, dict] = {
    "standard": {"meaning": "no pressure token in the label: a normal pass rush is presumed (rusher count unverified)", "status": "football_hypothesis",
                 "src": "domain_convention", "conf": "low"},
    "zone_blitz": {"meaning": "five or more rush while a normally-rushing player drops into coverage; fire zone = rush five with 3-3 coverage behind",
                   "status": "football_hypothesis", "src": "wiki_zone_blitz", "conf": "medium"},
    "man_blitz": {"meaning": "extra rushers while the remaining defenders play man", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "sim": {"meaning": "simulated pressure: EA describes a Zone Pressure STYLE built around simulated pressure with zone integrity behind it",
            "status": "ea_documented", "src": "ea_deep_dive", "conf": "medium"},
    "stunt": {"meaning": "rushers loop/exchange gaps (Loop / Texas stunt)", "status": "football_hypothesis", "src": "domain_convention", "conf": "low"},
    "unspecified": {"meaning": "the label names a blitz/pressure but not its type", "status": "name_derived", "src": None, "conf": "low"},
}
PRESSURE_KEYS = tuple(PRESSURE)

# ------------------------------------------------------------------------------------------------ per-name specs
# DSL: S(base, *variants, p=<pressure concept>, h=<digit hint after a pressure name>, origin=(...), u=(<tokens whose meaning is unknown>,),
#        basis=<why this classification>, target=<double-team target>)

def S(base=None, *variants, p="standard", h=None, origin=(), u=(), basis="name", target=None, shell_name=None, also=None):
    d = {"base": base, "variants": list(variants), "pressure": p, "hint": h, "origin": list(origin), "unknown_tokens": list(u),
         "basis": basis, "target": target, "shell_name": shell_name, "also": also}
    return d


AUDIT = "repo_audit_115"
NAME_SPECS: dict[str, dict] = {
    # --- Cover 1 family
    "1 Hole Press": S("c1", "hole", "press"), "1 Robber Press": S("c1", "robber", "press"),
    "Cover 1 Contain": S("c1", "contain"), "Cover 1 Hole": S("c1", "hole"), "Cover 1 Robber Press": S("c1", "robber", "press"),
    "1 Double WR1": S("c1", "double", target="WR1"), "1 Double WR2": S("c1", "double", target="WR2"),
    "1 Double Slot": S("c1", "double", target="Slot"), "1 Double TE": S("c1", "double", target="TE"),
    "1 LB Dog": S("c1", p="man_blitz", origin=("LB",), basis="name+convention"),
    "Cover 1 Spy": S("c1", "spy"), "Cover 1 Thief": S("c1", "thief", u=("thief",)), "Cover 1 Sting": S("c1", "sting", u=("sting",)),
    # --- Cover 2 family
    "Cover 2 Hard Flat": S("c2", "hard_flat"), "Cover 2 Man": S("c2", "man"), "Cover 2 Invert": S("c2", "invert"),
    "Cover 2 Roll": S("c2", "roll"), "Roll Cover 2 Invert": S("c2", "roll", "invert"), "2 Invert Hard Flat": S("c2", "invert", "hard_flat"),
    "Nickel 2 Trap": S("c2", "trap"), "SS 2 Trap": S("c2", "trap", origin=("SS",)), "Tampa 2": S("t2"), "Tampa 2 Drop": S("t2", "drop"),
    "Cover 2 Match": S("c2", "match"), "Cover 2 Drop": S("c2", "drop"), "Cover 2 Spy": S("c2", "spy"), "Cover 2 Man Spy": S("c2", "man", "spy"),
    # --- Cover 3 family
    "Cover 3": S("c3"), "Cover 3 Sky": S("c3", "sky"), "Cover 3 Cloud": S("c3", "cloud"), "Cover 3 Cloud Str": S("c3", "cloud", "str"),
    "Cover 3 Match": S("c3", "match"), "Cover 3 Buzz Match": S("c3", "buzz", "match"), "Cov 3 Buzz Match": S("c3", "buzz", "match"),
    "Cover 3 Buzz Match WK": S("c3", "buzz", "match", "wk"), "Cover 3 Hard Flat": S("c3", "hard_flat"), "Cover 3 Lock": S("c3", "lock", u=("lock",)),
    "Roll Cover 3 Cloud": S("c3", "roll", "cloud"), "Roll 23 Cloud": S(None, "roll", "cloud", u=("23",)),
    "Cover 3 Mable": S("c3", "mable"), "3 DBL Mable Cloud": S("c3", "double", "mable", "cloud", u=("dbl",)), "3 Double Mable": S("c3", "double", "mable"),
    "Cover 3 Buzz Seam": S("c3", "buzz", "seam"), "Cover 3 Buzz Press": S("c3", "buzz", "press"), "Cover 3 Buzz": S("c3", "buzz"),
    "Cover 3 Cloud Show 2": S("c3", "cloud", "show2"), "Cover 3 Sim Pressure": S("c3", p="sim", basis="name"),
    "Cover 3 Sammie": S("c3", "sammie", u=("sammie",)), "3 Double Buzz": S("c3", "double", "buzz"), "Cover 3 Drop": S("c3", "drop"),
    # --- Cover 4 / 6 / 9 / specialist
    "Cover 4 Press": S("c4", "press"), "Cover 4 Quarters": S("c4", "quarters"), "Cover 4 Palms": S("c4", "palms"), "Cover 4 Drop": S("c4", "drop"),
    "Cover 4 Drop Field": S("c4", "drop", "field"), "Cover 4 Show 2": S("c4", "show2"), "Cover 4 Zone": S("c4", "zone"),
    "Cover 6": S("c6"), "Cover 6 Willie": S("c6", "willie", u=("willie",)), "Cover 9 Show 2": S("c9", "show2"),
    "Prevent": S("prev"), "Man 3 Deep": S("m3d"),
    "Double Bracket": S(None, "bracket", "double"), "Double Bracket Switch": S(None, "bracket", "double", "switch"),
    "Bracket Switch Willie": S(None, "bracket", "switch", "willie", u=("willie",)),
    # --- pressure names (coverage behind them is UNKNOWN unless the label itself says it)
    "Fire Zone 3": S(None, p="zone_blitz", h=3), "Fire Zone 2": S(None, p="zone_blitz", h=2),
    "FS Zone Blitz": S(None, p="zone_blitz", origin=("FS",)), "SS Zone Blitz": S(None, p="zone_blitz", origin=("SS",)),
    "Buck Zone Blitz": S(None, p="zone_blitz", origin=("Buck",)), "FZ 3 Sky": S("c3", "sky", p="zone_blitz", basis="name"),
    "Zero Blitz": S("c0", p="man_blitz", basis="name+convention"), "OLB Fire Man": S(None, p="man_blitz", origin=("OLB",), basis="name+convention"),
    "Mid Blitz 0": S(None, p="unspecified", h=0, origin=("Mid",)), "Edge Blitz 0": S(None, p="unspecified", h=0, origin=("Edge",)),
    "FS Will Blitz 0": S(None, p="unspecified", h=0, origin=("FS", "Will")),
    "Edge Blitz 1": S(None, p="unspecified", h=1, origin=("Edge",)), "LB Blitz 1": S(None, p="unspecified", h=1, origin=("LB",)),
    "LB Sting 1": S(None, p="unspecified", h=1, origin=("LB",), u=("sting",), basis=AUDIT),
    "Edge Blitz 3": S(None, p="unspecified", h=3, origin=("Edge",)), "Hot Blitz 3": S(None, p="unspecified", h=3),
    "LB Blitz 3": S(None, p="unspecified", h=3, origin=("LB",)), "SS Blitz 3": S(None, p="unspecified", h=3, origin=("SS",)),
    "Nickel Blitz 3": S(None, p="unspecified", h=3, origin=("Nickel",)), "Sam Edge 3": S(None, p="unspecified", h=3, origin=("Sam", "Edge")),
    "DB Fire 2": S(None, p="unspecified", h=2, origin=("DB",), u=("fire",)), "DB Blitz Tex 3": S(None, p="stunt", h=3, origin=("DB",)),
    "Blitz Tex 3 Sim 3": S(None, p="sim", h=3, basis="name", also="stunt"), "Blitz Loop 3": S(None, p="stunt", h=3),
    "DT Mike Loop 3": S(None, p="stunt", h=3, origin=("DT", "Mike")),
    "Field Sim 3": S(None, "field", p="sim", h=3), "Nickel Sim 2": S(None, p="sim", h=2, origin=("Nickel",)),
    "Show SS Sim 2": S(None, "show2", p="sim", h=2, origin=("SS",)), "Sim Pressure 3": S(None, p="sim", h=3),
    "DBL Safety Blitz": S(None, p="unspecified", origin=("Safety",), u=("dbl",)), "DBL Safety Go": S(None, p="unspecified", origin=("Safety",), u=("dbl", "go"), basis=AUDIT),
    "Overload Blitz": S(None, p="unspecified", u=("overload",)), "Nickel Overload 3": S(None, p="unspecified", h=3, u=("overload",), origin=("Nickel",)),
    "Overload 3 Seam": S(None, "seam", p="unspecified", h=3, u=("overload",)), "Coffee House 3 Seam": S(None, "coffee_house", "seam", p="unspecified", h=3, u=("coffee house",)),
    "Okie Eagle 3": S(None, p="unspecified", h=3, u=("okie", "eagle"), basis=AUDIT), "Okie Roll 2": S(None, "roll", p="unspecified", h=2, u=("okie",), basis=AUDIT),
    "Over Storm Brave": S(None, p="unspecified", u=("storm brave",), basis=AUDIT), "Thunder Smoke": S(None, p="unspecified", u=("thunder smoke",), basis=AUDIT),
    "Free Fire": S(None, p="unspecified", u=("free fire",), basis=AUDIT), "Engage Eight": S(None, p="unspecified", u=("engage eight",), basis=AUDIT),
    "Inside Blitz": S(None, p="unspecified", origin=("Inside",)), "Rush Outside": S(None, p="unspecified", origin=("Outside",), basis=AUDIT),
    "3 Deep Blitz": S(None, p="unspecified", h=3, shell_name=3),
}

# External names known to exist outside the owner's installed book (the issue's required research list).
EXTERNAL_NAMES = (
    "Cover 1 Spy", "Cover 1 Thief", "Cover 1 Sting", "Cover 2 Match", "Cover 2 Drop", "Cover 2 Spy", "Cover 2 Man Spy", "Cover 3 Mable",
    "3 DBL Mable Cloud", "3 Double Mable", "Cover 3 Buzz Seam", "Cover 3 Buzz Press", "Cover 3 Cloud Show 2", "Cover 3 Sim Pressure",
    "Cover 3 Sammie", "Cover 4 Show 2", "Cover 4 Zone", "Prevent", "Man 3 Deep", "3 Double Buzz", "Cover 3 Drop", "Cover 3 Buzz",
    "3 Deep Blitz", "Fire Zone 2", "FZ 3 Sky", "Buck Zone Blitz", "SS 2 Trap",
)
# Where the Madden Prodigy catalog showed the play (existence only; the page was read partially).
EXTERNAL_OBSERVED_FORMATIONS = {
    "Cover 1 Spy": "Dime Normal", "Cover 1 Sting": "2-4-5 Over Wide", "Cover 3 Buzz Seam": "Dime Normal",
    "Cover 3 Buzz Press": "Dime Normal", "Cover 3 Cloud Show 2": "2-4-5 Over Wide", "Cover 3 Sim Pressure": "2-4-5 Over Wide",
    "3 Double Buzz": "Dime Normal", "Fire Zone 2": "5-2 Normal", "Buck Zone Blitz": "Dime 1-4 Dbl Mug", "SS 2 Trap": "3-4 Odd",
}

# Conditional offensive stress HYPOTHESES per structural trait. These are documentation, NEVER model inputs and never a play pick.
STRESS: dict[str, str] = {
    "c0": "No deep help: protection and quick separation decide the snap; a clean pocket exposes deep shots.",
    "c1": "Single-high man: crossing/rub routes and the hole between the safety and the underneath defenders.",
    "hole": "A middle underneath defender can jump crossers/digs over the middle (if the hole defender does what the source says).",
    "robber": "A safety-origin underneath defender can disguise as two-high before dropping into the middle.",
    "contain": "A spy/contain structure leaves fewer players on routes; scramble-heavy offenses are the target, not route combinations.",
    "c2": "Two-high zone: flats are squeezed by corners but the deep middle and sideline windows behind them are exposed.",
    "t2": "Linebacker takes the deep middle on a read pass; the deep middle is better defended, short crossers and tight ends are the usual stress.",
    "c3": "Three-deep zone: four underneath defenders leave seams and the flats stressed by floods/curl-flat combinations.",
    "c4": "Four deep: vertical shots are well covered; short and intermediate throws and runs are the usual stress.",
    "c6": "Quarters to one side and half to the other: the two sides defend differently, so route-side matters.",
    "man": "Man underneath: rubs, picks and crossers; a mobile QB can punish man with the safety(ies) tied to the deep field.",
    "match": "Pattern-match: defenders pass routes off; route distribution that overloads one defender's rules can create leverage.",
    "zone_blitz": "A normally-rushing defender may drop: hot/quick outlets to the vacated area, but the rusher's identity is unknown.",
    "man_blitz": "Extra rushers with man behind: beat the rush with timing; single coverage on every receiver.",
    "sim": "Simulated pressure shows rushers but may rush four: protection overreaction can leave extra defenders in coverage.",
    "stunt": "Looping/exchanging rushers stress interior gaps and protection communication.",
    "unspecified": "A pressure named without a type: the rush count and the coverage behind it are unknown.",
    "bracket": "Two defenders on one receiver: the doubled receiver is neutralized; the rest of the field is one-on-one.",
}
STRESS_LIMITATION = ("Hypothesis only. Actual defender alignment, user/opponent adjustments (unobservable on offense), route art and "
                     "personnel matchups decide the result; not a recommended play and not a guaranteed beater.")

# Differences between sibling variants worth stating explicitly (token -> token comparisons). Statuses follow the VARIANT records.
SIBLING_NOTES: dict[frozenset, str] = {
    frozenset({"cloud", "sky"}): "Cloud vs Sky: both Cover 3 flat-rotation names; under the (unverified) naming convention a CORNER takes the flat in Cloud and a SAFETY in Sky, which changes who is in deep thirds.",
    frozenset({"sky", "buzz"}): "Sky vs Buzz: under the (unverified) naming convention a SAFETY takes the flat in Sky and a LINEBACKER in Buzz.",
    frozenset({"cloud", "buzz"}): "Cloud vs Buzz: under the (unverified) naming convention a CORNER takes the flat in Cloud and a LINEBACKER in Buzz.",
    frozenset({"hole", "robber"}): "Hole vs Robber: per one pre-release article the hook/hole defender is a linebacker in Hole and comes from the safety in Robber.",
    frozenset({"hole", "contain"}): "Hole vs Contain: Contain turns the bonus defender into a QB spy and sets both edge rushers to contain instead of dropping a hook defender.",
    frozenset({"robber", "contain"}): "Robber vs Contain: a safety-origin hook defender versus a QB spy with contain rushers.",
    frozenset({"quarters", "palms"}): "Quarters vs Palms: both four-deep match families per EA; the exact corner/safety rules that differ are not documented.",
    frozenset({"man", "invert"}): "Man vs Invert: man underneath versus a zone with swapped roles; which defenders swap is unverified.",
    frozenset({"match", "sky"}): "Match vs Sky: match-rule pattern matching (EA documented) versus a flat-rotation spot-drop zone (the Sky rotation is an unverified naming convention).",
}
