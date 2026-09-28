"""Named-entity extraction for the CFB 27 meta (formations / plays / concepts).

Beyond keyword counts: every source document (web page, Reddit / Google News
item, YouTube title+description, YouTube transcript) is scanned for the exact
formation and play names in the CFB 27 playbook database
(``cfb_catalog``). A play mentioned near a formation that actually contains it
becomes a (formation, play) *pair* mention; otherwise it is a play-only mention.
Counts are per document (a long page can't dominate) with a small bonus for
repeated emphasis, and every document is recency-weighted (half-life
``HALF_LIFE_DAYS``). Transcript text is ASR (lowercase, no punctuation), so
matching is done on a normalised token stream with spoken-form aliases
("goal line" for GoalLine, "motion" for Mtn, "halfback" for HB, ...).
Pure stdlib, no network, no LLM.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterable

from cfb_coach.cfb_catalog import formations, pair_key

HALF_LIFE_DAYS = 21.0
UNDATED_WEIGHT = 0.6
ATTRIB_WINDOW = 60  # tokens: a play within this many tokens of a formation mention -> pair
REPEAT_BONUS = 0.25  # per extra mention in the same doc (up to MAX_REPEAT)
MAX_REPEAT = 4
KIND_WEIGHT = {"youtube_transcript": 1.0, "guide": 1.0, "patch": 0.6, "rss": 0.8, "youtube_meta": 0.5, "page": 0.8, "seed": 0.5}

# spoken / written aliases per token (regex alternatives, on the normalised stream)
_TOKEN_ALIASES: dict[str, str] = {
    "goalline": r"goal ?line",
    "mtn": r"(?:mtn|motion)",
    "hb": r"(?:hb|h b|halfback|half back|rb)",
    "qb": r"(?:qb|q b|quarterback)",
    "rpo": r"(?:rpo|r p o)",
    "pa": r"(?:pa|p a|play action)",
    "rz": r"(?:rz|red zone)",
    "te": r"(?:te|t e|tight end)",
    "wr": r"(?:wr|receiver)",
    "dbl": r"(?:dbl|double)",
    "wk": r"(?:wk|weak)",
    "5wr": r"(?:5wr|5 wide|five wide|5 wr)",
    "4wr": r"(?:4wr|4 wide|four wide)",
    "str": r"(?:str|strong)",
    "y": r"(?:y|why)",
}
# Short / generic play names that are concepts, not identifiable plays
_GENERIC_PLAYS = {"mesh", "bench", "stick", "flood", "verticals", "spacing", "curl", "duo", "levels",
                  "dagger", "smash", "shock", "drive", "cross", "slants", "spot", "flat", "screen",
                  "all curl", "all hitch", "curl flat", "hb draw", "qb draw", "hb dive", "inside zone",
                  "outside zone", "hb counter", "power", "trap", "hb zone"}
# Formation aliases people actually say (besides the full name)
_FORMATION_EXTRA: dict[str, list[str]] = {
    "Gun Bunch X Nasty": ["bunch x nasty", "bunch nasty"],
    "Gun Cluster": ["gun cluster", "cluster formation"],
    "Singleback Deuce Close": ["deuce close", "singleback deuce", "single back deuce close"],
    "Gun Deuce Close": ["gun deuce close"],
    "Gun Tight Doubles": ["tight doubles"],
    "Gun Power I Tight": ["power i tight", "gun power i", "power eye tight"],
    "Gun 5WR Tight": ["5 wide tight", "five wide tight", "5wr tight", "five-wide tight"],
    "Gun Trips TE Flex": ["trips te flex", "trips tight end flex"],
    "Gun Trips X Nasty": ["trips x nasty"],
    "Pistol Bunch X Nasty": ["pistol bunch x nasty", "pistol bunch nasty"],
    "Pistol Trips": ["pistol trips"],
    "Goal Line Normal": ["goal line normal"],
    "Singleback Bunch TE": ["singleback bunch te", "single back bunch"],
    "Singleback Bunch X Nasty": ["singleback bunch x nasty"],
    "Gun Bunch Str Nasty": ["bunch strong nasty", "bunch str nasty"],
    "Gun Bunch Spread": ["bunch spread"],
    "Gun Bunch TE": ["gun bunch te", "bunch te"],
    "Gun Empty Base Trio": ["empty base trio"],
    "Gun Trips TE": ["gun trips te"],
    "Gun Tight Y Off": ["tight y off"],
    "Gun Doubles Y Off Nasty": ["doubles y off nasty"],
    "Gun Bunch Open": ["bunch open"],
}


def normalise(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("&", " and ")
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return f" {re.sub(r'\s+', ' ', t).strip()} "


def _name_pattern(name: str) -> str:
    toks = normalise(name).split()
    parts = [_TOKEN_ALIASES.get(t, re.escape(t)) for t in toks]
    return r"(?<= )" + " ".join(parts) + r"(?= )"


@dataclass(frozen=True)
class _Vocab:
    forms: tuple[tuple[str, re.Pattern[str]], ...]
    plays: tuple[tuple[str, re.Pattern[str], tuple[str, ...], bool], ...]  # (play, rx, formations, generic)


def build_vocab(cat: dict[str, list[str]], extra: dict[str, list[str]] | None = None) -> _Vocab:
    """Vocabulary from any {formation: [plays]} catalog (CFB 27 or a Madden 27 side)."""
    extra = extra or {}
    forms = []
    for f in cat:
        alts = {normalise(f).strip()} | {normalise(a).strip() for a in extra.get(f, [])}
        rx = re.compile("|".join(_name_pattern(a) for a in sorted(alts, key=len, reverse=True)))
        forms.append((f, rx))
    by_play: dict[str, list[str]] = {}
    for f, plays in cat.items():
        for p in plays:
            by_play.setdefault(p, []).append(f)
    plays_out = []
    for p, fs in by_play.items():
        key = normalise(p).strip()
        if len(key.split()) < 2 or len(key) < 6:
            continue  # single words (Mesh, Duo, Bench...) are concepts, not identifiable plays
        generic = key in _GENERIC_PLAYS
        plays_out.append((p, re.compile(_name_pattern(p)), tuple(fs), generic))
    # longest names first so "Z Spot GoalLine" wins over "Z Spot"
    plays_out.sort(key=lambda t: -len(t[0]))
    return _Vocab(tuple(forms), tuple(plays_out))


@lru_cache(maxsize=1)
def vocab() -> _Vocab:
    return build_vocab(formations(), _FORMATION_EXTRA)


def extract_doc(text: str, *, voc: _Vocab | None = None) -> dict[str, dict[str, int]]:
    """One document -> {'formations': {F: n}, 'pairs': {F::P: n}, 'plays': {P: n}} (raw mention counts)."""
    t = normalise(text)
    if len(t) < 4:
        return {"formations": {}, "pairs": {}, "plays": {}}
    # token offsets for proximity
    starts = [m.start() for m in re.finditer(r"(?<= )\S", t)]

    def tok_index(pos: int) -> int:
        lo, hi = 0, len(starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        return max(0, lo - 1)

    v = voc or vocab()
    f_hits: list[tuple[int, str]] = []
    out_f: dict[str, int] = {}
    for f, rx in v.forms:
        for m in rx.finditer(t):
            f_hits.append((tok_index(m.start()), f))
            out_f[f] = out_f.get(f, 0) + 1
    taken: list[tuple[int, int]] = []
    out_pairs: dict[str, int] = {}
    out_plays: dict[str, int] = {}
    for p, rx, fs, generic in v.plays:
        for m in rx.finditer(t):
            if any(a <= m.start() < b for a, b in taken):
                continue  # part of a longer play name already counted
            taken.append((m.start(), m.end()))
            if not generic:  # generic names (HB Dive, Inside Zone) only count when tied to a formation
                out_plays[p] = out_plays.get(p, 0) + 1
            ti = tok_index(m.start())
            near = [(abs(ti - fi), f) for fi, f in f_hits if f in fs and abs(ti - fi) <= ATTRIB_WINDOW]
            if not near:
                continue  # our catalog is partial: never guess the formation from the play name alone
            f = min(near)[1]
            k = pair_key(f, p)
            out_pairs[k] = out_pairs.get(k, 0) + 1
    return {"formations": out_f, "pairs": out_pairs, "plays": out_plays}


def recency_weight(date_s: str | None, now: datetime) -> float:
    if not date_s:
        return UNDATED_WEIGHT
    try:
        dt = datetime.fromisoformat(str(date_s)[:10]).replace(tzinfo=timezone.utc)
    except ValueError:
        return UNDATED_WEIGHT
    age = max(0.0, (now - dt).total_seconds() / 86400.0)
    return round(0.5 ** (age / HALF_LIFE_DAYS), 4)


def aggregate(docs: Iterable[dict[str, Any]], *, now: datetime | None = None,
              voc: _Vocab | None = None) -> dict[str, Any]:
    """Docs ({label, kind, url, date, text}) -> recency-weighted named signals with per-source counts."""
    now = now or datetime.now(timezone.utc)
    agg: dict[str, dict[str, dict[str, Any]]] = {"formations": {}, "pairs": {}, "plays": {}}
    n_docs = 0
    for d in docs:
        text = d.get("text") or ""
        if not text:
            continue
        n_docs += 1
        ex = extract_doc(text, voc=voc)
        w = recency_weight(d.get("date"), now) * KIND_WEIGHT.get(str(d.get("kind") or "page"), 0.8)
        for bucket in ("formations", "pairs", "plays"):
            for k, n in ex[bucket].items():
                rec = agg[bucket].setdefault(k, {"score": 0.0, "docs": 0, "mentions": 0, "sources": []})
                bonus = 1.0 + REPEAT_BONUS * min(MAX_REPEAT, n - 1)
                rec["score"] = round(rec["score"] + w * bonus, 4)
                rec["docs"] += 1
                rec["mentions"] += n
                if len(rec["sources"]) < 6:
                    rec["sources"].append(
                        {"label": str(d.get("label") or "")[:140], "url": d.get("url") or "", "date": d.get("date") or "",
                         "kind": d.get("kind") or "", "n": n}
                    )
    for bucket in agg.values():
        for rec in bucket.values():
            rec["sources"].sort(key=lambda s: s.get("date") or "", reverse=True)
    return {
        "docs_scanned": n_docs,
        "half_life_days": HALF_LIFE_DAYS,
        "formations": dict(sorted(agg["formations"].items(), key=lambda kv: -kv[1]["score"])),
        "pairs": dict(sorted(agg["pairs"].items(), key=lambda kv: -kv[1]["score"])),
        "plays": dict(sorted(agg["plays"].items(), key=lambda kv: -kv[1]["score"])),
    }


def squash(x: float, cap: float, scale: float) -> float:
    return cap * math.tanh(max(0.0, x) / scale)


__all__ = ["aggregate", "build_vocab", "extract_doc", "normalise", "recency_weight", "vocab"]
