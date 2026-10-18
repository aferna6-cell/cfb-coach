"""One feature module for the v0.24 VOD success model: training, scoring, live.

The trainer (``train_vod_model_v6.prep``), the offline scorer and gate
reporting under ``scripts/madden_ml/``, and the live path
(``cfb_coach.madden.model.inference.vod_candidate_features``) all build
features here. Label rows and live inputs go through two thin adapters,
:func:`training_inputs` and :func:`live_inputs`, which only pick fields. Every
derived value comes from :func:`derive`, and every model input comes from
:func:`feature_levels`. That is what the G6 parity test checks.

Standard library only. This module must stay importable without numpy,
pandas or sklearn, because the live path imports it.

Rules (recipe f7c17d16, gate v2.2 4d4234aa):

* Family source. ``def_call`` is never a family source. Offense reads the
  observed look: ``coverage_seen`` then ``def_overlay`` in labels (each needs
  conf >= 0.5), the typed ``opp_look`` live. Defense reads our own candidate.
* Family mapping is the trainer's regex, digit reading kept
  ("DB Fire 2" -> cover_2). Each look token feeds one group at most: tokens
  the family used are removed before the pressure test, and a
  ``pressure_other`` family never also sets ``pressure``.
* Run/pass class: one word list, word boundaries. RPO first, then pass
  concepts, then run concepts, then formation/motion tags (jet, switch,
  bunch, double) that only mean pass when no concept word is present. RPO is its own class and folds into pass only in
  :func:`binary_run_pass`. QB Kneel has no class; it is a clock rule
  (:func:`kneel_clock_rule`) and never a candidate or training target.
* Distance: short <= 2, medium <= 6, long <= 10, xlong. Zone from yardline,
  else 100 - yards_to_goal: own_deep < 20, own < 50, opp < 80, red_zone.
  No location (or an untouched live yardline) is zone ``unk`` with
  ``location_missing`` set. No zone-keyed interaction is built for ``unk``.
* Defense family (addenda 1 and 3): exact lookup of our candidate's name in
  the defensive book (:class:`DefenseBook`) first; an off-book name falls
  back to the text pattern; only a name neither step resolves is ``other``.
  A blank name is ``none``. The path used (book/pattern/other) is recorded.
* Two-minute flag (group ``time``): ``on`` when clock <= 2:00 in Q2 or Q4,
  ``unknown`` when quarter or clock is blank (never ``off``), else ``off``.
  Situation term only; no clock interactions.
* No score, off_adj or def_adj features.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FEATURE_MODULE_VERSION = "vod_shared_features.v0.24.3"
RECIPE_SHA256 = "f7c17d16245116adad0cd30b3b2ab010b58cc3eb189aa6eb552abc5e85a28699"
GATE_SHA256 = "4d4234aae34894a005333f5e07845d516c57e12124f8cf55e16dddde93757ba8"

CONF_MIN = 0.5
MIN_KEYED_ROWS = 5
LAMBDA_KEYED_GRID = (64.0, 256.0, 1024.0, 4096.0)
TIE_MARGIN = 0.01

UNK = "unk"
TWO_MINUTE_ON = "on"
TWO_MINUTE_OFF = "off"
TWO_MINUTE_UNKNOWN = "unknown"
PRESSURE_FAMILIES = frozenset({"pressure_other", "cover_0"})

MANIFEST_NOTES = (
    "Live yardline defaults to own 25 when it is not typed, so an untyped "
    "yardline cannot be told apart from a real own 25 unless the live path "
    "sends yardline_touched=False, which maps it to the missing-location bucket.",
    "Live clock is usually blank, so the two-minute flag is usually 'unknown' "
    "live in Q2 and Q4 until the clock field ships.",
    "Dropped groups (both sides): score (a VOD label gap, known on about 8% of "
    "rows; live has score when typed), off_adj and def_adj (no live field).",
    "Dropped on defense: call, call_x_family, call_class, class_x_family "
    "(the offense call is not known before the snap; written choice: not used).",
    "def_call is never a family source.",
    "Defense family (addendum 3): exact lookup of our candidate's name in the "
    "defensive book (DefenseBook) first, then the text pattern for off-book names, "
    "else 'other'. The report records the book's source, size and sha256 and the "
    "book/pattern/other/none row counts. Default book: the public repo catalog; "
    "pass --defense-book for the applied book.",
)

# --- groups -----------------------------------------------------------------

BASELINE_GROUPS = ("down", "dist", "zone", "down_x_dist")
SITUATION_GROUPS = ("side", "down", "dist", "zone", "time", "down_x_dist")

OFFENSE_ENTITY_GROUPS = (
    "call", "call_class", "look", "look_family", "pressure", "call_x_family", "class_x_family",
)
OFFENSE_KEYED_GROUPS = ("call_x_down", "call_x_dist", "call_x_zone")
OFFENSE_BACKOFF_GROUPS = ("class_x_down_x_dist", "class_x_zone")

DEFENSE_ENTITY_GROUPS = ("look", "look_family", "pressure")
DEFENSE_KEYED_GROUPS = ("look_x_down", "look_x_dist", "look_x_zone")
DEFENSE_BACKOFF_GROUPS = ("family_x_down_x_dist", "family_x_zone", "pressure_x_down_x_dist")

KEYED_GROUPS = OFFENSE_KEYED_GROUPS + DEFENSE_KEYED_GROUPS
BACKOFF_GROUPS = OFFENSE_BACKOFF_GROUPS + DEFENSE_BACKOFF_GROUPS
DROPPED_GROUPS = {
    "offense": ("score", "off_adj", "def_adj"),
    "defense": ("score", "off_adj", "def_adj", "call", "call_x_family", "call_class", "class_x_family"),
}
ALL_GROUPS = tuple(
    dict.fromkeys(
        SITUATION_GROUPS
        + OFFENSE_ENTITY_GROUPS
        + DEFENSE_ENTITY_GROUPS
        + KEYED_GROUPS
        + BACKOFF_GROUPS
    )
)


def groups_for(side: str, *, situation_only: bool = False, enable_keyed: bool = True) -> tuple[str, ...]:
    """Every group the model for ``side`` can use."""
    if situation_only:
        return BASELINE_GROUPS
    if side == "defense":
        extra = DEFENSE_ENTITY_GROUPS + ((DEFENSE_KEYED_GROUPS + DEFENSE_BACKOFF_GROUPS) if enable_keyed else ())
    else:
        extra = OFFENSE_ENTITY_GROUPS + ((OFFENSE_KEYED_GROUPS + OFFENSE_BACKOFF_GROUPS) if enable_keyed else ())
    return SITUATION_GROUPS + extra


def group_penalty(group: str, hp: Sequence[float], lambda_keyed: float | None = None) -> float:
    """L2 penalty per group. hp = (situation, entity, interaction)."""
    lam_sit, lam_entity, lam_inter = (float(value) for value in hp)
    if group in KEYED_GROUPS:
        if lambda_keyed is None:
            raise ValueError("lambda_keyed is required for candidate-by-situation groups")
        return float(lambda_keyed)
    if group in BACKOFF_GROUPS:
        return 2.0 * lam_sit
    if group in ("side", "down", "dist", "zone", "time"):
        return lam_sit
    if group in ("down_x_dist", "call_class", "look_family", "pressure", "class_x_family"):
        return 2.0 * lam_sit
    if group in ("call", "look"):
        return lam_entity
    return lam_inter


# --- text helpers -------------------------------------------------------------

def norm_key(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    return text or None


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


# --- run / pass / rpo -------------------------------------------------------------

_KNEEL_RE = re.compile(r"\b(qb kneel|kneel|take a knee|victory)\b")
_RPO_RE = re.compile(r"\brpo\b")
PASS_WORDS = (
    r"\b(screens?|pa|flat|verts?|verticals|posts?|dig|curls?|slants?|mesh|smash|stick|corners?|"
    r"out|outs|seam|cross|drive|wheel|flood|spacing|dagger|levels|shallow|hitch|pivot|"
    r"bench|comebacks?|streak|fades?|bubble|spot|angle|texas|unders?|sail|choice|strike|"
    r"y-?cross|crossers?|crosses|clearout|vertical|hook|in|swirl|tare|gos?|cobra|scissors|snag|whip|pass|"
    r"hail mary|flea flicker|flicker|shovel|sprint|deep|swing|rub|salem|zig|sluggo|drags?|ohio|boot|clear|y-?wheel|"
    r"leak|trail|shake|glance|in n up|corndog|sit|ins|traffic|hi lo|replace|attack|divide|"
    r"mills|escape)\b"
)
# Formation, motion and tag words that also appear in run names ("Jet Sweep",
# "Switch HB Dive", "Bunch HB Draw", "Double Wing Power"). Any run-concept word
# (zone, dash, power, dive, blast, sweep, toss, draw, wrap, end around, sneak,
# push, ...) overrides them; they mean pass only when the name has no pass
# concept and no run concept.
WEAK_PASS_WORDS = r"\b(jet|switch|bunch|double)\b"
RUN_WORDS = (
    r"\b(hb|fb|qb sneak|qb power|qb draw|dive|zone|power|counter|stretch|iso|slam|"
    r"toss|sweep|draw|trap|option|sneak|lead|pitch|base|duo|wham|split|outside|inside|"
    r"off tackle|read option|speed option|load option|wk|reverse|g down|"
    r"dash|blast|wrap|end around|end arnd|push)\b"
)
_PASS_RE = re.compile(PASS_WORDS)
_RUN_RE = re.compile(RUN_WORDS)
_WEAK_PASS_RE = re.compile(WEAK_PASS_WORDS)


def is_kneel(play: Any) -> bool:
    """QB Kneel and victory formation. Never a candidate or training target."""
    text = norm_key(play)
    return bool(text and _KNEEL_RE.search(text))


def call_class(play: Any) -> str:
    """``none`` (no play, or a kneel), ``rpo``, ``pass``, ``run`` or ``unknown``."""
    text = norm_key(play)
    if not text or is_kneel(text):
        return "none"
    if _RPO_RE.search(text):
        return "rpo"
    if _PASS_RE.search(text):
        return "pass"
    if _RUN_RE.search(text):
        return "run"
    if _WEAK_PASS_RE.search(text):
        return "pass"
    return "unknown"


def binary_run_pass(cls: str) -> str:
    """Fold RPO into pass. Use only where a feature needs run versus pass."""
    return "pass" if cls == "rpo" else cls


def drop_kneel(candidates: Iterable[Any], play_of: Any = None) -> list[Any]:
    """Remove kneel candidates before ranking. ``play_of`` maps a candidate to its play name."""
    getter = play_of or (lambda item: getattr(item, "play", item))
    return [item for item in candidates if not is_kneel(getter(item))]


def kneel_clock_rule(
    *,
    quarter: Any,
    clock: Any,
    down: Any,
    score_us: Any,
    score_them: Any,
    opp_timeouts: Any = None,
) -> bool | None:
    """Whether kneeling out the clock applies, decided before ranking.

    ``None`` when any input is unknown (the rule never fires on a guess).
    True only in Q4 with a lead, downs 1-3, and a clock the remaining kneels
    can run out: about 40 s per kneel, minus a stop for each opponent timeout
    (three assumed when the count is unknown).
    """
    q = _quarter(quarter)
    secs = clock_seconds(clock)
    d = _int(down)
    us, them = _int(score_us), _int(score_them)
    if q is None or secs is None or d is None or us is None or them is None:
        return None
    if q != 4 or us <= them or d not in (1, 2, 3):
        return False
    kneels = 4 - d
    timeouts = _int(opp_timeouts)
    timeouts = 3 if timeouts is None else max(0, min(3, timeouts))
    stops = min(timeouts, kneels)
    burn = kneels * 40 - stops * 38
    return secs <= burn


# --- look family and pressure ---------------------------------------------------

PRESSURE = (
    r"(blitz|fire|dog|sting|crash|edge|overload|\bsim\b|pinch|zero|mug|shoot|pressure|gaps|lurk|buck|"
    r"loop|storm|smoke|thunder|safety go)"
)
_PRESSURE_RE = re.compile(PRESSURE)


def _tokens(text: str) -> tuple[str, list[tuple[int, int]]]:
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    return text, spans


def _token_ids_for_span(spans: list[tuple[int, int]], start: int, end: int) -> set[int]:
    return {i for i, (a, b) in enumerate(spans) if a < end and start < b}


def _family_with_tokens(name: Any) -> tuple[str, set[int], list[str]]:
    """Trainer regex mapping. Returns (family, token ids used, tokens)."""
    raw = norm_key(name)
    if not raw:
        return "none", set(), []
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", raw)).strip()
    if not text:
        return "other", set(), []
    text, spans = _tokens(text)
    tokens = [text[a:b] for a, b in spans]
    # compact string with an index map back to token ids
    compact_chars: list[str] = []
    owner: list[int] = []
    for i, (a, b) in enumerate(spans):
        for ch in text[a:b]:
            compact_chars.append(ch)
            owner.append(i)
    compact = "".join(compact_chars)

    def from_compact(match: re.Match[str]) -> set[int]:
        return {owner[k] for k in range(match.start(), match.end())}

    def from_text(match: re.Match[str], group: int = 0) -> set[int]:
        return _token_ids_for_span(spans, match.start(group), match.end(group))

    m = re.search(r"tampa", text)
    if m:
        used = from_text(m)
        used |= {i for i, tok in enumerate(tokens) if tok == "2"}
        return "tampa_2", used, tokens
    m = re.search(r"cov(er)?0|zero|blitz0\b", compact)
    if m:
        return "cover_0", from_compact(m), tokens
    m = re.search(r"\b(blitz|buck|pinch|fire) ?[0o]$", text)
    if m:
        return "cover_0", from_text(m), tokens
    m = re.search(r"cov(er)?2man", compact)
    if m:
        return "cover_2_man", from_compact(m), tokens
    m = re.search(r"\b2 man\b", text)
    if m:
        return "cover_2_man", from_text(m), tokens
    m = re.search(r"cov(er)?(\d)", compact)
    if m:
        return f"cover_{m.group(2)}", from_compact(m), tokens
    m = re.search(r"quarters|palms", text)
    if m:
        return "cover_4", from_text(m), tokens
    m = re.search(r"robber|hole", text)
    if m:
        return "cover_1", from_text(m), tokens
    m1 = re.search(r"\bcloud\b", text)
    if m1:
        # Cloud: a 3 anywhere in the name ("Roll 23 Cloud") is a Cover 3 rotation; no 3 is Cover 2.
        threes = {i for i, tok in enumerate(tokens) if "3" in tok}
        if threes:
            return "cover_3", from_text(m1) | threes, tokens
        return "cover_2", from_text(m1), tokens
    m1 = re.search(r"\b(buzz|sky)\b", text)
    m2 = re.search(r"\b3\b", text)
    if m1 and m2:
        return "cover_3", from_text(m1) | from_text(m2), tokens
    m1 = re.search(r"invert", text)
    m2 = re.search(r"\b2\b", text)
    if m1 and m2:
        return "cover_2", from_text(m1) | from_text(m2), tokens
    m = re.search(r"\b([0-9])\b", text)
    if m and m.group(1) in "012345679":
        family = "cover_0" if m.group(1) == "0" else f"cover_{m.group(1)}"
        return family, from_text(m), tokens
    m = re.search(r"\bbracket\b", text)
    if m:
        # Bracket coverage is a man family, never pressure on its own.
        return "man_other", from_text(m), tokens
    m = re.search(r"\bman\b", text)
    if m:
        return "man_other", from_text(m), tokens
    m = _PRESSURE_RE.search(text)
    if m:
        return "pressure_other", set(range(len(tokens))), tokens
    return "other", set(), tokens


def look_family(name: Any) -> str:
    """Coverage shell family of a look or play name. ``none`` when blank."""
    return _family_with_tokens(name)[0]


def family_and_pressure(name: Any) -> tuple[str, bool]:
    """Family, then pressure from the tokens the family did not use (de-dup rule)."""
    family, used, tokens = _family_with_tokens(name)
    if family in ("none", "pressure_other"):
        return family, False
    residue = " ".join(tok for i, tok in enumerate(tokens) if i not in used)
    return family, bool(residue and _PRESSURE_RE.search(residue))


def is_pressure(family: str | None, flag: bool | None) -> bool:
    """Blitz share: family pressure_other or cover_0, or the pressure flag on."""
    return bool(flag) or (family in PRESSURE_FAMILIES)


# --- defensive book lookup -----------------------------------------------------

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "madden27" / "playbooks.json"


def _book_names(source: Any) -> list[str]:
    """Play names from a list, a ``{"formations": {...}}`` side, a ``{"defense": ...}``
    playbook, or an ``active_playbook_json`` blob (``{"applied": {"defense": ...}}``)."""
    if source is None:
        return []
    if isinstance(source, (str, bytes)):
        return [source] if isinstance(source, str) else []
    if isinstance(source, Mapping):
        if "applied" in source:
            return _book_names(source.get("applied"))
        if "defense" in source:
            return _book_names(source.get("defense"))
        if "formations" in source:
            return _book_names(source.get("formations"))
        if "plays" in source:
            return _book_names(source.get("plays"))
        names: list[str] = []
        for value in source.values():
            names.extend(_book_names(value))
        return names
    names = []
    for item in source:
        names.extend(_book_names(item))
    return names


def pattern_family(name: Any) -> tuple[str, bool, str | None]:
    """Text pattern alone: (family, pressure, path).

    ``path`` is ``pattern`` when the pattern resolves the name, ``other`` when
    it does not, ``None`` when blank. There are no named exceptions (addendum 3).
    """
    key = norm_key(name)
    if not key:
        return "none", False, None
    family, pressure = family_and_pressure(key)
    if family == "other":
        return "other", False, "other"
    return family, pressure, "pattern"


class DefenseBook:
    """Exact (case- and space-normalized) name lookup in a defensive book.

    Each book play's family and pressure come from its own book name, once,
    when the book is built. Off-book names fall back to the text pattern.
    """

    def __init__(self, names: Iterable[Any], *, source: str = "custom") -> None:
        table: dict[str, tuple[str, bool]] = {}
        for name in names:
            key = norm_key(name)
            if key and key not in table:
                family, pressure, _path = pattern_family(key)
                table[key] = (family, pressure)
        self.table = table
        self.source = source
        self.sha256 = hashlib.sha256("\n".join(sorted(table)).encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self.table)

    def __contains__(self, name: Any) -> bool:
        key = norm_key(name)
        return bool(key) and key in self.table

    def lookup(self, name: Any) -> tuple[str, bool, str | None]:
        """(family, pressure, path), addendum 3.

        Book first: an exact book name gets its book family (path ``book``).
        An off-book name gets the text-pattern family (path ``pattern``); a
        name neither step resolves is ``other`` (path ``other``). A blank name
        is ``("none", False, None)``. Glued calls are blanked upstream by the
        trainer's frozen filter; a glued name that reaches here is mapped by
        the pattern like any other name.
        """
        key = norm_key(name)
        if not key:
            return "none", False, None
        hit = self.table.get(key)
        if hit is not None:
            return hit[0], hit[1], "book"
        return pattern_family(key)

    def describe(self) -> dict[str, Any]:
        return {"source": self.source, "plays": len(self.table), "sha256": self.sha256}

    @classmethod
    def from_playbook(cls, source: Any, *, label: str = "playbook") -> "DefenseBook":
        return cls(_book_names(source), source=label)

    @classmethod
    def from_file(cls, path: str | Path) -> "DefenseBook":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(_book_names(data), source=f"file:{Path(path).name}")


_CATALOG_BOOK: DefenseBook | None = None
_ACTIVE_BOOK: DefenseBook | None = None


def catalog_defense_book() -> DefenseBook:
    """Every defensive play in the public repo catalog (data/madden27/playbooks.json)."""
    global _CATALOG_BOOK
    if _CATALOG_BOOK is None:
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
        books = (data.get("books") or {}).get("defense") or {}
        _CATALOG_BOOK = DefenseBook(_book_names(books), source="catalog:madden27/playbooks.json")
    return _CATALOG_BOOK


def set_defense_book(book: DefenseBook | Any | None) -> DefenseBook | None:
    """Set the process-wide defensive book (``None`` goes back to the catalog)."""
    global _ACTIVE_BOOK
    if book is None or isinstance(book, DefenseBook):
        _ACTIVE_BOOK = book
    else:
        _ACTIVE_BOOK = DefenseBook.from_playbook(book)
    return _ACTIVE_BOOK


def active_defense_book() -> DefenseBook:
    return _ACTIVE_BOOK if _ACTIVE_BOOK is not None else catalog_defense_book()


def _resolve_book(book: DefenseBook | Any | None) -> DefenseBook:
    if book is None:
        return active_defense_book()
    if isinstance(book, DefenseBook):
        return book
    return DefenseBook.from_playbook(book)


def defense_family_path(name: Any, book: DefenseBook | Any | None = None) -> tuple[str, bool, str | None]:
    """Defense candidate (family, pressure, path) — book, then pattern, then other (addendum 3).

    The one function training, scoring, eval labels and live use. ``book`` is a
    :class:`DefenseBook`, any playbook shape :func:`DefenseBook.from_playbook`
    reads, or ``None`` for the active book (catalog unless one was set).
    ``path`` is ``book``, ``pattern`` or ``other``; ``None`` for a blank name.
    """
    return _resolve_book(book).lookup(name)


def defense_family(name: Any, book: DefenseBook | Any | None = None) -> tuple[str, bool]:
    """Family and pressure from :func:`defense_family_path`."""
    family, pressure, _path = defense_family_path(name, book)
    return family, pressure


def defense_on_book(name: Any, book: DefenseBook | Any | None = None) -> bool | None:
    """True for an exact book name, False off-book, None when blank."""
    if not norm_key(name):
        return None
    return name in _resolve_book(book)


def look_text_and_source(defense_look: Mapping[str, Any] | None, conf: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    """Observed look in a label row: coverage_seen, then def_overlay. Never def_call."""
    look = defense_look or {}
    conf = conf or {}
    for field in ("coverage_seen", "def_overlay"):
        value = look.get(field)
        if norm_key(value) and float(conf.get(field, 0) or 0) >= CONF_MIN:
            return str(value), field
    return None, None


# --- situation ------------------------------------------------------------------

def down_level(down: Any) -> str:
    d = _int(down)
    return str(d) if d in (1, 2, 3, 4) else UNK


def dist_bucket(distance: Any) -> str:
    yards = _int(distance)
    if yards is None or yards < 0:
        return UNK
    if yards <= 2:
        return "short"
    if yards <= 6:
        return "medium"
    if yards <= 10:
        return "long"
    return "xlong"


def location(yardline: Any, yards_to_goal: Any = None, yardline_touched: Any = None) -> int | None:
    """Yardline 0-100 from our goal. ``yardline_touched=False`` means missing."""
    if yardline_touched is False:
        return None
    spot = _int(yardline)
    if spot is None:
        ytg = _int(yards_to_goal)
        if ytg is not None:
            spot = 100 - ytg
    if spot is None or spot < 0 or spot > 100:
        return None
    return spot


def zone_bucket(yardline: Any, yards_to_goal: Any = None, yardline_touched: Any = None) -> str:
    spot = location(yardline, yards_to_goal, yardline_touched)
    if spot is None:
        return UNK
    if spot < 20:
        return "own_deep"
    if spot < 50:
        return "own"
    if spot < 80:
        return "opp"
    return "red_zone"


def clock_seconds(clock: Any) -> int | None:
    if clock is None or isinstance(clock, bool):
        return None
    if isinstance(clock, (int, float)):
        return int(clock) if clock >= 0 else None
    text = str(clock).strip()
    if not text:
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    if re.fullmatch(r"\d+", text):
        return int(text)
    return None


def _quarter(quarter: Any) -> int | None:
    if quarter is None or isinstance(quarter, bool):
        return None
    m = re.fullmatch(r"(?:q)?\s*(\d)", str(quarter).strip().lower())
    return int(m.group(1)) if m else None


def two_minute_flag(quarter: Any, clock: Any) -> str:
    """``on`` / ``off`` / ``unknown``. Blank quarter or clock is unknown, never off."""
    q = _quarter(quarter)
    secs = clock_seconds(clock)
    if q is None or secs is None:
        return TWO_MINUTE_UNKNOWN
    if q in (2, 4) and secs <= 120:
        return TWO_MINUTE_ON
    return TWO_MINUTE_OFF


# --- inputs -------------------------------------------------------------------

def _side(value: Any) -> str:
    text = norm_key(value) or ""
    if text.startswith("off"):
        return "offense"
    if text.startswith("def"):
        return "defense"
    return "unknown"


def derive(
    *,
    side: Any,
    candidate: Any,
    observed_look: Any,
    down: Any,
    distance: Any,
    yardline: Any,
    yards_to_goal: Any = None,
    yardline_touched: Any = None,
    quarter: Any = None,
    clock: Any = None,
    family_src: str | None = None,
    defense_book: DefenseBook | Any | None = None,
) -> dict[str, Any]:
    """The single derivation both paths use.

    Offense: ``candidate`` is our play; family/look come from ``observed_look``.
    Defense: ``candidate`` is our defensive play and is the family/look source,
    through the exact book lookup (:func:`defense_family`).
    """
    side_n = _side(side)
    cand_text = norm_key(candidate)
    family_path = None
    if side_n == "defense":
        look_raw = cand_text
        src = "candidate" if cand_text else None
        call = None
        family, pressure, family_path = _resolve_book(defense_book).lookup(cand_text)
    else:
        look_raw = norm_key(observed_look)
        src = (family_src or "opp_look") if look_raw else None
        call = cand_text
        family, pressure = family_and_pressure(look_raw)
    spot = location(yardline, yards_to_goal, yardline_touched)
    return {
        "side": side_n,
        "candidate_text": cand_text,
        "observed_text": norm_key(observed_look) if side_n != "defense" else None,
        "call": None if (call is None or is_kneel(call)) else call,
        "call_class": call_class(call) if side_n != "defense" else "none",
        "kneel": bool(cand_text and is_kneel(cand_text)),
        "look": look_raw,
        "look_family": family,
        "family_src": src,
        "family_path": family_path,
        "pressure": bool(pressure),
        "down": down_level(down),
        "dist": dist_bucket(distance),
        "zone": zone_bucket(spot),
        "location_missing": spot is None,
        "time": two_minute_flag(quarter, clock),
    }


def training_inputs(row: Mapping[str, Any], *, defense_book: DefenseBook | Any | None = None) -> dict[str, Any]:
    """Label-row adapter (v0.24 rows: offense_call / defense_look / conf)."""
    conf = row.get("conf") or {}
    side = _side(row.get("side"))
    defense_look = row.get("defense_look") or {}
    if side == "defense":
        value = defense_look.get("def_call")
        # On a defense row def_call is our own call: the candidate, not the observed look.
        candidate = value if float(conf.get("def_call", 0) or 0) >= CONF_MIN else None
        observed, src = None, None
    else:
        offense_call = row.get("offense_call") or {}
        candidate = offense_call.get("play") if float(conf.get("play", 0) or 0) >= CONF_MIN else None
        observed, src = look_text_and_source(defense_look, conf)
    return derive(
        side=side,
        candidate=candidate,
        observed_look=observed,
        down=row.get("down"),
        distance=row.get("distance"),
        yardline=row.get("yardline"),
        yards_to_goal=row.get("yards_to_goal"),
        yardline_touched=row.get("yardline_touched"),
        quarter=row.get("quarter"),
        clock=row.get("clock"),
        family_src=src,
        defense_book=defense_book,
    )


def live_inputs(
    side: Any,
    candidate: Any,
    *,
    opp_look: Any = None,
    down: Any = None,
    distance: Any = None,
    yardline: Any = None,
    yardline_touched: Any = None,
    quarter: Any = None,
    clock: Any = None,
    defense_book: DefenseBook | Any | None = None,
) -> dict[str, Any]:
    """Live adapter: what GameState and opp_look carry. No def_call, no coverage_seen."""
    return derive(
        side=side,
        candidate=candidate,
        observed_look=opp_look,
        down=down,
        distance=distance,
        yardline=yardline,
        yardline_touched=yardline_touched,
        quarter=quarter,
        clock=clock,
        defense_book=defense_book,
    )


def swap_candidate(
    inputs: Mapping[str, Any], candidate: Any, *, defense_book: DefenseBook | Any | None = None,
) -> dict[str, Any]:
    """Same situation and look, a different candidate (ranking, constant-pick checks)."""
    out = dict(inputs)
    cand_text = norm_key(candidate)
    out["candidate_text"] = cand_text
    out["kneel"] = bool(cand_text and is_kneel(cand_text))
    if inputs.get("side") == "defense":
        family, pressure, family_path = _resolve_book(defense_book).lookup(cand_text)
        out.update(look=cand_text, look_family=family, pressure=bool(pressure), family_path=family_path,
                   family_src="candidate" if cand_text else None)
    else:
        out["call"] = None if (cand_text is None or is_kneel(cand_text)) else cand_text
        out["call_class"] = call_class(cand_text)
    return out


def swap_look(inputs: Mapping[str, Any], look_text: Any) -> dict[str, Any]:
    """Offense only: same row, a different observed look (G4b counterfactual swap)."""
    out = dict(inputs)
    if inputs.get("side") == "defense":
        return out
    text = norm_key(look_text)
    family, pressure = family_and_pressure(text)
    out.update(observed_text=text, look=text, look_family=family, pressure=bool(pressure),
               family_src="swap" if text else None)
    return out


# --- model inputs -----------------------------------------------------------------

def feature_levels(
    inputs: Mapping[str, Any],
    *,
    situation_only: bool = False,
    enable_keyed: bool = True,
    allowed_keys: Mapping[str, set[str]] | None = None,
) -> dict[str, str]:
    """One level per active group. ``allowed_keys`` filters keyed groups (training fold)."""
    down, dist, zone = inputs["down"], inputs["dist"], inputs["zone"]
    feats: dict[str, str] = {
        "down": down,
        "dist": dist,
        "zone": zone,
        "down_x_dist": f"{down}|{dist}",
    }
    if situation_only:
        return feats
    feats["side"] = inputs.get("side") or "unknown"
    feats["time"] = inputs["time"]
    family = inputs.get("look_family") or "none"
    pressure = bool(inputs.get("pressure"))
    if inputs.get("look"):
        feats["look"] = inputs["look"]
    feats["look_family"] = family
    if pressure:
        feats["pressure"] = "yes"
    keyed: dict[str, str] = {}
    if inputs.get("side") == "defense":
        look = inputs.get("look")
        if enable_keyed:
            if look:
                keyed["look_x_down"] = f"{look}|{down}"
                keyed["look_x_dist"] = f"{look}|{dist}"
                if zone != UNK:
                    keyed["look_x_zone"] = f"{look}|{zone}"
            feats["family_x_down_x_dist"] = f"{family}|{down}|{dist}"
            if zone != UNK:
                feats["family_x_zone"] = f"{family}|{zone}"
            feats["pressure_x_down_x_dist"] = f"{'yes' if pressure else 'no'}|{down}|{dist}"
    else:
        call = inputs.get("call")
        cls = inputs.get("call_class") or "none"
        feats["call_class"] = cls
        if call:
            feats["call"] = call
            if family != "none":
                feats["call_x_family"] = f"{call}|{family}"
        feats["class_x_family"] = f"{cls}|{family}"
        if enable_keyed:
            if call:
                keyed["call_x_down"] = f"{call}|{down}"
                keyed["call_x_dist"] = f"{call}|{dist}"
                if zone != UNK:
                    keyed["call_x_zone"] = f"{call}|{zone}"
            feats["class_x_down_x_dist"] = f"{cls}|{down}|{dist}"
            if zone != UNK:
                feats["class_x_zone"] = f"{cls}|{zone}"
    for group, key in keyed.items():
        if allowed_keys is not None and key not in allowed_keys.get(group, ()):
            continue
        feats[group] = key
    return feats


def keyed_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {group: {} for group in KEYED_GROUPS}
    for inputs in rows:
        feats = feature_levels(inputs, enable_keyed=True, allowed_keys=None)
        for group in KEYED_GROUPS:
            key = feats.get(group)
            if key is not None:
                counts[group][key] = counts[group].get(key, 0) + 1
    return counts


def allowed_keys(rows: Iterable[Mapping[str, Any]], min_rows: int = MIN_KEYED_ROWS) -> dict[str, set[str]]:
    """Keyed-group keys with at least ``min_rows`` rows. Pass the training fold only."""
    counts = keyed_counts(rows)
    return {group: {key for key, n in table.items() if n >= min_rows} for group, table in counts.items()}


# --- ranking helpers -------------------------------------------------------------

def top_two_margin(probabilities: Sequence[float] | Mapping[Any, float] | Iterable[tuple[Any, float]]) -> float | None:
    """Top score minus the best *different play's* score. None with < 2 plays.

    Pass ``(play, score)`` pairs or a ``{play: score}`` map so the same play
    listed in two formations counts once (its best score); otherwise a book
    that repeats a play would always look like a tie. A bare list of numbers
    is treated as distinct candidates.
    """
    best: dict[Any, float] = {}
    if isinstance(probabilities, Mapping):
        items: Iterable[Any] = probabilities.items()
    else:
        items = probabilities
    for index, item in enumerate(items):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            name, score = item
            key = norm_key(name) or f"#{index}"
        else:
            key, score = f"#{index}", item
        value = float(score)
        if key not in best or value > best[key]:
            best[key] = value
    ordered = sorted(best.values(), reverse=True)
    if len(ordered) < 2:
        return None
    return ordered[0] - ordered[1]


def heuristic_fallback(
    observed_look: Any,
    margin: float | None,
    *,
    side: str = "offense",
    candidate: Any = None,
) -> str | None:
    """Q4/Q5: ``blank_look`` or ``tie`` sends the snap to the heuristic; else None.

    Offense needs an observed look. Defense ranks our own calls, so it never
    reads ``opp_look``; it is ``blank_look`` only when it has no candidate to
    score (``candidate`` given and blank). A margin under 0.01 is ``tie``.
    """
    if _side(side) == "defense":
        if candidate is not None and not norm_key(candidate):
            return "blank_look"
    elif not norm_key(observed_look):
        return "blank_look"
    if margin is None or margin < TIE_MARGIN:
        return "tie"
    return None


__all__ = [name for name in dir() if not name.startswith("_")]
