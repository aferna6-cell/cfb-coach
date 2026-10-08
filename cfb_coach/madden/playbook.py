"""Madden playbook of record — one active book per side, locked by the latest prep.

Contract (owner review on PR #2, trimmed plan is the default custom book):
  - Default prep saves the coach's trimmed plan as CUSTOM: the recommended source
    stock book (start Buccaneers O / 49ers D) plus the focused formations, with
    every play those formations have in that book. ``source_book`` records the
    source. It locks immediately. Explicit ``stock:NAME`` stays a stock book.
  - CUSTOM without ``source_book`` is the in-game editor book (explicit
    ``--o-book custom``): full checklist on the first one, then formation
    ADD/REMOVE. Those edits stay pending until ``prep --mark-applied``.
  - Switch any time via --o-book / --d-book.
  - Live `play` may only call formation+play pairs inside the locked (applied) book;
    with no locked book it refuses to call (run prep first).
  - The trimmed custom plan and stock picks lock at prep (nothing to build).

Persisted in the Madden DB meta key `active_playbook_json`:
  {"applied": {side: record}, "pending": {side: record}}
  record = {side, mode, name, formations: {formation: [plays]}, core, rev, locked_ts, reason}
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from cfb_coach.madden.data import _load_json

META_KEY = "active_playbook_json"
SIDES = ("offense", "defense")
STOCK, CUSTOM = "stock", "custom"
DEFAULT_STOCK = {"offense": "Buccaneers", "defense": "49ers"}
CUSTOM_NAME = {"offense": "PRIMARY META O (custom)", "defense": "PRIMARY META D (custom)"}

_BOOK_ALIASES = {
    "bucs": "Buccaneers", "buccaneers": "Buccaneers", "tb": "Buccaneers", "tampa": "Buccaneers",
    "tampa bay": "Buccaneers", "tampa bay buccaneers": "Buccaneers",
    "titans": "Titans", "tennessee": "Titans", "ten": "Titans", "tennessee titans": "Titans",
    "shotgun": "Shotgun Classic", "shotgun classic": "Shotgun Classic", "classic": "Shotgun Classic",
    "49ers": "49ers", "niners": "49ers", "sf": "49ers", "saleh": "49ers",
    "san francisco 49ers": "49ers",
}


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------

MAX_FOCUS = {"offense": 5, "defense": 5}  # formations the live caller works from in a stock book
MIN_FOCUS = 4
AUDIBLES_PER_FORMATION = 4
# Book recommendation (auto): start prior + hysteresis so research has to be clearly better
START_PRIOR = 0.30  # the configured starting book (Buccaneers O / 49ers D)
STAY_PRIOR = 0.15  # the book currently locked
TEAM_PRIOR = 0.10  # the primary team's own stock book (Lions)
RANK_PRIOR_MAX = 0.30  # seed rankings (Prodigy / TimeSaver / Civil / Operation Sports)
RESEARCH_BOOK_W = 0.55  # the book named in this prep's sources
RESEARCH_FORM_W = 0.30  # meta formations / plays named in this prep's sources that this book has
OWN_W = 0.30  # Aidan's own results in the current book
SWAP_MARGIN = 0.20  # a challenger must beat the current book by this much
SWITCH_COOLDOWN_PREPS = 2
# Formation focus inside a stock book
FOCUS_SEED = 0.50  # formation in the verified seed core for this side
FOCUS_STAY = 0.15
FOCUS_NAMED_W = 0.35
FOCUS_OWN_W = 0.40


def formation_catalog(side: str) -> dict[str, list[str]]:
    """Seed formations (custom-build vocabulary) with their verified callable plays."""
    seed = _load_json("seed.json")
    out: dict[str, list[str]] = {}
    if side == "offense":
        for name, meta in seed["playbooks"]["offense_formations"].items():
            out[name] = list(meta.get("verified") or meta.get("core") or [])
        for name, meta in (seed.get("delta_formations") or {}).items():
            plays = list(meta.get("verified") or [])
            if plays:
                out[name] = plays
    else:
        for name, meta in seed["playbooks"]["defense_packages"].items():
            out[name] = list(meta.get("calls") or [])
    return out


def formation_books(side: str, formation: str) -> str:
    seed = _load_json("seed.json")
    pool = (
        {**seed["playbooks"]["offense_formations"], **(seed.get("delta_formations") or {})}
        if side == "offense"
        else seed["playbooks"]["defense_packages"]
    )
    meta = pool.get(formation) or {}
    books = meta.get("books")
    return ", ".join(books) if isinstance(books, list) else str(meta.get("book") or "")


def stock_books(side: str) -> dict[str, dict[str, Any]]:
    """Every stock book we can call from: the per-book catalog (Huddle.gg) + seed metadata.

    ``formations`` = the default focus (seed-verified meta formations present in that book);
    ``all_formations`` = every formation in the book."""
    from cfb_coach.madden import catalog

    seed_books = dict((_load_json("seed.json").get("stock_books") or {}).get(side) or {})
    out: dict[str, dict[str, Any]] = {}
    for name in catalog.book_names(side):
        forms = catalog.book_formations(side, name)
        meta = dict(seed_books.get(name) or {})
        meta["team"] = meta.get("team") if "team" in meta else catalog.book_team(side, name)
        meta["formations"] = [f for f in (meta.get("formations") or []) if f in forms]
        meta["all_formations"] = list(forms)
        meta["catalogued"] = True
        out[name] = meta
    for name, meta in seed_books.items():  # seed-only books (not in the catalog yet)
        if name not in out:
            out[name] = dict(meta, catalogued=False)
    return out


def resolve_stock_book(side: str, raw: str) -> str:
    key = " ".join((raw or "").strip().lower().split())
    books = stock_books(side)
    name = _BOOK_ALIASES.get(key)
    if name not in books:
        name = next((b for b in books if b.lower() == key), None)
    if not name:  # team names: "detroit lions" / "lions" / "det"
        try:
            from cfb_coach.madden.franchise import resolve_nfl_team

            team, known = resolve_nfl_team(raw)
            if known:
                name = next((b for b, m in books.items() if m.get("team") == team), None)
        except ValueError:
            name = None
    if not name or name not in books:
        raise ValueError(
            f"Unknown {side} stock book {raw!r}. Known: {', '.join(books)}"
        )
    return name


def parse_choice(side: str, raw: str | None) -> tuple[str, str | None]:
    """--o-book / --d-book value → ("auto"|"custom"|"stock", book name|None)."""
    val = (raw or "auto").strip()
    low = val.lower()
    if low in ("", "auto"):
        return "auto", None
    if low == "custom":
        return CUSTOM, None
    if low == "stock":
        return STOCK, DEFAULT_STOCK[side]
    if low.startswith("stock:"):
        return STOCK, resolve_stock_book(side, val.split(":", 1)[1])
    return STOCK, resolve_stock_book(side, val)  # bare book name


def _book_plays(side: str, book: str) -> dict[str, list[str]]:
    """Formation -> plays IN THIS BOOK (catalog), else the seed-verified lists."""
    from cfb_coach.madden import catalog

    forms = catalog.book_formations(side, book)
    if forms:
        return forms
    cat = formation_catalog(side)
    return {f: cat[f] for f in (stock_books(side).get(book) or {}).get("formations") or [] if f in cat}


def _stock_formations(side: str, name: str, focus: list[str] | None = None) -> dict[str, list[str]]:
    plays = _book_plays(side, name)
    names = list(focus) if focus else select_focus(side, name)["formations"]
    return {f: list(plays[f]) for f in names if f in plays}


def _custom_formations(side: str, names: list[str]) -> dict[str, list[str]]:
    """Custom build: each formation brings the plays of the stock book it is added from."""
    from cfb_coach.madden import catalog

    cat = formation_catalog(side)
    out: dict[str, list[str]] = {}
    for f in names:
        src = next((b for b in (_load_json("seed.json").get("delta_formations") or {}).get(f, {}).get("books") or []
                    if catalog.book_formations(side, b).get(f)), None)
        src = src or (DEFAULT_STOCK[side] if catalog.book_formations(side, DEFAULT_STOCK[side]).get(f) else None)
        if src:
            out[f] = catalog.book_formations(side, src)[f]
        elif f in cat and cat[f]:
            out[f] = cat[f]
    return out


def custom_core(side: str) -> list[str]:
    return list((_load_json("seed.json").get("custom_core") or {}).get(side) or [])


def desired_custom(side: str, opp: dict[str, Any]) -> list[str]:
    """Core formations + persona formations (offense only)."""
    names = custom_core(side)
    if side == "offense":
        extras = (_load_json("seed.json").get("persona_custom_formations") or {}).get(
            (opp.get("archetype") or "").lower()
        ) or []
        names += [f for f in extras if f not in names and formation_catalog(side).get(f)]
    return names


# ---------------------------------------------------------------------------
# Research-driven choice (CFB parity: prep picks the book + formations)
# ---------------------------------------------------------------------------

def _tanh(x: float) -> float:
    import math

    return math.tanh(x)


def _own_formation_grade(lw: Any, formation: str, plays: list[str]) -> tuple[float, int]:
    """Mean normalised learned weight of Aidan's logged plays in a formation (and #plays logged)."""
    if lw is None:
        return 0.0, 0
    vals = []
    for p in plays:
        k = f"play::{formation}::{p}"
        if lw.n(k) > 0:
            vals.append(lw.norm(k))
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


def _pair_score(named: dict[str, Any], formation: str, plays: list[str]) -> float:
    pairs = (named or {}).get("pairs") or {}
    keys = {f"{formation}::{p}" for p in plays}
    return sum(float(v.get("score", 0.0)) for k, v in pairs.items() if k in keys)


def select_focus(
    side: str,
    book: str,
    *,
    named: dict[str, Any] | None = None,
    lw: Any = None,
    current: list[str] | None = None,
    cap: int | None = None,
) -> dict[str, Any]:
    """Pick the formations (full, every play) the coach works from in a stock book.

    Seed-verified meta formations first, then formations/plays named in this prep's
    research, Aidan's own results, and a small stay bonus (no churn)."""
    from cfb_coach.madden import catalog

    plays = _book_plays(side, book)
    seed_forms = set(formation_catalog(side))
    cap = cap or MAX_FOCUS[side]
    rows = []
    for f, ps in plays.items():
        if f.lower().startswith("hail mary"):
            continue
        s_seed = FOCUS_SEED if f in seed_forms else 0.0
        nf = float((((named or {}).get("formations") or {}).get(f) or {}).get("score", 0.0))
        s_named = FOCUS_NAMED_W * _tanh((nf + 0.5 * _pair_score(named or {}, f, ps)) / 2.0)
        own, n_own = _own_formation_grade(lw, f, ps)
        s_own = FOCUS_OWN_W * own
        s_stay = FOCUS_STAY if current and f in current else 0.0
        score = s_seed + s_named + s_own + s_stay
        why = []
        if s_seed:
            why.append("verified meta formation")
        if s_named > 0.01:
            why.append(f"named in this prep's research ({s_named:+.2f})")
        if n_own:
            why.append(f"your results {own:+.2f} over {n_own} play(s)")
        if s_stay:
            why.append("already in your plan")
        rows.append({"formation": f, "score": round(score, 3), "why": "; ".join(why), "n_plays": len(ps),
                     "family": catalog.formation_family(side, book, f)})
    rows.sort(key=lambda r: (-r["score"], r["formation"]))
    chosen = [r for r in rows if r["score"] > 0][:cap]
    if side == "offense":  # always keep a run / short-yardage answer
        if not any(any(catalog.is_run(p) for p in plays[r["formation"]]) for r in chosen):
            extra = next((r for r in rows if r not in chosen and any(catalog.is_run(p) for p in plays[r["formation"]])), None)
            if extra:
                chosen = chosen[: cap - 1] + [extra]
    if len(chosen) < MIN_FOCUS:  # little known about this book: add its largest formations, family-diverse
        seen = {r["family"] for r in chosen}
        for pass_ in (0, 1):
            for r in sorted(rows, key=lambda r: -r["n_plays"]):
                if len(chosen) >= MIN_FOCUS:
                    break
                if r in chosen or r["family"] in ("Goal Line", "Hail Mary", "Prevent"):
                    continue
                if pass_ == 0 and r["family"] in seen:
                    continue
                r["why"] = r["why"] or "fills the plan (largest formation in its family; no research signal yet)"
                chosen.append(r)
                seen.add(r["family"])
    return {"formations": [r["formation"] for r in chosen], "rows": rows[: max(12, cap)],
            "chosen": chosen}


def pick_audibles(side: str, book: str, formation: str, plays: list[str], named: dict[str, Any] | None = None,
                  lw: Any = None) -> list[str]:
    """4 audibles per formation: seed audibles present in this book's list, then research-named
    plays, then Aidan's best, keeping one run and one quick throw when the formation has them."""
    from cfb_coach.madden import catalog

    if side != "offense":
        return []
    seed = _load_json("seed.json")
    pool = {**seed["playbooks"]["offense_formations"], **(seed.get("delta_formations") or {})}
    out = [p for p in (pool.get(formation) or {}).get("audibles") or [] if p in plays]
    pairs = (named or {}).get("pairs") or {}
    ranked = sorted(plays, key=lambda p: -(float((pairs.get(f"{formation}::{p}") or {}).get("score", 0.0))
                                          + (lw.norm(f"play::{formation}::{p}") if lw is not None else 0.0)))
    for p in ranked:
        if len(out) >= AUDIBLES_PER_FORMATION:
            break
        if p in out:
            continue
        if float((pairs.get(f"{formation}::{p}") or {}).get("score", 0.0)) > 0 or (
            lw is not None and lw.n(f"play::{formation}::{p}") > 0 and lw.norm(f"play::{formation}::{p}") > 0):
            out.append(p)
    has_run = any(catalog.is_run(p) for p in out)
    if not has_run:
        run = next((p for p in plays if catalog.is_run(p) and p not in out), None)
        if run:
            out = (out[: AUDIBLES_PER_FORMATION - 1] + [run]) if len(out) >= AUDIBLES_PER_FORMATION else out + [run]
    if not any(re.search(r"slant|stick|mesh|spot|quick|drag|curl|hitch|smash|out\b", p, re.I) for p in out):
        q = next((p for p in plays if re.search(r"slant|stick|mesh|spot|quick|drag|curl|hitch", p, re.I) and p not in out), None)
        if q:
            out = (out[: AUDIBLES_PER_FORMATION - 1] + [q]) if len(out) >= AUDIBLES_PER_FORMATION else out + [q]
    for deep_ok in (False, True):  # fill in book order, underneath/run first
        for p in plays:
            if len(out) >= AUDIBLES_PER_FORMATION:
                break
            if p not in out and (deep_ok or not catalog.is_deep(p)):
                out.append(p)
    return out[:AUDIBLES_PER_FORMATION]


def side_named(research: dict[str, Any] | None, side: str) -> dict[str, Any]:
    """Named formation/play signals for one side (Madden research is split O/D)."""
    named = (research or {}).get("named") or {}
    if side in named and isinstance(named[side], dict):
        return named[side]
    return named


def _rank_prior(side: str, book: str) -> tuple[float, list[str]]:
    rk = ((_load_json("meta_baseline.json").get(side) or {}).get("rankings") or {})
    total, cites = 0.0, []
    for src, names in rk.items():
        for i, n in enumerate(names):
            if book.lower() in str(n).lower():
                total += 0.12 * (1.0 - i / max(1, len(names)))
                cites.append(f"{src} #{i + 1}")
                break
    return min(RANK_PRIOR_MAX, total), cites


def recommend_book(
    side: str,
    *,
    current: dict[str, Any] | None,
    research: dict[str, Any] | None,
    team: str | None,
    start_book: str | None = None,
    lw: Any = None,
    preps_since_switch: int = 99,
) -> dict[str, Any]:
    """Score every catalogued stock book for this side from this prep's research, the seed
    rankings, Aidan's team and his own results; stay unless a challenger clearly wins."""
    from cfb_coach.madden import catalog

    research = research or {}
    named = side_named(research, side)
    live = (research.get("mode") or "") == "live"
    book_hits = (research.get("books") or {}).get(side) or {}
    start = start_book or DEFAULT_STOCK[side]
    cur_name = incumbent_book_name(current)
    rows = []
    for b in catalog.book_names(side):
        plays = catalog.book_formations(side, b)
        s = {"start": START_PRIOR if b == start else 0.0,
             "stay": STAY_PRIOR if b == cur_name else 0.0,
             "team": TEAM_PRIOR if team and catalog.book_team(side, b) == team else 0.0}
        s["rank"], cites = _rank_prior(side, b)
        bh = float((book_hits.get(b) or {}).get("score", 0.0))
        s["research_book"] = RESEARCH_BOOK_W * _tanh(bh / 2.0)
        fsum = 0.0
        for f, ps in plays.items():
            nf = float(((named.get("formations") or {}).get(f) or {}).get("score", 0.0))
            fsum += 0.3 * nf + _pair_score(named, f, ps)
        s["research_formations"] = RESEARCH_FORM_W * _tanh(fsum / 4.0)
        own = 0.0
        if b == cur_name and lw is not None:
            vals = [g for g, n in (_own_formation_grade(lw, f, plays.get(f, [])) for f in (current or {}).get("formations") or {}) if n]
            own = OWN_W * (sum(vals) / len(vals)) if vals else 0.0
        s["own"] = own
        total = round(sum(s.values()), 3)
        rows.append({"book": b, "team": catalog.book_team(side, b), "score": total,
                     "parts": {k: round(v, 3) for k, v in s.items()}, "rank_cites": cites,
                     "research_docs": int((book_hits.get(b) or {}).get("docs", 0))})
    rows.sort(key=lambda r: -r["score"])
    if not rows:
        return {"book": cur_name or start, "rows": [], "reason": "no catalogued books", "switch": False}
    best = rows[0]
    incumbent = cur_name or start
    inc = next((r for r in rows if r["book"] == incumbent), best)
    reason = ""
    choice = incumbent
    if best["book"] != incumbent:
        margin = best["score"] - inc["score"]
        if not live:
            reason = (f"research favors {best['book']} ({best['score']:.2f} vs {inc['score']:.2f}) but live research "
                      "did not run this prep — staying")
        elif preps_since_switch < SWITCH_COOLDOWN_PREPS:
            reason = f"{best['book']} scores higher but the book switched {preps_since_switch} prep(s) ago (cooldown)"
        elif margin >= SWAP_MARGIN:
            choice = best["book"]
            reason = (f"this prep's research recommends {best['book']} over {incumbent} "
                      f"({best['score']:.2f} vs {inc['score']:.2f}, margin {margin:.2f} ≥ {SWAP_MARGIN})")
        else:
            reason = (f"{best['book']} is close ({best['score']:.2f} vs {inc['score']:.2f}) but not "
                      f"{SWAP_MARGIN} better — staying on {incumbent}")
    else:
        reason = f"{incumbent} is still the best-supported {side} book ({inc['score']:.2f})"
    return {"book": choice, "rows": rows, "reason": reason, "switch": choice != cur_name and cur_name is not None,
            "incumbent": incumbent}


# ---------------------------------------------------------------------------
# Record persistence
# ---------------------------------------------------------------------------

def incumbent_book_name(current: dict[str, Any] | None) -> str | None:
    """Stock book a locked record was trimmed from (custom plan) or selected (stock)."""
    if not current:
        return None
    if current.get("source_book"):
        return str(current["source_book"])
    if current.get("mode") == STOCK:
        return current.get("name")
    return None


def make_trimmed_record(
    side: str,
    source_book: str,
    *,
    rev: int = 1,
    reason: str = "",
    formations: list[str] | None = None,
    named: dict[str, Any] | None = None,
    lw: Any = None,
) -> dict[str, Any]:
    """Coach's trimmed plan, stored as mode custom.

    Plays are exactly the catalog plays of ``formations`` inside ``source_book``.
    This is the book of record prep assumes — not a separate in-game custom import.
    """
    forms = _stock_formations(side, source_book, formations)
    auds = {
        f: pick_audibles(side, source_book, f, ps, named, lw) for f, ps in forms.items()
    }
    return {
        "side": side,
        "mode": CUSTOM,
        "name": source_book,
        "source_book": source_book,
        "trimmed": True,
        "formations": forms,
        "audibles": {f: a for f, a in auds.items() if a},
        "core": list(forms),
        "rev": rev,
        "locked_ts": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }


def make_record(side: str, mode: str, name: str | None, *, rev: int = 1, reason: str = "",
                formations: list[str] | None = None, audibles: dict[str, list[str]] | None = None,
                named: dict[str, Any] | None = None, lw: Any = None) -> dict[str, Any]:
    if mode == STOCK:
        book = name or DEFAULT_STOCK[side]
        forms = _stock_formations(side, book, formations)
    else:
        book = CUSTOM_NAME[side]
        forms = _custom_formations(side, formations or custom_core(side))
    src_book = book if mode == STOCK else None
    auds = audibles if audibles is not None else {
        f: pick_audibles(side, src_book or DEFAULT_STOCK[side], f, ps, named, lw) for f, ps in forms.items()
    }
    return {
        "side": side,
        "mode": mode,
        "name": book,
        "formations": forms,
        "audibles": {f: a for f, a in auds.items() if f in forms and a},
        "core": custom_core(side) if mode == CUSTOM else list(forms),
        "rev": rev,
        "locked_ts": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }


class NoActivePlaybook(RuntimeError):
    """Live calls refused: no playbook of record has been locked by prep yet."""


def _load_state(db: Any) -> dict[str, dict[str, dict[str, Any]]]:
    state: dict[str, dict[str, dict[str, Any]]] = {"applied": {}, "pending": {}}
    if db is None:
        return state
    raw = db.get_meta(META_KEY)
    if not raw:
        return state
    try:
        data = dict(json.loads(raw))
    except ValueError:
        return state
    if "applied" in data or "pending" in data:
        for k in ("applied", "pending"):
            state[k] = {s: r for s, r in dict(data.get(k) or {}).items() if s in SIDES}
    else:  # pre-pending format: {side: record} == applied
        state["applied"] = {s: r for s, r in data.items() if s in SIDES}
    return state


def _save_state(db: Any, state: dict[str, dict[str, dict[str, Any]]]) -> None:
    db.set_meta(META_KEY, json.dumps(state))


def load_books(db: Any) -> dict[str, dict[str, Any]]:
    """Applied (locked) books — the only ones live calls may use."""
    return _load_state(db)["applied"]


def load_pending(db: Any) -> dict[str, dict[str, Any]]:
    """Custom builds/diffs proposed by prep but not yet confirmed installed."""
    return _load_state(db)["pending"]


def active_books(db: Any, sides: tuple[str, ...] = SIDES) -> dict[str, dict[str, Any]]:
    """Locked books for `sides`; raise NoActivePlaybook when any is missing."""
    books = load_books(db)
    missing = [s for s in sides if s not in books]
    if missing:
        pending = load_pending(db)
        hint = (
            " A custom book is pending — build it, then run `prep --game madden27 -o <opp> --mark-applied`."
            if any(s in pending for s in missing)
            else " Run `prep --game madden27 -o <opp>` first."
        )
        raise NoActivePlaybook(f"No {' / '.join(missing)} playbook locked yet.{hint}")
    return books


def eligible(books: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    return {side: dict(books[side]["formations"]) for side in SIDES if side in books}


# ---------------------------------------------------------------------------
# Prep decision (per side)
# ---------------------------------------------------------------------------

def _book_delta(action: str, target: str, detail: str, *, side: str, why: str,
                plays: list[str] | None = None) -> dict[str, Any]:
    return {
        "action": action,
        "kind": "playbook",
        "target": target,
        "field": side.title(),
        "before": "",
        "after": ", ".join(plays or []),
        "detail": detail,
        "why": why,
        "side": side,
        "validated_status": "meta_grounded",
    }


def _team_stock(side: str, team: str | None) -> str | None:
    if not team:
        return None
    return next((b for b, m in stock_books(side).items() if m.get("team") == team), None)


def plan_side(
    side: str,
    current: dict[str, Any] | None,
    *,
    opp: dict[str, Any],
    choice: str | None,
    team: str | None,
    research: dict[str, Any] | None = None,
    lw: Any = None,
    start_book: str | None = None,
    preps_since_switch: int = 99,
    pending: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide the book of record for one side + the formations (full, every play) to work from.

    auto (default): the coach's trimmed plan, stored as mode ``custom``. The source stock
    book is still scored (start: Buccaneers O / 49ers D, research can switch it), then
    ``select_focus`` keeps the formations to call from. That trimmed set is the custom
    playbook prep assumes. ``--o-book stock:NAME`` forces a stock book. ``--o-book custom``
    (or an already-applied editor custom with no ``source_book``) stays on the in-game
    custom-install path (checklist, then formation ADD/REMOVE)."""
    mode_req, book_req = parse_choice(side, choice)
    # An explicitly installed ML-designed custom book must not be silently
    # rewritten by ordinary heuristic-based prep. A new design must be staged
    # and confirmed through `ml offense-design`; explicit --o-book still wins.
    if (
        side == "offense" and current
        and current.get("name") == "ML Designed Offense (custom)"
        and mode_req == "auto"
    ):
        rec = dict(current)
        rec["formation_list"] = formation_list(side, rec, current)
        return {
            "side": side, "record": rec, "status": "applied",
            "change": "none", "deltas": [], "checklist": [],
            "reason": "ML-designed custom offense stays locked until an explicit redesign",
            "recommendation": None, "focus": None,
            "previous": {"mode": current.get("mode"), "name": current.get("name")},
        }
    named = side_named(research, side)
    cur_mode = (current or {}).get("mode")

    def _editor_custom(rec: dict[str, Any] | None) -> bool:
        return bool(rec) and rec.get("mode") == CUSTOM and not rec.get("source_book")

    on_editor = mode_req == CUSTOM or (mode_req == "auto" and (_editor_custom(current) or _editor_custom(pending)))
    recommendation: dict[str, Any] | None = None
    trimmed = False

    if mode_req == STOCK:
        mode, name, reason = STOCK, book_req, f"explicit --{side[0]}-book stock:{book_req}"
    elif on_editor:
        if mode_req == CUSTOM:
            mode, name, reason = CUSTOM, None, f"explicit --{side[0]}-book custom"
        else:
            mode, name, reason = CUSTOM, None, "staying on your custom book (formation ADD/REMOVE only — no rebuild churn)"
    else:
        recommendation = recommend_book(side, current=current, research=research, team=team,
                                        start_book=start_book, lw=lw, preps_since_switch=preps_since_switch)
        mode, name = CUSTOM, recommendation["book"]
        reason = (recommendation["reason"]
                  + " Saved as your custom plan: these trimmed formations, not the full stock book.")
        if team and next((True for r in recommendation["rows"] if r["book"] == name and r["team"] == team), False):
            reason += " (matches primary team)"
        trimmed = True

    rev = int((current or {}).get("rev") or 0)
    deltas: list[dict[str, Any]] = []
    checklist: list[dict[str, Any]] = []
    focus: dict[str, Any] | None = None

    if trimmed:
        cur_forms = list((current or {}).get("formations") or {}) if incumbent_book_name(current) == name else None
        focus = select_focus(side, name, named=named, lw=lw, current=cur_forms)
        new = make_trimmed_record(side, name, rev=rev, reason=reason, formations=focus["formations"],
                                  named=named, lw=lw)
        same = bool(current) and current.get("source_book") == name and current.get("mode") == CUSTOM
        if not same:
            new["rev"] = rev + 1
            change = "custom_plan"
            deltas.append(_book_delta(
                "USE CUSTOM", name,
                (f"{side.title()} custom plan → trimmed {name} "
                 f"({len(new['formations'])} formations, every play in each). "
                 "This is the book of record — nothing to build in the custom editor."),
                side=side, why=reason, plays=list(new["formations"]),
            ))
        else:
            before = set(current.get("formations") or {})
            after = set(new["formations"])
            for f in [x for x in new["formations"] if x not in before]:
                deltas.append(_book_delta(
                    "ADD", f, f"Add formation {f} to the custom plan (from {name}) → plays: {', '.join(new['formations'][f])}",
                    side=side, why="playbook suggestion added it this prep", plays=new["formations"][f],
                ))
            for f in [x for x in current.get("formations") or {} if x not in after]:
                deltas.append(_book_delta(
                    "REMOVE", f, f"Remove formation {f} from the custom plan",
                    side=side, why="playbook suggestion dropped it this prep",
                    plays=list((current.get("formations") or {}).get(f) or []),
                ))
            change = "focus" if deltas else "none"
            if deltas:
                new["rev"] = rev + 1
        mode = CUSTOM  # locks below via source_book
    elif mode == STOCK:
        cur_forms = list((current or {}).get("formations") or {}) if (current or {}).get("name") == name else None
        focus = select_focus(side, name, named=named, lw=lw, current=cur_forms)
        new = make_record(side, STOCK, name, rev=rev, reason=reason, formations=focus["formations"],
                          named=named, lw=lw)
        changed = not current or cur_mode != STOCK or current.get("name") != name
        if changed:
            new["rev"] = rev + 1
            verb = "USE STOCK" if not current or current.get("rev", 0) == 0 else "SWITCH"
            deltas.append(_book_delta(
                verb, name, f"{side.title()} playbook of record → in-game stock book \"{name}\" (no building needed)",
                side=side, why=reason, plays=list(new["formations"]),
            ))
            change = "stock_select"
        else:
            change = "focus" if set(new["formations"]) != set(current.get("formations") or {}) else "none"
            if change == "focus":
                new["rev"] = rev + 1
    else:
        new = make_record(side, CUSTOM, None, rev=rev, reason=reason, formations=desired_custom(side, opp),
                          named=named, lw=lw)
        if cur_mode != CUSTOM or (current or {}).get("source_book"):
            new["rev"] = rev + 1
            change = "switch" if current and current.get("rev", 0) > 0 else "first_custom"
            deltas.append(_book_delta(
                "CREATE CUSTOM", new["name"],
                f"Build a custom {side} playbook with these {len(new['formations'])} formations (full list below)",
                side=side, why=reason, plays=list(new["formations"]),
            ))
            for f, plays in new["formations"].items():
                checklist.append({"formation": f, "plays": plays, "books": formation_books(side, f) or DEFAULT_STOCK[side]})
        else:
            before = set(current.get("formations") or {})
            after = set(new["formations"])
            for f in [x for x in new["formations"] if x not in before]:
                deltas.append(_book_delta(
                    "ADD", f, f"Add formation {f} (from {formation_books(side, f)}) → plays: {', '.join(new['formations'][f])}",
                    side=side, why="needed vs this opponent's persona", plays=new["formations"][f],
                ))
            for f in [x for x in current.get("formations") or {} if x not in after]:
                deltas.append(_book_delta(
                    "REMOVE", f, f"Remove formation {f} (opponent-specific; not needed this week)",
                    side=side, why="keeps the custom book lean", plays=list((current.get("formations") or {}).get(f) or []),
                ))
            change = "diff" if deltas else "none"
            if deltas:
                new["rev"] = rev + 1
    new["formation_list"] = formation_list(side, new, current, focus=focus)
    locks_now = trimmed or mode == STOCK or change == "none"
    return {
        "side": side,
        "record": new,
        "status": "applied" if locks_now else "pending",
        "change": change,
        "deltas": deltas,
        "checklist": checklist,
        "reason": reason,
        "recommendation": recommendation,
        "focus": focus,
        "previous": {"mode": cur_mode, "name": (current or {}).get("name")} if current else None,
    }


def formation_list(side: str, rec: dict[str, Any], current: dict[str, Any] | None,
                   *, focus: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Rows for the minimal prep page (CFB `render_formations_min` shape)."""
    from cfb_coach.madden import catalog

    same_book = bool(current) and (
        (current.get("source_book") or current.get("name")) == (rec.get("source_book") or rec.get("name"))
    )
    prev = set((current or {}).get("formations") or {}) if same_book else set()
    why = {r["formation"]: r.get("why") or "" for r in (focus or {}).get("chosen") or []}
    rows = []
    src_name = rec.get("source_book") or (rec["name"] if rec["mode"] == STOCK else "")
    for f, ps in rec["formations"].items():
        st = "applied" if (not current or f in prev) else "new"
        if rec["mode"] == CUSTOM and not rec.get("source_book") and current and current.get("mode") == CUSTOM and f not in prev:
            st = "new"
        fam = catalog.formation_family(side, src_name, f) if src_name else ""
        source = src_name or (formation_books(side, f) or DEFAULT_STOCK[side])
        rows.append({"formation": f, "status": st, "n_plays": len(ps),
                     "source_book": source,
                     "note": "; ".join(x for x in (f"{fam} family" if fam and fam not in f.split()[0] else "", why.get(f, "")) if x)})
    if current and same_book:
        for f in current.get("formations") or {}:
            if f not in rec["formations"]:
                rows.append({"formation": f, "status": "remove", "n_plays": len(current["formations"][f]),
                             "source_book": src_name or rec.get("name"),
                             "note": "dropped from the plan this prep" if rec.get("source_book") or rec["mode"] == STOCK else "remove from the custom book"})
    return rows


def plan_books(
    db: Any,
    *,
    opp: dict[str, Any],
    team: str | None,
    offense_only: bool,
    o_book: str | None = None,
    d_book: str | None = None,
    research: dict[str, Any] | None = None,
    opponent_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    from cfb_coach.madden.franchise import DEFAULT_START_BOOK

    current = load_books(db)
    pending = load_pending(db) if db is not None else {}
    since = _preps_since_switch(db)
    lws = _learned(db, opponent_id or opp.get("_id") or "cpu")
    out = {"offense": plan_side("offense", current.get("offense"), opp=opp,
                                choice=o_book, team=team, research=research, lw=lws.get("offense"),
                                start_book=DEFAULT_START_BOOK["offense"], preps_since_switch=since.get("offense", 99),
                                pending=pending.get("offense"))}
    if offense_only:
        # CPU = offense-only: defense book untouched (report the locked/default one)
        rec = current.get("defense") or make_record("defense", STOCK, DEFAULT_STOCK["defense"], rev=0)
        rec.setdefault("formation_list", formation_list("defense", rec, None))
        out["defense"] = {"side": "defense", "record": rec, "change": "none", "deltas": [],
                          "checklist": [], "reason": "CPU = offense-only (defense book unchanged)",
                          "previous": None, "untouched": True, "recommendation": None, "focus": None}
    else:
        out["defense"] = plan_side("defense", current.get("defense"), opp=opp,
                                   choice=d_book, team=team, research=research, lw=lws.get("defense"),
                                   start_book=DEFAULT_START_BOOK["defense"], preps_since_switch=since.get("defense", 99),
                                   pending=pending.get("defense"))
    return out


def _learned(db: Any, opponent_id: str) -> dict[str, Any]:
    if db is None:
        return {}
    try:
        from cfb_coach.learning import LearnedWeights

        return {s: LearnedWeights.load(db, opponent_id, side=s) for s in SIDES}
    except Exception:  # noqa: BLE001 — never break prep
        return {}


SWITCH_KEY = "book_switch_log"


def _preps_since_switch(db: Any) -> dict[str, int]:
    if db is None:
        return {}
    try:
        log = json.loads(db.get_meta(SWITCH_KEY) or "{}")
    except ValueError:
        return {}
    return {s: int(v.get("preps_since", 99)) for s, v in log.items()}


def _note_prep(db: Any, plans: dict[str, dict[str, Any]]) -> None:
    try:
        log = json.loads(db.get_meta(SWITCH_KEY) or "{}")
    except ValueError:
        log = {}
    for side, p in plans.items():
        if p.get("untouched"):
            continue
        ent = log.setdefault(side, {"preps_since": 99})
        prev = (p.get("previous") or {}).get("name")
        if prev and prev != p["record"]["name"]:
            ent.update({"preps_since": 0, "from": prev, "to": p["record"]["name"],
                        "ts": datetime.now(timezone.utc).isoformat()})
        else:
            ent["preps_since"] = min(99, int(ent.get("preps_since", 99)) + 1)
    db.set_meta(SWITCH_KEY, json.dumps(log))


def lock_books(db: Any, plans: dict[str, dict[str, Any]], *, applied: bool = False) -> None:
    """Persist prep's book decision. Stock (nothing to build) locks now; custom
    builds/diffs stay pending until `applied=True` (prep --mark-applied)."""
    state = _load_state(db)
    for side, p in plans.items():
        if p.get("untouched"):
            continue
        rec = p["record"]
        if rec["mode"] == STOCK or rec.get("source_book") or p["change"] == "none" or applied:
            state["applied"][side] = rec
            state["pending"].pop(side, None)
        else:
            state["pending"][side] = rec
        p["status"] = "applied" if state["applied"].get(side) is rec else "pending"
    _save_state(db, state)
    _note_prep(db, plans)


def apply_pending(db: Any) -> list[str]:
    """Promote pending custom books to applied (Aidan built them). Returns sides applied."""
    state = _load_state(db)
    done = []
    for side, rec in list(state["pending"].items()):
        state["applied"][side] = rec
        del state["pending"][side]
        done.append(side)
    _save_state(db, state)
    return done


def live_book_info(db: Any, *, cpu: bool = False, profile_label: str = "") -> dict[str, Any]:
    """Live-window book panel (CFB `live_book_info` shape): locked O (+ D) book, pending builds."""
    locked = load_books(db)
    pending = load_pending(db)
    sides = ("offense",) if cpu else SIDES
    names = [f"{s[0].upper()} {locked[s]['name']} [{locked[s]['mode']}]" for s in sides if s in locked]
    off = locked.get("offense") or {}
    pend = [pending[s] for s in sides if s in pending]
    forms = dict(off.get("formations") or {})
    if not cpu and locked.get("defense"):
        forms.update({f"D · {f}": ps for f, ps in locked["defense"]["formations"].items()})
    return {
        "dynasty": profile_label or "Madden 27 Franchise",
        "context_label": "Franchise",
        "game_label": "Madden 27",
        "name": " · ".join(names),
        "callable_rev": off.get("rev") or (1 if off else None),
        "confirmed": bool(off),
        "pending_rev": max((int(p.get("rev") or 0) for p in pend), default=None) or None,
        "pending_edits": sum(len(p.get("formations") or {}) for p in pend),
        "pending_text": "\n".join(format_book(p) for p in pend),
        "formations": forms,
        "empty_text": "No Madden book locked — run prep --game madden27.",
    }


def live_apply(db: Any, rev: int | None = None) -> dict[str, Any] | None:
    done = apply_pending(db)
    if not done:
        return None
    return {"rev": max(int((load_books(db).get(s) or {}).get("rev") or 0) for s in done), "sides": done}


# ---------------------------------------------------------------------------
# Text views
# ---------------------------------------------------------------------------

def format_book(rec: dict[str, Any]) -> str:
    lines = [f"{rec['side'].title()}: {rec['name']} [{rec['mode']}] rev {rec.get('rev', 0)}"]
    if rec.get("reason"):
        lines.append(f"  ({rec['reason']})")
    auds = rec.get("audibles") or {}
    for f, plays in rec["formations"].items():
        lines.append(f"  - {f}: {', '.join(plays)}")
        if auds.get(f):
            lines.append(f"      audibles: {', '.join(auds[f])}")
    return "\n".join(lines)


def format_books(books: dict[str, dict[str, Any]], *, offense_only: bool = False) -> str:
    sides = ("offense",) if offense_only else SIDES
    return "\n".join(format_book(books[s]) for s in sides if s in books)


__all__ = [
    "CUSTOM",
    "STOCK",
    "NoActivePlaybook",
    "active_books",
    "apply_pending",
    "eligible",
    "load_pending",
    "format_book",
    "format_books",
    "lock_books",
    "parse_choice",
    "plan_books",
    "recommend_book",
    "resolve_stock_book",
    "select_focus",
]
