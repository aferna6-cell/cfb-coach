"""Autonomous CFB 27 custom playbook of record (per dynasty), versioned in the DB.

Each ``prep`` the coach decides, on its own, what is in Aidan's custom offensive
playbook: exact formations and the plays in each. It is heavily meta-weighted,
but his own results still count:

* KEEP a play that is working for him on a real sample, even if the meta is cold.
* CUT a play only when the meta doesn't support it AND his results say it is
  failing — and only after it fails on two preps with new data in between
  (hysteresis), unless the failure is overwhelming.
* DEMOTE (keep, flag, drop from the Active 8) a meta play that keeps failing for him.
* ADD meta-strong plays (seed research priors, Ohio State lab candidates, and
  formations/plays named by this prep's live research incl. YouTube transcripts),
  limited per prep; plays already proposed need a slightly lower bar to stay
  (hysteresis again), and swapping out an incumbent needs a clear margin.
* Meta-added plays that stay untested and lose meta support for two preps drop out.

Mirrors the Madden 27 playbook-of-record contract: a decision that changes the
book creates a PENDING revision with a click-to-copy edit list; the live caller
keeps using the APPLIED (current) book until Aidan confirms he made the edits
(``book apply`` / ``prep --mark-applied`` / the button in the live window).
Every revision is stored with timestamps and reasons (``cfb_playbook_revs``),
and ``book rollback --to N`` restores any earlier revision (with a cooldown so
the model doesn't immediately redo the change).

Constraints (CFB 25/26 custom playbook limits, unchanged as far as any CFB 27
source says — see ``LIMITS``): <= 500 plays, <= 56 formation sets, <= 50 plays per
set; formations may be pulled from any stock book. The coach deliberately stays
far below that (``PRACTICAL``): the editor can't reorder formations, so a small
book is faster to navigate on the sticks.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from typing import Any

from cfb_coach.cfb_catalog import (
    canonical_formation,
    canonical_pair,
    formations as catalog_formations,
    is_run,
    pair_key,
    zone_fit,
)

TABLE = "cfb_playbook_revs"
STATE_KEY = "cfb_book_state:{d}"
BOOK_NAME = {"ohio_state": "OSU LAB O (custom)", "alabama": "BAMA META O (custom)"}

# --- Real custom-playbook limits (sources cited on the prep page) --------------
LIMITS = {
    "max_plays": 500,
    "max_formation_sets": 56,
    "max_plays_per_set": 50,
    "audibles_per_formation": 4,
    "mix_formations_across_books": True,
    "reorder_formations": False,
}
LIMIT_SOURCES = [
    {"claim": "500-play cap per custom playbook (editor stops saving / plays vanish past ~500)",
     "source": "r/NCAAFBseries 'CFB 26 custom playbooks not showing certain formations or plays' + r/EASportsCFB 'Custom Playbook Update, Please!'",
     "url": "https://www.reddit.com/r/NCAAFBseries/comments/1lyvlny/cfb_26_custom_playbooks_not_showing_certain/",
     "date": "2025-07 / 2025-09", "title_era": "CFB 26"},
    {"claim": "56 formation sets max; an 'add all' set lists 50 plays max",
     "source": "r/NCAAFBseries 'Missing Plays in Custom Playbook' + Coach Dan Casey '56 Formations 500 Plays' custom book",
     "url": "https://www.reddit.com/r/NCAAFBseries/comments/1fijeaq/missing_plays_in_custom_playbook/",
     "date": "2024-09", "title_era": "CFB 25"},
    {"claim": "CFB 27 Custom Playbooks (Create & Share): 4 audibles per formation; All Plays lets you add any play; My Gameplan meter",
     "source": "ClutchPoints 'How to Create Custom Playbooks in College Football 27'",
     "url": "https://clutchpoints.com/gaming/how-to-create-custom-playbooks-in-college-football-27",
     "date": "2026-07-03", "title_era": "CFB 27"},
    {"claim": "CFB 27 custom books can pull formation blocks from any source playbook (e.g. Gun Bunch X Nasty + Dollar in one book)",
     "source": "MaddenTurf 'The Best Playbooks for College Football 27'",
     "url": "https://maddenturf.com/cfb-27-best-playbooks/", "date": "2026-07-21", "title_era": "CFB 27"},
    {"claim": "Title Update 1.010 (Sep 3 2026) rebuilt the Custom Playbook Designer; existing custom books must be re-saved. Formation/play reordering still absent.",
     "source": "SoVis Games CFB 27 Sep 3 title update write-up; EA Forums 'Custom Playbooks in 27'",
     "url": "https://www.sovisgames.com/posts/college-football-27-update-1-010-penalties-tuner-vision-occlusion/",
     "date": "2026-09-03", "title_era": "CFB 27"},
]
LIMIT_ASSUMPTION = (
    "No CFB 27-specific source states new numeric limits, so the CFB 25/26 limits (500 plays, 56 sets, "
    "50 plays per set) are assumed unchanged. The coach stays far below them on purpose."
)
PRACTICAL = {"max_formations": 8, "max_plays_per_formation": 12, "max_total_plays": 64}

# --- Decision constants (shown on the prep page) --------------------------------
OWN_PRIOR_N = 5.0  # pseudo-snaps at 0 when grading a play (2 lucky snaps != proven)
OWN_SQUASH = 0.7
OWN_MIN_N = 6  # real sample before his data can keep/cut a play on its own
OWN_WORKING = 0.10
OWN_FAILING = -0.15
OWN_STRONG_FAIL = -0.45  # with n >= OWN_STRONG_N: cut without waiting for a 2nd strike
OWN_STRONG_N = 10
META_SUPPORT = 0.10  # meta supports a play at/above this
META_COLD = 0.05  # meta "doesn't support" below this
ADD_MIN = {"ohio_state": 0.25, "alabama": 0.35}
STAY_DISCOUNT = 0.08  # a play already proposed (pending) stays with a slightly lower score
MAX_ADDS = {"ohio_state": 4, "alabama": 2}
MAX_CUTS = 3
SWAP_MARGIN = 0.25
CUT_STRIKES = 2
META_FADE_PREPS = 2
COOLDOWN_PREPS = 3
NEW_FORMATION_MIN_PLAYS = 2
PROMOTION_BONUS = 0.20  # Alabama: play working in the Ohio State lab
LAB_BONUS = {"ohio_state": 0.28, "alabama": 0.10}  # Ohio State is the lab: lab candidates get a real trial
ACTIVE_N = 8  # Aidan's max-8 rule, applied to the in-game quick set drawn from the book


def constants_table() -> list[tuple[str, str]]:
    return [
        ("Own-results sample needed (keep/cut on his data)", f"n >= {OWN_MIN_N} snaps (grades shrink toward 0 with {OWN_PRIOR_N:g} pseudo-snaps)"),
        ("Working / failing", f"own >= {OWN_WORKING:+.2f} / own <= {OWN_FAILING:+.2f}"),
        ("Meta supports / meta cold", f">= {META_SUPPORT:+.2f} / < {META_COLD:+.2f}"),
        ("Cut rule", f"meta cold AND failing on {CUT_STRIKES} preps with new snaps between (or n >= {OWN_STRONG_N} and own <= {OWN_STRONG_FAIL:+.2f})"),
        ("Demote rule", "meta supports it but it's failing for you -> keep, flag, not in Active 8"),
        ("Add bar", f"meta score >= {ADD_MIN['ohio_state']:.2f} (Ohio State lab) / {ADD_MIN['alabama']:.2f} (Alabama); pending plays stay at -{STAY_DISCOUNT:.2f}"),
        ("Adds / cuts per prep", f"<= {MAX_ADDS['ohio_state']} (OSU) / {MAX_ADDS['alabama']} (Bama) adds, <= {MAX_CUTS} cuts"),
        ("Swap margin (formation full)", f"candidate must beat the weakest incumbent by {SWAP_MARGIN:.2f}"),
        ("New formation", f"only with >= {NEW_FORMATION_MIN_PLAYS} qualifying plays"),
        ("Meta-added play fades", f"untested (n<3) and meta < {META_COLD:+.2f} for {META_FADE_PREPS} preps -> cut"),
        ("Rollback cooldown", f"{COOLDOWN_PREPS} preps before the model may redo a rolled-back change"),
        ("Practical size", f"<= {PRACTICAL['max_formations']} formations, <= {PRACTICAL['max_plays_per_formation']} plays each, <= {PRACTICAL['max_total_plays']} total"),
        ("Active set", f"{ACTIVE_N} calls max (quick set / audible priorities), drawn from the book"),
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================================
# Persistence
# ============================================================================

def ensure_table(conn: Any) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dynasty TEXT NOT NULL,
            rev INTEGER NOT NULL,
            status TEXT NOT NULL,
            kind TEXT NOT NULL,
            created_ts TEXT NOT NULL,
            applied_ts TEXT,
            parent_rev INTEGER,
            book_json TEXT NOT NULL,
            edits_json TEXT NOT NULL,
            summary TEXT,
            UNIQUE(dynasty, rev)
        )
        """
    )
    conn.commit()


def _row_to_rec(r: Any) -> dict[str, Any]:
    d = dict(r)
    d["book"] = json.loads(d.pop("book_json") or "{}")
    d["edits"] = json.loads(d.pop("edits_json") or "[]")
    return d


def _fetch(db: Any, dynasty: str, status: str) -> dict[str, Any] | None:
    ensure_table(db.conn)
    r = db.conn.execute(
        f"SELECT * FROM {TABLE} WHERE dynasty = ? AND status = ? ORDER BY rev DESC LIMIT 1", (dynasty, status)
    ).fetchone()
    return _row_to_rec(r) if r else None


def current_rev(db: Any, dynasty: str) -> dict[str, Any] | None:
    return _fetch(db, dynasty, "current")


def pending_rev(db: Any, dynasty: str) -> dict[str, Any] | None:
    return _fetch(db, dynasty, "pending")


def get_rev(db: Any, dynasty: str, rev: int) -> dict[str, Any] | None:
    ensure_table(db.conn)
    r = db.conn.execute(f"SELECT * FROM {TABLE} WHERE dynasty = ? AND rev = ?", (dynasty, int(rev))).fetchone()
    return _row_to_rec(r) if r else None


def history(db: Any, dynasty: str, limit: int = 20) -> list[dict[str, Any]]:
    ensure_table(db.conn)
    rows = db.conn.execute(
        f"SELECT * FROM {TABLE} WHERE dynasty = ? ORDER BY rev DESC LIMIT ?", (dynasty, limit)
    ).fetchall()
    return [_row_to_rec(r) for r in rows]


def _next_rev(db: Any, dynasty: str) -> int:
    r = db.conn.execute(f"SELECT MAX(rev) FROM {TABLE} WHERE dynasty = ?", (dynasty,)).fetchone()
    return int((r[0] if r else 0) or 0) + 1


def _insert(db: Any, dynasty: str, *, status: str, kind: str, book: dict[str, Any], edits: list[dict[str, Any]],
            summary: str, parent_rev: int | None, applied: bool = False) -> dict[str, Any]:
    rev = _next_rev(db, dynasty)
    ts = _now()
    book = dict(book, rev=rev, dynasty=dynasty)
    db.conn.execute(
        f"INSERT INTO {TABLE} (dynasty, rev, status, kind, created_ts, applied_ts, parent_rev, book_json, edits_json, summary) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (dynasty, rev, status, kind, ts, ts if applied else None, parent_rev, json.dumps(book), json.dumps(edits), summary),
    )
    db.conn.commit()
    return get_rev(db, dynasty, rev) or {}


def _set_status(db: Any, dynasty: str, rev: int, status: str, *, applied: bool = False) -> None:
    if applied:
        db.conn.execute(f"UPDATE {TABLE} SET status = ?, applied_ts = ? WHERE dynasty = ? AND rev = ?",
                        (status, _now(), dynasty, rev))
    else:
        db.conn.execute(f"UPDATE {TABLE} SET status = ? WHERE dynasty = ? AND rev = ?", (status, dynasty, rev))
    db.conn.commit()


def load_state(db: Any, dynasty: str) -> dict[str, Any]:
    raw = db.get_meta(STATE_KEY.format(d=dynasty)) if db is not None else None
    try:
        st = json.loads(raw) if raw else {}
    except ValueError:
        st = {}
    for k in ("strikes", "fade", "cooldown", "flags"):
        st.setdefault(k, {})
    st.setdefault("prep_count", 0)
    return st


def save_state(db: Any, dynasty: str, st: dict[str, Any]) -> None:
    db.set_meta(STATE_KEY.format(d=dynasty), json.dumps(st))


# ============================================================================
# Callable book (live caller)
# ============================================================================

def callable_book(db: Any, dynasty: str | None = None) -> dict[str, Any] | None:
    """The book live calls are locked to: the applied (current) revision.

    If nothing has been applied yet but a first build is pending, that pending
    book is used and marked unconfirmed. None when the dynasty has no book yet.
    """
    if db is None:
        return None
    try:
        from cfb_coach.dynasty import DEFAULT_DYNASTY, normalize_dynasty

        dyn = normalize_dynasty(dynasty or db.get_meta("dynasty_mode") or DEFAULT_DYNASTY)
        cur = current_rev(db, dyn)
        pend = pending_rev(db, dyn)
    except Exception:  # noqa: BLE001 — never break live calls
        return None
    if cur:
        return {"dynasty": dyn, "rev": cur["rev"], "formations": cur["book"].get("formations") or {},
                "confirmed": True, "pending_rev": pend["rev"] if pend else None,
                "pending_edits": len(pend["edits"]) if pend else 0,
                "flags": (cur["book"].get("flags") or {}), "active8": cur["book"].get("active8") or []}
    if pend:
        return {"dynasty": dyn, "rev": pend["rev"], "formations": pend["book"].get("formations") or {},
                "confirmed": False, "pending_rev": pend["rev"], "pending_edits": len(pend["edits"]),
                "flags": (pend["book"].get("flags") or {}), "active8": pend["book"].get("active8") or []}
    return None


def live_book_info(db: Any, dynasty: str | None = None) -> dict[str, Any]:
    """Compact status for the live window / terminal play mode."""
    b = callable_book(db, dynasty)
    if not b:
        return {"callable_rev": None, "confirmed": False, "pending_rev": None, "pending_edits": 0, "formations": {}}
    pend = pending_rev(db, b["dynasty"])
    return {
        "dynasty": b["dynasty"],
        "callable_rev": b["rev"],
        "confirmed": b["confirmed"],
        "pending_rev": b.get("pending_rev"),
        "pending_edits": b.get("pending_edits", 0),
        "pending_text": format_edit_list(pend["edits"], first_build=not b["confirmed"],
                                         book_name=pend["book"].get("name", "")) if pend else "",
        "formations": b["formations"],
    }


def live_status_line(db: Any, dynasty: str | None = None) -> str:
    i = live_book_info(db, dynasty)
    if not i.get("callable_rev"):
        return "Playbook: no custom book yet (run prep) — calls use the default menus."
    n = sum(len(v) for v in i["formations"].values())
    line = f"Playbook: locked to rev {i['callable_rev']} ({n} plays, {len(i['formations'])} formations)"
    if not i["confirmed"]:
        line += " — UNCONFIRMED first build: build it in CFB 27, then type `book apply`"
    elif i.get("pending_rev"):
        line += f" — {i['pending_edits']} pending edit(s) (rev {i['pending_rev']}) not callable until you type `book apply`"
    return line


def book_pairs(formations: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(f, p) for f, plays in formations.items() for p in plays]


# ============================================================================
# Evidence: his results + the meta
# ============================================================================

def _session_dynasty(conn: Any) -> dict[str, str]:
    try:
        return {str(r[0]): (r[1] or "") for r in conn.execute("SELECT session_id, dynasty FROM game_sessions").fetchall()}
    except Exception:  # noqa: BLE001
        return {}


def own_stats(db: Any, dynasty: str | None) -> dict[str, dict[str, Any]]:
    """Per (formation, play) offense grades from every logged snap in ``dynasty``
    (None = all dynasties). Uses the v2 retrain scores (leverage, turnovers,
    drive/W-L adjustments). Snaps without a session count for every dynasty."""
    from cfb_coach.learning import compute_all

    if db is None:
        return {}
    comp = compute_all(db.conn)
    sess_dyn = _session_dynasty(db.conn)
    out: dict[str, dict[str, Any]] = {}
    for e in comp["evals"]:
        if e.side != "offense" or e.play in ("?", "") or e.formation in ("?", ""):
            continue
        dyn = sess_dyn.get(str(e.session_id or ""), "")
        if dynasty and dyn and dyn != dynasty:
            continue
        f, p, verified = canonical_pair(e.formation, e.play)
        k = pair_key(f, p)
        s = out.setdefault(k, {"formation": f, "play": p, "verified": verified, "n": 0, "succ": 0, "score_sum": 0.0,
                               "turnovers": 0, "zones": {"open": 0, "rz": 0, "gl": 0}, "td": 0})
        s["n"] += 1
        s["succ"] += int(bool(e.success))
        s["score_sum"] += float(e.score)
        s["turnovers"] += int("turnover" in e.tags)
        s["td"] += int("td" in e.tags)
        s["zones"][e.zone if e.zone in s["zones"] else "open"] += 1
    for s in out.values():
        s["own"] = round(math.tanh(OWN_SQUASH * s["score_sum"] / (s["n"] + OWN_PRIOR_N)), 3)
        s["succ_rate"] = round(s["succ"] / s["n"], 3) if s["n"] else 0.0
        s["conf"] = round(s["n"] / (s["n"] + OWN_MIN_N), 3)
    return out


def _seed_research() -> dict[str, Any]:
    from cfb_coach.meta_align import load_research

    try:
        return load_research() or {}
    except Exception:  # noqa: BLE001
        return {}


def meta_scores(scout: dict[str, Any] | None, *, research: dict[str, Any] | None = None,
                dynasty: str = "ohio_state") -> dict[str, dict[str, Any]]:
    """Meta score per (formation, play) in -1..1 with reasons + citations.

    ``meta`` = everything (incl. generic concept/formation chatter); ``specific`` =
    only evidence that names this play (seed priors, lab list, named pairs). The
    keep/cut "does the meta support it" test uses ``specific`` so generic Bunch
    buzz can't shield a play nobody recommends."""
    from cfb_coach.meta_align import live_boosts

    research = research if research is not None else _seed_research()
    findings = {f.get("id"): f for f in research.get("findings") or []}
    out: dict[str, dict[str, Any]] = {}

    def rec(f: str, p: str) -> dict[str, Any] | None:
        cf, cp, ok = canonical_pair(f, p)
        if not ok:
            return None
        k = pair_key(cf, cp)
        return out.setdefault(k, {"formation": cf, "play": cp, "seed": 0.0, "lab": 0.0, "named": 0.0, "play_named": 0.0,
                                  "formation_named": 0.0, "concept": 0.0, "reasons": [], "cites": []})

    def cite(ref_ids: list[str]) -> list[dict[str, str]]:
        c = []
        for rid in ref_ids:
            fnd = findings.get(rid)
            if fnd:
                c.append({"source": fnd.get("source", ""), "url": fnd.get("url", ""), "date": str(fnd.get("published", ""))})
        return c

    for pr in research.get("priors") or []:
        r = rec(pr["formation"], pr["play"])
        if r is None:
            continue
        zones = pr.get("zones") or {}
        vals = [float(v) for v in zones.values()] or [0.0]
        r["seed"] = max(vals) if max(vals) > 0 else min(vals)
        best = max(zones, key=lambda z: zones[z]) if zones else ""
        r["reasons"].append(f"seed research {r['seed']:+.2f} ({best}): {pr.get('why', '')}")
        r["cites"] += cite(pr.get("refs") or [])
    for lc in research.get("lab_candidates") or []:
        r = rec(lc["formation"], lc["play"])
        if r is None:
            continue
        r["lab"] = LAB_BONUS.get(dynasty, 0.10)
        r["reasons"].append(f"lab candidate ({'/'.join(lc.get('zones') or [])}): {lc.get('note', '')}")
        r["cites"] += cite(lc.get("refs") or [])
    sd = scout or {}
    named = sd.get("named_signals") or {}
    for k, v in (named.get("pairs") or {}).items():
        f, p = k.split("::", 1)
        r = rec(f, p)
        if r is None:
            continue
        r["named"] = round(0.35 * math.tanh(float(v.get("score", 0.0)) / 1.5), 3)
        srcs = v.get("sources") or []
        r["reasons"].append(f"named in {v.get('docs', 0)} current source(s) (recency-weighted {v.get('score', 0):.2f})")
        r["cites"] += [{"source": s.get("label", ""), "url": s.get("url", ""), "date": s.get("date", "")} for s in srcs[:3]]
    play_named = named.get("plays") or {}
    form_named = named.get("formations") or {}
    for k, r in list(out.items()):
        pv = play_named.get(r["play"])
        if pv and not r["named"]:
            r["play_named"] = round(0.12 * math.tanh(float(pv.get("score", 0.0)) / 2.0), 3)
            r["reasons"].append(f"play name mentioned in {pv.get('docs', 0)} source(s)")
        fv = form_named.get(r["formation"])
        if fv:
            r["formation_named"] = round(0.06 * math.tanh(float(fv.get("score", 0.0)) / 2.0), 3)
    if sd.get("concept_signals") or sd.get("rz_signals"):
        for (f, p), zones in live_boosts(sd.get("concept_signals") or {}, sd.get("rz_signals") or {}).items():
            r = rec(f, p)
            if r is None:
                continue
            r["concept"] = round(max(zones.values()) if zones else 0.0, 3)
            if r["concept"] >= 0.03:
                r["reasons"].append(f"concept signals this prep {r['concept']:+.2f}")
    for r in out.values():
        r["specific"] = round(r["seed"] + r["lab"] + r["named"] + r["play_named"], 3)
        r["meta"] = round(max(-1.0, min(1.0, r["specific"] + r["formation_named"] + r["concept"])), 3)
        # dedupe cites
        seen = set()
        cc = []
        for c in r["cites"]:
            key = (c.get("source"), c.get("url"))
            if key in seen:
                continue
            seen.add(key)
            cc.append(c)
        r["cites"] = cc[:5]
    return out


def formation_meta(scout: dict[str, Any] | None) -> dict[str, float]:
    named = (scout or {}).get("named_signals") or {}
    return {f: float(v.get("score", 0.0)) for f, v in (named.get("formations") or {}).items()}


# ============================================================================
# Decision
# ============================================================================

def _book_obj(dynasty: str, formations: dict[str, list[str]], **extra: Any) -> dict[str, Any]:
    return {"name": BOOK_NAME.get(dynasty, f"{dynasty} O (custom)"), "dynasty": dynasty,
            "formations": {f: list(ps) for f, ps in formations.items() if ps}, **extra}


def _stat_txt(s: dict[str, Any] | None) -> str:
    if not s or not s.get("n"):
        return "untested by you"
    return (f"you: n={s['n']}, {s['succ_rate']:.0%} success, grade {s['own']:+.2f}"
            + (f", {s['turnovers']} TO" if s.get("turnovers") else ""))


def _meta_txt(m: dict[str, Any] | None) -> str:
    if not m:
        return "meta: no play-specific signal"
    sp = m.get("specific", m["meta"])
    return f"meta {m['meta']:+.2f} (play-specific {sp:+.2f})"


def _value(m: dict[str, Any] | None, s: dict[str, Any] | None) -> float:
    meta = (m or {}).get("meta", 0.0)
    own = (s or {}).get("own", 0.0) if (s or {}).get("n") else 0.0
    conf = (s or {}).get("conf", 0.0)
    return round(meta + own * conf, 3)


def seed_book_from_logs(stats: dict[str, dict[str, Any]], dynasty: str) -> dict[str, list[str]]:
    forms: dict[str, list[str]] = {}
    for s in sorted(stats.values(), key=lambda s: -s["n"]):
        forms.setdefault(s["formation"], [])
        if s["play"] not in forms[s["formation"]]:
            forms[s["formation"]].append(s["play"])
    return forms


def same_book(a: dict[str, list[str]], b: dict[str, list[str]]) -> bool:
    """Order-insensitive equality (the in-game editor can't reorder anyway)."""
    norm = lambda d: {f: frozenset(ps) for f, ps in d.items() if ps}  # noqa: E731
    return norm(a) == norm(b)


def diff_books(before: dict[str, list[str]], after: dict[str, list[str]]) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    for f, plays in after.items():
        if f not in before:
            edits.append({"op": "add_formation", "formation": f, "plays": list(plays)})
            for p in plays:
                edits.append({"op": "add_play", "formation": f, "play": p})
        else:
            for p in plays:
                if p not in before[f]:
                    edits.append({"op": "add_play", "formation": f, "play": p})
    for f, plays in before.items():
        if f not in after:
            edits.append({"op": "remove_formation", "formation": f, "plays": list(plays)})
            for p in plays:
                edits.append({"op": "remove_play", "formation": f, "play": p})
        else:
            for p in plays:
                if p not in after[f]:
                    edits.append({"op": "remove_play", "formation": f, "play": p})
    return edits


def _audibles(formations: dict[str, list[str]], value: dict[str, float]) -> dict[str, list[str]]:
    out = {}
    for f, plays in formations.items():
        ranked = sorted(plays, key=lambda p: -value.get(pair_key(f, p), 0.0))
        pick = ranked[:4]
        runs = [p for p in ranked if is_run(p)]
        passes = [p for p in ranked if not is_run(p)]
        if len(plays) >= 4 and runs and passes:
            if not any(is_run(p) for p in pick):
                pick[-1] = runs[0]
            elif not any(not is_run(p) for p in pick):
                pick[-1] = passes[0]
        out[f] = pick
    return out


def _active8(formations: dict[str, list[str]], value: dict[str, float], flags: dict[str, Any]) -> list[str]:
    pairs = [(f, p) for f, p in book_pairs(formations) if (flags.get(pair_key(f, p)) or {}).get("flag") not in ("demoted", "on_notice")]
    ranked = sorted(pairs, key=lambda fp: -value.get(pair_key(*fp), 0.0))
    chosen: list[tuple[str, str]] = []
    gl = [fp for fp in ranked if zone_fit(fp[1], "gl") and (not zone_fit(fp[1], "open") or is_run(fp[1]))]
    for fp in gl[:2]:
        chosen.append(fp)
    for fp in ranked:
        if len(chosen) >= ACTIVE_N:
            break
        if fp not in chosen:
            chosen.append(fp)
    return [pair_key(*fp) for fp in chosen[:ACTIVE_N]]


def decide(
    current: dict[str, list[str]],
    *,
    dynasty: str,
    stats: dict[str, dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    state: dict[str, Any],
    snap_count: int,
    pending: dict[str, list[str]] | None = None,
    other_stats: dict[str, dict[str, Any]] | None = None,
    form_meta: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Pure decision: current book + evidence -> target book, reasons, flags, new state."""
    st = copy.deepcopy(state)
    st["prep_count"] = int(st.get("prep_count", 0)) + 1
    prep_no = st["prep_count"]
    target = {f: list(ps) for f, ps in current.items()}
    pending_pairs = set(pair_key(f, p) for f, p in book_pairs(pending or {}))
    reasons: dict[str, str] = {}
    flags: dict[str, dict[str, Any]] = {}
    cooldown = {k: v for k, v in (st.get("cooldown") or {}).items() if int(v.get("until_prep", 0)) >= prep_no}
    st["cooldown"] = cooldown
    cats = catalog_formations()
    add_min = ADD_MIN.get(dynasty, 0.3)
    cuts: list[str] = []

    # 1) incumbents: keep / demote / strike / cut / fade
    for f, p in book_pairs(current):
        k = pair_key(f, p)
        s = stats.get(k)
        m = meta.get(k)
        mv = (m or {}).get("specific", (m or {}).get("meta", 0.0))
        n = (s or {}).get("n", 0)
        own = (s or {}).get("own", 0.0)
        working = n >= OWN_MIN_N and own >= OWN_WORKING
        failing = n >= OWN_MIN_N and own <= OWN_FAILING
        cd = cooldown.get(k) or {}
        if working:
            st["strikes"].pop(k, None)
            flags[k] = {"flag": "working", "why": f"working for you ({_stat_txt(s)}); {_meta_txt(m)}"}
            continue
        if failing and mv >= META_SUPPORT:
            st["strikes"].pop(k, None)
            flags[k] = {"flag": "demoted", "why": f"meta likes it ({_meta_txt(m)}) but it keeps failing for you ({_stat_txt(s)}) — kept as a changeup, out of the Active 8"}
            continue
        if failing:
            sk = st["strikes"].get(k) or {"n": 0, "last_snaps": -1}
            if sk.get("last_snaps") != snap_count:
                sk = {"n": int(sk.get("n", 0)) + 1, "last_snaps": snap_count}
            st["strikes"][k] = sk
            strong = n >= OWN_STRONG_N and own <= OWN_STRONG_FAIL
            if cd.get("block") == "remove":
                flags[k] = {"flag": "on_notice", "why": f"failing ({_stat_txt(s)}) but restored by rollback — cooldown until prep #{cd.get('until_prep')}"}
            elif (sk["n"] >= CUT_STRIKES or strong) and len(cuts) < MAX_CUTS:
                cuts.append(k)
                target[f] = [x for x in target[f] if x != p]
                reasons[k] = (f"CUT: no current source specifically backs it (play-specific meta {mv:+.2f}) and it keeps failing for you ({_stat_txt(s)}; "
                              f"{'strike ' + str(sk['n']) if not strong else 'overwhelming sample'})")
            else:
                flags[k] = {"flag": "on_notice", "why": f"strike {sk['n']}/{CUT_STRIKES}: no current source specifically backs it (play-specific meta {mv:+.2f}) and it's failing for you ({_stat_txt(s)}) — cut at the next prep with new snaps if it keeps failing"}
            continue
        st["strikes"].pop(k, None)
        # meta-added, still untested, meta support gone -> fade out
        origin = (state.get("origin") or {}).get(k, "")
        if origin in ("meta", "lab", "promoted") and n < 3:
            if mv < META_COLD:
                fd = int(st["fade"].get(k, 0)) + 1
                st["fade"][k] = fd
                if fd >= META_FADE_PREPS and cd.get("block") != "remove" and len(cuts) < MAX_CUTS:
                    cuts.append(k)
                    target[f] = [x for x in target[f] if x != p]
                    reasons[k] = f"CUT: added for the meta, still untested by you, and meta support is gone ({_meta_txt(m)}) for {fd} preps"
                    continue
                flags[k] = {"flag": "fading", "why": f"meta support gone ({_meta_txt(m)}); untested — drops out after {META_FADE_PREPS} preps"}
            else:
                st["fade"].pop(k, None)
                flags[k] = {"flag": "trial", "why": f"meta trial ({_meta_txt(m)}); {_stat_txt(s)}"}
        else:
            flags[k] = {"flag": "keep", "why": f"{_stat_txt(s)}; {_meta_txt(m)} — not enough evidence to cut"}
    target = {f: ps for f, ps in target.items() if ps}

    # 2) candidates
    promo = {}
    if dynasty == "alabama" and other_stats:
        for k, s in other_stats.items():
            if s.get("n", 0) >= OWN_MIN_N and s.get("own", 0) >= OWN_WORKING and s.get("verified"):
                promo[k] = s
    cand: list[tuple[float, str, str, str]] = []
    keys = set(meta) | set(promo)
    for k in keys:
        f, p = k.split("::", 1)
        if f not in cats or p not in cats[f]:
            continue
        if p in target.get(f, []) or k in cuts:
            continue
        if (cooldown.get(k) or {}).get("block") == "add":
            continue
        m = meta.get(k) or {"meta": 0.0, "reasons": [], "cites": []}
        score = m["meta"] + (PROMOTION_BONUS if k in promo else 0.0)
        bar = add_min - (STAY_DISCOUNT if k in pending_pairs else 0.0)
        s = stats.get(k)
        if s and s.get("n", 0) >= OWN_MIN_N and s.get("own", 0) <= OWN_FAILING:
            continue  # he already knows it fails
        o = (other_stats or {}).get(k)
        if o and o.get("n", 0) >= OWN_MIN_N and o.get("own", 0) <= OWN_FAILING:
            continue  # failed in the Ohio State lab — the serious dynasty doesn't retry it
        if score >= bar:
            why = "; ".join(m.get("reasons") or [])[:260]
            if k in promo:
                why = f"proven in your Ohio State lab ({_stat_txt(promo[k])}); " + why
            cand.append((round(score, 3), f, p, why))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))  # deterministic ties (no churn from hash order)
    values = {k: _value(meta.get(k), stats.get(k)) for k in set(meta) | set(stats)}

    adds: list[str] = []
    max_adds = MAX_ADDS.get(dynasty, 2)
    # group candidates for NEW formations: need >= NEW_FORMATION_MIN_PLAYS qualifying plays
    by_form: dict[str, list[tuple[float, str, str, str]]] = {}
    for c in cand:
        by_form.setdefault(c[1], []).append(c)
    total = sum(len(v) for v in target.values())
    for score, f, p, why in cand:
        if len(adds) >= max_adds:
            break
        k = pair_key(f, p)
        if total >= PRACTICAL["max_total_plays"]:
            break
        if f not in target:
            if len(target) >= PRACTICAL["max_formations"]:
                continue
            group = by_form.get(f) or []
            if len(group) < NEW_FORMATION_MIN_PLAYS or len(adds) + NEW_FORMATION_MIN_PLAYS > max_adds:
                continue
            target[f] = []
            for sc2, f2, p2, why2 in group[: max(NEW_FORMATION_MIN_PLAYS, 0)]:
                target[f].append(p2)
                k2 = pair_key(f2, p2)
                adds.append(k2)
                reasons[k2] = f"ADD (new formation {f}): meta {sc2:+.2f} — {why2}"
                total += 1
            continue
        if k in adds:
            continue
        if len(target[f]) >= PRACTICAL["max_plays_per_formation"]:
            removable = [
                (values.get(pair_key(f, x), 0.0), x) for x in target[f]
                if (flags.get(pair_key(f, x)) or {}).get("flag") not in ("working",)
                and (cooldown.get(pair_key(f, x)) or {}).get("block") != "remove"
            ]
            if not removable:
                continue
            wv, weakest = min(removable)
            if score < wv + SWAP_MARGIN or len(cuts) >= MAX_CUTS:
                continue
            target[f] = [x for x in target[f] if x != weakest]
            wk = pair_key(f, weakest)
            cuts.append(wk)
            reasons[wk] = f"SWAP OUT for {p}: weakest in a full formation (value {wv:+.2f} vs {score:+.2f}; margin {SWAP_MARGIN})"
        target[f].append(p)
        adds.append(k)
        reasons[k] = f"ADD: meta {score:+.2f} — {why}"
        total += 1

    # origins for fade logic
    origin = dict(state.get("origin") or {})
    for k in adds:
        origin[k] = "promoted" if k in promo else ("lab" if (meta.get(k) or {}).get("lab") else "meta")
    for k in cuts:
        origin.pop(k, None)
    st["origin"] = origin
    for k in adds:
        flags[k] = {"flag": "trial", "why": reasons[k]}
    values.update({k: _value(meta.get(k), stats.get(k)) for k in adds})
    st["flags"] = flags
    return {
        "target": target,
        "adds": adds,
        "cuts": cuts,
        "reasons": reasons,
        "flags": flags,
        "state": st,
        "values": values,
        "candidates": [{"formation": f, "play": p, "score": sc, "why": w} for sc, f, p, w in cand[:15]],
    }


# ============================================================================
# Prep entry point
# ============================================================================

def _edit_reason(e: dict[str, Any], reasons: dict[str, str], stats: dict[str, Any], meta: dict[str, Any],
                 other: dict[str, Any] | None = None) -> str:
    if e["op"] in ("add_play", "remove_play"):
        k = pair_key(e["formation"], e["play"])
        if k in reasons:
            return reasons[k]
        o = (other or {}).get(k)
        if o and e["op"] == "add_play":
            return f"starter: carried from your Ohio State lab ({_stat_txt(o).replace('you: ', '')}); {_meta_txt(meta.get(k))}"
        return f"{_stat_txt(stats.get(k))}; {_meta_txt(meta.get(k))}"
    if e["op"] == "add_formation":
        return f"new formation ({len(e.get('plays') or [])} play(s) below)"
    return "every play in it was cut"


def format_edit_list(edits: list[dict[str, Any]], *, first_build: bool = False, book_name: str = "") -> str:
    """Click-to-copy text for the CFB 27 custom playbook editor."""
    lines = []
    if first_build:
        lines.append(f"CREATE custom offense playbook \"{book_name}\" (Create & Share > Custom Playbooks), then:")
    forms_add = [e for e in edits if e["op"] == "add_formation"]
    forms_rm = [e for e in edits if e["op"] == "remove_formation"]
    rm_forms = {e["formation"] for e in forms_rm}
    add_forms = {e["formation"] for e in forms_add}
    for e in forms_add:
        lines.append(f"+ ADD FORMATION  {e['formation']}  — {e.get('reason', '')}")
    for e in forms_rm:
        lines.append(f"- REMOVE FORMATION  {e['formation']}  — {e.get('reason', '')}")
    by_f: dict[str, list[dict[str, Any]]] = {}
    for e in edits:
        if e["op"] in ("add_play", "remove_play") and e["formation"] not in rm_forms:
            by_f.setdefault(e["formation"], []).append(e)
    for f, es in by_f.items():
        lines.append(f"[{f}]" + ("  (new)" if f in add_forms else ""))
        for e in es:
            sign = "+ ADD   " if e["op"] == "add_play" else "- REMOVE"
            lines.append(f"  {sign} {e['play']}  — {e.get('reason', '')}")
    return "\n".join(lines)


def format_book_text(formations: dict[str, list[str]], *, audibles: dict[str, list[str]] | None = None,
                     name: str = "", rev: int | None = None) -> str:
    lines = [f"{name} rev {rev}" if rev is not None else name] if name else []
    for f, plays in formations.items():
        aud = (audibles or {}).get(f) or []
        lines.append(f"{f} ({len(plays)}): " + ", ".join(plays))
        if aud:
            lines.append(f"   audibles: {', '.join(aud)}")
    lines.append(f"TOTAL: {sum(len(v) for v in formations.values())} plays in {len(formations)} formations")
    return "\n".join(lines)


def plan_book(
    db: Any,
    dynasty: str,
    scout: dict[str, Any] | None,
    *,
    persist: bool = True,
    research: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the autonomous playbook decision for this prep. Never raises into prep."""
    from cfb_coach.dynasty import normalize_dynasty

    dynasty = normalize_dynasty(dynasty)
    ensure_table(db.conn)
    stats = own_stats(db, dynasty)
    other = own_stats(db, "ohio_state") if dynasty == "alabama" else {}
    meta = meta_scores(scout, research=research, dynasty=dynasty)
    fmeta = formation_meta(scout)
    snap_count = sum(s["n"] for s in stats.values())
    state = load_state(db, dynasty)
    cur = current_rev(db, dynasty)
    pend = pending_rev(db, dynasty)
    seeded_now = None

    if cur is None and pend is None:
        own_forms = seed_book_from_logs(stats, dynasty)
        if own_forms:
            book = _book_obj(dynasty, own_forms, seeded_from="logged snaps")
            summary = f"Seeded from the {sum(len(v) for v in own_forms.values())} plays you've actually logged in {dynasty} (already in your in-game book)"
            edits = [dict(e, reason="logged in your snaps") for e in diff_books({}, own_forms)]
            if persist:
                cur = _insert(db, dynasty, status="current", kind="seed", book=book, edits=edits, summary=summary,
                              parent_rev=None, applied=True)
                seeded_now = cur
            else:
                cur = {"rev": 1, "book": dict(book, rev=1), "edits": edits, "status": "current", "kind": "seed",
                       "created_ts": _now(), "summary": summary}
                seeded_now = cur
        else:
            carried = other or own_stats(db, None)
            # carry only what isn't failing (a serious dynasty doesn't inherit lab failures)
            carried = {k: v for k, v in carried.items()
                       if v.get("verified") and not (v["n"] >= OWN_MIN_N and v["own"] <= OWN_FAILING)
                       and not (v["n"] >= 3 and v["own"] <= OWN_STRONG_FAIL)}
            src_forms = seed_book_from_logs(carried, dynasty)
            if not src_forms:
                src_forms = {}
                for k, m in meta.items():
                    if m["meta"] >= META_SUPPORT:
                        src_forms.setdefault(m["formation"], []).append(m["play"])
            state.setdefault("origin", {})
            for f, p in book_pairs(src_forms):
                state["origin"][pair_key(f, p)] = "carried"
            pend_book = _book_obj(dynasty, src_forms, seeded_from="other dynasty logs / seed research")
            if persist:
                pend = _insert(db, dynasty, status="pending", kind="seed", book=pend_book,
                               edits=[dict(e, reason="starter book (carried from your other logs / seed research)") for e in diff_books({}, src_forms)],
                               summary="Starter custom book — build it, then `book apply`", parent_rev=None)
            else:
                pend = {"rev": 1, "book": dict(pend_book, rev=1), "edits": [], "status": "pending", "kind": "seed"}

    base = (cur or {}).get("book", {}).get("formations") or {}
    first_build = cur is None
    start_from = base if cur else ((pend or {}).get("book", {}).get("formations") or {})
    dec = decide(
        start_from,
        dynasty=dynasty,
        stats=stats,
        meta=meta,
        state=state,
        snap_count=snap_count,
        pending=(pend or {}).get("book", {}).get("formations") if pend else None,
        other_stats=other,
        form_meta=fmeta,
    )
    target = dec["target"]
    audibles = _audibles(target, dec["values"])
    active8 = _active8(target, dec["values"], dec["flags"])
    edits = diff_books(base, target)
    for e in edits:
        e["reason"] = _edit_reason(e, dec["reasons"], stats, meta, other)
        k = pair_key(e["formation"], e.get("play", "")) if e.get("play") else ""
        m = meta.get(k) if k else None
        e["cites"] = (m or {}).get("cites", [])[:3]
        e["stats"] = _stat_txt(stats.get(k)) if k else ""
    changed = bool(edits)
    status = "no_change"
    new_pending = pend
    book_extra = {"flags": dec["flags"], "audibles": audibles, "active8": active8}
    summary = (
        f"{len(dec['adds'])} add(s), {len(dec['cuts'])} cut(s): "
        + ", ".join([f"+{k.split('::')[1]}" for k in dec["adds"]] + [f"-{k.split('::')[1]}" for k in dec["cuts"]])
    ) if (dec["adds"] or dec["cuts"]) else "first build"
    if changed:
        if pend and same_book(pend["book"].get("formations") or {}, target):
            status = "pending"  # same proposal as last prep — keep it (no churn)
            if persist:
                db.conn.execute(f"UPDATE {TABLE} SET book_json = ?, edits_json = ? WHERE dynasty = ? AND rev = ?",
                                (json.dumps(dict(pend["book"], **book_extra)), json.dumps(edits), dynasty, pend["rev"]))
                db.conn.commit()
                new_pending = get_rev(db, dynasty, pend["rev"])
        else:
            status = "pending"
            if persist:
                if pend:
                    _set_status(db, dynasty, pend["rev"], "discarded")
                new_pending = _insert(db, dynasty, status="pending", kind="prep", book=_book_obj(dynasty, target, **book_extra),
                                      edits=edits, summary=summary, parent_rev=(cur or {}).get("rev"))
            else:
                new_pending = {"rev": ((pend or cur or {}).get("rev") or 0) + 1, "book": _book_obj(dynasty, target, **book_extra),
                               "edits": edits, "status": "pending", "summary": summary}
    else:
        if pend and persist and cur is not None:
            _set_status(db, dynasty, pend["rev"], "discarded")  # no longer recommended
            new_pending = None
        elif cur is not None:
            new_pending = None
        if persist and cur is not None:
            db.conn.execute(f"UPDATE {TABLE} SET book_json = ? WHERE dynasty = ? AND rev = ?",
                            (json.dumps(dict(cur["book"], **book_extra)), dynasty, cur["rev"]))
            db.conn.commit()
            cur = get_rev(db, dynasty, cur["rev"])
    if persist:
        save_state(db, dynasty, dec["state"])
    meta_table = []
    for f, p in book_pairs(target):
        k = pair_key(f, p)
        meta_table.append({"formation": f, "play": p, "meta": (meta.get(k) or {}).get("meta", 0.0),
                           "stats": _stat_txt(stats.get(k)), "own": (stats.get(k) or {}).get("own"),
                           "n": (stats.get(k) or {}).get("n", 0), "flag": (dec["flags"].get(k) or {}).get("flag", ""),
                           "why": (dec["flags"].get(k) or {}).get("why", ""), "value": dec["values"].get(k, 0.0),
                           "cites": (meta.get(k) or {}).get("cites", [])[:3]})
    name = BOOK_NAME.get(dynasty, dynasty)
    return {
        "dynasty": dynasty,
        "name": name,
        "status": status,  # no_change | pending
        "first_build": first_build,
        "seeded_now": bool(seeded_now),
        "seed_summary": (seeded_now or {}).get("summary", ""),
        "current": cur,
        "pending": new_pending if status == "pending" or (first_build and new_pending) else None,
        "edits": edits,
        "edit_text": format_edit_list(edits, first_build=first_build, book_name=name) if edits else "",
        "target": target,
        "book_text": format_book_text(target, audibles=audibles, name=name,
                                      rev=(new_pending or {}).get("rev") if changed else (cur or {}).get("rev")),
        "audibles": audibles,
        "active8": active8,
        "flags": dec["flags"],
        "candidates": dec["candidates"],
        "meta_table": sorted(meta_table, key=lambda r: -r["value"]),
        "adds": dec["adds"],
        "cuts": dec["cuts"],
        "snap_count": snap_count,
        "history": [{k: v for k, v in h.items() if k not in ("book",)} | {"n_plays": sum(len(x) for x in (h["book"].get("formations") or {}).values())}
                    for h in history(db, dynasty, 12)] if persist else [],
        "limits": LIMITS,
        "limit_sources": LIMIT_SOURCES,
        "limit_assumption": LIMIT_ASSUMPTION,
        "practical": PRACTICAL,
        "constants": constants_table(),
    }


# ============================================================================
# Apply / rollback
# ============================================================================

def apply_pending(db: Any, dynasty: str, rev: int | None = None) -> dict[str, Any] | None:
    """Aidan made the edits in-game: pending -> current (old current superseded)."""
    from cfb_coach.dynasty import normalize_dynasty

    dynasty = normalize_dynasty(dynasty)
    pend = pending_rev(db, dynasty)
    if not pend or (rev is not None and int(rev) != int(pend["rev"])):
        return None
    cur = current_rev(db, dynasty)
    if cur:
        _set_status(db, dynasty, cur["rev"], "superseded")
    _set_status(db, dynasty, pend["rev"], "current", applied=True)
    return get_rev(db, dynasty, pend["rev"])


def rollback(db: Any, dynasty: str, to_rev: int) -> dict[str, Any]:
    """Restore revision ``to_rev`` as a NEW current revision (history is append-only).

    Plays the rolled-back revisions added can't be re-added, and plays they cut
    can't be cut again, for ``COOLDOWN_PREPS`` preps.
    """
    from cfb_coach.dynasty import normalize_dynasty

    dynasty = normalize_dynasty(dynasty)
    target = get_rev(db, dynasty, to_rev)
    if not target:
        raise ValueError(f"no revision {to_rev} for {dynasty}")
    cur = current_rev(db, dynasty)
    pend = pending_rev(db, dynasty)
    before = (cur or {}).get("book", {}).get("formations") or {}
    after = target["book"].get("formations") or {}
    edits = diff_books(before, after)
    for e in edits:
        e["reason"] = f"rollback to rev {to_rev}"
    st = load_state(db, dynasty)
    until = int(st.get("prep_count", 0)) + COOLDOWN_PREPS
    for e in edits:
        if e["op"] == "remove_play":
            st["cooldown"][pair_key(e["formation"], e["play"])] = {"block": "add", "until_prep": until}
        elif e["op"] == "add_play":
            st["cooldown"][pair_key(e["formation"], e["play"])] = {"block": "remove", "until_prep": until}
    save_state(db, dynasty, st)
    if pend:
        _set_status(db, dynasty, pend["rev"], "discarded")
    if cur:
        _set_status(db, dynasty, cur["rev"], "superseded")
    return _insert(db, dynasty, status="current", kind="rollback", book=dict(target["book"]), edits=edits,
                   summary=f"Rollback to rev {to_rev} ({len(edits)} edit(s) to undo in-game)", parent_rev=(cur or {}).get("rev"),
                   applied=True)


def format_history(db: Any, dynasty: str) -> str:
    rows = history(db, dynasty, 30)
    if not rows:
        return f"No playbook revisions yet for {dynasty} — run prep."
    lines = [f"Playbook history — {dynasty}"]
    for h in rows:
        n = sum(len(x) for x in (h["book"].get("formations") or {}).values())
        lines.append(f"  rev {h['rev']:>3} [{h['status']:<10}] {h['kind']:<8} created {h['created_ts'][:16]}"
                     + (f" applied {h['applied_ts'][:16]}" if h.get("applied_ts") else "")
                     + f"  {n} plays — {h.get('summary') or ''}")
    return "\n".join(lines)


__all__ = [
    "LIMITS",
    "PRACTICAL",
    "apply_pending",
    "callable_book",
    "current_rev",
    "decide",
    "diff_books",
    "format_edit_list",
    "format_history",
    "history",
    "meta_scores",
    "own_stats",
    "pending_rev",
    "plan_book",
    "rollback",
]
