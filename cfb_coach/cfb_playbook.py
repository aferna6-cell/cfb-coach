"""Autonomous CFB 27 custom playbook of record (per dynasty), versioned in the DB.

v1.14: the unit is the FORMATION. In CFB 27's custom playbook editor, adding a
formation brings every play in it (from the stock book you take it from), so the
book of record is a list of formations with their source stock book, and the
callable play set is every play in those formations (the source book's list from
CFB.FAN, plus any play Aidan has actually logged in that formation — his logs are
ground truth).

Each ``prep`` the coach decides on its own which formations belong in the book.
It is heavily meta-weighted, but his results (aggregated per formation and per
play) still count:

* KEEP a formation that is working for him on a real sample, even if the meta is cold.
* CUT a formation only when the meta doesn't back it AND it is failing for him —
  and only after two preps with new snaps in between (hysteresis), unless the
  failure is overwhelming.
* DEMOTE (keep + flag) a meta formation that keeps failing for him; failing plays
  inside kept formations are demoted for the live caller (called less, not removed —
  the editor adds formations whole).
* ADD at most one meta-strong formation per prep (seed research formation priors,
  formations/plays named by this prep's live research incl. YouTube transcripts,
  Ohio State lab candidates). A formation already proposed stays at a slightly
  lower bar; swapping out an incumbent in a full book needs a clear margin.
* A meta-added formation that stays untested and loses meta support for two preps drops out.

Same contract as before: a decision that changes the book creates a PENDING
revision; the live caller keeps using the APPLIED book until Aidan confirms
(``book apply`` / ``prep --mark-applied`` / the live-window button). Every
revision is stored with timestamps and reasons (``cfb_playbook_revs``) and
``book rollback --to N`` restores any earlier revision with a cooldown.
v1.13 per-play revisions are migrated in place (``migrate_v1``).
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from typing import Any

from cfb_coach.cfb_catalog import (
    book_plays,
    book_source,
    canonical_formation,
    canonical_pair,
    formation_books,
    formations as catalog_formations,
    is_run,
    norm,
    pair_key,
)

TABLE = "cfb_playbook_revs"
STATE_KEY = "cfb_book_state:{d}"
SCHEMA_KEY = "cfb_book_schema"
SCHEMA = 2
BOOK_NAME = {"ohio_state": "OSU LAB O (custom)", "alabama": "BAMA META O (custom)"}
TEAM_BOOK = {"ohio_state": "Ohio State", "alabama": "Alabama"}

# --- Real custom-playbook limits (sources cited in the prep details) -----------
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
    {"claim": "CFB 27 Custom Playbooks (Create & Share): 4 audibles per formation; My Gameplan meter",
     "source": "ClutchPoints 'How to Create Custom Playbooks in College Football 27'",
     "url": "https://clutchpoints.com/gaming/how-to-create-custom-playbooks-in-college-football-27",
     "date": "2026-07-03", "title_era": "CFB 27"},
    {"claim": "CFB 27 custom books can pull formation blocks from any source playbook (e.g. Gun Bunch X Nasty + Dollar in one book)",
     "source": "MaddenTurf 'The Best Playbooks for College Football 27'",
     "url": "https://maddenturf.com/cfb-27-best-playbooks/", "date": "2026-07-21", "title_era": "CFB 27"},
    {"claim": "Adding a formation in the editor brings every play in it (the book unit is the formation)",
     "source": "Aidan, in-game (CFB 27)", "url": "", "date": "2026-09-27", "title_era": "CFB 27"},
    {"claim": "Title Update 1.010 (Sep 3 2026) rebuilt the Custom Playbook Designer; existing custom books must be re-saved. Formation/play reordering still absent.",
     "source": "SoVis Games CFB 27 Sep 3 title update write-up; EA Forums 'Custom Playbooks in 27'",
     "url": "https://www.sovisgames.com/posts/college-football-27-update-1-010-penalties-tuner-vision-occlusion/",
     "date": "2026-09-03", "title_era": "CFB 27"},
]
LIMIT_ASSUMPTION = (
    "No CFB 27-specific source states new numeric limits, so the CFB 25/26 limits (500 plays, 56 sets, "
    "50 plays per set) are assumed unchanged. The coach stays far below them on purpose."
)
PRACTICAL = {"max_formations": 8, "max_total_plays": 200}

# --- Play-level grading (per play, used for flags + audibles + the live caller) -
OWN_PRIOR_N = 5.0  # pseudo-snaps at 0 when grading a play (2 lucky snaps != proven)
OWN_SQUASH = 0.7
OWN_MIN_N = 6
OWN_WORKING = 0.10
OWN_FAILING = -0.15
LAB_BONUS = {"ohio_state": 0.28, "alabama": 0.10}  # play-level: lab candidates get a real trial

# --- Formation-level decision constants (prep details page) ---------------------
FORM_PRIOR_N = 8.0  # pseudo-snaps at 0 when grading a formation
FORM_MIN_N = 10  # real sample before his data can keep/cut a formation on its own
FORM_WORKING = 0.05
FORM_FAILING = -0.15
FORM_STRONG_FAIL = -0.40  # with n >= FORM_STRONG_N: cut without waiting for a 2nd strike
FORM_STRONG_N = 20
META_SUPPORT = 0.15  # meta backs a formation at/above this (formation-specific evidence)
META_COLD = 0.08
ADD_MIN = {"ohio_state": 0.25, "alabama": 0.35}
STAY_DISCOUNT = 0.08  # a formation already proposed (pending) stays at a slightly lower bar
MAX_ADDS = {"ohio_state": 1, "alabama": 1}
MAX_CUTS = 2
SWAP_MARGIN = 0.25
CUT_STRIKES = 2
META_FADE_PREPS = 2
FADE_MAX_N = 5  # "untested" = fewer snaps than this in the formation
COOLDOWN_PREPS = 3
PROMOTION_BONUS = 0.20  # Alabama: formation working in the Ohio State lab
LAB_FORM_BONUS = {"ohio_state": 0.10, "alabama": 0.0}
FORMATION_NAMED_W = 0.20
PLAYS_PART_W = 0.5
SOURCE_COVERAGE_W = 3.0
SOURCE_PRIOR_BOOK_W = 0.5
SOURCE_TEAM_W = 0.3


def constants_table() -> list[tuple[str, str]]:
    return [
        ("Book unit", "formation (adding a formation brings every play in it from its source stock book)"),
        ("Formation sample needed", f"n >= {FORM_MIN_N} snaps (grade shrinks toward 0 with {FORM_PRIOR_N:g} pseudo-snaps)"),
        ("Formation working / failing", f"own >= {FORM_WORKING:+.2f} / own <= {FORM_FAILING:+.2f}"),
        ("Meta backs / meta cold", f">= {META_SUPPORT:+.2f} / < {META_COLD:+.2f} (formation-specific evidence)"),
        ("Cut rule", f"meta doesn't back it AND failing on {CUT_STRIKES} preps with new snaps between (or n >= {FORM_STRONG_N} and own <= {FORM_STRONG_FAIL:+.2f})"),
        ("Demote rule", "meta backs it but it's failing for you -> keep + flag; failing plays inside are called less"),
        ("Add bar", f"formation meta >= {ADD_MIN['ohio_state']:.2f} (Ohio State lab) / {ADD_MIN['alabama']:.2f} (Alabama); pending stays at -{STAY_DISCOUNT:.2f}"),
        ("Adds / cuts per prep", f"<= {MAX_ADDS['ohio_state']} add, <= {MAX_CUTS} cuts"),
        ("Swap margin (book full)", f"candidate must beat the weakest incumbent by {SWAP_MARGIN:.2f}"),
        ("Meta-added formation fades", f"untested (n<{FADE_MAX_N}) and meta < {META_COLD:+.2f} for {META_FADE_PREPS} preps -> cut"),
        ("Rollback cooldown", f"{COOLDOWN_PREPS} preps before the model may redo a rolled-back change"),
        ("Formation meta", f"seed formation prior + {FORMATION_NAMED_W:g}*tanh(named/2) + {PLAYS_PART_W:g}*mean(top-3 play-specific meta) + lab bonus"),
        ("Source book", f"{SOURCE_COVERAGE_W:g}*share of your logged plays it carries + play meta + {SOURCE_PRIOR_BOOK_W:g} if research names that book + {SOURCE_TEAM_W:g} if it's your team's book; an existing choice is kept"),
        ("Practical size", f"<= {PRACTICAL['max_formations']} formations"),
        ("Play grades (flags/audibles)", f"n >= {OWN_MIN_N}: working >= {OWN_WORKING:+.2f}, failing <= {OWN_FAILING:+.2f} (failing -> demoted in live calls)"),
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================================
# Persistence (+ v1 per-play -> v2 formation migration)
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
    migrate_v1(conn)


def _schema_done(conn: Any) -> bool:
    try:
        r = conn.execute("SELECT value FROM meta WHERE key = ?", (SCHEMA_KEY,)).fetchone()
    except Exception:  # noqa: BLE001 — no meta table
        return False
    return bool(r) and str(r[0]) == str(SCHEMA)


def _set_schema(conn: Any) -> None:
    try:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (SCHEMA_KEY, str(SCHEMA)))
        conn.commit()
    except Exception:  # noqa: BLE001
        pass


def is_v1_book(book: dict[str, Any]) -> bool:
    return any(isinstance(v, list) for v in (book.get("formations") or {}).values())


def migrate_book_v1(book: dict[str, Any], dynasty: str) -> dict[str, Any]:
    """v1 {formation: [plays]} -> v2 {formation: {source_book, plays}} (pure)."""
    old = {f: list(ps) for f, ps in (book.get("formations") or {}).items()}
    forms: dict[str, dict[str, Any]] = {}
    for f, plays in old.items():
        if not isinstance(plays, list):
            forms[f] = plays
            continue
        cf = canonical_formation(f) or f
        src = team_source(cf, dynasty) or choose_source(cf, dynasty, set(plays), {})
        base = book_plays(cf, src) if src else []
        forms[cf] = {"source_book": src, "plays": base + [p for p in plays if p not in base]}
    new = {k: v for k, v in book.items() if k not in ("formations", "active8")}
    new["formations"] = forms
    new["schema"] = SCHEMA
    new["legacy_formations"] = old
    aud = book.get("audibles") or {}
    new["audibles"] = {f: [p for p in aud.get(f, []) if p in forms.get(f, {}).get("plays", [])] for f in forms if aud.get(f)}
    return new


def _migrate_edits_v1(edits: list[dict[str, Any]], book: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    forms = book.get("formations") or {}
    for e in edits:
        if e.get("op") in ("add_formation", "remove_formation"):
            f = canonical_formation(e["formation"]) or e["formation"]
            rec = forms.get(f) or {}
            out.append({"op": e["op"], "formation": f, "source_book": rec.get("source_book"),
                        "plays": rec.get("plays") or e.get("plays") or [], "reason": e.get("reason", "")})
    return out


def migrate_v1(conn: Any, *, force: bool = False) -> int:
    """Rewrite v1.13 per-play revisions to formation-level (idempotent). Returns rows changed."""
    if not force and _schema_done(conn):
        return 0
    rows = conn.execute(f"SELECT id, dynasty, rev, status, book_json, edits_json, summary FROM {TABLE} ORDER BY dynasty, rev").fetchall()
    changed = 0
    migrated: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        rid, dyn, rev, status, bj, ej, summary = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        try:
            book = json.loads(bj or "{}")
            edits = json.loads(ej or "[]")
        except ValueError:
            continue
        if not is_v1_book(book):
            migrated[(dyn, rev)] = book
            continue
        nb = migrate_book_v1(book, dyn)
        ne = _migrate_edits_v1(edits, nb)
        nb["legacy_edits"] = edits
        migrated[(dyn, rev)] = nb
        conn.execute(f"UPDATE {TABLE} SET book_json = ?, edits_json = ? WHERE id = ?", (json.dumps(nb), json.dumps(ne), rid))
        changed += 1
    # a v1 pending that only changed plays (same formation set as the current book) is superseded
    for r in rows:
        rid, dyn, rev, status = r[0], r[1], r[2], r[3]
        if status != "pending":
            continue
        cur = conn.execute(f"SELECT rev FROM {TABLE} WHERE dynasty = ? AND status = 'current' ORDER BY rev DESC LIMIT 1", (dyn,)).fetchone()
        if not cur:
            continue
        pb = migrated.get((dyn, rev)) or {}
        cb = migrated.get((dyn, cur[0])) or {}
        if pb.get("legacy_formations") is not None and set((pb.get("formations") or {})) == set((cb.get("formations") or {})):
            conn.execute(f"UPDATE {TABLE} SET status = 'discarded', summary = COALESCE(summary, '') || ? WHERE id = ?",
                         (" [v1.14: per-play edits superseded by the formation-level book]", rid))
            changed += 1
    # v1 state was keyed by (formation::play); formation-level state starts clean
    try:
        for k, v in conn.execute("SELECT key, value FROM meta WHERE key LIKE 'cfb_book_state:%'").fetchall():
            st = json.loads(v or "{}")
            if st.get("schema") == SCHEMA:
                continue
            for key in ("strikes", "fade", "cooldown", "origin", "flags"):
                st[key] = {kk: vv for kk, vv in (st.get(key) or {}).items() if "::" not in kk}
            st["schema"] = SCHEMA
            conn.execute("UPDATE meta SET value = ? WHERE key = ?", (json.dumps(st), k))
    except Exception:  # noqa: BLE001
        pass
    conn.commit()
    _set_schema(conn)
    return changed


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
# Book shape helpers
# ============================================================================

def form_sources(forms: dict[str, Any] | None) -> dict[str, str | None]:
    """{formation: source_book} from a v2 book (v1 lists -> None)."""
    out: dict[str, str | None] = {}
    for f, v in (forms or {}).items():
        if isinstance(v, dict):
            out[f] = v.get("source_book")
        elif isinstance(v, str):
            out[f] = v
        else:
            out[f] = None
    return out


def form_plays(forms: dict[str, Any] | None) -> dict[str, list[str]]:
    """{formation: [every callable play]} from a v2 (or v1) book."""
    out: dict[str, list[str]] = {}
    for f, v in (forms or {}).items():
        if isinstance(v, dict):
            out[f] = list(v.get("plays") or [])
        elif isinstance(v, list):
            out[f] = list(v)
        else:
            out[f] = book_plays(f, v if isinstance(v, str) else None)
    return {f: ps for f, ps in out.items() if ps}


def book_pairs(formations: dict[str, Any]) -> list[tuple[str, str]]:
    return [(f, p) for f, plays in form_plays(formations).items() for p in plays]


def formation_plays(formation: str, source_book: str | None, logged: set[str] | None = None) -> list[str]:
    """Every play you get with ``formation`` from ``source_book`` + plays you've logged in it."""
    base = book_plays(formation, source_book) if (source_book or canonical_formation(formation)) else []
    extra = sorted(p for p in (logged or set()) if p not in base)
    return base + extra


def build_formations(sources: dict[str, str | None], logged: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    return {f: {"source_book": b, "plays": formation_plays(f, b, logged.get(f))} for f, b in sources.items()}


def _book_label(b: str | None) -> str:
    return f"{b} playbook" if b else "any stock book"


# ============================================================================
# Callable book (live caller)
# ============================================================================

def callable_book(db: Any, dynasty: str | None = None) -> dict[str, Any] | None:
    """The book live calls are locked to: every play in the formations of the applied
    (current) revision. If nothing has been applied yet but a first build is pending,
    that pending book is used and marked unconfirmed. None when there's no book yet."""
    if db is None:
        return None
    try:
        from cfb_coach.dynasty import DEFAULT_DYNASTY, normalize_dynasty

        dyn = normalize_dynasty(dynasty or db.get_meta("dynasty_mode") or DEFAULT_DYNASTY)
        cur = current_rev(db, dyn)
        pend = pending_rev(db, dyn)
    except Exception:  # noqa: BLE001 — never break live calls
        return None
    rec = cur or pend
    if not rec:
        return None
    bk = rec["book"]
    return {"dynasty": dyn, "name": bk.get("name") or "", "rev": rec["rev"], "formations": form_plays(bk.get("formations")),
            "sources": form_sources(bk.get("formations")), "confirmed": bool(cur),
            "pending_rev": pend["rev"] if pend else None, "pending_edits": len(pend["edits"]) if pend else 0,
            "flags": bk.get("flags") or {}, "formation_flags": bk.get("formation_flags") or {},
            "audibles": bk.get("audibles") or {}}


def live_book_info(db: Any, dynasty: str | None = None) -> dict[str, Any]:
    """Compact status for the live window / terminal play mode."""
    from cfb_coach.dynasty import DEFAULT_DYNASTY, normalize_dynasty

    b = callable_book(db, dynasty)
    if not b:
        dyn = normalize_dynasty(dynasty or (db.get_meta("dynasty_mode") if db is not None else None) or DEFAULT_DYNASTY)
        return {"dynasty": dyn, "name": "", "callable_rev": None, "confirmed": False, "pending_rev": None,
                "pending_edits": 0, "formations": {}}
    pend = pending_rev(db, b["dynasty"])
    return {
        "dynasty": b["dynasty"],
        "name": b.get("name") or "",
        "callable_rev": b["rev"],
        "confirmed": b["confirmed"],
        "pending_rev": b.get("pending_rev"),
        "pending_edits": b.get("pending_edits", 0),
        "pending_text": format_edit_list(pend["edits"], first_build=not b["confirmed"],
                                         book_name=pend["book"].get("name", "")) if pend else "",
        "formations": b["formations"],
        "sources": b.get("sources") or {},
    }


def live_status_line(db: Any, dynasty: str | None = None) -> str:
    i = live_book_info(db, dynasty)
    if not i.get("callable_rev"):
        return f"Playbook ({i.get('dynasty')}): no custom book yet (run prep) — calls use the default menus."
    n = sum(len(v) for v in i["formations"].values())
    line = (f"Playbook ({i['dynasty']}): {i.get('name') or 'custom book'} — locked to rev {i['callable_rev']} "
            f"({len(i['formations'])} formations, {n} plays: {', '.join(i['formations'])})")
    if not i["confirmed"]:
        line += " — UNCONFIRMED first build: build it in CFB 27, then type `book apply`"
    elif i.get("pending_rev"):
        line += f" — {i['pending_edits']} pending formation change(s) (rev {i['pending_rev']}) not callable until you type `book apply`"
    return line


# ============================================================================
# Evidence: his results + the meta
# ============================================================================

def _session_dynasty(conn: Any) -> dict[str, str]:
    try:
        return {str(r[0]): (r[1] or "") for r in conn.execute("SELECT session_id, dynasty FROM game_sessions").fetchall()}
    except Exception:  # noqa: BLE001
        return {}


def _seed_research() -> dict[str, Any]:
    from cfb_coach.meta_align import load_research

    try:
        return load_research() or {}
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


def logged_by_formation(stats: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for s in stats.values():
        if s.get("n"):
            out.setdefault(s["formation"], set()).add(s["play"])
    return out


def formation_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """His results aggregated per formation (sum of per-play v2 retrain scores)."""
    out: dict[str, dict[str, Any]] = {}
    for s in stats.values():
        f = s["formation"]
        a = out.setdefault(f, {"formation": f, "n": 0, "succ": 0, "score_sum": 0.0, "turnovers": 0, "td": 0, "plays": 0})
        a["n"] += s["n"]
        a["succ"] += s["succ"]
        a["score_sum"] += s["score_sum"]
        a["turnovers"] += s.get("turnovers", 0)
        a["td"] += s.get("td", 0)
        a["plays"] += 1
    for a in out.values():
        a["own"] = round(math.tanh(OWN_SQUASH * a["score_sum"] / (a["n"] + FORM_PRIOR_N)), 3)
        a["succ_rate"] = round(a["succ"] / a["n"], 3) if a["n"] else 0.0
        a["conf"] = round(a["n"] / (a["n"] + FORM_MIN_N), 3)
    return out


def _prior_books(research: dict[str, Any] | None, formation: str) -> list[str]:
    for r in (research or {}).get("formation_priors") or []:
        if norm(r.get("formation")) == norm(formation):
            return list(r.get("books") or [])
    return []


def team_source(formation: str, dynasty: str) -> str | None:
    team = TEAM_BOOK.get(dynasty, "")
    for b in formation_books(formation):
        if norm(b) == norm(team):
            return b
    return None


def choose_source(formation: str, dynasty: str, logged: set[str] | None, pmeta: dict[str, dict[str, Any]],
                  *, research: dict[str, Any] | None = None, existing: str | None = None) -> str | None:
    """Which stock book to take ``formation`` from in the custom playbook editor.

    Keeps an existing choice. Otherwise: the book whose list carries the most of the
    plays you've logged in it, then play-specific meta, the books research names, and
    your own team's book."""
    books = formation_books(formation)
    if not books:
        return existing
    if existing and existing in books:
        return existing
    logged = set(logged or ())
    prior = _prior_books(research if research is not None else _seed_research(), formation)
    team = TEAM_BOOK.get(dynasty, "")

    def score(b: str) -> float:
        lst = set(book_plays(formation, b))
        cov = len(logged & lst) / len(logged) if logged else 0.0
        m = sum(max(0.0, float((pmeta.get(pair_key(formation, p)) or {}).get("specific", 0.0))) for p in lst)
        return (SOURCE_COVERAGE_W * cov + m + (SOURCE_PRIOR_BOOK_W if b in prior else 0.0)
                + (SOURCE_TEAM_W if norm(b) == norm(team) else 0.0))

    return sorted(books, key=lambda b: (-round(score(b), 4), -len(book_plays(formation, b)), b))[0]


def formation_meta_scores(scout: dict[str, Any] | None, *, research: dict[str, Any] | None = None,
                          pmeta: dict[str, dict[str, Any]] | None = None, dynasty: str = "ohio_state",
                          logged: dict[str, set[str]] | None = None,
                          existing: dict[str, str | None] | None = None) -> dict[str, dict[str, Any]]:
    """Meta score per formation (-1..1) with its best source book, reasons and citations.

    ``specific`` = evidence about this formation (seed formation prior, formation
    named in this prep's sources incl. YouTube, its plays' specific meta, lab bonus);
    ``meta`` adds generic concept chatter for its plays."""
    research = research if research is not None else _seed_research()
    pmeta = pmeta if pmeta is not None else meta_scores(scout, research=research, dynasty=dynasty)
    findings = {f.get("id"): f for f in research.get("findings") or []}
    fp = {canonical_formation(r.get("formation")) or r.get("formation"): r for r in research.get("formation_priors") or []}
    named = ((scout or {}).get("named_signals") or {}).get("formations") or {}
    named = {canonical_formation(k) or k: v for k, v in named.items()}
    lab_forms = {canonical_formation(lc.get("formation")) for lc in research.get("lab_candidates") or []}
    out: dict[str, dict[str, Any]] = {}
    for f in catalog_formations():
        src = choose_source(f, dynasty, (logged or {}).get(f), pmeta, research=research,
                            existing=(existing or {}).get(f))
        plays = formation_plays(f, src, (logged or {}).get(f))
        pm = [(float((pmeta.get(pair_key(f, p)) or {}).get("specific", 0.0)), p) for p in plays]
        top = sorted([t for t in pm if t[0] > 0], reverse=True)[:3]
        plays_part = round(PLAYS_PART_W * sum(t[0] for t in top) / 3.0, 3)
        cm = sorted([float((pmeta.get(pair_key(f, p)) or {}).get("concept", 0.0)) for p in plays], reverse=True)[:3]
        concept = round(0.5 * sum(max(0.0, c) for c in cm) / 3.0, 3)
        prior = fp.get(f) or {}
        seed = float(prior.get("score", 0.0))
        nv = named.get(f) or {}
        nm = round(FORMATION_NAMED_W * math.tanh(float(nv.get("score", 0.0)) / 2.0), 3) if nv else 0.0
        lab = LAB_FORM_BONUS.get(dynasty, 0.0) if f in lab_forms else 0.0
        reasons, cites, short = [], [], ""
        if seed:
            reasons.append(f"seed research {seed:+.2f}: {prior.get('why', '')}")
            short = prior.get("why", "")
            for rid in prior.get("refs") or []:
                fd = findings.get(rid)
                if fd:
                    cites.append({"source": fd.get("source", ""), "url": fd.get("url", ""), "date": str(fd.get("published", ""))})
        if nm:
            reasons.append(f"named in {nv.get('docs', 0)} current source(s) this prep (recency-weighted {float(nv.get('score', 0)):.2f})")
            short = short or f"named in {nv.get('docs', 0)} current sources this prep"
            cites += [{"source": s.get("label", ""), "url": s.get("url", ""), "date": s.get("date", "")} for s in (nv.get("sources") or [])[:3]]
        if plays_part:
            reasons.append(f"its plays' specific meta {plays_part:+.2f} ({', '.join(t[1] for t in top)})")
            short = short or f"meta plays: {', '.join(t[1] for t in top[:2])}"
        if lab:
            reasons.append("Ohio State lab formation (lab candidates live here)")
        specific = round(seed + nm + plays_part + lab, 3)
        seen, cc = set(), []
        for c in cites:
            key = (c.get("source"), c.get("url"))
            if key not in seen:
                seen.add(key)
                cc.append(c)
        out[f] = {"formation": f, "book": src, "n_plays": len(plays), "seed": seed, "named": nm,
                  "plays_part": plays_part, "lab": lab, "concept": concept, "specific": specific,
                  "meta": round(max(-1.0, min(1.0, specific + concept)), 3), "reasons": reasons,
                  "short": short[:90], "cites": cc[:5], "top_plays": [t[1] for t in top]}
    return out


def _fstat_txt(s: dict[str, Any] | None) -> str:
    if not s or not s.get("n"):
        return "untested by you"
    return (f"you: {s['n']} snaps, {s['succ_rate']:.0%} success, grade {s['own']:+.2f}"
            + (f", {s['turnovers']} TO" if s.get("turnovers") else ""))


_stat_txt = _fstat_txt


def _fmeta_txt(m: dict[str, Any] | None) -> str:
    if not m:
        return "meta: no signal"
    return f"meta {m['meta']:+.2f} (formation-specific {m.get('specific', m['meta']):+.2f})"


def _fvalue(m: dict[str, Any] | None, s: dict[str, Any] | None) -> float:
    meta = (m or {}).get("meta", 0.0)
    own = (s or {}).get("own", 0.0) if (s or {}).get("n") else 0.0
    return round(meta + own * (s or {}).get("conf", 0.0), 3)


def _pvalue(m: dict[str, Any] | None, s: dict[str, Any] | None) -> float:
    meta = (m or {}).get("meta", 0.0)
    own = (s or {}).get("own", 0.0) if (s or {}).get("n") else 0.0
    return round(meta + own * (s or {}).get("conf", 0.0), 3)


def seed_book_from_logs(stats: dict[str, dict[str, Any]], dynasty: str) -> list[str]:
    """Formations you've logged, most-used first (they're already in your in-game book)."""
    fs = formation_stats(stats)
    return [f for f, _ in sorted(fs.items(), key=lambda kv: (-kv[1]["n"], kv[0]))]


def same_book(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same formations from the same source books (order-insensitive; plays follow)."""
    return form_sources(a) == form_sources(b) if a and b else (not a and not b)


def diff_books(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Formation-level edits for the CFB 27 custom playbook editor."""
    bs, as_ = form_sources(before), form_sources(after)
    ap, bp = form_plays(after), form_plays(before)
    edits: list[dict[str, Any]] = []
    for f, src in as_.items():
        if f not in bs:
            edits.append({"op": "add_formation", "formation": f, "source_book": src, "plays": ap.get(f, [])})
        elif bs[f] and src and bs[f] != src:
            edits.append({"op": "change_source", "formation": f, "source_book": src, "from_book": bs[f], "plays": ap.get(f, [])})
    for f, src in bs.items():
        if f not in as_:
            edits.append({"op": "remove_formation", "formation": f, "source_book": src, "plays": bp.get(f, [])})
    return edits


def decide(
    current: dict[str, Any],
    *,
    dynasty: str,
    fstats: dict[str, dict[str, Any]],
    fmeta: dict[str, dict[str, Any]],
    state: dict[str, Any],
    snap_count: int,
    pending: dict[str, Any] | None = None,
    other_fstats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pure decision at formation granularity: current formations + evidence ->
    target {formation: source_book}, reasons, formation flags, new state."""
    st = copy.deepcopy(state)
    for k in ("strikes", "fade", "cooldown", "origin"):
        st.setdefault(k, {})
    st["prep_count"] = int(st.get("prep_count", 0)) + 1
    prep_no = st["prep_count"]
    cur_src = form_sources(current)
    target = dict(cur_src)
    pending_forms = set(form_sources(pending or {}))
    reasons: dict[str, str] = {}
    flags: dict[str, dict[str, Any]] = {}
    cooldown = {k: v for k, v in (st.get("cooldown") or {}).items() if int(v.get("until_prep", 0)) >= prep_no}
    st["cooldown"] = cooldown
    add_min = ADD_MIN.get(dynasty, 0.3)
    cuts: list[str] = []

    # 1) incumbents: keep / demote / strike / cut / fade
    for f in list(cur_src):
        s = fstats.get(f)
        m = fmeta.get(f)
        mv = (m or {}).get("specific", 0.0)
        n = (s or {}).get("n", 0)
        own = (s or {}).get("own", 0.0)
        working = n >= FORM_MIN_N and own >= FORM_WORKING
        failing = n >= FORM_MIN_N and own <= FORM_FAILING
        cd = cooldown.get(f) or {}
        if working:
            st["strikes"].pop(f, None)
            flags[f] = {"flag": "working", "why": f"working for you ({_fstat_txt(s)}); {_fmeta_txt(m)}"}
            continue
        if failing and mv >= META_SUPPORT:
            st["strikes"].pop(f, None)
            flags[f] = {"flag": "demoted", "why": f"meta backs it ({_fmeta_txt(m)}) but it keeps failing for you ({_fstat_txt(s)}) — kept; its failing plays are called less"}
            continue
        if failing:
            sk = st["strikes"].get(f) or {"n": 0, "last_snaps": -1}
            if sk.get("last_snaps") != snap_count:
                sk = {"n": int(sk.get("n", 0)) + 1, "last_snaps": snap_count}
            st["strikes"][f] = sk
            strong = n >= FORM_STRONG_N and own <= FORM_STRONG_FAIL
            if cd.get("block") == "remove":
                flags[f] = {"flag": "on_notice", "why": f"failing ({_fstat_txt(s)}) but restored by rollback — cooldown until prep #{cd.get('until_prep')}"}
            elif (sk["n"] >= CUT_STRIKES or strong) and len(cuts) < MAX_CUTS and len(target) > 1:
                cuts.append(f)
                target.pop(f, None)
                reasons[f] = (f"CUT: meta doesn't back it ({mv:+.2f}) and it keeps failing for you ({_fstat_txt(s)}; "
                              f"{'strike ' + str(sk['n']) if not strong else 'overwhelming sample'})")
            else:
                flags[f] = {"flag": "on_notice", "why": f"strike {sk['n']}/{CUT_STRIKES}: meta doesn't back it ({mv:+.2f}) and it's failing for you ({_fstat_txt(s)})"}
            continue
        st["strikes"].pop(f, None)
        origin = (state.get("origin") or {}).get(f, "")
        if origin in ("meta", "lab", "promoted") and n < FADE_MAX_N:
            if mv < META_COLD:
                fd = int(st["fade"].get(f, 0)) + 1
                st["fade"][f] = fd
                if fd >= META_FADE_PREPS and cd.get("block") != "remove" and len(cuts) < MAX_CUTS and len(target) > 1:
                    cuts.append(f)
                    target.pop(f, None)
                    reasons[f] = f"CUT: added for the meta, still untested by you, and meta support is gone ({_fmeta_txt(m)}) for {fd} preps"
                    continue
                flags[f] = {"flag": "fading", "why": f"meta support gone ({_fmeta_txt(m)}); untested — drops out after {META_FADE_PREPS} preps"}
            else:
                st["fade"].pop(f, None)
                flags[f] = {"flag": "trial", "why": f"meta trial ({_fmeta_txt(m)}); {_fstat_txt(s)}"}
        else:
            flags[f] = {"flag": "keep", "why": f"{_fstat_txt(s)}; {_fmeta_txt(m)}"}

    # 2) candidates (whole formations)
    promo = {}
    if dynasty == "alabama" and other_fstats:
        promo = {f: s for f, s in other_fstats.items() if s.get("n", 0) >= FORM_MIN_N and s.get("own", 0) >= FORM_WORKING}
    cand: list[tuple[float, str, str]] = []
    for f, m in fmeta.items():
        if f in target or f in cuts or (cooldown.get(f) or {}).get("block") == "add":
            continue
        s = fstats.get(f)
        if s and s.get("n", 0) >= FORM_MIN_N and s.get("own", 0) <= FORM_FAILING:
            continue  # he already knows it fails
        o = (other_fstats or {}).get(f)
        if o and o.get("n", 0) >= FORM_MIN_N and o.get("own", 0) <= FORM_FAILING:
            continue  # failed in the Ohio State lab
        score = m["meta"] + (PROMOTION_BONUS if f in promo else 0.0)
        bar = add_min - (STAY_DISCOUNT if f in pending_forms else 0.0)
        if score >= bar:
            why = "; ".join(m.get("reasons") or [])[:260]
            if f in promo:
                why = f"proven in your Ohio State lab ({_fstat_txt(promo[f])}); " + why
            cand.append((round(score, 3), f, why))
    cand.sort(key=lambda t: (-t[0], t[1]))  # deterministic ties
    values = {f: _fvalue(fmeta.get(f), fstats.get(f)) for f in set(fmeta) | set(fstats)}
    adds: list[str] = []
    for score, f, why in cand:
        if len(adds) >= MAX_ADDS.get(dynasty, 1):
            break
        if len(target) >= PRACTICAL["max_formations"]:
            removable = [(values.get(x, 0.0), x) for x in target
                         if (flags.get(x) or {}).get("flag") != "working" and (cooldown.get(x) or {}).get("block") != "remove"]
            if not removable or len(cuts) >= MAX_CUTS:
                continue
            wv, weakest = min(removable)
            if score < wv + SWAP_MARGIN:
                continue
            target.pop(weakest, None)
            cuts.append(weakest)
            reasons[weakest] = f"SWAP OUT for {f}: weakest formation in a full book (value {wv:+.2f} vs {score:+.2f}; margin {SWAP_MARGIN})"
        target[f] = (fmeta.get(f) or {}).get("book")
        adds.append(f)
        reasons[f] = f"ADD: meta {score:+.2f} — {why}"
    origin = dict(state.get("origin") or {})
    for f in adds:
        origin[f] = "promoted" if f in promo else ("lab" if (fmeta.get(f) or {}).get("lab") else "meta")
        flags[f] = {"flag": "trial", "why": reasons[f]}
    for f in cuts:
        origin.pop(f, None)
    st["origin"] = origin
    st["flags"] = flags
    return {
        "target": target,
        "adds": adds,
        "cuts": cuts,
        "reasons": reasons,
        "flags": flags,
        "state": st,
        "values": values,
        "candidates": [{"formation": f, "score": sc, "book": (fmeta.get(f) or {}).get("book"), "why": w} for sc, f, w in cand[:10]],
    }


def play_flags(formations: dict[str, list[str]], stats: dict[str, dict[str, Any]],
               pmeta: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-play flags for the live caller (failing plays are demoted, never removed)."""
    out: dict[str, dict[str, Any]] = {}
    for f, plays in formations.items():
        for p in plays:
            k = pair_key(f, p)
            s = stats.get(k) or {}
            n, own = s.get("n", 0), s.get("own", 0.0)
            if n >= OWN_MIN_N and own >= OWN_WORKING:
                out[k] = {"flag": "working", "why": _pstat_txt(s)}
            elif n >= OWN_MIN_N and own <= OWN_FAILING:
                sp = (pmeta.get(k) or {}).get("specific", 0.0)
                out[k] = {"flag": "demoted", "why": f"keeps failing for you ({_pstat_txt(s)}); play-specific meta {sp:+.2f} — called less"}
    return out


def _pstat_txt(s: dict[str, Any] | None) -> str:
    if not s or not s.get("n"):
        return "untested by you"
    return (f"you: n={s['n']}, {s['succ_rate']:.0%} success, grade {s['own']:+.2f}"
            + (f", {s['turnovers']} TO" if s.get("turnovers") else ""))


_CONCEPT_RE = {
    "run_first": r"zone|power|duo|counter|slam|base",
    "inside_zone": r"inside zone|zone split|hb zone",
    "duo_power": r"duo|power|counter",
    "rpo": r"rpo",
    "mesh": r"mesh",
    "spot_flat": r"spot|flat",
    "play_action": r"\bpa\b",
    "whip": r"whip",
    "dive": r"dive",
}


def research_hints(scout: dict[str, Any] | None, plays: dict[str, list[str]]) -> dict[str, float]:
    """Small tie-breakers for audibles from this prep's research: play names mentioned in
    current sources (web + YouTube) and concepts the sources push. Lets a newly added,
    untested formation get sensible audibles instead of alphabetical ones."""
    import re

    sd = scout or {}
    named = (sd.get("named_signals") or {}).get("plays") or {}
    named_n = {norm(k): float((v or {}).get("score", 0.0)) for k, v in named.items()}
    sig = {**(sd.get("concept_signals") or {})}
    for k, v in (sd.get("rz_signals") or {}).items():
        sig[k] = sig.get(k, 0) + v
    out: dict[str, float] = {}
    for f, ps in plays.items():
        for p in ps:
            h = 0.05 * math.tanh(named_n.get(norm(p), 0.0) / 2.0)
            for c, rx in _CONCEPT_RE.items():
                if sig.get(c) and re.search(rx, p, re.I):
                    h += 0.01 * min(3, int(sig[c]))
            if h:
                out[pair_key(f, p)] = round(h, 3)
    return out


def _audibles(formations: dict[str, list[str]], value: dict[str, float],
              flags: dict[str, dict[str, Any]] | None = None) -> dict[str, list[str]]:
    """4 audibles per formation (CFB 27 limit): best by value, run/pass mix, no demoted plays."""
    from cfb_coach.cfb_catalog import zone_fit

    out = {}
    for f, plays in formations.items():
        ok = [p for p in plays if ((flags or {}).get(pair_key(f, p)) or {}).get("flag") != "demoted"] or list(plays)
        ranked = sorted(ok, key=lambda p: (-value.get(pair_key(f, p), 0.0), plays.index(p)))
        # audibles are called at the line anywhere on the field: prefer plays legal in the open field
        openfit = [p for p in ranked if zone_fit(p, "open")] or ranked
        pick = openfit[:4]
        runs = [p for p in openfit if is_run(p)]
        passes = [p for p in openfit if not is_run(p)]
        if len(pick) >= 4 and runs and passes:
            if not any(is_run(p) for p in pick):
                pick[-1] = runs[0]
            elif not any(not is_run(p) for p in pick):
                pick[-1] = passes[0]
        out[f] = pick
    return out


# ============================================================================
# Text
# ============================================================================

def apply_command(dynasty: str) -> str:
    return f"PYTHONPATH=. python3 -m cfb_coach book apply --dynasty {dynasty}"


def format_edit_list(edits: list[dict[str, Any]], *, first_build: bool = False, book_name: str = "") -> str:
    """Click-to-copy text for the CFB 27 custom playbook editor (formation level)."""
    lines = []
    if first_build:
        lines.append(f"CREATE custom offense playbook \"{book_name}\" (Create & Share > Custom Playbooks), then:")
    for e in edits:
        n = len(e.get("plays") or [])
        why = f"  — {e['reason']}" if e.get("reason") else ""
        if e["op"] == "add_formation":
            lines.append(f"+ ADD FORMATION  {e['formation']}  (from the {_book_label(e.get('source_book'))}, {n} plays){why}")
        elif e["op"] == "remove_formation":
            lines.append(f"- REMOVE FORMATION  {e['formation']}  ({_book_label(e.get('source_book'))}){why}")
        elif e["op"] == "change_source":
            lines.append(f"~ RE-ADD FORMATION  {e['formation']}  from the {_book_label(e.get('source_book'))} "
                         f"(instead of {_book_label(e.get('from_book'))}, {n} plays){why}")
    return "\n".join(lines)


def format_book_text(formations: dict[str, Any], *, audibles: dict[str, list[str]] | None = None,
                     name: str = "", rev: int | None = None) -> str:
    lines = [f"{name} rev {rev}" if rev is not None else name] if name else []
    srcs = form_sources(formations)
    plays = form_plays(formations)
    for f, ps in plays.items():
        aud = (audibles or {}).get(f) or []
        lines.append(f"{f} [{_book_label(srcs.get(f))}] ({len(ps)} plays): " + ", ".join(ps))
        if aud:
            lines.append(f"   audibles: {', '.join(aud)}")
    lines.append(f"TOTAL: {len(plays)} formations, {sum(len(v) for v in plays.values())} plays")
    return "\n".join(lines)


def _edit_reason(e: dict[str, Any], reasons: dict[str, str], fstats: dict[str, Any], fmeta: dict[str, Any],
                 other: dict[str, Any] | None = None) -> str:
    f = e["formation"]
    if f in reasons:
        return reasons[f]
    o = (other or {}).get(f)
    if e["op"] == "add_formation" and o and o.get("n"):
        return f"starter: not failing in your Ohio State lab ({_fstat_txt(o).replace('you: ', '')}); {_fmeta_txt(fmeta.get(f))}"
    if e["op"] == "change_source":
        return f"take it from the {_book_label(e.get('source_book'))}"
    return f"{_fstat_txt(fstats.get(f))}; {_fmeta_txt(fmeta.get(f))}"


def _short(text: str, n: int = 90) -> str:
    t = (text or "").split(" — ", 1)[-1].split("; ")[0].strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def formation_list(base: dict[str, Any], target: dict[str, Any], *, first_build: bool,
                   reasons: dict[str, str], fmeta: dict[str, Any], fstats: dict[str, Any]) -> list[dict[str, Any]]:
    """What the prep page shows: every formation to have in the custom playbook."""
    bs, ts, tp = form_sources(base), form_sources(target), form_plays(target)
    out = []
    for f, src in ts.items():
        s = fstats.get(f) or {}
        if first_build or f not in bs:
            status = "new"
            m = fmeta.get(f) or {}
            note = "" if s.get("n") else (f"new: {m['short']}" if m.get("short") else "new")
        elif bs[f] and src and bs[f] != src:
            status, note = "changed", f"re-add from the {_book_label(src)} (was {bs[f]})"
        else:
            status, note = "applied", ""
        out.append({"formation": f, "source_book": src, "n_plays": len(tp.get(f, [])), "status": status,
                    "note": note, "snaps": s.get("n", 0)})
    for f, src in bs.items():
        if f not in ts:
            out.append({"formation": f, "source_book": src, "n_plays": len(form_plays(base).get(f, [])),
                        "status": "remove", "note": _short(reasons.get(f, "removed")), "snaps": (fstats.get(f) or {}).get("n", 0)})
    return out


# ============================================================================
# Prep entry point
# ============================================================================

def plan_book(
    db: Any,
    dynasty: str,
    scout: dict[str, Any] | None,
    *,
    persist: bool = True,
    research: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the autonomous formation-level playbook decision for this prep."""
    from cfb_coach.dynasty import normalize_dynasty

    dynasty = normalize_dynasty(dynasty)
    ensure_table(db.conn)
    research = research if research is not None else _seed_research()
    stats = own_stats(db, dynasty)
    other = own_stats(db, "ohio_state") if dynasty == "alabama" else {}
    pmeta = meta_scores(scout, research=research, dynasty=dynasty)
    logged = logged_by_formation(stats)
    fstats = formation_stats(stats)
    other_f = formation_stats(other)
    snap_count = sum(s["n"] for s in stats.values())
    state = load_state(db, dynasty)
    cur = current_rev(db, dynasty)
    pend = pending_rev(db, dynasty)
    existing = {**form_sources((pend or {}).get("book", {}).get("formations")),
                **form_sources((cur or {}).get("book", {}).get("formations"))}
    fmeta = formation_meta_scores(scout, research=research, pmeta=pmeta, dynasty=dynasty, logged=logged,
                                  existing=existing)
    seeded_now = None

    def src_for(f: str) -> str | None:
        return (fmeta.get(f) or {}).get("book") if f in fmeta else None

    def seed_src(f: str) -> str | None:
        # formations you've logged came from the book you've been playing: your team's
        # stock book when it has the formation (logged extras join it)
        return team_source(f, dynasty) or src_for(f)

    if cur is None and pend is None:
        own_forms = seed_book_from_logs(stats, dynasty)
        if own_forms:
            forms = build_formations({f: seed_src(f) for f in own_forms}, logged)
            book = _book_obj(dynasty, forms, seeded_from="logged snaps")
            n = sum(len(v["plays"]) for v in forms.values())
            summary = f"Seeded from the {len(forms)} formations you've logged in {dynasty} ({n} callable plays; already in your in-game book)"
            edits = [dict(e, reason="logged in your snaps") for e in diff_books({}, forms)]
            if persist:
                cur = _insert(db, dynasty, status="current", kind="seed", book=book, edits=edits, summary=summary,
                              parent_rev=None, applied=True)
            else:
                cur = {"rev": 1, "book": dict(book, rev=1), "edits": edits, "status": "current", "kind": "seed",
                       "created_ts": _now(), "summary": summary}
            seeded_now = cur
        else:
            carried_f = other_f or formation_stats(own_stats(db, None))
            # carry what isn't failing in the lab; a failing formation only if the meta backs it
            # (same keep/demote rule as decide) — a serious dynasty doesn't inherit lab failures
            carried = [f for f, s in sorted(carried_f.items(), key=lambda kv: (-kv[1]["n"], kv[0]))
                       if f in fmeta and (fmeta[f]["specific"] >= META_SUPPORT or not (
                           (s["n"] >= FORM_MIN_N and s["own"] <= FORM_FAILING) or (s["n"] >= 5 and s["own"] <= FORM_STRONG_FAIL)))]
            if not carried:
                carried = [f for f, m in sorted(fmeta.items(), key=lambda kv: -kv[1]["meta"]) if m["specific"] >= META_SUPPORT][:3]
            state.setdefault("origin", {})
            for f in carried:
                state["origin"][f] = "carried"
            forms = build_formations({f: src_for(f) for f in carried}, {})
            pend_book = _book_obj(dynasty, forms, seeded_from="other dynasty logs / seed research")
            edits = [dict(e, reason="starter book (formations working in your other logs / seed research)") for e in diff_books({}, forms)]
            if persist:
                pend = _insert(db, dynasty, status="pending", kind="seed", book=pend_book, edits=edits,
                               summary="Starter custom book — build it, then `book apply`", parent_rev=None)
            else:
                pend = {"rev": 1, "book": dict(pend_book, rev=1), "edits": edits, "status": "pending", "kind": "seed"}

    if seeded_now or (pend and not cur and pend.get("kind") == "seed"):
        existing = {**form_sources((pend or {}).get("book", {}).get("formations")),
                    **form_sources((cur or {}).get("book", {}).get("formations"))}
        fmeta = formation_meta_scores(scout, research=research, pmeta=pmeta, dynasty=dynasty, logged=logged,
                                      existing=existing)
    first_build = cur is None
    base_src = form_sources((cur or {}).get("book", {}).get("formations"))
    base = build_formations(base_src, logged)  # refreshed: new logged plays join their formation
    start_src = base_src if cur else form_sources((pend or {}).get("book", {}).get("formations"))
    dec = decide(
        start_src,
        dynasty=dynasty,
        fstats=fstats,
        fmeta=fmeta,
        state=state,
        snap_count=snap_count,
        pending=(pend or {}).get("book", {}).get("formations") if pend else None,
        other_fstats=other_f,
    )
    target = build_formations(dec["target"], logged)
    tplays = form_plays(target)
    pflags = play_flags(tplays, stats, pmeta)
    pvalues = {pair_key(f, p): _pvalue(pmeta.get(pair_key(f, p)), stats.get(pair_key(f, p))) for f, p in book_pairs(target)}
    hints = research_hints(scout, tplays)
    audibles = _audibles(tplays, {k: v + hints.get(k, 0.0) for k, v in pvalues.items()}, pflags)
    edits = diff_books(base if cur else {}, target)
    for e in edits:
        e["reason"] = _edit_reason(e, dec["reasons"], fstats, fmeta, other_f)
        e["cites"] = (fmeta.get(e["formation"]) or {}).get("cites", [])[:3]
        e["stats"] = _fstat_txt(fstats.get(e["formation"]))
    book_extra = {"flags": pflags, "formation_flags": dec["flags"], "audibles": audibles, "schema": SCHEMA}
    status = "no_change"
    new_pending = pend
    summary = (
        f"{len(dec['adds'])} formation add(s), {len(dec['cuts'])} cut(s): "
        + ", ".join([f"+{f}" for f in dec["adds"]] + [f"-{f}" for f in dec["cuts"]])
    ) if (dec["adds"] or dec["cuts"]) else "first build"
    # the applied book always carries fresh plays/flags/audibles (his logs are ground truth)
    if persist and cur is not None:
        cur_pl = form_plays(base)
        cur_flags = play_flags(cur_pl, stats, pmeta)
        cur_vals = {pair_key(f, p): _pvalue(pmeta.get(pair_key(f, p)), stats.get(pair_key(f, p))) for f, p in book_pairs(base)}
        cur_hints = research_hints(scout, cur_pl)
        cur_book = dict(cur["book"], formations=base, flags=cur_flags,
                        audibles=_audibles(cur_pl, {k: v + cur_hints.get(k, 0.0) for k, v in cur_vals.items()}, cur_flags),
                        formation_flags={f: v for f, v in dec["flags"].items() if f in base}, schema=SCHEMA)
        db.conn.execute(f"UPDATE {TABLE} SET book_json = ? WHERE dynasty = ? AND rev = ?",
                        (json.dumps(cur_book), dynasty, cur["rev"]))
        db.conn.commit()
        cur = get_rev(db, dynasty, cur["rev"])
    if edits:
        status = "pending"
        if pend and same_book(pend["book"].get("formations") or {}, target):
            if persist:  # same proposal as last prep — keep it (no churn)
                db.conn.execute(f"UPDATE {TABLE} SET book_json = ?, edits_json = ? WHERE dynasty = ? AND rev = ?",
                                (json.dumps(dict(pend["book"], formations=target, **book_extra)), json.dumps(edits), dynasty, pend["rev"]))
                db.conn.commit()
                new_pending = get_rev(db, dynasty, pend["rev"])
        else:
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
        if cur is not None:
            new_pending = None
    if persist:
        save_state(db, dynasty, dec["state"])
    name = BOOK_NAME.get(dynasty, dynasty)
    ftable = []
    for f in list(target) + [f for f in base if f not in target]:
        m, s = fmeta.get(f) or {}, fstats.get(f) or {}
        ftable.append({"formation": f, "source_book": form_sources(target).get(f) or base_src.get(f),
                       "n_plays": len(tplays.get(f) or form_plays(base).get(f, [])), "meta": m.get("meta", 0.0),
                       "specific": m.get("specific", 0.0), "own": s.get("own"), "n": s.get("n", 0),
                       "stats": _fstat_txt(s), "value": dec["values"].get(f, 0.0),
                       "flag": "cut" if f in dec["cuts"] else (dec["flags"].get(f) or {}).get("flag", ""),
                       "why": dec["reasons"].get(f) or (dec["flags"].get(f) or {}).get("why", ""),
                       "reasons": m.get("reasons", []), "cites": m.get("cites", [])[:3]})
    meta_table = []
    for f, p in book_pairs(target):
        k = pair_key(f, p)
        meta_table.append({"formation": f, "play": p, "meta": (pmeta.get(k) or {}).get("meta", 0.0),
                           "stats": _pstat_txt(stats.get(k)), "own": (stats.get(k) or {}).get("own"),
                           "n": (stats.get(k) or {}).get("n", 0), "flag": (pflags.get(k) or {}).get("flag", ""),
                           "why": (pflags.get(k) or {}).get("why", ""), "value": pvalues.get(k, 0.0),
                           "audible": p in (audibles.get(f) or []), "cites": (pmeta.get(k) or {}).get("cites", [])[:3]})
    has_pending = bool(status == "pending" or (first_build and new_pending))
    return {
        "dynasty": dynasty,
        "name": name,
        "status": status,  # no_change | pending
        "first_build": first_build,
        "seeded_now": bool(seeded_now),
        "seed_summary": (seeded_now or {}).get("summary", ""),
        "current": cur,
        "pending": new_pending if has_pending else None,
        "edits": edits,
        "edit_text": format_edit_list(edits, first_build=first_build, book_name=name) if edits else "",
        "apply_cmd": apply_command(dynasty) if has_pending else "",
        "target": target,
        "target_plays": tplays,
        "formation_list": formation_list(base if cur else {}, target, first_build=first_build,
                                         reasons=dec["reasons"], fmeta=fmeta, fstats=fstats),
        "book_text": format_book_text(target, audibles=audibles, name=name,
                                      rev=(new_pending or {}).get("rev") if edits else (cur or {}).get("rev")),
        "audibles": audibles,
        "flags": pflags,
        "formation_flags": dec["flags"],
        "formation_table": ftable,
        "formation_meta": {f: {k: v for k, v in m.items() if k != "cites"} for f, m in sorted(fmeta.items(), key=lambda kv: -kv[1]["meta"])[:12]},
        "candidates": dec["candidates"],
        "meta_table": sorted(meta_table, key=lambda r: -r["value"]),
        "adds": dec["adds"],
        "cuts": dec["cuts"],
        "snap_count": snap_count,
        "history": [{k: v for k, v in h.items() if k not in ("book",)}
                    | {"n_formations": len(form_plays(h["book"].get("formations"))),
                       "n_plays": sum(len(x) for x in form_plays(h["book"].get("formations")).values())}
                    for h in history(db, dynasty, 12)] if persist else [],
        "limits": LIMITS,
        "limit_sources": LIMIT_SOURCES,
        "limit_assumption": LIMIT_ASSUMPTION,
        "practical": PRACTICAL,
        "constants": constants_table(),
    }


def _book_obj(dynasty: str, formations: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"name": BOOK_NAME.get(dynasty, f"{dynasty} O (custom)"), "dynasty": dynasty, "schema": SCHEMA,
            "formations": {f: dict(v) for f, v in formations.items() if v.get("plays")}, **extra}


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

    Formations the rolled-back revisions added can't be re-added, and formations
    they cut can't be cut again, for ``COOLDOWN_PREPS`` preps."""
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
        if e["op"] == "remove_formation":
            st["cooldown"][e["formation"]] = {"block": "add", "until_prep": until}
        elif e["op"] in ("add_formation", "change_source"):
            st["cooldown"][e["formation"]] = {"block": "remove", "until_prep": until}
    save_state(db, dynasty, st)
    if pend:
        _set_status(db, dynasty, pend["rev"], "discarded")
    if cur:
        _set_status(db, dynasty, cur["rev"], "superseded")
    return _insert(db, dynasty, status="current", kind="rollback", book=dict(target["book"]), edits=edits,
                   summary=f"Rollback to rev {to_rev} ({len(edits)} formation change(s) to undo in-game)",
                   parent_rev=(cur or {}).get("rev"), applied=True)


def format_history(db: Any, dynasty: str) -> str:
    rows = history(db, dynasty, 30)
    if not rows:
        return f"No playbook revisions yet for {dynasty} — run prep."
    lines = [f"Playbook history — {dynasty} (times UTC)"]
    for h in rows:
        fp = form_plays(h["book"].get("formations"))
        lines.append(f"  rev {h['rev']:>3} [{h['status']:<10}] {h['kind']:<8} created {h['created_ts'][:16]}"
                     + (f" applied {h['applied_ts'][:16]}" if h.get("applied_ts") else "")
                     + f"  {len(fp)} formations / {sum(len(x) for x in fp.values())} plays — {h.get('summary') or ''}")
        for e in h.get("edits") or []:
            if h["kind"] != "seed" and e.get("op") in ("add_formation", "remove_formation", "change_source"):
                sign = {"add_formation": "+", "remove_formation": "-", "change_source": "~"}[e["op"]]
                lines.append(f"        {sign} {e['formation']} ({e.get('source_book') or 'any'}) — {(e.get('reason') or '')[:110]}")
    return "\n".join(lines)


__all__ = [
    "LIMITS",
    "PRACTICAL",
    "apply_pending",
    "callable_book",
    "choose_source",
    "current_rev",
    "decide",
    "diff_books",
    "format_edit_list",
    "format_history",
    "formation_meta_scores",
    "formation_stats",
    "history",
    "meta_scores",
    "migrate_v1",
    "own_stats",
    "pending_rev",
    "plan_book",
    "rollback",
]
