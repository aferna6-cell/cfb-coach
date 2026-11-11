"""Madden 27 online league: per-game opponent NFL team tags, stock-book scout, inventory.

Aidan plays this league as the Kansas City Chiefs (``franchise.online_team``).
It is separate from the Detroit Lions Franchise (``primary``) and never writes
``primary_team``.

Every online game must name the opponent's NFL team (``--opponent-team``). The
team is stored on the session row and copied onto every snap, together with
``league = "online"`` and ``user_team``. ``--opponent`` stays the person
(gamertag / cold-start human); one person may play different teams in
different games, so learning by team reads the session tags, not the person.

The scout is a STOCK-BOOK prior from Huddle's team playbooks
(``playbooks_full.json``). It is not learned user tendency. Once games exist
for a team, the logged summary (games, snaps, coverage mix, success) is shown
next to it, labelled as logged data.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from cfb_coach.madden.franchise import (
    DEFAULT_ONLINE_TEAM,
    ONLINE,
    load_config,
    nfl_teams,
    resolve_nfl_team,
)

LEAGUE_ID = ONLINE
STOCK_SCOUT_LABEL = "stock-book scout (Huddle team playbook; NOT learned user tendencies)"
LOGGED_LABEL = "logged online-league games vs this team (actual users)"


def user_team(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg if cfg is not None else load_config()
    return str(cfg.get("online_team") or DEFAULT_ONLINE_TEAM)


def opponent_teams(user: str | None = None) -> list[str]:
    """The 31 NFL teams that can be an opponent (every team except ours)."""
    me = user or user_team()
    return [t for t in nfl_teams() if t != me]


def resolve_opponent_team(raw: str | None, user: str | None = None) -> str:
    """Canonical NFL team name for this game's opponent. Raises ValueError otherwise."""
    me = user or user_team()
    if not raw or not str(raw).strip():
        raise ValueError(
            "--opponent-team is required for --franchise online "
            '(e.g. --opponent-team "Buffalo Bills" or BUF).'
        )
    name, known = resolve_nfl_team(str(raw))
    if not known:
        raise ValueError(
            f"{raw!r} is not an NFL team. Use a full name, nickname, or abbreviation "
            '(e.g. "Buffalo Bills", Bills, BUF).'
        )
    if name == me:
        raise ValueError(
            f"{name} is your team in the online league; it cannot be the opponent team "
            "(no opponent data is collected on it)."
        )
    return name


# ---------------------------------------------------------------------------
# Stock-book scout (Huddle playbooks_full.json)
# ---------------------------------------------------------------------------

_COVER = re.compile(r"\bcover\s*(\d)", re.I)
_RUN_WORDS = re.compile(
    r"\b(hb|fb|qb|zone|dive|counter|power|toss|sweep|draw|option|sneak|stretch|trap|iso|"
    r"wham|duo|pin pull|off tackle|dart|slash|sprint|blast|lead)\b",
    re.I,
)
_PASS_HINT = re.compile(r"\b(pa|rpo|screen|boot|pass|flood|mesh|verticals|post|corner|curl|slant|wheel|"
                        r"spacing|stick|smash|cross|out|levels|seam|angle|drive|dig|shallow|choice)\b", re.I)


def _book_for(side: str, team: str) -> dict[str, Any] | None:
    from cfb_coach.madden.data import _load_json

    books = ((_load_json("playbooks_full.json").get("books") or {}).get(side)) or {}
    for key, book in books.items():
        if book.get("kind") == "team" and book.get("team") == team:
            return {"key": key, **book}
    return None


def _defense_scout(book: dict[str, Any]) -> dict[str, Any]:
    fams: Counter[str] = Counter()
    covers: Counter[str] = Counter()
    blitz = man = total = 0
    for fname, f in (book.get("formations") or {}).items():
        plays = list(f.get("plays") or [])
        fams[str(f.get("family") or fname.split()[0])] += len(plays)
        for p in plays:
            total += 1
            m = _COVER.search(p)
            if m:
                covers[f"Cover {m.group(1)}"] += 1
            if re.search(r"\bman\b", p, re.I):
                man += 1
            if re.search(r"blitz|fire|zero|sting|storm|mug", p, re.I):
                blitz += 1
    return {
        "book": book.get("key"),
        "formations": len(book.get("formations") or {}),
        "plays": total,
        "families": dict(fams.most_common()),
        "coverage_names": dict(covers.most_common()),
        "man_named_plays": man,
        "pressure_named_plays": blitz,
        "url": book.get("url"),
    }


def _offense_scout(book: dict[str, Any]) -> dict[str, Any]:
    fams: Counter[str] = Counter()
    runs = total = 0
    for fname, f in (book.get("formations") or {}).items():
        plays = list(f.get("plays") or [])
        fams[str(f.get("family") or fname.split()[0])] += len(plays)
        for p in plays:
            total += 1
            if _RUN_WORDS.search(p) and not _PASS_HINT.search(p):
                runs += 1
    return {
        "book": book.get("key"),
        "formations": len(book.get("formations") or {}),
        "plays": total,
        "families": dict(fams.most_common()),
        "run_named_plays_approx": runs,
        "url": book.get("url"),
    }


def stock_scout(team: str, user: str | None = None) -> dict[str, Any]:
    """Stock offense + defense book summary for an opponent team (never our own team)."""
    me = user or user_team()
    if team == me:
        raise ValueError(f"{team} is your team; it is not scouted as an opponent.")
    out: dict[str, Any] = {"team": team, "source": STOCK_SCOUT_LABEL}
    d = _book_for("defense", team)
    o = _book_for("offense", team)
    out["defense"] = _defense_scout(d) if d else None
    out["offense"] = _offense_scout(o) if o else None
    return out


# ---------------------------------------------------------------------------
# Logged data by opponent team
# ---------------------------------------------------------------------------

def _has_league_columns(db: Any) -> bool:
    try:
        cols = {r[1] for r in db.conn.execute("PRAGMA table_info(game_sessions)").fetchall()}
        scols = {r[1] for r in db.conn.execute("PRAGMA table_info(snaps)").fetchall()}
    except Exception:  # noqa: BLE001
        return False
    return {"league", "opponent_team"} <= cols and {"league", "opponent_team"} <= scols


def logged_summary(db: Any, team: str) -> dict[str, Any]:
    """Games / snaps / coverage mix / success for this team in the online league."""
    out: dict[str, Any] = {
        "team": team, "source": LOGGED_LABEL, "games": 0, "snaps": 0,
        "offense_snaps": 0, "defense_snaps": 0, "last_played": None,
        "opponents": [], "coverage_seen": {}, "their_plays_seen": {},
        "offense_success_rate": None, "defense_success_rate": None,
    }
    if db is None or not _has_league_columns(db):
        return out
    sess = db.conn.execute(
        "SELECT session_id, opponent_id, started_ts FROM game_sessions "
        "WHERE league = ? AND opponent_team = ? ORDER BY started_ts",
        (LEAGUE_ID, team),
    ).fetchall()
    out["games"] = len(sess)
    out["last_played"] = sess[-1][2] if sess else None
    out["opponents"] = sorted({str(r[1]) for r in sess})
    rows = db.conn.execute(
        "SELECT * FROM snaps WHERE league = ? AND opponent_team = ?",
        (LEAGUE_ID, team),
    ).fetchall()
    from cfb_coach.madden.postgame import snap_success

    cov: Counter[str] = Counter()
    con: Counter[str] = Counter()
    o_ok = o_n = d_ok = d_n = 0
    for r in rows:
        side = str(r["side"] or "")
        try:
            ok = snap_success(r)
        except Exception:  # noqa: BLE001
            ok = None
        if side.startswith("o"):
            out["offense_snaps"] += 1
            if r["coverage_seen"]:
                cov[str(r["coverage_seen"])] += 1
            if ok is not None:
                o_n += 1
                o_ok += int(ok)
        elif side.startswith("d"):
            out["defense_snaps"] += 1
            if r["concept_seen"]:
                con[str(r["concept_seen"])] += 1
            if ok is not None:
                d_n += 1
                d_ok += int(ok)
    out["snaps"] = len(rows)
    out["coverage_seen"] = dict(cov.most_common(8))
    out["their_plays_seen"] = dict(con.most_common(8))
    out["offense_success_rate"] = round(o_ok / o_n, 3) if o_n else None
    out["defense_success_rate"] = round(d_ok / d_n, 3) if d_n else None
    return out


def inventory(db: Any, user: str | None = None) -> list[dict[str, Any]]:
    """One row per possible opponent team (31; never our own team)."""
    me = user or user_team()
    rows = []
    for team in opponent_teams(me):
        s = logged_summary(db, team)
        rows.append({
            "team": team,
            "games": s["games"],
            "snaps": s["snaps"],
            "last_played": s["last_played"],
            "opponents": s["opponents"],
            "status": "none yet" if not s["games"] else "logged",
        })
    return rows


def _local(ts: str | None) -> str:
    if not ts:
        return "none yet"
    from datetime import datetime

    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except ValueError:
        return ts[:19]


def format_inventory(db: Any, user: str | None = None) -> str:
    me = user or user_team()
    rows = inventory(db, me)
    have = sum(1 for r in rows if r["games"])
    lines = [
        f"Online league — you: {me}. Opponent teams with data: {have}/{len(rows)}",
        f"{'Team':<24} {'Games':>5} {'Snaps':>6}  {'Last played (local)':<26} Opponents",
    ]
    for r in rows:
        last = _local(r["last_played"])
        who = ", ".join(r["opponents"]) if r["opponents"] else "-"
        lines.append(f"{r['team']:<24} {r['games']:>5} {r['snaps']:>6}  {last:<26} {who}")
    return "\n".join(lines)


def scout_report(db: Any, team: str, user: str | None = None) -> dict[str, Any]:
    return {"stock": stock_scout(team, user), "logged": logged_summary(db, team)}


def format_scout(report: dict[str, Any]) -> str:
    st = report["stock"]
    lg = report["logged"]
    lines = [f"SCOUT — {st['team']}", f"  [{st['source']}]"]
    d = st.get("defense")
    if d:
        fams = ", ".join(f"{k} {v}" for k, v in list(d["families"].items())[:6])
        covs = ", ".join(f"{k} {v}" for k, v in d["coverage_names"].items()) or "-"
        lines.append(f"  Defense book {d['book']}: {d['formations']} formations / {d['plays']} plays — {fams}")
        lines.append(f"    coverage-named plays: {covs}; man-named {d['man_named_plays']}, "
                     f"pressure-named {d['pressure_named_plays']}")
    else:
        lines.append("  Defense book: none in playbooks_full.json")
    o = st.get("offense")
    if o:
        fams = ", ".join(f"{k} {v}" for k, v in list(o["families"].items())[:6])
        lines.append(f"  Offense book {o['book']}: {o['formations']} formations / {o['plays']} plays "
                     f"(~{o['run_named_plays_approx']} run-named) — {fams}")
    else:
        lines.append("  Offense book: none in playbooks_full.json")
    lines.append(f"  [{lg['source']}]")
    if not lg["games"]:
        lines.append("    none yet — this game will be the first logged sample.")
    else:
        lines.append(f"    games {lg['games']}, snaps {lg['snaps']} (O {lg['offense_snaps']} / D {lg['defense_snaps']}), "
                     f"last {_local(lg['last_played'])}, users: {', '.join(lg['opponents'])}")
        if lg["coverage_seen"]:
            lines.append("    coverage seen: " + ", ".join(f"{k} {v}" for k, v in lg["coverage_seen"].items()))
        if lg["their_plays_seen"]:
            lines.append("    their plays seen: " + ", ".join(f"{k} {v}" for k, v in lg["their_plays_seen"].items()))
        if lg["offense_success_rate"] is not None:
            lines.append(f"    our offense success {lg['offense_success_rate']:.0%}")
        if lg["defense_success_rate"] is not None:
            lines.append(f"    our defense success {lg['defense_success_rate']:.0%}")
    return "\n".join(lines)


def league_context(db: Any, opponent_id: str, opponent_team: str, *, cpu: bool) -> dict[str, Any]:
    """Tags stored on the session row (and copied onto each snap)."""
    me = user_team()
    team = resolve_opponent_team(opponent_team, me)
    report = scout_report(db, team, me)
    return {
        "league": LEAGUE_ID,
        "user_team": me,
        "opponent_team": team,
        "opponent_type": "cpu" if cpu else "human",
        "opponent_id": opponent_id,
        "scout_json": json.dumps(report, default=str),
        "_report": report,
    }
