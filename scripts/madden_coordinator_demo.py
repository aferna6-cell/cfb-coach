"""Reproducible coordinator demo built on Gameplan Coach Demo C.

Scenarios, the trimmed book, and the macro ids come from
``cfb_coach.madden.adjustments.demo``. The down-and-distance lines below are
synthetic. Plays are whatever that fixture book already contains. Database
files are temporary ``CFB_COACH_DB`` and ``CFB_COACH_MADDEN_DB`` paths.
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_OFFENSE_STATES = ("1&10", "2&7", "3&2", "1&10 my 40", "3&8")
_DEFENSE_STATES = ("d 1&10", "d 3&6", "d 2&6", "d 1&10 my 25")
_DEFENSE_INPUTS = ("d 3&6", "d 2&8", "d 1&10")
_CALLS = 50
_FIXTURE_LABEL = "TEST FIXTURE - not an Aidan-verified macro"
_REPORT = Path("docs/madden_ml/coordinator_demo_report.md")
_OFFENSE_SHAPE = "Formation — Play | Adjustment | Read 1 → Read 2"
_DEFENSE_SHAPE = "Formation — Play | Adjustment/Macro | User responsibility"


def _prepare_env() -> tempfile.TemporaryDirectory[str]:
    folder = tempfile.TemporaryDirectory(prefix="madden-coordinator-demo-")
    root = Path(folder.name)
    empty = root / "empty-home"
    empty.mkdir()
    os.environ["HOME"] = str(empty)
    os.environ["CFB_COACH_DB"] = str(root / "coach.db")
    os.environ["CFB_COACH_MADDEN_DB"] = str(root / "madden27.db")
    os.environ["CFB_COACH_NO_VOD_PRIOR"] = "1"
    os.environ["CFB_COACH_AI_RESEARCH"] = "off"
    return folder


def _nearest_rank(samples: list[float], pct: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return ordered[index]


def _split_reads(raw: str) -> tuple[str | None, str | None]:
    text = (raw or "").strip()
    if "→" not in text:
        return (text or None), None
    parts = [part.strip() for part in text.split("→") if part.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return parts[0], None
    return None, None


def _why(old: str, new: str) -> str:
    if old == new:
        return "same line"
    reasons: list[str] = []
    if "Checkdown" in old and "Checkdown" not in new:
        reasons.append(
            "heuristic formatter fills a missing second read with Checkdown; the live line uses an em dash"
        )
    if "| none |" in old and "| No adj |" in new:
        reasons.append(
            "heuristic defense prints none for an empty adjustment; the live line prints No adj"
        )
    if not reasons:
        reasons.append("live exact-line slots differ from the heuristic formatter")
    return "; ".join(reasons)


def run() -> str:
    folder = _prepare_env()
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import data as mdata
    from cfb_coach.madden.adjustments.candidates import is_live_selectable
    from cfb_coach.madden.adjustments.demo import _EVIDENCE
    from cfb_coach.madden.adjustments.synthesis import (
        DESIGN_VERSION,
        MacroState,
        NotLiveSelectable,
        generate_from_db,
        install_trimmed_book,
        installed_formations,
        require_live,
        select_live,
        transition,
        trimmed_formations,
    )
    from cfb_coach.madden.data import reads_for, user_job_for
    from cfb_coach.madden.model.defense_coordinator import live_pick
    from cfb_coach.madden.model.inference import (
        ML_LATENCY_BUDGET_MS,
        exact_defense_line,
        exact_offense_line,
        present_snap,
        rank_live,
    )
    from cfb_coach.madden.model.schema import (
        CoachingMode,
        MLStatus,
        Possession,
        candidate_in_book,
    )
    from cfb_coach.madden.playcaller import make_call
    from cfb_coach.madden.situation import parse_madden_situation

    def displayed(call) -> str:
        if call.side == "defense":
            return exact_defense_line(
                call.formation,
                call.play,
                None if call.adj_or_macro in ("", "none", "No adj") else call.adj_or_macro,
                call.read_or_user or user_job_for(call.play),
                macro_name=call.macro,
                macro_state=MacroState.ARMED if call.macro else None,
            )
        read1, read2 = _split_reads(call.read_or_user or reads_for(call.play))
        return exact_offense_line(
            call.formation,
            call.play,
            None if call.adj_or_macro in ("", "No adj") else call.adj_or_macro,
            read1,
            read2,
            macro_name=call.macro,
            macro_state=MacroState.ARMED if call.macro else None,
        )

    def in_book(side: str, formation: str, play: str) -> None:
        plays = forms[side].get(formation) or []
        if play not in plays:
            raise SystemExit(f"{side} play is not in the Demo C book: {formation} / {play}")

    class Side:
        """One coordinator. It keeps its own last line and its own seed stream."""

        def __init__(self, side: str) -> None:
            self.side = side
            self.last: str | None = None

        def pick(self, raw: str, seed: int, *, time_model: bool) -> dict[str, str]:
            sit = parse_madden_situation(raw, default_side=self.side)
            started = time.perf_counter()
            call = make_call(
                sit,
                "gavin",
                db,
                rng=random.Random(seed),
                live_macros=False,
                active_macros={"offense": [], "defense": []},
            )
            whole_ms = (time.perf_counter() - started) * 1000.0
            in_book(call.side, call.formation, call.play)
            new = displayed(call)
            old = call.format().splitlines()[0]
            self.last = new
            model_ms = ""
            status = ""
            if time_model:
                elapsed, model_status = model_call(self.side, sit, seed)
                whole_ms += elapsed
                model_ms = f"{elapsed:.3f}"
                status = model_status.value
            return {
                "side": self.side,
                "state": raw,
                "old": old,
                "new": new,
                "why": _why(old, new),
                "whole_ms": f"{whole_ms:.3f}",
                "model_ms": model_ms,
                "status": status,
                "play": call.play,
                "formation": call.formation,
            }

    def rank_side(sit, book: dict, index: int, side: str):
        from cfb_coach.madden.model.inference import _SITUATION_ADAPTER

        poss = Possession.DEFENSE if side == "defense" else Possession.OFFENSE
        state, obs = _SITUATION_ADAPTER(
            sit,
            game_id="coordinator-demo",
            snap_id=f"demo-{side}-{index}",
            opponent_id="gavin",
            madden_version="madden27",
        )
        candidates = [
            candidate_in_book(book, side=poss, formation=formation, play=play)
            for formation, plays in book.items()
            for play in plays
        ]
        return rank_live(
            state,
            obs,
            candidates,
            None,
            None,
            (),
            mode=CoachingMode.SHADOW,
            registry_dir=str(Path(folder.name) / "empty-registry"),
            session_id=f"demo-{side}",
            snap_seq=index,
        )

    def model_call(side: str, sit, index: int) -> tuple[float, MLStatus]:
        book = forms[side]
        started = time.perf_counter()
        try:
            if side == "defense":
                picked = live_pick(
                    sit, db, "gavin", book, session_id="demo-def-latency", snap_seq=index,
                )
                elapsed = float(picked["elapsed_ms"])
                decision = picked.get("decision")
                status = getattr(decision, "shadow_status", None) or MLStatus.OK
                chosen = picked["selected"]
                in_book("defense", chosen["formation"], chosen["play"])
            else:
                decision = rank_side(sit, book, index, side)
                recorded = getattr(decision, "latency_ms_model", None)
                elapsed = float(recorded) if recorded is not None else (
                    (time.perf_counter() - started) * 1000.0
                )
                status = getattr(decision, "shadow_status", None) or MLStatus.MODEL_MISSING
            if elapsed > ML_LATENCY_BUDGET_MS:
                return elapsed, MLStatus.TIMEOUT
            return elapsed, status
        except TimeoutError:
            return (time.perf_counter() - started) * 1000.0, MLStatus.TIMEOUT
        except Exception:
            return (time.perf_counter() - started) * 1000.0, MLStatus.EXCEPTION

    def macro_section() -> list[str]:
        records = generate_from_db(db)
        installed = installed_formations(db)
        draft = next(
            row for row in records
            if row.candidate.formation == "Dime 3-2 Odd"
            and row.candidate.play == "DB Fire 2"
            and not row.candidate.is_no_adjustment
        )
        candidate = draft.candidate
        in_book("defense", candidate.formation, candidate.play)
        try:
            require_live(candidate)
        except NotLiveSelectable as exc:
            refused = str(exc)
        else:
            raise SystemExit("Demo C draft was selectable")
        hidden = present_snap(
            side="defense",
            formation=candidate.formation,
            play=candidate.play,
            user_job=user_job_for(candidate.play),
            macro_name=candidate.macro_id,
            macro_state=MacroState.DRAFT,
            mode=CoachingMode.HEURISTIC,
        )
        if candidate.macro_id in hidden.line or is_live_selectable(MacroState.DRAFT):
            raise SystemExit("DRAFT macro was shown")
        armed = draft
        for state, evidence in (
            (MacroState.EDITOR_VERIFIED, _EVIDENCE[0]),
            (MacroState.INSTALLED, _EVIDENCE[1]),
            (MacroState.ARMED, _EVIDENCE[2]),
        ):
            armed = transition(armed, state, evidence)
        if not is_live_selectable(armed.candidate.state):
            raise SystemExit("ARMED macro was not selectable")
        shown = present_snap(
            side="defense",
            formation=armed.candidate.formation,
            play=armed.candidate.play,
            user_job=user_job_for(armed.candidate.play),
            macro_name=armed.candidate.macro_id,
            macro_state=armed.candidate.state,
            mode=CoachingMode.HEURISTIC,
        )
        if armed.candidate.macro_id not in shown.line:
            raise SystemExit("ARMED macro was not displayed")
        updated = [
            armed if row.candidate.macro_id == armed.candidate.macro_id else row
            for row in records
        ]
        chosen = select_live(
            updated, formation="Dime 3-2 Odd", play="DB Fire 2", installed=installed,
        )
        baseline = select_live(
            records, formation="Pistol Trips", play="HB Dive", installed=installed,
        )
        in_book("offense", baseline.formation, baseline.play)
        return [
            f"Source: Demo C (`cfb_coach.madden.adjustments.demo`), design version {DESIGN_VERSION}.",
            (
                "Checked-in data has no Aidan-verified live macro. "
                "Catalog rows marked confirmed are Aidan's setting notes, not a MacroState record. "
                f"This run uses Demo C's generated macro. {_FIXTURE_LABEL}."
            ),
            f"Macro used: {candidate.macro_id} ({candidate.formation} / {candidate.play}). {_FIXTURE_LABEL}.",
            f"Draft refused: {candidate.macro_id}. {refused}",
            f"Draft line (macro absent): {hidden.line}",
            f"Armed line: {shown.line}",
            f"select_live armed: {chosen.macro_id} state {chosen.state.value}",
            (
                f"Unarmed Demo C play falls back to baseline {baseline.macro_id} "
                f"is_no_adjustment={baseline.is_no_adjustment} ({baseline.formation} / {baseline.play})"
            ),
        ]

    def latency_section(samples: list[dict[str, str]]) -> list[str]:
        whole = [float(row["whole_ms"]) for row in samples]
        model = [float(row["model_ms"]) for row in samples]
        counts = Counter(row["status"] for row in samples)
        fallback = sum(1 for row in samples if row["status"] != MLStatus.OK.value)
        reasons = {
            MLStatus.MODEL_MISSING.value: "no model artifact is installed in the temp registry, so the live rank keeps the heuristic line",
            MLStatus.TIMEOUT.value: "the model call stepped past ml.latency_budget_ms",
            MLStatus.EXCEPTION.value: "the model call raised",
            MLStatus.INVALID_OUTPUT.value: "the model result was not a legal line",
            MLStatus.LOW_CONFIDENCE.value: "the model result was below the confidence bar",
            MLStatus.CIRCUIT_OPEN.value: "the model path is closed and the heuristic line stays",
            MLStatus.WORKER_BUSY.value: "the model worker was busy",
            MLStatus.UNKNOWN.value: "the model status was unknown",
        }
        lines = [
            f"ml.latency_budget_ms stays {ML_LATENCY_BUDGET_MS}.",
            f"Calls timed: {len(samples)}.",
            (
                "Whole call (make_call plus the model call) ms: "
                f"p50 {_nearest_rank(whole, 50):.3f}, "
                f"p95 {_nearest_rank(whole, 95):.3f}, "
                f"max {max(whole):.3f}."
            ),
            (
                "Model call only (defense live_pick elapsed_ms, offense rank_live latency) ms: "
                f"p50 {_nearest_rank(model, 50):.3f}, "
                f"p95 {_nearest_rank(model, 95):.3f}, "
                f"max {max(model):.3f}."
            ),
            f"Fallback count: {fallback}. A fallback is any model result whose MLStatus is not {MLStatus.OK.value}.",
            "MLStatus counts: " + ", ".join(f"{name}={counts[name]}" for name in sorted(counts)),
        ]
        for name in sorted(counts):
            if name == MLStatus.OK.value:
                continue
            lines.append(f"Fallback {name}: {reasons.get(name, name)} ({counts[name]}).")
        if ML_LATENCY_BUDGET_MS != 150:
            raise SystemExit("latency budget was rewritten")
        return lines

    db = CoachDB(Path(os.environ["CFB_COACH_MADDEN_DB"]), seed=mdata.load_seed())
    try:
        install_trimmed_book(db)
        db.log_snap(
            opponent_id="gavin",
            side="defense",
            situation_raw="d 3&6",
            our_call="fixture",
            formation="Nickel Over",
            play="Cover 4 Quarters",
            result="stop",
            macro="TAMPA MABLE",
        )
        from cfb_coach.madden.macros import store_selection

        store_selection(
            db, "gavin", {"offense": ["MAN", "C2"], "defense": ["TAMPA MABLE", "SKY CONTAIN"]},
        )
        db.conn.commit()
        forms = trimmed_formations()
        empty_home = Path(os.environ["HOME"])
        snaps_before = db.conn.execute("SELECT COUNT(*) FROM snaps").fetchone()[0]

        offense = Side("offense")
        defense = Side("defense")
        before = [offense.pick(raw, 100 + index, time_model=False) for index, raw in enumerate(_OFFENSE_STATES)]
        defense_picks = []
        for index, raw in enumerate(_DEFENSE_INPUTS):
            sit = parse_madden_situation(raw, default_side="defense")
            picked = live_pick(
                sit, db, "gavin", forms["defense"], session_id="demo-def", snap_seq=index,
            )
            chosen = picked["selected"]
            in_book("defense", chosen["formation"], chosen["play"])
            defense_picks.append(f"{raw} -> {chosen['formation']} / {chosen['play']}")
        after = [offense.pick(raw, 100 + index, time_model=False) for index, raw in enumerate(_OFFENSE_STATES)]
        snaps_after = db.conn.execute("SELECT COUNT(*) FROM snaps").fetchone()[0]
        if snaps_before != snaps_after:
            raise SystemExit("defense live_pick wrote a snap and could couple the next offense call")
        changed = [row["state"] for left, row in zip(before, after) if left["new"] != row["new"]]
        if changed:
            raise SystemExit(f"offense pick changed after defense inputs changed: {changed}")

        sequence = list(before)
        for index, raw in enumerate(_DEFENSE_STATES):
            sequence.append(defense.pick(raw, 200 + index, time_model=False))
        timed: list[dict[str, str]] = []
        for index in range(_CALLS):
            if index % 2:
                raw = _DEFENSE_STATES[index % len(_DEFENSE_STATES)]
                timed.append(defense.pick(raw, 300 + index, time_model=True))
            else:
                raw = _OFFENSE_STATES[index % len(_OFFENSE_STATES)]
                timed.append(offense.pick(raw, 400 + index, time_model=True))
        macro_lines = macro_section()
        latency_lines = latency_section(timed)
        home_files = [path for path in empty_home.rglob("*") if path.is_file()]
        if home_files:
            raise SystemExit(f"demo wrote under HOME: {home_files[:3]}")
    finally:
        db.close()
        folder.cleanup()

    rows = []
    for row in sequence:
        rows.append(f"- {row['side']} `{row['state']}` (synthetic) — {row['formation']} / {row['play']}")
        rows.append(f"  - old: {row['old']}")
        rows.append(f"  - live: {row['new']}")
        rows.append(f"  - why: {row['why']}")
    differed = [row for row in sequence if row["old"] != row["new"]]
    unchanged = sum(1 for left, right in zip(before, after) if left["new"] == right["new"])
    body = [
        "# Coordinator demo",
        "",
        "Down-and-distance lines are synthetic. Formations and plays come from Demo C's trimmed Buccaneers and 49ers book. "
        "No VOD prior is enabled (`CFB_COACH_NO_VOD_PRIOR=1`). "
        f"`CFB_COACH_DB` and `CFB_COACH_MADDEN_DB` are separate temp files. {_FIXTURE_LABEL}.",
        "",
        "## Independent coordinators",
        "",
        "Offense and defense are separate coordinators. Each keeps its own last line. "
        "The offense sequence is called again with the same seeds after the defensive inputs change. "
        "Defense `live_pick` reads the defensive book and does not log a snap, so it does not change the offense read of the fixture.",
        "",
        f"Offense lines unchanged: {unchanged}/{len(before)}.",
        "Offense before:",
    ]
    body.extend(f"- `{row['state']}`: {row['new']}" for row in before)
    body.append("Defense inputs changed:")
    body.extend(f"- {line}" for line in defense_picks)
    body.append("Offense after:")
    body.extend(f"- `{row['state']}`: {row['new']}" for row in after)
    body.extend(["", "## Sequence", ""])
    body.extend(rows)
    body.extend(["", "## Macro lifecycle", ""])
    body.extend(macro_lines)
    body.extend(["", "## Latency", ""])
    body.extend(latency_lines)
    body.extend(["", "## Diff versus the heuristic caller", ""])
    body.append(
        f"{len(differed)} of {len(sequence)} displayed lines differ from `make_call().format()` "
        "(the heuristic caller at df81381). The live line is "
        f"`{_OFFENSE_SHAPE}` on offense and "
        f"`{_DEFENSE_SHAPE}` on defense."
    )
    body.append("")
    if differed:
        for row in differed:
            body.append(f"- {row['side']} {row['state']}: {row['why']}")
            body.append(f"  - old: {row['old']}")
            body.append(f"  - live: {row['new']}")
    else:
        body.append("No displayed line differed on this sequence.")
    body.append("")
    return "\n".join(body) + "\n"


def main() -> int:
    report = run()
    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
