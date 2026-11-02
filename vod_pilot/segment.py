"""Group per-frame OCR observations into snaps and infer the result.

A snap is a stretch of frames that share one down-and-distance. The result is
the change into the next down-and-distance when the game clock is still the
same drive. Yards are ``previous distance - next distance`` when the down
advances by exactly one. Anything else (skipped down, quarter change, big
clock jump) is left blank rather than invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from vod_pilot.adjustments import read_adjustments
from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.parse_hud import (
    clocks_compatible,
    parse_clock,
    parse_down_distance,
    parse_flags,
    parse_yardline_phrase,
)
from vod_pilot.schema import SnapRow, result_text, situation_raw
from vod_pilot.scorebug import ScoreRead, parse_scorebug
from vod_pilot.screen_read import read_defensive_formation, read_offensive_formation
from vod_pilot.tokens import token_text

_MENU_MARKERS = (
    "SELECT GAME MODE",
    "SELECTGAMEMODE",
    "COIN TOSS",
    "COINTOSS",
    "DYNASTY CENTRAL",
    "DYNASTYCENTRAL",
    "RECRUITING",
    "JOB SECURITY",
    "JOBSECURITY",
    "CREATE SAVEPOINT",
    "CREATESAVEPOINT",
    "MANAGE BLUEPRINT",
    "ADVANCE TO",
    "PLAY GAME",
    "PLAYGAME",
    "NEXT WEEK",
    "NEXTWEEK",
)
_PLAYCALL_MARKERS = (
    "COACH SUGGEST",
    "PLAY CALL",
    "PLAYCALL",
    "PREVIOUS PLAY",
    "HOT ROUTE",
    "HOTROUTES",
    "FORMATION",
)
_ROUTE_WORDS = (
    "mesh",
    "smash",
    "spot",
    "corner",
    "post",
    "streak",
    "wheel",
    "slant",
    "drag",
    "screen",
    "out",
    "curl",
    "flat",
    "seam",
)
_DEF_PREFIXES = (
    "3-3",
    "3-4",
    "4-2",
    "4-3",
    "4-4",
    "5-2",
    "46",
    "DIME",
    "DOLLAR",
    "NICKEL",
    "GOAL LINE",
    "BIG NICKEL",
    "QUARTER",
)


@dataclass
class FrameObs:
    t: float
    kind: str  # menu | kickoff | play_call | live | other
    green: float = 0.0
    down: int | None = None
    distance: int | None = None
    quarter: int | None = None
    clock: int | None = None
    formation: str = ""
    play: str = ""
    coverage: str = ""
    formation_hits: list[str] = field(default_factory=list)
    coverage_hits: list[str] = field(default_factory=list)
    yardline: int | None = None
    yard_phrase: str = ""
    flags: set[str] = field(default_factory=set)
    texts: list[str] = field(default_factory=list)
    previous_coverage: str = ""
    previous_play: str = ""
    def_formation: str = ""
    def_formation_conf: float = 0.0
    previous_def_formation: str = ""
    previous_def_conf: float = 0.0
    off_formation_conf: float = 0.0
    previous_off_formation: str = ""
    previous_off_conf: float = 0.0
    team_left: str = ""
    team_right: str = ""
    score_left: int | None = None
    score_right: int | None = None
    record_left: str = ""
    record_right: str = ""
    score_conf: float = 0.0
    adjustments: list[str] = field(default_factory=list)
    adjustment_conf: float = 0.0

    @property
    def dd(self) -> tuple[int, int] | None:
        if self.down is None or self.distance is None:
            return None
        return self.down, self.distance


def classify_frame(
    t: float,
    texts: list,
    green: float,
    lexicon: Lexicon,
    extract: ExtractFlags | None = None,
) -> FrameObs:
    extract = extract or ExtractFlags()
    raw_tokens = list(texts)
    texts = [token_text(item) for item in raw_tokens]
    blob = " ".join(texts).upper()
    compact = blob.replace(" ", "")
    dd = parse_down_distance(texts)
    clock = parse_clock(texts)
    flags = parse_flags(texts)
    yardline, yard_phrase = parse_yardline_phrase(texts, dd)
    from vod_pilot.lexicon import _hits

    formation_hits = lexicon.formation_hits(texts)
    coverage_hits = _hits(lexicon._cov_keys, texts)
    play = lexicon.match_play(texts)
    menu_call = any(marker in compact for marker in ("PREVIOUSPLAY", "SELECTAPLAY", "PLAYCALLSUBSTITUTION"))
    penalty = "PENALTY" in compact
    # The formation tab prints every row. A single lexicon hit on that screen
    # is whichever name OCR happened to keep, not the highlighted call.
    # "PREVIOUS PLAY" names the call that just ended, so it must not tag the
    # next snap. A penalty accept/decline screen is not a play call at all.
    previous_coverage = ""
    previous_play = ""
    if menu_call or penalty:
        if "PREVIOUSPLAY" in compact and not penalty:
            if len(coverage_hits) == 1:
                previous_coverage = coverage_hits[0]
            if play:
                previous_play = play
        formation_hits = []
        coverage_hits = []
        if penalty or "PREVIOUSPLAY" in compact:
            play = ""
    formation = formation_hits[0] if len(formation_hits) == 1 else ""
    coverage = coverage_hits[0] if len(coverage_hits) == 1 else ""
    def_formation = ""
    def_formation_conf = 0.0
    previous_def_formation = ""
    previous_def_conf = 0.0
    off_formation_conf = 0.0
    previous_off_formation = ""
    previous_off_conf = 0.0
    if extract.read_defense:
        # Packaged fronts from every book, plus one custom line when it is not
        # a prefix of two real fronts. Several names and no tooltip: no read.
        read_name, read_conf = read_defensive_formation(texts)
        if "PREVIOUSPLAY" in compact and not penalty and read_name:
            previous_def_formation = read_name
            previous_def_conf = read_conf
        elif read_name and not formation_hits:
            def_formation = read_name
            def_formation_conf = read_conf
            formation_hits = [read_name]
            formation = read_name
        elif read_name:
            def_formation = read_name
            def_formation_conf = read_conf
    if extract.read_offense_formation:
        off_name, off_conf = read_offensive_formation(texts)
        if "PREVIOUSPLAY" in compact and not penalty and off_name:
            previous_off_formation = off_name
            previous_off_conf = off_conf
        elif off_name and not formation and not penalty:
            # Lexicon abstains when Gun and Shotgun share the set. The set is
            # still what was painted, so keep that string and not a book family.
            formation = off_name
            formation_hits = [off_name]
            off_formation_conf = off_conf
    score = parse_scorebug(raw_tokens) if extract.read_score else ScoreRead()
    adjustments: list[str] = []
    adjustment_conf = 0.0
    if extract.read_adjustments:
        found, adjustment_conf = read_adjustments(texts)
        adjustments = [item.label() for item in found]

    kind = "other"
    if any(marker in compact or marker in blob for marker in _MENU_MARKERS):
        kind = "menu"
    elif "KICKOFF" in compact or "KICK OFF" in blob:
        kind = "kickoff"
    elif green < 0.40 and (
        coverage
        or formation
        or play
        or any(marker in compact or marker in blob for marker in _PLAYCALL_MARKERS)
    ):
        kind = "play_call"
    elif dd and green >= 0.20:
        kind = "live"
    elif dd:
        kind = "live"

    return FrameObs(
        t=t,
        kind=kind,
        green=green,
        down=dd[0] if dd else None,
        distance=dd[1] if dd else None,
        quarter=clock[0] if clock else None,
        clock=clock[1] if clock else None,
        formation=formation,
        play=play,
        coverage=coverage,
        formation_hits=formation_hits,
        coverage_hits=coverage_hits,
        yardline=yardline,
        yard_phrase=yard_phrase,
        flags=flags,
        texts=list(texts),
        previous_coverage=previous_coverage,
        previous_play=previous_play,
        def_formation=def_formation,
        def_formation_conf=def_formation_conf,
        previous_def_formation=previous_def_formation,
        previous_def_conf=previous_def_conf,
        off_formation_conf=off_formation_conf,
        previous_off_formation=previous_off_formation,
        previous_off_conf=previous_off_conf,
        team_left=score.team_left,
        team_right=score.team_right,
        score_left=score.score_left,
        score_right=score.score_right,
        record_left=score.record_left,
        record_right=score.record_right,
        score_conf=score.confidence,
        adjustments=adjustments,
        adjustment_conf=adjustment_conf,
    )


def smooth_down_distance(frames: list[FrameObs]) -> list[FrameObs]:
    """Drop a one-frame down/distance blip surrounded by the same reading."""
    if len(frames) < 3:
        return list(frames)
    out = list(frames)
    for i in range(1, len(out) - 1):
        prev, cur, nxt = out[i - 1], out[i], out[i + 1]
        if prev.dd and prev.dd == nxt.dd and cur.dd != prev.dd:
            if 0 < (nxt.t - prev.t) <= 8:
                out[i] = replace(cur, down=prev.down, distance=prev.distance)
    return out


@dataclass
class _Run:
    frames: list[FrameObs]

    @property
    def t_start(self) -> float:
        return self.frames[0].t

    @property
    def t_end(self) -> float:
        return self.frames[-1].t

    def majority_dd(self) -> tuple[int, int] | None:
        counts: dict[tuple[int, int], int] = {}
        for frame in self.frames:
            if frame.dd:
                counts[frame.dd] = counts.get(frame.dd, 0) + 1
        if not counts:
            return None
        return max(counts, key=lambda key: (counts[key], key[0]))


def _runs(frames: list[FrameObs]) -> list[_Run]:
    runs: list[_Run] = []
    current: list[FrameObs] = []

    def close() -> None:
        nonlocal current
        if current:
            runs.append(_Run(current))
        current = []

    for frame in frames:
        usable = frame.kind == "kickoff" or (frame.kind == "live" and frame.dd is not None)
        if not usable:
            # A play-call or menu frame between two looks at the same down
            # is not a new snap. Only the next live frame can close the run.
            continue
        if not current:
            current = [frame]
            continue
        gap = frame.t - current[-1].t
        same_kind = frame.kind == current[-1].kind
        same_dd = frame.dd == current[-1].dd
        # Pre-snap play-art can sit between two looks at the same down.
        if same_kind and same_dd and gap <= 25:
            current.append(frame)
        else:
            close()
            current = [frame]
    close()
    runs.sort(key=lambda run: run.t_start)
    return runs


def _attach_play_calls(runs: list[_Run], frames: list[FrameObs]) -> dict[int, list[FrameObs]]:
    attached: dict[int, list[FrameObs]] = {i: [] for i in range(len(runs))}
    for frame in frames:
        if frame.kind != "play_call":
            continue
        if not (frame.formation or frame.play or frame.coverage):
            continue
        best: int | None = None
        best_dist = 1e9
        for i, run in enumerate(runs):
            if run.t_start - 16 <= frame.t <= run.t_end + 1:
                dist = abs(frame.t - run.t_start)
                if dist < best_dist:
                    best, best_dist = i, dist
        if best is not None:
            attached[best].append(frame)
    return attached


def _clock_of(frames: list[FrameObs], *, last: bool) -> tuple[int | None, int | None]:
    ordered = frames if not last else list(reversed(frames))
    for frame in ordered:
        if frame.clock is not None:
            return frame.quarter, frame.clock
    return None, None


def _score_flags(flags: set[str]) -> tuple[str, int | None, float] | None:
    if "int" in flags:
        return "int", 0, 0.8
    if "fumble" in flags:
        return "fumble", 0, 0.7
    if "td" in flags:
        return "td", None, 0.75
    return None


def _infer_between(prev: _Run, nxt: _Run, boundary_flags: set[str]) -> tuple[str, int | None, float]:
    flags = boundary_flags
    # A kickoff has no down. A score or turnover banner just before it is
    # still the result of the snap that just ended.
    if prev.frames and prev.frames[0].kind == "kickoff":
        return "", None, 0.0
    if nxt.frames and nxt.frames[0].kind == "kickoff":
        scored = _score_flags(flags)
        return scored if scored is not None else ("", None, 0.0)

    dd0 = prev.majority_dd()
    dd1 = nxt.majority_dd()
    if not dd0 or not dd1:
        scored = _score_flags(flags)
        return scored if scored is not None else ("", None, 0.0)
    q0, c0 = _clock_of(prev.frames, last=True)
    q1, c1 = _clock_of(nxt.frames, last=False)
    if not clocks_compatible(q0, c0, q1, c1):
        scored = _score_flags(flags)
        return scored if scored is not None else ("", None, 0.0)
    clock_known = c0 is not None and c1 is not None
    conf = 0.72 if clock_known else 0.4
    scored = _score_flags(flags)
    if scored is not None:
        return scored

    down0, dist0 = dd0
    down1, dist1 = dd1
    if down1 == down0 + 1:
        yards = dist0 - dist1
        if "sack" in flags:
            return "sack", yards, min(0.85, conf + 0.1)
        if "incomplete" in flags and yards == 0:
            return "incomplete", 0, 0.75
        if yards > 0:
            return "gain", yards, conf
        if yards < 0:
            return "loss", yards, conf
        return "no_gain", 0, conf - 0.15
    if down1 == 1 and down0 in {2, 3} and dist1 == 10:
        return "convert", None, 0.4 if clock_known else 0.25
    return "", None, 0.0


def _side(formation: str, coverage: str) -> str:
    """Defense only when the formation itself is a front.

    ``coverage_seen`` on an offensive snap is what the defense showed. A
    coverage name alone must not flip the row to the defensive side.
    """
    del coverage
    upper = formation.upper()
    if any(upper.startswith(prefix) for prefix in _DEF_PREFIXES):
        return "defense"
    return "offense"


def _concepts(frames: list[FrameObs]) -> str:
    blob = " ".join(t for frame in frames for t in frame.texts).lower()
    found = [word for word in _ROUTE_WORDS if re_word(word, blob)]
    if len(found) < 2:
        return ""
    return " ".join(found[:4])


def re_word(word: str, blob: str) -> bool:
    import re

    return re.search(rf"\b{re.escape(word)}\b", blob) is not None


def _field_zone(yardline: int | None, distance: int | None) -> str:
    if yardline is None:
        return ""
    to_goal = 100 - yardline
    if to_goal <= 5 or (distance is not None and distance >= to_goal and to_goal <= 10):
        return "gl"
    if to_goal <= 20:
        return "rz"
    return "open"


def build_snaps(
    frames: list[FrameObs],
    *,
    video_id: str,
    channel: str,
    game: str,
    extract: ExtractFlags | None = None,
) -> list[SnapRow]:
    extract = extract or ExtractFlags()
    frames = smooth_down_distance(frames)
    runs = _runs(frames)
    calls = _attach_play_calls(runs, frames)
    backward = _tags_from_previous(runs, frames)
    # Flags that show up just after a play (sack bug, incomplete banner)
    # belong to the play that just ended.
    rows: list[SnapRow] = []
    for i, run in enumerate(runs):
        dd = run.majority_dd()
        call_frames = calls.get(i) or []
        # A play-call screen lists the whole tab. One OCR frame often catches
        # only one of those names. Pool every hit in the window and keep a
        # name only when the window agrees on exactly one.
        formation = _agreed(call_frames, "formation_hits") or _agreed(run.frames, "formation_hits")
        play = _agreed_text(call_frames, "play") or _agreed_text(run.frames, "play")
        coverage = _agreed(call_frames, "coverage_hits") or _agreed(run.frames, "coverage_hits")
        back_def, back_def_conf = "", 0.0
        back_off, back_off_conf = "", 0.0
        if run.frames[0].kind == "kickoff":
            formation, play, coverage = "", "", ""
        else:
            back_play, back_cov, back_def, back_def_conf, back_off, back_off_conf = backward.get(
                i, ("", "", "", 0.0, "", 0.0)
            )
            if not play and back_play:
                play = back_play
            if not coverage and back_cov:
                coverage = back_cov
            if extract.read_offense_formation and not formation and back_off:
                formation = back_off
        quarter, clock = _clock_of(run.frames, last=False)
        yardline = next((f.yardline for f in run.frames if f.yardline is not None), None)
        yard_phrase = next((f.yard_phrase for f in run.frames if f.yard_phrase and f.yardline is not None), "")
        if not yard_phrase:
            yard_phrase = next((f.yard_phrase for f in run.frames if f.yard_phrase), "")
            if yard_phrase.startswith("ball on"):
                yardline = None

        kind = ""
        yards: int | None = None
        conf = 0.0
        if i + 1 < len(runs):
            nxt = runs[i + 1]
            gap_frames = [f for f in frames if run.t_end < f.t < nxt.t_start]
            flags: set[str] = set()
            for frame in list(run.frames[-2:]) + gap_frames:
                flags |= frame.flags
            # The sack / incomplete glyph usually prints on the next down's
            # scorebug. A touchdown banner on that next snap is a different play.
            for frame in nxt.frames[:2]:
                if "sack" in frame.flags:
                    flags.add("sack")
                if "incomplete" in frame.flags:
                    flags.add("incomplete")
            kind, yards, conf = _infer_between(run, nxt, flags)
        else:
            flags = set()
            for frame in run.frames:
                flags |= frame.flags
            if "sack" in flags:
                kind, yards, conf = "sack", None, 0.55
            elif "incomplete" in flags:
                kind, yards, conf = "incomplete", 0, 0.55
            elif "int" in flags:
                kind, yards, conf = "int", 0, 0.7
            elif "td" in flags:
                kind, yards, conf = "td", None, 0.6

        if run.frames[0].kind == "kickoff":
            play = play or "Kickoff"
            kind, yards, conf = "", None, 0.0

        def_form = ""
        def_conf = 0.0
        if extract.read_defense:
            live_name, live_conf = _unique_labeled(
                call_frames + list(run.frames), "def_formation", "def_formation_conf"
            )
            # The call screen and the previous-play strip are different snaps.
            # A disagreement is not a vote, so the field stays empty.
            if live_name and back_def and live_name != back_def:
                def_form, def_conf = "", 0.0
            elif live_name:
                def_form, def_conf = live_name, live_conf
            else:
                def_form, def_conf = back_def, back_def_conf
        score_row = _agree_score(run.frames) if extract.read_score else ScoreRead()
        snap_adjustments: list[str] = []
        snap_adj_conf = 0.0
        if extract.read_adjustments:
            snap_adjustments, snap_adj_conf = _agree_adjustments(call_frames + list(run.frames))
        down = dd[0] if dd else None
        distance = dd[1] if dd else None
        agree = 0.0
        if dd:
            agree = sum(1 for f in run.frames if f.dd == dd) / len(run.frames)
        clock_text = _fmt_clock(clock)
        sit = situation_raw(
            down=down,
            distance=distance,
            quarter=quarter,
            clock=clock_text,
            yardline_phrase=yard_phrase if yardline is not None else "",
        )
        channel_slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in channel).strip("-") or "vod"
        notes = (
            f"vod_pilot game={game} video={video_id} "
            f"t={run.t_start:.1f}-{run.t_end:.1f} "
            f"result_kind={kind or 'unknown'} "
            f"not logged to coach.db"
        )
        if yard_phrase and yardline is None:
            notes += f" | spot={yard_phrase} (not converted to 0-100)"
        rows.append(
            SnapRow(
                ts=f"vod:{video_id}:{run.t_start:.1f}",
                opponent_id=f"vod:{channel_slug}",
                side=_side(formation, coverage),
                down=down,
                distance=distance,
                yardline=yardline,
                quarter=quarter,
                situation_raw=sit,
                our_call="",
                formation=formation,
                play=play,
                macro="",
                result=result_text(kind=kind, yards=yards),
                coverage_seen=coverage,
                concept_seen=_concepts(call_frames + run.frames),
                notes=notes,
                session_id=f"vod:{video_id}",
                video_id=video_id,
                t_start=run.t_start,
                t_end=run.t_end,
                game=game,
                result_kind=kind,
                yards=yards,
                field_zone=_field_zone(yardline, distance),
                def_formation=def_form,
                team_left=score_row.team_left,
                team_right=score_row.team_right,
                score_left=score_row.score_left,
                score_right=score_row.score_right,
                record_left=score_row.record_left,
                record_right=score_row.record_right,
                adjustments=snap_adjustments,
                confidence=_confidence(
                    agree=agree,
                    formation=formation,
                    play=play,
                    coverage=coverage,
                    result=conf,
                    yardline=yardline,
                    extra=_extra_conf(
                        extract,
                        def_conf=def_conf,
                        score_conf=score_row.confidence,
                        formation_conf=_off_conf(
                            extract, formation, call_frames, run.frames, back_off, back_off_conf
                        ),
                        adjustment_conf=snap_adj_conf,
                    ),
                ),
            )
        )
    return rows


def _tags_from_previous(
    runs: list[_Run], frames: list[FrameObs]
) -> dict[int, tuple[str, str, str, float, str, float]]:
    """Play, coverage, and formations named on a PREVIOUS PLAY screen.

    Keyed by the snap that ended. A later screen must not fill the next snap.
    """
    out: dict[int, tuple[str, str, str, float, str, float]] = {}
    for frame in frames:
        if not (
            frame.previous_coverage
            or frame.previous_play
            or frame.previous_def_formation
            or frame.previous_off_formation
        ):
            continue
        best: int | None = None
        for i, run in enumerate(runs):
            if run.frames[0].kind == "kickoff":
                continue
            if run.t_end < frame.t <= run.t_end + 20:
                if best is None or run.t_end > runs[best].t_end:
                    best = i
        if best is None:
            continue
        play, cov, def_name, def_conf, off_name, off_conf = out.get(best, ("", "", "", 0.0, "", 0.0))
        if frame.previous_play and not play:
            play = frame.previous_play
        if frame.previous_coverage and not cov:
            cov = frame.previous_coverage
        if frame.previous_def_formation and not def_name:
            def_name = frame.previous_def_formation
            def_conf = frame.previous_def_conf
        if frame.previous_off_formation and not off_name:
            off_name = frame.previous_off_formation
            off_conf = frame.previous_off_conf
        out[best] = (play, cov, def_name, def_conf, off_name, off_conf)
    return out


def _unique_labeled(frames: list[FrameObs], attr: str, conf_attr: str) -> tuple[str, float]:
    """One agreed name and the weakest confidence among the frames that said it."""
    vals: list[tuple[str, float]] = []
    for frame in frames:
        name = str(getattr(frame, attr) or "")
        if name:
            vals.append((name, float(getattr(frame, conf_attr) or 0.0)))
    names = list(dict.fromkeys(name for name, _ in vals))
    if len(names) != 1:
        return "", 0.0
    confs = [conf for name, conf in vals if name == names[0] and conf > 0]
    return names[0], (min(confs) if confs else 0.0)


def _agree_score(frames: list[FrameObs]) -> ScoreRead:
    """The score at the snap. Conflicting reads are dropped, not averaged."""
    from collections import Counter

    reads = [
        frame
        for frame in frames
        if frame.team_left and frame.team_right and frame.score_left is not None and frame.score_right is not None
    ]
    if not reads:
        return ScoreRead()

    def key(frame: FrameObs) -> tuple:
        return (
            frame.team_left,
            frame.score_left,
            frame.team_right,
            frame.score_right,
            frame.record_left,
            frame.record_right,
        )

    counts = Counter(key(frame) for frame in reads)
    best, votes = counts.most_common(1)[0]
    if sum(1 for count in counts.values() if count == votes) > 1:
        return ScoreRead()
    if votes < 2 and len(reads) > 1:
        return ScoreRead()
    sample = next(frame for frame in reads if key(frame) == best)
    conf = min(frame.score_conf for frame in reads if key(frame) == best)
    return ScoreRead(
        team_left=sample.team_left,
        team_right=sample.team_right,
        score_left=sample.score_left,
        score_right=sample.score_right,
        record_left=sample.record_left,
        record_right=sample.record_right,
        confidence=conf,
    )


def _agree_adjustments(frames: list[FrameObs]) -> tuple[list[str], float]:
    labels: list[str] = []
    confs: list[float] = []
    for frame in frames:
        for label in frame.adjustments:
            if label not in labels:
                labels.append(label)
        if frame.adjustments and frame.adjustment_conf:
            confs.append(frame.adjustment_conf)
    if not labels:
        return [], 0.0
    return labels, min(confs) if confs else 0.0


def _off_conf(
    extract: ExtractFlags,
    formation: str,
    call_frames: list[FrameObs],
    run_frames: list[FrameObs],
    back_off: str,
    back_off_conf: float,
) -> float:
    if not extract.read_offense_formation or not formation:
        return 0.0
    confs = [
        frame.off_formation_conf
        for frame in call_frames + list(run_frames)
        if frame.formation == formation and frame.off_formation_conf
    ]
    if not confs and formation == back_off and back_off_conf:
        confs = [back_off_conf]
    return min(confs) if confs else 0.0


def _extra_conf(
    extract: ExtractFlags,
    *,
    def_conf: float = 0.0,
    score_conf: float = 0.0,
    formation_conf: float = 0.0,
    adjustment_conf: float = 0.0,
) -> dict[str, float] | None:
    extra: dict[str, float] = {}
    if extract.read_defense:
        extra["def_formation"] = def_conf
    if extract.read_score:
        extra["score"] = score_conf
    if extract.read_offense_formation and formation_conf:
        extra["formation"] = formation_conf
    if extract.read_adjustments:
        extra["adjustments"] = adjustment_conf
    return extra or None


def _confidence(
    *,
    agree: float,
    formation: str,
    play: str,
    coverage: str,
    result: float,
    yardline: int | None,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    out = {
        "down": round(agree, 2),
        "formation": 0.7 if formation else 0.0,
        "play": 0.6 if play and play != "Kickoff" else 0.0,
        "coverage": 0.65 if coverage else 0.0,
        "result": round(result, 2),
        "yardline": 0.7 if yardline is not None else 0.0,
    }
    if extra:
        for key, val in extra.items():
            out[key] = round(val, 2)
    return out


def _agreed(frames: list[FrameObs], attr: str) -> str:
    hits: list[str] = []
    for frame in frames:
        hits.extend(getattr(frame, attr) or [])
    unique = list(dict.fromkeys(hits))
    if len(unique) == 1:
        return unique[0]
    return ""


def _agreed_text(frames: list[FrameObs], attr: str) -> str:
    vals = [str(getattr(frame, attr)) for frame in frames if getattr(frame, attr)]
    unique = list(dict.fromkeys(vals))
    if len(unique) == 1:
        return unique[0]
    return ""


def _first(frames: list[FrameObs], attr: str) -> str:
    for frame in frames:
        val = getattr(frame, attr)
        if val:
            return str(val)
    return ""


def _fmt_clock(seconds: int | None) -> str:
    if seconds is None:
        return ""
    return f"{seconds // 60}:{seconds % 60:02d}"
