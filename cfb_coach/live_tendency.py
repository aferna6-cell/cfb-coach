"""This-game live tendency engine (Milestone 2) — MOST IMPORTANT.

Sample tiers: 1=log, 2=mild, 3+=actionable, 5+=strong.
Full-game + recent windows (last 5 / last 8). Anti-whiplash preserved.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from cfb_coach.vision.play_family import family_to_concept_hint
from cfb_coach.vision.play_record import PlayRecord


SAMPLE_LOG = 1
SAMPLE_MILD = 2
SAMPLE_ACTIONABLE = 3
SAMPLE_STRONG = 5

DEFAULT_EXPLOSIVE_RUN = 10
DEFAULT_EXPLOSIVE_PASS = 15


@dataclass
class LiveTendency:
    signal: str
    sample_size: int
    hit_rate: float
    confidence: str  # log|mild|actionable|strong
    recent_rate: float
    bucket: str
    side: str = ""  # offense_when_opp | defense_when_we | either
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        return self.sample_size >= SAMPLE_ACTIONABLE and self.confidence in (
            "actionable",
            "strong",
        )


def _tier(n: int) -> str:
    if n >= SAMPLE_STRONG:
        return "strong"
    if n >= SAMPLE_ACTIONABLE:
        return "actionable"
    if n >= SAMPLE_MILD:
        return "mild"
    if n >= SAMPLE_LOG:
        return "log"
    return "none"


def _dist_bucket(distance: int | None) -> str:
    if distance is None:
        return "unk"
    if distance <= 2:
        return "short"
    if distance <= 6:
        return "med"
    return "long"


def contextual_bucket(rec: PlayRecord) -> str:
    """down|dist|zone|quarter|score|formation|shell|pressure"""
    parts = [
        f"d{rec.down or 'x'}",
        _dist_bucket(rec.distance),
        rec.field_zone or "open",
        f"q{rec.quarter or 'x'}",
        rec.score_state(),
        (rec.formation or "unk").lower(),
        (rec.shell or "unk").lower(),
        (rec.pressure or "unk").lower(),
    ]
    return "|".join(parts)


@dataclass
class Alert:
    kind: str  # TENDENCY_CONFIRMED | SHIFT | QB_ESCAPE
    message: str
    signal: str
    ts: float = field(default_factory=time.time)


@dataclass
class CounterValidation:
    signal: str
    macro_or_call: str
    window: int
    successes: int
    attempts: int

    @property
    def rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0


@dataclass
class LiveTendencyEngine:
    """Accumulate PlayRecords → LiveTendency list + alerts."""

    explosive_run: int = DEFAULT_EXPLOSIVE_RUN
    explosive_pass: int = DEFAULT_EXPLOSIVE_PASS
    recent_windows: tuple[int, ...] = (5, 8)
    alert_cooldown_s: float = 45.0

    plays: list[PlayRecord] = field(default_factory=list)
    _alerts: list[Alert] = field(default_factory=list)
    _last_alert: dict[str, float] = field(default_factory=dict)
    _counters: list[dict[str, Any]] = field(default_factory=list)
    _prev_top: dict[str, str] = field(default_factory=dict)

    def add_play(self, rec: PlayRecord) -> list[Alert]:
        self.plays.append(rec)
        return self._maybe_alerts(rec)

    def correct_play(self, play_id: str, **fields: Any) -> None:
        for i, p in enumerate(self.plays):
            if p.play_id == play_id:
                for k, v in fields.items():
                    if hasattr(p, k) and v is not None:
                        setattr(p, k, v)
                p.corrected = True
                self.plays[i] = p
                return

    def tendencies_for_context(
        self,
        *,
        down: int | None = None,
        distance: int | None = None,
        field_zone: str = "",
        formation: str = "",
        shell: str = "",
        pressure: str = "",
        side: str = "offense",
        limit: int = 8,
    ) -> list[LiveTendency]:
        """LiveTendency list for current pre-snap context (feed playcaller)."""
        all_t = self.compute(side=side)
        # Prefer matching bucket dimensions
        scored: list[tuple[float, LiveTendency]] = []
        for t in all_t:
            score = 0.0
            b = t.bucket
            if down is not None and f"d{down}" in b:
                score += 2
            if distance is not None and _dist_bucket(distance) in b:
                score += 2
            if field_zone and field_zone in b:
                score += 1.5
            if formation and formation.lower() in b:
                score += 1.5
            if shell and shell.lower() in b:
                score += 1
            if pressure and pressure.lower() in b:
                score += 1
            # Always keep global signals
            if t.bucket.startswith("global|") or t.bucket.startswith("recent"):
                score += 0.5
            if t.actionable:
                score += 1
            scored.append((score, t))
        scored.sort(key=lambda x: (-x[0], -x[1].sample_size, -x[1].hit_rate))
        out: list[LiveTendency] = []
        seen = set()
        for _, t in scored:
            key = (t.signal, t.bucket.split("|")[0])
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
            if len(out) >= limit:
                break
        return out

    def compute(self, *, side: str | None = None) -> list[LiveTendency]:
        plays = self.plays
        if side:
            # offense-when-opp-has-ball ⇒ our defense snaps watching their O
            # defense-when-we-have-ball ⇒ our offense snaps watching their D
            if side == "defense":
                plays = [p for p in self.plays if p.side == "defense"]
            elif side == "offense":
                plays = [p for p in self.plays if p.side == "offense"]

        out: list[LiveTendency] = []
        out.extend(self._family_tendencies(plays, scope="global"))
        for w in self.recent_windows:
            recent = plays[-w:] if len(plays) >= 1 else []
            out.extend(self._family_tendencies(recent, scope=f"recent{w}"))
        out.extend(self._shell_tendencies(plays))
        out.extend(self._scramble_tendencies(plays))
        out.extend(self._explosive_tendencies(plays))
        out.extend(self._motion_tendencies(plays))
        out.extend(self._context_family(plays))
        return out

    def top_lines(self, *, n: int = 3, side: str = "offense") -> list[str]:
        ts = [t for t in self.tendencies_for_context(side=side, limit=12) if t.sample_size >= SAMPLE_MILD]
        lines = []
        for t in ts[:n]:
            pct = int(round(t.hit_rate * 100))
            lines.append(f"{t.signal} {t.sample_size}× ({pct}%) [{t.confidence}]")
        return lines

    def current_counter(self, *, side: str = "offense") -> str:
        """Short CURRENT COUNTER suggestion from strongest actionable tendency."""
        ts = [t for t in self.tendencies_for_context(side=side, limit=12) if t.actionable]
        ts = _prefer_specific(ts)
        if not ts:
            mild = [t for t in self.tendencies_for_context(side=side, limit=12) if t.sample_size >= SAMPLE_MILD]
            mild = _prefer_specific(mild)
            if not mild:
                return "base situational"
            t = mild[0]
            return f"soft vs {t.signal} (n={t.sample_size})"
        t = ts[0]
        return _counter_phrase(t)

    def register_counter(self, signal: str, macro_or_call: str) -> None:
        self._counters.append(
            {
                "signal": signal,
                "macro_or_call": macro_or_call,
                "after_idx": len(self.plays),
            }
        )

    def validate_counters(self, *, window: int = 3) -> list[CounterValidation]:
        """Did recommended macro/call succeed next N snaps?"""
        out: list[CounterValidation] = []
        for c in self._counters:
            start = int(c["after_idx"])
            chunk = self.plays[start : start + window]
            if not chunk:
                continue
            succ = 0
            for p in chunk:
                if _play_success(p):
                    succ += 1
            out.append(
                CounterValidation(
                    signal=str(c["signal"]),
                    macro_or_call=str(c["macro_or_call"]),
                    window=window,
                    successes=succ,
                    attempts=len(chunk),
                )
            )
        return out

    def _maybe_alerts(self, rec: PlayRecord) -> list[Alert]:
        new: list[Alert] = []
        now = time.time()
        ts = self.compute()
        # Confirmed
        for t in ts:
            if t.sample_size >= SAMPLE_ACTIONABLE and t.hit_rate >= 0.5:
                key = f"CONF:{t.signal}"
                if now - self._last_alert.get(key, 0) >= self.alert_cooldown_s:
                    msg = f"TENDENCY CONFIRMED: {t.signal} ({t.sample_size}×, {int(t.hit_rate*100)}%)"
                    a = Alert("TENDENCY_CONFIRMED", msg, t.signal, now)
                    new.append(a)
                    self._alerts.append(a)
                    self._last_alert[key] = now
        # Shift — recent rate diverges from full-game
        by_sig: dict[str, list[LiveTendency]] = defaultdict(list)
        for t in ts:
            by_sig[t.signal].append(t)
        for sig, group in by_sig.items():
            glob = next((g for g in group if g.bucket.startswith("global")), None)
            rec5 = next((g for g in group if g.bucket.startswith("recent5")), None)
            if glob and rec5 and glob.sample_size >= 4 and rec5.sample_size >= 3:
                if abs(rec5.recent_rate - glob.hit_rate) >= 0.35:
                    key = f"SHIFT:{sig}"
                    if now - self._last_alert.get(key, 0) >= self.alert_cooldown_s:
                        msg = f"TENDENCY SHIFT: {sig} recent={int(rec5.recent_rate*100)}% vs game={int(glob.hit_rate*100)}%"
                        a = Alert("SHIFT", msg, sig, now)
                        new.append(a)
                        self._alerts.append(a)
                        self._last_alert[key] = now
        # QB escape
        if rec.play_family == "SCRAMBLE" or rec.result_type == "scramble":
            scrams = [p for p in self.plays if p.play_family == "SCRAMBLE" or p.result_type == "scramble"]
            if len(scrams) >= SAMPLE_ACTIONABLE:
                key = "QB_ESCAPE"
                if now - self._last_alert.get(key, 0) >= self.alert_cooldown_s:
                    dirs = [p.scramble_dir or "unknown" for p in scrams]
                    msg = f"QB ESCAPE: {len(scrams)}× scramble (dirs={dirs[-3:]})"
                    a = Alert("QB_ESCAPE", msg, "SCRAMBLE", now)
                    new.append(a)
                    self._alerts.append(a)
                    self._last_alert[key] = now
        return new

    def _family_tendencies(self, plays: Iterable[PlayRecord], *, scope: str) -> list[LiveTendency]:
        plays = list(plays)
        if not plays:
            return []
        counts: dict[str, int] = defaultdict(int)
        for p in plays:
            sig = p.play_family if p.play_family != "UNKNOWN" else ""
            if not sig and p.concept_tags:
                sig = p.concept_tags[0]
            if not sig:
                hint = family_to_concept_hint(p.play_family, p.concept_tags)
                sig = hint.upper().replace(" ", "_") if hint else ""
            if sig:
                counts[sig] += 1
            for tag in p.concept_tags:
                counts[tag] += 1
        n = len(plays)
        out = []
        for sig, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            rate = c / n
            out.append(
                LiveTendency(
                    signal=sig,
                    sample_size=c,
                    hit_rate=rate,
                    confidence=_tier(c),
                    recent_rate=rate,
                    bucket=f"{scope}|family",
                    details={"n_plays": n},
                )
            )
        return out

    def _context_family(self, plays: list[PlayRecord]) -> list[LiveTendency]:
        buckets: dict[str, list[PlayRecord]] = defaultdict(list)
        for p in plays:
            buckets[contextual_bucket(p)].append(p)
        out = []
        for b, group in buckets.items():
            if len(group) < SAMPLE_MILD:
                continue
            fam_counts: dict[str, int] = defaultdict(int)
            for p in group:
                if p.play_family and p.play_family != "UNKNOWN":
                    fam_counts[p.play_family] += 1
            for sig, c in fam_counts.items():
                out.append(
                    LiveTendency(
                        signal=sig,
                        sample_size=c,
                        hit_rate=c / len(group),
                        confidence=_tier(c),
                        recent_rate=c / len(group),
                        bucket=b,
                    )
                )
        return out

    def _shell_tendencies(self, plays: list[PlayRecord]) -> list[LiveTendency]:
        counts: dict[str, int] = defaultdict(int)
        for p in plays:
            if p.shell and p.shell != "unknown":
                counts[f"SHELL:{p.shell}"] += 1
            if p.pressure and p.pressure not in ("unknown", "none"):
                counts[f"PRESSURE:{p.pressure}"] += 1
        n = max(1, len(plays))
        return [
            LiveTendency(
                signal=sig,
                sample_size=c,
                hit_rate=c / n,
                confidence=_tier(c),
                recent_rate=c / n,
                bucket="global|shell",
            )
            for sig, c in counts.items()
        ]

    def _scramble_tendencies(self, plays: list[PlayRecord]) -> list[LiveTendency]:
        scrams = [p for p in plays if p.play_family == "SCRAMBLE" or p.result_type == "scramble"]
        if not scrams:
            return []
        dirs: dict[str, int] = defaultdict(int)
        for p in scrams:
            dirs[p.scramble_dir or "unknown"] += 1
        out = [
            LiveTendency(
                signal="SCRAMBLE",
                sample_size=len(scrams),
                hit_rate=len(scrams) / max(1, len(plays)),
                confidence=_tier(len(scrams)),
                recent_rate=len(scrams) / max(1, len(plays[-8:])),
                bucket="global|scramble",
                details={"dirs": dict(dirs)},
            )
        ]
        for d, c in dirs.items():
            out.append(
                LiveTendency(
                    signal=f"SCRAMBLE_{d}",
                    sample_size=c,
                    hit_rate=c / len(scrams),
                    confidence=_tier(c),
                    recent_rate=c / len(scrams),
                    bucket="global|scramble_dir",
                )
            )
        return out

    def _explosive_tendencies(self, plays: list[PlayRecord]) -> list[LiveTendency]:
        out = []
        expl = [p for p in plays if p.explosive or _explosive(p, self.explosive_run, self.explosive_pass)]
        if expl:
            out.append(
                LiveTendency(
                    signal="EXPLOSIVE",
                    sample_size=len(expl),
                    hit_rate=len(expl) / max(1, len(plays)),
                    confidence=_tier(len(expl)),
                    recent_rate=len([p for p in plays[-8:] if p in expl or p.explosive]) / max(1, min(8, len(plays))),
                    bucket="global|explosive",
                )
            )
        return out

    def _motion_tendencies(self, plays: list[PlayRecord]) -> list[LiveTendency]:
        known = [p for p in plays if p.motion_present is not None or p.motion_direction not in ("", "UNKNOWN")]
        if not known:
            return [
                LiveTendency(
                    signal="NO_MOTION",
                    sample_size=0,
                    hit_rate=0.0,
                    confidence="none",
                    recent_rate=0.0,
                    bucket="global|motion",
                    details={"note": "UNKNOWN"},
                )
            ]
        with_m = sum(1 for p in known if p.motion_present or p.motion_direction in ("L", "R"))
        out = [
            LiveTendency(
                signal="MOTION",
                sample_size=with_m,
                hit_rate=with_m / len(known),
                confidence=_tier(with_m),
                recent_rate=with_m / len(known),
                bucket="global|motion",
            )
        ]
        for d in ("L", "R", "NO_MOTION"):
            c = sum(1 for p in known if p.motion_direction == d)
            if c:
                out.append(
                    LiveTendency(
                        signal=f"MOTION_{d}",
                        sample_size=c,
                        hit_rate=c / len(known),
                        confidence=_tier(c),
                        recent_rate=c / len(known),
                        bucket="global|motion_dir",
                    )
                )
        return out

    def summary_report(self) -> str:
        lines = ["# LIVE TENDENCY REPORT", f"Plays: {len(self.plays)}", ""]
        ts = self.compute()
        actionable = [t for t in ts if t.actionable]
        mild = [t for t in ts if t.confidence == "mild"]
        lines.append("## Actionable")
        if not actionable:
            lines.append("  (none yet — need 3+ samples)")
        for t in actionable[:12]:
            lines.append(
                f"  - {t.signal}: n={t.sample_size} rate={t.hit_rate:.0%} "
                f"recent={t.recent_rate:.0%} bucket={t.bucket}"
            )
        lines.append("")
        lines.append("## Mild (n=2)")
        if not mild:
            lines.append("  (none)")
        for t in mild[:8]:
            lines.append(f"  - {t.signal}: n={t.sample_size} rate={t.hit_rate:.0%}")
        cvs = self.validate_counters()
        if cvs:
            lines.append("")
            lines.append("## Counter validation")
            for c in cvs:
                lines.append(
                    f"  - {c.macro_or_call} vs {c.signal}: "
                    f"{c.successes}/{c.attempts} ({c.rate:.0%})"
                )
        return "\n".join(lines)



_SPECIFIC_TAGS = frozenset({
    "CROSSERS", "VERTICALS", "FLOOD", "MESH", "SPACING", "SLANTS",
    "SCREEN", "SCRAMBLE", "RPO", "SACK", "QB_RUN",
})


def _prefer_specific(ts: list[LiveTendency]) -> list[LiveTendency]:
    """Prefer concept-tag signals over broad play families for COUNTER line."""
    if not ts:
        return ts
    specific = [t for t in ts if t.signal in _SPECIFIC_TAGS or t.signal.startswith("SCRAMBLE_")]
    if specific:
        return sorted(specific, key=lambda t: (-t.sample_size, -t.hit_rate))
    fam = [t for t in ts if not t.signal.startswith(("SHELL:", "PRESSURE:", "MOTION", "EXPLOSIVE"))]
    return fam or ts



def _explosive(p: PlayRecord, run_n: int, pass_n: int) -> bool:
    if p.yards is None:
        return False
    if p.play_family in ("RUN_INSIDE", "RUN_OUTSIDE", "QB_RUN", "SCRAMBLE", "RPO"):
        return p.yards >= run_n
    return p.yards >= pass_n


def _play_success(p: PlayRecord) -> bool:
    if p.result_type in ("td", "completion"):
        return True
    if p.yards is not None and p.yards >= 4:
        return True
    if p.result_type in ("int", "sack", "turnover", "incompletion"):
        return False
    return False


def _counter_phrase(t: LiveTendency) -> str:
    sig = t.signal.upper()
    if "CROSSER" in sig or sig == "CROSSERS":
        return "CROSS / Tampa-Quarters"
    if "VERTICAL" in sig:
        return "VERT / Quarters"
    if "MESH" in sig or "BUNCH" in sig:
        return "BUNCH / Tampa"
    if "RPO" in sig:
        return "RPO flats"
    if "RUN_INSIDE" in sig or "INSIDE" in sig:
        return "fit C3 / RUN-IN if short"
    if "RUN_OUTSIDE" in sig:
        return "RUN-OUT"
    if "SCRAMBLE" in sig:
        return "SCRAM + contain"
    if "SHELL:TWO_HIGH" in sig or "TWO_HIGH" in sig:
        return "run/easy vs two-high"
    if "SHELL:ONE_HIGH" in sig:
        return "flood/cross check"
    if t.signal == "DROPBACK_PASS":
        return "Quarters/Tampa soft"
    return f"lean vs {t.signal}"
