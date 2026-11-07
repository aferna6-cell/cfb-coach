"""CPU-only pilot harness for the structure-aware offense. Inert unless explicitly approved.

* sidecar: nothing here is imported by the live caller, and it never writes the game DB;
* human-vs-CPU only: any other ``opponent_type`` is refused;
* primary = structural model; the existing coordinator stays available as the baseline and as the
  fallback ONLY on a technical failure (exception / invalid artifact), never as a routine override;
* only installed plays; only registry-validated adjustments (``decide`` enforces); NO ADJUSTMENT preserved;
* every recommendation is logged with scores and a unique id; a recommendation is NOT an execution:
  executions are separate ``execution`` records the user confirms.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_presnap_advice as advice
from cfb_coach.madden.model import offense_structural_model as sm

CONFIG_SCHEMA = "madden27.offense_pilot.config.v1"
LOG_SCHEMA = "madden27.offense_pilot.log.v1"
LIVE_DIR = Path.home() / ".cfb-coach"


class PilotRefused(RuntimeError):
    """The pilot declined to run (unapproved, wrong opponent type, bad artifact, unsafe path)."""


@dataclass
class PilotConfig:
    schema: str = CONFIG_SCHEMA
    mode: str = "shadow"                      # shadow = recommend+log only | pilot = primary for CPU games
    approved: bool = False                    # flipped ONLY by the explicit `approve` command
    approved_by: str = ""
    approved_at: str = ""
    opponent_scope: str = "cpu"
    artifact_path: str = ""
    artifact_digest: str = ""
    assignment_digest: str = ""
    installed_fingerprint: str = oa.FINGERPRINT
    log_path: str = ""
    loadout_path: str = ""
    confirmations_path: str = ""
    baseline_available: bool = True
    allow_adjustments: bool = True
    notes: str = "Inert until approved. Does not modify the installed playbook or the live caller."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "PilotConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema") != CONFIG_SCHEMA:
            raise PilotRefused("unknown pilot config schema")
        return cls(**{k: raw[k] for k in cls().__dict__ if k in raw})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True), encoding="utf-8")


def _outside_live(p: str | Path) -> bool:
    r = Path(p).expanduser().resolve()
    live = LIVE_DIR.resolve()
    return not (r == live or live in r.parents)


def validate_config(cfg: PilotConfig, *, artifact: sm.StructuralArtifact | None = None) -> list[str]:
    """Every reason this pilot must not start (empty = ready). Approval is checked separately for pilot mode."""
    errs: list[str] = []
    if cfg.mode not in ("shadow", "pilot"):
        errs.append(f"mode must be shadow or pilot, got {cfg.mode!r}")
    if cfg.opponent_scope != "cpu":
        errs.append("pilot scope is CPU only")
    if cfg.installed_fingerprint != oa.FINGERPRINT:
        errs.append(f"installed fingerprint {cfg.installed_fingerprint} != {oa.FINGERPRINT}")
    for label, path in (("artifact", cfg.artifact_path), ("log", cfg.log_path)):
        if not path:
            errs.append(f"{label}_path not set")
        elif not _outside_live(path):
            errs.append(f"{label}_path is inside the live data directory {LIVE_DIR}")
    if cfg.artifact_path and artifact is None:
        try:
            artifact = sm.load_artifact(cfg.artifact_path)
        except Exception as exc:  # noqa: BLE001
            errs.append(f"artifact unreadable: {exc}")
    if artifact is not None:
        errs += [f"artifact: {e}" for e in sm.validate_artifact(artifact)]
        if cfg.artifact_digest and cfg.artifact_digest != sm.artifact_digest(artifact):
            errs.append("artifact digest differs from the one recorded in the config")
    if cfg.mode == "pilot" and not cfg.approved:
        errs.append("pilot mode requires explicit approval (run the `approve` command)")
    if cfg.mode == "pilot" and cfg.approved and not (cfg.approved_by and cfg.approved_at):
        errs.append("approval record incomplete")
    return errs


def approve(cfg: PilotConfig, by: str) -> PilotConfig:
    if not by.strip():
        raise PilotRefused("approval needs a name")
    cfg.approved, cfg.approved_by = True, by.strip()
    cfg.approved_at = datetime.now(timezone.utc).isoformat()
    return cfg


def _baseline_rank(artifact: sm.StructuralArtifact, candidates: Sequence[tuple[str, str]],
                   sit: od.Situation) -> list[dict[str, Any]]:
    """The existing coordinator's ranking (hierarchical shrinkage), with no structural features."""
    return em.rank_candidates(artifact.base, list(candidates), down=sit.down, distance=sit.distance,
                              yardline=sit.yardline, opponent_id=sit.opponent_id, opponent_type=sit.opponent_type)


class PilotSession:
    def __init__(self, cfg: PilotConfig, *, artifact: sm.StructuralArtifact | None = None,
                 loadout: ex.Loadout | None = None, confirmations: Mapping[str, Any] | None = None,
                 candidates: Sequence[tuple[str, str]] | None = None):
        problems = validate_config(cfg, artifact=artifact)
        if problems:
            raise PilotRefused("; ".join(problems))
        self.cfg = cfg
        self.artifact = artifact or sm.load_artifact(cfg.artifact_path)
        self.loadout = loadout
        self.confirmations = confirmations or {}
        book = [(p["formation"], p["play"]) for p in oa.load_effective()["plays"]]
        self.candidates = list(candidates) if candidates is not None else book
        bad = [c for c in self.candidates if c not in set(book)]
        if bad:
            raise PilotRefused(f"candidates not in the installed offense: {bad[:3]}")
        self.log_path = Path(cfg.log_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.recent: list[tuple[str, str]] = []

    def _append(self, rec: Mapping[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"schema": LOG_SCHEMA, **rec}, sort_keys=True, default=str) + "\n")

    def recommend(self, situation: Mapping[str, Any], *, game_id: str = "",
                  coverage_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        sit = od.Situation.from_mapping({k: v for k, v in situation.items() if k != "game_id"})
        if (sit.opponent_type or "").lower() != "cpu":
            raise PilotRefused("pilot is human-vs-CPU only; opponent_type must be 'cpu'")
        t0 = time.perf_counter()
        fallback = ""
        try:
            dec = od.decide(self.artifact, sit, self.candidates, loadout=self.loadout, confirmations=self.confirmations,
                            mode="live", allow_adjustments=self.cfg.allow_adjustments, recent_calls=self.recent[::-1])
            card = advice.build_card(dec, sit.mix or sm.pre_snap_mix(self.artifact, sit.opponent_id)[0])
            primary = {"formation": dec.formation, "play": dec.play, "adjustment": dec.adjustment}
            detail = dec.as_log()
        except Exception as exc:  # noqa: BLE001 - technical failure only
            fallback = f"{type(exc).__name__}: {exc}"
            top = _baseline_rank(self.artifact, self.candidates, sit)[0]
            primary = {"formation": top["formation"], "play": top["play"], "adjustment": od.NO_ADJUSTMENT}
            card = {"FORMATION": top["formation"], "PLAY": top["play"], "ADJUSTMENT": od.NO_ADJUSTMENT,
                    "NOTE": "baseline coordinator used because the structural model failed technically"}
            detail = {"error": fallback}
        base = _baseline_rank(self.artifact, self.candidates, sit)[:5] if self.cfg.baseline_available else []
        rec_id = uuid.uuid4().hex[:12]
        record = {
            "type": "recommendation", "rec_id": rec_id, "ts": datetime.now(timezone.utc).isoformat(),
            "game_id": game_id, "mode": self.cfg.mode, "situation": {k: v for k, v in situation.items() if k != "mix"},
            "primary": primary, "decision": detail, "baseline_top5": [
                {"formation": b["formation"], "play": b["play"], "probability": round(b["probability"], 4)} for b in base],
            "baseline_agrees": bool(base) and (base[0]["formation"], base[0]["play"]) == (primary["formation"], primary["play"]),
            "fallback_reason": fallback, "artifact_digest": sm.artifact_digest(self.artifact),
            "assignment_digest": self.artifact.assignment_digest, "executed": "not_confirmed",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "card": card,
            "coverage_history": dict(coverage_context or {}),
        }
        self._append(record)
        self.recent.append((primary["formation"], primary["play"]))
        return record

    def recommend_from_logged_game(self, conn: Any, situation: Mapping[str, Any], *,
                                  session_id: str, next_snap_seq: int,
                                  game_id: str = "") -> dict[str, Any]:
        """Read previously CLOSED user-entered coverage without any extra logging.

        The prior is the already trained model's opponent/global coverage mix.
        Only previous snaps of this SAME opponent and game contribute to the
        current-game adjustment. This method is opt-in for a CPU-only pilot.
        """
        from cfb_coach.madden.model import offense_live_coverage as live_cov
        opp = str(situation.get("opponent_id") or "")
        prior, _ = sm.pre_snap_mix(self.artifact, opp)
        context = live_cov.coverage_for_next_snap(
            conn, session_id=session_id, opponent_id=opp,
            next_snap_seq=next_snap_seq, prior_mix=prior,
        )
        updated = dict(situation)
        updated["mix"] = context.mix
        return self.recommend(updated, game_id=game_id or session_id,
                              coverage_context=context.as_log())

    def confirm_execution(self, rec_id: str, *, formation: str, play: str, adjustment: str = od.NO_ADJUSTMENT,
                          status: str = "executed") -> dict[str, Any]:
        if status not in ("executed", "not_executed", "unknown"):
            raise PilotRefused("status must be executed | not_executed | unknown")
        known = {r["rec_id"] for r in read_log(self.log_path) if r["type"] == "recommendation"}
        if rec_id not in known:
            raise PilotRefused(f"no recommendation {rec_id!r} in this log")
        rec = {"type": "execution", "rec_id": rec_id, "ts": datetime.now(timezone.utc).isoformat(),
               "formation": formation, "play": play, "adjustment": adjustment, "status": status}
        self._append(rec)
        return rec


def read_log(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_log(path: str | Path) -> dict[str, Any]:
    rows = read_log(path)
    recs = {r["rec_id"]: r for r in rows if r["type"] == "recommendation"}
    exe: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["type"] == "execution":
            exe[r["rec_id"]] = r          # latest confirmation wins
    executed = [e for e in exe.values() if e["status"] == "executed"]
    followed = [e for e in executed if (e["formation"], e["play"]) == (recs[e["rec_id"]]["primary"]["formation"],
                                                                      recs[e["rec_id"]]["primary"]["play"])]
    lat = sorted(r["latency_ms"] for r in recs.values())
    return {
        "recommendations": len(recs), "executions_confirmed": len(executed),
        "unconfirmed": len(recs) - len(exe),
        "followed_recommended_play": len(followed),
        "adjusted": sum(1 for r in recs.values() if r["primary"]["adjustment"] != od.NO_ADJUSTMENT),
        "fallbacks": sum(1 for r in recs.values() if r["fallback_reason"]),
        "baseline_agreement_rate": (sum(r["baseline_agrees"] for r in recs.values()) / len(recs)) if recs else None,
        "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))] if lat else None,
        "note": "a recommendation is not an execution; only confirmed executions may enter training",
    }
