"""Executable Madden 27 (Xbox) adjustments: registry, loadout, validation, candidates, coaching cards.

Research and shadow only: nothing here is read by the live caller.

An adjustment may only be recommended when it is a real in-game action:
an entry button from EA's official Madden 27 Xbox pre-play controls, an
on-screen label to pick for any sub-menu EA does not spell out, or a saved
Custom Adjustment the user confirmed is installed, active and in a named slot.
No face-button mapping is invented for Custom Adjustment slots.

Readiness, strictest first:
  live_ready      EA-documented entry, every option confirmed by the user in-game
  needs_user_test EA-documented entry, but the user has not physically confirmed it
  not_installed   a preset or saved Custom Adjustment the user has not confirmed in a slot
  unsupported     no EA-documented input, or EA contradicts the old mapping
Live selection only ever uses ``live_ready`` actions and the unadjusted play.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DATA = Path(__file__).resolve().parents[2] / "data" / "madden27" / "executable_actions.json"
BUTTONS = frozenset({"A", "B", "X", "Y", "LB", "RB", "LT", "RT", "LS", "RS", "VIEW", "MENU",
                     "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT"})
STICKS = frozenset({"RS_LEFT_RIGHT", "LS_LEFT_RIGHT"})
_STEP = re.compile(r"^(tap|hold):([A-Z_]+)$|^(stick|hold_stick):([A-Z_]+)$|^icon:(receiver|defender)$|"
                   r"^onscreen:(\{[a-z_]+\}|[^{}]+)$|^slot:(\{slot\}|[^{}]+)$")

# Which EA entry function each category must start with (button meaning per EA's Xbox map).
ENTRY_FUNCTION = {
    ("offense", "hot_route"): "Hot Route", ("offense", "audible"): "Audible",
    ("offense", "protection"): "Pass Protection", ("offense", "motion"): "Switch Player",
    ("offense", "snap_control"): "Fake Snap", ("offense", "custom_adjustment_preset"): "Custom Adjustments",
    ("offense", "custom_adjustment_saved"): "Custom Adjustments",
    ("defense", "coverage_hub"): "Global Coverage", ("defense", "coverage_audible"): "Coverage Audible",
    ("defense", "individual_assignment"): "DEF Hot Route", ("defense", "front"): "DL Shift / LB Shift",
    ("defense", "keys"): "Defensive Keys", ("defense", "custom_adjustment_preset"): "Custom Adjustments",
    ("defense", "custom_adjustment_saved"): "Custom Adjustments",
}
GET_SET_SECONDS = 1.5
SNAP_MARGIN_SECONDS = 3.0
MIN_ADVANTAGE = 0.01
EVIDENCE_WEIGHT = {"measured": 1.0, "prior": 0.35, "hypothesis": 0.15}
CONFIDENCE_TEXT = {
    "measured": "MEASURED in your games",
    "prior": "LOW - coded football reasoning, not measured in Madden",
    "hypothesis": "VERY LOW - untested hypothesis",
}
STATUS_TEXT = {
    "live_ready": "LIVE-READY: EA-documented inputs, confirmed by you in-game",
    "needs_user_test": "NEEDS YOUR TEST: inputs from EA's Madden 27 controls, not yet confirmed by you",
    "not_installed": "NOT INSTALLED: not confirmed in an active slot for this game",
    "unsupported": "UNSUPPORTED: no verified Madden 27 input",
}

# The project's older research control table, checked against EA's Madden 27 Xbox map.
RESEARCH_CONTROL_AUDIT = {
    ("offense", "hot_route"): ("confirmed", "Y = Hot Route on EA's map"),
    ("offense", "audible"): ("confirmed", "X = Audible on EA's map"),
    ("offense", "pass_protection"): ("confirmed", "LT = Pass Protection on EA's map"),
    ("offense", "custom_adjustments"): ("confirmed", "LB = Custom Adjustments on EA's map"),
    ("offense", "fake_snap"): ("confirmed", "RB = Fake Snap on EA's map"),
    ("offense", "motion"): ("resolved", "EA's map: Motion Player = hold left stick left/right (the old table listed a conflict)"),
    ("offense", "flip_run"): ("not_on_ea_map", "no flip-run input on EA's map; right stick click is the Pre-Play Menu"),
    ("defense", "custom_adjustments"): ("confirmed", "LB = Custom Adjustments on EA's map"),
    ("defense", "coverage_hub"): ("confirmed", "LT = Global Coverage on EA's map"),
    ("defense", "press_back_off"): ("hub_only", "inside the LT hub per EA; the old 'left stick' step is not on EA's map"),
    ("defense", "shade"): ("hub_only", "inside the LT hub per EA (per-receiver shade)"),
    ("defense", "commit"): ("hub_only", "Pass Commit inside/outside is in the LT hub per EA"),
    ("defense", "zone_strategy"): ("hub_only", "Smart Zone strategy is in the LT hub per EA; the old 'D-pad' step is not"),
    ("defense", "dline"): ("partly", "right stick left/right = DL Shift / LB Shift on EA's map; the follow-up step is not documented"),
    ("defense", "lb"): ("partly", "right stick left/right = DL Shift / LB Shift on EA's map; the follow-up step is not documented"),
    ("defense", "stunt"): ("conflict", "EA puts stunts and twists on the play-call screen; the old 'RS left then RB' is not on EA's map"),
    ("defense", "show_blitz"): ("conflict", "RB = Defensive Keys on EA's map; no Show Blitz input"),
    ("defense", "qb_contain"): ("conflict", "old 'RB then LB' = Defensive Keys then Custom Adjustments on EA's map"),
    ("defense", "individual"): ("conflict", "old 'Y then defender' - on EA's map Y = Coverage Audible and A = DEF Hot Route"),
}


class AdjustmentError(ValueError):
    """A planned adjustment that cannot be executed as written."""


# ------------------------------------------------------------------ registry


@dataclass(frozen=True)
class ActionSpec:
    id: str
    name: str
    side: str
    category: str
    kind: str
    inputs: tuple[str, ...]
    inputs_grade: str
    requires: Mapping[str, Any]
    exclusive_groups: tuple[str, ...]
    touches: tuple[str, ...]
    est_seconds: float
    sources: tuple[str, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    effects: Mapping[str, Any] = field(default_factory=dict)
    effects_by_param: Mapping[str, Any] = field(default_factory=dict)
    option_groups: Mapping[str, str] = field(default_factory=dict)
    name_grade: str = "ea_official"
    blocked_reason: str = ""

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ActionSpec":
        return cls(
            id=d["id"], name=d["name"], side=d["side"], category=d["category"], kind=d["kind"],
            inputs=tuple(d.get("inputs") or ()), inputs_grade=d["inputs_grade"], requires=dict(d.get("requires") or {}),
            exclusive_groups=tuple(d.get("exclusive_groups") or ()), touches=tuple(d.get("touches") or ()),
            est_seconds=float(d.get("est_seconds") or 0), sources=tuple(d.get("sources") or ()),
            params=dict(d.get("params") or {}), effects=dict(d.get("effects") or {}),
            effects_by_param=dict(d.get("effects_by_param") or {}), option_groups=dict(d.get("option_groups") or {}),
            name_grade=d.get("name_grade", "ea_official"), blocked_reason=d.get("blocked_reason", ""),
        )


@lru_cache(maxsize=1)
def load_registry_data(path: str | None = None) -> dict[str, Any]:
    return json.loads(Path(path or DATA).read_text(encoding="utf-8"))


def registry() -> dict[str, ActionSpec]:
    return {a["id"]: ActionSpec.from_dict(a) for a in load_registry_data()["actions"]}


def find_by_name(side: str, name: str) -> ActionSpec:
    """Exact in-game name lookup. Unknown names are rejected, never fuzzy-matched."""
    for a in registry().values():
        if a.side == side and a.name == name:
            return a
    raise AdjustmentError(f"unknown Madden 27 {side} adjustment name: {name!r}")


def entry_buttons(side: str) -> Mapping[str, str]:
    return load_registry_data()["entry_buttons"][side]


def validate_inputs(side: str, category: str, steps: Sequence[str]) -> list[str]:
    """Problems with an Xbox input sequence (empty list = valid)."""
    errs: list[str] = []
    entry = entry_buttons(side)
    for i, s in enumerate(steps):
        m = _STEP.match(s)
        if not m:
            errs.append(f"step {i + 1} {s!r} is not a valid input")
            continue
        if m.group(1) and m.group(2) not in BUTTONS:
            errs.append(f"step {i + 1}: {m.group(2)!r} is not an Xbox button")
        if m.group(3) and m.group(4) not in STICKS:
            errs.append(f"step {i + 1}: {m.group(4)!r} is not a stick direction")
        if side == "offense" and s == "tap:A":
            errs.append(f"step {i + 1}: A snaps the ball on offense; it cannot be part of an adjustment")
    if steps:
        want = ENTRY_FUNCTION.get((side, category))
        first = _STEP.match(steps[0])
        if want and first:
            key = first.group(2) or ("LS_HOLD_LEFT_RIGHT" if first.group(4) == "LS_LEFT_RIGHT" else first.group(4))
            got = entry.get(key or "")
            if got != want:
                errs.append(f"first input {steps[0]!r} is {got or 'not mapped'} on EA's {side} map, not {want}")
    return errs


# ------------------------------------------------------------------ loadout and confirmations


@dataclass(frozen=True)
class LoadoutEntry:
    side: str
    name: str
    slot: str
    installed: bool
    active: bool
    confirmed_by_user: bool
    confirmed_at: str = ""
    preset_id: str | None = None


@dataclass(frozen=True)
class Loadout:
    """What the user confirmed in-game for this game. Software-generated designs never appear here."""

    entries: tuple[LoadoutEntry, ...] = ()
    game_label: str = ""
    active_limit_per_side_confirmed: bool = False

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Loadout":
        entries = []
        for side in ("offense", "defense"):
            for e in d.get(side) or []:
                entries.append(LoadoutEntry(
                    side=side, name=str(e["name"]), slot=str(e.get("slot") or ""),
                    installed=bool(e.get("installed")), active=bool(e.get("active")),
                    confirmed_by_user=bool(e.get("confirmed_by_user")), confirmed_at=str(e.get("confirmed_at") or ""),
                    preset_id=e.get("preset_id"),
                ))
        out = cls(tuple(entries), str(d.get("game_label") or ""), bool(d.get("active_limit_per_side_confirmed")))
        problems = out.problems()
        if problems:
            raise AdjustmentError("invalid loadout: " + "; ".join(problems))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Loadout":
        return cls.from_dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))

    def problems(self) -> list[str]:
        out = []
        rules = load_registry_data()["rules"]["saved_and_active_limits"]
        for side in ("offense", "defense"):
            rows = [e for e in self.entries if e.side == side]
            names = [e.name for e in rows]
            slots = [e.slot for e in rows if e.slot]
            if len(set(names)) != len(names):
                out.append(f"{side}: duplicate adjustment names")
            if len(set(slots)) != len(slots):
                out.append(f"{side}: two adjustments in the same slot")
            if sum(e.installed for e in rows) > rules["saved_per_side"]:
                out.append(f"{side}: more than {rules['saved_per_side']} saved adjustments")
            if sum(e.active for e in rows) > rules["active_per_game"]:
                out.append(f"{side}: more than {rules['active_per_game']} active adjustments")
            for e in rows:
                if e.active and not e.installed:
                    out.append(f"{side}: {e.name!r} is active but not installed")
                if e.preset_id and e.preset_id not in registry():
                    out.append(f"{side}: unknown preset id {e.preset_id!r}")
        total = sum(e.active for e in self.entries)
        if not self.active_limit_per_side_confirmed and total > rules["active_per_game"]:
            out.append(f"{total} active adjustments: EA's limit of {rules['active_per_game']} may be per game, "
                       "not per side; confirm in-game before using more")
        return out

    def usable(self, side: str, name: str) -> LoadoutEntry | None:
        for e in self.entries:
            if e.side == side and e.name == name and e.installed and e.active and e.confirmed_by_user and e.slot:
                return e
        return None


def confirmed(confirmations: Mapping[str, Any] | None, action_id: str, option: str | None = None) -> bool:
    c = dict(confirmations or {})
    key = f"{action_id}|{option}" if option else action_id
    rec = c.get(key)
    return bool(rec and (rec.get("confirmed") if isinstance(rec, Mapping) else rec))


def saved_action(entry: LoadoutEntry) -> ActionSpec:
    """A user-confirmed saved Custom Adjustment, as a registry action."""
    effects: dict[str, Any] = {"addresses": {}, "exposes": {}, "evidence": "hypothesis"}
    if entry.side == "defense":
        try:
            from cfb_coach.madden import research_db as rdb

            for m in rdb.defense_macros():
                if str(m.get("name")) == entry.name:
                    effects["addresses"] = {str(a): 0.15 for a in m.get("answers") or [] if a != "prevent"}
        except Exception:  # noqa: BLE001 - research file missing leaves effects unknown
            pass
    return ActionSpec(
        id=f"{entry.side[:3]}.saved.{entry.name}", name=entry.name, side=entry.side,
        category="custom_adjustment_saved", kind="saved_custom", inputs=("tap:LB", "slot:{slot}"),
        inputs_grade="ea_entry_documented", requires={"loadout": True},
        exclusive_groups=("custom_adjustment",) if entry.side == "defense" else ("routes:all",),
        touches=("coverage:all",) if entry.side == "defense" else ("route:all",), est_seconds=2,
        sources=("ea-deep-dive", "ea-controls-xb-off" if entry.side == "offense" else "ea-controls-xb-def"),
        effects=effects,
    )


def readiness(action: ActionSpec, option: str | None, *, loadout: Loadout | None,
              confirmations: Mapping[str, Any] | None) -> tuple[str, str]:
    if action.inputs_grade in ("undocumented", "conflict"):
        return "unsupported", action.blocked_reason or "no verified Madden 27 input"
    if action.kind in ("preset", "saved_custom"):
        if not (loadout and loadout.usable(action.side, action.name)):
            return "not_installed", f"{action.name!r} is not confirmed installed and active in a slot"
        return "live_ready", "confirmed by you in an active slot"
    if action.kind == "stock":
        return "live_ready", "the called play"
    entry_ok = confirmed(confirmations, action.id)
    per_option = option is not None and any(
        (meta.get("options_grade") != "user_playbook") for meta in action.params.values())
    if entry_ok and (not per_option or confirmed(confirmations, action.id, option)):
        return "live_ready", "confirmed by you in-game"
    if entry_ok and per_option:
        return "needs_user_test", f"entry confirmed; menu item {option!r} not yet confirmed by you"
    return "needs_user_test", "inputs follow EA's Madden 27 map; not yet confirmed by you"


# ------------------------------------------------------------------ plans and validation


@dataclass(frozen=True)
class PlannedAction:
    action_id: str
    params: Mapping[str, str] = field(default_factory=dict)


@dataclass
class PlayContext:
    side: str
    formation: str
    play: str
    play_type: str  # offense: pass | run ; defense: coverage | goal_line_run
    eligible_receivers: tuple[str, ...] = ()
    defenders: tuple[str, ...] = ()
    formation_audibles: Mapping[str, str] = field(default_factory=dict)  # play name -> run | pass
    play_clock_remaining: float | None = None
    coach_mode: bool = False
    current_settings: Mapping[str, str] = field(default_factory=dict)  # touch key -> value already in effect
    receiver_share: Mapping[str, float] = field(default_factory=dict)


def _resolve(spec: ActionSpec, params: Mapping[str, str], loadout: Loadout | None) -> dict[str, str]:
    p = dict(params)
    if spec.option_groups and p.get("option"):
        p["option_group"] = spec.option_groups.get(p["option"], p["option"])
    if spec.kind in ("preset", "saved_custom") and loadout:
        e = loadout.usable(spec.side, spec.name)
        if e:
            p["slot"] = e.slot
    return p


def _fill(template: str, p: Mapping[str, str]) -> str:
    return re.sub(r"\{([a-z_]+)\}", lambda m: str(p.get(m.group(1), "{" + m.group(1) + "}")), template)


def _overlaps(a: str, b: str) -> bool:
    if a == b:
        return True
    pa, _, ta = a.partition(":")
    pb, _, tb = b.partition(":")
    return pa.rstrip("s") == pb.rstrip("s") and "all" in (ta, tb)


def _spec_for(pa: PlannedAction, side: str, loadout: Loadout | None) -> ActionSpec | None:
    reg = registry()
    if pa.action_id in reg:
        return reg[pa.action_id]
    prefix = f"{side[:3]}.saved."
    if pa.action_id.startswith(prefix) and loadout:
        name = pa.action_id[len(prefix):]
        for e in loadout.entries:
            if e.side == side and e.name == name:
                return saved_action(e)
    return None


def plan_seconds(specs: Sequence[ActionSpec]) -> float:
    real = [s for s in specs if s.kind != "stock"]
    return sum(s.est_seconds for s in real) + GET_SET_SECONDS * max(0, len(real) - 1)


def validate_plan(plan: Sequence[PlannedAction], ctx: PlayContext, *, loadout: Loadout | None = None,
                  confirmations: Mapping[str, Any] | None = None, require_live: bool = False) -> list[str]:
    """Every reason this plan cannot be executed in Madden 27 as written (empty = executable)."""
    errs: list[str] = []
    specs: list[tuple[ActionSpec, dict[str, str]]] = []
    for pa in plan:
        spec = _spec_for(pa, ctx.side, loadout)
        if spec is None:
            errs.append(f"unknown adjustment {pa.action_id!r}")
            continue
        if spec.side != ctx.side:
            errs.append(f"{spec.name}: {spec.side} adjustment on a {ctx.side} snap")
            continue
        p = _resolve(spec, pa.params, loadout)
        specs.append((spec, p))
        option = next((p[k] for k in spec.params if k in p), None)
        status, why = readiness(spec, option, loadout=loadout, confirmations=confirmations)
        if status in ("unsupported", "not_installed"):
            errs.append(f"{spec.name}: {why}")
        elif require_live and status != "live_ready":
            errs.append(f"{spec.name}: {why}")
        errs += [f"{spec.name}: {e}" for e in validate_inputs(spec.side, spec.category, [_fill(s, p) for s in spec.inputs])]
        for key, meta in spec.params.items():
            val = p.get(key)
            if val is None:
                errs.append(f"{spec.name}: missing {key}")
                continue
            opts = meta.get("options")
            if isinstance(opts, list) and val not in opts:
                errs.append(f"{spec.name}: {val!r} is not a {key} this action offers")
            if opts == "formation_audibles" and val not in ctx.formation_audibles:
                errs.append(f"{spec.name}: {val!r} is not an audible in {ctx.formation!r}")
        req = spec.requires
        if req.get("play_type") and ctx.play_type not in req["play_type"]:
            errs.append(f"{spec.name}: not available on a {ctx.play_type} play")
        if req.get("target") == "eligible_receiver":
            tgt = p.get("target")
            if not ctx.eligible_receivers:
                errs.append(f"{spec.name}: eligible receivers in {ctx.formation!r} are unknown")
            elif tgt not in ctx.eligible_receivers:
                errs.append(f"{spec.name}: {tgt!r} is not an eligible receiver in {ctx.formation!r}")
        if req.get("target") == "defender" and p.get("target") not in ctx.defenders:
            errs.append(f"{spec.name}: {p.get('target')!r} is not a defender on the field")
        if req.get("target_positions") and not any(str(p.get("target", "")).startswith(t) for t in req["target_positions"]):
            errs.append(f"{spec.name}: only {', '.join(req['target_positions'])} can do this")
        for key in spec.touches:
            k = _fill(key, p)
            for cur, val in ctx.current_settings.items():
                if _overlaps(k, cur):
                    errs.append(f"{spec.name}: conflicts with {val} already in effect")
    for (a, pa_), (b, pb_) in itertools.combinations(specs, 2):
        ga = {_fill(g, pa_) for g in a.exclusive_groups}
        gb = {_fill(g, pb_) for g in b.exclusive_groups}
        ta = {_fill(t, pa_) for t in a.touches}
        tb = {_fill(t, pb_) for t in b.touches}
        if any(_overlaps(x, y) for x in ga for y in gb) or any(_overlaps(x, y) for x in ta for y in tb):
            errs.append(f"{a.name} and {b.name} cannot both be applied (they change the same thing)")
    if ctx.play_clock_remaining is not None:
        need = plan_seconds([s for s, _ in specs])
        deadline = ctx.play_clock_remaining - SNAP_MARGIN_SECONDS - (
            load_registry_data()["rules"]["coach_mode_cutoff_seconds"]["value"] if ctx.coach_mode else 0)
        if need > max(0.0, deadline):
            errs.append(f"needs about {need:.0f}s of inputs but only {max(0.0, deadline):.0f}s remain before the snap")
    return errs


def render_inputs(spec: ActionSpec, params: Mapping[str, str], loadout: Loadout | None = None) -> str:
    p = _resolve(spec, params, loadout)
    words = []
    for s in spec.inputs:
        s = _fill(s, p)
        kind, _, val = s.partition(":")
        if kind == "tap":
            words.append(f"Press {val}")
        elif kind == "hold":
            words.append(f"Hold {val}")
        elif kind == "stick":
            words.append("Move the right stick left or right" if val == "RS_LEFT_RIGHT" else f"Move {val}")
        elif kind == "hold_stick":
            words.append("hold the left stick left or right")
        elif kind == "icon":
            who = p.get("target", f"the {val}")
            words.append(f"press the button shown over {who}")
        elif kind == "onscreen":
            words.append(f"choose '{val}' on the on-screen menu")
        elif kind == "slot":
            words.append(f"choose '{spec.name}' in your confirmed slot ({val})")
    return " -> ".join(words) if words else ("Snap with A when ready" if spec.side == "offense" else "No pre-snap input")


# ------------------------------------------------------------------ football decision


@dataclass
class Option:
    spec: ActionSpec
    params: dict[str, str]
    status: str
    status_reason: str
    gain: float
    score: float
    evidence: str
    helps: list[str]
    exposes: list[str]
    errors: list[str]


def _effects(spec: ActionSpec, params: Mapping[str, str]) -> Mapping[str, Any]:
    if spec.effects_by_param:
        for key in ("route", "option", "play_family", "protection"):
            val = params.get(key)
            if val in spec.effects_by_param:
                return spec.effects_by_param[val]
        return {"addresses": {}, "exposes": {}, "evidence": "hypothesis"}
    return spec.effects or {"addresses": {}, "exposes": {}, "evidence": "hypothesis"}


def _entropy(dist: Mapping[str, float]) -> float:
    vals = [v for v in dist.values() if v > 0]
    return -sum(v * math.log(v) for v in vals) / math.log(len(dist)) if len(dist) > 1 else 0.0


def _param_grid(spec: ActionSpec, ctx: PlayContext) -> Iterable[dict[str, str]]:
    if spec.category == "hot_route":
        for t in ctx.eligible_receivers:
            for r in spec.effects_by_param:
                if r == "Block" and not t.startswith(("TE", "HB", "FB")):
                    continue
                yield {"target": t, "route": r}
    elif spec.category == "audible":
        for play, fam in ctx.formation_audibles.items():
            if play != ctx.play:
                yield {"play": play, "play_family": fam}
    elif spec.category == "protection":
        for opt in (spec.params.get("protection") or {}).get("options") or [None]:
            yield {"protection": opt} if opt else {}
    elif spec.id == "def.double_team":
        for t in ctx.eligible_receivers:
            yield {"target": t}
    elif spec.id == "def.coverage_hub":
        for opt in spec.effects_by_param:
            yield {"option": opt}
    elif spec.requires.get("target"):
        return
    else:
        yield {}


def candidate_options(ctx: PlayContext, distribution: Mapping[str, float], *, loadout: Loadout | None = None,
                      confirmations: Mapping[str, Any] | None = None) -> list[Option]:
    """Every adjustment for this snap with its expected effect; the unadjusted play is always first."""
    specs = [s for s in registry().values() if s.side == ctx.side]
    if loadout:
        specs += [saved_action(e) for e in loadout.entries if e.side == ctx.side and not e.preset_id]
    h = _entropy(distribution) if distribution else 1.0
    out: list[Option] = []
    for spec in specs:
        for params in _param_grid(spec, ctx):
            eff = _effects(spec, params)
            share = 1.0
            if params.get("target") and ctx.receiver_share:
                top = max(ctx.receiver_share.values()) or 1.0
                share = ctx.receiver_share.get(params["target"], 0.0) / top
            addr = {k: v * share for k, v in (eff.get("addresses") or {}).items()}
            expo = dict(eff.get("exposes") or {})
            gain = sum(distribution.get(k, 0.0) * v for k, v in addr.items()) - \
                sum(distribution.get(k, 0.0) * v for k, v in expo.items())
            evidence = eff.get("evidence", "hypothesis")
            timing = 0.01 * spec.est_seconds / 3.0
            score = 0.0 if spec.kind == "stock" else gain * EVIDENCE_WEIGHT[evidence] * (1 - 0.5 * h) - timing
            option = next((params[k] for k in spec.params if k in params), None)
            status, why = readiness(spec, option, loadout=loadout, confirmations=confirmations)
            errors = validate_plan([PlannedAction(spec.id, params)], ctx, loadout=loadout, confirmations=confirmations)
            out.append(Option(spec, dict(params), status, why, gain, score, evidence,
                              sorted(addr, key=lambda k: -distribution.get(k, 0) * addr[k])[:3],
                              sorted(expo, key=lambda k: -distribution.get(k, 0) * expo[k])[:3], errors))
    out.sort(key=lambda o: (o.spec.kind != "stock", -o.score))
    return out


@dataclass
class Recommendation:
    chosen: Option
    unadjusted: Option
    theoretical_best: Option | None
    considered: int
    note: str


def recommend(ctx: PlayContext, distribution: Mapping[str, float], *, loadout: Loadout | None = None,
              confirmations: Mapping[str, Any] | None = None, mode: str = "live") -> Recommendation:
    """Adjust only when an executable option beats the unadjusted play by a real margin.

    ``live`` uses live-ready actions only. ``research`` also shows actions that still need
    your in-game test, clearly labeled. Not-installed and unsupported actions are never chosen.
    """
    opts = candidate_options(ctx, distribution, loadout=loadout, confirmations=confirmations)
    unadjusted = opts[0]
    allowed = {"live_ready"} if mode == "live" else {"live_ready", "needs_user_test"}
    theoretical = max(opts[1:], key=lambda o: o.score, default=None)
    usable = [o for o in opts[1:] if o.status in allowed and not o.errors]
    best = max(usable, key=lambda o: o.score, default=None)
    if best is None or best.score < MIN_ADVANTAGE:
        note = "no executable adjustment beats the called play by a meaningful margin"
        if best is None:
            note = "no adjustment is executable here with what is confirmed"
        chosen = unadjusted
    else:
        chosen, note = best, f"beats the called play by {best.score:+.3f} (weighted, not measured)"
    if theoretical is not None and chosen is not theoretical and theoretical.score > max(chosen.score, MIN_ADVANTAGE):
        detail = ", ".join(v for k, v in theoretical.params.items() if k not in ("play_family", "option_group"))
        reason = theoretical.status_reason if theoretical.status != "live_ready" or not theoretical.errors \
            else theoretical.errors[0]
        note += f"; best in theory: {theoretical.spec.name}{' (' + detail + ')' if detail else ''} - {reason}"
    return Recommendation(chosen, unadjusted, theoretical, len(opts), note)


def coaching_card(rec: Recommendation, ctx: PlayContext, *, loadout: Loadout | None = None) -> str:
    o = rec.chosen
    target = o.params.get("target") or (o.params.get("slot") or "")
    if o.spec.kind in ("preset", "saved_custom") and loadout:
        e = loadout.usable(o.spec.side, o.spec.name)
        target = f"slot {e.slot}" if e else "not in an active slot"
    label = o.spec.name
    extra = ", ".join(f"{v}" for k, v in o.params.items() if k not in ("target", "slot", "play_family", "option_group"))
    if extra:
        label = f"{label}: {extra}"
    helps = ", ".join(o.helps) or ("-" if o.spec.kind == "stock" else "no modeled benefit")
    exposes = ", ".join(o.exposes) or "-"
    lines = [
        f"FORMATION: {ctx.formation}",
        f"PLAY: {ctx.play}",
        f"MADDEN ADJUSTMENT: {label if o.spec.kind != 'stock' else 'None - run the called play as is'}",
        f"XBOX INPUTS: {render_inputs(o.spec, o.params, loadout)}",
        f"REQUIRED PLAYER OR SLOT: {target or '-'}",
        f"WHY THIS HELPS: {('answers ' + helps) if o.spec.kind != 'stock' else rec.note}",
        f"WHAT IT EXPOSES: {exposes}",
        f"CONFIDENCE: {CONFIDENCE_TEXT[o.evidence] if o.spec.kind != 'stock' else 'n/a'}",
        f"EXECUTION STATUS: {STATUS_TEXT[o.status].split(':')[0]}: {o.status_reason}",
    ]
    if o.spec.kind != "stock":
        lines.append(f"NOTE: {rec.note}")
    return "\n".join(lines)


# ------------------------------------------------------------------ pre-game loadout planning


def plan_loadout(side: str, situations: Sequence[tuple[float, Mapping[str, float]]], *,
                 limit: int | None = None, designs: Sequence[ActionSpec] = ()) -> list[dict[str, Any]]:
    """Rank presets and saved designs for the game's active slots. Nothing is marked installed."""
    rules = load_registry_data()["rules"]["saved_and_active_limits"]
    limit = int(limit or rules["active_per_game"])
    pool = [s for s in registry().values() if s.side == side and s.kind == "preset"] + list(designs)
    rows = []
    for spec in pool:
        eff = spec.effects or {}
        value = 0.0
        for weight, dist in situations:
            g = sum(dist.get(k, 0.0) * v for k, v in (eff.get("addresses") or {}).items()) - \
                sum(dist.get(k, 0.0) * v for k, v in (eff.get("exposes") or {}).items())
            value += weight * max(0.0, g)
        rows.append({"name": spec.name, "kind": spec.kind, "expected_value": round(value, 4),
                     "evidence": eff.get("evidence", "hypothesis"),
                     "status": "to install: not installed until you confirm it in-game with its slot"})
    rows.sort(key=lambda r: -r["expected_value"])
    return [r for r in rows if r["expected_value"] > 0][:limit]


def catalog_report(*, confirmations: Mapping[str, Any] | None = None, loadout: Loadout | None = None) -> dict[str, list[str]]:
    """Every registered action sorted into verified / needs-test / not-installed / unsupported."""
    out: dict[str, list[str]] = {"fully_verified": [], "ea_documented_end_to_end_needs_test": [],
                                 "needs_user_test": [], "not_installed": [], "unsupported": []}
    for spec in registry().values():
        if spec.kind == "stock":
            continue
        status, why = readiness(spec, None, loadout=loadout, confirmations=confirmations)
        label = f"{spec.side}: {spec.name}"
        if status == "live_ready":
            out["fully_verified"].append(label)
        elif status == "needs_user_test":
            key = "ea_documented_end_to_end_needs_test" if spec.inputs_grade == "ea_documented" else "needs_user_test"
            out[key].append(label)
        else:
            out[status].append(f"{label} ({why})")
    return out
