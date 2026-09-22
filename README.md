# cfb_coach

Xbox **CFB 27** dynasty play-caller for Aidan's Alabama online Dynasty.

Heuristics + packaged `seed.json` + CFB27 META baseline + SQLite log learning. No neural net.

**Prepper + live caller.** Prep emits an **INSTALL SHEET** (CREATE/ADD/EDIT/BENCH steps for Xbox custom O/D books + macros + custom adjs) and remembers per-opponent install diffs in SQLite. Live caller stays sharp: two reads on O, one user job on D, anti-repeat, no single-snap whiplash. Mid-game **PIVOT** fires when the last 3 snaps fail on a side.

## Install / run

```bash
cd /workspace/cfb-coach
# stdlib only — no pip deps required
PYTHONPATH=. python3 -m cfb_coach opponents
PYTHONPATH=. python3 -m cfb_coach prep --opponent gavin
PYTHONPATH=. python3 -m cfb_coach play --opponent gavin
```

Optional editable install:

```bash
pip install -e .
cfb-coach prep --opponent quen
```

DB seeds on first run to `~/.cfb-coach/coach.db` (fallback: `/workspace/cfb-coach/data/coach.db`). Override with `CFB_COACH_DB=/path/to/coach.db`.

## Commands

| Command | Purpose |
|--------|---------|
| `python3 -m cfb_coach opponents` | List IDs, teams, aliases |
| `python3 -m cfb_coach prep --opponent <id>` | INSTALL SHEET → BASELINE META → OVERLAY → EFFECTIVE → OPENING → THREATS |
| `python3 -m cfb_coach play --opponent <id>` | Interactive live loop |
| `python3 -m cfb_coach play --opponent <id> --once "2&7 c2 invert"` | One-shot non-interactive call |
| `python3 -m cfb_coach call -o gavin -s "2&7 cover 2 invert" --why` | Scripted one-shot |
| `python3 -m cfb_coach postgame --opponent gavin` | Learn from recent snaps — adjust weights |

Opponent aliases: ids (`gavin`), display names (`Gavin`), teams (`Auburn`, `Houston`, `SMU`, …).

## Live call syntax (UX lock)

**Offense — always two reads:**

```text
Formation — Play | Adj | Read1 → Read2
```

Example:

```text
Gun Bunch X Nasty — Mesh Spot | No adj | Spot → Drag
```

**Defense — one user job:**

```text
Formation — Play | Macro|none | User <one job>
```

Example:

```text
Nickel Over — Cover 4 Quarters | VERT | User #3 seam
```

Platform: Xbox. Prefer formation / play / macro **names**. Do not invent PlayStation button sequences.

## Play loop

```text
[O] sit> 1&10
Gun Bunch X Nasty — Inside Zone | No adj | Front → Cutback
[O] sit> 2&7 c2 invert
Gun Bunch X Nasty — Mesh Spot | No adj | Spot → Drag
[O] sit> d 3&8 verts
Nickel Over — Cover 4 Quarters | VERT | User #3 seam
[O] sit> result +4 run
  logged: +4 run
[O] sit> quit
```

- Prefix `d ` (or type `d` / `side d`) for defense.
- `result <text>` / `log <text>` updates SQLite tendencies.
- `why` prints the last call's rationale.
- After 3 failed snaps on a side, live caller tags **PIVOT:** and switches family/macro plan (no hero-shot whiplash).

## Baseline → Learn → Adjust (CFB27)

Every user/CPU game starts from a **generic CFB27 META baseline** (`cfb_coach/data/meta_baseline.json`, version `cfb27-2026-09`) — Bunch-first Wazzu/OSU-style offense; Nickel Over zones home on D.

1. **INSTALL SHEET** — prepper has full permission to CREATE/ADD/EDIT/BENCH Xbox custom books, macros, and custom adjs. Concrete steps printed first. Per-opponent install diffs saved in SQLite so next prep remembers.
2. **Baseline** — offense opening menu / early-down mix / run-first vs Cover 6/9/Quarters / Mesh Spot + Whip Trail pressure answers / RZ possession; defense Nickel Over home (C3 Sky / C4 Quarters / Tampa 2) + situational Even 6-1 + selective Cub/Mug pressure; macros KEEP CROSS VERT BUNCH RPO SCRAM RUN-IN RUN-OUT HEAT, BENCH FLOOD + SCREEN; CREATE candidates GLASS / CONTAIN-SCRAM / SPOT-LOCK / PROT.
3. **Learn** — logged snaps and `postgame --opponent X` bump gameplan / macro weights in SQLite for that opponent, plus soft `global` (and `cpu`) buckets. Successes up-weight what worked; failures down-weight. Anti-repeat penalizes spamming the same play. Free reign is never removed.
4. **Adjust** — opponent overlays stack **on top** of baseline. Thin film (few snaps) stays near META. `prep` prints **INSTALL → BASELINE → OVERLAY → EFFECTIVE → OPENING → THREATS**.

**Opponent install leans:** Gavin → run-vs-C6; Quen → protection/hot macros (PROT); CPU → patience / no forced Mesh Post on money.

**Patch 1.012 (Sep 22 2026):** improved run-action blocking; reduced QB spin effectiveness; contain custom adj fix — lean run-action more; scram less spin-hero; CONTAIN-SCRAM installs cleaner.

```bash
PYTHONPATH=. python3 -m cfb_coach prep --opponent gavin   # rich film → run-vs-C6 install lean
PYTHONPATH=. python3 -m cfb_coach prep --opponent quen    # protection macro lean
PYTHONPATH=. python3 -m cfb_coach prep --opponent cpu     # same baseline; overlay depth differs
PYTHONPATH=. python3 -m cfb_coach postgame --opponent gavin
```

## Doctrine (from seed)

- Free reign: any playbook call is legal.
- Default defense prior: **Nickel Over** (C3 Sky / C4 Quarters / Tampa 2). Macros situational.
- Meta priors are soft weights; **dynasty evidence outranks**.
- Ignore rosters (new season).
- Cover 2 Invert / Invert Hard Flat as a *live* soft lean → prefer **Mesh Spot / easy underneath or Inside Zone** — not forced Deep Flood.
- **Tendency discipline (symmetric O + D):**
  - One tell (one Cross Wheels, one IZ, one C2 Invert) = **log + mild probability bump only**.
  - Do **not** hard-counter the previous coverage/concept every snap.
  - Targeted macro / coverage-specific beater only when the same signal is a **REPEATED** tendency in a similar D&D/field zone.
  - Default next snap = base situational call (D&D, field, opponent archetype priors).
- **PIVOT:** last 3 snaps fail on a side → switch family (O: run ↔ Mesh Spot ↔ Cluster) or reset D to Nickel Over base (clear chase macros). No single-snap whiplash.

## Package layout

```text
cfb-coach/
  pyproject.toml
  requirements.txt      # empty — stdlib only
  README.md
  data/                 # fallback coach.db
  cfb_coach/
    __main__.py
    cli.py
    seed.py / db.py
    opponents.py
    situation.py
    playcaller.py       # live caller + PIVOT
    format_call.py      # UX lock formatter
    prep.py             # INSTALL → BASELINE → OVERLAY → EFFECTIVE → OPENING → THREATS
    gameplan.py         # baseline + overlays + postgame learning + pivot
    install_sheet.py    # CREATE/ADD/EDIT/BENCH recipes + SQLite install diffs
    tendency.py
    data/seed.json
    data/meta_baseline.json  # CFB27 META (cfb27-2026-09)
```

## Smoke

```bash
PYTHONPATH=. python3 -m cfb_coach prep --opponent gavin
PYTHONPATH=. python3 -m cfb_coach prep --opponent quen
PYTHONPATH=. python3 -m cfb_coach prep --opponent cpu
PYTHONPATH=. python3 -m cfb_coach call -o gavin -s "2&7 cover 2 invert" --why
PYTHONPATH=. python3 -m cfb_coach postgame --opponent gavin
```
