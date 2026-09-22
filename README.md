# cfb_coach

Xbox **CFB 27** dynasty play-caller for Aidan's Alabama online Dynasty.

Heuristics + packaged `seed.json` + CFB27 META baseline + SQLite log learning. No neural net.

**Prepper + live caller.** Prep opens a **browser** with **playbook/macro diffs only** (never a full recreate install sheet). Live caller stays sharp: two reads on O, one user job on D, anti-repeat, no single-snap whiplash. Mid-game **PIVOT** fires when the last 3 snaps fail on a side.

## Install / run

```bash
cd /workspace/cfb-coach
# stdlib only — no pip deps required
PYTHONPATH=. python3 -m cfb_coach opponents
PYTHONPATH=. python3 -m cfb_coach prep --opponent gavin          # opens browser (deltas only)
PYTHONPATH=. python3 -m cfb_coach prep --opponent gavin --text   # terminal delta dump
PYTHONPATH=. python3 -m cfb_coach play --opponent gavin
```

Optional editable install:

```bash
pip install -e .
cfb-coach prep --opponent quen
```

DB seeds on first run to `~/.cfb-coach/coach.db` (fallback: `/workspace/cfb-coach/data/coach.db`). Override with `CFB_COACH_DB=/path/to/coach.db`.

Prep HTML writes to `~/.cfb-coach/prep_<opponent>.html` (same data dir as the DB).

## Commands

| Command | Purpose |
|--------|---------|
| `python3 -m cfb_coach opponents` | List IDs, teams, aliases |
| `python3 -m cfb_coach prep --opponent <id>` | Open browser: **deltas only** + call tips + collapsible inventory |
| `python3 -m cfb_coach prep -o <id> --text` | Compact terminal delta dump (no browser) |
| `python3 -m cfb_coach prep -o <id> --no-open` | Write HTML without opening browser |
| `python3 -m cfb_coach prep -o <id> --mark-applied` | Mark proposed deltas applied (next prep shows only NEW) |
| `python3 -m cfb_coach play --opponent <id>` | Interactive live loop |
| `python3 -m cfb_coach play --opponent <id> --once "2&7 c2 invert"` | One-shot non-interactive call |
| `python3 -m cfb_coach call -o gavin -s "2&7 cover 2 invert" --why` | Scripted one-shot |
| `python3 -m cfb_coach postgame --opponent gavin` | Learn from recent snaps — adjust weights |

Opponent aliases: ids (`gavin`), display names (`Gavin`), teams (`Auburn`, `Houston`, `SMU`, …).

## Prep = diffs only (v1.3)

Inventory (seed playbooks + 8 active / 2 benched macros) is the **already-stocked** book. Prep never says CREATE BAMA META O/D from scratch or re-ADD all 8 macros.

- **Playbook adjustments** — ADD/REMOVE formation, EDIT audible slot (only when changed for this opponent)
- **Macro adjustments** — EDIT field (e.g. HEAT when-to-arm), ADD one new recipe, BENCH/UNBENCH
- If no deltas: browser says **“No playbook changes — run baseline as-is”** + short call tips
- Collapsible **Current inventory** (formations→plays) is read-only reference, collapsed by default
- `--mark-applied` persists applied deltas so the next prep only shows NEW changes

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

1. **Inventory** — seed O/D formations + plays + 8 D macros (BENCH FLOOD + SCREEN) are assumed fully stocked on Xbox.
2. **Prep deltas** — opponent overlays propose only ADD/REMOVE/EDIT/BENCH tweaks. Thin film (few snaps) → zero or minimal deltas.
3. **Learn** — logged snaps and `postgame --opponent X` bump gameplan / macro weights in SQLite. Anti-repeat penalizes spamming the same play.
4. **Adjust** — overlays stack on top of baseline; live caller unchanged.

**Opponent lean examples:** Gavin → elevate run audibles, demote Mesh Post early, maybe ADD CONTAIN-SCRAM / EDIT HEAT; Quen → protection/hot (PROT) + audible edits; Ryan (thin) → no/minimal deltas.

**Patch 1.012 (Sep 22 2026):** improved run-action blocking; reduced QB spin effectiveness; contain custom adj fix — lean run-action more; scram less spin-hero; CONTAIN-SCRAM installs cleaner.

```bash
PYTHONPATH=. python3 -m cfb_coach prep --opponent gavin   # opponent-specific edits in browser
PYTHONPATH=. python3 -m cfb_coach prep --opponent ryan    # thin → often zero deltas
PYTHONPATH=. python3 -m cfb_coach prep --opponent quen
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
    prep.py             # text prep helpers + --text dump
    prep_browser.py     # dark HTML deltas → ~/.cfb-coach/prep_<opp>.html
    gameplan.py         # baseline + overlays + postgame learning + pivot
    install_sheet.py    # delta engine (inventory vs proposed; mark_prep_applied)
    tendency.py
    data/seed.json
    data/meta_baseline.json  # CFB27 META (cfb27-2026-09)
```

## Smoke

```bash
PYTHONPATH=. python3 -m cfb_coach prep --opponent gavin --no-open
PYTHONPATH=. python3 -m cfb_coach prep --opponent ryan --no-open
PYTHONPATH=. python3 -m cfb_coach prep --opponent quen --text
PYTHONPATH=. python3 -m cfb_coach call -o gavin -s "2&7 cover 2 invert" --why
PYTHONPATH=. python3 -m cfb_coach postgame --opponent gavin
```

HTML for Gavin must **not** say CREATE whole custom books or ADD all 8 macros from scratch — only opponent-specific edits. Ryan (thin) may show zero or minimal deltas.
