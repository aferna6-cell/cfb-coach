# cfb_coach

Xbox CFB dynasty play-caller for Aidan's Alabama online Dynasty (Year 1).

Heuristics + packaged `seed.json` + meta priors + SQLite log learning. No neural net.

## Install / run

```bash
cd /workspace/cfb-coach
# stdlib only — no pip deps required
PYTHONPATH=. python -m cfb_coach opponents
PYTHONPATH=. python -m cfb_coach prep --opponent gavin
PYTHONPATH=. python -m cfb_coach play --opponent gavin
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
| `python -m cfb_coach opponents` | List IDs, teams, aliases |
| `python -m cfb_coach prep --opponent <id>` | Threat sheet, D-vs-us, macros, emphasis, opening O+D menus |
| `python -m cfb_coach play --opponent <id>` | Interactive live loop |
| `python -m cfb_coach play --opponent <id> --once "2&7 c2 invert"` | One-shot non-interactive call |
| `python -m cfb_coach call -o gavin -s "2&7 cover 2 invert" --why` | Scripted one-shot |

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
    playcaller.py
    format_call.py      # UX lock formatter
    prep.py
    data/seed.json      # source of truth (copied from cfb-coach-seed)
```

## Smoke

```bash
PYTHONPATH=. python -m cfb_coach prep --opponent gavin
PYTHONPATH=. python -m cfb_coach prep --opponent quen
PYTHONPATH=. python -m cfb_coach prep --opponent cpu
PYTHONPATH=. python -m cfb_coach call -o gavin -s "2&7 cover 2 invert" --why
```
