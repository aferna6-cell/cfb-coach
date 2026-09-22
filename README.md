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

**Browser open (Linux/WSL):** `prep` tries `webbrowser` → `xdg-open`, then on WSL (`/proc/version` contains Microsoft, or `WSL_DISTRO_NAME` set): `wslview <path>` → `explorer.exe $(wslpath -w <path>)` → prints the Windows path for manual open. Use `--no-open` to skip and only write the file.

```bash
PYTHONPATH=. python3 -m cfb_coach prep --opponent cpu --no-open
# then open manually, e.g.:
#   wslview ~/.cfb-coach/prep_cpu.html
#   explorer.exe $(wslpath -w ~/.cfb-coach/prep_cpu.html)
```

## Commands

| Command | Purpose |
|--------|---------|
| `python3 -m cfb_coach opponents` | List IDs, teams, aliases |
| `python3 -m cfb_coach prep --opponent <id>` | Open browser: **Live meta scout** + deltas only + call tips + Active loadout |
| `python3 -m cfb_coach prep -o <id> --text` | Compact terminal delta dump (no browser) |
| `python3 -m cfb_coach prep -o <id> --no-open` | Write HTML without opening browser |
| `python3 -m cfb_coach prep -o <id> --mark-applied` | Mark proposed deltas applied (next prep shows only NEW) |
| `python3 -m cfb_coach prep -o <id> --offline` | Skip network meta scout (use cache/baseline `cfb27-2026-09`) |
| `python3 -m cfb_coach prep -o <id> --refresh-meta` | Force refetch meta scout (ignore <6h cache) |
| `python3 -m cfb_coach play --opponent <id>` | Interactive live loop (CPU = offense-only) |
| `python3 -m cfb_coach play --opponent <id> --once "2&7 c2 invert"` | One-shot non-interactive call |
| `python3 -m cfb_coach call -o gavin -s "2&7 cover 2 invert" --why` | Scripted one-shot |
| `python3 -m cfb_coach postgame --opponent gavin` | Learn from snaps; ohio_state successes → Alabama promotion notes |
| `python3 -m cfb_coach promote` | List pending Alabama promotions (from ohio_state lab) |
| `python3 -m cfb_coach promote --accept-all` | Accept all pending Alabama promotions |
| `python3 -m cfb_coach play --opponent cpu` | CPU = offense-only live loop |

Opponent aliases: ids (`gavin`), display names (`Gavin`), teams (`Auburn`, `Houston`, `SMU`, …).

## Prep = rich opening plan (v1.6.0)

Alabama user games are ~**once per season** — you cannot lean on thick per-user datasets. Prep is the main edge:

1. **Live meta scout** (default on) hits trusted CFB 27 patch/meta URLs, caches ~6h under `~/.cfb-coach/meta_cache.json`, and maps hits into Aidan's book language as soft **Meta-grounded (live scout)** suggestions.
2. **Thin seeds + baseline** (`cfb27-2026-09`) still drive the stocked inventory.
3. Browser top section: **Patch radar**, **What's meta (O/D)**, **How it affects THIS prep** (ties to any deltas). Offline/failed → "Scout unavailable — using cached/baseline cfb27-2026-09" without breaking prep.
4. Lightly biases gameplan/macro **deltas only** (still ≤8 active macros, CPU O-only, alabama vs ohio_state rules).

Inventory (seed playbooks + 8 active / 2 benched macros) is the **already-stocked** book. Prep never says CREATE BAMA META O/D from scratch or re-ADD all 8 macros.

- **Playbook adjustments** — ADD/REMOVE formation, EDIT audible slot (only when changed for this opponent)
- **Macro adjustments** — EDIT field (e.g. HEAT when-to-arm), ADD one new recipe, BENCH/UNBENCH
- If no deltas: browser says **"No playbook changes — run baseline as-is"** + short call tips
- Main view = **Active loadout only (≤8)**; collapsible inventory is read-only reference (no bench catalog dump)
- `--mark-applied` persists applied deltas so the next prep only shows NEW changes
- `--offline` skips network; `--refresh-meta` forces refetch


## Dynasty modes (v1.6.0)

```bash
prep --opponent gavin --dynasty alabama      # default: serious USER dynasty
prep --opponent gavin --dynasty ohio_state   # experimental lab / practice
prep --opponent cpu --dynasty ohio_state     # CPU = offense-only coaching
```

- **alabama** (serious USER dynasty, default): stick to proven Active-8; experimental/meta-grounded macros stay benched unless promoted; tighter pivots. Alabama user opponents still get **O+D** coaching.
- **ohio_state** (experimental lab / practice): freer strategies — meta-grounded experimental macros / gameplan tweaks OK in prep deltas.
- **Promotion:** when an experimental strategy **works** in ohio_state (strong postgame weight bumps), `postgame` records an Alabama promotion note (macro loadout change or gameplan overlay). Review/accept with `cfb_coach promote` / `promote --accept-all`.
- Dynasty is stored in the session DB for `play` / `postgame`.


## CPU games = offense-only

`prep --opponent cpu` and `play --opponent cpu` are **offense-only** coaching: no defense calls, no D macros emphasis. Active loadout D section shows **N/A — offense only**. Alabama user opponents (gavin/quen/…) still get full **O+D**.

## Exact macros (Aidan sheets)

Defense Active-8 copy blocks are Aidan's **exact** Custom Adjustments ticks (General / DL/LB / Secondary / Zone Drops / Strategy / Coverage Checks / Individuals) — not approx checklists.
- ACTIVE: CROSS VERT BUNCH RPO SCRAM RUN-IN RUN-OUT HEAT
- BENCHED: FLOOD SCREEN (still `proven`, just not Active)
- UI note: Safety Midpoint **Strong** = toward pass strength
- Doctrine: do **not** auto-use a macro from one concept appearance — most snaps Cover 3 Sky / Quarters / Tampa 2 with no macro.

## Macros — Active loadout only, click-to-copy, validation (v1.6.0)

- **USER Active hard cap = 8** Custom Adjustments across **Offense + Defense combined** (Aidan rule for online dynasty). EA's UI may advertise 10 — honor **8**.
- Path: **Create & Share → Custom Adjustments → Offense/Defense** → edit/save → set Active → in-game **LB** to use.
- Prep browser shows **ONLY the Active loadout (≤8)** clickable accordion cards — **never** dumps benched FLOOD/SCREEN (or the whole bench catalog) in the main prep view.
- If a swap is proposed: loadout shows the **8 after the swap**, plus one line **“replacing X with Y”** (still no bench catalog dump). Expand a card for full settings + **Copy** button.
- If prep proposes **ADD** while already at 8/8: also shows an exact **swap plan** (which Active macro to deactivate) with Xbox steps.
- Validation badges on playbook deltas + live call suggestions: `proven` | `meta_grounded` | `failed` | `unvalidated`.
  - Temple 8D baseline (CROSS VERT BUNCH RPO SCRAM RUN-IN RUN-OUT HEAT) = **proven** (survived his games).
  - CREATE candidates grounded in CFB27 meta (GLASS, CONTAIN-SCRAM, SPOT-LOCK, PROT, O macros) = **meta_grounded** — OK to bring into game.
  - **failed** / cooking — got cooked; demote via postgame.
  - Prep deltas only include tweaks that are at least **meta_grounded** (no ungrounded invention).
  - Live caller prefers proven inventory; tags `meta_grounded` / `failed` / `unvalidated` when suggesting others.
- **Workflow:** bring a suggested macro into game → if cooked, adjust via postgame → if it holds, mark **proven**.
- Catalog: `cfb_coach/data/macro_catalog.json` (full_settings + copy_block per macro).

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

## Doctrine (Aidan)

**Priority stack**

1. **Rich prep** — live meta scout + patch notes + thin seeds = strong opening gameplan.
2. **Mid-game adaptation THIS game** — after a few snaps of their tendencies (2+ tells), pivot calls/macros for *this* game. Still no single-snap whiplash.
3. **Per-user long-term learning** — light carry-forward only, not the main edge.

Ohio State / CPU = volume lab. Alabama users = prepare well + adjust live.

- Free reign: any playbook call is legal.
- Default defense prior: **Nickel Over** (C3 Sky / C4 Quarters / Tampa 2). Macros situational.
- Meta priors are soft weights; **in-game evidence outranks** thin career film.
- Ignore rosters (new season).
- Cover 2 Invert / Invert Hard Flat as a *live* soft lean → prefer **Mesh Spot / easy underneath or Inside Zone** — not forced Deep Flood.
- **Tendency discipline (symmetric O + D):**
  - One tell = **log + mild probability bump only**.
  - Do **not** hard-counter the previous coverage/concept every snap.
  - Targeted macro / coverage-specific beater only when the same signal is a **REPEATED** tendency (2+) in a similar D&D/field zone — for user opponents, the **this-game window** (recent snaps) counts first.
  - Default next snap = base situational call (D&D, field, opponent archetype priors).
- **PIVOT:** user games soft-pivot after **2** fails on a side; hard PIVOT at **3** for everyone (switch family / reset Nickel Over). No single-snap whiplash.


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
    meta_scout.py       # live CFB27 patch/meta scout (urllib, 6h cache)
    gameplan.py         # baseline + overlays + postgame learning + pivot
    install_sheet.py    # delta engine (inventory vs proposed; mark_prep_applied)
    tendency.py
    data/seed.json
    data/meta_baseline.json  # CFB27 META (cfb27-2026-09)
```

## Smoke

```bash
PYTHONPATH=. python3 -m cfb_coach prep --opponent cpu --dynasty ohio_state --offline --no-open
PYTHONPATH=. python3 -m cfb_coach prep --opponent cpu --dynasty ohio_state --no-open   # live scout section even if some URLs 404
```

Offline prep must succeed. With network, prep HTML includes **Live meta scout** (fetch budget ≤~12s).
