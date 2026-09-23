# cfb_coach

Xbox **CFB 27** dynasty play-caller for Aidan's Alabama online Dynasty.

Heuristics + packaged `seed.json` + CFB27 META baseline + SQLite log learning. No neural net.

**v1.10.0:** also a typed **Madden 27 Franchise** coach behind `--game madden27` (see [Madden 27 Franchise](#madden-27-franchise---game-madden27)). CFB 27 stays the default. Nothing changes for CFB unless you pass `--game madden27`.

**Prepper + live caller.** Prep opens a **browser** with **playbook/macro diffs only** (never a full recreate install sheet). Live caller stays sharp: two reads on O, one user job on D, anti-repeat, no single-snap whiplash. Mid-game **PIVOT** fires when the last 3 snaps fail on a side.

## Install / run

```bash
cd /workspace/cfb-coach
# stdlib only — no pip deps required
PYTHONPATH=. python3 -m cfb_coach opponents
PYTHONPATH=. python3 -m cfb_coach prep --opponent gavin          # opens browser (deltas only)
PYTHONPATH=. python3 -m cfb_coach prep --opponent gavin --text   # terminal delta dump
PYTHONPATH=. python3 -m cfb_coach play --opponent gavin          # typed live + overlay browser
PYTHONPATH=. python3 -m cfb_coach play --opponent gavin --no-overlay
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
| `python3 -m cfb_coach play --opponent <id>` | Typed live loop + **browser overlay** (CPU = offense-only) |
| `python3 -m cfb_coach play --opponent <id> --no-overlay` | Same loop without HTML overlay |
| `python3 -m cfb_coach play --opponent <id> --once "2&7 c2 invert"` | One-shot non-interactive call |
| `python3 -m cfb_coach call -o gavin -s "2&7 cover 2 invert" --why` | Scripted one-shot |
| `python3 -m cfb_coach postgame --opponent gavin` | Learn from snaps; ohio_state successes → Alabama promotion notes |
| `python3 -m cfb_coach promote` | List pending Alabama promotions (from ohio_state lab) |
| `python3 -m cfb_coach promote --accept-all` | Accept all pending Alabama promotions |
| `python3 -m cfb_coach play --opponent cpu` | CPU = offense-only live loop |
| `python3 -m cfb_coach watch --demo` | Screen co-pilot demo: sample DefenseLooks → ≤3 tips |
| `python3 -m cfb_coach watch` | Live tip loop (typed look injection) |
| `python3 -m cfb_coach watch --setup` | Remote Play / capture setup notes |
| `python3 -m cfb_coach watch --window Xbox --debug` | Live vision (Windows + `[vision]`) |
| `python3 -m cfb_coach watch --calibrate` | Save window/crop → `~/.cfb-coach/vision_calib.json` |
| `python3 -m cfb_coach copilot --demo --once` | Alias of `watch` |

Opponent aliases: ids (`gavin`), display names (`Gavin`), teams (`Auburn`, `Houston`, `SMU`, …).

## Madden 27 Franchise (`--game madden27`)

Typed live calls for **Madden 27 Franchise** (not MUT, not MCS tournament tooling), same UX as CFB: prep browser with **deltas only + Active-8**, then typed `play` with `heard:` echo and the HTML overlay. **Sidecar only**: no controller input. Vision/`watch` is **not** used for Madden.

```bash
# Franchise CPU week (offense-only)
PYTHONPATH=. python3 -m cfb_coach prep --game madden27 --opponent cpu
PYTHONPATH=. python3 -m cfb_coach play --game madden27 --opponent cpu

# Franchise user game (shared persona, O + D within the 8-macro cap)
PYTHONPATH=. python3 -m cfb_coach prep --game madden27 --opponent gavin
PYTHONPATH=. python3 -m cfb_coach play --game madden27 --opponent gavin

# After the game
PYTHONPATH=. python3 -m cfb_coach postgame --game madden27 --opponent gavin

# Primary Franchise team (TBD until you set it)
PYTHONPATH=. python3 -m cfb_coach config --game madden27                          # show
PYTHONPATH=. python3 -m cfb_coach config --game madden27 --primary-team Buccaneers  # set (name, nickname or abbr: TB)
PYTHONPATH=. python3 -m cfb_coach config --game madden27 --clear-primary            # back to TBD
```

`--game madden` works as an alias. Every CFB prep/play flag carries over (`--text`, `--no-open`, `--offline`, `--refresh-meta`, `--mark-applied`, `--once`, `--why`, `--overlay`, `--no-overlay`).

| Command | Madden 27 purpose |
|---|---|
| `opponents --game madden27` | Shared personas with archetype + Madden film confidence |
| `prep --game madden27 -o <id>` | Browser: live Madden meta scout + baseline patch radar + deltas + Active-8 (CPU: offense only) |
| `prep --game madden27 -o <id> --franchise lab` | Lab profile: freer (e.g. ADD benched HEAT with a swap plan) |
| `prep --game madden27 -o <id> --o-book stock:Buccaneers` | Force the offensive book of record (`auto` default \| `custom` \| `stock:<name>`); `--d-book` for defense |
| `playbook --game madden27` | Print the full playbook of record (every formation + play live calls may use) |
| `play --game madden27 -o <id>` | Typed live loop + overlay (`~/.cfb-coach/madden27_overlay.html`), hard-locked to the book |
| `call --game madden27 -o <id> -s "d 3&8" --why` | One-shot call |
| `postgame --game madden27 -o <id> [--franchise lab]` | Learn from snaps; macro `proven`/`failed`; lab → primary promotions |
| `promote --game madden27 [--accept-all \| --target X]` | Review/accept lab → primary promotions |
| `config --game madden27 --primary-team X [--lab-team Y]` | Set/clear Franchise teams (TBD default) |

**Franchise profiles** (`--franchise`, stored per Madden DB like `--dynasty` in CFB): `primary` (serious save; id `franchise_primary`) and `lab` (optional practice save; id `franchise_lab`). Strong lab postgame results record promotions for the primary profile. `--dynasty` stays CFB-only and `--franchise` stays Madden-only; the CLI errors if you mix them.

**Playbook of record (every prep chooses).** Each prep locks one book per side, and live `play` / `call` may only use formation + play pairs from it. If a situation's menu has nothing in the book, the caller picks from another in-book menu; it never soft-warns.
- **Stock:** an existing in-game book by exact name. Offense: `Buccaneers` (Gun Doubles Clamp Stack, Gun Trips X Nasty, Gun 5WR Tight, Pistol Trips) or `Shotgun Classic` (Clamp Stack + 5WR Tight). Defense: `49ers` (Saleh 4-3: Nickel Over, Nickel Single / Double Mug, Dime 3-2 Odd, 4-3 Over). Nothing to build.
- **Custom:** on the first build, or when switching to custom, prep lists **every formation** to install (with its plays and source books). Later preps on the custom book show only **ADD / REMOVE of whole formations**, e.g. `ADD Gun Tight (Texans)` / `REMOVE Pistol Deuce Close` when the opponent changes.
- **Auto rule:**
  - Stay on the current book if it covers this opponent.
  - Otherwise pick a stock book that has everything (preferring your primary team's book).
  - Otherwise go custom (core Bucs-formation pack + the persona formation: Pistol Deuce Close vs split-field, Texans Gun Tight vs pressure, Gun Off Trips Close vs C2/C3 mixers).
  - Once custom, it stays custom (no rebuild churn).
- **Switch any time:** `--o-book stock:"Shotgun Classic"`, `--o-book custom`, `--d-book stock:49ers`. `playbook --game madden27` and the prep page's **Show full playbook** list the whole book. Optional audible-slot tweaks show up as tips, never as install steps.

**Primary team TBD.** Books come from the verified meta catalog until you set a team. Once the primary team is set, prep prefers that team's stock book when it's in the catalog. Personas and call language don't change. Config lives in `~/.cfb-coach/madden27_config.json`; `CFB_COACH_MADDEN_PRIMARY_TEAM` / `CFB_COACH_MADDEN_LAB_TEAM` override it.

**Shared personas.** Same cast as CFB (gavin, quen, tiano, gio, michael, harrison, jaxon, ryan, james, cpu) and the same aliases. Madden reuses only the persona **archetype + traits**, never CFB playbook/concept names. Madden film confidence starts one step below the CFB persona confidence (no Madden film yet).

**Doctrine (same as CFB):** calls stay inside the locked playbook; two reads on O, one user job on D; one tell = log + mild bump; macros arm only on a **REPEATED live** tendency (2+ this game), never from a previous-snap tell; PIVOT after 2 fails (user) / 3 fails (hard). **CPU = offense-only.** **User games = O + D** with a hard **8-macro O+D cap** (default Active-8 = MATCH-4, FLAT-CAP, MESH-RAT, STACK, SPY, RUN-FIT + O-PROT, O-MAN; benched HEAT, O-RPO).

```text
[O] sit> 1&10 my 35 stick wheel
heard: 1&10 yl35 [prev:Stick Wheel]
Gun Doubles Clamp Stack — Inside Zone | No adj | Front → Cutback
[O] sit> d 2&6 showing 4 verts
heard: 2&6 [live:Four Verticals]
Nickel Over — Cover 4 Quarters | none | User #3 seam
  SUGGEST macro: MATCH-4 [meta_grounded] — after one more Four Verticals tell
```

**Meta snapshot `madden27-2026-09`, cross-checked 2026-09-23 (soft priors, will drift).** Checked against Madden Prodigy, TimeSaver, Civil.GG and Operation Sports rankings, the Huddle.gg / madden.tools playbook databases, EA's gameplay deep dive and the 2026-09-03 / 2026-09-16 title-update notes. The live calls use only play names found in those formation lists; anything not found is tagged **"unverified name"** in the prep inventory. What the check changed from the original brief:
- **Fixed names:** Mtn Shuffle **Verts** Smash; Pistol Trips (no "RZ" formation); HB Zone WK.
- **Formations:** the Saleh book has no "Nickel Mug", so Nickel Over is home, with Single / Double Mug as disguise.
- **Pressure:** "DB Fire 2" wasn't found, so the pressure calls are Nickel Sim 2 / Field Sim 3.
- **Added:** Texas Y-Stutter Wheel and Mesh Post (the #1 and #2 meta plays).
- **Contested:** "Bucs = #1 offense" is Prodigy's launch pick; later rankings lead with Shotgun Classic / Texans.

Macros are Madden 27 **Custom Adjustments** (EA: Create & Share → Custom Adjustments, 20 saved per side, 10 active, LB in-game). Aidan's cap of 8 applies, same as CFB. EA doesn't publish the exact option labels, so every setting is tagged **approx**, and no button sequences are invented. Everything starts `meta_grounded`; postgame marks a macro `proven` (3+ uses, ≥60% success) or `failed` (≤30%). Full verdict table: `cfb_coach/data/madden27/VALIDATION_NOTES.txt`. Refresh after any title update with `prep --game madden27 --refresh-meta`.

**Files (separate namespace, same data dir):** `madden27.db` (override `CFB_COACH_MADDEN_DB`), `prep_madden27_<opp>.html`, `madden27_overlay.html`, `madden27_meta_cache.json`, `madden27_config.json`. CFB's `coach.db` / `prep_<opp>.html` / `copilot_overlay.html` are never touched by Madden runs.

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

## Play loop (typed live + overlay) — v1.9.6

**Typing-only live play** is the supported game-day path. Vision `watch` screen-watching is **on hold**; keep the same browser overlay for a big PLAY glance.

```bash
PYTHONPATH=. python3 -m cfb_coach play --opponent gavin
# overlay default ON → ~/.cfb-coach/copilot_overlay.html (auto-opens once)
# PYTHONPATH=. python3 -m cfb_coach play --opponent gavin --no-overlay
```

```text
[O] sit> 1&10
heard: 1&10
Gun Bunch X Nasty — Inside Zone | No adj | Front → Cutback
[O] sit> 1&10 my 35 mesh spot
heard: 1&10 yl35 [prev:Mesh Spot]
Gun Bunch X Nasty — HB Base | No adj | Front → Cutback
[O] sit> 2&7 deep flood
heard: 2&7 [prev:Deep Flood]
Gun Bunch X Nasty — Mesh Spot | No adj | Spot → Drag
[O] sit> 1&10 cover 2
heard: 1&10 [prev:Cover 2]
Gun Bunch X Nasty — Inside Zone | No adj | Front → Cutback
[O] sit> 1&10 showing cover 2
heard: 1&10 [live:Cover 2]
Gun Bunch X Nasty — Mesh Spot | No adj | Spot → Drag
[O] sit> result +4 run
  logged: +4 run
[O] sit> quit
```

**Aidan UX (CPU offense typing):** type **down/distance** (+ optional yardline) + **previous play name only**. Do **not** say `last`.

| You type | Coach hears |
|---|---|
| `1&10 my 35 mesh spot` | `[prev:Mesh Spot]` — mild bump only |
| `2&7 deep flood` | `[prev:Deep Flood]` |
| `1&10 cover 2` | `[prev:Cover 2]` |
| `1&10 showing cover 2` | `[live:Cover 2]` — soft live lean |
| `1&10 live cover 2` / `pre-snap c2` / `aligned c2` | live |

Bare book names map to concept hints: Mesh Spot, Deep Flood, Mtn RPO, HB Base, Counter Y, Inside Zone, Four Verticals, Cross Wheels, Whip Trail, etc.

- Each `sit>` line prints `heard:` then a **PLAY** call and refreshes the overlay (big call text).
- Yardlines: `my 35` / `our 35` / `ball on 35` (from own goal); `opp 40` / `their 25` (opp yardline → 100−N).
- **Prev vs live:** bare coverage/play name = **prev** (mild bump, never hard-counter). Only `showing` / `live` / `pre-snap` / `aligned` / `they're in` force **live**.
- CPU (`--opponent cpu`) stays **offense-only**; user opponents still accept `d ` for defense.
- `result <text>` / `log <text>` updates SQLite tendencies.
- `why` prints the last call's rationale.
- After 3 failed snaps on a side, live caller tags **PIVOT:** and switches family/macro plan (no hero-shot whiplash).
- `--no-overlay` disables the HTML strip; default is **ON** for interactive play.

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



## Screen Co-Pilot / vision watch (on hold)

> **v1.9.6:** Game-day live = **`play` + typed situations + overlay**. Vision screen-watching (`watch --window` / capture) is **on hold** — overlay HTML is reused by typed `play`.

**Live UX (when watch returns):** pre-snap cadence is **PLAY → ADJUST → HIKE** (playcaller `Formation — Play | Adj | Reads`), not generic “base look” tips as the main output. Pressure/shell changes print short **ADJUST** lines without replacing the play unless audible is clearly warranted (e.g. Cover 0). When the look is stable (~1.8s) or timed out (~4.5s), coach prints **HIKE — go** — you snap; coach never presses buttons.

**Remote Play focus:** Xbox Remote Play often **pauses/freezes when unfocused** (e.g. PowerShell focused). Keep the Remote Play / Xbox stream window focused and visible. Use the **HTML overlay** (`~/.cfb-coach/copilot_overlay.html`; auto-opens from `play` or live watch) on the other half of the screen for a big PLAY glance. Capture card later removes this. Disable with `--no-overlay`.

Heartbeat: single overwritten line (\\r) ≤ every **5s** waiting for frames; troubleshoot once at 5s, then every **15s**. Status spam reduced when PLAY unchanged (longer gap after HIKE). `--list-windows` / dxcam→mss fallback unchanged.


**Doctrine:** Aidan keeps **full Xbox control** on the HDMI monitor/console. The Windows laptop is a **sidecar** — it runs Xbox Remote Play so coach can **see** the game, analyze locally (classical CV, ~5–10 FPS, no LLM per frame), and suggest **pre-snap adjustments** only. **Never auto-play / never press buttons.**

### Capture architecture

| Path | Role |
|------|------|
| **Xbox Remote Play** on Windows laptop | **Active prototype video source.** Play stays on direct HDMI monitor + Xbox controller. |
| **HDMI capture card** (Elgato Cam Link / HD60 / 4K X / …) | **Future drop-in** `CaptureBackend` (same pipeline). |
| `--demo` + typed hotkeys | Always available (Linux/WSL/Windows) — no vision deps. |
| `--image` / `--video` | Offline pipeline / naive heuristic testing. |

**Honesty:** Cover 3 vs Quarters (and fine shell calls) is hard from pixels. Milestone 1 = coarse **formation / shell / pressure / play_state**. Milestone 2 adds snap/play-end, PlayRecord, and this-game live tendencies. Hotkeys remain first-class.

### Try it (WSL / Linux — no vision extras)

```bash
cd /workspace/cfb-coach   # or your clone
PYTHONPATH=. python3 -m cfb_coach watch --demo --once
PYTHONPATH=. python3 -m cfb_coach watch --demo --overlay   # also writes ~/.cfb-coach/copilot_overlay.html
PYTHONPATH=. python3 -m cfb_coach watch                    # interactive: type looks
PYTHONPATH=. python3 -m cfb_coach watch --image tests/fixtures/synthetic_field.png --once
```

Interactive injection examples:

```text
look> look nickel two_high none
look> f nickel
look> s c3
look> p blitz_left
look> c3 heat
look> demo
look> help
```

Tips are ≤3 short lines, e.g. `slide protect left`, `hot X ready`, `two-high → run first`. Reuses macro inventory language (PROT / ZERO / C3 / RUN / Active-8 names) — does **not** change prep/play/call/meta scout.

Overlay (default ON for live window/region): auto-opens `~/.cfb-coach/copilot_overlay.html` (refresh ~1.5s) — pin on the other half of the screen; **keep Xbox Remote Play focused**. `--no-overlay` to disable.

## Xbox Remote Play Vision

Live vision Milestone 1 — **sidecar only**. Play on HDMI; laptop sees Remote Play.

1. **Enable Xbox remote features** — Xbox Settings → Devices & connections → Remote features → enable.
2. **Open Remote Play on the Windows laptop** — Xbox app → Remote Play → connect to your console.
3. **Keep playing via direct HDMI monitor** with the Xbox controller. Laptop is tips-only.
4. **Install vision extras on native Windows Python** (not WSL for live dxcam grab):
   ```bat
   cd cfb-coach
   pip install -e ".[vision]"
   ```
5. **Calibrate** (saves window title + crop to `%USERPROFILE%\.cfb-coach\vision_calib.json`):
   ```bat
   cfb-coach watch --calibrate
   ```
6. **List window titles** (if dxcam gets zero frames):
   ```bat
   cfb-coach watch --list-windows
   ```
7. **Watch live:**
   ```bat
   cfb-coach watch --window "Xbox" --debug
   ```
   dxcam tries first; if the Xbox app yields no frames (~2–3s), coach auto-falls back to mss on the matched window rect (or calib crop). **Xbox app may need `--screen-region` / mss fallback** (UWP/protected content).
   Calib crop via mss: `cfb-coach watch --screen-region` (prints `capture=mss`).
   Other useful flags: `--screen-region L,T,W,H`, `--tts`, `--video sample.mp4`, `--image test.png --debug`.
8. **Debug view meaning** — OpenCV window shows the captured frame, analyzed FPS, crop, HUD/field ROIs, and short state text (`3&7 | BUNCH R | 2-HIGH | PRESSURE L`). Console prints the same short live line + CALL tips.
9. **Xbox control is separate from vision** — coach never injects controller input / auto-play.
10. **Remote Play = prototype source** — convenient because the laptop already mirrors the game while you play on the HDMI monitor.
11. **Capture card = future drop-in backend** — same `CaptureBackend` protocol / pipeline; swap Remote Play window grab for a device/index later.

Full notes anytime: `cfb-coach watch --setup`.

Live terminal cadence:

```text
1&10 | BUNCH | 2-HIGH | PRESSURE ?
PLAY
Gun Bunch X Nasty — Mesh Spot | No adj | Spot → Drag
…
ADJUST  pressure show left — slide L, hot ready
HIKE — go
```

Overlay (secondary): big PLAY text in the browser strip; keep Xbox focused.



## Live Vision Milestone 2 (on hold; last ship v1.9.4)

**Extend M1 — do not rewrite.** Sidecar only; Remote Play = prototype capture; same pipeline for `--video`. False snaps worse than late. Sample tiers: **1=log, 2=mild, 3+=actionable, 5+=strong**. Recency windows (last 5 / last 8) + full-game. Anti-whiplash preserved.

### What M2 adds

| Piece | Role |
|-------|------|
| Play state machine | `UNKNOWN/MENU/BETWEEN_PLAYS/PRE_SNAP/PLAY_ACTIVE/PLAY_ENDING/POST_PLAY` (+ M1 compat) with timestamps |
| `snap_detect` / `play_end` / `play_tracker` | Multi-signal snap + debounced end → one `PlayRecord` per snap |
| `play_family` / `result_detect` | Coarse families + results; yards only from confident HUD delta (never fabricate) |
| `session` + DB | `game_sessions`, `play_records`, `live_tendency_events` (additive migrations) |
| `live_tendency` | This-game contextual buckets, alerts, counter validation → feeds playcaller |
| Watch UX | Threaded capture (drop stale), compact LIVE GAME overlay, R/P/S/I/X corrections, optional `--record-plays` |
| `postgame --report` | Concise tendency summary from session play_records |

### Windows — first live test checklist

```bat
cd cfb-coach
pip install -e ".[vision]"
cfb-coach watch --calibrate
cfb-coach watch --window "Xbox" --opponent gavin --dynasty alabama --debug
:: overlay auto-opens (big PLAY). Keep Xbox Remote Play focused.
:: optional: --no-overlay   or   --screen-region
:: optional clips when low-conf / explosive:
cfb-coach watch --window "Xbox" --opponent gavin --record-plays --debug
cfb-coach postgame -o gavin --report
```

**Checklist**

1. Remote Play open on laptop; play on HDMI + Xbox controller (coach never drives sticks).
2. Calibrate window/crop once (`--calibrate`).
3. Confirm debug view shows FPS, state, short line (`3&7 | BUNCH R | 2-HIGH | …`).
4. Confirm console shows **PLAY** (real Formation — Play | Adj | Reads), then **ADJUST** / **HIKE — go** (not generic base-look tips as the main line).
5. After a few snaps, verify `~/.cfb-coach/coach.db` has `game_sessions` / `play_records` rows.
6. Manual correct last play: type `R` / `P` / `S` / `I` / `X` or `correct family CROSSERS`.
7. Postgame: `cfb-coach postgame -o <opp> --report` for tendency summary.
8. `--video sample.mp4` uses the **same** analysis path as live (no Xbox required for dry runs).

### Linux / WSL smoke (no vision extras)

```bash
cd /workspace/cfb-coach
PYTHONPATH=. python3 -m cfb_coach watch --demo --once
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

### Known limitations (honest)

- **HUD OCR is weak** — down/distance/score/yards often unknown; we never invent yards.
- **Concept families are coarse** — CROSSERS/VERTICALS/MESH/… tags, not exact route grading.
- **Field / formation accuracy is unmeasured** on real Remote Play; treat live looks as soft evidence.
- **Remote Play often pauses when unfocused** — keep stream focused; use overlay strip; capture card later.
- No LLM every frame, no controller automation, no huge NN training.
- Cover 3 vs Quarters still hard from pixels (M1 honesty unchanged).

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
    vision/             # DefenseLook + GameObservation + live CV + M2 play tracker
    live_tendency.py    # this-game LiveTendency engine (M2)
    counter_evidence.py # tendency API for playcaller
    session.py          # game session ids for watch
    copilot.py          # pre-snap tip engine + tiny HTML overlay
    watch.py            # `watch` / `copilot` live loop
    data/seed.json
    data/meta_baseline.json  # CFB27 META (cfb27-2026-09)
    games.py            # --game registry (cfb27 default | madden27) + per-game paths
    madden/             # Madden 27 Franchise: data, situation, playcaller, prep(+browser),
                        #   meta_scout, franchise (primary/lab + team config), postgame, cli
    data/madden27/      # seed / meta_baseline (madden27-2026-09) / macro_catalog / VALIDATION_NOTES
```

## Smoke

```bash
PYTHONPATH=. python3 -m cfb_coach prep --opponent cpu --dynasty ohio_state --offline --no-open
PYTHONPATH=. python3 -m cfb_coach prep --opponent cpu --dynasty ohio_state --no-open   # live scout section even if some URLs 404
PYTHONPATH=. python3 -m cfb_coach watch --demo --once
PYTHONPATH=. python3 -m cfb_coach watch --setup
PYTHONPATH=. python3 -m unittest discover -s tests -v
# or: pip install pytest && pytest -q
PYTHONPATH=. python3 -m cfb_coach postgame -o cpu --report   # tendency report if session logged
```

Offline prep must succeed. M2: all old + new tests pass; `watch --demo --once` still works. With network, prep HTML includes **Live meta scout** (fetch budget ≤~12s). `watch --demo --once` must print sample pre-snap tips (stdlib only — no dxcam/cv2 required on Linux).
