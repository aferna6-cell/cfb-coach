# Madden NFL 27 defensive coverage and pressure gap audit — 2026-11-07

## Scope, evidence, and meaning

- Source of truth for **our installed defense**: `cfb_coach/data/madden27/installed_playbook_snapshot.json` (DEF-F1CF-1862; 6 formations; **123 formation–play entries**, 82 unique names).
- Runtime/historical recognizer audited: `cfb_coach/madden/model/historical_defense_baseline.py` `ALIASES` / `normalize_label` at PR #112. A name not in its alias table becomes `unknown` unless generic pressure terms exactly match. **38 / 123 installed entries** resolve to a named coverage by exact normalizing; **85 / 123** do not. This is an **exact-name recognition audit**, **not** a percentage of football understanding and not a measurement of winning.
- Offline football explanations: `coverage_attack_knowledge.json` has **20** scheme entries and is not equivalent to exact player responsibility/pressure knowledge.
- Names must not automatically establish number of rushers, which defender drops, coverage check, coverage side, or which concept will succeed. Distinguish **source text**, *hypothesized coverage*, *hypothesized pressure*, and **verified screenshot/game behavior**.
- User's offensive play-art change in this PR allows two **user-assumed** name aliases, but does not certify their exact in-game Xbox route assignments.

## Gaps by category (name normalization only)

- Coverage variants / rotations / spy / match: **28** unrecognized installed formation–play entries
- Pressure, blitz, simulated pressure / other named defense: **47** unrecognized installed formation–play entries
- Man double teams / brackets: **10** unrecognized installed formation–play entries

## Every unrecognized installed call by formation (85 entries)

| Installed formation | Unrecognized exact defensive play names | Count |
|---|---|---:|
| 4-3 Over | `1 Hole Press`, `1 Robber Press`, `Cover 2 Hard Flat`, `Cover 3 Cloud Str`, `Cover 4 Press`, `DBL Safety Blitz`, `Free Fire`, `LB Sting 1`, `OLB Fire Man`, `Sam Edge 3`, `Thunder Smoke` | 11 |
| 46 Normal | `Engage Eight`, `Fire Zone 3`, `Inside Blitz`, `Rush Outside` | 4 |
| Dime 3-2 Odd | `1 Double WR1`, `1 Double WR2`, `Cover 1 Robber Press`, `Cover 3 Buzz Match WK`, `Cover 3 Cloud`, `Cover 3 Hard Flat`, `Cover 3 Lock`, `Cover 6 Willie`, `DB Fire 2`, `DBL Safety Go`, `Double Bracket`, `Edge Blitz 1`, `Edge Blitz 3`, `FS Zone Blitz`, `Hot Blitz 3`, `Okie Eagle 3`, `Okie Roll 2`, `Overload 3 Seam`, `SS Zone Blitz`, `Tampa 2 Drop`, `Zero Blitz` | 21 |
| Nickel Double Mug | `2 Invert Hard Flat`, `Blitz Loop 3`, `Coffee House 3 Seam`, `Field Sim 3`, `LB Blitz 3`, `Mid Blitz 0`, `Nickel 2 Trap`, `Nickel Overload 3`, `Nickel Sim 2`, `SS Blitz 3` | 10 |
| Nickel Over | `1 Double Slot`, `1 Double TE`, `1 Double WR1`, `1 Double WR2`, `1 Robber Press`, `Bracket Switch Willie`, `Cov 3 Buzz Match`, `Cover 2 Roll`, `Cover 3 Buzz Match`, `Cover 3 Cloud`, `Cover 4 Drop Field`, `Cover 6 Willie`, `Cover 9 Show 2`, `DBL Safety Blitz`, `Double Bracket`, `Double Bracket Switch`, `Edge Blitz 0`, `Edge Blitz 3`, `Field Sim 3`, `FS Will Blitz 0`, `LB Blitz 1`, `Nickel 2 Trap`, `Nickel Blitz 3`, `Nickel Sim 2`, `Over Storm Brave`, `Overload 3 Seam`, `Overload Blitz`, `Roll 23 Cloud`, `Roll Cover 2 Invert`, `Roll Cover 3 Cloud`, `Show SS Sim 2`, `Sim Pressure 3` | 32 |
| Nickel Single Mug | `1 LB Dog`, `Blitz Tex 3 Sim 3`, `Cover 3 Buzz Match`, `DB Blitz Tex 3`, `DT Mike Loop 3`, `Nickel 2 Trap`, `Nickel Blitz 3` | 7 |

## High-priority missing defensive **coverage variants** (not just blitz names)

1. **Cover 3 family**: Cloud; Cloud Str; Roll Cover 3 Cloud; Roll 23 Cloud; Buzz Match and Buzz Match WK (also `Cov 3 Buzz Match`); Hard Flat; Lock; and externally observed Buzz Press, Buzz Seam, Mable / double Mable Cloud, Cloud Show 2, Sky WK, Cover 3 Drop and simulated-pressure Cover 3. The present recognizer knows generic Cover 3, Sky, and Match but fails these more specific strings.
2. **Cover 2 / two-high**: Cover 2 Hard Flat, Cover 2 Roll, 2 Invert Hard Flat, Roll Cover 2 Invert, Tampa 2 Drop, 2 Trap / Nickel 2 Trap, plus externally observed Cover 2 Match, Drop, Man Spy, Cover 2 Spy. It currently knows basic Cover 2 Man, Spot, Invert, Tampa 2 and Trap 2, but not all those concrete installed spellings/variations.
3. **Cover 4 / split-safety**: Cover 4 Press and Cover 4 Drop Field in our book; externally observed Cover 4 Show 2 and Cover 4 Zone. Quarters, Palms and Drop have only limited generic concepts and not verified check-by-check responsibility behavior.
4. **Split-field / hybrid**: Cover 6 Willie, Cover 9 Show 2, externally observed Cover 6 Zone, Cover 9 Zone. Distinguish field/boundary, roll/show and pattern-matching behavior when evidence exists.
5. **Cover 1 / man variations**: 1 Hole Press, 1 Robber Press, Cover 1 Robber Press, 1 Double Slot, 1 Double TE, 1 Double WR1/WR2, Double Bracket, Double Bracket Switch, Bracket Switch Willie; externally observed Cover 1 Spy, Thief, Sting. These cannot be reduced to plain Cover 1 without losing man-help and bracket intent.
6. **Prevent and specialist coverages**: Prevent, Man 3 Deep, Cover 2 Drop, 3 Double Buzz, 3 Double Mable, 3 DBL Mable Cloud, other goal-line/short-yardage designs (check exact roles before assigning a schematic family).

## Pressure / blitz gaps (independent axis, NOT extra Cover 0 classes)

- **Zone/fire-zone blitz** examples: Fire Zone 3, FS Zone Blitz, SS Zone Blitz, DB Fire 2, Edge Blitz 3, Hot Blitz 3, Nickel Blitz 3, LB Blitz 3, DB Blitz Tex 3, plus Overload 3 Seam and Coffee House 3 Seam. `3` in a name is only a clue; preserve coverage unknown until sourced.
- **Simulated pressure / creepers / disguise**: Field Sim 3, Nickel Sim 2, Sim Pressure 3, Show SS Sim 2, Blitz Tex 3 Sim 3, DT Mike Loop 3, Roll 23 Cloud. A simulated pressure may rush four with nonstandard identities; avoid marking it as a five-man blitz without proof.
- **Man pressure / zero**: Mid Blitz 0, Zero Blitz, Edge Blitz 0, FS Will Blitz 0, Over Storm Brave, Overload Blitz, Engage Eight, Free Fire, OLB Fire Man, 1 LB Dog and other named pressures. Do not convert *any* generic `blitz` observation into Cover 0; coverage is a separate field.
- **Other edge / interior / safety pressure names**: DBL Safety Blitz, DBL Safety Go, LB Sting 1, Sam Edge 3, Thunder Smoke, Inside Blitz, Rush Outside, Blitz Loop 3, Nickel Overload 3, SS Blitz 3, Nickel Blitz 3, Overload 3 Seam, Edge Blitz 1, Edge Blitz 3, Okie Eagle 3, Okie Roll 2. Unknown by current exact recognizer; source-check each assignment.

## Madden 27 features that are NOT simply extra coverage classes

EA gameplay documentation identifies dynamic match checks — **Box, Bingo, Triangle, Point Triangle, Skate, Skinny, Skinny Meg, Stress, Stubbie, Stump, Solo, Solo Cut, Zone It** — and additional man switches against stacks/bunches, Smart Zones, plaster logic, Roll Coverage, alignment/leverage and matchup focus. Some are conceptually documented in #107, but **not observed as current defender-by-defender execution states**. Model these as **coverage behavior/adjustment axes** when actually observed; do not inflate the coverage class count or assume they are active every snap.

## Additional Madden 27 plays outside the user's installed defense

The Madden 27 Multiple D public play catalog includes, among others, `Cover 2 Match`, `Cover 3 Mable`, `3 DBL Mable Cloud`, `Cover 3 Cloud Show 2`, `Cover 3 Sim Pressure`, `Cover 3 Sammie`, `Cover 4 Show 2`, `Cover 1 Sting`, `Cover 1 Thief`, `Cover 1 Spy`, `Cover 2 Spy`, `Cover 2 Man Spy`, `Cover 2 Drop`, `Cover 4 Zone`, `Man 3 Deep` and `Prevent`. This is **illustrative, not an exhaustive all-team Madden 27 catalog**. Other human franchise opponents may have more options than our current installed 123 calls.

## Recommended integration strategy

1. Preserve the exact raw post-play revealed call string. Map into **separate axes**: base deep structure, variant/check, match vs drop vs man, pressure type, pressure origin/rusher count if verified, rotation/show, front/formation, provenance and confidence.
2. Normalize high-frequency exact Madden names first; preserve `unknown` for unsourced roles instead of treating a name as proof. Allow pressure+coverage coexistence (e.g., a zone blitz behind three deep) and preserve generic Cover 3 as ambiguous.
3. Reuse existing result/coverage entry, and use only **earlier closed snaps** to update next-snap opponent beliefs. Do not leak current play's reveal to current play's call.
4. Evaluate name recognition separately from football-value gains: unit tests for mapping, then held-out predictive calibration, then CPU franchise comparison; never claim automatic performance from adding aliases.
5. Source/legal note: public third-party play-name catalogs can be cited for research; **do not scrape or reproduce restricted third-party play art or route assignments into ML features** without permission. Favor the user's own in-game screenshots for assignment evidence.

## Public sources

- [EA Sports Madden NFL 27 gameplay deep dive](https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay) — official defensive alignment, man/match checks, Smart Zones, roll and adjustments.
- [EA Help — Madden 27 playbooks guide](https://help.ea.com/en/articles/madden-nfl/madden-playbooks-guide/) — defensive playbook families and pressure/zone blitz approaches.
- [Madden Prodigy — Madden 27 Multiple D playbook](https://www.maddenprodigy.com/playbooks/madden-27-multiple-d-defense/) — reference list of named play variants not necessarily in the user's own playbook.

Research audit only: **no defensive coverage normalization, training artifact or live defensive caller changed by this report**.
