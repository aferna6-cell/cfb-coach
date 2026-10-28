# Coordinator demo

Down-and-distance lines are synthetic. Formations and plays come from Demo C's trimmed Buccaneers and 49ers book. No VOD prior is enabled (`CFB_COACH_NO_VOD_PRIOR=1`). `CFB_COACH_DB` and `CFB_COACH_MADDEN_DB` are separate temp files. TEST FIXTURE - not an Aidan-verified macro.

## Independent coordinators

Offense and defense are separate coordinators. Each keeps its own last line. The offense sequence is called again with the same seeds after the defensive inputs change. Defense `live_pick` reads the defensive book and does not log a snap, so it does not change the offense read of the fixture.

Offense lines unchanged: 5/5.
Offense before:
- `1&10`: Gun Doubles Clamp Stack — Texas Y-Stutter Wheel | No adj | Y Stutter → HB Wheel
- `2&7`: Gun 5WR Tight — QB Draw | No adj | Primary → Checkdown
- `3&2`: Pistol Trips — HB Stretch | No adj | Edge → Cut up
- `1&10 my 40`: Pistol Trips — Strong Power | No adj | Primary → Checkdown
- `3&8`: Pistol Deuce Close — Counter Y | No adj | Primary → Checkdown
Defense inputs changed:
- d 3&6 -> 4-3 Over / Free Fire
- d 2&8 -> 4-3 Over / Cover 3 Sky
- d 1&10 -> Nickel Double Mug / Coffee House 3 Seam
Offense after:
- `1&10`: Gun Doubles Clamp Stack — Texas Y-Stutter Wheel | No adj | Y Stutter → HB Wheel
- `2&7`: Gun 5WR Tight — QB Draw | No adj | Primary → Checkdown
- `3&2`: Pistol Trips — HB Stretch | No adj | Edge → Cut up
- `1&10 my 40`: Pistol Trips — Strong Power | No adj | Primary → Checkdown
- `3&8`: Pistol Deuce Close — Counter Y | No adj | Primary → Checkdown

## Sequence

- offense `1&10` (synthetic) — Gun Doubles Clamp Stack / Texas Y-Stutter Wheel
  - old: Gun Doubles Clamp Stack — Texas Y-Stutter Wheel | No adj | Y Stutter → HB Wheel
  - live: Gun Doubles Clamp Stack — Texas Y-Stutter Wheel | No adj | Y Stutter → HB Wheel
  - why: same line
- offense `2&7` (synthetic) — Gun 5WR Tight / QB Draw
  - old: Gun 5WR Tight — QB Draw | No adj | Primary → Checkdown
  - live: Gun 5WR Tight — QB Draw | No adj | Primary → Checkdown
  - why: same line
- offense `3&2` (synthetic) — Pistol Trips / HB Stretch
  - old: Pistol Trips — HB Stretch | No adj | Edge → Cut up
  - live: Pistol Trips — HB Stretch | No adj | Edge → Cut up
  - why: same line
- offense `1&10 my 40` (synthetic) — Pistol Trips / Strong Power
  - old: Pistol Trips — Strong Power | No adj | Primary → Checkdown
  - live: Pistol Trips — Strong Power | No adj | Primary → Checkdown
  - why: same line
- offense `3&8` (synthetic) — Pistol Deuce Close / Counter Y
  - old: Pistol Deuce Close — Counter Y | No adj | Primary → Checkdown
  - live: Pistol Deuce Close — Counter Y | No adj | Primary → Checkdown
  - why: same line
- defense `d 1&10` (synthetic) — Nickel Single Mug / Cover 4 Quarters
  - old: Nickel Single Mug — Cover 4 Quarters | none | User #3 seam
  - live: Nickel Single Mug — Cover 4 Quarters | No adj | User #3 seam
  - why: heuristic defense prints none for an empty adjustment; the live line prints No adj
- defense `d 3&6` (synthetic) — 4-3 Over / Cover 4 Quarters
  - old: 4-3 Over — Cover 4 Quarters | none | User #3 seam
  - live: 4-3 Over — Cover 4 Quarters | No adj | User #3 seam
  - why: heuristic defense prints none for an empty adjustment; the live line prints No adj
- defense `d 2&6` (synthetic) — Nickel Over / Cover 1 Hole
  - old: Nickel Over — Cover 1 Hole | none | User robber
  - live: Nickel Over — Cover 1 Hole | No adj | User robber
  - why: heuristic defense prints none for an empty adjustment; the live line prints No adj
- defense `d 1&10 my 25` (synthetic) — Nickel Over / Cover 4 Drop Field
  - old: Nickel Over — Cover 4 Drop Field | none | User the deep middle (rob the post)
  - live: Nickel Over — Cover 4 Drop Field | No adj | User the deep middle (rob the post)
  - why: heuristic defense prints none for an empty adjustment; the live line prints No adj

## Macro lifecycle

Source: Demo C (`cfb_coach.madden.adjustments.demo`), design version 1.
Checked-in data has no Aidan-verified live macro. Catalog rows marked confirmed are Aidan's setting notes, not a MacroState record. This run uses Demo C's generated macro. TEST FIXTURE - not an Aidan-verified macro.
Macro used: ADJ-DEF-4F5ED0CB0F (Dime 3-2 Odd / DB Fire 2). TEST FIXTURE - not an Aidan-verified macro.
Draft refused: ADJ-DEF-4F5ED0CB0F. ADJ-DEF-4F5ED0CB0F is draft and is not live-selectable
Draft line (macro absent): Dime 3-2 Odd — DB Fire 2 | No adj | User hook
Armed line: Dime 3-2 Odd — DB Fire 2 | ADJ-DEF-4F5ED0CB0F | User hook
select_live armed: ADJ-DEF-4F5ED0CB0F state armed
Unarmed Demo C play falls back to baseline BASE-OFF-85936814A2 is_no_adjustment=True (Pistol Trips / HB Dive)

## Latency

ml.latency_budget_ms stays 150.
Calls timed: 50.
Whole call (make_call plus the model call) ms: p50 2.473, p95 2.682, max 2.729.
Model call only (defense live_pick elapsed_ms, offense rank_live latency) ms: p50 0.384, p95 1.285, max 1.324.
Fallback count: 25. A fallback is any model result whose MLStatus is not ok.
MLStatus counts: model_missing=25, ok=25
Fallback model_missing: no model artifact is installed in the temp registry, so the live rank keeps the heuristic line (25).

## Diff versus the heuristic caller

4 of 9 displayed lines differ from `make_call().format()` (the heuristic caller at df81381). The live line is `Formation — Play | Adjustment | Read 1 → Read 2` on offense and `Formation — Play | Adjustment/Macro | User responsibility` on defense.

- defense d 1&10: heuristic defense prints none for an empty adjustment; the live line prints No adj
  - old: Nickel Single Mug — Cover 4 Quarters | none | User #3 seam
  - live: Nickel Single Mug — Cover 4 Quarters | No adj | User #3 seam
- defense d 3&6: heuristic defense prints none for an empty adjustment; the live line prints No adj
  - old: 4-3 Over — Cover 4 Quarters | none | User #3 seam
  - live: 4-3 Over — Cover 4 Quarters | No adj | User #3 seam
- defense d 2&6: heuristic defense prints none for an empty adjustment; the live line prints No adj
  - old: Nickel Over — Cover 1 Hole | none | User robber
  - live: Nickel Over — Cover 1 Hole | No adj | User robber
- defense d 1&10 my 25: heuristic defense prints none for an empty adjustment; the live line prints No adj
  - old: Nickel Over — Cover 4 Drop Field | none | User the deep middle (rob the post)
  - live: Nickel Over — Cover 4 Drop Field | No adj | User the deep middle (rob the post)

