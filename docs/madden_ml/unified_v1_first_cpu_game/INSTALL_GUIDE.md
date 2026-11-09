# Madden playbook installation guide - cpu

**Nothing below is installed or confirmed.** These are proposals. After you build a book in Madden and CHECK it, run `confirm-install` for that side. The coordinator refuses to play a side whose book is not attested.

## OFFENSE - 15 formations, 324 plays

- proposed fingerprint: `OFF-INST-F8478F3356`
- confirmed fingerprint: **NOT CONFIRMED - only the owner can confirm after installing and checking in Madden**
- editor limits: formations 15/15, plays 324/500, sets 15/56, families 3/15 (third_party_reported_not_verified_in_game)
- unmet situational requirements: none
- selection basis: cold-start prior (uncalibrated): formation ranking is driven by situational coverage and catalog breadth, not learned strength
- previous book OFF-3C6D-E543: objective 0.88027 vs proposed 0.95997; formations kept from it: none

| # | formation (set) | source playbook | plays | editor status | why chosen |
|---|---|---|---|---|---|
| 1 | Gun Trips TE Flex Close Wk | Colts | 30 | UNVERIFIED_IN_EDITOR | adds +11.9075 portfolio value when selected (coverage +11.7775, requirement fill +0.1300); strongest in goal line and red zone; play mix: run 6, quick pass 7, deep shot 7, screen 2, play action 3 |
| 2 | Gun Trips TE | Seahawks | 33 | UNVERIFIED_IN_EDITOR | adds +0.0524 portfolio value when selected (coverage +0.0484, requirement fill +0.0040); strongest in goal line and red zone; play mix: run 6, quick pass 3, deep shot 6, screen 4, play action 4 |
| 3 | Gun Spread Y-Slot | Texans | 24 | UNVERIFIED_IN_EDITOR | adds +0.0182 portfolio value when selected (coverage +0.0183, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 4, quick pass 6, deep shot 3, screen 1, play action 4 |
| 4 | Singleback Wing Pair | Bengals | 15 | UNVERIFIED_IN_EDITOR | adds +0.0120 portfolio value when selected (coverage +0.0120, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 6, quick pass 3, deep shot 1, play action 2 |
| 5 | Singleback Bunch X Nasty | Vikings | 21 | UNVERIFIED_IN_EDITOR | adds +0.0057 portfolio value when selected (coverage +0.0057, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 6, quick pass 2, deep shot 4, screen 1, play action 10 |
| 6 | Gun Deuce Close | Spread Classic | 18 | UNVERIFIED_IN_EDITOR | adds +0.0046 portfolio value when selected (coverage +0.0046, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 3, quick pass 2, deep shot 5, play action 6 |
| 7 | Singleback Bunch TE | Commanders | 18 | UNVERIFIED_IN_EDITOR | adds +0.0042 portfolio value when selected (coverage +0.0042, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 8, quick pass 5, deep shot 1, screen 1, play action 2 |
| 8 | Gun Y Trips Wk | Bengals | 24 | UNVERIFIED_IN_EDITOR | adds +0.0040 portfolio value when selected (coverage +0.0040, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 3, quick pass 3, deep shot 3, screen 1, play action 2 |
| 9 | Gun Trio HB Wk | Benkert's Dimes | 21 | UNVERIFIED_IN_EDITOR | adds +0.0040 portfolio value when selected (coverage +0.0040, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 3, quick pass 3, deep shot 2, screen 1, play action 3 |
| 10 | Gun Trio Y-Flex | Browns | 21 | UNVERIFIED_IN_EDITOR | adds +0.0040 portfolio value when selected (coverage +0.0040, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 1, quick pass 4, deep shot 3, screen 1, play action 2 |
| 11 | Gun Box Offset Wk | Raiders | 15 | UNVERIFIED_IN_EDITOR | adds +0.0040 portfolio value when selected (coverage +0.0040, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 2, quick pass 1, deep shot 2, screen 1, play action 1 |
| 12 | Gun Y Off Trips Wk | Browns | 33 | UNVERIFIED_IN_EDITOR | adds +0.0022 portfolio value when selected (coverage +0.0022, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 8, quick pass 5, deep shot 3, screen 1, play action 2 |
| 13 | I Form Close | Lions | 12 | UNVERIFIED_IN_EDITOR | adds +0.0020 portfolio value when selected (coverage +0.0020, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 4, quick pass 2, deep shot 4, play action 5 |
| 14 | Gun Y Off Trio Close | 49ers | 24 | UNVERIFIED_IN_EDITOR | adds +0.0020 portfolio value when selected (coverage +0.0020, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 7, quick pass 3, deep shot 1, play action 4 |
| 15 | Gun Tight Doubles | Buccaneers | 15 | UNVERIFIED_IN_EDITOR | adds +0.0020 portfolio value when selected (coverage +0.0020, requirement fill +0.0000); strongest in goal line and red zone; play mix: run 2, quick pass 5, deep shot 1, screen 1, play action 2 |

### Step by step

1. **Gun Trips TE Flex Close Wk** - Playbook editor: family 'Shotgun' -> set 'Trips TE Flex Close Wk' -> source playbook 'Colts' -> keep ALL 30 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Colts', confirm set 'Trips TE Flex Close Wk' lists 30 plays
   - plays: Bounce Inside Zone, Bounce Read Option, Bounce Shallow Cross, China Y-Shake, HB Power, HB Slip Screen, Inside Zone, Mid Zone, Mtn Dagger, Mtn Fake Screen Wheel, Mtn HB Swing, Mtn Mesh Gos, Mtn Mesh Rail, Mtn Smash Slot Choice, Outside Zone, PA Deep Sail, PA Flood, Read Y Corner, Return PA Deep Sail, Return Post Curl, Return RPO Alert Swing, Return Sprint Comeback, Return Sprint Spot, Return Sprint TE Screen, RPO Read Bubble, Same Side Zone, Shallow Slot Fade, Stick, Switch Dagger, Verticals
2. **Gun Trips TE** - Playbook editor: family 'Shotgun' -> set 'Trips TE' -> source playbook 'Seahawks' -> keep ALL 33 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Seahawks', confirm set 'Trips TE' lists 33 plays
   - plays: 45 Quick Base, All Curl, China Y Post, Curl Flat, Dagger Texas, Double In Sail, Drive Post, Flood, HB Counter, HB Mid Draw, HB Slip Screen, Inside Zone, Mtn Bench, Mtn Return Levels, Mtn Z Spot, Orbit Cross Screen, Orbit PA HB Wheel, Orbit RPO Alert Screen, PA Boot Shot, PA Counter Go, PA Shot Wheel, PA Slot Corner, QB Strong Power, QB Zone, Read Option Bubble, RPO Alert Bubble, Speed Option, TE Middle Screen, TE Swirl, Verticals, Verts Y Delay, WR Short Post, Z Cross
3. **Gun Spread Y-Slot** - Playbook editor: family 'Shotgun' -> set 'Spread Y-Slot' -> source playbook 'Texans' -> keep ALL 24 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Texans', confirm set 'Spread Y-Slot' lists 24 plays
   - plays: Corner Strike, Curl Seams, Digs H Option, Double Slant, Four Verticals, HB Draw, HB Slip Screen, In N Up, Inside Zone, Levels Divide, Mid Zone, PA All Curl, PA Comebacks, PA Read, Quick Slants, Read Option, Return Curls Texas, Return Inside Zone Split, Return PA Post Curls, RPO Peek Slant, Slot Outs, Stick Nod, Y-Seam, Z Shallow Cross
4. **Singleback Wing Pair** - Playbook editor: family 'Singleback' -> set 'Wing Pair' -> source playbook 'Bengals' -> keep ALL 15 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Bengals', confirm set 'Wing Pair' lists 15 plays
   - plays: 0 1 Trap, Dragon Spacing, HB Counter WK, HB Dive Alert Smoke, HB Duo, HB Sweep, Mtn Counter Str, Mtn HB Plunge, Mtn Spot Heavy, PA Sprint HB Flat, PA X Burst Heavy, Power O, Stretch Alert Lookie, Verticals, Y Stick Heavy
5. **Singleback Bunch X Nasty** - Playbook editor: family 'Singleback' -> set 'Bunch X Nasty' -> source playbook 'Vikings' -> keep ALL 21 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Vikings', confirm set 'Bunch X Nasty' lists 21 plays
   - plays: Cheat HB Slash, Cheat HB Zone WK, Counter Y, Crack Toss, HB Dive, HB Slam, Inside Zone Split, PA Boot, PA Boot Cheat Slide Levels, PA Cheat Deep Corner, PA Cheat Wheel Post, PA Counter, PA Cross, PA Deep Cross Go, PA Post Curl Shot, PA Post Dig, PA WR Cheat Screen, PA X Out Cross, Spacing, Verticals, Wide Zone WK
6. **Gun Deuce Close** - Playbook editor: family 'Shotgun' -> set 'Deuce Close' -> source playbook 'Spread Classic' -> keep ALL 18 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Spread Classic', confirm set 'Deuce Close' lists 18 plays
   - plays: Bench Swap, Both Stick, Counter Y, Inside Zone Split, Mesh Spot, Outside Zone, PA Deep Cross, PA Deep Curls, PA Deep Out, PA Over Whip Out, PA Slide, PA Y Leak, Post Wheel, RPO Alert Omaha, RPO Peek Stick, RPO Read Flat Wheel, RPO Ready Jab Flat, RPO Split Peek Stick
7. **Singleback Bunch TE** - Playbook editor: family 'Singleback' -> set 'Bunch TE' -> source playbook 'Commanders' -> keep ALL 18 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Commanders', confirm set 'Bunch TE' lists 18 plays
   - plays: Cross Drag, HB Pump Draw, Inside Zone, Mtn DBL Swing, Mtn FK Swing, Mtn HB Screen, Mtn HB Zone WK, Mtn PA Boot Over, Mtn PA Switch Replace, PA Draw Shot, PA Jet Sweep, Power Alert Bubble, Spacing, Stick, Switch Snag, Verticals, WR Power Fake Jet, Zone Fake Jet
8. **Gun Y Trips Wk** - Playbook editor: family 'Shotgun' -> set 'Y Trips Wk' -> source playbook 'Bengals' -> keep ALL 24 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Bengals', confirm set 'Y Trips Wk' lists 24 plays
   - plays: Clearout, Deep In, Four Verticals, GoalLine Fade, HB Mid Draw, HB Mid Zone, HB Slip Screen, Inside Zone, Mtn PA Smash Fork, Mtn RPO Base Alert Smash, Mtn Smash HB Angle, PA Read, Quick Fork Zig, Quick Short Post, Quick Unders, RPO Base Alert Lookie, RZ Mesh Crossers, Shockh Option, Slants, Slot 2 Buc, Slot Cross, Stick N Nod, Under, Y Corner
9. **Gun Trio HB Wk** - Playbook editor: family 'Shotgun' -> set 'Trio HB Wk' -> source playbook 'Benkert's Dimes' -> keep ALL 21 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Benkert's Dimes', confirm set 'Trio HB Wk' lists 21 plays
   - plays: Dagger Sit, Flood Pylon Shot, HB Slip Screen, Inside Zone, Jet Chip Wasp, Mtn Spot Smash, Mtn Stick Nod Swirl, Mtn Trail HB Sneak, PA Cross Sit, PA Y Cross, QB Blast, Reload Inside Zone, Reload PA Flood, Reload RPO Alert Buck, RPO Alert Bubble, Sprint Rub Flat, Sprint Smash, Sprint Spot, Sticks, Verticals, Wide Zone
10. **Gun Trio Y-Flex** - Playbook editor: family 'Shotgun' -> set 'Trio Y-Flex' -> source playbook 'Browns' -> keep ALL 21 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Browns', confirm set 'Trio Y-Flex' lists 21 plays
   - plays: Curl Flat, Dagger, Flood Y-Post, HB Quick Base, HB Slip Screen, Inside Zone, Mesh Dagger, Mesh Whips, Mills, Mtn Corner Slot Choice, Mtn Mills, Mtn Out H-Angle, Mtn Out N Up, Mtn Shock Slot Dig, Mtn Stick, PA Crossers, PA Dagger, RPO Alert Bubble, Shockh Option, Slot Out, Verticals
11. **Gun Box Offset Wk** - Playbook editor: family 'Shotgun' -> set 'Box Offset Wk' -> source playbook 'Raiders' -> keep ALL 15 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Raiders', confirm set 'Box Offset Wk' lists 15 plays
   - plays: Curl Flat, HB Angle, HB Slip Screen, Inside Zone, Middle Hi Lo, Mtn Curls HB Angle, Mtn Verts HB Angle, Mtn Y Cross, Outside Zone, PA Verticals, Read Option, Slot Trail, Verticals, Wide Trap, Y Cross Post
12. **Gun Y Off Trips Wk** - Playbook editor: family 'Shotgun' -> set 'Y Off Trips Wk' -> source playbook 'Browns' -> keep ALL 33 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Browns', confirm set 'Y Off Trips Wk' lists 33 plays
   - plays: 22 Z In, Bounce Inside Zone, Bounce PA Post Curls, Bounce Slot Cross, Dig Cross, Fade Stops, Four Verticals, HB Flip Counter, HB Slip Screen, Inside Zone, Inside Zone Split, Mtn Curls, Mtn Fade Dagger, Mtn HB Swing, Mtn PA WR Zone Flood, Mtn Slot Whip In, Mtn WR Counter Rt, Mtn WR Zone, Mtn X Slant Spot, Mtn Z Spot, PA Flip Counter, PA Slide, Quick Fork Zig, Quick Short Post, Quick Unders, RPO Alert Flip Counter, Shallow Sail, Shockh Option, Slants, Slot 2 Buc, Sluggo Verts, Stick, Strong Flood
13. **I Form Close** - Playbook editor: family 'I Form' -> set 'Close' -> source playbook 'Lions' -> keep ALL 12 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Lions', confirm set 'Close' lists 12 plays
   - plays: HB Lead Dive, HB Stretch, HB Zone Toss, Inside Zone, Mesh, PA Boot Flow, PA Deep Cross Go, PA Deep Read, PA HB Wheel, PA Spider Mesh, Stutter Gos, Y-Post
14. **Gun Y Off Trio Close** - Playbook editor: family 'Shotgun' -> set 'Y Off Trio Close' -> source playbook '49ers' -> keep ALL 24 plays listed.
   - check: open the Playbook Editor, choose source playbook '49ers', confirm set 'Y Off Trio Close' lists 24 plays
   - plays: 0 1 Trap, Boomerang Mesh Z Rail, Boomerang PA X Over, Boomerang RPO Alert Swing, Bounce Dash Crosser, Bounce Inside Zone, Bounce PA Deep Out, Counter Y, Dash Flood, Empty Mtn Flood H-Trail, Empty Mtn Mesh Flat, Empty Mtn Salem, Flood, HB Pump Draw, Mtn Inside Zone, Mtn Mesh Rail, Mtn PA Cross Switch, Outside Zone, PA Power Shallow Cross, PA Toss Lead Swing Crosser, Pin Pull Toss, Power O, RPO Zone Peek, Smash Return
15. **Gun Tight Doubles** - Playbook editor: family 'Shotgun' -> set 'Tight Doubles' -> source playbook 'Buccaneers' -> keep ALL 15 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Buccaneers', confirm set 'Tight Doubles' lists 15 plays
   - plays: Cross Wheels, Curl, DBL Stick, Flood, HB Draw, HB Quick Base, HB Slip Screen, Inside Zone, L Spot, Mesh Spot, Mtn Bunch Option, Mtn Cross HB Sneak, Mtn FK Cross, PA Deep Curls, PA WR Cross

### Unresolved installation issues

- 15 of 15 sets have never been verified in the editor: open each before relying on this book
- editor limits (15 formations / 56 sets / 500 plays) are third-party reported, not verified in-game
- personnel for offense sets is inferred from names; defender/route assignments are UNKNOWN for most plays

## DEFENSE - 13 formations, 297 plays

- proposed fingerprint: `DEF-INST-CA849BF417`
- confirmed fingerprint: **NOT CONFIRMED - only the owner can confirm after installing and checking in Madden**
- editor limits: formations 13/15, plays 297/500, sets 13/56, families 5/15 (third_party_reported_not_verified_in_game)
- unmet situational requirements: none
- selection basis: cold-start prior (uncalibrated): formation ranking is driven by situational coverage and catalog breadth, not learned strength
- previous book DEF-F1CF-1862: objective -0.97528 vs proposed -0.91528; formations kept from it: ['4-3 Over', 'Dime 3-2 Odd', 'Nickel Over']

| # | formation (set) | source playbook | plays | editor status | why chosen |
|---|---|---|---|---|---|
| 1 | 4-3 Over | 49ers | 18 | PREVIOUSLY_INSTALLED_FROM_THIS_SOURCE | adds +10.0297 portfolio value when selected (coverage +9.9117, requirement fill +0.1180); strongest in two-minute-ish deep and 1st&10 open; play mix: man 5, zone 4, match 3, pressure 2, deep shell 7 |
| 2 | Nickel Wide | Bears | 24 | UNVERIFIED_IN_EDITOR | adds +0.0574 portfolio value when selected (coverage +0.0494, requirement fill +0.0080); strongest in two-minute-ish deep and 1st&10 open; play mix: man 3, zone 7, match 7, pressure 4 |
| 3 | Nickel Over | 4-3 | 42 | UNVERIFIED_IN_EDITOR | adds +0.0186 portfolio value when selected (coverage +0.0186, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 7, zone 9, match 8, pressure 11, deep shell 15 |
| 4 | Nickel 2-4 | Broncos | 42 | UNVERIFIED_IN_EDITOR | adds +0.0100 portfolio value when selected (coverage +0.0100, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 4, zone 9, match 8, pressure 5 |
| 5 | Quarter 1-3-7 | 3-4 | 15 | UNVERIFIED_IN_EDITOR | adds +0.0060 portfolio value when selected (coverage +0.0060, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 2, zone 4, match 1 |
| 6 | Dime 3-2 Odd | 46 | 27 | UNVERIFIED_IN_EDITOR | adds +0.0060 portfolio value when selected (coverage +0.0060, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 5, zone 6, match 5, pressure 6, deep shell 10 |
| 7 | 4-2-5 Over G | 4-3 | 24 | UNVERIFIED_IN_EDITOR | adds +0.0040 portfolio value when selected (coverage +0.0040, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 2, zone 4, match 7, pressure 10 |
| 8 | Dime Normal | 4-3 | 18 | UNVERIFIED_IN_EDITOR | adds +0.0040 portfolio value when selected (coverage +0.0040, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 3, zone 5, match 3, pressure 5 |
| 9 | Dime 2-3 Odd Jack | 3-4 | 18 | UNVERIFIED_IN_EDITOR | adds +0.0020 portfolio value when selected (coverage +0.0020, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 3, zone 4, match 2, pressure 5 |
| 10 | 4-3 Even 6-1 | 49ers | 6 | UNVERIFIED_IN_EDITOR | adds +0.0020 portfolio value when selected (coverage +0.0020, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 2, zone 1, match 1, pressure 3 |
| 11 | 4-2-5 Over Wide | 49ers | 18 | UNVERIFIED_IN_EDITOR | adds +0.0020 portfolio value when selected (coverage +0.0020, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 3, zone 4, match 2, pressure 3 |
| 12 | 4-3 Under | Bears | 18 | UNVERIFIED_IN_EDITOR | adds +0.0020 portfolio value when selected (coverage +0.0020, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 3, zone 4, match 2, pressure 1 |
| 13 | Nickel 3-3 Odd | Seahawks | 27 | UNVERIFIED_IN_EDITOR | adds +0.0020 portfolio value when selected (coverage +0.0020, requirement fill +0.0000); strongest in two-minute-ish deep and 1st&10 open; play mix: man 2, zone 6, match 6, pressure 2 |

### Step by step

1. **4-3 Over** - Playbook editor: family '4-3' -> set 'Over' -> source playbook '49ers' -> keep ALL 18 plays listed.
   - check: open the Playbook Editor, choose source playbook '49ers', confirm set 'Over' lists 18 plays
   - plays: 1 Hole Press, 1 Robber Press, Cover 1 Contain, Cover 1 Hole, Cover 2 Hard Flat, Cover 2 Man, Cover 3 Cloud Str, Cover 3 Sky, Cover 4 Press, Cover 4 Quarters, Cover 6, DBL Safety Blitz, Free Fire, LB Sting 1, OLB Fire Man, Sam Edge 3, Tampa 2, Thunder Smoke
2. **Nickel Wide** - Playbook editor: family 'Nickel' -> set 'Wide' -> source playbook 'Bears' -> keep ALL 24 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Bears', confirm set 'Wide' lists 24 plays
   - plays: CB Zone Blitz, Cov 2 Invert Hard Flat, Cov 3 Buzz Match WK, Cover 1 Hole, Cover 1 Robber Press, Cover 2 Invert, Cover 2 Man, Cover 3 Cloud, Cover 3 Match, Cover 3 Sky Press, Cover 3 Zone, Cover 4 Palms, Cover 4 Quarters, Cover 4 Zone, Cover 6, Cover 6 Zone, Cover 9 Show 2, DBL Safety Blitz, I LB Blitz, Nickel 2 Trap, Nickel Blitz 3, Over Storm Brave, Silver Shoot Pinch, Tampa 2 Contain
3. **Nickel Over** - Playbook editor: family 'Nickel' -> set 'Over' -> source playbook '4-3' -> keep ALL 42 plays listed.
   - check: open the Playbook Editor, choose source playbook '4-3', confirm set 'Over' lists 42 plays
   - plays: 1 Double Slot, 1 Double WR1, 1 Double WR2, 1 Robber Press, Bracket Switch Willie, Cov 3 Buzz Match, Cover 1 Contain, Cover 1 Hole, Cover 2 Invert, Cover 2 Man, Cover 2 Roll, Cover 3 Buzz Match, Cover 3 Cloud, Cover 3 Match, Cover 3 Sky, Cover 4 Drop Field, Cover 4 Palms, Cover 4 Quarters, Cover 6, Cover 6 Willie, Cover 9 Show 2, DBL Safety Blitz, Double Bracket, Double Bracket Switch, Double TE, Edge Blitz 0, Edge Blitz 3, Field Sim 3, FS Will Blitz 0, LB Blitz 1, Nickel 2 Trap, Nickel Blitz 3, Nickel Sim 2, Over Storm Brave, Overload 3 Seam, Overload Blitz, Roll 23 Cloud, Roll Cover 2 Invert, Roll Cover 3 Cloud, Show SS Sim 2, Sim Pressure 3, Tampa 2
4. **Nickel 2-4** - Playbook editor: family 'Nickel' -> set '2-4' -> source playbook 'Broncos' -> keep ALL 42 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Broncos', confirm set '2-4' lists 42 plays
   - plays: 1 Double TE, 1 Double WR1, 2 Invert Lurk, Bracket Switch Willie, Buck Slant 3, Cover 1 Contain, Cover 1 Hole, Cover 1 Robber, Cover 2 Invert, Cover 2 Man, Cover 2 Match, Cover 3 Buzz Match, Cover 3 Cloud, Cover 3 Match, Cover 3 Sammie, Cover 3 Sky, Cover 4 Drop Field, Cover 4 Palms, Cover 4 Quarters, Cover 6, Cover 6 Willie, Cover 9 Show 2, Cross Sim 2, DBL Safety Blitz, Double Bracket, Double Bracket Switch, Edge Blitz 0, Edge Blitz 3, Field Sim 3, FS Will Blitz 0, LB Blitz 1, Nickel 2 Trap, Nickel Sim 2, Over Storm Brave, Overload 3 Seam, Roll 23 Cloud, Roll Cover 2 Invert, Roll Cover 3 Cloud, Show SS Sim 2, Silver Shoot Pinch, Sim Pressure 3, Tampa 2
5. **Quarter 1-3-7** - Playbook editor: family 'Quarter' -> set '1-3-7' -> source playbook '3-4' -> keep ALL 15 plays listed.
   - check: open the Playbook Editor, choose source playbook '3-4', confirm set '1-3-7' lists 15 plays
   - plays: 3 DBL Mable Cloud, 3 Double Buzz, 3 Double Mable, Cover 1 Thief, Cover 2 Man Spy, Cover 2 Spy, Cover 3 Mable, Cover 3 Spy, Cover 4 Drop, DB Strike 1, Fz 2 Deep, Fz 3 Sky, Man Up 3 Deep, Overload 3 Seam, Under Smoke
6. **Dime 3-2 Odd** - Playbook editor: family 'Dime' -> set '3-2 Odd' -> source playbook '46' -> keep ALL 27 plays listed.
   - check: open the Playbook Editor, choose source playbook '46', confirm set '3-2 Odd' lists 27 plays
   - plays: 1 Double WR1, 1 Double WR2, Cover 1 Robber Press, Cover 2 Man, Cover 3 Buzz Match WK, Cover 3 Cloud, Cover 3 Hard Flat, Cover 3 Lock, Cover 4 Drop, Cover 4 Palms, Cover 4 Quarters, Cover 6, Cover 6 Willie, DB Fire 2, DBL Safety Go, Double Bracket, Edge Blitz 1, Edge Blitz 3, FS Zone Blitz, Hot Blitz 3, Okie Eagle 3, Okie Roll 2, Overload 3 Seam, SS Zone Blitz, Tampa 2, Tampa 2 Drop, Zero Blitz
7. **4-2-5 Over G** - Playbook editor: family '4-2-5' -> set 'Over G' -> source playbook '4-3' -> keep ALL 24 plays listed.
   - check: open the Playbook Editor, choose source playbook '4-3', confirm set 'Over G' lists 24 plays
   - plays: Casino Blitz, Corner Blitz 3, Cover 1 Hole, Cover 2, Cover 2 Contain, Cover 2 Man, Cover 3, Cover 3 Buzz Match WK, Cover 3 Match, Cover 3 Sky, Cover 4 Palms, Cover 4 Quarters, Cover 6, Cover 6 Invert, Cover 6 Trap, LB Blitz, Mike SS 3, Pinch Blitz, SS Blitz 2, SS Blitz 3, SS LB Blitz, Ws Blitz 1, Ws Blitz 2, Ws Blitz 3
8. **Dime Normal** - Playbook editor: family 'Dime' -> set 'Normal' -> source playbook '4-3' -> keep ALL 18 plays listed.
   - check: open the Playbook Editor, choose source playbook '4-3', confirm set 'Normal' lists 18 plays
   - plays: 3 Double Buzz, Cover 1 Robber Press, Cover 1 Spy, Cover 2 Hard Flat, Cover 2 Man, Cover 3 Buzz Press, Cover 3 Buzz Seam, Cover 3 Cloud, Cover 4 Palms, Cover 4 Quarters, Cover 6, DB Blitz 0, DBL Safety Blitz, Dime Blitz 3, LB Blitz 1, Overload 3 Show 2, SS Blitz 3, Tampa 2
9. **Dime 2-3 Odd Jack** - Playbook editor: family 'Dime' -> set '2-3 Odd Jack' -> source playbook '3-4' -> keep ALL 18 plays listed.
   - check: open the Playbook Editor, choose source playbook '3-4', confirm set '2-3 Odd Jack' lists 18 plays
   - plays: Cover 1 Contain, Cover 1 Robber Press, Cover 2 Hard Flat, Cover 2 Man, Cover 2 Match, Cover 3 Lock, Cover 4 Drop, Cover 6 Press, DB Fire 2, DBL Safety Go, Edge Blitz 1, Edge Blitz 3, FS Zone Blitz, Overload 3 Seam Press, SS Zone Blitz, Str Eagle Slant 3, Tampa 2 Spy, Zero Blitz
10. **4-3 Even 6-1** - Playbook editor: family '4-3' -> set 'Even 6-1' -> source playbook '49ers' -> keep ALL 6 plays listed.
   - check: open the Playbook Editor, choose source playbook '49ers', confirm set 'Even 6-1' lists 6 plays
   - plays: Cover 1 Hole, Cover 1 MLB Blitz, Cover 3 Buzz, Cover 4 Quarters, Sam Blitz 3, Sam Will Blitz
11. **4-2-5 Over Wide** - Playbook editor: family '4-2-5' -> set 'Over Wide' -> source playbook '49ers' -> keep ALL 18 plays listed.
   - check: open the Playbook Editor, choose source playbook '49ers', confirm set 'Over Wide' lists 18 plays
   - plays: Cover 1 Hole, Cover 1 Robber, Cover 2 Man, Cover 3 Cloud Show 2, Cover 3 Sim Pressure, Cover 3 Sky, Cover 4 Quarters, Cover 6, Cover Sting, Double CB Blitz, Fire Man, Free Fire 3, Invert 2 Sim, Mike Sam Cross 3, Sam Blitz 3, Tampa 2, Thunder Smoke, Will Blitz 3
12. **4-3 Under** - Playbook editor: family '4-3' -> set 'Under' -> source playbook 'Bears' -> keep ALL 18 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Bears', confirm set 'Under' lists 18 plays
   - plays: 1 Contain Press, 1 Hole Press, 1 Robber Press, 2 Invert Hard Flat, Cover 1 Hole, Cover 2 Invert, Cover 3 Hard Flat, Cover 3 Match, Cover 3 Skywk, Cover 4 Quarters, DBL Safety Blitz, Free Fire O, FS Fire 1, FS Fire 3, Hammer 0 Blast, Sam Crash 1, Sam Crash 3, Will Go Fire 3
13. **Nickel 3-3 Odd** - Playbook editor: family 'Nickel' -> set '3-3 Odd' -> source playbook 'Seahawks' -> keep ALL 27 plays listed.
   - check: open the Playbook Editor, choose source playbook 'Seahawks', confirm set '3-3 Odd' lists 27 plays
   - plays: 1 Double WR1, 1 Double WR2, 3 Buzz Mable, Cover 1 Robber Press, Cover 2 Man, Cover 3, Cover 3 Cloud, Cover 3 Match, Cover 3 Sky, Cover 4 Drop, Cover 4 Palms, Cover 4 Quarters, Cover 6, Cover 6 Willie, Double Bracket, Hot Blitz 3, Hot Blitz Bail, LB Cross 3 Show 2, LB Cross O, Nickel 2 Trap, Overload 3 Press, Pinch 0, Sam Mike 1, Sam Mike 3 Press, Tampa 2, Tampa 2 Drop, Tampa Sim Pressure

### Unresolved installation issues

- 12 of 13 sets have never been verified in the editor: open each before relying on this book
- editor limits (15 formations / 56 sets / 500 plays) are third-party reported, not verified in-game
- personnel for offense sets is inferred from names; defender/route assignments are UNKNOWN for most plays

## After installing

1. Verify every set and play count in Madden's editor; fix anything missing and re-run `game-prep` if a set is unavailable.
2. `confirm-install --proposal proposal_offense.json --attested-by YOU --confirmed-books books.json` (and again for `proposal_defense.json`).
3. Run `preflight`, then `game`.
