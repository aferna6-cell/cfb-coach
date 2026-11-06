# Madden 27 scheme semantics (research only)

## Current achievement

Source: exact 93 offense / 123 defense playbook snapshot, user-confirmed as built. Each installed play is indexed with **name-level concept/scheme intelligence**. The defensive matcher does **not** conflate Palms with Quarters, match with spot-drop, Cover 1 Hole with Robber, or Cover 6 with Cover 9. It preserves name-only confidence and unknown 11-player assignments. It recognizes EA's 13 documented match checks as conditional behaviors, not automatically installed adjustments.

Generate a standalone database with:

```bash
python scripts/madden_ml/build_installed_semantics.py --output /tmp/madden27-installed-semantics.sqlite
```

Creates `.review.csv` and `.report.json` as well. It refuses to replace an existing file or write into `~/.cfb-coach`. No live caller changes.

## Critical semantic distinctions

- `cover_4_quarters`: pattern match, 4 deep as the initial structure. Safeties can relate to #2 vertical; release changes alter responsibilities.
- `cover_4_palms`: two-read pattern match, notably #2 quick out/flat can cause the corner to trigger and safety to cover #1's vertical. This is **not** Quarters with a different name.
- `cover_4_drop`: spatial four-deep zone; unlike match it does not mechanically convert because a receiver stemmed vertical.
- `cover_3_match` vs `cover_3_spot`: both nominal three-deep but only the former uses release-dependent match assignments.
- `cover_1_hole`: man + single deep + middle hook/hole defender, often a linebacker.
- `cover_1_robber`: man + single deep + mobile low-middle robber, commonly safety rotation; shell may disguise this.
- `cover_1_contain`: man + single deep + QB spy and edge containment principles.
- `cover_2_man`: two-deep *man underneath*, unlike Cover 2 zone.
- `cover_2_invert`: inverted deep/underneath roles post snap, unlike normal Cover 2 or Cover 2 Man.
- `tampa_2`: two-deep with middle runner relating to the seam.
- `cover_6`: quarter-quarter-half split field, normally quarters toward strength; **direction depends on actual check, not established from name**.
- `cover_9`: reversed split field relative to Cover 6, not the same call.
- Willie, roll, press, weak, show-2 are recorded as variants, with unsupported special responsibilities remaining **unknown**.

**Known limitation:** A particular Madden play may behave differently after title updates, selected match checks, zone drops or personnel assignments. `name_inferred_not_player_verified` cannot be used as eleven assignments. Unusual blitz labels and ambiguous Cover 4 Press / Cover 3 Lock retain `unknown` scheme instead of invented details.

## Public source research and reuse policy

General coverage mechanics verified from:

- [EA Madden NFL 27 Gameplay Deep Dive](https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay) — official 2026 match checks, zone-it conversion, coverage hubs and custom adjustments.
- [MUT.GG Ask Huddle #42: Cover 1 in Madden](https://www.mut.gg/news/ask-huddle-42-cover-1-explained-in-madden/) — core man, Cover 1 Hole versus Robber versus Contain, January 2026.
- Independent original explanatory taxonomy (football principles, **not** scraped third-party diagram assets).

**Do not claim this is 216 full play diagrams.** A web survey found Huddle.gg, Madden Prodigy, 818 Madden, Madden School and Madden Tools list lots of play names/diagrams. However, Madden Tools disallows systematic ML extraction; 818 Madden explicitly forbids scraping/copying its library to another product. Public listings verify membership, not authorized complete-player training data. Huddle.gg names and playbook metadata are already captured in this repo, but owning a name list does not authorize extracting all protected diagrams. This commit doesn't download images or bypass access controls. For full 2,376 player duties, use user-owned Xbox play art/film or a source that expressly permits ML reuse.

## Train and evaluate responsibly

The structured **scheme** knowledge can be used as a tagged football-prior feature in future offline tests if source-use permissions allow. It cannot be represented as observed Madden player execution or as learned success data. The existing separate `installed_assignment_truth.py` player-duty ledger still requires verifiable player-level evidence. No new verified player duties are claimed. No live changes or model promotions.