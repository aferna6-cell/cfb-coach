# Elite Madden 27 VOD pilot — source and honest readiness

## First real source (not yet imported)
- **Official Madden Championship Series 27 / Pro League: Game Time**, broadcast September 15, 2026.
- EA SPORTS MADDEN NFL official YouTube video: https://www.youtube.com/watch?v=iNiG0_ib9Kc
- EA event guide and match schedule: https://www.ea.com/games/madden-nfl/news/mcs-pro-league-game-time
- Full broadcasts include multiple experts and matchups, notably Henry vs JonBeast, Drini vs Wes, and Drini vs Astro. **Select one complete match and one offensive player per initial annotation group**, not a highlight montage.
- This is **Madden Ultimate Team, human vs human**. Do not automatically transfer player attributes, settings, or execution-dependent tactics to Lions Franchise. Preserve title update/patch when verified.

Finding an official public stream confirms that it can be watched. It **does not** authorize automated downloading, redistribution, or model training. Use a locally supplied file only when its owner or license permits the intended analysis. Never use a downloader that bypasses a service's restrictions. If authorized processing is unavailable, stop at source discovery; do not claim real expert VOD training.

## Local first pilot

1. Obtain an authorized local recording of an entire match. If your authorized source is a full event, separate matches locally; keep the source link and permission note.
2. Review ~20-30 offensive snaps of a specified elite player first. This proves the pipeline, not generalization.
3. Record match identity, competitor ID, opponent, patch, MUT mode, down/distance and **only visible** formation, concept, adjustment and post-snap result.
4. Keep coverage hypotheses apart from observed pre-snap safety structure. Never backfill pre-snap inputs from post-snap film.
5. Review failures and incompletions as carefully as conversions. Keep inaccessible plays and unknown adjustments as null.
6. Only after reviewed examples cover multiple independent experts/matches, evaluate disjoint expert-and-match holdouts. Keep the live coach in shadow mode until a real baseline and reliable personal outcome comparison exist.

### Commands (Ubuntu)

Use an existing authorized file (replace placeholders; no clip has been obtained in this repository):

\`\`\`bash
cd ~/cfb-coach
git fetch origin
git switch cursor/sprint15a-elite-vod-validation-5f33
git pull --ff-only
sudo apt-get install -y ffmpeg

FILM="$HOME/.cfb-coach/film/expert"
LEARN="$HOME/.cfb-coach/learning"

python3 -m cfb_coach ml expert-film-import /absolute/path/to/authorized-match.mp4 \
  --expert-id EXPLICIT_PLAYER_ID \
  --match-id mcs27-gametime-match-1 \
  --store "$FILM" --permission-status user_authorized_local \
  --competitive-mode ultimate_team --opponent-type human \
  --game-version madden27

python3 -m cfb_coach ml film-review \
  --game-id mcs27-gametime-match-1 --store "$FILM" --serve

# After reviewing and correcting labels. The expert annotation workflow also
# supports expert-annotate --labels when fields not covered by film-review need review.
python3 -m cfb_coach ml expert-export-evidence \
  --store "$FILM" --learning-store "$LEARN" \
  --match-id mcs27-gametime-match-1

python3 -m cfb_coach ml prepare-expert-dataset --learning-store "$LEARN"
python3 -m cfb_coach ml train-expert-policy --learning-store "$LEARN"
python3 -m cfb_coach ml expert-learning --summary \
  --learning-store "$LEARN" --store "$FILM"
python3 -m cfb_coach ml learning-eval --learning-store "$LEARN"
\`\`\`

**Do not use \`learning-eval --promote\` on early VODs or synthetic fixtures.** This validation branch intentionally blocks promotion until held-out real footage, a measured coordinator baseline, and a valid personal outcome evaluation exist.

## Verified constraints of this branch

- This code does not automatically recognize football plays in real recordings. Snap review, clip identity, visible offensive concept and outcome labels require a person.
- The test fixtures under \`tests/fixtures/expert_film\` are generated annotations, not pro-game proof.
- The historical personal game DB \`~/.cfb-coach/madden27.db\` exists on the user's Ubuntu system, not in the coding environment.
- Evaluation fits must not overwrite the saved full-data policy or personalization artifacts.
- An expert VOD is a **demonstrated decision**, not a proof that unchosen alternatives fail or that a professional's timing is transferable to another player.
