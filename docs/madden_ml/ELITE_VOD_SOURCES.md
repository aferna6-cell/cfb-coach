# Elite Madden 27 VOD source research

Research date: 2026-10-10  
Machine-readable pack: `research/elite_vods/candidate_manifest.json`  
Permission draft (not sent): `research/elite_vods/PERMISSION_REQUEST_DRAFT.md`

## Verdict

At least **seven** high-value full competitive recordings were identified from official EA channels, including the requested **Sept 15 Game Time** and **Oct 6 Breakouts** broadcasts.

**Authorized local file: not available.** Public viewing is open on YouTube/Twitch. Download and model-training rights are **not** granted by that access. No unauthorized downloader was used.

**Recommended first match:** [Noah vs Sebatron](https://www.youtube.com/watch?v=F8Hm324SMtg) (official single-match upload from MCS Pro League Game Time).

## Ranked source table

| Rank | Match / package | Date | Players | Mode | View links | Res (typical) | Est. usable OC snaps | View | Download | Train | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Noah vs Sebatron (single) | 2026-09-15 | Noah, Sebatron | MCS Pro League MUT | [YouTube](https://www.youtube.com/watch?v=F8Hm324SMtg) | ≤1080p | 18–36 | Yes | **No** | **Pending EA** | Best first annotation target; result 43–42 Sebatron |
| 2 | MCS Pro League: Game Time (main) | 2026-09-15 | Henry–JonBeast, Drini–Wes, Noah–Sebatron, Drini–Astro | MCS Pro League MUT | [YouTube](https://www.youtube.com/watch?v=iNiG0_ib9Kc) · [Twitch 4h55m](https://www.twitch.tv/videos/2875171708) | ≤1080p | 70–140 | Yes | **No** | **Pending EA** | User-supplied official starting URL |
| 3 | MCS Pro League: Breakouts (main) | 2026-10-06 | Pro League eight | MCS Pro League MUT | [YouTube](https://www.youtube.com/watch?v=B50_glA4it4) · [Twitch 4h52m](https://www.twitch.tv/videos/2893769780) | ≤1080p | 90–180 | Yes | **No** | **Pending EA** | Current-patch priority |
| 4 | Game Time Bonus Stream | 2026-09-15 | Chosen–Noah, Astro–Sebatron, JonBeast–Chosen | MCS Pro League MUT | [YouTube](https://www.youtube.com/watch?v=iB3O8H_p5MU) | ≤1080p | 50–100 | Yes | **No** | **Pending EA** | Completes Sept 15 day |
| 5 | Breakouts Bonus Stream | 2026-10-06 | Pro League secondary matches | MCS Pro League MUT | [YouTube](https://www.youtube.com/watch?v=Y9DZ6fhD5_0) | ≤1080p | 40–90 | Yes | **No** | **Pending EA** | Completes Oct 6 day |
| 6 | Chosen vs Millz OQ Final | 2026-10-06 | Chosen, Millz | MCS Open Qualifier | [YouTube](https://www.youtube.com/watch?v=xGH8SOUrTT0) | ≤1080p | 18–40 | Yes | **No** | **Pending EA** | Chosen target player |
| 7 | Henry vs JonBeast (early official) | 2026-08-08 | Henry, JonBeast | H2H gameplay | [YouTube](https://www.youtube.com/watch?v=aoIYILGdvOE) | ≤1080p | 18–36 | Yes | **No** | **Pending EA** | Pre–Game Time sample |
| 8 | Henry_773 MCS ladder/elim POV | 2026-09-19–27 | Henry (+opponents) | Qualifier / practice | [Twitch channel](https://www.twitch.tv/henry_773) | 720p–1080p | 40–200 | Yes* | **No** | **Pending EA + creator** | Better menu visibility; dual rights |

\*If the VOD has not expired.

Deprioritized: TD-only highlights ([GbwHuUPUNGc](https://www.youtube.com/watch?v=GbwHuUPUNGc)), shorts.

## Exact Game Time match card (official EA schedule)

Source: https://www.ea.com/games/madden-nfl/madden-nfl-27/news/mcs-pro-league-game-time  
Results: Robinhood News Game Time write-up (2026-09-16)

| Time (ET) | Main | Bonus |
| --- | --- | --- |
| 6:00 | Henry 33–29 JonBeast | Noah 47–35 Chosen |
| 7:00 | Wesley 35–30 Drini | Sebatron 34–28 Astro |
| 8:00 | Sebatron 43–42 Noah | Chosen 50–37 JonBeast |
| 9:00 | Astro 31–27 Drini | — |

## Rights and acquisition status

| Permission | Status | Evidence |
| --- | --- | --- |
| View on official channels | Allowed for public viewers | EA hosts on YouTube `@EAMaddenNFL` and Twitch `EAMaddenNFL` |
| Download / local retain | **Not authorized** here | No EA download endpoint found; platform ToS disallow unauthorized downloaders |
| Local analysis / ML training | **Unclear → pending written EA permission** | MCS 27 Official Rules reserve commercial/media rights for EA; EA publishes a Content Usage Permission Request Form to `PermissionRequests@ea.com` |
| Authorized organizer copy | **Not found** | No press-kit full-match package or research mirror located |

**Blocker (clear):** acquisition stopped at rights clearance. Continuing alternatives were catalogued (bonus streams, OQ final, player POV) without bypassing DRM or using unauthorized downloaders.

## Annotation practicality

Broadcast VODs are excellent for **competitive strategy** and **outcomes**, but play-selection menus are often not fully framed. Expect human review. Player POV (Henry Twitch) may improve dial visibility after dual permission.

Do not claim raw video is automatically understood.

## Pipeline handoff

After EA (and any creator) grants rights and you place an authorized local file:

```bash
python3 -m cfb_coach ml expert-film-import /path/to/authorized.mp4 \
  --expert-id sebatron --match-id mcs27-gametimen-noah-vs-sebatron \
  --permission-status user_authorized_local \
  --competitive-mode ultimate_team --opponent-type human \
  --game-version madden27

# Register remote candidates now (no download):
python3 -m cfb_coach ml expert-vod-catalog \
  --manifest research/elite_vods/candidate_manifest.json
```

Until then, keep `permission_status` as `unknown` and leave live expert-signal influence in shadow.
