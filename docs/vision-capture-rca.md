# Live screen capture — root-cause analysis (v1.9.7)

**Question:** why did `watch --window "XBOX"` never work reliably on Aidan's laptop
(Win11, native Python 3.14, Xbox Remote Play / Xbox PC app, stream on the left and
PowerShell on the right), and what should we build instead?

**Short answer**

| Symptom | Cause | Whose fault | Status |
|---|---|---|---|
| dxcam: window matched, then endless `waiting for frames` | dxcam rejects any region outside the primary monitor (`Invalid Region`). A **snapped** window's `GetWindowRect` includes the invisible 7px border, so its rect always leaks past the screen edge. The exception was **swallowed** by `ThreadedCapture`. | **Ours** | **Fixed** |
| mss got frames but the crop / window was wrong | Window rects were read **before** the process became DPI-aware (logical px on a scaled laptop vs physical px captured). The matcher could also pick **cloaked** windows (suspended UWP Xbox app, Game Bar), and it used the full window rect, title bar included. | **Ours** | **Fixed** |
| Stream shows "Click or tap here to continue playing" / freezes when PowerShell is focused | The Remote Play client pauses when it loses focus. No capture API can un-pause another app. | **Remote Play** | Can't be fixed from our side. Design around it (below). |
| Our own overlay made it worse | `watch` auto-ran `webbrowser.open(overlay.html)`. That steals focus, and with Edge as the default browser it opened a **tab in the same Edge window** as `XBOX PC app \| … - Microsoft Edge`. That explains why `--list-windows` showed different titles "depending on focus/tab". | **Ours** | **Fixed** (not auto-opened during live capture) |
| Vision stuck at `unknown`, conf ≈ 0.17 | 0.17 is **exactly** `mean(0.15, 0.20, 0.15)`: the fixed fallback scores returned when **zero player blobs** were found, i.e. no grass was detected in the crop. The pipeline never saw a football field. | Wrong/paused/non-field frames reaching the CV | Explained. Diagnose with `--probe`. |
| Even on a perfect frame, shell was wrong | Image y (0 = top = far) was fed into classifiers written for field y (0 = near camera). Offense and defense were swapped, and the "deep safeties" were our QB/RB. | **Ours** | **Fixed** |
| Even with that fixed, shell/pressure is unreliable | Classical "not-grass blobs" CV breaks as soon as yard lines, numbers, logos or overlays are in frame. | Honest limit of classical CV | Won't fix with heuristics |

---

## 1. dxcam "no frames": our bug, not an Xbox limitation

Verified against the dxcam 0.3.0 source (`dxcam/dxcam.py::_validate_region`):

```python
if not (self.width >= right > left >= 0 and self.height >= bottom > top >= 0):
    raise ValueError("Invalid Region: Region should be in {w}x{h}")
```

Old code: `dxcam.create(region=GetWindowRect(hwnd))`.

- Win11 snap-left on a 1920×1080 screen: `GetWindowRect` ≈ `(-7, 0, 967, 1047)`. `left < 0`, so it raises.
- Snap-right: ≈ `(953, 0, 1927, 1047)`. `right > 1920`, so it raises.
- Maximized: `(-8, -8, 1928, 1048)`, which also raises.

So in the side-by-side layout, **every** dxcam window capture raised. `ThreadedCapture._loop`
did `except Exception: frame = None`, so the only visible output was
`waiting for frames… capture=threaded:dxcam`, forever. That matches the logs exactly.
The v1.9.3 dxcam→mss fallback hid the error rather than fixing it.

The Xbox app / UWP / "protected content" theory does not explain it. Desktop
Duplication (dxcam) and GDI BitBlt (mss) both honor display affinity the same way:
protected windows come back **black**, not missing. mss *did* return frames, so
the surface is very likely not capture-protected. `--probe` settles this on the
real machine (verdict `BLACK` vs `OK`).

A second issue: dxcam only switches the process to DPI-aware when the camera is
created, which is **after** our code had already read the window rect. On a
125%/150% laptop the rect was in logical px and the capture in physical px.

**Fixes (v1.9.7):**
- `vision/winenum.py` makes the process per-monitor DPI-aware before reading any coordinate.
- It uses the **client rect** (no title bar, no invisible border) via ctypes, so pywin32 is no longer needed.
- dxcam now creates a full-output camera and grabs a **clamped** region. A window that isn't on the primary monitor raises a clear error.
- `ThreadedCapture` records `last_error`, and the heartbeat prints it: `waiting for frames… · last error: ValueError: Invalid Region…`.

## 2. Wrong window, wrong crop (mss path)

- `IsWindowVisible()` is **True for DWM-cloaked windows**: suspended UWP apps like the Xbox app, Xbox Game Bar, and windows on other virtual desktops. Their titles still match "Xbox" and "Game Bar" (both were in our alias list). mss then copied whatever was on screen at that stale rect, often PowerShell. The finder now skips cloaked, minimized and terminal windows. `--list-windows` shows them tagged `[CLOAKED]` so you can see which is which.
- mss copies **screen pixels**, not the window. Anything on top (the terminal, a browser overlay, the "click to continue" dimmer) is what gets analyzed.
- With the stream in Edge, the client area includes the tab strip and address bar. Use Edge full-screen (F11) or an installed-app window so the client area is only the video. Otherwise set a relative crop in calibration.

## 3. The focus/pause problem: a hard Remote Play constraint

Remote Play pausing when the stream loses focus is the client's behavior (observed
on Aidan's machine). It is not something capture code can change.

- WGC and dxcam/mss all faithfully capture the **paused** frame. `--probe` reports it as `FROZEN`.
- WGC *does* fix occlusion: it captures the stream window even when a terminal sits on top of it. It does not fix focus.
- **Typing and Remote Play vision can't run at the same time.** Typing needs the terminal focused, and that pauses the stream. This is the main reason the typing-only `play` flow won.

## 4. Why vision said `unknown` @ 0.17

`to_defense_look()` confidence = mean of formation/shell/pressure confidences.
Their "not enough players" floors are 0.15 / 0.15 / 0.20, and **(0.15+0.15+0.20)/3 = 0.1667**.
So conf ≈ 0.17 on every frame means `extract_player_centroids` found ~0 blobs, which
means its green-grass mask found no grass. Reproduced with synthetic frames:

| Synthetic frame (1280×720) | blobs | pipeline look |
|---|---|---|
| black / protected | 0 | unknown, conf 0.17 |
| play-call menu (no grass) | 0 | unknown, conf 0.17 |
| washed-out grass (S < 40) | 0 | unknown, conf 0.17 |
| clean field, 22 players, 2 deep safeties | 22 | **before fix:** single_high (wrong) · **after y-flip:** two_high ✓ |
| same + yard lines | 14 | single_high (still wrong) |
| same + pause-dimmer overlay | 21 | confident "trips" from text blobs (garbage) |

Conclusions:
1. The live 0.17 means the frames were **not gameplay field frames**: wrong window/region, pause dimmer, menus, or black. That's a capture problem, not a classifier-threshold problem.
2. There was a real **pipeline bug**: a y-axis inversion. It's fixed (`pipeline.image_to_field`), and a regression test proves two-high now reads two-high on a clean frame.
3. **Classical CV is honestly weak.** Once yard lines, numbers or overlays are present, blob geometry misreads shells. Pre-snap shell/pressure needs a trained player detector (for example a small YOLO fine-tuned on a few hundred labeled CFB 27 frames) plus yard-line homography. That's a separate project, and no amount of threshold tuning replaces it.

## 5. Capture API comparison for this app class

| API | Captures | Occluded window OK? | DPI/region pitfalls | Verdict here |
|---|---|---|---|---|
| **dxcam** (DXGI Desktop Duplication) | Screen pixels of the primary output | No (captures what's on top) | Region must be inside the output; primary monitor only | Fast, but brittle for window capture. Fixed, but not preferred. |
| **mss** (GDI BitBlt of the desktop) | Screen pixels of any region | No | Needs DPI-aware coordinates | Works as a fixed-ROI fallback while the stream stays focused and uncovered |
| **WGC** (Windows Graphics Capture, `windows-capture`) | The target window's own surface, by HWND | **Yes** (not minimized) | None, it's per-window | **Best Remote Play option.** New `--capture wgc`. abi3 wheel, works on Python 3.14. |
| PrintWindow / window-DC BitBlt | Asks the app to paint itself | Sometimes | GPU/flip-model apps (browser video, UWP) often return black | Not worth it |
| **HDMI capture card** (UVC → OpenCV `--device`) | The console's actual HDMI signal | N/A (no window) | None | **Production path.** Removes Remote Play entirely. |
| OBS Virtual Camera | Whatever OBS composes | Via OBS | Adds OBS to the loop | Only useful as a bridge (e.g. card → OBS → virtual cam → `--device`); unnecessary if the card is UVC |

## 6. Recommended architecture: "Xbox focused, coach visible, terminal never focused"

**Production (recommended): capture card with HDMI passthrough.**

```
Xbox Series ──HDMI──▶ [capture card IN]──passthrough OUT──HDMI──▶ monitor (Aidan plays, ~0 added latency)
                            └──USB 3 (UVC)──▶ laptop: cfb-coach watch --device 0
```

- No Remote Play: no focus rule, no pause, no network compression, no window-finding.
- The laptop keyboard is free, so **typed `play` and vision run together**. Vision handles bookkeeping, and you type only when you want to.
- Buy any **UVC** (driverless) card **with HDMI loop-out/passthrough** that captures 1080p60. Cheap generic USB 3.0 loop-out dongles (~$30–60) or an Elgato HD60 X (~$150+) both fit. Check that passthrough supports your monitor's mode; some cheap loop-outs cap refresh rate or drop VRR.
- Xbox Series doesn't apply HDCP to games. If the feed is black, turn off *Allow HDCP* under Settings → General → TV & display options → Video fidelity & overscan. **Verify on the console:** this menu path is from docs, not tested here.
- `DeviceCapture` now opens with DirectShow at 1920×1080. Many UVC cards default to 640×480, which is too small for HUD digits.

**Interim (free): Remote Play + WGC, stream always focused.**

1. `pip install windows-capture`
2. Put the stream on the left and **click it once. Never click the terminal during live watch.**
3. `cfb-coach watch --window "XBOX" --capture wgc --tts`
4. Tips reach you without focusing anything:
   - The PowerShell window prints PLAY/ADJUST/HIKE. Console output is visible without focus.
   - `--tts` speaks them.
   - The HTML overlay is still written but **not** auto-opened. If you want it, open it on a **phone** or in a separate, non-Edge window, then click back on the stream.
5. You can't type corrections mid-drive (that pauses the stream). Accept that, or use the capture card.

**What vision should do, even with a card:** HUD read (down/distance/clock/score at fixed 1080p pixel positions), snap and play-end detection (motion), and play-call-screen detection. Those automate the *bookkeeping* part of typing. Keep the **defensive read (shell/pressure) typed/manual** until a trained detector exists.

## 7. Won't fix without a capture card (definitive)

- Remote Play pausing when unfocused.
- Using typed commands and Remote Play vision at the same time.
- Remote Play's compression and resolution, which hurt HUD OCR.

None of these are code bugs. A capture card makes all three go away. The shell/pressure
accuracy problem remains either way until there's a trained detector.

## 8. What Aidan runs next (≈5 minutes, on the laptop)

```powershell
cd <repo>; py -m pip install -e ".[vision]"      # now includes windows-capture on Windows
py -m cfb_coach watch --list-windows             # which "Xbox" windows exist; which are CLOAKED
py -m cfb_coach watch --probe --window "XBOX"    # click the stream during the 5s countdown
```

Read the verdict per backend (`wgc`, `dxcam`, `mss`):

- `OK`: that path works. Use `--capture <backend>`.
- `FROZEN`: the stream paused. Focus wasn't on it, or you were in a menu.
- `NO FIELD`: frames arrive but show no grass. Check `~/.cfb-coach/probe/probe_*.png` for the crop, browser chrome, or a pause overlay.
- `BLACK`: the surface is protected, or the window is hidden. Use a capture card.
- `NO FRAMES — <error>`: the exact exception is shown.

Run the probe a second time **without** clicking the stream. If the verdict flips to
`FROZEN`, that proves the focus/pause cause on your machine.

## Code map (v1.9.7)

- `cfb_coach/vision/winenum.py`: DPI awareness, ctypes enumeration (cloaked/minimized/terminal filter, client rect), ranking, clamp math, alias needles.
- `cfb_coach/vision/capture.py`: dxcam full-output camera plus clamped region; `DeviceCapture` DSHOW at 1080p.
- `cfb_coach/vision/wgc_capture.py`: `WgcWindowCapture` (by HWND, copies frames, stale returns None).
- `cfb_coach/vision/threaded_capture.py`: `last_error` / `errors` in `debug_stats()`.
- `cfb_coach/vision/pipeline.py`: `image_to_field` y-flip; `--capture wgc|mss` selection.
- `cfb_coach/vision/probe.py`: `watch --probe`.
- `cfb_coach/watch.py`: heartbeat shows the last error; overlay not auto-opened during live capture; detailed `--list-windows`.
- `tests/test_vision_capture_rca.py`: one regression test per root cause. 7 of them fail on v1.9.6.

Sidecar doctrine unchanged: capture reads pixels only; nothing sends input to the Xbox.
The typing-only `play` flow is untouched.
