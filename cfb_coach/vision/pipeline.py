"""Vision pipeline: capture → crop → CV → smooth → GameObservation → DefenseLook."""

from __future__ import annotations

import time
from typing import Any, Callable

from cfb_coach.vision.calibrate import load_calib
from cfb_coach.vision.capture import (
    CaptureBackend,
    DxcamWindowCapture,
    Frame,
    ImageFileCapture,
    MssRegionCapture,
    StubCapture,
    VideoFileCapture,
    frame_to_bytes,
)
from cfb_coach.vision.formation import classify_formation
from cfb_coach.vision.hud import extract_hud
from cfb_coach.vision.look import DefenseLook, naive_roi_heuristic
from cfb_coach.vision.observation import GameObservation, SituationHUD
from cfb_coach.vision.play_state import PlayStateTracker
from cfb_coach.vision.play_tracker import PlayTracker
from cfb_coach.vision.players import extract_player_centroids, split_offense_defense
from cfb_coach.vision.pressure import classify_pressure, classify_shell
from cfb_coach.vision.smooth import TemporalSmoother


class VisionPipeline:
    """Classical CV pipeline targeting ~5–10 analyzed FPS (sidecar only)."""

    def __init__(
        self,
        capture: CaptureBackend | None = None,
        *,
        calib: dict[str, Any] | None = None,
        smooth_window: int = 5,
        smooth_threshold: int = 3,
        conf_threshold: float = 0.35,
        use_ocr: bool = False,
    ) -> None:
        self.capture = capture or StubCapture()
        self.calib = calib if calib is not None else load_calib()
        self.conf_threshold = conf_threshold
        self.use_ocr = use_ocr
        self.smoother = TemporalSmoother(window=smooth_window, threshold=smooth_threshold)
        self.play_tracker = PlayStateTracker()
        self.lifecycle = PlayTracker(state=self.play_tracker)
        self._last_play = None
        self._last_gray: Any = None
        self._thread_stats: dict[str, Any] = {}
        self._fps = 0.0
        self._last_t = 0.0
        self._frames = 0

    @property
    def fps(self) -> float:
        return self._fps

    def process_frame(self, frame: Frame | bytes | None) -> GameObservation:
        now = time.time()
        self._tick_fps(now)

        bgr = None
        if isinstance(frame, Frame):
            bgr = frame.bgr
            if bgr is None and frame.png_bytes:
                # decode later if needed
                pass
        elif isinstance(frame, (bytes, bytearray)):
            # Legacy bytes — naive heuristic path
            look = naive_roi_heuristic(bytes(frame))
            obs = GameObservation(
                shell="two_high" if look.shell == "two_high" else (
                    "one_high" if look.shell == "single_high" else "unknown"
                ),
                pressure={
                    "none": "none",
                    "show": "show",
                    "blitz_left": "left",
                    "blitz_right": "right",
                    "blitz_middle": "middle",
                    "all_out": "all_out",
                }.get(look.pressure, "unknown"),
                confidence={"shell": look.confidence, "pressure": look.confidence},
                source="image",
                notes=look.notes,
                ts=now,
            )
            return obs.normalized(conf_threshold=self.conf_threshold)

        if bgr is None and isinstance(frame, Frame) and frame.png_bytes:
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore

                arr = np.frombuffer(frame.png_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                bgr = None

        bgr = self._apply_crop(bgr)
        rois = (self.calib or {}).get("rois") or {}

        sit = extract_hud(bgr, rois=rois, use_ocr=self.use_ocr)
        centroids = extract_player_centroids(
            bgr, field_roi=rois.get("field") if rois else None
        )
        offense, defense = split_offense_defense(centroids)

        formation, form_side, form_conf = classify_formation(
            offense or centroids, conf_threshold=self.conf_threshold
        )
        shell, shell_conf = classify_shell(defense or centroids, conf_threshold=self.conf_threshold)
        pressure, press_conf = classify_pressure(
            defense or centroids, conf_threshold=self.conf_threshold
        )

        motion = self._motion_energy(bgr)
        # Lifecycle (snap/end) drives state machine with multi-signal evidence
        hud_dict = {}
        if isinstance(sit, SituationHUD):
            hud_dict = {
                "down": sit.down,
                "distance": sit.distance,
                "quarter": sit.quarter,
                "score_us": sit.score_us,
                "score_them": sit.score_them,
                "yardline": None,
            }
        cents = [(float(c[0]), float(c[1])) for c in (centroids or [])[:22]]
        play_state, completed = self.lifecycle.feed(
            now=now,
            motion=motion,
            centroids=cents,
            play_clock=getattr(sit, "clock", None) if isinstance(sit, SituationHUD) else None,
            hud=hud_dict,
            formation=formation,
            formation_side=form_side,
            shell=shell,
            pressure=pressure,
        )
        if completed is not None:
            self._last_play = completed

        raw_vals = {
            "formation": formation,
            "shell": shell,
            "pressure": pressure,
            "play_state": play_state,
        }
        smoothed = self.smoother.update(raw_vals)

        extras = {
            "n_centroids": len(centroids),
            "motion": motion,
            "fps": self._fps,
            "last_play_id": getattr(self._last_play, "play_id", None),
        }
        cap = self.capture
        if hasattr(cap, "debug_stats"):
            try:
                extras.update(cap.debug_stats())
            except Exception:
                pass
        else:
            for a, k in (
                ("capture_fps", "capture_fps"),
                ("dropped", "dropped"),
                ("latency_ms", "latency_ms"),
                ("queue_depth", "queue"),
            ):
                if hasattr(cap, a):
                    try:
                        extras[k] = getattr(cap, a)
                    except Exception:
                        pass

        obs = GameObservation(
            situation=sit if isinstance(sit, SituationHUD) else SituationHUD(),
            formation=smoothed.get("formation", formation),
            formation_side=form_side,
            shell=smoothed.get("shell", shell),
            pressure=smoothed.get("pressure", pressure),
            play_state=smoothed.get("play_state", play_state),
            confidence={
                "formation": form_conf,
                "shell": shell_conf,
                "pressure": press_conf,
                "play_state": 0.6,
            },
            source=getattr(self.capture, "name", "capture"),
            notes="",
            extras=extras,
            ts=now,
        )
        return obs.normalized(conf_threshold=self.conf_threshold)

    def step(self) -> tuple[GameObservation, DefenseLook]:
        frame = self.capture.grab()
        obs = self.process_frame(frame)
        return obs, obs.to_defense_look()

    def close(self) -> None:
        try:
            self.capture.close()
        except Exception:
            pass

    def _apply_crop(self, bgr: Any) -> Any:
        if bgr is None:
            return None
        crop = (self.calib or {}).get("crop")
        if not isinstance(crop, dict):
            return bgr
        # If crop is absolute screen coords, we assume capture already region-limited.
        # Relative crop keys: rx,ry,rw,rh in 0..1
        if all(k in crop for k in ("rx", "ry", "rw", "rh")):
            h, w = int(bgr.shape[0]), int(bgr.shape[1])
            x0 = int(float(crop["rx"]) * w)
            y0 = int(float(crop["ry"]) * h)
            x1 = int((float(crop["rx"]) + float(crop["rw"])) * w)
            y1 = int((float(crop["ry"]) + float(crop["rh"])) * h)
            return bgr[y0:y1, x0:x1]
        return bgr

    def _motion_energy(self, bgr: Any) -> float:
        if bgr is None:
            return 0.0
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if self._last_gray is None:
                self._last_gray = gray
                return 0.0
            diff = cv2.absdiff(gray, self._last_gray)
            self._last_gray = gray
            return float(np.mean(diff) / 255.0)
        except Exception:
            return 0.0

    def _tick_fps(self, now: float) -> None:
        self._frames += 1
        if self._last_t <= 0:
            self._last_t = now
            return
        dt = now - self._last_t
        if dt >= 0.5:
            self._fps = self._frames / dt
            self._frames = 0
            self._last_t = now


def _is_calib_region_flag(screen_region: Any) -> bool:
    """True when --screen-region was passed with no L,T,W,H (use calib)."""
    return screen_region in ("calib", True, "")


def build_capture_from_args(args: Any) -> CaptureBackend:
    """Select capture backend from CLI namespace."""
    import sys

    image = getattr(args, "image", None)
    video = getattr(args, "video", None)
    window = getattr(args, "window", None)
    screen_region = getattr(args, "screen_region", None)
    device = getattr(args, "device", None)

    calib = load_calib()
    region_dict = None
    prefer_mss = False
    region_source = ""

    if _is_calib_region_flag(screen_region):
        prefer_mss = True
        region_dict = calib.get("screen_region") or calib.get("crop")
        if not isinstance(region_dict, dict):
            print(
                "No crop/screen_region in ~/.cfb-coach/vision_calib.json — "
                "run: cfb-coach watch --calibrate",
                file=sys.stderr,
            )
            region_dict = None
        else:
            region_source = "calib (~/.cfb-coach/vision_calib.json)"
    elif isinstance(screen_region, str) and screen_region.strip():
        from cfb_coach.vision.calibrate import parse_screen_region

        prefer_mss = True
        region_dict = parse_screen_region(screen_region)
        region_source = f"CLI ({screen_region})"
    elif isinstance(calib.get("screen_region") or calib.get("crop"), dict):
        # Soft fallback crop when capturing a window (dxcam region)
        region_dict = calib.get("screen_region") or calib.get("crop")
        region_source = "calib crop (with window)"

    if image:
        return ImageFileCapture(image)
    if video:
        return VideoFileCapture(video)
    if device is not None:
        from cfb_coach.vision.capture import DeviceCapture

        return DeviceCapture(int(device))

    win = window or calib.get("window_substring")
    want_live = (
        bool(win)
        or region_dict is not None
        or bool(getattr(args, "live", False))
        or _is_calib_region_flag(screen_region)
        or (isinstance(screen_region, str) and bool(screen_region.strip()))
    )
    if want_live:
        if prefer_mss and region_dict is not None:
            try:
                print(
                    f"Using mss --screen-region from {region_source}: "
                    f"left={region_dict.get('left')} top={region_dict.get('top')} "
                    f"width={region_dict.get('width')} height={region_dict.get('height')}"
                )
                return MssRegionCapture(region=region_dict)
            except ImportError:
                print(
                    "mss unavailable — falling back to dxcam/window if possible",
                    file=sys.stderr,
                )

        if win or region_dict:
            try:
                region_tuple = None
                if region_dict and all(
                    k in region_dict for k in ("left", "top", "width", "height")
                ):
                    l, t = int(region_dict["left"]), int(region_dict["top"])
                    region_tuple = (
                        l,
                        t,
                        l + int(region_dict["width"]),
                        t + int(region_dict["height"]),
                    )
                cap = DxcamWindowCapture(
                    window_substring=win or "Xbox",
                    region=region_tuple,
                )
                return cap
            except ImportError:
                pass
            try:
                if region_dict is not None:
                    print(
                        f"Using mss region from {region_source or 'calib'}: {region_dict}"
                    )
                return MssRegionCapture(region=region_dict)
            except ImportError:
                pass
        # Fall through to stub if deps missing
    return StubCapture()


def observation_logger(
    db: Any,
) -> Callable[[GameObservation], None]:
    """Return a callback that logs observations to SQLite."""

    def _log(obs: GameObservation) -> None:
        try:
            db.log_vision_observation(obs.to_dict())
        except Exception:
            pass

    return _log
