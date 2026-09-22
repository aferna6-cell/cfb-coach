"""Pluggable vision for SCREEN CO-PILOT (v1.8 — live vision Milestone 1).

DefenseLook remains the coach tip contract. GameObservation is the richer
live-vision state; pipeline maps GameObservation → DefenseLook.

Capture: Xbox Remote Play on Windows laptop is the **active prototype** video
source (play stays on HDMI monitor + Xbox controller). Capture card is a
future drop-in CaptureBackend. Side-car only — NEVER control Xbox.

Windows-only deps (dxcam, win32, cv2) are lazy / try-except so
`import cfb_coach` and prep/play work on Linux without vision extras.
"""

from __future__ import annotations

from cfb_coach.vision.look import (
    CAPTURE_SETUP_NOTES,
    FRONTS,
    PRESSURES,
    SHELLS,
    CaptureBackend as LegacyCaptureBackend,
    DefenseLook,
    ImageFileCapture as LegacyImageFileCapture,
    StubCapture as LegacyStubCapture,
    demo_sequence,
    naive_roi_heuristic,
    parse_look_tokens,
)
from cfb_coach.vision.observation import (
    DEFAULT_CONF_THRESHOLD,
    FORMATIONS,
    OBS_PRESSURES,
    OBS_SHELLS,
    PLAY_STATES,
    GameObservation,
    SituationHUD,
)
from cfb_coach.vision.capture import (
    CaptureBackend,
    DeviceCapture,
    DxcamWindowCapture,
    Frame,
    ImageFileCapture,
    MssRegionCapture,
    StubCapture,
    VideoFileCapture,
    frame_to_bytes,
)
from cfb_coach.vision.smooth import TemporalSmoother, majority_vote
from cfb_coach.vision.formation import classify_formation
from cfb_coach.vision.pressure import classify_pressure, classify_shell
from cfb_coach.vision.play_state import PlayStateTracker
from cfb_coach.vision.pipeline import VisionPipeline, build_capture_from_args

__all__ = [
    "CAPTURE_SETUP_NOTES",
    "FRONTS",
    "PRESSURES",
    "SHELLS",
    "FORMATIONS",
    "OBS_PRESSURES",
    "OBS_SHELLS",
    "PLAY_STATES",
    "DEFAULT_CONF_THRESHOLD",
    "DefenseLook",
    "GameObservation",
    "SituationHUD",
    "CaptureBackend",
    "Frame",
    "StubCapture",
    "ImageFileCapture",
    "VideoFileCapture",
    "MssRegionCapture",
    "DxcamWindowCapture",
    "DeviceCapture",
    "frame_to_bytes",
    "demo_sequence",
    "naive_roi_heuristic",
    "parse_look_tokens",
    "TemporalSmoother",
    "majority_vote",
    "classify_formation",
    "classify_pressure",
    "classify_shell",
    "PlayStateTracker",
    "VisionPipeline",
    "build_capture_from_args",
    # legacy aliases
    "LegacyCaptureBackend",
    "LegacyStubCapture",
    "LegacyImageFileCapture",
]
