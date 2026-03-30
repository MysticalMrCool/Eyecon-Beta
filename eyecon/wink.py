"""EAR-based wink detection state machine."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from config import WinkConfig


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class WinkEvent:
    eye: str            # "left" or "right"
    timestamp: float    # time.monotonic()
    duration_frames: int


# ---------------------------------------------------------------------------
# EAR computation
# ---------------------------------------------------------------------------

def compute_ear(eye_landmarks: np.ndarray) -> float:
    """Compute Eye Aspect Ratio from 6 landmark points.

    Point order: [outer_corner, upper_1, upper_2, inner_corner, lower_2, lower_1]
    EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
    """
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


# ---------------------------------------------------------------------------
# Per-eye state machine
# ---------------------------------------------------------------------------

class _EyeState(Enum):
    OPEN = auto()
    CLOSING = auto()
    CLOSED = auto()


class _EyeTracker:
    """Track open/close state for a single eye."""

    def __init__(self):
        self.state = _EyeState.OPEN
        self.close_frames: int = 0
        self.last_close_duration: int = 0   # preserved after re-open
        self.close_start_frame: int = 0

    def update(self, below_thresh: bool, frame_num: int, min_wink: int) -> bool:
        """Update state machine. Returns True if the eye just re-opened from CLOSING/CLOSED."""
        just_opened = False

        if self.state == _EyeState.OPEN:
            if below_thresh:
                self.state = _EyeState.CLOSING
                self.close_frames = 1
                self.close_start_frame = frame_num

        elif self.state == _EyeState.CLOSING:
            if below_thresh:
                self.close_frames += 1
                if self.close_frames >= min_wink:
                    self.state = _EyeState.CLOSED
            else:
                # Noise — reopened before min duration.
                self.last_close_duration = self.close_frames
                self.close_frames = 0
                self.state = _EyeState.OPEN
                just_opened = True

        elif self.state == _EyeState.CLOSED:
            if below_thresh:
                self.close_frames += 1
            else:
                self.last_close_duration = self.close_frames
                self.close_frames = 0
                self.state = _EyeState.OPEN
                just_opened = True

        return just_opened


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class WinkDetector:
    def __init__(self, wink_config: WinkConfig):
        self._cfg = wink_config
        self._left = _EyeTracker()
        self._right = _EyeTracker()

        self._baseline_left: float | None = None
        self._baseline_right: float | None = None
        self._baseline_buf_left: list[float] = []
        self._baseline_buf_right: list[float] = []
        self._baseline_done = False

        self._frame_num = 0
        self._refractory_until = 0  # frame number

        # EMA decay for baseline adaptation during OPEN state.
        self._ema_alpha = 0.005

    # ---- baseline calibration ------------------------------------------

    def calibrate_baseline(self, left_ear: float, right_ear: float) -> None:
        """Feed EAR values during the initial baseline period (eyes open)."""
        if self._baseline_done:
            return
        self._baseline_buf_left.append(left_ear)
        self._baseline_buf_right.append(right_ear)
        if len(self._baseline_buf_left) >= self._cfg.ear_baseline_frames:
            self._baseline_left = float(np.median(self._baseline_buf_left))
            self._baseline_right = float(np.median(self._baseline_buf_right))
            self._baseline_done = True

    @property
    def baseline_ready(self) -> bool:
        return self._baseline_done

    # ---- per-frame update ----------------------------------------------

    def update(
        self,
        left_eye_lm: np.ndarray,
        right_eye_lm: np.ndarray,
    ) -> WinkEvent | None:
        self._frame_num += 1
        left_ear = compute_ear(left_eye_lm)
        right_ear = compute_ear(right_eye_lm)

        # Still collecting baseline — just accumulate and return.
        if not self._baseline_done:
            self.calibrate_baseline(left_ear, right_ear)
            return None

        thresh_left = self._baseline_left * self._cfg.ear_close_ratio
        thresh_right = self._baseline_right * self._cfg.ear_close_ratio

        left_below = left_ear < thresh_left
        right_below = right_ear < thresh_right

        left_opened = self._left.update(
            left_below, self._frame_num, self._cfg.min_wink_frames)
        right_opened = self._right.update(
            right_below, self._frame_num, self._cfg.min_wink_frames)

        # Blink rejection: if both eyes were closing at roughly the same time,
        # this is a blink, not a wink.
        both_were_closing = (
            abs(self._left.close_start_frame - self._right.close_start_frame)
            <= self._cfg.blink_sync_tolerance
            and self._left.close_start_frame > 0
            and self._right.close_start_frame > 0
        )

        event: WinkEvent | None = None

        if self._frame_num >= self._refractory_until and not both_were_closing:
            if left_opened and self._right.state == _EyeState.OPEN:
                dur = self._left.last_close_duration
                if self._cfg.min_wink_frames <= dur <= self._cfg.max_wink_frames:
                    event = WinkEvent(
                        eye="left",
                        timestamp=time.monotonic(),
                        duration_frames=dur,
                    )
                    self._refractory_until = self._frame_num + self._cfg.refractory_frames

            elif right_opened and self._left.state == _EyeState.OPEN:
                dur = self._right.last_close_duration
                if self._cfg.min_wink_frames <= dur <= self._cfg.max_wink_frames:
                    event = WinkEvent(
                        eye="right",
                        timestamp=time.monotonic(),
                        duration_frames=dur,
                    )
                    self._refractory_until = self._frame_num + self._cfg.refractory_frames

        # Adapt baseline slowly during OPEN periods.
        if self._left.state == _EyeState.OPEN and left_ear > thresh_left:
            self._baseline_left += self._ema_alpha * (left_ear - self._baseline_left)
        if self._right.state == _EyeState.OPEN and right_ear > thresh_right:
            self._baseline_right += self._ema_alpha * (right_ear - self._baseline_right)

        return event

    # ---- state queries (for debug overlay) -----------------------------

    def get_state(self) -> dict:
        return {
            "left_state": self._left.state.name,
            "right_state": self._right.state.name,
            "baseline_left": self._baseline_left,
            "baseline_right": self._baseline_right,
            "baseline_ready": self._baseline_done,
        }
