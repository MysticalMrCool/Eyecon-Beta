"""One-Euro adaptive filter for cursor smoothing."""
from __future__ import annotations

import time
import math

from config import SmoothingConfig, GazeSmoothingConfig


class OneEuroFilter:
    """Speed-adaptive low-pass filter (Casiez et al., 2012).

    Low speed → heavy smoothing (stable fixation).
    High speed → light smoothing (responsive saccades).
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.5,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_hat: float | None = None
        self._dx_hat: float = 0.0
        self._last_time: float | None = None
        self._freq = freq

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        if timestamp is None:
            timestamp = time.monotonic()

        if self._last_time is None or self._x_hat is None:
            self._x_hat = x
            self._dx_hat = 0.0
            self._last_time = timestamp
            return x

        dt = timestamp - self._last_time
        if dt <= 0:
            return self._x_hat
        self._last_time = timestamp

        freq = 1.0 / dt

        # Filter the derivative.
        dx = (x - self._x_hat) * freq
        alpha_d = self._alpha(self._d_cutoff, freq)
        self._dx_hat = alpha_d * dx + (1 - alpha_d) * self._dx_hat

        # Adaptive cutoff based on speed.
        cutoff = self._min_cutoff + self._beta * abs(self._dx_hat)

        # Filter the signal.
        alpha = self._alpha(cutoff, freq)
        self._x_hat = alpha * x + (1 - alpha) * self._x_hat

        return self._x_hat

    def reset(self) -> None:
        self._x_hat = None
        self._dx_hat = 0.0
        self._last_time = None


class ScreenSmoother:
    """Apply One-Euro filtering independently to x and y screen coordinates."""

    def __init__(self, smoothing_config: SmoothingConfig):
        self._filter_x = OneEuroFilter(
            min_cutoff=smoothing_config.min_cutoff,
            beta=smoothing_config.beta,
            d_cutoff=smoothing_config.d_cutoff,
        )
        self._filter_y = OneEuroFilter(
            min_cutoff=smoothing_config.min_cutoff,
            beta=smoothing_config.beta,
            d_cutoff=smoothing_config.d_cutoff,
        )

    def smooth(self, x: float, y: float) -> tuple[float, float]:
        t = time.monotonic()
        return self._filter_x(x, t), self._filter_y(y, t)

    def reset(self) -> None:
        self._filter_x.reset()
        self._filter_y.reset()


class GazeSmoother:
    """Apply One-Euro filtering to raw gaze angles BEFORE ray-plane intersection.

    This is the most important smoothing stage: small angular jitter at the
    gaze level gets geometrically amplified by the ray-plane projection
    (0.5° noise at 500 mm ≈ 4 mm ≈ 16 px on screen).  Filtering in angular
    space keeps fixations rock-solid while still allowing fast saccades.
    """

    def __init__(self, config: GazeSmoothingConfig):
        self._filter_pitch = OneEuroFilter(
            min_cutoff=config.min_cutoff,
            beta=config.beta,
            d_cutoff=config.d_cutoff,
        )
        self._filter_yaw = OneEuroFilter(
            min_cutoff=config.min_cutoff,
            beta=config.beta,
            d_cutoff=config.d_cutoff,
        )

    def smooth(self, pitch: float, yaw: float) -> tuple[float, float]:
        t = time.monotonic()
        return self._filter_pitch(pitch, t), self._filter_yaw(yaw, t)

    def reset(self) -> None:
        self._filter_pitch.reset()
        self._filter_yaw.reset()
