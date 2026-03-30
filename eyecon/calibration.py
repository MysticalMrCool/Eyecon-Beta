"""Per-user calibration: UI, polynomial gaze→screen mapping, and profile persistence.

Replaces the original MLP-based mapping with a geometric approach:
1. Compute gaze ray-plane intersection (eye position + gaze direction → screen plane)
2. Map intersection coordinates to screen pixels via polynomial ridge regression

The ray-plane intersection naturally accounts for head translation (parallax),
and the polynomial corrects for the unknown camera-screen geometry.
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CalibrationConfig, ScreenConfig, CameraConfig


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationData:
    features: np.ndarray      # (N, 2) gaze ray-plane intersection coords (mm)
    targets: np.ndarray       # (N, 2) screen coordinates in pixels
    screen_width: int
    screen_height: int
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))


# ---------------------------------------------------------------------------
# Polynomial ridge regression (replaces the MLP)
# ---------------------------------------------------------------------------

def _poly_features(xy: np.ndarray, degree: int = 2) -> np.ndarray:
    """Build polynomial feature matrix from (N, 2) input.

    degree=2: [1, x, y, x², xy, y²]               → 6 terms
    degree=3: [1, x, y, x², xy, y², x³, x²y, xy², y³] → 10 terms
    """
    x, y = xy[:, 0:1], xy[:, 1:2]
    terms = [np.ones((len(xy), 1))]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            terms.append((x ** (d - i)) * (y ** i))
    return np.hstack(terms)


# ---------------------------------------------------------------------------
# Calibration model (train / predict / save / load)
# ---------------------------------------------------------------------------

class CalibrationModel:
    """Maps gaze ray-plane intersection coordinates to screen pixels.

    Uses polynomial ridge regression instead of a neural network.
    With degree=2 this fits 6 coefficients per axis (12 total) from ~270
    calibration samples — massively overdetermined and robust.
    """

    def __init__(self, calibration_config: CalibrationConfig):
        self._cfg = calibration_config
        self._coeffs_x: np.ndarray | None = None   # (n_terms,)
        self._coeffs_y: np.ndarray | None = None   # (n_terms,)
        self._feat_mean: np.ndarray | None = None   # (2,) centering
        self._feat_scale: np.ndarray | None = None  # (2,) scaling
        self._screen_w: int = 0
        self._screen_h: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self._coeffs_x is not None

    def train(self, data: CalibrationData) -> float:
        """Fit polynomial mapping on collected calibration data.

        Returns final mean squared pixel error on training data.
        """
        self._screen_w = data.screen_width
        self._screen_h = data.screen_height

        # Center and scale features for numerical stability.
        self._feat_mean = data.features.mean(axis=0).astype(np.float64)
        self._feat_scale = data.features.std(axis=0).astype(np.float64)
        self._feat_scale[self._feat_scale < 1e-7] = 1.0

        xy = (data.features - self._feat_mean) / self._feat_scale
        Phi = _poly_features(xy.astype(np.float64), self._cfg.poly_degree)
        targets = data.targets.astype(np.float64)

        # Ridge regression: (Φ^T Φ + αI) β = Φ^T y
        alpha = self._cfg.ridge_alpha
        A = Phi.T @ Phi + alpha * np.eye(Phi.shape[1])
        self._coeffs_x = np.linalg.solve(A, Phi.T @ targets[:, 0])
        self._coeffs_y = np.linalg.solve(A, Phi.T @ targets[:, 1])

        # Training MSE (pixel²).
        pred_x = Phi @ self._coeffs_x
        pred_y = Phi @ self._coeffs_y
        mse = float(np.mean((pred_x - targets[:, 0]) ** 2
                            + (pred_y - targets[:, 1]) ** 2))
        return mse

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Map a (2,) intersection-coordinate vector to (2,) screen pixels."""
        if self._coeffs_x is None:
            raise RuntimeError("CalibrationModel is not calibrated.")
        xy = ((features - self._feat_mean) / self._feat_scale).astype(np.float64)
        Phi = _poly_features(xy.reshape(1, -1), self._cfg.poly_degree)
        sx = float(Phi @ self._coeffs_x)
        sy = float(Phi @ self._coeffs_y)
        return np.array([sx, sy])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": 2,
            "coeffs_x": self._coeffs_x,
            "coeffs_y": self._coeffs_y,
            "feat_mean": self._feat_mean,
            "feat_scale": self._feat_scale,
            "screen_width": self._screen_w,
            "screen_height": self._screen_h,
            "poly_degree": self._cfg.poly_degree,
        }, str(path))

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        if data.get("version") != 2:
            # Old MLP-based profile — incompatible, needs recalibration.
            return False
        self._coeffs_x = data["coeffs_x"]
        self._coeffs_y = data["coeffs_y"]
        self._feat_mean = data["feat_mean"]
        self._feat_scale = data["feat_scale"]
        self._screen_w = data["screen_width"]
        self._screen_h = data["screen_height"]
        return True


# ---------------------------------------------------------------------------
# Profile path helper
# ---------------------------------------------------------------------------

def get_profile_path(
    cfg: CalibrationConfig,
    camera_id: int,
    screen_w: int,
    screen_h: int,
) -> Path:
    key = f"{camera_id}_{screen_w}x{screen_h}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    base = Path(__file__).resolve().parent / cfg.save_dir
    return base / f"profile_{h}.pt"


# ---------------------------------------------------------------------------
# Calibration UI
# ---------------------------------------------------------------------------

class CalibrationUI:
    def __init__(
        self,
        screen_config: ScreenConfig,
        calibration_config: CalibrationConfig,
        camera_config: CameraConfig,
    ):
        self._scr = screen_config
        self._cal = calibration_config
        self._cam = camera_config

    def get_calibration_points(self) -> list[tuple[int, int]]:
        """Return calibration dot positions in screen pixel coordinates."""
        mx = int(self._scr.width * self._cal.margin_fraction)
        my = int(self._scr.height * self._cal.margin_fraction)

        cols = self._cal.grid_cols
        rows = self._cal.grid_rows
        xs = np.linspace(mx, self._scr.width - mx, cols).astype(int).tolist()
        ys = np.linspace(my, self._scr.height - my, rows).astype(int).tolist()

        points = [(x, y) for y in ys for x in xs]
        random.shuffle(points)
        return points

    def run_calibration(self, pipeline_step_fn) -> CalibrationData | None:
        """Run the full calibration flow.

        Args:
            pipeline_step_fn: callable that takes a BGR frame and returns a
                (2,) feature vector (x_int, y_int) — gaze ray-plane
                intersection coordinates in mm — or None if detection failed.

        Returns:
            CalibrationData on success, None if the user cancelled (ESC).
        """
        win_name = "Eyecon Calibration"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        points = self.get_calibration_points()
        all_features: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        cancelled = False

        from capture import FrameGrabber
        grabber = FrameGrabber(self._cam)
        grabber.start()

        try:
            for pt_idx, (tx, ty) in enumerate(points):
                # --- countdown phase ---
                for countdown in range(3, 0, -1):
                    t0 = time.monotonic()
                    while time.monotonic() - t0 < 0.6:
                        canvas = np.zeros((self._scr.height, self._scr.width, 3), dtype=np.uint8)
                        # Draw target dot (white, pulsing)
                        radius = 18 + int(6 * np.sin(time.monotonic() * 4))
                        cv2.circle(canvas, (tx, ty), radius, (255, 255, 255), -1)
                        # Countdown text
                        cv2.putText(canvas, str(countdown),
                                    (tx + 30, ty + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (200, 200, 200), 2)
                        # Progress indicator
                        cv2.putText(canvas, f"Point {pt_idx + 1}/{len(points)}",
                                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (150, 150, 150), 1)
                        cv2.imshow(win_name, canvas)
                        key = cv2.waitKey(16) & 0xFF
                        if key == 27:
                            cancelled = True
                            break
                    if cancelled:
                        break
                if cancelled:
                    break

                # --- settle phase: discard frames while gaze moves to dot ---
                settled = 0
                while settled < self._cal.settle_frames:
                    ret, frame = grabber.read()
                    if not ret:
                        continue
                    _ = pipeline_step_fn(frame)
                    settled += 1

                    canvas = np.zeros((self._scr.height, self._scr.width, 3), dtype=np.uint8)
                    cv2.circle(canvas, (tx, ty), 20, (0, 200, 200), -1)
                    cv2.putText(canvas, f"Point {pt_idx + 1}/{len(points)} — settling",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (150, 150, 150), 1)
                    cv2.imshow(win_name, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        cancelled = True
                        break
                if cancelled:
                    break

                # --- collection phase ---
                collected: list[np.ndarray] = []
                while len(collected) < self._cal.samples_per_point:
                    ret, frame = grabber.read()
                    if not ret:
                        continue
                    feat = pipeline_step_fn(frame)
                    if feat is not None:
                        collected.append(feat)

                    canvas = np.zeros((self._scr.height, self._scr.width, 3), dtype=np.uint8)
                    cv2.circle(canvas, (tx, ty), 20, (0, 255, 0), -1)
                    pct = len(collected) / self._cal.samples_per_point
                    cv2.putText(canvas,
                                f"Point {pt_idx + 1}/{len(points)} — collecting {int(pct * 100)}%",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (150, 150, 150), 1)
                    cv2.imshow(win_name, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        cancelled = True
                        break
                if cancelled:
                    break

                # Reject outlier samples (saccades, detection glitches)
                # using median-distance thresholding.
                stacked = np.array(collected, dtype=np.float32)
                med = np.median(stacked, axis=0)
                dists = np.linalg.norm(stacked - med, axis=1)
                cutoff = np.median(dists) * 3.0
                if cutoff > 1e-6:
                    mask = dists <= cutoff
                    if mask.sum() >= 5:
                        stacked = stacked[mask]

                # Store all kept samples for polynomial regression.
                target = np.array([tx, ty], dtype=np.float32)
                for feat in stacked:
                    all_features.append(feat)
                    all_targets.append(target)

                # Brief blue flash to signal "done".
                canvas = np.zeros((self._scr.height, self._scr.width, 3), dtype=np.uint8)
                cv2.circle(canvas, (tx, ty), 24, (255, 100, 0), -1)
                cv2.imshow(win_name, canvas)
                cv2.waitKey(300)

        finally:
            grabber.release()
            cv2.destroyWindow(win_name)

        if cancelled or len(all_features) < 4:
            return None

        return CalibrationData(
            features=np.array(all_features, dtype=np.float32),
            targets=np.array(all_targets, dtype=np.float32),
            screen_width=self._scr.width,
            screen_height=self._scr.height,
        )
