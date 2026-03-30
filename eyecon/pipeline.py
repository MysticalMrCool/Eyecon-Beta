"""Main processing loop — ties all modules together."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from config import EyeconConfig
from capture import FrameGrabber
from landmarks import FaceLandmarkDetector, LandmarkResult, _PNP_LANDMARK_IDX, _MODEL_POINTS_3D
from normalization import GazeNormalizer
from gaze_model import GazeEstimator, unnormalize_gaze
from calibration import (
    CalibrationModel, CalibrationUI, CalibrationData, get_profile_path,
)
from smoothing import ScreenSmoother, GazeSmoother
from wink import WinkDetector
from cursor import CursorController
from utils import pitchyaw_to_vector, compute_face_center_3d


def _gaze_ray_intersect(
    pitch_cam: float,
    yaw_cam: float,
    landmark_result: LandmarkResult,
) -> np.ndarray | None:
    """Intersect the gaze ray with the z=0 (camera/screen) plane.

    Returns (2,) array [x_int, y_int] in mm on the virtual screen plane,
    or None if the gaze direction is nearly parallel to the plane.
    """
    face_center_3d = compute_face_center_3d(
        landmark_result.rotation_matrix,
        landmark_result.translation_vec,
        _MODEL_POINTS_3D,
    ).flatten()

    gaze_dir = pitchyaw_to_vector(pitch_cam, yaw_cam)

    # The gaze must point toward the camera (negative z) to hit the screen.
    if gaze_dir[2] > -0.1:
        return None

    # Ray-plane intersection: origin + t * direction, solve for z = 0.
    t = -face_center_3d[2] / gaze_dir[2]
    x_int = face_center_3d[0] + t * gaze_dir[0]
    y_int = face_center_3d[1] + t * gaze_dir[1]
    return np.array([x_int, y_int], dtype=np.float32)


def fallback_projection(
    features: np.ndarray,
    config: EyeconConfig,
) -> np.ndarray:
    """Rough linear projection when no calibration profile is available.

    Maps intersection coordinates (mm) to screen pixels assuming the camera
    is centred above the screen and the screen subtends ~60° horizontal FOV.
    """
    # Approximate screen physical size from assumed viewing distance.
    # At ~550 mm distance with 60° HFOV the screen is about 640 mm wide.
    assumed_screen_w_mm = 640.0
    assumed_screen_h_mm = assumed_screen_w_mm * config.screen.height / config.screen.width

    x = config.screen.width / 2 - features[0] * (config.screen.width / assumed_screen_w_mm)
    y = features[1] * (config.screen.height / assumed_screen_h_mm)
    return np.array([x, y])


class EyeconPipeline:
    def __init__(self, config: EyeconConfig, move_cursor: bool = True):
        self._cfg = config
        self._move_cursor = move_cursor

        # Resolve model paths relative to the package directory.
        pkg_dir = Path(__file__).resolve().parent
        gaze_model_path = pkg_dir / config.gaze.model_path
        config.gaze.model_path = gaze_model_path

        self._grabber = FrameGrabber(config.camera)
        self._detector = FaceLandmarkDetector(config.camera, config.wink)
        self._normalizer = GazeNormalizer(config.gaze, config.camera)
        self._estimator = GazeEstimator(config.gaze)
        self._smoother = ScreenSmoother(config.smoothing)
        self._gaze_smoother = GazeSmoother(config.gaze_smoothing)
        self._wink = WinkDetector(config.wink)
        self._cursor = CursorController(config.screen)
        self._calibration = CalibrationModel(config.calibration)

        self._running = False

        # Try to load an existing calibration profile.
        profile = get_profile_path(
            config.calibration,
            config.camera.device_id,
            config.screen.width,
            config.screen.height,
        )
        if profile.exists():
            if self._calibration.load(profile):
                print(f"Loaded calibration profile: {profile}")

    # ---- feature extraction (used by calibration UI) --------------------

    def get_features(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Run the pipeline up to the feature-vector stage.

        Returns: (2,) array [x_int, y_int] — gaze ray intersection with the
                 z=0 (camera/screen) plane in mm — or None if detection failed.

        NOTE: This intentionally does NOT apply gaze smoothing. During
        calibration the 30 raw samples per point are enough for the
        polynomial regression to average out noise, and the stateful
        smoother would introduce lag-bias between calibration dots.
        Smoothing is applied only at runtime in the main loop.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.process(frame_rgb)
        if result is None:
            return None

        norm_result = self._normalizer.normalize(frame_rgb, result)
        if norm_result is None:
            return None

        pitch_n, yaw_n = self._estimator.predict(norm_result.face_patch)
        pitch_cam, yaw_cam = unnormalize_gaze(
            pitch_n, yaw_n, norm_result.rotation_matrix,
        )

        # Apply systematic bias correction.
        pitch_cam += self._cfg.gaze.pitch_offset
        yaw_cam += self._cfg.gaze.yaw_offset

        return _gaze_ray_intersect(pitch_cam, yaw_cam, result)

    # ---- calibration ----------------------------------------------------

    def run_calibration(self) -> bool:
        """Run the calibration UI. Returns True if calibration succeeded."""
        ui = CalibrationUI(self._cfg.screen, self._cfg.calibration, self._cfg.camera)
        data = ui.run_calibration(self.get_features)
        if data is None:
            print("Calibration cancelled.")
            return False

        loss = self._calibration.train(data)

        # --- Per-point diagnostics ---
        unique_targets = []
        seen = set()
        for t in data.targets:
            key = (float(t[0]), float(t[1]))
            if key not in seen:
                seen.add(key)
                unique_targets.append(t)

        print(f"\nCalibration trained — overall MSE: {loss:.1f} px²")
        print(f"  {'Target':>20s}    {'Predicted':>20s}    {'Error'}")
        print(f"  {'------':>20s}    {'---------':>20s}    {'-----'}")
        errors = []
        point_data = []  # for visual validation
        for tgt in unique_targets:
            mask = np.all(np.abs(data.targets - tgt) < 0.5, axis=1)
            mean_feat = data.features[mask].mean(axis=0)
            pred = self._calibration.predict(mean_feat)
            err = np.sqrt(np.sum((pred - tgt) ** 2))
            n_samples = int(mask.sum())
            errors.append(err)
            point_data.append((tgt, pred, mean_feat, n_samples))
            print(f"  ({tgt[0]:7.0f}, {tgt[1]:5.0f})    "
                  f"({pred[0]:7.0f}, {pred[1]:5.0f})    "
                  f"{err:5.1f}px  ({n_samples} samples)")
        print(f"  Mean error: {np.mean(errors):.1f}px, "
              f"Max: {np.max(errors):.1f}px\n")

        # --- Visual validation screen ---
        self._show_calibration_validation(point_data)

        profile = get_profile_path(
            self._cfg.calibration,
            self._cfg.camera.device_id,
            self._cfg.screen.width,
            self._cfg.screen.height,
        )
        self._calibration.save(profile)
        print(f"Saved calibration profile: {profile}")
        self._smoother.reset()
        self._gaze_smoother.reset()
        return True

    def _show_calibration_validation(
        self,
        point_data: list[tuple],
    ) -> None:
        """Show a fullscreen overlay comparing target vs predicted positions."""
        w, h = self._cfg.screen.width, self._cfg.screen.height
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        for tgt, pred, _, n_samples in point_data:
            tx, ty = int(tgt[0]), int(tgt[1])
            px, py = int(np.clip(pred[0], 0, w)), int(np.clip(pred[1], 0, h))
            err = np.sqrt((pred[0] - tgt[0]) ** 2 + (pred[1] - tgt[1]) ** 2)

            # Grey line connecting target to prediction.
            cv2.line(canvas, (tx, ty), (px, py), (80, 80, 80), 1)
            # Green hollow circle = target (where the dot was).
            cv2.circle(canvas, (tx, ty), 18, (0, 255, 0), 2)
            # Red filled circle = polynomial prediction.
            cv2.circle(canvas, (px, py), 8, (0, 0, 255), -1)
            # Error label.
            cv2.putText(canvas, f"{err:.0f}px",
                        (tx + 25, ty + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1)

        cv2.putText(canvas,
                    "Green = target dot position, Red = predicted cursor position",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(canvas,
                    "Press any key to continue...",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        val_win = "Calibration Validation"
        cv2.namedWindow(val_win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(val_win, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.imshow(val_win, canvas)
        cv2.waitKey(0)
        cv2.destroyWindow(val_win)

    # ---- main loop ------------------------------------------------------

    def start(self) -> None:
        self._grabber.start()
        self._running = True
        print("Eyecon pipeline running.")
        print("  Debug window: ESC to stop")
        print("  F9 = toggle cursor | F10 = recalibrate | ESC = quit")

        fps_timer = time.monotonic()
        frame_count = 0
        fps_display = 0.0

        try:
            while self._running:
                ret, frame = self._grabber.read()
                if not ret:
                    continue

                frame_count += 1
                elapsed = time.monotonic() - fps_timer
                if elapsed >= 1.0:
                    fps_display = frame_count / elapsed
                    frame_count = 0
                    fps_timer = time.monotonic()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self._detector.process(frame_rgb)

                if result is None:
                    if self._cfg.debug:
                        cv2.putText(frame, "No face detected", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self._show_debug(frame, fps_display)
                    continue

                # Wink detection (runs on raw landmarks).
                wink_event = self._wink.update(
                    result.left_eye_landmarks,
                    result.right_eye_landmarks,
                )
                if wink_event and self._cursor.enabled and self._move_cursor:
                    if wink_event.eye == "left":
                        self._cursor.left_click()
                    else:
                        self._cursor.right_click()

                # Data normalisation.
                norm_result = self._normalizer.normalize(frame_rgb, result)
                if norm_result is None:
                    if self._cfg.debug:
                        cv2.putText(frame, "Normalization failed", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self._show_debug(frame, fps_display)
                    continue

                # Gaze estimation.
                pitch_n, yaw_n = self._estimator.predict(norm_result.face_patch)
                pitch_cam, yaw_cam = unnormalize_gaze(
                    pitch_n, yaw_n, norm_result.rotation_matrix,
                )

                # Apply systematic bias correction.
                pitch_cam += self._cfg.gaze.pitch_offset
                yaw_cam += self._cfg.gaze.yaw_offset

                # Smooth gaze angles before geometric projection.
                pitch_cam, yaw_cam = self._gaze_smoother.smooth(
                    pitch_cam, yaw_cam)

                # Gaze ray → screen-plane intersection (2D feature vector).
                features = _gaze_ray_intersect(pitch_cam, yaw_cam, result)
                if features is None:
                    if self._cfg.debug:
                        cv2.putText(frame, "Gaze behind camera", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self._show_debug(frame, fps_display)
                    continue

                # Map to screen coordinates.
                if self._calibration.is_calibrated:
                    screen_xy = self._calibration.predict(features)
                else:
                    screen_xy = fallback_projection(features, self._cfg)

                # Clamp to screen bounds before smoothing.
                raw_x = max(0.0, min(float(screen_xy[0]),
                                     float(self._cfg.screen.width)))
                raw_y = max(0.0, min(float(screen_xy[1]),
                                     float(self._cfg.screen.height)))

                # Temporal smoothing.
                sx, sy = self._smoother.smooth(raw_x, raw_y)

                # Move cursor.
                if self._move_cursor and self._cursor.enabled:
                    self._cursor.move(sx, sy)

                # Debug visualisation.
                if self._cfg.debug:
                    self._draw_debug_overlay(
                        frame, result, pitch_cam, yaw_cam, sx, sy)
                    self._show_debug(frame, fps_display)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self._grabber.release()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")

    @property
    def cursor(self) -> CursorController:
        return self._cursor

    @property
    def calibration(self) -> CalibrationModel:
        return self._calibration

    # ---- debug helpers --------------------------------------------------

    def _draw_debug_overlay(self, frame, result, pitch_cam, yaw_cam, sx, sy):
        from landmarks import _PNP_LANDMARK_IDX
        from wink import compute_ear

        # Face bounding box.
        bx, by, bw, bh = result.face_bbox
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)

        # Head pose axes.
        rvec, _ = cv2.Rodrigues(result.rotation_matrix)
        origin = np.float64([[0, 0, 0]])
        axes = np.float64([[50, 0, 0], [0, 50, 0], [0, 0, 50]])
        cam_mat = self._cfg.camera.camera_matrix
        dist = self._cfg.camera.dist_coeffs
        o2d, _ = cv2.projectPoints(origin, rvec, result.translation_vec, cam_mat, dist)
        a2d, _ = cv2.projectPoints(axes, rvec, result.translation_vec, cam_mat, dist)
        o = tuple(o2d[0].ravel().astype(int))
        for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            p = tuple(a2d[i].ravel().astype(int))
            cv2.arrowedLine(frame, o, p, c, 2, tipLength=0.2)

        # Gaze arrow (yellow).
        ox, oy = int(result.face_center_px[0]), int(result.face_center_px[1])
        length = 150.0
        dx = -length * np.sin(yaw_cam) * np.cos(pitch_cam)
        dy = -length * np.sin(pitch_cam)
        cv2.arrowedLine(frame, (ox, oy),
                        (int(ox + dx), int(oy + dy)),
                        (0, 255, 255), 2, tipLength=0.18)

        # EAR values.
        left_ear = compute_ear(result.left_eye_landmarks)
        right_ear = compute_ear(result.right_eye_landmarks)
        cv2.putText(frame, f"L-EAR: {left_ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"R-EAR: {right_ear:.3f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Gaze angles + screen target.
        cv2.putText(frame,
                    f"Gaze P:{np.degrees(pitch_cam):+.1f} Y:{np.degrees(yaw_cam):+.1f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Screen: ({int(sx)}, {int(sy)})",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Wink state.
        wstate = self._wink.get_state()
        cv2.putText(frame,
                    f"Wink L:{wstate['left_state']} R:{wstate['right_state']}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Calibration status.
        cal_str = "calibrated" if self._calibration.is_calibrated else "fallback"
        cv2.putText(frame, f"[{cal_str}]", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 150), 1)

    def _show_debug(self, frame, fps_display):
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Eyecon Debug", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self._running = False
