"""Phase 2 test script — normalization, gaze model, and gaze direction arrows.

Displays:
  - Main window: webcam feed with landmarks, head pose axes, and gaze arrows.
  - Side window: normalized (warped) face patch — should appear upright regardless
    of head tilt/rotation.

If the gaze model (.pth) is not present the script still runs to validate the
normalization pipeline; gaze arrows are simply skipped.
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure the eyecon package root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EyeconConfig, GazeModelConfig
from capture import FrameGrabber
from landmarks import FaceLandmarkDetector, _MODEL_POINTS_3D, _PNP_LANDMARK_IDX
from normalization import GazeNormalizer
from utils import pitchyaw_to_vector

# Optional — only needed when the gaze model (.pth) exists.
_gaze_available = False
try:
    from gaze_model import GazeEstimator, unnormalize_gaze
    _gaze_available = True
except ImportError:
    pass


# ── visualisation helpers ──────────────────────────────────────────────────


def draw_head_pose_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, length=50.0):
    """Draw RGB axes (X=red, Y=green, Z=blue) from the nose tip."""
    origin = np.float64([[0, 0, 0]])
    axes = np.float64([
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length],
    ])
    origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, camera_matrix, dist_coeffs)
    axes_2d, _ = cv2.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)

    o = tuple(origin_2d[0].ravel().astype(int))
    colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i, (c, label) in enumerate(zip(colours, "XYZ")):
        p = tuple(axes_2d[i].ravel().astype(int))
        cv2.arrowedLine(frame, o, p, c, 2, tipLength=0.2)
        cv2.putText(frame, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)


def draw_landmarks(frame, landmarks_px, indices=None, colour=(0, 255, 0), radius=1):
    if indices is None:
        indices = range(0, len(landmarks_px), 3)
    for i in indices:
        pt = tuple(landmarks_px[i].astype(int))
        cv2.circle(frame, pt, radius, colour, -1)


def draw_eye_landmarks(frame, eye_lm, colour=(255, 255, 0)):
    pts = eye_lm.astype(int)
    for i in range(len(pts)):
        cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), colour, 1)
        cv2.circle(frame, tuple(pts[i]), 2, colour, -1)


def compute_ear(eye_landmarks: np.ndarray) -> float:
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def draw_gaze_arrow(frame, face_center_px, pitch_cam, yaw_cam, length=150.0):
    """Draw a yellow arrow from the face center in the estimated gaze direction.

    Uses the same 2D projection as the official ETH-XGaze draw_gaze():
      dx = -length * sin(yaw) * cos(pitch)
      dy = -length * sin(pitch)
    This avoids the bug of passing camera-space vectors through projectPoints
    (which would double-apply the head rotation).
    """
    ox, oy = int(face_center_px[0]), int(face_center_px[1])
    dx = -length * np.sin(yaw_cam) * np.cos(pitch_cam)
    dy = -length * np.sin(pitch_cam)
    ex, ey = int(ox + dx), int(oy + dy)
    cv2.arrowedLine(frame, (ox, oy), (ex, ey), (0, 255, 255), 2, tipLength=0.18)


def face_patch_to_display(face_patch: np.ndarray, mean, std) -> np.ndarray:
    """Undo ImageNet normalization and convert CHW→HWC BGR for display."""
    img = face_patch.copy().transpose(1, 2, 0)  # CHW → HWC
    img = img * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ── main ───────────────────────────────────────────────────────────────────


def main():
    cfg = EyeconConfig()
    cam_cfg = cfg.camera
    gaze_cfg = cfg.gaze

    # Resolve model path relative to the eyecon package directory.
    model_path = Path(__file__).resolve().parent.parent / gaze_cfg.model_path
    gaze_cfg.model_path = model_path

    grabber = FrameGrabber(cam_cfg)
    detector = FaceLandmarkDetector(cam_cfg, cfg.wink)
    normalizer = GazeNormalizer(gaze_cfg, cam_cfg)

    estimator = None
    if _gaze_available and model_path.exists():
        print(f"Loading gaze model from {model_path}")
        estimator = GazeEstimator(gaze_cfg)
    else:
        print("Gaze model not found — running normalization-only mode.")
        print(f"  Expected at: {model_path}")
        print("  Run  python scripts/download_model.py  to download weights.")

    grabber.start()
    print("Phase 2 test — press ESC or Q to quit")

    fps_timer = time.monotonic()
    frame_count = 0
    fps_display = 0.0

    try:
        while True:
            ret, frame = grabber.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(frame_rgb)

            norm_patch_display = None  # will hold the de-normalized face patch

            if result is not None:
                # Draw sparse landmarks + PnP keypoints.
                draw_landmarks(frame, result.landmarks_px)
                for idx in _PNP_LANDMARK_IDX:
                    pt = tuple(result.landmarks_px[idx].astype(int))
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)

                # Head pose axes.
                rvec, _ = cv2.Rodrigues(result.rotation_matrix)
                draw_head_pose_axes(frame, rvec, result.translation_vec,
                                    cam_cfg.camera_matrix, cam_cfg.dist_coeffs)

                # Eye landmarks + EAR.
                draw_eye_landmarks(frame, result.left_eye_landmarks, (255, 255, 0))
                draw_eye_landmarks(frame, result.right_eye_landmarks, (0, 255, 255))
                left_ear = compute_ear(result.left_eye_landmarks)
                right_ear = compute_ear(result.right_eye_landmarks)
                cv2.putText(frame, f"L-EAR: {left_ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(frame, f"R-EAR: {right_ear:.3f}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # Face bounding box.
                x, y, bw, bh = result.face_bbox
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

                # ── Phase 2: normalization + gaze ─────────────────────────
                norm_result = normalizer.normalize(frame_rgb, result)

                if norm_result is not None:
                    # Prepare face patch thumbnail for display.
                    norm_patch_display = face_patch_to_display(
                        norm_result.face_patch, gaze_cfg.mean, gaze_cfg.std)

                    if estimator is not None:
                        pitch_n, yaw_n = estimator.predict(norm_result.face_patch)
                        pitch_cam, yaw_cam = unnormalize_gaze(
                            pitch_n, yaw_n, norm_result.rotation_matrix)

                        # Gaze direction arrow (yellow).
                        draw_gaze_arrow(frame, result.face_center_px,
                                        pitch_cam, yaw_cam)

                        # Overlay gaze angles as text.
                        cv2.putText(frame,
                                    f"Gaze  P:{np.degrees(pitch_cam):+.1f}  Y:{np.degrees(yaw_cam):+.1f}",
                                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 1)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # FPS counter.
            frame_count += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_timer = time.monotonic()
            cv2.putText(frame, f"FPS: {fps_display:.1f}",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Eyecon Phase 2 — Gaze Estimation", frame)

            # Show normalized face patch in a separate window.
            if norm_patch_display is not None:
                # Scale up for visibility.
                disp = cv2.resize(norm_patch_display, (336, 336),
                                  interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Normalized Face Patch", disp)
            else:
                blank = np.zeros((336, 336, 3), dtype=np.uint8)
                cv2.putText(blank, "No patch", (100, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
                cv2.imshow("Normalized Face Patch", blank)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        grabber.release()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
