"""Phase 1 test script — webcam with face landmarks and head pose axes."""
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure the eyecon package root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CameraConfig, EyeconConfig
from capture import FrameGrabber
from landmarks import FaceLandmarkDetector, _MODEL_POINTS_3D, _PNP_LANDMARK_IDX


def draw_head_pose_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, length=50.0):
    """Draw RGB axes (X=red, Y=green, Z=blue) from the nose tip."""
    origin = np.float64([[0, 0, 0]])
    axes = np.float64([
        [length, 0, 0],   # X – red
        [0, length, 0],   # Y – green
        [0, 0, length],   # Z – blue
    ])

    origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, camera_matrix, dist_coeffs)
    axes_2d, _ = cv2.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)

    o = tuple(origin_2d[0].ravel().astype(int))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: red, green, blue
    labels = ["X", "Y", "Z"]
    for i in range(3):
        p = tuple(axes_2d[i].ravel().astype(int))
        cv2.arrowedLine(frame, o, p, colors[i], 2, tipLength=0.2)
        cv2.putText(frame, labels[i], p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)


def draw_landmarks(frame, landmarks_px, indices=None, color=(0, 255, 0), radius=1):
    """Draw face landmarks as small dots."""
    if indices is None:
        # Draw a sparse subset for cleaner visualization.
        indices = range(0, len(landmarks_px), 3)
    for i in indices:
        pt = tuple(landmarks_px[i].astype(int))
        cv2.circle(frame, pt, radius, color, -1)


def draw_eye_landmarks(frame, eye_lm, color=(255, 255, 0)):
    """Draw eye contour from EAR landmarks."""
    pts = eye_lm.astype(int)
    for i in range(len(pts)):
        cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), color, 1)
        cv2.circle(frame, tuple(pts[i]), 2, color, -1)


def compute_ear(eye_landmarks: np.ndarray) -> float:
    """Compute Eye Aspect Ratio from 6 landmark points."""
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def main():
    cfg = EyeconConfig()
    cam_cfg = cfg.camera

    grabber = FrameGrabber(cam_cfg)
    detector = FaceLandmarkDetector(cam_cfg, cfg.wink)

    grabber.start()
    print("Phase 1 test — press ESC or Q to quit")

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

            if result is not None:
                # Draw sparse face mesh landmarks.
                draw_landmarks(frame, result.landmarks_px)

                # Highlight the 6 PnP landmarks used for head pose.
                for idx in _PNP_LANDMARK_IDX:
                    pt = tuple(result.landmarks_px[idx].astype(int))
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)

                # Draw head pose axes.
                rvec, _ = cv2.Rodrigues(result.rotation_matrix)
                draw_head_pose_axes(
                    frame, rvec, result.translation_vec,
                    cam_cfg.camera_matrix, cam_cfg.dist_coeffs,
                )

                # Draw eye contours and EAR values.
                draw_eye_landmarks(frame, result.left_eye_landmarks, (255, 255, 0))
                draw_eye_landmarks(frame, result.right_eye_landmarks, (0, 255, 255))

                left_ear = compute_ear(result.left_eye_landmarks)
                right_ear = compute_ear(result.right_eye_landmarks)

                cv2.putText(frame, f"L-EAR: {left_ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(frame, f"R-EAR: {right_ear:.3f}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # Display face bounding box.
                x, y, bw, bh = result.face_bbox
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
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
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Eyecon Phase 1 — Landmarks + Head Pose", frame)
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
