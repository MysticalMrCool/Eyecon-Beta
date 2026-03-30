"""Phase 3 test — full pipeline diagnostic visualisation (no cursor movement).

Shows the webcam feed with all overlays (landmarks, gaze arrow, screen target dot)
plus a second window showing the predicted screen position as a moving dot.
Validates that all modules integrate correctly before enabling actual cursor control.

Usage:
    python eyecon/scripts/test_pipeline.py [--calibrate]
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure the eyecon package root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EyeconConfig, ScreenConfig
from capture import FrameGrabber
from landmarks import FaceLandmarkDetector
from normalization import GazeNormalizer
from gaze_model import GazeEstimator, unnormalize_gaze
from calibration import CalibrationModel, CalibrationUI, get_profile_path
from smoothing import ScreenSmoother
from wink import WinkDetector, compute_ear


def draw_gaze_arrow(frame, face_center_px, pitch_cam, yaw_cam, length=150.0):
    ox, oy = int(face_center_px[0]), int(face_center_px[1])
    dx = -length * np.sin(yaw_cam) * np.cos(pitch_cam)
    dy = -length * np.sin(pitch_cam)
    cv2.arrowedLine(frame, (ox, oy), (int(ox + dx), int(oy + dy)),
                    (0, 255, 255), 2, tipLength=0.18)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration before test")
    args = parser.parse_args()

    cfg = EyeconConfig()
    cfg.debug = True
    cfg.screen = ScreenConfig.from_system()
    print(f"Screen: {cfg.screen.width}x{cfg.screen.height}")

    # Resolve model path.
    pkg_dir = Path(__file__).resolve().parent.parent
    gaze_model_path = pkg_dir / cfg.gaze.model_path
    cfg.gaze.model_path = gaze_model_path

    if not gaze_model_path.exists():
        print(f"ERROR: Gaze model not found at {gaze_model_path}")
        print("Run:  python scripts/setup.py")
        sys.exit(1)

    grabber = FrameGrabber(cfg.camera)
    detector = FaceLandmarkDetector(cfg.camera, cfg.wink)
    normalizer = GazeNormalizer(cfg.gaze, cfg.camera)
    estimator = GazeEstimator(cfg.gaze)
    smoother = ScreenSmoother(cfg.smoothing)
    wink_det = WinkDetector(cfg.wink)
    cal_model = CalibrationModel(cfg.calibration)

    # Try loading existing profile.
    profile = get_profile_path(
        cfg.calibration, cfg.camera.device_id,
        cfg.screen.width, cfg.screen.height)
    if profile.exists():
        cal_model.load(profile)
        print(f"Loaded calibration: {profile}")

    # ---------- calibration if requested or missing ----------

    if args.calibrate or not cal_model.is_calibrated:
        print("Running calibration (press ESC to skip)...")
        grabber.start()

        def step_fn(frame_bgr):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = detector.process(frame_rgb)
            if result is None:
                return None
            norm = normalizer.normalize(frame_rgb, result)
            if norm is None:
                return None
            pn, yn = estimator.predict(norm.face_patch)
            pc, yc = unnormalize_gaze(pn, yn, norm.rotation_matrix)
            fcx = result.face_center_px[0] / cfg.camera.width
            fcy = result.face_center_px[1] / cfg.camera.height
            fsc = result.face_bbox[2] / cfg.camera.width
            return np.array([pc, yc, fcx, fcy, fsc], dtype=np.float32)

        # CalibrationUI creates its own grabber, so release ours first.
        grabber.release()

        ui = CalibrationUI(cfg.screen, cfg.calibration, cfg.camera)
        data = ui.run_calibration(step_fn)

        if data is not None:
            loss = cal_model.train(data)
            cal_model.save(profile)
            print(f"Calibration trained (MSE={loss:.6f}), saved to {profile}")
        else:
            print("Calibration skipped.")

        # Re-create grabber after calibration.
        grabber = FrameGrabber(cfg.camera)

    grabber.start()
    print("Phase 3 test — press ESC or Q to quit")
    print(f"  Calibration: {'active' if cal_model.is_calibrated else 'fallback mode'}")

    fps_timer = time.monotonic()
    frame_count = 0
    fps_display = 0.0

    # Mini screen-position canvas (scaled down).
    CANVAS_W, CANVAS_H = 640, 400
    sx_ratio = CANVAS_W / cfg.screen.width
    sy_ratio = CANVAS_H / cfg.screen.height

    try:
        while True:
            ret, frame = grabber.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(frame_rgb)

            screen_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

            if result is None:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # EAR + wink.
                left_ear = compute_ear(result.left_eye_landmarks)
                right_ear = compute_ear(result.right_eye_landmarks)
                wink_event = wink_det.update(
                    result.left_eye_landmarks, result.right_eye_landmarks)

                cv2.putText(frame, f"L-EAR: {left_ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"R-EAR: {right_ear:.3f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                wstate = wink_det.get_state()
                cv2.putText(frame,
                            f"Wink L:{wstate['left_state']} R:{wstate['right_state']}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if wink_event:
                    cv2.putText(frame, f"WINK: {wink_event.eye}!",
                                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)

                # Face bbox.
                bx, by, bw, bh = result.face_bbox
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)

                # Head pose axes.
                rvec, _ = cv2.Rodrigues(result.rotation_matrix)
                origin = np.float64([[0, 0, 0]])
                axes = np.float64([[50, 0, 0], [0, 50, 0], [0, 0, 50]])
                cam_m = cfg.camera.camera_matrix
                dist_c = cfg.camera.dist_coeffs
                o2d, _ = cv2.projectPoints(
                    origin, rvec, result.translation_vec, cam_m, dist_c)
                a2d, _ = cv2.projectPoints(
                    axes, rvec, result.translation_vec, cam_m, dist_c)
                o = tuple(o2d[0].ravel().astype(int))
                for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    p = tuple(a2d[i].ravel().astype(int))
                    cv2.arrowedLine(frame, o, p, c, 2, tipLength=0.2)

                # Normalization + gaze.
                norm_result = normalizer.normalize(frame_rgb, result)
                if norm_result is not None:
                    pn, yn = estimator.predict(norm_result.face_patch)
                    pc, yc = unnormalize_gaze(pn, yn, norm_result.rotation_matrix)

                    draw_gaze_arrow(frame, result.face_center_px, pc, yc)
                    cv2.putText(frame,
                                f"Gaze P:{np.degrees(pc):+.1f} Y:{np.degrees(yc):+.1f}",
                                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)

                    # Feature vector → screen coords.
                    fcx = result.face_center_px[0] / cfg.camera.width
                    fcy = result.face_center_px[1] / cfg.camera.height
                    fsc = result.face_bbox[2] / cfg.camera.width
                    features = np.array([pc, yc, fcx, fcy, fsc], dtype=np.float32)

                    if cal_model.is_calibrated:
                        screen_xy = cal_model.predict(features)
                    else:
                        from pipeline import fallback_projection
                        screen_xy = fallback_projection(pc, yc, cfg)

                    sx, sy = smoother.smooth(float(screen_xy[0]),
                                             float(screen_xy[1]))

                    cv2.putText(frame, f"Screen: ({int(sx)}, {int(sy)})",
                                (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

                    # Draw on mini-canvas.
                    cx = int(sx * sx_ratio)
                    cy = int(sy * sy_ratio)
                    cx = max(0, min(cx, CANVAS_W - 1))
                    cy = max(0, min(cy, CANVAS_H - 1))
                    cv2.circle(screen_canvas, (cx, cy), 8, (0, 255, 0), -1)
                    cv2.circle(screen_canvas, (cx, cy), 12, (0, 255, 0), 1)

                    # Draw crosshair at screen center.
                    scx = CANVAS_W // 2
                    scy = CANVAS_H // 2
                    cv2.line(screen_canvas, (scx - 10, scy), (scx + 10, scy),
                             (60, 60, 60), 1)
                    cv2.line(screen_canvas, (scx, scy - 10), (scx, scy + 10),
                             (60, 60, 60), 1)

            # FPS.
            frame_count += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_timer = time.monotonic()

            cv2.putText(frame, f"FPS: {fps_display:.1f}",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cal_str = "calibrated" if cal_model.is_calibrated else "fallback"
            cv2.putText(frame, f"[{cal_str}]",
                        (frame.shape[1] - 140, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 150), 1)

            cv2.imshow("Eyecon Phase 3 — Pipeline Test", frame)
            cv2.imshow("Screen Position", screen_canvas)

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
