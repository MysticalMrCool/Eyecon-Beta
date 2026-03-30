"""MediaPipe Face Landmarker wrapper + 6-DoF head pose estimation via solvePnP."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

from config import CameraConfig, WinkConfig


# Canonical 3D face model points (mm) for solvePnP – origin at nose tip.
_MODEL_POINTS_3D = np.array([
    [0.0,    0.0,    0.0],    # Nose tip        (landmark 1)
    [0.0,  -63.6,  -12.5],    # Chin            (landmark 199)
    [-43.3,  32.7,  -26.0],   # Left eye outer  (landmark 33)
    [43.3,   32.7,  -26.0],   # Right eye outer (landmark 263)
    [-28.9, -28.9,  -24.1],   # Left mouth      (landmark 61)
    [28.9,  -28.9,  -24.1],   # Right mouth     (landmark 291)
], dtype=np.float64)

# Landmark indices corresponding to the 3D model points above.
_PNP_LANDMARK_IDX = [1, 199, 33, 263, 61, 291]

# Default path to the MediaPipe FaceLandmarker .task model.
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


@dataclass
class LandmarkResult:
    landmarks_px: np.ndarray        # (478, 2) pixel coordinates
    landmarks_3d: np.ndarray        # (478, 3) MediaPipe relative 3D coords
    rotation_matrix: np.ndarray     # (3, 3) head rotation in camera coords
    translation_vec: np.ndarray     # (3, 1) head translation in camera coords
    face_center_px: np.ndarray      # (2,) face center in pixel coords
    face_bbox: tuple                # (x, y, w, h) bounding box in pixels
    left_eye_landmarks: np.ndarray  # (6, 2) landmarks for left eye EAR
    right_eye_landmarks: np.ndarray # (6, 2) landmarks for right eye EAR


class FaceLandmarkDetector:
    def __init__(
        self,
        camera_config: CameraConfig,
        wink_config: WinkConfig | None = None,
        model_path: Path | None = None,
    ):
        self._cam_cfg = camera_config
        self._wink_cfg = wink_config or WinkConfig()
        self._frame_count = 0

        mp_model = str(model_path or _DEFAULT_MODEL_PATH)
        base_options = mp.tasks.BaseOptions(model_asset_path=mp_model)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_rgb: np.ndarray) -> LandmarkResult | None:
        h, w = frame_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_count += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_count)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]

        # Extract all landmarks in pixel coords and normalised 3D.
        lm_px = np.array(
            [(lm.x * w, lm.y * h) for lm in face], dtype=np.float64
        )
        lm_3d = np.array(
            [(lm.x, lm.y, lm.z) for lm in face], dtype=np.float64
        )

        # --- Head pose via solvePnP ---
        image_points = lm_px[_PNP_LANDMARK_IDX].astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            self._cam_cfg.camera_matrix,
            self._cam_cfg.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        R, _ = cv2.Rodrigues(rvec)
        tvec = tvec.reshape(3, 1)

        # Face center in camera 3D coordinates.
        model_center = np.mean(_MODEL_POINTS_3D, axis=0).reshape(3, 1)

        # Face center projected to pixel coords.
        face_center_proj, _ = cv2.projectPoints(
            model_center.T, rvec, tvec,
            self._cam_cfg.camera_matrix, self._cam_cfg.dist_coeffs,
        )
        face_center_px = face_center_proj.reshape(2)

        # Bounding box from pixel landmarks.
        xs, ys = lm_px[:, 0], lm_px[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        face_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

        # Eye landmarks for EAR computation.
        left_eye_lm = lm_px[list(self._wink_cfg.LEFT_EYE_EAR_IDX)]
        right_eye_lm = lm_px[list(self._wink_cfg.RIGHT_EYE_EAR_IDX)]

        return LandmarkResult(
            landmarks_px=lm_px,
            landmarks_3d=lm_3d,
            rotation_matrix=R,
            translation_vec=tvec,
            face_center_px=face_center_px,
            face_bbox=face_bbox,
            left_eye_landmarks=left_eye_lm,
            right_eye_landmarks=right_eye_lm,
        )
