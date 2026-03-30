"""Data normalization — warp face to canonical pose for gaze estimation.

Implements the normalization procedure from
"Revisiting Data Normalization for Appearance-Based Gaze Estimation"
(Zhang et al., 2018).
"""
from dataclasses import dataclass
import cv2
import numpy as np

from config import CameraConfig, GazeModelConfig
from landmarks import LandmarkResult, _MODEL_POINTS_3D
from utils import compute_face_center_3d


@dataclass
class NormalizationResult:
    face_patch: np.ndarray        # (3, 224, 224) normalized face image, float32
    rotation_matrix: np.ndarray   # (3, 3) R_norm — the normalization rotation
    head_rotation: np.ndarray     # (3, 3) original head R in camera coords


class GazeNormalizer:
    def __init__(self, gaze_config: GazeModelConfig, camera_config: CameraConfig):
        self._gaze_cfg = gaze_config
        self._cam_cfg = camera_config

        # Normalized camera intrinsic matrix.
        # focal_length_norm must match the value used during ETH-XGaze training (960).
        s = gaze_config.input_size
        f_norm = gaze_config.focal_length_norm
        self._K_norm = np.array([
            [f_norm, 0,      s / 2.0],
            [0,      f_norm, s / 2.0],
            [0,      0,      1.0],
        ], dtype=np.float64)

    def normalize(
        self,
        frame: np.ndarray,
        landmark_result: LandmarkResult,
    ) -> NormalizationResult | None:
        """Warp the face image into a canonical normalized space.

        Returns None if the normalization cannot be computed.
        """
        R = landmark_result.rotation_matrix
        t = landmark_result.translation_vec
        K = self._cam_cfg.camera_matrix
        d_n = self._gaze_cfg.normalize_distance
        size = self._gaze_cfg.input_size

        # ------------------------------------------------------------------
        # Step 0 — compute 3D face center in camera coordinates (mm)
        # ------------------------------------------------------------------
        face_center_3d = compute_face_center_3d(R, t, _MODEL_POINTS_3D)
        distance = float(np.linalg.norm(face_center_3d))
        if distance < 1e-6:
            return None

        # ------------------------------------------------------------------
        # Step 1 — build the normalization rotation R_norm
        # ------------------------------------------------------------------
        # Z-axis: camera → face center
        z_axis = face_center_3d.flatten() / distance

        # X-axis of the head coordinate system in camera coords (first column of R)
        head_x = R[:, 0].copy()

        # Y-axis: perpendicular to z and head_x
        y_axis = np.cross(z_axis, head_x)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            return None
        y_axis /= y_norm

        # X-axis: perpendicular to y and z (ensures orthonormal basis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        # Rows are the new basis vectors
        R_norm = np.array([x_axis, y_axis, z_axis], dtype=np.float64)

        # ------------------------------------------------------------------
        # Step 2 — scaling matrix to move face to normalized distance
        # ------------------------------------------------------------------
        S = np.diag([1.0, 1.0, d_n / distance])

        # ------------------------------------------------------------------
        # Step 3 — the full perspective warp: M = K_norm @ S @ R_norm @ K_inv
        # ------------------------------------------------------------------
        K_inv = np.linalg.inv(K)
        W = self._K_norm @ S @ R_norm @ K_inv

        # ------------------------------------------------------------------
        # Step 4 — warp the original frame
        # ------------------------------------------------------------------
        face_patch = cv2.warpPerspective(frame, W, (size, size))

        # ------------------------------------------------------------------
        # Step 5 — preprocess for the gaze model
        # ------------------------------------------------------------------
        face_patch = face_patch.astype(np.float32) / 255.0

        mean = np.array(self._gaze_cfg.mean, dtype=np.float32)
        std = np.array(self._gaze_cfg.std, dtype=np.float32)
        face_patch = (face_patch - mean) / std

        # HWC → CHW
        face_patch = face_patch.transpose(2, 0, 1)

        return NormalizationResult(
            face_patch=face_patch,
            rotation_matrix=R_norm,
            head_rotation=R,
        )
