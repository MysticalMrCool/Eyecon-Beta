"""Shared math utilities used across multiple Eyecon modules."""
import numpy as np


def pitchyaw_to_vector(pitch: float, yaw: float) -> np.ndarray:
    """Convert (pitch, yaw) angles to a 3D unit direction vector.

    Convention: pitch+ = down, yaw+ = right (subject's perspective).
    """
    x = -np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = -np.cos(pitch) * np.cos(yaw)
    return np.array([x, y, z])


def vector_to_pitchyaw(v: np.ndarray) -> tuple[float, float]:
    """Convert a 3D direction vector to (pitch, yaw) angles."""
    v = v / np.linalg.norm(v)
    pitch = np.arcsin(-v[1])
    yaw = np.arctan2(-v[0], -v[2])
    return pitch, yaw


def compute_face_center_3d(
    R: np.ndarray,
    t: np.ndarray,
    model_points: np.ndarray,
) -> np.ndarray:
    """Compute 3D face center in camera coords from PnP result.

    Returns: (3, 1) column vector — face center in camera coordinate frame (mm).
    """
    center_model = np.mean(model_points, axis=0).reshape(3, 1)
    center_cam = R @ center_model + t.reshape(3, 1)
    return center_cam


def normalize_screen_coords(
    x: float, y: float, w: int, h: int
) -> tuple[float, float]:
    """Normalize pixel coordinates to [0, 1] range."""
    return x / w, y / h
