"""Central configuration for Eyecon. All tunable parameters in one place."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class CameraConfig:
    """Webcam parameters."""
    width: int = 1280
    height: int = 720
    fps: int = 30
    device_id: int = 0

    @property
    def focal_length(self) -> float:
        """Estimated focal length for ~60° HFOV webcam."""
        return self.width / (2 * np.tan(np.radians(30)))

    @property
    def camera_matrix(self) -> np.ndarray:
        fx = fy = self.focal_length
        cx, cy = self.width / 2, self.height / 2
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0,  0,  1]], dtype=np.float64)

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.zeros((4, 1), dtype=np.float64)


@dataclass
class GazeModelConfig:
    """Gaze estimation backbone parameters."""
    model_path: Path = Path("models/gaze_resnet50.pth")
    input_size: int = 224
    normalize_distance: float = 600.0  # mm, virtual camera distance
    focal_length_norm: float = 960.0   # focal length of normalized virtual camera (ETH-XGaze default)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class CalibrationConfig:
    """Per-user calibration parameters."""
    grid_rows: int = 3
    grid_cols: int = 3
    samples_per_point: int = 30
    settle_frames: int = 15
    dot_display_ms: int = 2000
    margin_fraction: float = 0.08

    mlp_input_dim: int = 5
    mlp_hidden_dim: int = 64
    mlp_output_dim: int = 2
    mlp_epochs: int = 500
    mlp_lr: float = 0.01
    mlp_weight_decay: float = 1e-4

    save_dir: Path = Path("calibration_data")


@dataclass
class SmoothingConfig:
    """One-Euro filter parameters."""
    min_cutoff: float = 1.5
    beta: float = 0.5
    d_cutoff: float = 1.0


@dataclass
class WinkConfig:
    """Wink/blink detection parameters."""
    ear_baseline_frames: int = 60
    ear_close_ratio: float = 0.68
    ear_asymmetry_threshold: float = 0.4

    min_wink_frames: int = 4
    max_wink_frames: int = 12
    refractory_frames: int = 15
    blink_sync_tolerance: int = 2

    RIGHT_EYE_EAR_IDX: tuple = (33, 159, 158, 133, 153, 145)
    LEFT_EYE_EAR_IDX: tuple = (362, 386, 385, 263, 374, 380)


@dataclass
class ScreenConfig:
    """Display configuration — auto-detected at startup."""
    width: int = 2560
    height: int = 1600

    @classmethod
    def from_system(cls):
        """Detect primary monitor resolution."""
        try:
            from screeninfo import get_monitors
            m = get_monitors()[0]
            return cls(width=m.width, height=m.height)
        except Exception:
            return cls()


@dataclass
class EyeconConfig:
    """Top-level configuration combining all sub-configs."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    gaze: GazeModelConfig = field(default_factory=GazeModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    wink: WinkConfig = field(default_factory=WinkConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)

    debug: bool = False
