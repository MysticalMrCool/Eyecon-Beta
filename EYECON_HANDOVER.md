# Eyecon — Complete Implementation Handover

## What this document is

This is a **complete architecture specification and implementation guide** for **Eyecon**, a webcam-based eye-tracking cursor controller for Windows. An expert machine learning engineer designed this architecture after thorough research into the state of the art in appearance-based gaze estimation.

**Your job is to implement this system exactly as specified.** Every stage, every coordinate transform, every threshold, and every module is defined below. Do not simplify the pipeline or skip stages — each one exists for a critical reason.

---

## Project overview

Eyecon lets a user control their mouse cursor by looking at their screen, using only a standard webcam. Left wink = left click, right wink = right click. No special hardware required.

**Target platform:** Windows 10/11, Python 3.10+, PyTorch, OpenCV, ONNX Runtime
**Hardware assumptions:** 2560×1600 display, standard 720p webcam (~60° horizontal FOV), consumer laptop CPU (GPU optional)
**Performance target:** 30+ fps real-time, <25ms total pipeline latency per frame
**Accuracy target:** ~1.5–2.5cm screen error after 9-point calibration

---

## Project structure

```
eyecon/
├── main.py                  # Entry point — arg parsing, mode selection
├── pipeline.py              # Main processing loop (capture → gaze → cursor)
├── config.py                # All constants, thresholds, paths
├── capture.py               # Threaded webcam capture (producer)
├── landmarks.py             # MediaPipe Face Mesh wrapper + head pose (R, t)
├── normalization.py         # Data normalization: warp face to canonical pose
├── gaze_model.py            # ONNX Runtime inference wrapper for ResNet-18
├── calibration.py           # Calibration UI + MLP/SVR training + persistence
├── smoothing.py             # One-Euro adaptive filter
├── wink.py                  # Per-eye EAR wink/blink state machine
├── cursor.py                # Win32 cursor control (SetCursorPos, mouse_event)
├── utils.py                 # Shared math utilities
├── models/                  # Directory for model weights
│   └── .gitkeep
├── calibration_data/        # Saved per-user calibration profiles
│   └── .gitkeep
├── scripts/
│   ├── download_model.py    # Script to download + export pretrained weights
│   └── test_pipeline.py     # Diagnostic visualization of each pipeline stage
├── requirements.txt
└── README.md
```

---

## Dependencies (requirements.txt)

```
opencv-python>=4.8.0
mediapipe>=0.10.9
torch>=2.0.0
torchvision>=0.15.0
onnxruntime>=1.16.0
numpy>=1.24.0
scipy>=1.10.0
pynput>=1.7.6
screeninfo>=0.8.1
```

**Note on ONNX Runtime:** Install `onnxruntime-gpu` instead of `onnxruntime` if the user has an NVIDIA GPU and wants GPU acceleration. The code must handle both gracefully — try GPU provider first, fall back to CPU.

---

## Module-by-module specification

---

### config.py — Central configuration

All magic numbers live here. No magic numbers anywhere else in the codebase.

```python
"""Central configuration for Eyecon. All tunable parameters in one place."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

@dataclass
class CameraConfig:
    """Webcam parameters."""
    width: int = 1280          # Capture width (request from webcam)
    height: int = 720          # Capture height
    fps: int = 30              # Target frame rate
    device_id: int = 0         # OpenCV camera index

    # Estimated intrinsics for ~60° HFOV webcam at 1280x720
    # fx ≈ width / (2 * tan(FOV/2)) ≈ 1280 / (2 * tan(30°)) ≈ 1108
    # If the webcam is 720x480 effective, recalculate accordingly
    @property
    def focal_length(self) -> float:
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
        return np.zeros((4, 1), dtype=np.float64)  # Assume no distortion


@dataclass
class GazeModelConfig:
    """Gaze estimation backbone parameters."""
    model_path: Path = Path("models/gaze_resnet18.onnx")
    input_size: int = 224          # Normalized face patch size (224x224)
    normalize_distance: float = 600.0  # mm, virtual camera distance for normalization
    # ImageNet normalization (used by ETH-XGaze pretrained models)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class CalibrationConfig:
    """Per-user calibration parameters."""
    grid_rows: int = 3             # 3x3 = 9 calibration points
    grid_cols: int = 3
    samples_per_point: int = 30    # Frames to collect per calibration point
    settle_frames: int = 15        # Frames to discard while user's gaze settles
    dot_display_ms: int = 2000     # How long each dot is shown (ms)
    margin_fraction: float = 0.08  # Screen margin for calibration grid (8% from edges)

    # MLP calibration head architecture
    mlp_input_dim: int = 5         # [pitch, yaw, face_cx, face_cy, face_scale]
    mlp_hidden_dim: int = 64
    mlp_output_dim: int = 2        # [screen_x, screen_y]
    mlp_epochs: int = 500
    mlp_lr: float = 0.01
    mlp_weight_decay: float = 1e-4

    save_dir: Path = Path("calibration_data")


@dataclass
class SmoothingConfig:
    """One-Euro filter parameters."""
    min_cutoff: float = 1.5     # Low cutoff = more smoothing at rest
    beta: float = 0.5           # High beta = less lag during fast movement
    d_cutoff: float = 1.0       # Cutoff for derivative computation


@dataclass
class WinkConfig:
    """Wink/blink detection parameters."""
    # EAR thresholds (will be calibrated per-user relative to baseline)
    ear_baseline_frames: int = 60       # Frames to compute baseline EAR
    ear_close_ratio: float = 0.68       # Eye considered closed when EAR < baseline * this
    ear_asymmetry_threshold: float = 0.4 # |left - right| / max(left, right) > this = wink

    # Temporal constraints (in frames at 30fps)
    min_wink_frames: int = 4           # ~133ms minimum wink duration
    max_wink_frames: int = 12          # ~400ms maximum wink duration
    refractory_frames: int = 15        # ~500ms cooldown between winks
    blink_sync_tolerance: int = 2      # Frames within which both eyes closing = blink

    # MediaPipe landmark indices for EAR calculation
    # Right eye (from camera's perspective — user's left eye)
    RIGHT_EYE_EAR_IDX: tuple = (33, 159, 158, 133, 153, 145)
    # Left eye (from camera's perspective — user's right eye)
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
            return cls()  # Fall back to defaults


@dataclass
class EyeconConfig:
    """Top-level configuration combining all sub-configs."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    gaze: GazeModelConfig = field(default_factory=GazeModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    wink: WinkConfig = field(default_factory=WinkConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)

    debug: bool = False  # Show debug visualization window
```

---

### capture.py — Threaded webcam capture

```
PURPOSE: Continuously grab frames from the webcam on a background thread so
         the processing loop never blocks waiting for a frame.

CLASS: FrameGrabber
  - __init__(camera_config: CameraConfig)
  - start() -> None            # Start capture thread
  - read() -> tuple[bool, np.ndarray | None]  # Get latest frame (non-blocking)
  - stop() -> None             # Stop capture thread
  - release() -> None          # Clean up resources

IMPLEMENTATION NOTES:
  - Use threading.Thread with daemon=True
  - Store only the LATEST frame in a threading.Lock-protected variable
    (drop old frames — we always want the most recent)
  - Set cv2.VideoCapture properties: CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FPS, CAP_PROP_BUFFERSIZE=1
  - The read() method should return (False, None) if no frame is available yet
  - Flip the frame horizontally (cv2.flip(frame, 1)) so the display acts as a mirror
    — this is critical for intuitive cursor mapping (look right → cursor goes right)
```

---

### landmarks.py — MediaPipe Face Mesh + head pose

This is the foundation of the entire pipeline. It provides face landmarks AND computes the 6-DoF head pose.

```
PURPOSE: Detect 468 face landmarks using MediaPipe, extract head pose (R, t),
         and provide the raw landmark data to downstream modules.

CLASS: FaceLandmarkDetector
  - __init__(camera_config: CameraConfig)
  - process(frame_rgb: np.ndarray) -> LandmarkResult | None

DATACLASS: LandmarkResult
  - landmarks_px: np.ndarray       # (468, 2) pixel coordinates
  - landmarks_3d: np.ndarray       # (468, 3) MediaPipe's relative 3D coords
  - rotation_matrix: np.ndarray    # (3, 3) head rotation in camera coords
  - translation_vec: np.ndarray    # (3, 1) head translation in camera coords
  - face_center_px: np.ndarray     # (2,) face center in pixel coords
  - face_bbox: tuple               # (x, y, w, h) bounding box in pixels
  - left_eye_landmarks: np.ndarray # (6, 2) landmarks for left eye EAR
  - right_eye_landmarks: np.ndarray# (6, 2) landmarks for right eye EAR

HEAD POSE ESTIMATION PROCEDURE:
  1. Select 6 landmark indices for solvePnP:
     - Nose tip: 1
     - Chin: 199
     - Left eye outer corner: 33
     - Right eye outer corner: 263
     - Left mouth corner: 61
     - Right mouth corner: 291

  2. Define corresponding 3D points in a canonical face model (in mm):
     These are approximate positions in a generic face coordinate system
     where the origin is at the nose tip:
       nose_tip     = [0.0,    0.0,    0.0]
       chin         = [0.0,   -63.6, -12.5]
       left_eye     = [-43.3,  32.7, -26.0]
       right_eye    = [43.3,   32.7, -26.0]
       left_mouth   = [-28.9, -28.9, -24.1]
       right_mouth  = [28.9,  -28.9, -24.1]

  3. Call cv2.solvePnP with:
     - objectPoints: the 3D model points (6, 3)
     - imagePoints: the 2D detected landmarks at those indices (6, 2)
     - cameraMatrix: from CameraConfig
     - distCoeffs: zeros (no distortion)
     - flags: cv2.SOLVEPNP_ITERATIVE

  4. Convert the returned rotation vector (rvec) to a rotation matrix (R)
     using cv2.Rodrigues(rvec)

  5. Return R and tvec as part of LandmarkResult

FACE CENTER COMPUTATION:
  - Average the 3D positions of the 6 PnP landmarks in camera coordinates:
    face_center_3d = R @ model_center + t
    where model_center is the mean of the 6 3D model points
  - This is used in the normalization step

MEDIAPIPE CONFIGURATION:
  - mp.solutions.face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,     # Enables iris landmarks (important)
      min_detection_confidence=0.7,
      min_tracking_confidence=0.5
    )
  - Feed RGB images (not BGR — convert before passing)
```

---

### normalization.py — Data normalization (THE CRITICAL MODULE)

This implements the normalization procedure from "Revisiting Data Normalization for Appearance-Based Gaze Estimation" (Zhang et al., 2018). **This is the most mathematically precise module — every matrix operation must be exact or the gaze model outputs will be garbage.**

```
PURPOSE: Warp a face image into a canonical normalized space where:
         - Head roll is cancelled (face appears upright)
         - Face is at a fixed virtual distance (consistent eye pixel size)
         - The head coordinate system's x-axis is horizontal

CLASS: GazeNormalizer
  - __init__(gaze_config: GazeModelConfig, camera_config: CameraConfig)
  - normalize(frame: np.ndarray, landmark_result: LandmarkResult)
      -> NormalizationResult | None

DATACLASS: NormalizationResult
  - face_patch: np.ndarray          # (224, 224, 3) normalized face image, float32, [0,1]
  - rotation_matrix: np.ndarray     # (3, 3) R_norm — the normalization rotation
  - head_rotation: np.ndarray       # (3, 3) original head R in camera coords

THE NORMALIZATION ALGORITHM (step by step):

  Given:
    - R: (3, 3) head rotation matrix in camera coords (from landmarks.py)
    - t: (3, 1) head translation in camera coords
    - K: (3, 3) camera intrinsic matrix
    - d_n: normalized distance (600 mm)
    - face_center_3d: the 3D face center in camera coords
      Compute as: fc = R @ np.mean(model_3d_points, axis=0).reshape(3,1) + t

  Step 1 — Compute the normalization rotation R_norm:

    # Distance from camera to face center
    distance = np.linalg.norm(face_center_3d)

    # Z-axis of normalized camera: points from camera to face center
    z_axis = face_center_3d.flatten() / distance

    # X-axis of head coordinate system in camera coords
    # This is the first column of the head rotation matrix R
    head_x = R[:, 0]

    # Y-axis: perpendicular to both z_axis and head_x
    y_axis = np.cross(z_axis, head_x)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # X-axis: perpendicular to y and z (ensures orthonormal basis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Normalization rotation matrix (rows are the new basis vectors)
    R_norm = np.array([x_axis, y_axis, z_axis])

  Step 2 — Compute the scaling matrix S:

    # Scale factor to move face to normalized distance
    S = np.diag([1.0, 1.0, d_n / distance])

  Step 3 — Compute the full warp transformation:

    # Normalized camera matrix (can use same focal length, centered principal point)
    K_norm = np.array([
        [gaze_config.focal_length_norm, 0, gaze_config.input_size / 2],
        [0, gaze_config.focal_length_norm, gaze_config.input_size / 2],
        [0, 0, 1]
    ])
    # where focal_length_norm ≈ input_size (224) for a reasonable FOV in normalized space

    # Actually, a simpler approach that works well in practice:
    # Use the same focal length as the original camera, just center the principal point
    # at the center of the output image.
    # K_norm = K.copy()
    # K_norm[0, 2] = input_size / 2
    # K_norm[1, 2] = input_size / 2

    # The perspective warp matrix
    # M transforms a pixel in the original image to a pixel in the normalized image
    # M = K_norm @ S @ R_norm @ K_inv
    K_inv = np.linalg.inv(K)
    W = K_norm @ S @ R_norm @ K_inv

  Step 4 — Apply the warp:

    face_patch = cv2.warpPerspective(frame, W, (input_size, input_size))

  Step 5 — Preprocess for the model:

    face_patch = face_patch.astype(np.float32) / 255.0
    # Apply ImageNet normalization
    face_patch = (face_patch - mean) / std
    # Transpose to CHW format: (3, 224, 224)
    face_patch = face_patch.transpose(2, 0, 1)

  Return NormalizationResult with face_patch, R_norm, R (original head rotation)

IMPORTANT IMPLEMENTATION NOTES:
  - The face_center_3d MUST be computed from the PnP result, not from
    MediaPipe's relative Z coordinates (which are not metrically calibrated)
  - R_norm is needed later to un-normalize the gaze prediction
  - All matrix operations must use float64 for numerical stability
  - The warp matrix W must be a 3x3 homogeneous matrix for cv2.warpPerspective
  - If solvePnP fails (face not detected clearly), return None
```

---

### gaze_model.py — ONNX Runtime gaze inference

```
PURPOSE: Run the pretrained gaze estimation ResNet-18 on a normalized face patch.
         Output is (pitch, yaw) in the NORMALIZED HEAD coordinate system.

CLASS: GazeEstimator
  - __init__(gaze_config: GazeModelConfig)
  - predict(face_patch: np.ndarray) -> tuple[float, float]  # (pitch, yaw) in radians

LOADING THE MODEL:
  - Try ONNX Runtime with CUDAExecutionProvider first, fall back to CPUExecutionProvider
  - Session options: enable graph optimization (ORT_ENABLE_ALL)
  - Set intra_op_num_threads = 4 for CPU inference

INFERENCE:
  - Input: face_patch as np.ndarray, shape (1, 3, 224, 224), float32
  - Output: np.ndarray shape (1, 2) → [pitch, yaw] in radians

CONVERTING MODEL OUTPUT TO 3D GAZE VECTOR (helper function):
  def pitchyaw_to_vector(pitch: float, yaw: float) -> np.ndarray:
      """Convert pitch/yaw angles to a 3D unit gaze direction vector.

      Convention (ETH-XGaze / data normalization standard):
        - pitch: vertical angle (positive = looking down)
        - yaw: horizontal angle (positive = looking right from subject's perspective)

      Returns: (3,) unit vector in the head coordinate system
      """
      x = -np.cos(pitch) * np.sin(yaw)
      y = -np.sin(pitch)
      z = -np.cos(pitch) * np.cos(yaw)
      return np.array([x, y, z])

UN-NORMALIZING THE GAZE (helper function):
  def unnormalize_gaze(pitch: float, yaw: float, R_norm: np.ndarray) -> tuple[float, float]:
      """Convert gaze from normalized head coords back to camera coords.

      Args:
        pitch, yaw: model output in normalized head coordinate system
        R_norm: the normalization rotation matrix from normalization.py

      Returns:
        (pitch_cam, yaw_cam): gaze direction in camera coordinate system
      """
      gaze_norm = pitchyaw_to_vector(pitch, yaw)
      # Un-rotate: multiply by inverse of normalization rotation
      gaze_cam = R_norm.T @ gaze_norm  # R_norm is orthonormal so R_inv = R^T
      # Convert back to pitch/yaw in camera coords
      pitch_cam = np.arcsin(-gaze_cam[1])
      yaw_cam = np.arctan2(-gaze_cam[0], -gaze_cam[2])
      return pitch_cam, yaw_cam
```

---

### calibration.py — Per-user calibration system

This module handles the calibration UI, data collection, model training, and persistence.

```
PURPOSE: Display calibration targets on screen, collect gaze features at each
         target, train a small MLP to map features → screen coordinates,
         and save/load calibration profiles.

CLASSES:

1. CalibrationUI
   - __init__(screen_config: ScreenConfig, calibration_config: CalibrationConfig)
   - get_calibration_points() -> list[tuple[int, int]]
       Returns the (screen_x, screen_y) positions for each calibration dot,
       arranged in a grid with margins from screen edges.
   - run_calibration(pipeline_step_fn) -> CalibrationData
       Opens a fullscreen window, displays dots one at a time,
       calls pipeline_step_fn for each frame to get gaze features,
       collects samples, returns CalibrationData.

   CALIBRATION DOT POSITIONS:
     For a 3×3 grid on a 2560×1600 screen with 8% margin:
       margin_x = 2560 * 0.08 = 205
       margin_y = 1600 * 0.08 = 128
       x_positions = [205, 1280, 2355]   # left, center, right
       y_positions = [128, 800, 1472]     # top, center, bottom
       Points are all 9 combinations, presented in a randomized order.

   CALIBRATION UI BEHAVIOR:
     - Fullscreen black window (cv2.namedWindow with WINDOW_NORMAL + setWindowProperty FULLSCREEN)
     - Display a pulsing white dot (radius ~20px) at each calibration point
     - Show a brief countdown (3, 2, 1) before data collection starts
     - During collection: dot turns green, gaze features are recorded
     - After collection: dot turns briefly blue, then moves to next point
     - Show a progress indicator (e.g., "Point 3/9")
     - Allow ESC to cancel calibration

   DATA COLLECTION PER POINT:
     - Discard first `settle_frames` frames (user's gaze is still moving to the dot)
     - Collect `samples_per_point` frames of feature vectors
     - Feature vector per frame: [pitch_cam, yaw_cam, face_cx, face_cy, face_scale]
       where:
         pitch_cam, yaw_cam = un-normalized gaze angles in camera coords
         face_cx = face_center_px.x / frame_width   (normalized 0–1)
         face_cy = face_center_px.y / frame_height   (normalized 0–1)
         face_scale = face_bbox_width / frame_width   (normalized, proxy for distance)
     - Average the collected feature vectors to get ONE robust sample per point

2. CalibrationData
   - features: np.ndarray       # (N, 5) averaged feature vectors
   - targets: np.ndarray        # (N, 2) corresponding screen coordinates in pixels
   - screen_width: int
   - screen_height: int
   - timestamp: str

3. CalibrationModel
   - __init__(calibration_config: CalibrationConfig)
   - train(data: CalibrationData) -> None
   - predict(features: np.ndarray) -> np.ndarray  # (2,) screen coords
   - save(path: Path) -> None
   - load(path: Path) -> None
   - is_calibrated: bool

   MLP ARCHITECTURE:
     - Input: 5 features
     - Hidden: 64 units, ReLU activation
     - Hidden: 32 units, ReLU activation
     - Output: 2 (screen_x, screen_y)

   TRAINING:
     - Normalize features to zero mean, unit variance (fit StandardScaler on calibration data,
       save the scaler with the model)
     - Normalize targets to [0, 1] range by dividing by screen dimensions
     - Loss: MSE
     - Optimizer: Adam, lr=0.01, weight_decay=1e-4
     - Train for 500 epochs (data is tiny, this is fast)
     - Denormalize predictions back to pixel coordinates at inference time

   PERSISTENCE:
     - Save as a dict with torch.save():
       { 'model_state_dict': ...,
         'scaler_mean': ..., 'scaler_scale': ...,
         'screen_width': ..., 'screen_height': ...,
         'timestamp': ..., 'camera_device': ... }
     - Save to calibration_data/{hash}.pt where hash is based on
       camera device ID + screen resolution
     - On startup, try to load an existing calibration profile
     - If screen resolution or camera changed, prompt recalibration

   FALLBACK (no calibration available):
     - Use a simple linear approximation:
       screen_x = screen_width/2 + yaw_cam * (screen_width / fov_horizontal)
       screen_y = screen_height/2 + pitch_cam * (screen_height / fov_vertical)
     - This gives ~quadrant-level accuracy, enough to navigate to the calibration button
```

---

### smoothing.py — One-Euro adaptive filter

```
PURPOSE: Smooth the cursor position to eliminate jitter at rest while maintaining
         responsiveness during gaze shifts (saccades).

CLASS: OneEuroFilter
  - __init__(freq: float, min_cutoff: float, beta: float, d_cutoff: float)
  - __call__(x: float, timestamp: float = None) -> float  # Filter one value
  - reset() -> None

  THE ONE-EURO FILTER ALGORITHM:
    The key idea: the cutoff frequency adapts to signal speed.
    When the signal is stable (low speed), use a low cutoff → heavy smoothing.
    When the signal is moving fast (saccade), use a high cutoff → pass through.

    def __call__(self, x, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()

        if self._last_time is None:
            # First sample
            self._x_hat = x
            self._dx_hat = 0.0
            self._last_time = timestamp
            return x

        dt = timestamp - self._last_time
        if dt <= 0:
            return self._x_hat
        self._last_time = timestamp

        # Estimate derivative
        freq = 1.0 / dt
        dx = (x - self._x_hat) * freq

        # Filter the derivative (low pass)
        alpha_d = self._alpha(self._d_cutoff, freq)
        self._dx_hat = alpha_d * dx + (1 - alpha_d) * self._dx_hat

        # Adaptive cutoff based on speed
        cutoff = self._min_cutoff + self._beta * abs(self._dx_hat)

        # Filter the signal
        alpha = self._alpha(cutoff, freq)
        self._x_hat = alpha * x + (1 - alpha) * self._x_hat

        return self._x_hat

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

CLASS: ScreenSmoother
  - __init__(smoothing_config: SmoothingConfig)
  - smooth(x: float, y: float) -> tuple[float, float]
  - reset() -> None

  Uses TWO OneEuroFilter instances: one for x, one for y.
  Clamps output to screen bounds after smoothing.
```

---

### wink.py — Wink detection state machine

```
PURPOSE: Detect deliberate single-eye winks (for clicks) while rejecting
         natural blinks and squinting. Must have near-zero false positive rate
         — a false click is worse than a missed click.

CLASS: WinkDetector
  - __init__(wink_config: WinkConfig)
  - update(left_eye_lm: np.ndarray, right_eye_lm: np.ndarray)
      -> WinkEvent | None
  - calibrate_baseline(left_ear: float, right_ear: float) -> None

DATACLASS: WinkEvent
  - eye: str                   # "left" or "right"
  - timestamp: float           # time.monotonic()
  - duration_frames: int       # How long the wink lasted

EYE ASPECT RATIO (EAR):
  def compute_ear(eye_landmarks: np.ndarray) -> float:
      """Compute Eye Aspect Ratio from 6 landmark points.

      Points are ordered: [outer_corner, upper_1, upper_2, inner_corner, lower_2, lower_1]
      Indices for MediaPipe Face Mesh:
        Right eye: [33, 159, 158, 133, 153, 145]
        Left eye:  [362, 386, 385, 263, 374, 380]

      EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
      where p0,p3 are corners (horizontal) and p1,p2,p4,p5 are vertical points.
      """
      # Vertical distances
      A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
      B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
      # Horizontal distance
      C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
      if C < 1e-6:
          return 0.0
      return (A + B) / (2.0 * C)

STATE MACHINE (per eye):
  States: OPEN, CLOSING, CLOSED

  Transitions:
    OPEN → CLOSING:     when EAR < threshold for this frame
    CLOSING → CLOSED:   when EAR < threshold for min_wink_frames consecutive frames
    CLOSING → OPEN:     when EAR >= threshold before min_wink_frames reached (noise)
    CLOSED → OPEN:      when EAR >= threshold (eye reopened)

WINK DETECTION LOGIC (called every frame):
  1. Compute EAR_left, EAR_right from current landmarks
  2. Compute thresholds: thresh_left = baseline_left * ear_close_ratio
                         thresh_right = baseline_right * ear_close_ratio
  3. Update state machine for each eye
  4. Check for BLINK (both eyes closing within blink_sync_tolerance frames):
     - If both eyes enter CLOSING/CLOSED within 2 frames of each other → BLINK, ignore
  5. Check for WINK:
     - If one eye is in CLOSED state AND the other eye has been continuously
       OPEN for the entire duration → potential WINK
     - When the closed eye reopens (CLOSED → OPEN):
       duration = frames spent in CLOSING + CLOSED
       if min_wink_frames <= duration <= max_wink_frames AND not in refractory:
           emit WinkEvent(eye=which_eye, ...)
           start refractory timer
  6. Update baseline EAR using exponential moving average during OPEN periods
     (baseline slowly adapts to lighting/position changes)

BASELINE CALIBRATION:
  - During first ear_baseline_frames (60 frames = 2 seconds), collect
    EAR values for both eyes while user has eyes open
  - baseline_left = median(collected_left_ears)
  - baseline_right = median(collected_right_ears)
  - Use median, not mean, to be robust to any accidental blinks during baseline
  - This can also run during the calibration routine (user is looking at dots with eyes open)
```

---

### cursor.py — Win32 cursor control

```
PURPOSE: Move the mouse cursor and generate click events on Windows.

CLASS: CursorController
  - __init__(screen_config: ScreenConfig)
  - move(x: float, y: float) -> None       # Move cursor to (x, y) screen pixels
  - left_click() -> None
  - right_click() -> None
  - set_enabled(enabled: bool) -> None      # Pause/resume cursor control

IMPLEMENTATION:
  - Use ctypes to call Win32 API directly (faster than pynput for cursor movement):
      import ctypes
      user32 = ctypes.windll.user32

      def move(self, x: float, y: float):
          # Clamp to screen bounds
          ix = max(0, min(int(x), self.screen.width - 1))
          iy = max(0, min(int(y), self.screen.height - 1))
          user32.SetCursorPos(ix, iy)

      def left_click(self):
          MOUSEEVENTF_LEFTDOWN = 0x0002
          MOUSEEVENTF_LEFTUP = 0x0004
          user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
          user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

      def right_click(self):
          MOUSEEVENTF_RIGHTDOWN = 0x0008
          MOUSEEVENTF_RIGHTUP = 0x0010
          user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
          user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

  - The enabled flag allows the user to pause gaze control
    (e.g., with a keyboard shortcut) to use the regular mouse
  - All coordinates are in Windows screen pixels (not DPI-scaled logical pixels)
    — use user32.SetProcessDPIAware() at startup to get physical pixels

KEYBOARD SHORTCUT LISTENER (in a separate thread):
  - Use pynput.keyboard.Listener to detect:
    - F9: Toggle gaze cursor on/off
    - F10: Trigger recalibration
    - ESC: Exit Eyecon entirely
```

---

### utils.py — Shared math utilities

```
PURPOSE: Math helpers used across multiple modules.

FUNCTIONS:

def pitchyaw_to_vector(pitch: float, yaw: float) -> np.ndarray:
    """Convert (pitch, yaw) angles to a 3D unit direction vector.
    Convention: pitch+ = down, yaw+ = right (subject's perspective)"""
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
    model_points: np.ndarray
) -> np.ndarray:
    """Compute 3D face center in camera coords from PnP result."""
    center_model = np.mean(model_points, axis=0).reshape(3, 1)
    center_cam = R @ center_model + t.reshape(3, 1)
    return center_cam

def normalize_screen_coords(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    """Normalize pixel coordinates to [0, 1] range."""
    return x / w, y / h
```

---

### pipeline.py — Main processing loop

```
PURPOSE: Tie all modules together into the real-time processing loop.

CLASS: EyeconPipeline
  - __init__(config: EyeconConfig)
  - start() -> None              # Enter the main loop
  - stop() -> None               # Clean shutdown
  - run_calibration() -> None    # Trigger calibration procedure
  - get_features(frame) -> np.ndarray | None   # Used by calibration to get feature vector

THE MAIN LOOP (runs in start()):

  while running:
      1. frame = capture.read()
         if frame is None: continue

      2. result = landmarks.process(frame_rgb)
         if result is None: continue   # No face detected

      3. # Wink detection (runs on raw landmarks, parallel to gaze)
         wink_event = wink_detector.update(
             result.left_eye_landmarks,
             result.right_eye_landmarks
         )
         if wink_event and cursor.enabled:
             if wink_event.eye == "left":
                 cursor.left_click()
             else:
                 cursor.right_click()

      4. # Data normalization
         norm_result = normalizer.normalize(frame, result)
         if norm_result is None: continue

      5. # Gaze estimation
         pitch_norm, yaw_norm = gaze_model.predict(norm_result.face_patch)

      6. # Un-normalize to camera coordinates
         pitch_cam, yaw_cam = unnormalize_gaze(
             pitch_norm, yaw_norm, norm_result.rotation_matrix
         )

      7. # Assemble feature vector
         face_cx = result.face_center_px[0] / config.camera.width
         face_cy = result.face_center_px[1] / config.camera.height
         face_scale = result.face_bbox[2] / config.camera.width
         features = np.array([pitch_cam, yaw_cam, face_cx, face_cy, face_scale])

      8. # Map to screen coordinates
         if calibration_model.is_calibrated:
             screen_xy = calibration_model.predict(features)
         else:
             # Fallback: rough geometric projection
             screen_xy = fallback_projection(pitch_cam, yaw_cam, config)

      9. # Temporal smoothing
         smooth_x, smooth_y = smoother.smooth(screen_xy[0], screen_xy[1])

     10. # Move cursor
         if cursor.enabled:
             cursor.move(smooth_x, smooth_y)

     11. # Debug visualization (if enabled)
         if config.debug:
             draw_debug(frame, result, pitch_cam, yaw_cam, smooth_x, smooth_y)
             cv2.imshow("Eyecon Debug", frame)
             if cv2.waitKey(1) & 0xFF == 27:
                 break

  FALLBACK PROJECTION (used before calibration):
    def fallback_projection(pitch_cam, yaw_cam, config):
        """Rough geometric projection — assumes camera is centered above screen."""
        fov_h = np.radians(60)  # Approximate webcam HFOV
        fov_v = fov_h * config.screen.height / config.screen.width
        x = config.screen.width / 2 - yaw_cam * (config.screen.width / fov_h)
        y = config.screen.height / 2 + pitch_cam * (config.screen.height / fov_v)
        return np.array([x, y])

DEBUG VISUALIZATION:
  When config.debug is True, draw onto the frame:
  - Face bounding box (green rectangle)
  - Head pose axes (RGB arrows from nose, using cv2.projectPoints)
  - Gaze direction arrow (yellow, projected from face center)
  - Current screen target (text overlay: "Screen: (1234, 567)")
  - Per-eye EAR values (text overlay)
  - FPS counter
  - Wink state for each eye (OPEN / CLOSING / CLOSED)
```

---

### main.py — Entry point

```
PURPOSE: Parse arguments, initialize config, and start the pipeline.

MODES:
  - `python main.py`                    → Start with existing calibration (or fallback)
  - `python main.py --calibrate`        → Force recalibration on startup
  - `python main.py --debug`            → Show debug visualization window
  - `python main.py --camera 1`         → Use camera device index 1
  - `python main.py --no-cursor`        → Run pipeline without moving cursor (for testing)

STARTUP SEQUENCE:
  1. Parse arguments
  2. Initialize EyeconConfig (auto-detect screen resolution)
  3. Set DPI awareness: ctypes.windll.user32.SetProcessDPIAware()
  4. Check that gaze model exists at config.gaze.model_path
     - If not, print instructions to run scripts/download_model.py
  5. Initialize all pipeline modules
  6. Try to load existing calibration profile
  7. If no calibration or --calibrate flag, run calibration
  8. Start the main pipeline loop
  9. Handle KeyboardInterrupt → clean shutdown
```

---

### scripts/download_model.py — Model acquisition

```
PURPOSE: Download and export the pretrained gaze estimation model.

APPROACH:
  The pl_gaze_estimation repository (github.com/hysts/pl_gaze_estimation) provides
  pretrained weights for ResNet-18 trained on ETH-XGaze.

  Steps:
  1. Define a ResNet-18 model matching the ETH-XGaze baseline architecture:
     - Standard torchvision.models.resnet18(pretrained=False)
     - Replace the final fc layer: nn.Linear(512, 2) — outputs (pitch, yaw)

  2. Load the pretrained checkpoint (download the .pth file)

  3. Export to ONNX:
     dummy_input = torch.randn(1, 3, 224, 224)
     torch.onnx.export(
         model,
         dummy_input,
         "models/gaze_resnet18.onnx",
         input_names=["face_patch"],
         output_names=["gaze"],
         opset_version=12,
         dynamic_axes={"face_patch": {0: "batch"}, "gaze": {0: "batch"}}
     )

  4. Verify the ONNX model runs correctly with onnxruntime

  ALTERNATIVE (if ETH-XGaze weights are hard to obtain):
  Use the L2CS-Net pretrained ResNet-50 as a drop-in alternative.
  The L2CS-Net model outputs pitch and yaw separately via two FC heads,
  so the inference code needs to handle the dual-output format.
  Repo: github.com/Ahmednull/L2CS-Net — pretrained weights are directly downloadable.
  This is larger (ResNet-50) but easier to get started with.

  The model architecture difference:
  - ETH-XGaze baseline: single FC → 2 outputs (pitch, yaw) as regression
  - L2CS-Net: two FC heads, each using softmax+expectation for bin classification
    Output needs: pitch = sum(softmax(bins) * bin_centers), same for yaw

  If using L2CS-Net, update gaze_model.py predict() to handle the dual-head output format.
```

---

## Implementation order (recommended)

Build and test each stage incrementally. Do not try to build the whole pipeline at once.

### Phase 1: Foundations (get a debug window showing landmarks + head pose)
1. `config.py` — all configs
2. `capture.py` — threaded webcam capture
3. `landmarks.py` — MediaPipe face mesh + solvePnP head pose
4. Write a test script that shows the webcam with drawn landmarks and head pose axes

### Phase 2: Gaze backbone (get gaze arrows rendering)
5. `scripts/download_model.py` — get model weights
6. `normalization.py` — face patch warping (**test this thoroughly**: visualize the warped face patches, they should always appear upright and consistently sized regardless of head pose)
7. `gaze_model.py` — ONNX inference
8. `utils.py` — coordinate conversion functions
9. Add gaze direction arrows to the debug visualization

### Phase 3: Screen mapping (get a dot tracking your gaze)
10. `calibration.py` — calibration UI + MLP training
11. `smoothing.py` — One-Euro filter
12. `cursor.py` — Win32 cursor control
13. `pipeline.py` — full pipeline integration

### Phase 4: Click detection
14. `wink.py` — EAR-based wink detection
15. Integrate wink → click into pipeline

### Phase 5: Polish
16. `main.py` — entry point with arg parsing
17. Keyboard shortcuts (F9 toggle, F10 recalibrate, ESC quit)
18. Calibration profile persistence
19. README.md with setup instructions

---

## Testing checklist

After each phase, verify:

- [ ] **Phase 1**: Debug window shows face mesh landmarks overlaid on webcam feed. Head pose axes (red=X, green=Y, blue=Z) render from the nose tip and track head rotation smoothly.

- [ ] **Phase 2**: Normalized face patches appear upright regardless of head tilt/rotation. Gaze direction arrow points in the correct direction (look left → arrow points left on screen). No NaN or wildly unstable values.

- [ ] **Phase 3**: Calibration routine displays dots, collects data, trains MLP. After calibration, a visualization dot follows your gaze across the screen with reasonable accuracy. Cursor is stable during fixation (no jitter) and responsive during gaze shifts.

- [ ] **Phase 4**: Winking one eye triggers a click. Blinking both eyes does NOT trigger a click. No false clicks during normal use (talking, squinting, looking around). The refractory period prevents double-clicks.

- [ ] **Phase 5**: Full system runs from a single command. Persists calibration across restarts. Keyboard shortcuts work. Clean exit without hanging threads.

---

## Common pitfalls (avoid these)

1. **Forgetting to flip the frame horizontally.** If the webcam feed isn't mirrored, looking right moves the cursor left. This is deeply confusing. Always `cv2.flip(frame, 1)` before processing.

2. **Using MediaPipe's z-coordinate as metric depth.** MediaPipe's landmark z-values are relative to the face, NOT metrically calibrated. You must use `solvePnP` to get actual depth/translation in mm.

3. **Skipping data normalization.** Without normalization, the gaze model's output will shift wildly with head movement even when gaze direction hasn't changed. This is the #1 reason webcam gaze demos fail.

4. **Applying normalization rotation to gaze incorrectly.** The gaze must be un-normalized by multiplying with `R_norm.T` (transpose = inverse for rotation matrices). Multiplying with `R_norm` instead will double the rotation error.

5. **Not handling the face-not-detected case.** Every processing step after landmarks can return None. The pipeline must gracefully skip frames where the face isn't found, without crashing or sending the cursor to (0, 0).

6. **Using raw cursor coordinates without smoothing.** The raw gaze estimate has ~30-50px of frame-to-frame jitter. Without the One-Euro filter, the cursor vibrates and is unusable.

7. **Setting EAR thresholds as absolute values.** People have wildly different eye shapes. An EAR of 0.2 might be "wide open" for someone with narrow eyes and "half closed" for someone with large eyes. Always calibrate thresholds relative to the individual's baseline.

8. **Thread safety in the capture module.** The frame buffer MUST be protected by a lock. Without it, you'll get occasional corrupted frames (half old, half new) that cause downstream crashes.

---

## Key references

- Zhang et al. (2020) "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation" — ECCV 2020
- Zhang et al. (2018) "Revisiting Data Normalization for Appearance-Based Gaze Estimation" — ETRA 2018
- Abdelrahman et al. (2022) "L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments"
- Soukupová & Čech (2016) "Real-Time Eye Blink Detection using Facial Landmarks"
- Casiez et al. (2012) "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input"
- Zhu et al. (2025) "GazeFollower: An open-source system for deep learning-based gaze tracking with web cameras"
