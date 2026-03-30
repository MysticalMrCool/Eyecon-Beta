# Eyecon — Architecture & Implementation Reference

## What this document is

This is the **living architecture reference** for **Eyecon**, a webcam-based eye-tracking cursor controller for Windows. It describes the system **as actually implemented**, not the original design spec. When the code and this document disagree, the code is authoritative.

---

## Project overview

Eyecon lets a user control their mouse cursor by looking at their screen, using only a standard webcam. Left wink = left click, right wink = right click. No special hardware required.

**Target platform:** Windows 10/11, Python 3.10+, PyTorch, OpenCV, MediaPipe
**Hardware assumptions:** 2560×1600 display, standard 720p webcam (~60° horizontal FOV), consumer laptop CPU (GPU optional)
**Performance target:** 30+ fps real-time, <25ms total pipeline latency per frame
**Accuracy target:** ~1.5–2.5cm screen error after 9-point calibration

**Environment:** Python 3.10, Windows, venv at `.venv`. Torch 2.11.0+cpu, torchvision 0.26.0+cpu.

---

## Project structure

```
eyecon/
├── main.py                  # Entry point — arg parsing, mode selection
├── pipeline.py              # Main processing loop (capture → gaze → cursor)
├── config.py                # All constants, thresholds, paths
├── capture.py               # Threaded webcam capture (producer)
├── landmarks.py             # MediaPipe FaceLandmarker wrapper + head pose (R, t)
├── normalization.py         # Data normalization: warp face to canonical pose
├── gaze_model.py            # PyTorch inference wrapper for ResNet-50 (ETH-XGaze)
├── calibration.py           # Calibration UI + polynomial mapping + persistence
├── smoothing.py             # One-Euro adaptive filter
├── wink.py                  # Per-eye EAR wink/blink state machine
├── cursor.py                # Win32 cursor control (SetCursorPos, mouse_event)
├── utils.py                 # Shared math utilities
├── models/                  # Downloaded model weights (gitignored)
│   ├── face_landmarker.task # MediaPipe FaceLandmarker model
│   └── gaze_resnet50.pth    # ETH-XGaze ResNet-50 state dict
├── calibration_data/        # Saved per-user calibration profiles
│   └── profile_*.pt
├── scripts/
│   ├── download_model.py    # Download + convert ETH-XGaze checkpoint
│   ├── test_phase1.py       # Test: landmarks + head pose
│   ├── test_phase2.py       # Test: normalization + gaze arrows
│   └── test_pipeline.py     # Test: full pipeline (Phase 3)
├── requirements.txt
└── README.md
scripts/
└── setup.py                 # One-command setup for cloners
```

---

## Dependencies (requirements.txt)

```
opencv-python>=4.8.0
mediapipe>=0.10.9
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
pynput>=1.7.6
screeninfo>=0.8.1
```

**There is no ONNX dependency.** Gaze inference uses native PyTorch (`torch.no_grad()`, `model.eval()`).

Model files are gitignored. Weights are downloaded at setup time via `scripts/setup.py` or `eyecon/scripts/download_model.py`.

---

## Module-by-module specification

---

### config.py — Central configuration (IMPLEMENTED)

All magic numbers live here. No magic numbers anywhere else in the codebase.

Key values that differ from common defaults:
- `GazeModelConfig.model_path = Path("models/gaze_resnet50.pth")` — PyTorch state dict, NOT ONNX
- `GazeModelConfig.focal_length_norm = 960.0` — must match the ETH-XGaze training value
- `GazeModelConfig.normalize_distance = 600.0` — mm, virtual camera distance

```python
@dataclass
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    device_id: int = 0

    @property
    def focal_length(self) -> float:
        return self.width / (2 * np.tan(np.radians(30)))  # ~1108 for 1280px

    @property
    def camera_matrix(self) -> np.ndarray:
        fx = fy = self.focal_length
        cx, cy = self.width / 2, self.height / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.zeros((4, 1), dtype=np.float64)

@dataclass
class GazeModelConfig:
    model_path: Path = Path("models/gaze_resnet50.pth")
    input_size: int = 224
    normalize_distance: float = 600.0
    focal_length_norm: float = 960.0   # ETH-XGaze training value
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

@dataclass
class CalibrationConfig:
    grid_rows: int = 3
    grid_cols: int = 3
    samples_per_point: int = 30
    settle_frames: int = 15
    margin_fraction: float = 0.08
    poly_degree: int = 2       # polynomial degree for gaze→screen regression
    ridge_alpha: float = 1e-3  # ridge regularisation strength
    save_dir: Path = Path("calibration_data")

@dataclass
class SmoothingConfig:
    min_cutoff: float = 1.5
    beta: float = 0.5
    d_cutoff: float = 1.0

@dataclass
class WinkConfig:
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
    width: int = 2560
    height: int = 1600
    # from_system() auto-detects via screeninfo

@dataclass
class EyeconConfig:
    camera / gaze / calibration / smoothing / wink / screen sub-configs
    debug: bool = False
```

---

### capture.py — Threaded webcam capture (IMPLEMENTED)

Daemon thread continuously grabs frames. Only the latest frame is kept (lock-protected). Frames are horizontally flipped (`cv2.flip(frame, 1)`) for mirror behavior.

- `FrameGrabber.__init__(camera_config)`, `start()`, `read()`, `stop()`, `release()`
- `read()` returns `(False, None)` if no frame is ready yet
- Sets `CAP_PROP_BUFFERSIZE=1` to minimise latency

---

### landmarks.py — MediaPipe FaceLandmarker + head pose (IMPLEMENTED)

**Uses `mp.tasks.vision.FaceLandmarker` (NOT `mp.solutions.face_mesh`).**

- `FaceLandmarkDetector.__init__(camera_config, wink_config, model_path)`
  - Creates a `FaceLandmarker` via `mp.tasks.vision.FaceLandmarkerOptions`
  - `RunningMode.VIDEO`, `num_faces=1`
  - Calls `detect_for_video(mp_image, frame_count)` — requires incrementing timestamp
- Returns **478** landmarks (not 468) because the `.task` model includes iris points

**LandmarkResult** dataclass:
- `landmarks_px: (478, 2)` pixel coordinates
- `landmarks_3d: (478, 3)` MediaPipe relative 3D (NOT metrically calibrated)
- `rotation_matrix: (3, 3)` head R from solvePnP
- `translation_vec: (3, 1)` head t from solvePnP
- `face_center_px: (2,)` projected face center in pixels
- `face_bbox: (x, y, w, h)`
- `left_eye_landmarks, right_eye_landmarks: (6, 2)` for EAR

**Head pose:** solvePnP with 6 canonical 3D model points (nose, chin, eye corners, mouth corners) at indices `[1, 199, 33, 263, 61, 291]`. Camera matrix from `CameraConfig`. `face_center_px` is obtained by `cv2.projectPoints` of the model-point centroid.

---

### normalization.py — Zhang et al. 2018 data normalization (IMPLEMENTED)

Warps the face image to a canonical normalized space: roll cancelled, fixed virtual distance, x-axis horizontal.

- `GazeNormalizer.__init__(gaze_config, camera_config)`
- `normalize(frame_rgb, landmark_result) -> NormalizationResult | None`

**K_norm** uses `focal_length_norm = 960` (must match ETH-XGaze training).

Algorithm:
1. `face_center_3d = compute_face_center_3d(R, t, _MODEL_POINTS_3D)` from `utils.py`
2. Build orthonormal basis: z = face direction, x from head R column 0, y = cross(z, head_x)
3. `R_norm` = rows [x_axis, y_axis, z_axis]
4. `S = diag(1, 1, d_n / distance)` where `d_n = 600mm`
5. `W = K_norm @ S @ R_norm @ K_inv`
6. `cv2.warpPerspective(frame, W, (224, 224))`
7. ImageNet normalise, transpose HWC → CHW

**NormalizationResult**: `face_patch (3,224,224)`, `rotation_matrix (R_norm)`, `head_rotation (R)`

---

### gaze_model.py — PyTorch ResNet-50 gaze inference (IMPLEMENTED)

**Uses native PyTorch inference, NOT ONNX Runtime.**

**GazeNetwork** (nn.Module):
- ResNet-50 backbone (`torchvision.models.resnet50`) layers packed into `nn.Sequential` (indices 0–8: conv1, bn1, relu, maxpool, layer1–4, avgpool)
- Head: `nn.Sequential(nn.Linear(2048, 2))` → outputs `(pitch, yaw)` in radians

**GazeEstimator**:
- Loads state dict from `models/gaze_resnet50.pth` via `torch.load()`
- Auto-selects CUDA or CPU
- `predict(face_patch) -> (pitch, yaw)` — normalized head coords

**`unnormalize_gaze(pitch, yaw, R_norm)`**: converts gaze from normalised head coords back to camera coords using `R_norm.T @ gaze_norm`, then `vector_to_pitchyaw()`.

---

### calibration.py — Calibration UI + polynomial gaze→screen mapping (IMPLEMENTED)

**CalibrationData**: `features (N,2)` — gaze ray-plane intersection coords in mm; `targets (N,2)` — screen pixels; screen dims, timestamp. N = grid_points × samples_per_point (all individual samples kept, not averaged).

**Gaze-to-screen mapping approach:**
The gaze estimation pipeline produces a 3D gaze direction vector in camera coordinates, and solvePnP provides the 3D face center. The gaze ray (origin=face_center_3d, direction=gaze_vector) is intersected with the z=0 (camera/screen) plane to produce a 2D intersection point `(x_int, y_int)` in millimetres. This naturally accounts for head translation (parallax).

A degree-2 polynomial ridge regression maps these intersection coordinates to screen pixels:
- Feature basis: `[1, x, y, x², xy, y²]` → 6 terms per axis, **12 total parameters**
- Ridge regression: `(Φ^T Φ + αI) β = Φ^T y`, with α=1e-3
- ~270 training samples for 12 parameters → massively overdetermined, robust
- Features are z-score centred/scaled before polynomial expansion

**CalibrationModel** (train / predict / save / load):
- `train(data)` → fits polynomial coefficients via ridge regression, returns MSE
- `predict(features)` → polynomial evaluation, returns screen pixel coords
- Saves/loads via `torch.save`/`torch.load` as `.pt` dicts with `version: 2`
- Old v1 MLP profiles are detected and rejected (returns False on load)

**CalibrationUI**:
- 3×3 grid with 8% screen margin, randomised order
- Fullscreen black window via OpenCV
- Per-point flow: 3-2-1 countdown → settle_frames (15) → collect samples_per_point (30)
- All individual samples stored (not averaged) for better regression statistics
- ESC cancels

**Profile path:** `calibration_data/profile_{md5hash}.pt` keyed on camera_id + resolution

**`_gaze_ray_intersect(pitch_cam, yaw_cam, landmark_result)`** (in pipeline.py):
Computes `face_center_3d` from solvePnP, converts `(pitch_cam, yaw_cam)` to a 3D gaze vector, intersects with z=0 plane. Returns `(2,)` array or None if gaze points away from screen.

**Fallback projection** (in pipeline.py, used before calibration):
Linear mapping from intersection mm to screen pixels assuming camera centred above 640 mm-wide screen.

**Suspected causes:**
- Only 9 training samples for a 5→64→32→2 MLP (massively underconstrained)
- Angular features (radians, range ~±0.5) and spatial features (normalised 0–1) are at very different scales; z-score normalisation on 9 points may not help
- The mapping is fundamentally nonlinear (gaze ray → screen plane intersection) but the MLP has no geometric prior
- The gaze arrow projection (`dx = -length * sin(yaw) * cos(pitch)`) works well — the same geometric relationship could likely map directly to screen coordinates more reliably than a learned MLP

**This is the primary open issue.** The fix should likely replace or heavily supplement the MLP with a geometric projection model (e.g., a PoR-based approach that intersects the gaze ray with the screen plane, with a small affine correction learned from calibration data).

---

### smoothing.py — One-Euro adaptive filter (IMPLEMENTED)

- `OneEuroFilter`: speed-adaptive low-pass filter (Casiez et al., 2012)
  - Low speed → heavy smoothing (stable fixation)
  - High speed → light smoothing (responsive saccades)
  - `__call__(x, timestamp)` → filtered value
- `ScreenSmoother`: wraps two independent OneEuroFilter instances (x, y)
  - `smooth(x, y) -> (sx, sy)`
  - `reset()` clears both filters

---

### wink.py — EAR wink/blink state machine (IMPLEMENTED)

**`compute_ear(eye_landmarks)`**: Eye Aspect Ratio from 6 points. `EAR = (|p1−p5| + |p2−p4|) / (2·|p0−p3|)`

**`_EyeTracker`**: per-eye state machine with states OPEN → CLOSING → CLOSED.
- Tracks `close_frames`, `last_close_duration`, `close_start_frame`
- `update(below_thresh, frame_num, min_wink) -> bool` (True if eye just reopened)

**`WinkDetector`**:
- Automatic baseline calibration: first 60 frames, median EAR per eye
- Thresholds: `baseline * ear_close_ratio (0.68)`
- Blink rejection: if both eyes' `close_start_frame` within `blink_sync_tolerance (2)` frames → ignored
- Wink: one eye reopens from CLOSING/CLOSED, other eye is OPEN, duration in [4, 12] frames
- Refractory period: 15 frames between winks
- Adaptive baseline: slow EMA update during OPEN periods (`alpha=0.005`)

---

### cursor.py — Win32 cursor control (IMPLEMENTED)

- `set_dpi_aware()` → `SetProcessDPIAware()` via ctypes
- `CursorController`:
  - `move(x, y)` → `SetCursorPos` (clamped to screen bounds)
  - `left_click()` / `right_click()` → `mouse_event` down+up
  - `toggle_enabled()` → thread-safe toggle
  - All coords in physical pixels (DPI-aware)

---

### utils.py — Shared math (IMPLEMENTED)

- `pitchyaw_to_vector(pitch, yaw) -> (3,)` — convention: pitch+=down, yaw+=right
- `vector_to_pitchyaw(v) -> (pitch, yaw)`
- `compute_face_center_3d(R, t, model_points) -> (3,1)` — 3D face center in camera coords
- `normalize_screen_coords(x, y, w, h) -> (nx, ny)` — normalise to [0,1]

---

### pipeline.py — Main processing loop (IMPLEMENTED)

**EyeconPipeline**: orchestrates all modules.

Constructor:
- Initialises FrameGrabber, FaceLandmarkDetector, GazeNormalizer, GazeEstimator, ScreenSmoother, WinkDetector, CursorController, CalibrationModel
- Auto-loads existing calibration profile

**`get_features(frame_bgr) -> (5,) | None`**: runs the pipeline up to the feature vector stage (used by CalibrationUI)

**`run_calibration() -> bool`**: opens CalibrationUI, trains MLP, saves profile

**`start()`** main loop:
1. `grabber.read()` → BGR frame
2. `detector.process(frame_rgb)` → LandmarkResult
3. `wink.update(left_eye_lm, right_eye_lm)` → WinkEvent → click
4. `normalizer.normalize(frame_rgb, result)` → NormalizationResult
5. `estimator.predict(face_patch)` → (pitch_norm, yaw_norm)
6. `unnormalize_gaze(...)` → (pitch_cam, yaw_cam)
7. Assemble features: `[pitch_cam, yaw_cam, face_cx, face_cy, face_scale]`
8. `calibration.predict(features)` or `fallback_projection(...)`
9. `smoother.smooth(x, y)`
10. `cursor.move(sx, sy)`
11. Debug overlay (if enabled): face bbox, head pose axes, gaze arrow (yellow), screen target text, EAR values, FPS, wink states

**Gaze arrow formula** (debug overlay): `dx = -length * sin(yaw) * cos(pitch)`, `dy = -length * sin(pitch)`

---

### main.py — Entry point (IMPLEMENTED)

```
python main.py                    # Start with existing calibration
python main.py --calibrate        # Force recalibration on startup
python main.py --debug            # Show debug window
python main.py --camera 1         # Use camera device 1
python main.py --no-cursor        # Testing mode (no cursor movement)
```

Startup: parse args → `set_dpi_aware()` → auto-detect screen → check model exists → init pipeline → start key listener → calibrate if needed → `pipeline.start()`

Keyboard shortcuts via pynput: F9 toggle cursor, F10 recalibrate, ESC quit.

---

### scripts/download_model.py — Model download & conversion (IMPLEMENTED)

Downloads the official ETH-XGaze **ResNet-50** checkpoint from Google Drive (file ID: `1Ma6zJrECNTjo_mToZ5GKk7EF-0FS4nEC`) and remaps the state dict keys to match `GazeNetwork`'s `nn.Sequential` backbone indexing.

Key remapping: `gaze_network.conv1.weight → gaze_network.0.weight`, etc. (`_LAYER_MAP` dict)

Also supports:
- `--checkpoint path/to/file.pth.tar` for manually downloaded checkpoints
- `--dummy` to create a model with random weights (correct architecture) for testing

---

## Implementation status

### Phase 1: Foundations — COMPLETE ✓
- config.py, capture.py, landmarks.py, test_phase1.py
- Debug window shows landmarks + head pose axes

### Phase 2: Gaze backbone — COMPLETE ✓
- utils.py, normalization.py, gaze_model.py, download_model.py, test_phase2.py
- Gaze arrows render accurately

### Phase 3: Screen mapping — IMPLEMENTED, MAPPING BROKEN
- calibration.py, smoothing.py, wink.py, cursor.py, pipeline.py, main.py, test_pipeline.py
- All unit tests pass
- Calibration UI works, MLP trains, profile saved/loaded
- **Gaze-to-screen mapping does not produce usable cursor positions** (see known bug above)

---

## Testing checklist

- [x] **Phase 1**: Debug window shows landmarks overlaid on webcam. Head pose axes (RGB) from nose, smooth tracking.
- [x] **Phase 2**: Normalized face patches upright regardless of head tilt. Gaze arrow points correctly.
- [ ] **Phase 3**: Calibration dot tracks gaze across screen with reasonable accuracy. **BLOCKED — mapping broken.**
- [x] **Phase 4**: Wink detection works (integrated into Phase 3 pipeline). Blinks rejected, refractory period works.
- [x] **Phase 5**: Single-command startup, calibration persistence, keyboard shortcuts, clean exit.

---

## Common pitfalls

1. **Frame flip.** `cv2.flip(frame, 1)` in capture.py — without this, look-right → cursor-left.
2. **MediaPipe z is not metric.** Use solvePnP for real 3D translation.
3. **Data normalization is mandatory.** Without it, gaze shifts with head movement.
4. **Unnormalize with R_norm.T**, not R_norm. They're transposes.
5. **Handle None everywhere.** Every step after landmarks can fail.
6. **Smooth cursor output.** Raw gaze has ~30-50px jitter per frame.
7. **EAR thresholds are relative.** Calibrate per-user as ratio of baseline.
8. **Thread-safe capture.** Lock-protected frame buffer prevents corruption.

---

## Key references

- Zhang et al. (2020) "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation" — ECCV 2020
- Zhang et al. (2018) "Revisiting Data Normalization for Appearance-Based Gaze Estimation" — ETRA 2018
- Abdelrahman et al. (2022) "L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments"
- Soukupová & Čech (2016) "Real-Time Eye Blink Detection using Facial Landmarks"
- Casiez et al. (2012) "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input"
- Zhu et al. (2025) "GazeFollower: An open-source system for deep learning-based gaze tracking with web cameras"

