# Eyecon

A webcam-based eye-tracking cursor controller for Windows. Look at your screen to move the cursor — left wink to click, right wink to right-click. No special hardware required.

Built with **PyTorch**, **OpenCV**, and **MediaPipe**.

## How It Works

1. **Face Detection & Landmarks** — MediaPipe Face Landmarker detects 478 face landmarks in real time
2. **Head Pose Estimation** — SolvePnP computes 3D head rotation and translation from key landmarks
3. **Data Normalization** — Zhang et al. 2018 normalization warps the face to a canonical pose/distance
4. **Gaze Estimation** — A pretrained ETH-XGaze ResNet-50 predicts gaze direction (pitch, yaw) from the normalized face patch
5. **Calibration** — 9-point screen calibration maps raw gaze to screen coordinates via an MLP *(Phase 3)*
6. **Cursor Control** — Smoothed gaze coordinates drive the Windows cursor with wink-based clicking *(Phase 3)*

## Quick Start

```bash
# Clone the repo
git clone https://github.com/<your-username>/eyecon.git
cd eyecon

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Run setup (installs dependencies + downloads model weights ~290MB)
python scripts/setup.py

# Test it — opens webcam with gaze arrows overlay
python eyecon/scripts/test_phase2.py
```

Press **ESC** or **Q** to close the test window.

## Project Structure

```
eyecon/
├── config.py              # All constants and tunable parameters
├── capture.py             # Threaded webcam capture
├── landmarks.py           # MediaPipe face landmarks + head pose (R, t)
├── normalization.py       # Zhang et al. 2018 face patch normalization
├── gaze_model.py          # PyTorch ResNet-50 gaze inference (ETH-XGaze)
├── utils.py               # Shared math utilities
├── models/                # Downloaded model weights (gitignored)
├── calibration_data/      # Per-user calibration profiles
├── scripts/
│   ├── download_model.py  # Download + convert ETH-XGaze checkpoint
│   ├── test_phase1.py     # Test: landmarks + head pose
│   └── test_phase2.py     # Test: normalization + gaze arrows
scripts/
└── setup.py               # One-command setup
```

## Tech Stack

- **PyTorch / torchvision** — ResNet-50 gaze estimation model (ETH-XGaze pretrained weights)
- **OpenCV** — Webcam capture, image processing, SolvePnP head pose
- **MediaPipe** — Real-time face landmark detection (478 points)
- **NumPy / SciPy** — Linear algebra, coordinate transforms

## Requirements

- Python 3.10+
- Windows 10/11
- Standard webcam (720p recommended)
- ~500MB disk space (model weights)

## Architecture Details

The gaze estimation pipeline follows the approach from [ETH-XGaze](https://ait.ethz.ch/xgaze) (Zhang et al., ECCV 2020):

- **Normalization**: Warps the face region to a canonical virtual camera at a fixed distance (600mm) and focal length (960px), removing head pose variation before gaze inference
- **Model**: ResNet-50 backbone with a 2-unit linear head outputting (pitch, yaw) in radians
- **Unnormalization**: Rotates the predicted gaze vector back from the normalized coordinate system to camera coordinates using the inverse of the normalization rotation

## License

MIT
