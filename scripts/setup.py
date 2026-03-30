"""One-command setup: install dependencies and download all required model files."""
import subprocess
import sys
import urllib.request
from pathlib import Path

EYECON_DIR = Path(__file__).resolve().parent.parent / "eyecon"
MODELS_DIR = EYECON_DIR / "models"

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"


def install_requirements():
    """Install Python dependencies from requirements.txt."""
    req_file = EYECON_DIR / "requirements.txt"
    print(f"Installing dependencies from {req_file} ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q",
    ])
    print("Dependencies installed.\n")


def download_face_landmarker():
    """Download the MediaPipe face landmarker model."""
    if FACE_LANDMARKER_PATH.exists():
        print(f"Face landmarker already exists: {FACE_LANDMARKER_PATH}")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading MediaPipe face landmarker ...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(FACE_LANDMARKER_PATH))
    print(f"  Saved: {FACE_LANDMARKER_PATH} ({FACE_LANDMARKER_PATH.stat().st_size:,} bytes)\n")


def download_gaze_model():
    """Download and convert the ETH-XGaze pretrained weights."""
    gaze_pth = MODELS_DIR / "gaze_resnet50.pth"
    if gaze_pth.exists():
        print(f"Gaze model already exists: {gaze_pth}")
        return
    print("Downloading ETH-XGaze gaze model (this may take a minute) ...")
    subprocess.check_call([
        sys.executable,
        str(EYECON_DIR / "scripts" / "download_model.py"),
    ])
    print()


def main():
    print("=" * 60)
    print("  Eyecon Setup")
    print("=" * 60)
    print()

    install_requirements()
    download_face_landmarker()
    download_gaze_model()

    print("=" * 60)
    print("  Setup complete!")
    print()
    print("  To test:  python eyecon/scripts/test_phase2.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
