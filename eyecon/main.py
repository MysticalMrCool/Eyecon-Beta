"""Eyecon entry point — argument parsing, mode selection, startup sequence."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the eyecon package directory is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EyeconConfig, ScreenConfig
from cursor import set_dpi_aware


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eyecon — webcam-based eye-tracking cursor controller")
    parser.add_argument("--calibrate", action="store_true",
                        help="Force recalibration on startup")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug visualisation window")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--no-cursor", action="store_true",
                        help="Run pipeline without moving the cursor (for testing)")
    args = parser.parse_args()

    set_dpi_aware()

    cfg = EyeconConfig()
    cfg.camera.device_id = args.camera
    cfg.debug = args.debug
    cfg.screen = ScreenConfig.from_system()

    print(f"Screen: {cfg.screen.width}x{cfg.screen.height}")
    print(f"Camera: device {cfg.camera.device_id}")

    # Check gaze model exists.
    pkg_dir = Path(__file__).resolve().parent
    model_path = pkg_dir / cfg.gaze.model_path
    if not model_path.exists():
        print(f"\nERROR: Gaze model not found at {model_path}")
        print("Run:  python scripts/setup.py   OR   python eyecon/scripts/download_model.py --dummy")
        sys.exit(1)

    from pipeline import EyeconPipeline

    pipeline = EyeconPipeline(cfg, move_cursor=not args.no_cursor)

    # Keyboard shortcut listener (F9 toggle, F10 recalibrate, ESC quit).
    _start_key_listener(pipeline)

    if args.calibrate or not pipeline.calibration.is_calibrated:
        print("Starting calibration...")
        pipeline.run_calibration()

    pipeline.start()


def _start_key_listener(pipeline) -> None:
    """Start a background thread listening for hotkeys."""
    try:
        from pynput import keyboard
    except ImportError:
        print("pynput not installed — keyboard shortcuts disabled.")
        return

    def on_press(key):
        try:
            if key == keyboard.Key.f9:
                state = pipeline.cursor.toggle_enabled()
                print(f"Cursor {'enabled' if state else 'disabled'}")
            elif key == keyboard.Key.f10:
                print("Recalibration requested...")
                pipeline.run_calibration()
            elif key == keyboard.Key.esc:
                pipeline.stop()
                return False  # stop listener
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()


if __name__ == "__main__":
    main()
