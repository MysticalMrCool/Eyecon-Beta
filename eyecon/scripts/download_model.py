"""Download the official ETH-XGaze pretrained checkpoint and save as a .pth state dict.

Usage:
    # Auto-download official ETH-XGaze ResNet-50 weights from Google Drive:
    python scripts/download_model.py

    # Convert a manually-downloaded PyTorch checkpoint:
    python scripts/download_model.py --checkpoint path/to/epoch_24_ckpt.pth.tar

    # Create a dummy model (correct architecture, random weights) for testing:
    python scripts/download_model.py --dummy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add eyecon package root so imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PTH_OUTPUT = Path(__file__).resolve().parent.parent / "models" / "gaze_resnet50.pth"

# Google Drive file ID for the official ETH-XGaze checkpoint.
_GDRIVE_FILE_ID = "1Ma6zJrECNTjo_mToZ5GKk7EF-0FS4nEC"


# ---------------------------------------------------------------------------
# Checkpoint loading & key remapping
# ---------------------------------------------------------------------------

# Map original backbone attribute names to nn.Sequential indices.
_LAYER_MAP = {
    "conv1": "0", "bn1": "1", "relu": "2", "maxpool": "3",
    "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7",
    "avgpool": "8",
}


def _remap_checkpoint(ckpt_path: Path) -> dict:
    """Load an official ETH-XGaze checkpoint and remap keys to match GazeNetwork.

    The official .pth.tar stores a raw state_dict with keys like:
      gaze_network.conv1.weight   (backbone layers)
      gaze_fc.0.weight            (linear head)
    We remap the backbone keys because GazeNetwork wraps layers in
    nn.Sequential (indexed 0-8) instead of keeping separate attributes.
    """
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Some checkpoints wrap in 'state_dict', 'model_state_dict', etc.
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model_state", "model"):
            if key in state:
                state = state[key]
                break

    remapped = {}
    for k, v in state.items():
        # Strip DataParallel "module." prefix if present.
        if k.startswith("module."):
            k = k[len("module."):]

        if k.startswith("gaze_network."):
            suffix = k[len("gaze_network."):]         # e.g. "conv1.weight"
            top_attr = suffix.split(".")[0]            # e.g. "conv1"
            if top_attr in _LAYER_MAP:
                rest = suffix[len(top_attr):]          # e.g. ".weight"
                new_key = f"gaze_network.{_LAYER_MAP[top_attr]}{rest}"
                remapped[new_key] = v
            # else: skip unknown backbone keys (e.g. the original fc layer)
        else:
            # gaze_fc.* keys pass through unchanged.
            remapped[k] = v

    return remapped


def _verify_state_dict(state_dict: dict) -> None:
    """Verify the remapped state dict loads into GazeNetwork cleanly."""
    from gaze_model import GazeNetwork

    model = GazeNetwork()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        print("  All keys matched. Checkpoint verified successfully.")
    else:
        print("  Checkpoint loaded (with warnings above).")


# ---------------------------------------------------------------------------
# Download from Google Drive
# ---------------------------------------------------------------------------

def download_from_gdrive(file_id: str, dest: Path) -> bool:
    """Download a file from Google Drive using gdown."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        print("  'gdown' package not installed. Installing ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading from Google Drive (id={file_id}) ...")
    output = gdown.download(url, str(dest), quiet=False)
    if output is None or not dest.exists() or dest.stat().st_size < 50_000:
        print("  ERROR: download failed or file is too small.")
        dest.unlink(missing_ok=True)
        return False
    print(f"  Downloaded {dest.stat().st_size:,} bytes -> {dest}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download official ETH-XGaze model and save as a PyTorch state dict.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to an ETH-XGaze .pth.tar checkpoint to convert.")
    parser.add_argument("--output", type=Path, default=PTH_OUTPUT,
                        help="Output .pth path (default: models/gaze_resnet50.pth).")
    parser.add_argument("--dummy", action="store_true",
                        help="Save a GazeNetwork with random weights for testing.")
    args = parser.parse_args()

    if args.output.exists():
        print(f"Model already exists at {args.output}")
        resp = input("Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dummy:
        from gaze_model import GazeNetwork
        print("Saving dummy GazeNetwork (random weights) ...")
        model = GazeNetwork()
        torch.save(model.state_dict(), str(args.output))
    else:
        if args.checkpoint:
            if not args.checkpoint.exists():
                print(f"  ERROR: checkpoint not found at {args.checkpoint}")
                sys.exit(1)
            ckpt_path = args.checkpoint
        else:
            # Auto-download from Google Drive.
            ckpt_path = args.output.parent / "_tmp_ethxgaze_ckpt.pth.tar"
            print("Downloading official ETH-XGaze checkpoint from Google Drive ...")
            if not download_from_gdrive(_GDRIVE_FILE_ID, ckpt_path):
                ckpt_path.unlink(missing_ok=True)
                print()
                print("=" * 68)
                print("Auto-download failed.")
                print()
                print("Please download the checkpoint manually:")
                print("  https://drive.google.com/file/d/"
                      f"{_GDRIVE_FILE_ID}/view?usp=sharing")
                print()
                print("Then run:")
                print("  python scripts/download_model.py "
                      "--checkpoint <downloaded_file>")
                print("=" * 68)
                sys.exit(1)

        print(f"Loading & remapping checkpoint from {ckpt_path} ...")
        state_dict = _remap_checkpoint(ckpt_path)
        _verify_state_dict(state_dict)

        print(f"Saving remapped state dict to {args.output} ...")
        torch.save(state_dict, str(args.output))

        # Clean up temp download.
        if not args.checkpoint and ckpt_path.exists():
            ckpt_path.unlink(missing_ok=True)

    print(f"Done: {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
