"""PyTorch gaze inference wrapper for the pretrained ETH-XGaze ResNet-50."""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from config import GazeModelConfig
from utils import pitchyaw_to_vector, vector_to_pitchyaw


class GazeNetwork(nn.Module):
    """ETH-XGaze gaze estimation network (ResNet-50 backbone + Linear head).

    Architecture mirrors the official implementation at
    https://github.com/xucong-zhang/ETH-XGaze/blob/master/model.py
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.gaze_network = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )
        self.gaze_fc = nn.Sequential(nn.Linear(2048, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.gaze_network(x)          # (B, 2048, 1, 1)
        feat = feat.view(feat.size(0), -1)   # (B, 2048)
        gaze = self.gaze_fc(feat)            # (B, 2)
        return gaze


class GazeEstimator:
    """Load a pretrained ETH-XGaze model and run inference with PyTorch."""

    def __init__(self, gaze_config: GazeModelConfig):
        self._cfg = gaze_config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model()

    def _load_model(self) -> GazeNetwork:
        model = GazeNetwork()
        checkpoint = torch.load(
            str(self._cfg.model_path), map_location=self._device, weights_only=False,
        )
        model.load_state_dict(checkpoint, strict=True)
        model.to(self._device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, face_patch: np.ndarray) -> tuple[float, float]:
        """Run gaze estimation on a normalized face patch.

        Args:
            face_patch: (3, 224, 224) float32, already ImageNet-normalized.

        Returns:
            (pitch, yaw) in radians, in the *normalized* head coordinate system.
        """
        tensor = torch.from_numpy(face_patch[np.newaxis, ...]).to(self._device)
        gaze = self._model(tensor).cpu().numpy().flatten()
        return float(gaze[0]), float(gaze[1])


def unnormalize_gaze(
    pitch: float,
    yaw: float,
    R_norm: np.ndarray,
) -> tuple[float, float]:
    """Convert gaze from normalized head coords back to camera coords.

    Args:
        pitch, yaw: model output in normalized head coordinate system.
        R_norm: the normalization rotation matrix from normalization.py.

    Returns:
        (pitch_cam, yaw_cam) in camera coordinate system.
    """
    gaze_norm = pitchyaw_to_vector(pitch, yaw)
    # Un-rotate: R_norm is orthonormal so R_inv = R^T
    gaze_cam = R_norm.T @ gaze_norm
    return vector_to_pitchyaw(gaze_cam)
