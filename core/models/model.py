import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from core.models.backbones.mobilepixor import MobilePixorBackBone
from core.models.backbones.pixor import PixorBackBone
from core.models.backbones.rpn import RPN
from core.models.backbones.pointpillars import PointPillarsBackbone
from core.models.heads.cnn import Header


class CustomModel(nn.Module):
    """3D LiDAR Bird's-Eye-View Object Detection Model."""
    BACKBONES = {
        "mobilepixor": MobilePixorBackBone,
        "pixor": PixorBackBone,
        "rpn": RPN,
        "pointpillars": PointPillarsBackbone,
    }

    def __init__(self, cfg: dict, num_classes: int = 4):
        super(CustomModel, self).__init__()
        backbone_name = cfg.get("backbone", "mobilepixor").lower()
        backbone_cls = self.BACKBONES.get(backbone_name)
        if backbone_cls is None:
            raise ValueError(f"Unknown backbone '{backbone_name}'. Available: {list(self.BACKBONES.keys())}")

        self.backbone = backbone_cls()
        self.num_classes = num_classes
        if cfg.get("cls_encoding") == "binary":
            self.num_classes += 1

        self.header = Header(self.num_classes, cfg["backbone_out_dim"])

    def forward(self, x):
        features = self.backbone(x)
        return self.header(features)


if __name__ == "__main__":
    test_cfg = {
        "backbone": "mobilepixor",
        "cls_encoding": "gaussian",
        "backbone_out_dim": 16,
    }
    model = CustomModel(test_cfg, num_classes=4)
    dummy_input = torch.randn(1, 30, 240, 240)
    pred = model(dummy_input)
    print("Inference successful:")
    for k, v in pred.items():
        print(f"  {k}: {v.shape}")
