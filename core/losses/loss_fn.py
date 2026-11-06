import torch
import torch.nn as nn
from core.losses.focal_loss import modified_focal_loss, focal_loss
from core.losses.l1_loss import l1_loss


class LossFunction(nn.Module):
    """Multi-task detection loss combining focal classification loss and masked L1 regression."""
    def __init__(self, cls_encoding: str = "gaussian", device: str = "cpu"):
        super(LossFunction, self).__init__()
        self.cls_encoding = cls_encoding
        self.device = device

    def forward(self, pred: dict, target: dict) -> dict:
        # Classification loss
        if self.cls_encoding == "binary":
            cls_loss = focal_loss(pred["cls"], target["cls"])
        else:
            cls_loss = modified_focal_loss(pred["cls"], target["cls"])

        # Bounding box regression losses
        offset_loss = l1_loss(pred["offset"], target["offset"], target["reg_mask"])
        size_loss = l1_loss(pred["size"], target["size"], target["reg_mask"])
        yaw_loss = l1_loss(pred["yaw"], target["yaw"], target["reg_mask"])

        total_loss = cls_loss + offset_loss + size_loss + yaw_loss

        return {
            "loss": total_loss,
            "cls": cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            "offset": offset_loss.item(),
            "size": size_loss.item(),
            "yaw": yaw_loss.item(),
        }
