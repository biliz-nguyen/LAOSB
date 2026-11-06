import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked L1 Loss for bounding box regression targets."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss


def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked Smooth L1 Loss for bounding box regression targets."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(pred).float()
    loss = F.smooth_l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss
