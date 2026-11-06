import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.one_hot import one_hot


def modified_focal_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Modified focal loss for Gaussian heatmap regression (CenterNet / CornerNet style).

    Args:
        pred: Predicted class logits of shape (B, C, H, W).
        gt: Ground truth target heatmap with values in [0, 1].

    Returns:
        torch.Tensor: Scalar loss.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Heatmap/logit shape mismatch: {pred.shape} vs {gt.shape}")

    logits = pred.float()
    pred_prob = logits.sigmoid()

    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    pos_pred = pred_prob[pos_inds]
    neg_pred = pred_prob[neg_inds]

    pos_loss = F.logsigmoid(logits[pos_inds]) * torch.pow(1 - pos_pred, 2)
    neg_loss = F.logsigmoid(-logits[neg_inds]) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
    eps: float = 1e-8,
) -> torch.Tensor:
    """Standard Focal Loss for dense binary/multi-class classification.

    Args:
        input: Logits of shape (N, C, H, W).
        target: Ground truth indices or masks.
        alpha: Weighting factor alpha in [0, 1].
        gamma: Focusing parameter gamma >= 0.
        reduction: 'none', 'mean', or 'sum'.
        eps: Small epsilon for numerical stability.

    Returns:
        torch.Tensor: Computed loss.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect (N, C, *). Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f"Batch size mismatch: {input.size(0)} vs {target.size(0)}")

    probs = F.softmax(input, dim=1) + eps
    target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    weight = torch.pow(1.0 - probs, gamma)
    focal = -alpha * weight * torch.log(probs)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        return loss_tmp
    elif reduction == 'mean':
        return torch.mean(loss_tmp)
    elif reduction == 'sum':
        return torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean', eps: float = 1e-8):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)
