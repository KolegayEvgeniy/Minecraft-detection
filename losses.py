import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss

bbox_loss = nn.SmoothL1Loss(reduction='mean')

def compute_bbox_loss(preds, targets, mask=None):
    if mask is not None:
        preds = preds[mask]
        targets = targets[mask]
    if preds.numel() == 0:
        return torch.tensor(0.0, device=preds.device)
    return bbox_loss(preds, targets)
