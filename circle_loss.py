import torch
from torch import nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """Simple loss for circle detection."""

    def __init__(self, box_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight

    def forward(self, preds, targets, nc: int):
        circles, scores = preds.split([3, nc], 1)
        batch_loss = 0.0
        n = 0
        for i, t in enumerate(targets):
            if len(t):
                pc = circles[i][:, None, :].expand(-1, len(t), -1)
                tc = t[:, 1:4].to(pc.device)
                cls_t = t[:, 0].long().to(pc.device)
                l_box = F.l1_loss(pc[:, :, :3], tc, reduction='none').mean()
                l_cls = F.cross_entropy(scores[i], cls_t, reduction='mean') if scores[i].numel() else 0.0
                batch_loss += self.box_weight * l_box + self.cls_weight * l_cls
                n += 1
        return batch_loss / max(n, 1)
