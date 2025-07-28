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
        # preds has shape [batch, channels, height, width]
        # We need to reshape it to [batch, height*width, channels]
        batch_size, channels, height, width = preds.shape
        preds_reshaped = preds.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        
        # Split into circle predictions and classification scores
        # The model outputs: reg_max * 3 for circles + nc for classification
        reg_channels = channels - nc
        circles, scores = preds_reshaped.split([reg_channels, nc], dim=-1)
        
        batch_loss = 0.0
        n = 0
        for i, t in enumerate(targets):
            if len(t):
                # For now, just use a simple MSE loss on the circle predictions
                # This is a simplified loss - in practice you'd want more sophisticated matching
                pc = circles[i]  # [height*width, reg_channels]
                tc = t[:, 1:4].to(pc.device)  # [num_targets, 3] - x, y, r
                
                # Simple loss: find the best matching prediction for each target
                if len(tc) > 0:
                    # For simplicity, just use the first prediction
                    best_pred = pc[0:len(tc), :3]  # Take first 3 channels as x, y, r
                    l_box = F.mse_loss(best_pred, tc, reduction='mean')
                    
                    # Classification loss
                    cls_t = t[:, 0].long().to(pc.device)
                    if scores[i].numel() > 0:
                        l_cls = F.cross_entropy(scores[i][0:len(tc)], cls_t, reduction='mean')
                    else:
                        l_cls = 0.0
                    
                    batch_loss += self.box_weight * l_box + self.cls_weight * l_cls
                    n += 1
        
        return batch_loss / max(n, 1)
