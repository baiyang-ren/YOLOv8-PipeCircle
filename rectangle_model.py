import math
from typing import List, Tuple
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.block import DFL
from ultralytics.utils.tal import make_anchors


class RectangleDetect(nn.Module):
    """YOLO detection head that predicts rectangle bounding boxes (x, y, w, h)."""

    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False

    def __init__(self, nc: int = 80, ch: Tuple = ()):  # channels from previous layer
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # 4 for x, y, w, h
        self.stride = torch.zeros(self.nl)
        self.f = [15, 18, 21]  # Feature layer indices (same as original Detect)
        self.i = 22  # Layer index (same as original Detect)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: List[torch.Tensor]):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x: List[torch.Tensor]) -> torch.Tensor:
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        bbox, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbbox = self.decode_bboxes(self.dfl(bbox), self.anchors.unsqueeze(0)) * self.strides
        if self.export and self.format == "imx":
            return dbbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        return torch.cat((dbbox, cls.sigmoid()), 1)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rectangle predictions relative to anchor centers."""
        # bboxes shape: [batch, anchors, 4*reg_max, H*W]
        # anchors shape: [batch, anchors, 2, H*W]
        
        # Reshape bboxes to [batch, anchors, 4, reg_max, H*W]
        bboxes = bboxes.view(bboxes.shape[0], bboxes.shape[1], 4, -1, bboxes.shape[2])
        
        # Apply DFL to each coordinate
        decoded = []
        for i in range(4):
            coord = self.dfl(bboxes[:, :, i, :, :])  # [batch, anchors, H*W]
            decoded.append(coord)
        
        # Stack coordinates: [batch, anchors, 4, H*W]
        decoded = torch.stack(decoded, dim=2)
        
        # Decode relative to anchors
        x = decoded[:, :, 0] + anchors[:, :, 0]  # center x
        y = decoded[:, :, 1] + anchors[:, :, 1]  # center y
        w = decoded[:, :, 2].exp()  # width (exponential to ensure positive)
        h = decoded[:, :, 3].exp()  # height (exponential to ensure positive)
        
        return torch.stack((x, y, w, h), 2)  # [batch, anchors, 4, H*W]

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
        batch_size, anchors, _ = preds.shape
        bboxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        bboxes = bboxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]
        return torch.cat([bboxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


def load_yolov8_rectangle(weights: str = "yolov8n.pt", nc: int = 80) -> YOLO:
    """Load a pretrained YOLOv8 model and replace its head with RectangleDetect."""
    model = YOLO(weights)
    head = model.model.model[-1]
    ch = [m[0].conv.in_channels for m in head.cv2]
    rectangle_head = RectangleDetect(nc=nc, ch=ch)
    rectangle_head.stride = head.stride

    # Copy classification weights for any layers with matching shapes
    for src, dst in zip(head.cv3, rectangle_head.cv3):
        src_state = src.state_dict()
        dst_state = dst.state_dict()
        for k, v in dst_state.items():
            if k in src_state and src_state[k].shape == v.shape:
                dst_state[k] = src_state[k]
        dst.load_state_dict(dst_state, strict=False)

    # Copy regression weights where possible (adapting from 3 to 4 coordinates)
    for src, dst in zip(head.cv2, rectangle_head.cv2):
        src_state = src.state_dict()
        dst_state = dst.state_dict()
        for k, v in dst_state.items():
            if k in src_state:
                src_tensor = src_state[k]
                if k == '2.weight':  # Final conv layer
                    # Adapt from 3*reg_max to 4*reg_max
                    if src_tensor.shape[0] == 3 * 16:  # 3 coordinates * 16 (reg_max)
                        # Repeat the pattern for the 4th coordinate
                        new_weight = torch.zeros(4 * 16, src_tensor.shape[1], src_tensor.shape[2], src_tensor.shape[3])
                        new_weight[:3*16] = src_tensor  # Copy first 3 coordinates
                        new_weight[3*16:] = src_tensor[:16]  # Use first coordinate as template for 4th
                        dst_state[k] = new_weight
                    else:
                        dst_state[k] = src_tensor
                else:
                    dst_state[k] = src_tensor
        dst.load_state_dict(dst_state, strict=False)

    model.model.model[-1] = rectangle_head
    return model 