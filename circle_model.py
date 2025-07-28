import math
from typing import List, Tuple
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.block import DFL
from ultralytics.utils.tal import make_anchors


class CircleDetect(nn.Module):
    """YOLO detection head that predicts circle center and radius."""

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
        self.no = nc + self.reg_max * 3
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 3)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 3 * self.reg_max, 1)) for x in ch
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
        circle, cls = x_cat.split((self.reg_max * 3, self.nc), 1)
        dcircle = self.decode_bboxes(self.dfl(circle), self.anchors.unsqueeze(0)) * self.strides
        if self.export and self.format == "imx":
            return dcircle.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        return torch.cat((dcircle, cls.sigmoid()), 1)

    def decode_bboxes(self, circles: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode circle predictions relative to anchor centers."""
        x = circles[:, 0] + anchors[:, 0]
        y = circles[:, 1] + anchors[:, 1]
        r = circles[:, 2].exp()
        return torch.stack((x, y, r), 1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
        batch_size, anchors, _ = preds.shape
        circles, scores = preds.split([3, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        circles = circles.gather(dim=1, index=index.repeat(1, 1, 3))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]
        return torch.cat([circles[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


def load_yolov8_circle(weights: str = "yolov8n.pt", nc: int = 80) -> YOLO:
    """Load a pretrained YOLOv8 model and replace its head with CircleDetect."""
    model = YOLO(weights)
    head = model.model.model[-1]
    ch = [m[0].conv.in_channels for m in head.cv2]
    circle_head = CircleDetect(nc=nc, ch=ch)
    circle_head.stride = head.stride

    # Copy classification weights for any layers with matching shapes
    for src, dst in zip(head.cv3, circle_head.cv3):
        src_state = src.state_dict()
        dst_state = dst.state_dict()
        for k, v in dst_state.items():
            if k in src_state and src_state[k].shape == v.shape:
                dst_state[k] = src_state[k]
        dst.load_state_dict(dst_state, strict=False)

    model.model.model[-1] = circle_head
    return model
