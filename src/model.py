"""Compact image->trajectory student policy for the CoVLA E2E demo.

Design goals:
  * real-time (>20Hz) on an RTX 5070 Ti so it can drive CARLA closed-loop
  * robust visual features (DINOv2 option) to survive the real(CoVLA)->sim(CARLA) gap
  * predicts a short future trajectory (waypoints in the ego BEV frame), which a
    downstream pure-pursuit / PID controller turns into steer+throttle.

Optionally exposes a feature vector for Phase-2 knowledge distillation from Alpamayo.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class TrajectoryPolicy(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet34",
        n_waypoints: int = 10,
        use_speed: bool = True,
        pretrained: bool = True,
        feat_dim: int = 256,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.use_speed = use_speed

        # timm backbone as a pooled feature extractor (num_classes=0 -> global pooled vec)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        b_dim = self.backbone.num_features
        # input transform config (mean/std/size) travels with the model -> see data_config()
        self.data_cfg = timm.data.resolve_model_data_config(self.backbone)

        extra = 1 if use_speed else 0
        self.neck = nn.Sequential(
            nn.Linear(b_dim + extra, feat_dim),
            nn.GELU(),
            nn.LayerNorm(feat_dim),
        )
        # student feature used for distillation against the teacher's trajectory/embedding
        self.feat_dim = feat_dim
        # predict dx,dy per waypoint in ego frame (x fwd, y left), meters
        self.head = nn.Linear(feat_dim, n_waypoints * 2)

    def forward(self, image: torch.Tensor, speed: torch.Tensor | None = None):
        f = self.backbone(image)  # (B, b_dim)
        if self.use_speed:
            if speed is None:
                speed = torch.zeros(f.size(0), 1, device=f.device, dtype=f.dtype)
            f = torch.cat([f, speed.view(-1, 1)], dim=1)
        feat = self.neck(f)
        wp = self.head(feat).view(-1, self.n_waypoints, 2)
        return {"waypoints": wp, "feat": feat}


def build_model(cfg: dict) -> TrajectoryPolicy:
    return TrajectoryPolicy(
        backbone=cfg.get("backbone", "resnet34"),
        n_waypoints=cfg.get("n_waypoints", 10),
        use_speed=cfg.get("use_speed", True),
        pretrained=cfg.get("pretrained", True),
        feat_dim=cfg.get("feat_dim", 256),
    )


if __name__ == "__main__":
    # smoke test on GPU/CPU
    m = build_model({"backbone": "resnet34", "n_waypoints": 10})
    x = torch.randn(2, 3, 224, 224)
    out = m(x, speed=torch.tensor([3.0, 8.0]))
    print("waypoints:", out["waypoints"].shape, "feat:", out["feat"].shape)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"params: {n_params:.1f}M | data_cfg: {m.data_cfg}")
