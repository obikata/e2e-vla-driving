"""CoVLA trajectory dataset: reads the preprocessed index (scripts/prepare_dataset.py).

Each item:
    image:     (3,H,W) normalized for the timm backbone
    speed:     () ego speed [m/s]
    waypoints: (n_waypoints,2) future (x_fwd, y_left) in ego frame [m]
    valid:     (n_waypoints,) all ones (CoVLA labels are dense)
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CoVLATrajectory(Dataset):
    def __init__(self, root, index_file, transform, augment=False):
        self.root = root
        self.items = json.load(open(os.path.join(root, index_file)))
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        img = Image.open(os.path.join(self.root, r["img"])).convert("RGB")
        wp = torch.tensor(r["waypoints"], dtype=torch.float32)  # (n,2) x_fwd, y_left
        # mirror augmentation: flip L<->R and negate lateral target -> kills left/right bias
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            wp = wp.clone()
            wp[:, 1] = -wp[:, 1]
        x = self.transform(img)
        speed = torch.tensor(r["speed"], dtype=torch.float32)
        valid = torch.ones(wp.size(0), dtype=torch.float32)
        return {"image": x, "speed": speed, "waypoints": wp, "valid": valid}


def make_transform(backbone, train):
    m = timm.create_model(backbone, pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(m)
    return timm.data.create_transform(**cfg, is_training=train)


def build_datasets(cfg):
    root = os.path.join(HERE, cfg["data_root"])
    backbone = cfg.get("backbone", "resnet34")
    # keep geometry intact for trajectory labels: avoid flips/crops that change the path
    train_tf = make_transform(backbone, train=False)
    val_tf = make_transform(backbone, train=False)
    train = CoVLATrajectory(root, "index_train.json", train_tf, augment=True)
    val = CoVLATrajectory(root, "index_val.json", val_tf, augment=False)
    return train, val


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open(os.path.join(HERE, "configs", "train.yaml")))
    tr, va = build_datasets(cfg)
    print("train", len(tr), "val", len(va))
    b = tr[0]
    print({k: tuple(v.shape) for k, v in b.items()})
    print("waypoints[0..3]:", b["waypoints"][:4].tolist(), "speed", float(b["speed"]))
