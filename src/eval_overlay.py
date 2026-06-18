"""Offline qualitative eval: predicted vs ground-truth trajectory on CoVLA val frames.

Loads checkpoints/best.pt, runs the policy on N val samples, and projects BOTH the
predicted waypoints (green) and the GT waypoints (yellow) onto the camera image, plus a
BEV panel. Saves a montage to assets/eval_overlay.jpg.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import timm
import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import build_model        # noqa: E402
from viz import project_ego_to_image  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "data", "covla_mini")


def scene_frame_from_img(img_rel):
    # frames/<scene>/<idx>.jpg
    parts = img_rel.split("/")
    return parts[1], int(parts[2].split(".")[0])


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(HERE, "checkpoints", "best.pt"), map_location=device)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    dcfg = timm.data.resolve_model_data_config(model.backbone)
    tf = timm.data.create_transform(**dcfg, is_training=False)

    val = json.load(open(f"{ROOT}/index_val.json"))
    # cache states per scene
    states = {}

    def get_state(scene, fi):
        if scene not in states:
            states[scene] = [json.loads(l) for l in open(f"{ROOT}/states/{scene}.jsonl") if l.strip()]
        return states[scene][fi]

    picks = np.linspace(0, len(val) - 1, 8).astype(int)
    tiles = []
    ades = []
    for i in picks:
        r = val[i]
        scene, fi = scene_frame_from_img(r["img"])
        s = get_state(scene, fi)
        img = cv2.imread(os.path.join(ROOT, r["img"]))
        h, w = img.shape[:2]
        x = tf(Image.open(os.path.join(ROOT, r["img"])).convert("RGB")).unsqueeze(0).to(device)
        speed = torch.tensor([r["speed"]], device=device)
        pred = model(x, speed)["waypoints"][0].cpu().numpy()  # (n,2)
        gt = np.array(r["waypoints"])                          # (n,2)
        ades.append(np.linalg.norm(pred - gt, axis=1).mean())

        K, T = s["intrinsic_matrix"], s["extrinsic_matrix"]
        sx, sy = w / 1928.0, h / 1208.0
        for wp, color in [(gt, (0, 255, 255)), (pred, (0, 255, 0))]:
            pts3 = np.concatenate([wp, np.zeros((len(wp), 1))], axis=1)  # z=0
            uv = project_ego_to_image(pts3, K, T)
            uv[:, 0] *= sx
            uv[:, 1] *= sy
            for p in uv.astype(int):
                cv2.circle(img, tuple(p), 5, color, -1, cv2.LINE_AA)
        cv2.putText(img, f"ADE {ades[-1]:.2f}m v{r['speed']:.0f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (480, 300)))

    grid = np.vstack([np.hstack(tiles[0:4]), np.hstack(tiles[4:8])])
    cv2.putText(grid, "yellow=GT  green=pred  (best.pt val ADE %.2fm)" % ckpt.get("val_ade", -1),
                (10, grid.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out = os.path.join(HERE, "assets", "eval_overlay.jpg")
    cv2.imwrite(out, grid)
    print(f"mean ADE over picks: {np.mean(ades):.3f} m | wrote {out}")


if __name__ == "__main__":
    main()
