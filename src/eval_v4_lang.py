"""Evaluate v4 (option C): the parallel language head + demonstrate its limit.

Shows two things on CoVLA val:
  1. the language head *can* read semantics (red/green/should_stop accuracy);
  2. but it is DECOUPLED from the action — among frames the model says "should_stop",
     the trajectory head still predicts a long (going) path. Saying != deciding.

Outputs assets/v4_lang_eval.jpg (montage) and prints metrics.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import timm
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model          # noqa: E402
from viz import project_ego_to_image   # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "data", "covla_mini")
LABELS = ["RED", "GREEN", "STOP"]


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(HERE, "checkpoints", "best_v4_lang.pt"), map_location=device)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    dcfg = timm.data.resolve_model_data_config(model.backbone)
    tf = timm.data.create_transform(**dcfg, is_training=False)

    val = json.load(open(f"{ROOT}/index_val.json"))
    states = {}

    def get_state(scene, fi):
        if scene not in states:
            states[scene] = [json.loads(l) for l in open(f"{ROOT}/states/{scene}.jsonl") if l.strip()]
        return states[scene][fi]

    probs, gts, traj_len = [], [], []
    rows = []
    for r in val:
        x = tf(Image.open(os.path.join(ROOT, r["img"])).convert("RGB")).unsqueeze(0).to(device)
        out = model(x, torch.tensor([r["speed"]], device=device))
        p = torch.sigmoid(out["lang_logits"])[0].cpu().numpy()
        wp = out["waypoints"][0].cpu().numpy()
        probs.append(p)
        gts.append(np.array(r["lang"]))
        traj_len.append(float(np.linalg.norm(wp[-1])))  # how far the plan goes (m)
    probs, gts, traj_len = np.array(probs), np.array(gts), np.array(traj_len)

    # 1) language accuracy per label
    acc = ((probs > 0.5).astype(int) == gts).mean(0)
    print("=== language head accuracy ===")
    for i, n in enumerate(LABELS):
        print(f"  {n:6s} acc {acc[i]*100:5.1f}%  (base-rate {gts[:,i].mean()*100:.0f}%)")

    # 2) THE LIMIT: when it says STOP, does the trajectory actually stop?
    says_stop = probs[:, 2] > 0.5
    print("\n=== decoupling (limit) ===")
    print(f"  frames model says STOP: {says_stop.sum()}/{len(val)}")
    print(f"  mean planned distance when it says STOP : {traj_len[says_stop].mean():.1f} m")
    print(f"  mean planned distance when it says GO   : {traj_len[~says_stop].mean():.1f} m")
    print("  -> if these are similar, the language does NOT drive the action (decoupled).")

    # montage: pick frames where it says STOP but plans a long path (the contradiction)
    contradiction = np.where(says_stop & (traj_len > 5.0))[0]
    pick = contradiction[:: max(1, len(contradiction) // 6)][:6] if len(contradiction) else range(6)
    for idx in pick:
        r = val[idx]
        scene = r["img"].split("/")[1]
        fi = int(r["img"].split("/")[2].split(".")[0])
        s = get_state(scene, fi)
        img = cv2.imread(os.path.join(ROOT, r["img"]))
        h, w = img.shape[:2]
        wp = model(tf(Image.open(os.path.join(ROOT, r["img"])).convert("RGB")).unsqueeze(0).to(device),
                   torch.tensor([r["speed"]], device=device))["waypoints"][0].cpu().numpy()
        pts = np.concatenate([wp, np.zeros((len(wp), 1))], axis=1)
        uv = project_ego_to_image(pts, s["intrinsic_matrix"], s["extrinsic_matrix"])
        for u in uv:
            cv2.circle(img, (int(u[0] * w / 1928), int(u[1] * h / 1208)), 5, (0, 255, 0), -1)
        p = probs[idx]
        cv2.putText(img, f"says: RED {p[0]:.2f} STOP {p[2]:.2f}  | plans {traj_len[idx]:.0f}m AHEAD",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, "narrates STOP, drives ON", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        rows.append(cv2.resize(img, (520, 325)))
    if rows:
        grid = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows) - len(rows) % 2, 2)])
        out_path = os.path.join(HERE, "assets", "v4_lang_eval.jpg")
        cv2.imwrite(out_path, grid)
        print("\nwrote", out_path)


if __name__ == "__main__":
    main()
