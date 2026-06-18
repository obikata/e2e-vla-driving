"""Sanity-check / demo overlay: project an ego-frame trajectory onto the camera image.

CoVLA convention (verified from states):
  trajectory points are in the ego/vehicle frame: x_fwd, y_left, z_up [m].
  extrinsic_matrix is ego->camera (P_cam = R @ P_ego + t).
  intrinsic_matrix is the standard 3x3 K for the 1928x1208 capture.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "data", "covla_mini")


def project_ego_to_image(pts_ego, K, T_ego2cam, img_w_full=1928):
    """pts_ego: (N,3). Returns (M,2) pixel coords for points in front of camera."""
    P = np.asarray(pts_ego, dtype=np.float64)
    R = np.asarray(T_ego2cam)[:3, :3]
    t = np.asarray(T_ego2cam)[:3, 3]
    cam = (R @ P.T).T + t  # (N,3) in camera frame: x right, y down, z fwd
    z = cam[:, 2]
    front = z > 0.1
    cam = cam[front]
    uv = (np.asarray(K) @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv


def draw_trajectory(img_bgr, uv, color=(0, 255, 0), scale_x=1.0, scale_y=1.0):
    pts = uv.copy()
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    pts = pts.astype(int)
    for j in range(len(pts) - 1):
        cv2.line(img_bgr, tuple(pts[j]), tuple(pts[j + 1]), color, 4, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img_bgr, tuple(p), 5, color, -1, cv2.LINE_AA)
    return img_bgr


def draw_bev(waypoints, size=300, max_m=20.0):
    """Top-down view of ego-frame waypoints (x_fwd up, y_left left)."""
    bev = np.zeros((size, size, 3), np.uint8)
    cx, cy = size // 2, size - 20
    s = (size - 40) / max_m
    cv2.circle(bev, (cx, cy), 5, (0, 0, 255), -1)  # ego
    for wp in waypoints:
        px = int(cx - wp[1] * s)   # y_left -> screen left
        py = int(cy - wp[0] * s)   # x_fwd  -> screen up
        cv2.circle(bev, (px, py), 4, (0, 255, 0), -1)
    cv2.putText(bev, "BEV", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return bev


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(HERE, "assets", "label_check.jpg"))
    args = ap.parse_args()
    scene = args.scene or open(f"{ROOT}/scenes.txt").readline().strip()
    st = [json.loads(l) for l in open(f"{ROOT}/states/{scene}.jsonl") if l.strip()]
    s = st[args.frame]

    img = cv2.imread(os.path.join(ROOT, "frames", scene, f"{args.frame:04d}.jpg"))
    if img is None:
        raise SystemExit(f"frame jpg not found; run prepare_dataset.py first")
    h, w = img.shape[:2]
    uv = project_ego_to_image(s["trajectory"], s["intrinsic_matrix"], s["extrinsic_matrix"])
    # frames were resized from 1928-wide capture -> scale projection accordingly
    draw_trajectory(img, uv, scale_x=w / 1928.0, scale_y=h / 1208.0)
    bev = draw_bev(np.asarray(s["trajectory"])[:, :2])
    bev = cv2.resize(bev, (h * bev.shape[1] // bev.shape[0], h)) if False else cv2.resize(bev, (h, h))
    canvas = np.hstack([img, bev])
    cv2.putText(canvas, f"{scene} f{args.frame}  v={s['ego_state']['vEgo']:.1f}m/s",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print("wrote", args.out, canvas.shape)


if __name__ == "__main__":
    main()
