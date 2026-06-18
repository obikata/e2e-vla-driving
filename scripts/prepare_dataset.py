#!/usr/bin/env python3
"""Preprocess CoVLA-Mini into a fast training index.

For each scene: sequentially decode the mp4 (fast), save strided frames as jpg,
and build per-sample labels from states:
    speed     = ego_state.vEgo  [m/s]
    waypoints = future trajectory resampled to n_waypoints (x_fwd, y_left) [m]

Scene-level train/val split (held-out scenes) to avoid frame leakage.
Outputs: data/covla_mini/frames/<scene>/<idx>.jpg  +  index_{train,val}.json
"""
from __future__ import annotations

import argparse
import json
import os

import av
import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "data", "covla_mini")


def load_states(scene):
    return [json.loads(l) for l in open(f"{ROOT}/states/{scene}.jsonl") if l.strip()]


def resample_traj(traj, horizon_frames, n_waypoints):
    """traj: (60,3) ego-frame future path @20Hz. Take first horizon_frames, pick n evenly."""
    a = np.asarray(traj, dtype=np.float32)  # (60,3) x_fwd, y_left, z_up
    h = min(horizon_frames, len(a))
    idx = np.linspace(0, h - 1, n_waypoints).round().astype(int)
    wp = a[idx, :2]  # (n,2) -> x_fwd, y_left
    return wp


def save_frame(img_rgb, path, out_w):
    import cv2
    h, w = img_rgb.shape[:2]
    if out_w and w > out_w:
        out_h = round(h * out_w / w)
        img_rgb = cv2.resize(img_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])  # RGB->BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "train.yaml"))
    ap.add_argument("--out-width", type=int, default=960)
    ap.add_argument("--val-scenes", type=int, default=6, help="held-out scenes for val")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fps = cfg["fps"]
    stride = cfg["frame_stride"]
    n_wp = cfg["n_waypoints"]
    horizon_frames = round(cfg["horizon_s"] * fps)

    scenes = [s.strip() for s in open(f"{ROOT}/scenes.txt") if s.strip()]
    val_set = set(scenes[: args.val_scenes])
    print(f"{len(scenes)} scenes | val={len(val_set)} | stride={stride} | "
          f"horizon={horizon_frames}f -> {n_wp} wp")

    train_idx, val_idx = [], []
    for si, scene in enumerate(scenes):
        st = load_states(scene)
        n = len(st)
        os.makedirs(f"{ROOT}/frames/{scene}", exist_ok=True)
        want = set(range(0, n, stride))
        try:
            container = av.open(f"{ROOT}/video_samples/{scene}.mp4")
        except av.error.FFmpegError as e:
            print(f"[{si+1}/{len(scenes)}] {scene}: SKIP bad mp4 ({type(e).__name__})")
            continue
        kept = 0
        for fi, frame in enumerate(container.decode(video=0)):
            if fi not in want:
                continue
            s = st[fi]
            # need a full future horizon for a clean label
            if fi + horizon_frames >= n:
                continue
            jpg = f"{ROOT}/frames/{scene}/{fi:04d}.jpg"
            if not os.path.exists(jpg):
                save_frame(frame.to_ndarray(format="rgb24"), jpg, args.out_width)
            wp = resample_traj(s["trajectory"], horizon_frames, n_wp)
            rec = {
                "img": f"frames/{scene}/{fi:04d}.jpg",
                "speed": float(s["ego_state"]["vEgo"]),
                "steer_deg": float(s["ego_state"]["steeringAngleDeg"]),
                "waypoints": wp.tolist(),
            }
            (val_idx if scene in val_set else train_idx).append(rec)
            kept += 1
        container.close()
        print(f"[{si+1}/{len(scenes)}] {scene}: kept {kept}")

    json.dump(train_idx, open(f"{ROOT}/index_train.json", "w"))
    json.dump(val_idx, open(f"{ROOT}/index_val.json", "w"))
    print(f"DONE: train={len(train_idx)} val={len(val_idx)} samples")


if __name__ == "__main__":
    main()
