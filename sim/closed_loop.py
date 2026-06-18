"""CARLA closed-loop runner: our CoVLA-trained policy drives a Tesla online.

Loop @ fixed dt:
  front RGB camera -> normalize -> TrajectoryPolicy -> ego-frame waypoints
  -> pure-pursuit steering + speed controller -> carla.VehicleControl -> tick
Renders an overlay (camera + predicted path + BEV + telemetry) and writes mp4.

Run a CARLA server first (see carla/README.md), then:
    python carla/closed_loop.py --ckpt checkpoints/best.pt --town Town10HD_Opt --seconds 60
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_model  # noqa: E402
from viz import draw_bev       # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----- controller -----
def pure_pursuit_steer(waypoints, wheelbase=2.8, lookahead=6.0, max_steer_rad=0.6):
    """waypoints: (N,2) ego frame x_fwd,y_left [m]. Returns steer in [-1,1]."""
    wp = np.asarray(waypoints)
    dists = np.linalg.norm(wp, axis=1)
    idx = int(np.argmin(np.abs(dists - lookahead)))
    tx, ty = wp[idx]
    Ld = max(math.hypot(tx, ty), 1e-3)
    # bicycle pure pursuit: curvature = 2*y / Ld^2 ; steer angle = atan(wheelbase*curv)
    curv = 2.0 * ty / (Ld * Ld)
    angle = math.atan(wheelbase * curv)
    return float(np.clip(angle / max_steer_rad, -1.0, 1.0))


def speed_target_from_traj(waypoints, dt_horizon):
    """Crude target speed = path length / horizon time."""
    wp = np.asarray(waypoints)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1).sum()
    return seg / max(dt_horizon, 1e-3)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "best.pt"))
    ap.add_argument("--town", default="Town10HD_Opt")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "assets", "carla_run.mp4"))
    args = ap.parse_args()

    import carla  # imported here so the rest of the repo doesn't require it

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    # build the same normalization the backbone expects
    import timm
    dcfg = timm.data.resolve_model_data_config(model.backbone)
    mean = torch.tensor(dcfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(dcfg["std"], device=device).view(1, 3, 1, 1)
    in_size = dcfg["input_size"][1:]
    horizon_s = cfg["horizon_s"]

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.load_world(args.town)
    # synchronous, fixed dt
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    bp = world.get_blueprint_library()
    ego_bp = bp.find("vehicle.tesla.model3")  # Tesla ego (narrative bonus)
    sp = world.get_map().get_spawn_points()
    ego = world.spawn_actor(ego_bp, sp[0])

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "960")
    cam_bp.set_attribute("image_size_y", "600")
    # match CoVLA optics: hfov = 2*atan(W/(2*fx)) with fx=2648,W=1928 -> ~40 deg
    cam_bp.set_attribute("fov", "40")
    cam_tf = carla.Transform(carla.Location(x=1.4, z=1.3))  # windshield, ~CoVLA mount height
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    latest = {"img": None}
    cam.listen(lambda im: latest.__setitem__(
        "img", np.frombuffer(im.raw_data, np.uint8).reshape(im.height, im.width, 4)[:, :, :3][:, :, ::-1].copy()))

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), int(1 / args.dt), (960 + 600, 600))
    n_steps = int(args.seconds / args.dt)
    print(f"running {n_steps} steps in {args.town} ...")
    try:
        for step in range(n_steps):
            world.tick()
            if latest["img"] is None:
                continue
            rgb = latest["img"]
            t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = torch.nn.functional.interpolate(t, size=in_size, mode="bilinear", align_corners=False)
            t = (t - mean) / std
            v = ego.get_velocity()
            speed = math.hypot(v.x, v.y)
            out = model(t, torch.tensor([speed], device=device))
            wp = out["waypoints"][0].cpu().numpy()  # (N,2)

            steer = pure_pursuit_steer(wp)
            tgt_v = speed_target_from_traj(wp, horizon_s)
            err = tgt_v - speed
            throttle = float(np.clip(0.3 * err + 0.15, 0.0, 0.7))
            brake = 0.0 if err > -1.0 else float(np.clip(-0.2 * err, 0, 1))
            ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

            # overlay
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bev = cv2.resize(draw_bev(wp), (600, 600))
            cv2.putText(frame, f"v={speed:4.1f} -> {tgt_v:4.1f} m/s  steer={steer:+.2f} thr={throttle:.2f}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            vw.write(np.hstack([frame, bev]))
    finally:
        vw.release()
        cam.destroy()
        ego.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
