"""Collect a CARLA driving dataset for sim-domain adaptation (D4 plan A).

Drives an ego on autopilot (Traffic Manager) through Town10HD with traffic, logs the
front camera + ego pose every tick, then builds ego-frame future-waypoint labels in the
SAME convention as CoVLA (x_fwd, y_left, meters). Output mirrors the CoVLA index so we
can reuse src/train.py for fine-tuning.

CARLA local frame is left-handed (x fwd, y RIGHT, z up); CoVLA uses y_LEFT, so we negate y.

Run with the UE5 server up:
    python sim/collect_carla.py --minutes 4 --traffic 40
"""
from __future__ import annotations

import argparse
import json
import os
import queue

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "carla_sim")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town10HD_Opt")
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--traffic", type=int, default=40)
    ap.add_argument("--fps", type=int, default=20)        # match CoVLA 20Hz
    ap.add_argument("--horizon-s", type=float, default=2.0)
    ap.add_argument("--n-waypoints", type=int, default=10)
    ap.add_argument("--stride", type=int, default=2)      # keep every 2nd frame for training
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--ego", default="vehicle.lincoln.mkz")
    args = ap.parse_args()
    import carla

    dt = 1.0 / args.fps
    H = round(args.horizon_s * args.fps)
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    # avoid a fragile map reload right after server boot: reuse the loaded world if it matches
    world = client.get_world()
    if args.town not in world.get_map().name:
        world = client.load_world(args.town)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    world.apply_settings(settings)

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    rng = np.random.default_rng(0)

    actors = []
    try:
        ego_bp = (bp.filter(args.ego) or bp.filter("vehicle.lincoln.mkz") or bp.filter("vehicle.*"))[0]
        ego = world.spawn_actor(ego_bp, spawns[0])
        actors.append(ego)
        ego.set_autopilot(True, tm.get_port())

        # background traffic
        cars = bp.filter("vehicle.*")
        for sp in rng.permutation(len(spawns))[: args.traffic]:
            sp = int(sp)
            if sp == 0:
                continue
            v = world.try_spawn_actor(cars[int(rng.integers(len(cars)))], spawns[sp])
            if v:
                v.set_autopilot(True, tm.get_port())
                actors.append(v)

        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "960")
        cam_bp.set_attribute("image_size_y", "600")
        cam_bp.set_attribute("fov", "40")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.4, z=1.3)), attach_to=ego)
        actors.append(cam)
        q: queue.Queue = queue.Queue()
        cam.listen(q.put)

        n_ticks = int(args.minutes * 60 * args.fps)
        print(f"collecting {n_ticks} ticks ({args.minutes} min @ {args.fps}Hz), H={H} ...")
        poses, speeds, img_idx = [], [], []
        # warmup so traffic settles
        for _ in range(20):
            world.tick(); q.get()
        for t in range(n_ticks):
            world.tick()
            try:
                image = q.get(timeout=5.0)
            except queue.Empty:
                print(f"camera stalled at t={t} (ego likely destroyed) — stopping early")
                break
            tf = ego.get_transform()
            M_inv = np.array(tf.get_inverse_matrix())   # world -> ego local
            poses.append(M_inv)
            loc = ego.get_location()
            poses_world = (loc.x, loc.y, loc.z)
            v = ego.get_velocity()
            speeds.append((v.x ** 2 + v.y ** 2) ** 0.5)
            # store world location separately for trajectory build
            img_idx.append(poses_world)
            if t % args.stride == 0:
                arr = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
                cv2.imwrite(os.path.join(OUT, "frames", f"{t:06d}.jpg"), arr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if t % (args.fps * 20) == 0:
                print(f"  t={t}/{n_ticks} speed={speeds[-1]:.1f}")

        cam.stop()  # stop the sensor stream before teardown (avoids 0.10 crash on destroy)
        # build waypoint labels from future world positions transformed into each ego frame
        world_xyz = np.array(img_idx)  # (T,3)
        T = len(world_xyz)              # actual collected ticks (may be < n_ticks if stalled)
        items = []
        for t in range(0, T, args.stride):
            if t + H >= T:
                continue
            jpg = os.path.join(OUT, "frames", f"{t:06d}.jpg")
            if not os.path.exists(jpg):
                continue
            M_inv = poses[t]
            idxs = np.linspace(t + 1, t + H, args.n_waypoints).round().astype(int)
            fut = world_xyz[idxs]                       # (n,3) world
            homog = np.concatenate([fut, np.ones((len(fut), 1))], axis=1)  # (n,4)
            local = (M_inv @ homog.T).T[:, :3]          # ego-local: x fwd, y right, z up
            wp = np.stack([local[:, 0], -local[:, 1]], axis=1)  # -> x_fwd, y_left (CoVLA)
            # skip near-stationary frames (autopilot stopped at light) to avoid degenerate labels
            if np.linalg.norm(wp[-1]) < 1.0:
                continue
            items.append({"img": f"frames/{t:06d}.jpg", "speed": float(speeds[t]),
                          "waypoints": wp.tolist()})

        rng.shuffle(items)
        nval = int(len(items) * args.val_frac)
        json.dump(items[nval:], open(os.path.join(OUT, "index_train.json"), "w"))
        json.dump(items[:nval], open(os.path.join(OUT, "index_val.json"), "w"))
        print(f"DONE: {len(items)} samples (train {len(items)-nval}, val {nval}) -> {OUT}")
    finally:
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:
                pass
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()
