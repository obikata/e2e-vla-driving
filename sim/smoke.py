"""CARLA connection smoke test: verify server+client handshake before the closed loop.

Connects, lists maps, confirms the Tesla blueprint exists, spawns the ego + camera,
ticks a few synchronous frames, and reports camera frame shape. Cleans up actors.
"""
import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town10HD_Opt")
    args = ap.parse_args()
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print("client version:", client.get_client_version(), "| server:", client.get_server_version())
    print("available maps:", [m.split("/")[-1] for m in client.get_available_maps()])

    world = client.load_world(args.town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    bp = world.get_blueprint_library()
    ego_bp = bp.find("vehicle.tesla.model3")
    print("ego blueprint:", ego_bp.id)
    sp = world.get_map().get_spawn_points()
    ego = world.spawn_actor(ego_bp, sp[0])
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "960")
    cam_bp.set_attribute("image_size_y", "600")
    cam_bp.set_attribute("fov", "40")
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.4, z=1.3)), attach_to=ego)
    got = {"n": 0, "shape": None}
    cam.listen(lambda im: got.update(n=got["n"] + 1,
                                     shape=(im.height, im.width)))
    try:
        for _ in range(20):
            world.tick()
            time.sleep(0.02)
        print(f"camera frames received: {got['n']} shape={got['shape']}")
        print("SMOKE OK" if got["n"] > 0 else "SMOKE FAIL: no camera frames")
    finally:
        cam.destroy()
        ego.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
