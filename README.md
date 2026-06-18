# E2E VLA Driving Demo (CoVLA → closed-loop in CARLA)

Self-built end-to-end autonomous-driving demo: train a vision→trajectory policy on the
**CoVLA dataset**, then run it **online, closed-loop, in CARLA** driving a Tesla.
Phase 2 distills NVIDIA **Alpamayo-R1** reasoning into the student.

## Hardware / stack
- RTX 5070 Ti (16GB, Blackwell sm_120) · torch 2.11+cu128 · uv venv
- Data: CoVLA-Dataset-Mini (gated, HF) — front camera + ego state + future trajectory + captions
- Sim: CARLA 0.9.15 (closed-loop)

## Pipeline
```
CoVLA mp4+states ──prepare_dataset──▶ frames/*.jpg + index_{train,val}.json
                                          │
                          TrajectoryPolicy (timm backbone → N waypoints)
                                          │  train.py (imitation, masked L2)
                                          ▼
                                   checkpoints/best.pt
                                          │
   CARLA front cam ──▶ policy ──▶ waypoints ──pure-pursuit──▶ VehicleControl ──▶ tick (online)
```

## Key design decisions
- **Label = CoVLA `trajectory`** (60 pts, ego frame x_fwd/y_left/z_up, meters) resampled to
  `n_waypoints` over `horizon_s`. Verified by projecting onto the camera image (`assets/label_check.jpg`).
- **Geometry-preserving transforms only** (no flip/RandomResizedCrop) — those would corrupt the
  left/right trajectory target.
- **real→sim domain gap** is the #1 risk: policy trained on real Tokyo frames must drive on
  CARLA renders. Mitigations (D4): DINOv2 backbone option, color aug, light in-sim adaptation.

## Layout
- `src/model.py` — TrajectoryPolicy (image[+speed] → waypoints, real-time)
- `src/covla_dataset.py` — dataset over the preprocessed index
- `src/train.py` — AMP imitation trainer, reports ADE (m)
- `src/viz.py` — trajectory→image projection + BEV overlay
- `scripts/download_covla.py` / curl path — selective gated download
- `scripts/prepare_dataset.py` — mp4 → frames + waypoint labels
- `sim/closed_loop.py` — online closed-loop runner (Tesla ego)

## Run
```bash
. .venv/bin/activate
export HF_HUB_DISABLE_XET=1                     # hf python stack hangs here; curl path used for DL
python scripts/prepare_dataset.py               # build frames + index
python src/train.py --config configs/train.yaml # D2 train
# D3: start CARLA server, then:
python sim/closed_loop.py --ckpt checkpoints/best.pt
```

## Status
- [x] D1 env + GPU (Blackwell) + data pipeline + label sanity check
- [x] D2 train policy, offline ADE (best val ADE 0.786 m)
- [x] D3 CARLA closed-loop (upgraded to CARLA 0.10 / Unreal Engine 5.5)
- [~] D4 sim-domain adaptation — see [EXPERIMENTS.md](EXPERIMENTS.md); v2 cruises ~22 km/h,
  next: stop at red lights
- [ ] D5 Alpamayo distillation + reasoning overlay
- [ ] D6 demo polish

See [EXPERIMENTS.md](EXPERIMENTS.md) for the model-version evolution and metrics.

## License
MIT (see [LICENSE](LICENSE)). Applies only to the original source here; CoVLA / CARLA / Alpamayo are third-party and not redistributed.
