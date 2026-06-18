# Experiments log

Evolution of the driving policy, with reproduction commands and closed-loop metrics.
Weights and data are gitignored; each model is reproduced from code + the command below.
Closed-loop metrics: 30 s in CARLA UE5 Town10HD, ego `vehicle.lincoln.mkz`, dt=0.1.
(`distance` is inflated by jitter when stuck; read it together with `mean_speed`.)

| ver | model | key change | dist (m) | collisions | mean speed (m/s) | verdict |
|-----|-------|-----------|---------:|-----------:|-----------------:|---------|
| v0 | `best.pt` | CoVLA-only imitation (front cam + speed → 10 waypoints) | 102.9 | 135 | 1.1 | drives a bit, then pins on a parked car |
| v1 | `best_carla.pt` | + fine-tune on CARLA autopilot data (sim-domain adapt) | 104.4 | 214 | 1.2 | worse — **right-drift** into parked cars (data was 31% right / 4% left turns) |
| v2 | `best_carla_mirror.pt` | + horizontal-mirror augmentation (flip img + negate lateral target) | **247.1** | **41** | **6.0** | **cruises ~22 km/h**; lane-keeping imperfect, runs red lights |
| v3 | _planned_ | keep stopped frames in collection (currently filtered out) so it learns to **stop at red** | — | — | — | TODO |

## Reproduce
```bash
. .venv/bin/activate
# data (deterministic seed)
python scripts/download_covla.py --scenes 50 --image-tars 0   # gated; needs HF token
python scripts/prepare_dataset.py
python sim/collect_carla.py --minutes 3 --traffic 20          # needs UE5 server up

# v0  CoVLA-only
python src/train.py --config configs/train.yaml

# v1  + CARLA sim-domain fine-tune
python src/train.py --init checkpoints/best.pt --data-root data/carla_sim \
    --tag carla --lr 1e-4 --epochs 12

# v2  + mirror augmentation (now default in src/covla_dataset.py for the train split)
python src/train.py --init checkpoints/best.pt --data-root data/carla_sim \
    --tag carla_mirror --lr 1e-4 --epochs 12

# evaluate any model (closed-loop metrics + overlay video)
python sim/closed_loop.py --ckpt checkpoints/best_carla_mirror.pt --seconds 30
```

## Lessons (the evolution story)
- **v0→v1 regression**: naive sim-adaptation made it *worse*. Root cause found by inspecting the
  approach-to-collision frames: a **rightward steering bias** from right-heavy autopilot turns.
- **v1→v2 fix**: mirror augmentation symmetrizes left/right → 5× speed, ~4× fewer collisions.
- **open-loop ADE ≠ closed-loop quality**: v1 had lower val ADE than v0 but drove worse.
- **v2 runs red lights**: the collector filtered out near-stationary frames, deleting every
  "stopped at a light" example, so the policy never learned to stop. → v3.
