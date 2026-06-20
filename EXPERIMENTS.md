# Experiments log

Evolution of the driving policy, with reproduction commands and closed-loop metrics.
Weights and data are gitignored; each model is reproduced from code + the command below.
Closed-loop metrics: 30 s in CARLA UE5 Town10HD, ego `vehicle.lincoln.mkz`, dt=0.1.
(`distance` is inflated by jitter when stuck; read it together with `mean_speed`.)

| ver | model | key change | dist (m) | collisions | mean speed (m/s) | verdict |
|-----|-------|-----------|---------:|-----------:|-----------------:|---------|
| v0 | `best.pt` | CoVLA-only imitation (front cam + speed → 10 waypoints) | 102.9 | 135 | 1.1 | drives a bit, then pins on a parked car |
| v1 | `best_carla.pt` | + fine-tune on CARLA autopilot data (sim-domain adapt) | 104.4 | 214 | 1.2 | worse — **right-drift** into parked cars (data was 31% right / 4% left turns) |
| v2 | `best_carla_mirror.pt` | + horizontal-mirror augmentation (flip img + negate lateral target) | **247.1** | **41** | **6.0** | **best driver so far** — cruises ~22 km/h; lane-keeping imperfect, runs red lights |
| v3 | `best_carla_v3_stop.pt` | + keep 1/3 of stopped frames (24% of set) to learn to stop at red | 257.4 | 295 | 4.7 | **regressed** — learned to stop when *confused* (not at red lights), drove into a building and pinned. Pure VA can't do the semantic "red light → stop" reliably |
| v4 | `best_v4_lang.pt` | **add the "L" the cheap way**: parallel language head supervised by CoVLA captions | ADE 0.65 | — | — | narrates (RED acc 75%) **but intervention proves it's causally inert: forcing it to scream RED+STOP changes the trajectory by 0.000000 m**. saying ≠ deciding → motivates v5 |
| v5 | _planned_ | **world model**: a real (small) VLM so reasoning *drives* the action (reason-then-act); Alpamayo-R1-10B itself needs 24 GB (we have 16) | — | — | — | option B / A-stretch |

**D4 endpoint = v2.** Remaining gaps (red lights, intersections/navigation, obstacle avoidance) are
semantic/reasoning problems — the domain of VLA, not more VA tuning.

## Roadmap for the "L": C → limit → world model
Deliberately the same "try the simple thing, watch it break, escalate" arc used for v0→v2.

- **v4 = C (no world model):** paste a language head onto the VA policy, supervised by CoVLA's
  auto-captions. Goal is **not** to make it drive well — it's to *empirically expose the limit*:
  - **limit 1 (grounding):** the language is just another readout of the same visual features, so
    on out-of-domain CARLA it degrades like the vision does.
  - **limit 2 (decoupling):** even when it correctly *narrates* "red light → should stop", the
    trajectory head is a **parallel branch** that never reads the language, so the car narrates
    "red" and drives through anyway. **Saying ≠ deciding.**
- **v5 = world model:** make reasoning *causal* to the action (reason-then-act) using a real (small)
  VLM with actual world knowledge. Alpamayo is the dream teacher but is 24 GB-blocked on this 16 GB
  box, so the realistic path is a 2–3 B VLM (e.g. Qwen2-VL / InternVL) as backbone or teacher;
  Alpamayo via 4-bit/CPU-offload is a stretch.

This makes the case for the world model **empirically** (show C's ceiling) instead of asserting it.

## Reproduce
```bash
. .venv/bin/activate
# data (deterministic seed)
python scripts/download_covla.py --scenes 50 --image-tars 0   # gated; needs HF token
python scripts/prepare_dataset.py
python sim/collect_carla.py --minutes 3 --traffic 20          # needs UE5 server up

# v0  CoVLA-only (mirror off)            -> checkpoints/best.pt
python src/train.py --no-mirror

# v1  + CARLA sim-domain fine-tune, mirror off (the regression)
python src/train.py --init checkpoints/best.pt --data-root data/carla_sim \
    --no-mirror --tag carla --lr 1e-4 --epochs 12

# v2  + mirror augmentation                -> checkpoints/best_carla_mirror.pt
python src/train.py --init checkpoints/best.pt --data-root data/carla_sim \
    --tag carla_mirror --lr 1e-4 --epochs 12

# v3  + keep stopped frames in collection (collect with current sim/collect_carla.py)
python src/train.py --init checkpoints/best.pt --data-root data/carla_sim \
    --tag carla_v3_stop --lr 1e-4 --epochs 12

# v4  + parallel language head (option C)   -> checkpoints/best_v4_lang.pt
python scripts/add_caption_labels.py        # attach red/green/stop labels to the CoVLA index
python src/train.py --init checkpoints/best.pt --n-lang 3 --lang-weight 3.0 \
    --tag v4_lang --lr 2e-4 --epochs 12

# evaluate: closed-loop metrics + overlay video
python sim/closed_loop.py --ckpt checkpoints/best_carla_mirror.pt --seconds 30
# evaluate v4 language head + the decoupling intervention
python src/eval_v4_lang.py
```

## Lessons (the evolution story)
- **v0→v1 regression**: naive sim-adaptation made it *worse*. Root cause found by inspecting the
  approach-to-collision frames: a **rightward steering bias** from right-heavy autopilot turns.
- **v1→v2 fix**: mirror augmentation symmetrizes left/right → 5× speed, ~4× fewer collisions.
- **open-loop ADE ≠ closed-loop quality**: v1 had lower val ADE than v0 but drove worse.
- **v2 runs red lights**: the collector filtered out near-stationary frames, deleting every
  "stopped at a light" example, so the policy never learned to stop. → v3.
- **v3: stopped frames weren't enough**: adding back 24% stopped frames made the policy stop when
  *visually confused* (it drove into a building and halted), not specifically at red lights. The
  red light is a tiny, rare visual cue; a pure **Vision→Action** regressor has no world knowledge
  ("red = stop") and no reasoning. This is the structural limit of VA.
- **VA → VLA**: our policy uses only Vision + Action; it drops CoVLA's **Language** (captions).
  Semantic decisions (lights, signs, right-of-way) are exactly what the "L" / reasoning provides.
- **v4: a bolt-on language head is a *correlational illusion*, not a VLA.** It narrates (RED acc
  75%, weak — GREEN ≈ chance) and on in-distribution data its narration even agrees with the
  action — but only because both heads read the same features. **Intervention proves it is causally
  inert**: forcing the language to scream RED+STOP on every frame changes the trajectory by exactly
  **0.000000 m**. The trajectory head never reads the language. *Saying ≠ deciding.*
- **→ v5 (world model):** to make reasoning *cause* the action (reason-then-act) you need the
  action conditioned on the reasoning **and** real world knowledge so the reasoning is reliable
  out-of-distribution. Alpamayo-R1-10B is the ideal teacher but needs 24 GB (we have 16); the
  realistic path is a 2–3 B VLM (Qwen2-VL / InternVL) as backbone/teacher.
