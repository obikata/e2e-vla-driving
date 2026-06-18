"""Imitation-learning trainer for the CoVLA trajectory policy.

Dataset contract (see src/covla_dataset.py): each item is a dict with
    image:     float tensor (3,H,W), already normalized for the backbone
    speed:     float tensor () ego speed [m/s]
    waypoints: float tensor (n_waypoints, 2) future (dx,dy) in ego frame [m]
    valid:     float tensor (n_waypoints,) 1 if waypoint exists else 0

Loss = masked L2 on waypoints (optionally distance-weighted so near-term
waypoints, which dominate control, are emphasized).
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from covla_dataset import build_datasets
from model import build_model

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def masked_traj_loss(pred, target, valid, near_weight=True):
    # pred/target: (B,T,2)  valid: (B,T)
    err = (pred - target).pow(2).sum(-1)  # (B,T) squared L2 per waypoint
    if near_weight:
        T = pred.size(1)
        w = torch.linspace(1.5, 0.5, T, device=pred.device)  # emphasize near-term
        err = err * w
    err = err * valid
    return err.sum() / valid.sum().clamp(min=1.0)


def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in loader:
            img = b["image"].to(device, non_blocking=True)
            spd = b["speed"].to(device, non_blocking=True)
            tgt = b["waypoints"].to(device, non_blocking=True)
            val = b["valid"].to(device, non_blocking=True)
            out = model(img, spd)
            # report ADE (avg displacement error, meters) on valid waypoints
            disp = (out["waypoints"] - tgt).norm(dim=-1) * val
            tot += disp.sum().item()
            n += val.sum().item()
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "train.yaml"))
    ap.add_argument("--init", default=None, help="checkpoint to warm-start from (fine-tuning)")
    ap.add_argument("--data-root", default=None, help="override cfg data_root")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tag", default="", help="checkpoint name suffix, e.g. 'carla' -> best_carla.pt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.lr:
        cfg["lr"] = args.lr
    suffix = f"_{args.tag}" if args.tag else ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    train_ds, val_ds = build_datasets(cfg)
    print(f"train={len(train_ds)} val={len(val_ds)}")
    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg.get("num_workers", 8), pin_memory=True, drop_last=True,
                          persistent_workers=cfg.get("num_workers", 8) > 0)
    val_ld = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=4, pin_memory=True)

    model = build_model(cfg).to(device)
    if args.init:
        sd = torch.load(args.init, map_location=device)["model"]
        model.load_state_dict(sd)
        print(f"warm-started from {args.init}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=len(train_ld))
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    best = 1e9
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        for i, b in enumerate(train_ld):
            img = b["image"].to(device, non_blocking=True)
            spd = b["speed"].to(device, non_blocking=True)
            tgt = b["waypoints"].to(device, non_blocking=True)
            val = b["valid"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                out = model(img, spd)
                loss = masked_traj_loss(out["waypoints"], tgt, val)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if i % cfg.get("log_every", 50) == 0:
                print(f"ep{ep} it{i}/{len(train_ld)} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}")
        ade = evaluate(model, val_ld, device)
        dt = time.time() - t0
        print(f"== epoch {ep} done in {dt:.0f}s | val ADE {ade:.3f} m ==")
        ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": ep, "val_ade": ade}
        torch.save(ckpt, os.path.join(HERE, "checkpoints", f"last{suffix}.pt"))
        if ade < best:
            best = ade
            torch.save(ckpt, os.path.join(HERE, "checkpoints", f"best{suffix}.pt"))
            print(f"  new best ADE {best:.3f} m -> checkpoints/best{suffix}.pt")


if __name__ == "__main__":
    main()
