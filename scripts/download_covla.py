#!/usr/bin/env python3
"""Selective downloader for the (gated) CoVLA-Dataset-Mini.

Disk-aware: the full Mini repo is ~75GB because images/*.tar.gz are ~1.5GB each.
By default we grab the cheap per-frame annotations + the small mp4 clips for ALL
scenes (~3GB total) and only pull full-res image tars for a few scenes.

Usage (after `huggingface-cli login` and accepting the dataset license):
    python scripts/download_covla.py --scenes 50 --image-tars 0     # light, mp4-based
    python scripts/download_covla.py --scenes 8  --image-tars 8     # full-res for 8 scenes
"""
import argparse
import os
from huggingface_hub import HfApi, snapshot_download

REPO = "turing-motors/CoVLA-Dataset-Mini"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(HERE, "data", "covla_mini")


def list_scenes(api: HfApi) -> list[str]:
    files = api.list_repo_files(REPO, repo_type="dataset")
    scenes = sorted(
        f[len("captions/"):-len(".jsonl")]
        for f in files
        if f.startswith("captions/") and f.endswith(".jsonl")
    )
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=50, help="how many scenes (annotations+mp4)")
    ap.add_argument("--image-tars", type=int, default=0, help="how many scenes to also pull full-res images/*.tar.gz")
    args = ap.parse_args()

    api = HfApi()
    scenes = list_scenes(api)[: args.scenes]
    print(f"selected {len(scenes)} scenes")

    patterns = ["metadata.json", "index.csv", "README.md"]
    for s in scenes:
        patterns += [
            f"captions/{s}.jsonl",
            f"states/{s}.jsonl",
            f"front_car/{s}.jsonl",
            f"traffic_lights/{s}.jsonl",
            f"video_samples/{s}.mp4",
        ]
    for s in scenes[: args.image_tars]:
        patterns.append(f"images/{s}.tar.gz")

    print(f"downloading {len(patterns)} patterns -> {DEST}")
    snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=DEST,
        allow_patterns=patterns,
        max_workers=8,
    )
    print("done:", DEST)


if __name__ == "__main__":
    main()
