#!/usr/bin/env python3
"""Print the real schema of one CoVLA scene so we can lock the dataloader.

Run after `huggingface-cli login` + dataset access. Downloads just the small
jsonl files for ONE scene (no images), then dumps keys + sample values.
"""
import json
import os
from huggingface_hub import HfApi, hf_hub_download

REPO = "turing-motors/CoVLA-Dataset-Mini"


def first_scene(api):
    files = api.list_repo_files(REPO, repo_type="dataset")
    for f in sorted(files):
        if f.startswith("captions/") and f.endswith(".jsonl"):
            return f[len("captions/"):-len(".jsonl")]
    raise RuntimeError("no scene found")


def head_jsonl(path, n=2):
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


def main():
    api = HfApi()
    scene = first_scene(api)
    print("scene:", scene)
    for kind in ["states", "captions", "front_car", "traffic_lights"]:
        try:
            p = hf_hub_download(REPO, f"{kind}/{scene}.jsonl", repo_type="dataset")
            rows = head_jsonl(p, 2)
            print(f"\n===== {kind} =====")
            with open(p) as fh:
                n_lines = sum(1 for _ in fh)
            print(f"lines: {n_lines}")
            if rows:
                print("keys:", list(rows[0].keys()))
                print("sample[0]:", json.dumps(rows[0], ensure_ascii=False)[:1200])
        except Exception as e:
            print(f"\n===== {kind} ===== ERROR {e}")
    # metadata
    try:
        mp = hf_hub_download(REPO, "metadata.json", repo_type="dataset")
        print("\n===== metadata.json =====")
        print(open(mp).read()[:800])
    except Exception as e:
        print("metadata err", e)


if __name__ == "__main__":
    main()
