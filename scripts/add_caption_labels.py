#!/usr/bin/env python3
"""Augment the CoVLA index with semantic language labels parsed from captions.

This is the data side of option C (the "L"): we attach, to each training frame, a few
binary semantic flags read from CoVLA's plain_caption, plus the caption text itself (for
the demo overlay). The language head will be trained to predict these flags from vision.

Labels: [light_red, light_green, should_stop]  (multi-label, 0/1)
"""
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "data", "covla_mini")

LABEL_NAMES = ["light_red", "light_green", "should_stop"]


def parse_concat_json(path):
    """CoVLA caption files are concatenated JSON objects (no newlines)."""
    txt = open(path).read()
    dec = json.JSONDecoder()
    i, out = 0, []
    while i < len(txt):
        while i < len(txt) and txt[i] in " \n\r\t":
            i += 1
        if i >= len(txt):
            break
        obj, j = dec.raw_decode(txt, i)
        out.append(obj)
        i = j
    return out


def labels_from_caption(c: str):
    cl = c.lower()
    red = int("red" in cl)
    green = int("green" in cl)
    stop = int(bool(re.search(r"stopp|should stop|comes to a stop|is stopped", cl)))
    return [red, green, stop], c


def scene_frame(img_rel):
    p = img_rel.split("/")           # frames/<scene>/<idx>.jpg
    return p[1], int(p[2].split(".")[0])


def main():
    cap_cache = {}

    def get_caps(scene):
        if scene not in cap_cache:
            cap_cache[scene] = parse_concat_json(f"{ROOT}/captions/{scene}.jsonl")
        return cap_cache[scene]

    for split in ["index_train.json", "index_val.json"]:
        path = os.path.join(ROOT, split)
        items = json.load(open(path))
        n_red = n_stop = 0
        for r in items:
            scene, fi = scene_frame(r["img"])
            caps = get_caps(scene)
            cap = caps[fi]["plain_caption"] if fi < len(caps) else ""
            lab, text = labels_from_caption(cap)
            r["lang"] = lab
            r["caption"] = text
            n_red += lab[0]
            n_stop += lab[2]
        json.dump(items, open(path, "w"))
        print(f"{split}: {len(items)} items | labels={LABEL_NAMES} | "
              f"red {100*n_red/len(items):.0f}% stop {100*n_stop/len(items):.0f}%")


if __name__ == "__main__":
    main()
