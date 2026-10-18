#!/usr/bin/env python3
"""Desk-crop dataset: crop each labelled frame to the work area (union of counting_machine + safe boxes
+ margin), keep boxes mostly inside, re-normalise. Split = same video split as data/<src>/split.json.
Usage: python3 make_crops.py v1 crop_v1"""
import json, os, sys
from PIL import Image
SRC, DST = sys.argv[1], sys.argv[2]
R = "/nlp/nurlan/tonton/exp/data"
M = .08   # margin (fraction of frame)
names = open("/nlp/nurlan/tonton/dataset/yolo/classes.txt").read().split()
ANCH = {names.index("counting_machine"), names.index("safe_open"), names.index("safe_closed")}
n = {"train": 0, "val": 0}; kept = [0] * len(names)
for part in ("train", "val"):
    os.makedirs(f"{R}/{DST}/images/{part}", exist_ok=True); os.makedirs(f"{R}/{DST}/labels/{part}", exist_ok=True)
    for fn in os.listdir(f"{R}/{SRC}/labels/{part}"):
        L = [l.split() for l in open(f"{R}/{SRC}/labels/{part}/{fn}") if l.strip()]
        B = [(int(c), float(x) - float(w) / 2, float(y) - float(h) / 2, float(x) + float(w) / 2, float(y) + float(h) / 2) for c, x, y, w, h in L]
        a = [b for b in B if b[0] in ANCH]
        if not a: continue
        x1 = max(0, min(b[1] for b in a) - M); y1 = max(0, min(b[2] for b in a) - M)
        x2 = min(1, max(b[3] for b in a) + M); y2 = min(1, max(b[4] for b in a) + M)
        img = Image.open(f"{R}/{SRC}/images/{part}/{fn[:-4]}.jpg"); W, H = img.size
        img.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))).save(f"{R}/{DST}/images/{part}/{fn[:-4]}.jpg", quality=95)
        cw, ch = x2 - x1, y2 - y1; out = []
        for c, bx1, by1, bx2, by2 in B:
            ix1, iy1, ix2, iy2 = max(bx1, x1), max(by1, y1), min(bx2, x2), min(by2, y2)
            if ix2 <= ix1 or iy2 <= iy1: continue
            if (ix2 - ix1) * (iy2 - iy1) < .5 * (bx2 - bx1) * (by2 - by1): continue
            out.append(f"{c} {((ix1 + ix2) / 2 - x1) / cw:.6f} {((iy1 + iy2) / 2 - y1) / ch:.6f} {(ix2 - ix1) / cw:.6f} {(iy2 - iy1) / ch:.6f}")
            kept[c] += 1
        open(f"{R}/{DST}/labels/{part}/{fn}", "w").write("\n".join(out) + ("\n" if out else ""))
        n[part] += 1
open(f"{R}/{DST}/data.yaml", "w").write(f"path: {R}/{DST}\ntrain: images/train\nval: images/val\nnc: {len(names)}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(names)))
print(n, dict(zip(names, kept)))
