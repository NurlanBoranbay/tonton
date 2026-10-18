#!/usr/bin/env python3
"""Freeze the current annotations into a YOLO dataset snapshot, split by VIDEO.
Usage: python3 make_split.py <snapshot_name> [--val X-2,Х-17,...]
Output: tonton/exp/data/<name>/{images,labels}/{train,val}/ (hard links) + data.yaml + split.json
Val videos are chosen greedily so every class has val boxes (seeded, deterministic)."""
import collections, json, os, random, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "..", "dataset", "yolo")
name = sys.argv[1]
OUT = os.path.join(ROOT, "data", name)
names = open(os.path.join(SRC, "classes.txt")).read().split()
recs = {}
for l in open(os.path.join(SRC, "annotations.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if r.get("deleted"): recs.pop(r["stem"], None)
    else: recs[r["stem"]] = r
recs = {s: r for s, r in recs.items() if os.path.exists(os.path.join(SRC, "images", s + ".jpg"))}
by_vid = collections.defaultdict(list)
for r in recs.values(): by_vid[r["video"]].append(r)
cls_vids = collections.defaultdict(set)
for v, rs in by_vid.items():
    for r in rs:
        for b in r["boxes"]: cls_vids[b["c"]].add(v)
if "--val" in sys.argv:
    val = set(sys.argv[sys.argv.index("--val") + 1].split(","))
else:
    rng = random.Random(0); val = set()
    for c in sorted(cls_vids, key=lambda c: len(cls_vids[c])):   # rarest class first
        if not cls_vids[c] & val:
            val.add(rng.choice(sorted(cls_vids[c])))
    vids = sorted(by_vid); rng.shuffle(vids)
    for v in vids:
        if len(val) >= 9: break
        val.add(v)
for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
n = collections.Counter()
for s, r in recs.items():
    part = "val" if r["video"] in val else "train"
    for d, ext in (("images", ".jpg"), ("labels", ".txt")):
        dst = os.path.join(OUT, d, part, s + ext)
        if not os.path.exists(dst): os.link(os.path.join(SRC, d, s + ext), dst)
    n[part] += 1
with open(os.path.join(OUT, "data.yaml"), "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnc: {len(names)}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(names)))
json.dump({"val_videos": sorted(val), "counts": n}, open(os.path.join(OUT, "split.json"), "w"), ensure_ascii=False, indent=1)
vb = collections.Counter(b["c"] for r in recs.values() if r["video"] in val for b in r["boxes"])
print(n, "val videos:", sorted(val)); print({names[c]: vb[c] for c in range(len(names))})
