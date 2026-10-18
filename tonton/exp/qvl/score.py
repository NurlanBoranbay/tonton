#!/usr/bin/env python3
import csv, json, sys
L = {r["file"]: (int(r["violation"]), float(r["timestamp_sec"]) if r["timestamp_sec"] else None) for r in csv.DictReader(open("/nlp/nurlan/tonton/dataset/labels.csv"))}
EX = {}
for r in csv.DictReader(open("/nlp/nurlan/tonton/dataset/violation_classes.csv")):
    for e in r["example_videos"].split(";"):
        if e: EX.setdefault(e + ".mov", set()).add(r["class_id"])
def score(path, allowed=None, verbose=True):
    R = [json.loads(l) for l in open(path)]
    tp = fp = fn = tn = typ = near = 0
    for r in sorted(R, key=lambda r: r["video"]):
        vs = [x for x in r["parsed"].get("violations", []) if isinstance(x, dict) and (allowed is None or x.get("class_id") in allowed)]
        y, t = L[r["video"]]; p = int(bool(vs))
        tp += y and p; fp += (not y) and p; fn += y and not p; tn += (not y) and not p
        dt = None
        if y and p:
            try: dt = min(abs(float(x.get("time_sec", -99)) - t) for x in vs)
            except Exception: dt = None
            near += dt is not None and dt <= 2
            typ += bool(EX.get(r["video"], set()) & {x.get("class_id") for x in vs})
        if verbose: print(f"{r['video']:9} y={y} p={p} dt={dt} gt={sorted(EX.get(r['video'], []))} pred={[ (x.get('class_id'), x.get('time_sec')) for x in vs]}")
    n = len(R)
    print(f"{path}: n={n} acc={(tp+tn)/n:.2f} TP={tp} FP={fp} FN={fn} TN={tn} recall={tp/max(tp+fn,1):.2f} prec={tp/max(tp+fp,1):.2f} type_ok={typ}/{tp} |dt|<=2s {near}/{tp}")
if __name__ == "__main__":
    score(sys.argv[1], verbose="-q" not in sys.argv)
