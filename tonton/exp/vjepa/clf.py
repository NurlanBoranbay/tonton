#!/usr/bin/env python3
"""Window classifier on V-JEPA 2 embeddings, leave-one-group-out (group = staged pair / duplicate).
Positive windows: centre within ±POS s of the labelled violation time. Negatives: all windows of clean videos
(+ windows of violation videos farther than 6 s from the label). Video score = max window prob."""
import csv, os, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
E = "/nlp/nurlan/tonton/exp/vjepa/emb"; POS = 1.5
L = {r["file"][:-4]: (int(r["violation"]), float(r["timestamp_sec"]) if r["timestamp_sec"] else None) for r in csv.DictReader(open("/nlp/nurlan/tonton/dataset/labels.csv"))}
L["Х-4"] = (1, [57.0, 68.0]); L["Х-47"] = (1, [57.0, 68.0])          # identical files, two labelled moments
def group(s):
    if s in ("Х-4", "Х-47", "X-2", "Х-3"): return "g4"
    if s in ("Х-5", "Х-48"): return "g5"
    n = int(re.sub(r"\D", "", s)); return f"p{(n - (n % 2 == 1 and n > 9)) // 2}" if n >= 10 else f"s{n}"
D = {s: np.load(f"{E}/{s}.npz") for s in L if os.path.exists(f"{E}/{s}.npz")}
def xy(stems):
    X, Y = [], []
    for s in stems:
        y, t = L[s]; ts = t if isinstance(t, list) else [t]
        for tc, e in zip(D[s]["t"], D[s]["e"]):
            d = min(abs(tc - x) for x in ts) if y else 99
            if y and POS < d < 6: continue
            X.append(e); Y.append(int(y and d <= POS))
    return np.array(X), np.array(Y)
groups = sorted({group(s) for s in D}); res = {}
for g in groups:
    tr = [s for s in D if group(s) != g]; te = [s for s in D if group(s) == g]
    X, Y = xy(tr); sc = StandardScaler().fit(X); pca = PCA(64, random_state=0).fit(sc.transform(X))
    clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(pca.transform(sc.transform(X)), Y)
    for s in te:
        p = clf.predict_proba(pca.transform(sc.transform(D[s]["e"])))[:, 1]
        res[s] = (p.max(), float(D[s]["t"][p.argmax()]))
print("groups", len(groups))
for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
    tp = fp = fn = tn = near = 0
    for s, (pm, tm) in res.items():
        y, t = L[s]; p = pm >= thr
        tp += y and p; fp += (not y) and p; fn += y and not p; tn += (not y) and not p
        if y and p: near += min(abs(tm - x) for x in (t if isinstance(t, list) else [t])) <= 2
    print(f"thr {thr}: acc {(tp+tn)/len(res):.2f} TP {tp} FP {fp} FN {fn} TN {tn} |dt|<=2s {near}/{tp}")
for s in sorted(res, key=lambda s: (L[s][0], s)): print(s, L[s][0], f"{res[s][0]:.2f} @{res[s][1]:.1f}s label {L[s][1]}")
