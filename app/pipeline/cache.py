"""Offline rows for the 42 training videos from tonton's cached detections (no GPU needed).

  base   : feats/base/<stem>.jsonl      pose tracks + COCO (yolo26x, stride 3)
  custom : feats/<cust>/<stem>.jsonl    custom 10-class detector (cust_s_v1 = in-sample, cust_cvb = 4-fold CV)
  luma   : cache/luma/<stem>.jsonl      laptop screen luma, built here by `python3 -m app.pipeline.cache --luma`
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATS = os.path.join(ROOT, "cache", "feats_tonton")
LUMA = os.path.join(ROOT, "cache", "luma")
VIDEOS = os.path.join(ROOT, "data", "videos")


def _read(path):
    return {r["f"]: r for r in map(json.loads, open(path, encoding="utf-8"))} if os.path.exists(path) else {}


def stems():
    return sorted(os.path.splitext(v)[0] for v in os.listdir(VIDEOS) if v.endswith(".mov"))


def load_rows(stem, cust="cust_s_v1"):
    B = _read(os.path.join(FEATS, "base", stem + ".jsonl"))
    own = os.path.join(ROOT, "cache", "feats", cust, stem + ".jsonl")      # detections made here (tools/extract_custom.py)
    C = _read(own if os.path.exists(own) else os.path.join(FEATS, cust, stem + ".jsonl"))
    L = _read(os.path.join(LUMA, stem + ".jsonl"))
    rows = []
    for f in sorted(set(B) & set(C)):
        r = dict(B[f]); r["custom"] = C[f].get("custom", [])
        r["laptops"] = L.get(f, {}).get("laptops", [])
        r["frame_luma"] = L.get(f, {}).get("frame_luma")
        rows.append(r)
    return rows


def build_luma(stem, upper=.6):
    import cv2
    from app.pipeline.timeline import CONF, screen_luma
    B = _read(os.path.join(FEATS, "base", stem + ".jsonl"))
    cap = cv2.VideoCapture(os.path.join(VIDEOS, stem + ".mov"))
    out, f = [], -1
    while True:
        ok, img = cap.read(); f += 1
        if not ok: break
        if f not in B: continue
        lap = [d["box"] for d in B[f].get("coco", []) if d["c"] == "laptop" and d["conf"] >= CONF["laptop"]]
        out.append({"f": f, "frame_luma": round(float(img.mean()), 1),
                    "laptops": [{"box": b, "luma": screen_luma(img, b, upper)} for b in lap]})
    os.makedirs(LUMA, exist_ok=True)
    with open(os.path.join(LUMA, stem + ".jsonl"), "w") as fo:
        for r in out: fo.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--luma", action="store_true")
    if ap.parse_args().luma:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(8) as ex:
            list(ex.map(build_luma, stems()))
        print("luma done", len(stems()))
