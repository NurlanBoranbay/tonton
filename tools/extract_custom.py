"""Custom-detector rows for chosen videos -> cache/feats/<tag>/<stem>.jsonl (same stride/frames as tonton's base feats).
  GPU=0 ./run.sh python3 tools/extract_custom.py <tag> <weights.pt> [imgsz] stem ..."""
import json, os, sys
import cv2
from ultralytics import YOLO
from app.pipeline.cache import ROOT, VIDEOS
from app.pipeline.extract import _boxes

tag, w = sys.argv[1], sys.argv[2]
imgsz = int(sys.argv[3]) if sys.argv[3].isdigit() else 960
stems = sys.argv[4:] if sys.argv[3].isdigit() else sys.argv[3:]
m = YOLO(w); out = os.path.join(ROOT, "cache", "feats", tag); os.makedirs(out, exist_ok=True)
for s in stems:
    cap = cv2.VideoCapture(os.path.join(VIDEOS, s + ".mov")); fps = cap.get(cv2.CAP_PROP_FPS) or 30; f = -1; rows = []
    while True:
        ok, img = cap.read(); f += 1
        if not ok: break
        if f % 3: continue
        H, W = img.shape[:2]
        rows.append({"f": f, "t": round(f / fps, 3), "custom": _boxes(m(img, imgsz=imgsz, conf=.2, verbose=False)[0], W, H)})
    with open(os.path.join(out, s + ".jsonl"), "w") as fo:
        for r in rows: fo.write(json.dumps(r) + "\n")
    print(tag, s, len(rows), flush=True)
