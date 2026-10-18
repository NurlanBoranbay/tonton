#!/usr/bin/env python3
"""Per-frame facts for every video -> feats/<tag>/<video stem>.jsonl, one line per sampled frame:
{"f": frame, "t": sec, "W":w, "H":h,
 "persons": [{"id", "box":[x1,y1,x2,y2] (0..1), "conf", "kpts": [[x,y,conf]*17]}],   # --pose (tracked)
 "coco":    [{"c": name, "box", "conf"}],                                           # --coco
 "custom":  [{"c": name, "box", "conf"}]}                                           # --custom
Usage: python3 extract.py --tag base --stride 3 [--pose yolo26x-pose.pt] [--coco yolo26x.pt] [--custom best.pt] [--videos a,b]"""
import argparse, json, os, cv2
from ultralytics import YOLO

r4 = lambda v: round(float(v), 4)


def _boxes(res, W, H):
    return [{"c": res.names[int(c)], "box": [r4(b[0]/W), r4(b[1]/H), r4(b[2]/W), r4(b[3]/H)], "conf": r4(s)}
            for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]


def extract_rows(path, pose=None, coco=None, cust=None, stride=3, imgsz=960, conf=0.15, cust_conf=0.2):
    """Per-frame facts for one video (same row format as the jsonl files). Models are ultralytics YOLO objects."""
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows, f = [], -1
    if pose: pose.predictor = None
    while True:
        ok, img = cap.read(); f += 1
        if not ok: break
        if f % stride: continue
        H, W = img.shape[:2]; row = {"f": f, "t": round(f / fps, 3), "W": W, "H": H}
        if pose:
            r = pose.track(img, persist=True, imgsz=imgsz, conf=conf, classes=[0], verbose=False, tracker="bytetrack.yaml")[0]
            ids = r.boxes.id.tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            kp = r.keypoints.data.tolist() if r.keypoints is not None else [[]] * len(r.boxes)
            row["persons"] = [{"id": int(i) if i is not None else -1, "box": [r4(b[0]/W), r4(b[1]/H), r4(b[2]/W), r4(b[3]/H)],
                               "conf": r4(s), "kpts": [[r4(x/W), r4(y/H), r4(c)] for x, y, c in k]}
                              for b, s, i, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), ids, kp)]
        if coco: row["coco"] = [b for b in _boxes(coco(img, imgsz=imgsz, conf=conf, verbose=False)[0], W, H) if b["c"] != "person"]
        if cust: row["custom"] = _boxes(cust(img, imgsz=imgsz, conf=cust_conf, verbose=False)[0], W, H)
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pose"); ap.add_argument("--coco"); ap.add_argument("--custom")
    ap.add_argument("--videos", default=""); ap.add_argument("--vdir", default="/nlp/nurlan/tonton/dataset/videos")
    ap.add_argument("--imgsz", type=int, default=960); ap.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args()
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feats", a.tag); os.makedirs(OUT, exist_ok=True)
    pose = YOLO(a.pose) if a.pose else None
    coco = YOLO(a.coco) if a.coco else None
    cust = YOLO(a.custom) if a.custom else None
    vids = a.videos.split(",") if a.videos else sorted(v for v in os.listdir(a.vdir) if v.endswith(".mov"))
    r4 = lambda v: round(float(v), 4)
    def boxes(res, W, H):
        out = []
        for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            out.append({"c": res.names[int(c)], "box": [r4(b[0]/W), r4(b[1]/H), r4(b[2]/W), r4(b[3]/H)], "conf": r4(s)})
        return out
    for v in vids:
        stem = os.path.splitext(v)[0]; path = os.path.join(OUT, stem + ".jsonl")
        if os.path.exists(path): continue
        cap = cv2.VideoCapture(os.path.join(a.vdir, v)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
        rows, f = [], -1
        if pose: pose.predictor = None   # reset tracker state per video
        while True:
            ok, img = cap.read(); f += 1
            if not ok: break
            if f % a.stride: continue
            H, W = img.shape[:2]; row = {"f": f, "t": round(f / fps, 3), "W": W, "H": H}
            if pose:
                r = pose.track(img, persist=True, imgsz=a.imgsz, conf=a.conf, classes=[0], verbose=False, tracker="bytetrack.yaml")[0]
                ps = []
                ids = r.boxes.id.tolist() if r.boxes.id is not None else [None] * len(r.boxes)
                kp = r.keypoints.data.tolist() if r.keypoints is not None else [[]] * len(r.boxes)
                for b, s, i, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), ids, kp):
                    ps.append({"id": int(i) if i is not None else -1, "box": [r4(b[0]/W), r4(b[1]/H), r4(b[2]/W), r4(b[3]/H)],
                               "conf": r4(s), "kpts": [[r4(x/W), r4(y/H), r4(c)] for x, y, c in k]})
                row["persons"] = ps
            if coco: row["coco"] = [b for b in boxes(coco(img, imgsz=a.imgsz, conf=a.conf, verbose=False)[0], W, H) if b["c"] != "person"]
            if cust: row["custom"] = boxes(cust(img, imgsz=a.imgsz, conf=a.conf, verbose=False)[0], W, H)
            rows.append(row)
        with open(path + ".tmp", "w") as fo:
            for row in rows: fo.write(json.dumps(row) + "\n")
        os.replace(path + ".tmp", path); print(v, len(rows), "frames", flush=True)
