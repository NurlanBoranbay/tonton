#!/usr/bin/env python3
"""Video -> result in the case's API format.

  result = Pipeline().analyze("/path/video.mov")
  {"verdict": "violation"|"no_violation",
   "violations": [{"type", "class_id", "timecode_sec", "description", "rule": {...}, "evidence_frames": [...]}],
   "facts": {...}}            # scene anchors + timings, for the UI / debugging (extra field, ignored by the jury)

Steps: (1) per-frame facts with YOLO26 pose+COCO+custom detector (extract.extract_rows), (2) state timeline +
rules (engine), (3) optional narrow VLM checks (laptop screen), (4) per violation: evidence JPEGs at the timecode,
rule citation + verbatim quote from dataset/violation_classes.csv.
"""
import base64, csv, io, os, subprocess, time
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "dataset")
WEIGHTS = os.path.join(HERE, "weights")
DEFAULT_CUSTOM = os.path.join(HERE, "runs", "y26s_v1", "weights", "best.pt")

CLASSES = {r["class_id"]: r for r in csv.DictReader(open(os.path.join(DATA, "violation_classes.csv"), encoding="utf-8"))}


def frame_jpeg_b64(video, t, max_side=848, quality=85):
    b = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", video, "-frames:v", "1",
                        "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    if not b:
        return None
    img = Image.open(io.BytesIO(b)).convert("RGB")
    s = max_side / max(img.size)
    if s < 1:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def rule_of(cid):
    r = CLASSES.get(cid, {})
    return {"class_id": cid, "name_ru": r.get("name_ru", ""), "act": r.get("act", ""),
            "citation": r.get("citation", ""), "quote": r.get("quote", ""), "link": r.get("link", ""),
            "rule_match": r.get("rule_match", "")}


class Pipeline:
    def __init__(self, custom=DEFAULT_CUSTOM, device=0, vlm=True, stride=3):
        from ultralytics import YOLO
        self.pose = YOLO(os.path.join(WEIGHTS, "yolo26x-pose.pt"))
        self.coco = YOLO(os.path.join(WEIGHTS, "yolo26x.pt"))
        self.cust = YOLO(custom)
        for m in (self.pose, self.coco, self.cust):
            m.to(f"cuda:{device}" if device is not None else "cpu")
        self.vlm, self.stride = vlm, stride

    def analyze(self, video):
        import extract, engine
        t0 = time.time()
        rows = extract.extract_rows(video, self.pose, self.coco, self.cust, stride=self.stride)
        t1 = time.time()
        stem = os.path.splitext(os.path.basename(video))[0]
        v = engine.Video(stem, None, None, vdir=os.path.dirname(video), rows=rows)
        tl = v.timeline()
        try:
            ev = engine.rules(v, tl, vlm=self.vlm)
        except Exception:                       # VLM server down -> rules without the VLM check
            ev = engine.rules(v, tl, vlm=False)
        # one entry per (class, timecode); keep the earliest when the same class fires twice within 3 s
        ev.sort(key=lambda e: (e["class_id"], e["t"]))
        dedup = []
        for e in ev:
            if dedup and dedup[-1]["class_id"] == e["class_id"] and e["t"] - dedup[-1]["t"] < 3:
                continue
            dedup.append(e)
        dedup.sort(key=lambda e: e["t"])
        out = []
        for e in dedup:
            frames = []
            for dt in (0.0, 1.0):
                b = frame_jpeg_b64(video, e["t"] + dt)
                if b:
                    frames.append({"timestamp_sec": round(e["t"] + dt, 2), "image_base64": b})
            r = rule_of(e["class_id"])
            out.append({"type": r["name_ru"] or e["class_id"], "class_id": e["class_id"],
                        "timecode_sec": round(e["t"], 1), "interval_sec": [e["start"], e["end"]],
                        "description": e["why"], "rule": r, "evidence_frames": frames})
        return {"verdict": "violation" if out else "no_violation", "violations": out,
                "facts": {"door": v.door, "safe": v.safe, "desk": v.desk, "frames_analysed": len(rows),
                          "seconds": {"detection": round(t1 - t0, 1), "total": round(time.time() - t0, 1)}}}


if __name__ == "__main__":
    import json, sys
    res = Pipeline().analyze(sys.argv[1])
    for x in res["violations"]:
        for f in x["evidence_frames"]:
            f["image_base64"] = f["image_base64"][:20] + "..."
    print(json.dumps(res, ensure_ascii=False, indent=1))
