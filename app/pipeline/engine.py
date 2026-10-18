"""Video -> result in the case's API format.

  Pipeline().analyze("/path/video.mov") ->
  {"verdict": "violation" | "no_violation",
   "violations": [{"type", "timecode_sec", "description", "evidence_frames": [{"timestamp_sec", "image_base64"}],
                   "rule": {act, citation, quote, link, ...}, "debug": {"rule_id", "interval_sec", "evidence"}}],
   "facts": {...}}

detectors (extract.py) -> Timeline -> SceneCompiler -> evaluate_rules -> evidence JPEGs + legal citation.
"""
import base64, csv, io, os, subprocess, time

from app.config import load_settings
from app.pipeline.timeline import Timeline
from app.rules.base import REGISTRY, evaluate_rules
from app.rules.scene import SceneCompiler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.path.join(ROOT, "weights")
CLASSES = {r["class_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "violation_classes.csv"), encoding="utf-8"))}


def frame_jpeg_b64(video, t, max_side=848, quality=85):
    from PIL import Image
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


def legal_rule(class_id):
    r = CLASSES.get(class_id, {})
    return {"class_id": class_id, "name_ru": r.get("name_ru", ""), "act": r.get("act", ""), "citation": r.get("citation", ""),
            "quote": r.get("quote", ""), "link": r.get("link", ""), "rule_match": r.get("rule_match", "")}


def analyse(rows, name, settings=None):
    settings = settings or load_settings()
    tl = Timeline(rows, name)
    scene = SceneCompiler(settings["scene"]).compile(tl)
    return tl, scene, evaluate_rules(tl, scene, settings["rules"])


def to_result(video, tl, scene, alarms, with_frames=True):
    out = []
    for a in alarms:
        frames = []
        for dt in ((0.0, 1.0) if with_frames else ()):
            b = frame_jpeg_b64(video, a.timecode_sec + dt)
            if b: frames.append({"timestamp_sec": round(a.timecode_sec + dt, 2), "image_base64": b})
        d = a.as_dict()
        cid = REGISTRY[a.rule_id].class_id
        out.append({"type": a.type, "timecode_sec": d["timecode_sec"], "description": a.description,
                    "evidence_frames": frames, "rule": legal_rule(cid), "class_id": cid, "interval_sec": d["interval_sec"],
                    "debug": {"rule_id": a.rule_id, "interval_sec": d["interval_sec"], "evidence": d["evidence"]}})
    return {"verdict": "violation" if out else "no_violation", "violations": out,
            "facts": {"door": scene.door, "safe": scene.safe, "machine": scene.machine, "laptop": scene.laptop,
                      "departure_sec": round(scene.departure[0], 1) if scene.departure else None,
                      "counts_sec": [round(c.t0, 1) for c in scene.counts],
                      "roles": {str(p): t.role for p, t in scene.tracks.items() if t.n >= 10},
                      "frames_analysed": len(tl)}}


CUSTOM_WEIGHTS_FILE = os.path.join(ROOT, "service", "custom_weights.txt")   # ONE line: path to the custom detector


def custom_weights():
    """$CUSTOM_WEIGHTS, else the path in service/custom_weights.txt (read at every start), else weights/custom_y26s_all_stamp.pt."""
    try:
        line = open(CUSTOM_WEIGHTS_FILE).read().strip()
    except OSError:
        line = ""
    p = os.environ.get("CUSTOM_WEIGHTS") or line or os.path.join(WEIGHTS, "custom_y26s_all_stamp.pt")
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


class Pipeline:
    def __init__(self, device=0, stride=3, custom=None):
        custom = custom or custom_weights()
        print("custom detector:", custom, flush=True)
        from ultralytics import YOLO
        self.pose = YOLO(os.path.join(WEIGHTS, "yolo26x-pose.pt"))
        self.coco = YOLO(os.path.join(WEIGHTS, "yolo26x.pt"))
        self.cust = YOLO(custom)
        for m in (self.pose, self.coco, self.cust):
            m.to(f"cuda:{device}" if device is not None else "cpu")
        self.stride = stride

    def rows(self, video):
        from app.pipeline.extract import extract_rows
        return extract_rows(video, self.pose, self.coco, self.cust, stride=self.stride)

    def analyze(self, video, settings=None, detections_path=None):
        t0 = time.time()
        rows = self.rows(video)
        t1 = time.time()
        tl, scene, alarms = analyse(rows, os.path.splitext(os.path.basename(video))[0], settings)
        res = to_result(video, tl, scene, alarms)
        res["facts"]["seconds"] = {"detection": round(t1 - t0, 1), "total": round(time.time() - t0, 1)}
        if detections_path:
            save_detections(rows, detections_path)
        return res


def save_detections(rows, path):
    """Per-frame boxes for the UI overlay -> gzipped JSON. Coordinates are normalised to [0, 1].

    {"W", "H", "frames": [{"t", "p": [[track_id, x1, y1, x2, y2, [x, y, c] * 17]], "o": [[class, x1, y1, x2, y2, conf]]}]}
    Only detections the rules would use (CONF thresholds) are kept.
    """
    import gzip, json
    from app.pipeline.timeline import CONF, PERSON_CONF
    r3 = lambda v: round(v, 3)
    frames = []
    for r in rows:
        p = [[q["id"], *map(r3, q["box"]), [[r3(x), r3(y), round(c, 2)] for x, y, c in q["kpts"]]]
             for q in r.get("persons", []) if q["conf"] >= PERSON_CONF]
        o = [[b["c"], *map(r3, b["box"]), round(b["conf"], 2)]
             for b in r.get("custom", []) + r.get("coco", []) if b["conf"] >= CONF.get(b["c"], 1.1)]
        frames.append({"t": r["t"], "p": p, "o": o})
    W, H = (rows[0]["W"], rows[0]["H"]) if rows else (0, 0)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"W": W, "H": H, "frames": frames}, f, separators=(",", ":"))


if __name__ == "__main__":
    import json, sys
    res = Pipeline().analyze(sys.argv[1])
    for x in res["violations"]:
        for f in x["evidence_frames"]: f["image_base64"] = f["image_base64"][:20] + "..."
    print(json.dumps(res, ensure_ascii=False, indent=1))
