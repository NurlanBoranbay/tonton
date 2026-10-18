#!/usr/bin/env python3
"""Box annotation tool for the ForteBank videos -> YOLO dataset.

Serves annotate.html, the videos (with HTTP Range, so the browser can seek) and a small
JSON API. Every save writes one frame:

  <out>/images/<stem>.jpg   the frame exactly as the annotator saw it (native resolution)
  <out>/labels/<stem>.txt   YOLO format: "<class> <cx> <cy> <w> <h>", normalised 0..1;
                            an empty file = a frame with no objects (negative sample)
  <out>/annotations.jsonl   one line per save: video, frame, time, boxes, user, saved_at
                            (the last line for a stem wins)
  <out>/data.yaml, classes.txt

<stem> = <video name with Cyrillic "Х" written as Latin "X">_f<frame:06d>, e.g. X-14_f000123.
X-1/X-2 (Latin) and Х-3.. (Cyrillic) have different numbers, so nothing collides.

Access: if ANNOTATOR_PASSWORD is set, every request needs HTTP Basic Auth (any user name,
that password). Set it whenever the port is exposed through a tunnel.

Stdlib only. Usage: ANNOTATOR_PASSWORD=... python3 serve.py [--port 8097] [--videos DIR] [--labels CSV] [--out DIR]
"""
import argparse
import base64
import csv
import hmac
import json
import os
import re
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "dataset")

# Custom box classes (object_classes.csv, team decision 2026-10-17). Order = YOLO class id.
CLASSES = [
    ("money_stack", "пачка денег"),
    ("collection_bag", "инкассаторская сумка"),
    ("counting_machine", "машина для счёта денег"),
    ("safe_open", "открытый сейф"),
    ("safe_closed", "закрытый сейф"),
    ("key_in_lock", "ключ в скважине"),
    ("key_out_of_lock", "ключ вне скважины"),
    ("door_open", "открытая дверь"),
    ("door_closed", "закрытая дверь"),
    ("ID-identification", "удостоверение личности (показ / проверка)"),
]

LOCK = threading.Lock()


def stem_for(video, frame):
    base = os.path.splitext(video)[0].replace("Х", "X").replace("х", "x")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return f"{base}_f{int(frame):06d}"


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,nb_frames,width,height:format=duration", "-of", "json", path],
        capture_output=True, text=True).stdout
    d = json.loads(out or "{}")
    s = (d.get("streams") or [{}])[0]
    num, den = (s.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den or 1)
    dur = float((d.get("format") or {}).get("duration") or 0)
    frames = int(s.get("nb_frames") or round(dur * fps))
    return {"fps": fps, "frames": frames, "duration": dur,
            "width": s.get("width"), "height": s.get("height")}


class State:
    def __init__(self, videos, labels_csv, out):
        self.videos_dir, self.out = videos, out
        os.makedirs(os.path.join(out, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, "labels"), exist_ok=True)
        with open(os.path.join(out, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(c for c, _ in CLASSES) + "\n")
        with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"# written by annotator/serve.py; split images into train/val before training\n"
                    f"path: {os.path.abspath(out)}\ntrain: images\nval: images\n"
                    f"nc: {len(CLASSES)}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, (c, _) in enumerate(CLASSES)))
        labels = {}
        if labels_csv and os.path.exists(labels_csv):
            for r in csv.DictReader(open(labels_csv, encoding="utf-8")):
                labels[r["file"]] = {"violation": int(r["violation"]),
                                     "t": float(r["timestamp_sec"]) if r["timestamp_sec"] else None}
        self.videos = {}
        for name in sorted(os.listdir(videos), key=lambda n: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", n)]):
            if name.lower().endswith((".mov", ".mp4")):
                self.videos[name] = probe(os.path.join(videos, name)) | labels.get(name, {"violation": None, "t": None})
        self.ann = {}  # stem -> last record
        p = os.path.join(out, "annotations.jsonl")
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if r.get("deleted"):
                    self.ann.pop(r["stem"], None)
                else:
                    self.ann[r["stem"]] = r

    def log(self, rec):
        with open(os.path.join(self.out, "annotations.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def save(self, d):
        video, frame = d["video"], int(d["frame"])
        if video not in self.videos:
            raise ValueError("unknown video")
        stem = stem_for(video, frame)
        img = base64.b64decode(d["image"].split(",", 1)[-1])
        lines = []
        for b in d.get("boxes", []):
            c = int(b["c"])
            x, y, w, h = (min(max(float(b[k]), 0.0), 1.0) for k in ("x", "y", "w", "h"))
            if 0 <= c < len(CLASSES) and w > 0 and h > 0:
                lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        with LOCK:
            for sub, data, mode in (("images", img, "wb"), ("labels", "\n".join(lines) + ("\n" if lines else ""), "w")):
                path = os.path.join(self.out, sub, stem + (".jpg" if sub == "images" else ".txt"))
                tmp = path + ".tmp"
                with open(tmp, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
                    f.write(data)
                os.replace(tmp, path)
            rec = {"stem": stem, "video": video, "frame": frame, "time": float(d.get("time", 0)),
                   "boxes": d.get("boxes", []), "user": (d.get("user") or "")[:40],
                   "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.ann[stem] = rec
            self.log(rec)
        return rec

    def delete(self, video, frame):
        stem = stem_for(video, frame)
        with LOCK:
            for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
                p = os.path.join(self.out, sub, stem + ext)
                if os.path.exists(p):
                    os.remove(p)
            self.ann.pop(stem, None)
            self.log({"stem": stem, "video": video, "frame": int(frame), "deleted": True,
                      "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    def stats(self):
        per_class = [0] * len(CLASSES)
        empty = 0
        for r in self.ann.values():
            if not r["boxes"]:
                empty += 1
            for b in r["boxes"]:
                if 0 <= int(b["c"]) < len(CLASSES):
                    per_class[int(b["c"])] += 1
        return {"images": len(self.ann), "empty_images": empty, "per_class": per_class}


def make_handler(st, password):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def authorized(self):
            if not password:
                return True
            h = self.headers.get("Authorization", "")
            if h.startswith("Basic "):
                try:
                    _, _, pw = base64.b64decode(h[6:]).decode().partition(":")
                except Exception:
                    pw = ""
                if hmac.compare_digest(pw, password):
                    return True
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="annotator", charset="UTF-8"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def send_json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if not self.authorized():
                return
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                b = open(os.path.join(HERE, "annotate.html"), "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/api/config":
                vids = []
                for name, v in st.videos.items():
                    saved = sorted(r["frame"] for r in st.ann.values() if r["video"] == name)
                    vids.append({"name": name, **v, "saved": saved})
                self.send_json({"classes": [{"id": i, "name": c, "ru": ru} for i, (c, ru) in enumerate(CLASSES)],
                                "videos": vids, "stats": st.stats(), "out": os.path.abspath(st.out)})
            elif u.path == "/api/ann":
                r = st.ann.get(stem_for(q.get("video", ""), int(q.get("frame", 0))))
                self.send_json(r or {})
            elif u.path == "/api/last":
                rs = [r for r in st.ann.values() if r["video"] == q.get("video") and r["frame"] < int(q.get("frame", 0))]
                self.send_json(max(rs, key=lambda r: r["frame"]) if rs else {})
            elif u.path == "/review":
                b = open(os.path.join(HERE, "review.html"), "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/api/all":
                self.send_json({"classes": [c for c, _ in CLASSES],
                                "items": sorted(st.ann.values(), key=lambda r: (r["video"], r["frame"]))})
            elif u.path.startswith("/img/"):
                stem = os.path.basename(unquote(u.path[len("/img/"):]))
                path = os.path.join(st.out, "images", stem + ".jpg")
                if not os.path.exists(path):
                    return self.send_error(404)
                b = open(path, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/api/stats":
                self.send_json(st.stats())
            elif u.path.startswith("/video/"):
                self.send_video(unquote(u.path[len("/video/"):]))
            else:
                self.send_error(404)

        def send_video(self, name):
            if name not in st.videos:
                return self.send_error(404)
            path = os.path.join(st.videos_dir, name)
            size = os.path.getsize(path)
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else end
                else:
                    start = size - int(m.group(2))
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_POST(self):
            if not self.authorized():
                return
            u = urlparse(self.path)
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if u.path == "/api/save":
                    rec = st.save(d)
                    self.send_json({"ok": True, "stem": rec["stem"], "stats": st.stats()})
                elif u.path == "/api/delete":
                    st.delete(d["video"], int(d["frame"]))
                    self.send_json({"ok": True, "stats": st.stats()})
                else:
                    self.send_error(404)
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)}, 400)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--videos", default=os.path.join(DATASET, "videos"))
    ap.add_argument("--labels", default=os.path.join(DATASET, "labels.csv"))
    ap.add_argument("--out", default=os.path.join(DATASET, "yolo"))
    a = ap.parse_args()
    st = State(a.videos, a.labels, a.out)
    print(f"{len(st.videos)} videos, {len(st.ann)} annotated frames -> {os.path.abspath(a.out)}")
    print(f"open http://<this-server>:{a.port}/")
    password = os.environ.get("ANNOTATOR_PASSWORD", "")
    print("password protection:", "on" if password else "OFF (set ANNOTATOR_PASSWORD before exposing the port)")
    ThreadingHTTPServer((a.host, a.port), make_handler(st, password)).serve_forever()


if __name__ == "__main__":
    main()
