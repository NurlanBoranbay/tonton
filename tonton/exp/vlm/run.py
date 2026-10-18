"""VLM baseline: show Gemma 4 a video as tiled frame grids, ask for violations as JSON.

Frames come from frames/<video>/%05d.jpg (ffmpeg fps=2, frame i -> t=(i-1)/2 s).
Usage: python3 run.py --design W12 --prompt P1 [--think] [--videos X-1,Х-4]
Raw responses -> raw/<design>_<prompt>[_think].jsonl (one line per call, resumable).
"""
import argparse, base64, csv, io, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = "/nlp/nurlan/tonton/dataset"
URL = "http://localhost:8887/v1/chat/completions"
MODEL = "/models/gemma-4-26B-A4B-it"
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)

# design: tiles per grid, fps of tiles, columns, tile size, stride (s); OV = whole-video overview
DESIGNS = {
    "W12": dict(n=12, fps=1, cols=6, tw=232, th=424, stride=6),    # 12 s window, 1 fps
    "W16": dict(n=16, fps=2, cols=8, tw=174, th=318, stride=4),    # 8 s window, 2 fps
    "W12c": dict(n=12, fps=1, cols=6, tw=290, th=329, stride=6, crop=0.62),  # as W12, top 62% only (action area)
    "CAND": dict(n=12, fps=1, cols=6, tw=232, th=424, stride=None, cand=True),  # one 12 s window per video centred on a candidate time (label ts / clean: middle)
    "OV":  dict(n=24, fps=None, cols=8, tw=174, th=318, stride=None),  # 24 tiles over the whole video
}

def classes():
    rows = [r for r in csv.DictReader(open(f"{DS}/violation_classes.csv")) if r["selected"] == "1"]
    out = []
    for r in rows:
        note = f" ({r['selection_note']})" if r["selection_note"] else ""
        out.append(f"- {r['class_id']} {r['name_en']}: {r['visual_cue']}{note}")
    return "\n".join(out)

SEQ = """Correct procedure for a cash deposit:
1. Cashier examines the client's ID document.
2. Counts the cash through the counting machine.
3. Records the operation on the laptop.
4. Opens the safe and puts the money in.
5. Client leaves.
Notes: foreign objects and unrelated actions can be in frame; the money on the desk is not always a
client's money; without a client, several piles of money can lie on the desk - that alone is not a violation."""

def prompt(kind, t0, t1, n, overview):
    scope = (f"The image is an overview of the WHOLE video ({n} frames sampled evenly from 0 to {t1:.0f} s)."
             if overview else
             f"The image shows {n} consecutive frames covering seconds {t0:.1f}-{t1:.1f} of a longer video.")
    p = f"""You are auditing cash-discipline at a bank cash desk. The image is a grid of frames from a fixed
overhead CCTV camera in a small cash room (cashier desk with banknote counting machine and laptop, a safe,
a door). One cashier plus, during an operation, one client at the desk is normal. Frames are in reading order (left to right, top to bottom); each tile has its timestamp in seconds
burned into the top-left corner. {scope}

Violation classes:
{classes()}
"""
    if kind in ("P1",):
        p += "\n" + SEQ + "\n"
    p += """
Report only violations you can actually see in these frames; do not guess. If everything looks like normal
work, return an empty list. Use the burned-in timestamps for start_sec/end_sec.
Answer with strict JSON only, no prose:
{"violations": [{"class_id": "Cxx", "start_sec": <number>, "end_sec": <number>, "evidence": "<what is visible>"}]}"""
    return p

def frames_of(video):
    d = f"{ROOT}/frames/{video}"
    return [f"{d}/{f}" for f in sorted(os.listdir(d))]

def grid(paths_ts, cols, tw, th, crop=1.0):
    rows = (len(paths_ts) + cols - 1) // cols
    g = Image.new("RGB", (cols * tw, rows * th), "black")
    for i, (p, t) in enumerate(paths_ts):
        im = Image.open(p).convert("RGB")
        im = im.crop((0, 0, im.width, round(im.height * crop))).resize((tw, th))
        dr = ImageDraw.Draw(im)
        s = f"{t:.1f}s" if t % 1 else f"{t:.0f}s"
        dr.rectangle([0, 0, 8 + 14 * len(s), 28], fill="black")
        dr.text((4, 2), s, fill="yellow", font=FONT)
        g.paste(im, ((i % cols) * tw, (i // cols) * th))
    return g

def windows(video, design):
    D = DESIGNS[design]
    fr = frames_of(video)  # 2 fps
    dur = len(fr) / 2
    if D.get("cand"):
        lab = {r["file"][:-4]: r["timestamp_sec"] for r in csv.DictReader(open(f"{DS}/labels.csv"))}
        c = float(lab[video]) if lab[video] else dur / 2
        s = int(max(0, min(c - 6, dur - 12)))
        idx = list(range(s * 2, len(fr), 2))[:12]
        yield 0, (s, idx[-1] / 2), [(fr[i], i / 2) for i in idx]
        return
    if D["fps"] is None:
        idx = [round(i * (len(fr) - 1) / (D["n"] - 1)) for i in range(D["n"])]
        yield 0, (0.0, dur), [(fr[i], i / 2) for i in idx]
        return
    step = 2 // D["fps"]
    span = D["n"] / D["fps"]
    starts = list(range(0, max(1, int(dur - span) + 1), D["stride"]))
    if starts[-1] + span < dur - 1:
        starts.append(max(0, int(dur - span)))
    for k, s in enumerate(starts):
        idx = [i for i in range(s * 2, len(fr), step)][: D["n"]]
        yield k, (s, idx[-1] / 2), [(fr[i], i / 2) for i in idx]

def parse(txt):
    """Last {"violations": [...]} object in the text (thinking output is inline before it)."""
    if txt is None:
        return None
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r'\{\s*"violations"', txt))):
        try:
            v = dec.raw_decode(txt[m.start():])[0].get("violations")
            if isinstance(v, list):
                return v
        except Exception:
            pass
    return None

def call(img, text, think):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=88)
    body = {"model": MODEL, "temperature": 0, "max_tokens": 6000 if think else 800,
            "mm_processor_kwargs": {"max_soft_tokens": 1120},
            "chat_template_kwargs": {"enable_thinking": think},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": text}]}]}
    for a in range(3):
        try:
            t = time.time()
            r = requests.post(URL, json=body, timeout=600).json()
            m = r["choices"][0]["message"]
            return m.get("content"), m.get("reasoning") or m.get("reasoning_content"), r.get("usage"), time.time() - t
        except Exception as e:
            err = e; time.sleep(3)
    return None, f"ERROR {err}", None, 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True); ap.add_argument("--prompt", default="P1")
    ap.add_argument("--think", action="store_true"); ap.add_argument("--videos", default="")
    ap.add_argument("--save-grids", action="store_true")
    a = ap.parse_args()
    D = DESIGNS[a.design]
    tag = f"{a.design}_{a.prompt}" + ("_think" if a.think else "")
    os.makedirs(f"{ROOT}/raw", exist_ok=True)
    out = f"{ROOT}/raw/{tag}.jsonl"
    done = set()
    if os.path.exists(out):
        done = {(j["video"], j["win"]) for j in map(json.loads, open(out)) if j["content"] is not None}
    vids = [r["file"][:-4] for r in csv.DictReader(open(f"{DS}/labels.csv"))]
    if a.videos:
        vids = [v for v in vids if v in a.videos.split(",")]
    jobs = []
    for v in vids:
        for k, (t0, t1), pt in windows(v, a.design):
            if (v, k) not in done:
                jobs.append((v, k, t0, t1, pt))
    print(tag, len(jobs), "calls", flush=True)

    def work(j):
        v, k, t0, t1, pt = j
        g = grid(pt, D["cols"], D["tw"], D["th"], D.get("crop", 1.0))
        if a.save_grids:
            os.makedirs(f"{ROOT}/grids/{a.design}", exist_ok=True); g.save(f"{ROOT}/grids/{a.design}/{v}_{k:03d}.jpg")
        content, reasoning, usage, sec = call(g, prompt(a.prompt, t0, t1, len(pt), D["fps"] is None), a.think)
        return dict(video=v, win=k, t0=t0, t1=t1, content=content, reasoning=reasoning, usage=usage,
                    sec=round(sec, 2), violations=parse(content))

    with ThreadPoolExecutor(4) as ex, open(out, "a") as f:
        for i, r in enumerate(ex.map(work, jobs)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 50 == 0:
                print(i, r["video"], r["win"], r["sec"], (r["content"] or "")[:120].replace("\n", " "), flush=True)

if __name__ == "__main__":
    main()
