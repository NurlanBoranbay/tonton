#!/usr/bin/env python3
"""Qwen3.8-27B (video) judges each whole video against the procedure + violation list. Saves raw JSON."""
import csv, json, os, re, sys, concurrent.futures as cf
from qvl import clip, ask
V = "/nlp/nurlan/tonton/dataset/videos"
TAG = sys.argv[1]; THINK = "think" in TAG; FPS = 2
OUT = f"/nlp/nurlan/tonton/exp/qvl/{TAG}.jsonl"
CLASSES = [(r["class_id"], r["name_ru"], r["visual_cue"]) for r in csv.DictReader(open("/nlp/nurlan/tonton/dataset/violation_classes.csv")) if r["selected"] == "1"]
PROMPT = """You are an auditor of bank cash-desk discipline. The video is an overhead security camera of one cash room:
a cashier works at a desk with a banknote counting machine, a laptop and a safe; clients and cash collectors (инкассаторы)
may enter through a glass door. People seen THROUGH the glass wall outside the room are not in the room.

Correct procedure for a client operation: (1) the cashier examines the client's ID document, (2) counts the cash in the
counting machine in front of the client, (3) records the operation on the laptop, (4) opens the safe and puts the money in,
then closes the safe, (5) the client leaves. When the cashier leaves the room: safe closed, key taken out of the safe lock,
no money / bags / personal items left on the desk, laptop locked. The door of the cash room is kept closed.
Only working equipment belongs on the desk (no drinks, bags, books, personal things).

Possible violations (class_id: name — what it looks like):
""" + "\n".join(f"{c}: {n} — {cue}" for c, n, cue in CLASSES) + """

Watch the whole video carefully. Report ONLY violations you can actually see; if everything is done correctly, report none.
Answer with JSON only:
{"violations": [{"class_id": "C..", "time_sec": <second when it starts>, "evidence": "<what is visible>"}]}"""
def one(v):
    try:
        c, rz, dt, u = ask(clip(os.path.join(V, v), fps=FPS, height=640), PROMPT, think=THINK, max_tokens=6000 if THINK else 1500, fps=FPS)
    except Exception as e:
        c, rz, dt, u = "", f"ERROR {e}", 0, {}
    m = re.search(r"\{.*\}", c, re.S)
    try: js = json.loads(m.group(0)) if m else {}
    except Exception: js = {}
    return {"video": v, "resp": c, "parsed": js, "sec": dt, "tokens": u.get("total_tokens")}
vids = sorted(x for x in os.listdir(V) if x.endswith(".mov"))
done = {json.loads(l)["video"] for l in open(OUT)} if os.path.exists(OUT) else set()
with cf.ThreadPoolExecutor(4) as ex, open(OUT, "a") as fo:
    for r in ex.map(one, [v for v in vids if v not in done]):
        fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush(); print(r["video"], r["sec"], len(r["parsed"].get("violations", [])), flush=True)
