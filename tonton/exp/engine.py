#!/usr/bin/env python3
"""State timeline + rule engine.

Input: feats/<base_tag>/<video>.jsonl (persons + coco) and feats/<cust_tag>/<video>.jsonl (custom det).
Per sampled frame we build a state, smooth it over a ~1 s window, then rules turn sustained states
into violation candidates: {class_id, start, end, t (timecode), evidence: {...state that fired}}.

Every rule is a plain, readable condition on named states, so each verdict can be explained.
Usage:  python3 engine.py --base base --cust cust_s_v1 [--timeline Х-11] [--eval]
"""
import argparse, collections, csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "dataset")

PERSONAL = {"cup", "bottle", "wine glass", "cell phone", "handbag", "backpack", "book", "banana", "apple",
            "sandwich", "orange", "donut", "cake", "umbrella", "scissors", "teddy bear", "bowl", "suitcase",
            "remote", "toothbrush", "vase", "hair drier"}
CUST_CONF = {"money_stack": .35, "collection_bag": .35, "counting_machine": .4, "safe_open": .4,
             "safe_closed": .4, "key_in_lock": .3, "key_out_of_lock": .35, "door_open": .4,
             "door_closed": .4, "ID-identification": .3}


def load(tag, stem):
    p = os.path.join(HERE, "feats", tag, stem + ".jsonl")
    return {r["f"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def area(b): return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
def inter(a, b): return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
def expand(b, m): return [b[0] - m, b[1] - m, b[2] + m, b[3] + m]
def union(bs): return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


class Video:
    def __init__(self, stem, base, cust, vdir=None, rows=None):
        self.vdir = vdir or os.path.join(DATA, "videos")
        if rows is not None:                       # in-memory rows from extract.extract_rows (service path)
            B = C = {r["f"]: r for r in rows}
        else:
            B, C = load(base, stem), load(cust, stem)
        self.stem, self.frames = stem, sorted(set(B) & set(C)) if C else sorted(B)
        self.B, self.C = B, C
        # Static scene anchors: median boxes of objects that do not move (robust to camera angle).
        def med(cls_set, src="custom"):
            bs = [d["box"] for f in self.frames for d in (C if src == "custom" else B).get(f, {}).get(src, [])
                  if d["c"] in cls_set and d["conf"] >= .4]
            if not bs: return None
            return [sorted(x[i] for x in bs)[len(bs) // 2] for i in range(4)]
        self.door = med({"door_open", "door_closed"})
        self.safe = med({"safe_open", "safe_closed"})
        self.machine = med({"counting_machine"})
        self.laptop = med({"laptop"}, "coco")
        anchors = [b for b in (self.machine, self.laptop) if b]
        self.desk = expand(union(anchors), .08) if anchors else None
        self.track_max = collections.defaultdict(float)     # largest box height per person track
        for f in self.frames:
            for p in B.get(f, {}).get("persons", []):
                if p["conf"] >= .4 and p["id"] >= 0:
                    self.track_max[p["id"]] = max(self.track_max[p["id"]], p["box"][3] - p["box"][1])

    def outside(self, p):
        """Person seen through the glass wall/door, not in the room.
        Outside people stand metres behind the glass: their box lies in the glass/door area, is short
        relative to the door (<0.38 of its height) and ends above the top of the counting machine (an
        in-room person behind the desk always reaches down to the desk/machine level)."""
        b = p["box"]
        if self.door is None:
            return False
        if inter(b, expand(self.door, .05)) / max(area(b), 1e-6) <= .5:
            return False
        short = (b[3] - b[1]) < .38 * (self.door[3] - self.door[1])
        above_desk = self.machine is not None and b[3] < self.machine[1] + .01
        return short and above_desk

    def state(self, f):
        b, c = self.B.get(f, {}), self.C.get(f, {})
        ps = [p for p in b.get("persons", []) if p["conf"] >= .4]
        inroom = [p for p in ps if not self.outside(p)]
        cu = collections.defaultdict(list)
        for d in c.get("custom", []):
            if d["conf"] >= CUST_CONF.get(d["c"], .35): cu[d["c"]].append(d["box"])
        on_desk = lambda bx: self.desk is not None and inter(bx, self.desk) / max(area(bx), 1e-6) > .5
        pers = sorted({d["c"] for d in b.get("coco", []) if d["c"] in PERSONAL and d["conf"] >= .35 and on_desk(d["box"])})
        near_door = sum(1 for p in inroom if self.door and inter(p["box"], expand(self.door, .02)) > 0)
        return {
            "t": b.get("t", c.get("t")), "n_room": len(inroom), "n_out": len(ps) - len(inroom), "near_door": near_door,
            "money": len(cu["money_stack"]), "money_desk": sum(on_desk(x) for x in cu["money_stack"]),
            "bag": len(cu["collection_bag"]), "safe_open": int(bool(cu["safe_open"])),
            "safe_closed": int(bool(cu["safe_closed"])), "key_in": int(bool(cu["key_in_lock"])),
            "key_out": int(bool(cu["key_out_of_lock"])), "door_open": int(bool(cu["door_open"])),
            "door_closed": int(bool(cu["door_closed"])), "id": int(bool(cu["ID-identification"])),
            "personal": pers,
        }

    def timeline(self, win=5):
        """Raw states, then majority/median smoothing over ±win samples (stride 3 → ±0.5 s)."""
        raw = [self.state(f) for f in self.frames]
        out = []
        for i, s in enumerate(raw):
            w = raw[max(0, i - win): i + win + 1]
            sm = {"t": s["t"]}
            for k, v in s.items():
                if k == "t": continue
                if k == "personal":
                    cnt = collections.Counter(x for r in w for x in r[k])
                    sm[k] = sorted(x for x, n in cnt.items() if n > len(w) / 2)
                else:
                    sm[k] = sorted(r[k] for r in w)[len(w) // 2]
            out.append(sm)
        return out


def runs(tl, cond, min_dur):
    """Maximal intervals where cond(state) holds for at least min_dur seconds."""
    out, start = [], None
    for i, s in enumerate(tl + [None]):
        ok = s is not None and cond(s)
        if ok and start is None: start = i
        if not ok and start is not None:
            a, b = tl[start]["t"], tl[i - 1]["t"]
            if b - a >= min_dur: out.append((a, b, start, i - 1))
            start = None
    return out


DRINKS = {"cup", "bottle", "wine glass"}
RU = {"book": "книга", "handbag": "сумка", "backpack": "рюкзак", "cup": "чашка", "bottle": "бутылка", "wine glass": "бокал",
      "umbrella": "зонт", "suitcase": "чемодан"}
LEFT_ITEMS = {"book", "handbag", "backpack", "cup", "bottle", "wine glass", "umbrella", "suitcase"}


def hip_dist(v, f):
    """Distance (in desk widths) from the main in-room person's hips to the desk centre; None if nobody."""
    ps = [p for p in v.B.get(f, {}).get("persons", []) if p["conf"] >= .4 and not v.outside(p)]
    if not ps or not v.desk: return None
    p = max(ps, key=lambda p: p["box"][3] - p["box"][1]); k = p["kpts"]
    if len(k) < 13: return None
    hx, hy = (k[11][0] + k[12][0]) / 2, (k[11][1] + k[12][1]) / 2
    cx, cy = (v.desk[0] + v.desk[2]) / 2, (v.desk[1] + v.desk[3]) / 2
    return ((hx - cx) ** 2 + (hy - cy) ** 2) ** .5 / (v.desk[2] - v.desk[0])


def rules(v, tl, P=None, vlm=False):
    """Violation candidates from named, sustained states. P = thresholds (seconds)."""
    P = {"door": 8.0, "safe": 8.0, "drink": 2.0, "leave_min": .5, "work_zone": .17, "crowd": 2.0, "visit": 5.0, "after": 4.0, "money_share": .6, "return_min": 10, "visit_id": 3.0, **(P or {})}
    ev = []
    def add(cid, t, a, b, why):
        ev.append({"class_id": cid, "t": round(t, 2), "start": round(a, 2), "end": round(b, 2), "why": why})
    T = [s["t"] for s in tl]
    # 1. Cashier leaves the room: judge what was left behind while the room is empty.
    raw = {f: v.state(f) for f in v.frames}
    for a, b, i, j in runs(tl, lambda s: s["n_room"] == 0, P["leave_min"]):
        if i == 0: continue                                   # empty from the start: nobody left anything
        fr = [f for f in v.frames if a <= raw[f]["t"] <= b]
        frac = lambda k: sum(bool(raw[f][k]) for f in fr) / max(len(fr), 1)
        items = sorted({x for f in fr for x in raw[f]["personal"]} & LEFT_ITEMS)
        # departure = last moment the cashier was in the work zone (desk/safe) before the room emptied
        td = a
        for f in reversed([f for f in v.frames if raw[f]["t"] < a]):
            d = hip_dist(v, f)
            if d is not None and d <= P["work_zone"]:
                td = raw[f]["t"]; break
        left = []
        if frac("key_in") >= .5: add("C11", td, td, b, "Кассир покинул кассовое помещение, ключ остался в замке сейфа"); left.append("key")
        if frac("money_desk") >= .5: add("C08", td, td, b, "Кассир покинул кассовое помещение, на столе остались деньги"); left.append("money")
        if frac("bag") >= .5: add("C08", td, td, b, "Кассир покинул кассовое помещение, инкассаторская сумка осталась открыто на месте"); left.append("bag")
        if frac("safe_open") >= .5: add("C10", td, td, b, "Кассир покинул кассовое помещение, сейф остался открытым")
        if items: add("C12", td, td, b, "Кассир покинул кассовое помещение, на рабочем столе остались посторонние предметы: " + ", ".join(RU.get(x, x) for x in items))
        if vlm and v.laptop:                                  # screen state needs pixels: narrow VLM question
            from screen import frame
            from vlm_checks import laptop_unlocked
            ans, txt = laptop_unlocked(frame(os.path.join(v.vdir, v.stem + ".mov"), (a + b) / 2), v.laptop)
            if ans == "unlocked": add("C17", td, td, b, "Кассир покинул кассовое помещение, на экране ноутбука открыто рабочее приложение (компьютер не заблокирован)")
    # 2. Door open for a long time (people only pass through it).
    for a, b, i, j in runs(tl, lambda s: s["door_open"] and not s["door_closed"], P["door"]):
        add("C16", a, a, b, f"Дверь кассового помещения остаётся открытой {b - a:.0f} с")
    # 3. Safe open for a long time (should be closed right after taking/putting money).
    for a, b, i, j in runs(tl, lambda s: s["safe_open"] and not s["safe_closed"], P["safe"]):
        add("C10", a, a, b, f"Сейф остаётся открытым {b - a:.0f} с")
    # 4. A third person in the cash room (cashier + client is the maximum).
    for a, b, i, j in runs(tl, lambda s: s["n_room"] >= 3, P["crowd"]):
        add("C14", a, a, b, f"В кассовом помещении одновременно находятся {tl[i]['n_room']} человека")
    # 5. A client leaves but money stays on the desk (not put in the safe) -> next client's money gets mixed in.
    for a, b, i, j in runs(tl, lambda s: s["n_room"] >= 2, P["visit"]):
        after = tl[j + 1: j + 1 + int(P["after"] * 10)]
        if after and sum(s["money_desk"] > 0 for s in after) / len(after) >= P["money_share"] and not any(s["safe_open"] for s in after):
            if any(s["n_room"] >= 2 for s in after):
                add("C03", b, b, after[-1]["t"], "Клиент ушёл, деньги не убраны в сейф; следующий клиент обслуживается, пока деньги предыдущего остаются на столе")
            else:
                add("C07", b, b, after[-1]["t"], "Клиент ушёл, деньги остаются на столе и не убраны в сейф")
    # 6. Cash return (2026-10-17): after the cashier counts in the machine, cash must go to the safe;
    #    cash appearing in the hands of another person before the safe opens = money handed back.
    from cashflow import cash_flow
    cf = cash_flow(v)
    if cf and cf.get("count_start") is not None:
        cash = [t for t, w in cf["returns"] if w == "cash"]
        if len(cash) >= P["return_min"]:
            add("C05", cash[0], cash[0], cash[-1], "После пересчёта на счётной машине наличные снова оказываются в руках клиента до того, как деньги убраны в сейф")
    # 7. C01: cash accepted / counted without the client's ID having been shown (п.19 пп.1).
    from cashflow import no_id_before_count
    for a, b, i, j in runs(tl, lambda s: s["n_room"] >= 2, P["visit_id"]):
        r = no_id_before_count(v, a)
        if r:
            add("C01", r[0], r[0], r[1], "Кассир принимает и пересчитывает наличные, но удостоверение личности клиента не было предъявлено")
            break
    # 8. Drinks on the cash desk.
    for a, b, i, j in runs(tl, lambda s: bool(set(s["personal"]) & DRINKS), P["drink"]):
        add("C12", a, a, b, "Напиток на рабочем столе кассира: " + ", ".join(RU.get(x, x) for x in sorted(set(tl[i]["personal"]) & DRINKS)))
    return ev


def labels():
    return {r["file"][:-4]: (int(r["violation"]), float(r["timestamp_sec"]) if r["timestamp_sec"] else None)
            for r in csv.DictReader(open(os.path.join(DATA, "labels.csv"), encoding="utf-8"))}


def evaluate(base, cust, verbose=True, vlm=False, vdir=None):
    L = labels(); res = []
    types = {r["file"][:-4]: r for r in csv.DictReader(open(os.path.join(DATA, "violations.csv"), encoding="utf-8"))}
    for stem, (y, t) in sorted(L.items()):
        v = Video(stem, base, cust, vdir)
        if not v.frames: continue
        ev = rules(v, v.timeline(), vlm=vlm)
        pred = int(bool(ev))
        best = min(ev, key=lambda e: abs(e["t"] - t)) if (ev and t is not None) else None
        res.append({"video": stem, "y": y, "pred": pred, "t": t,
                    "dt": None if best is None else round(best["t"] - t, 1),
                    "classes": sorted({e["class_id"] for e in ev}), "ev": ev})
    tp = sum(r["y"] and r["pred"] for r in res); fp = sum((not r["y"]) and r["pred"] for r in res)
    fn = sum(r["y"] and not r["pred"] for r in res); tn = sum((not r["y"]) and not r["pred"] for r in res)
    if verbose:
        for r in res:
            mark = "TP" if r["y"] and r["pred"] else "FP" if r["pred"] else "FN" if r["y"] else "TN"
            print(f"{r['video']:6} {mark} label_t={r['t']} dt={r['dt']} pred={r['classes']}")
    acc = (tp + tn) / len(res)
    within = sum(1 for r in res if r["y"] and r["pred"] and r["dt"] is not None and abs(r["dt"]) <= 2)
    print(f"videos {len(res)}  acc {acc:.3f}  TP {tp} FP {fp} FN {fn} TN {tn}  "
          f"recall {tp / max(tp + fn, 1):.2f}  precision {tp / max(tp + fp, 1):.2f}  |dt|<=2s {within}/{tp}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base"); ap.add_argument("--cust", default="cust_s_v1")
    ap.add_argument("--timeline"); ap.add_argument("--eval", action="store_true"); ap.add_argument("--vlm", action="store_true"); ap.add_argument("--vdir")
    a = ap.parse_args()
    if a.timeline:
        v = Video(a.timeline, a.base, a.cust)
        print("door", v.door, "safe", v.safe, "desk", v.desk)
        last = None
        for s in v.timeline():
            key = {k: s[k] for k in s if k != "t"}
            if key != last:
                print(f"{s['t']:6.1f}  " + " ".join(f"{k}={v}" for k, v in key.items() if v not in (0, [], None)))
                last = key
        for e in rules(v, v.timeline()): print(e)
    if a.eval:
        evaluate(a.base, a.cust, vlm=a.vlm, vdir=a.vdir)
