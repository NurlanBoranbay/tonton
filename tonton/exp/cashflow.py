#!/usr/bin/env python3
"""Signals for the cash-flow rules: counting-machine use (wrist in machine box), safe opened afterwards,
ID / cash near another person's wrist after counting started and before the safe opened."""
from engine import Video, inter, area, expand

WR = (9, 10)          # COCO keypoints: left/right wrist


def wrists(p, kmin=.3):
    return [(p["kpts"][i][0], p["kpts"][i][1]) for i in WR if len(p["kpts"]) > i and p["kpts"][i][2] >= kmin]


def inside(pt, b, m=0.0):
    return b[0] - m <= pt[0] <= b[2] + m and b[1] - m <= pt[1] <= b[3] + m


def iou(a, b):
    i = inter(a, b); return i / max(area(a) + area(b) - i, 1e-9)


def cash_flow(v, near=.03, count_min=.5, id_conf=.3, money_conf=.35):
    """Returns dict: count_start, counter_box, safe_open_after, returns=[(t, what, track)]."""
    if not v.machine:
        return None
    m = expand(v.machine, .01)
    fr = v.frames
    persons = {f: [p for p in v.B[f].get("persons", []) if p["conf"] >= .4 and not v.outside(p)] for f in fr}
    # 1) the cashier = the in-room person track seen in the most frames; counting = the cashier's wrist
    #    inside the machine box for >= count_min s (a client passing a hand over the machine doesn't count)
    import collections
    seen = collections.Counter(p["id"] for f in fr for p in persons[f] if p["id"] >= 0)
    cashier_id = seen.most_common(1)[0][0] if seen else None
    run, start, count_start, counter = 0, None, None, None
    for f in fr:
        hit = [p for p in persons[f] if p["id"] == cashier_id and any(inside(w, m) for w in wrists(p))]
        if hit:
            if start is None: start = v.B[f]["t"]
            if v.B[f]["t"] - start >= count_min and count_start is None:
                count_start, counter = start, hit[0]
        else:
            start = None
        if count_start is not None: break
    if count_start is None:
        return {"count_start": None}
    # 2) first safe opening after counting started
    cust = lambda f, c, cf: [d["box"] for d in v.C.get(f, {}).get("custom", []) if d["c"] == c and d["conf"] >= cf]
    safe_t = next((v.B[f]["t"] for f in fr if v.B[f]["t"] >= count_start and cust(f, "safe_open", .4)), None)
    # 3) ID / cash near the wrist of a person who is not the counter, between counting and safe opening
    cbox = counter["box"]; returns = []
    for f in fr:
        t = v.B[f]["t"]
        if t < count_start or (safe_t is not None and t >= safe_t): continue
        ps = persons[f]
        if not ps: continue
        me = max(ps, key=lambda p: iou(p["box"], cbox))          # the counter in this frame (follows motion)
        if iou(me["box"], cbox) > .2: cbox = me["box"]
        for p in ps:
            if p is me or iou(p["box"], me["box"]) > .3: continue
            for what, cf in (("ID", id_conf), ("cash", money_conf)):
                for b in cust(f, "ID-identification" if what == "ID" else "money_stack", cf):
                    if any(inside(w, b, near) for w in wrists(p)):
                        returns.append((t, what))
    return {"count_start": count_start, "safe_open_after": safe_t, "returns": returns}


if __name__ == "__main__":
    import sys
    from engine import labels
    cust = sys.argv[1] if len(sys.argv) > 1 else "cust_s_v1"
    L = labels()
    for stem, (y, t) in sorted(L.items(), key=lambda x: (x[1][0], x[0])):
        v = Video(stem, "base", cust); r = cash_flow(v)
        if not r or r["count_start"] is None:
            print(f"{stem:5} y={y} t={t}  no counting"); continue
        rid = [x for x in r["returns"] if x[1] == "ID"]; rc = [x for x in r["returns"] if x[1] == "cash"]
        print(f"{stem:5} y={y} t={t}  count@{r['count_start']:.1f} safe_open_after={r['safe_open_after']}  "
              f"ID->other: {len(rid)} first {rid[0][0] if rid else '-'}  cash->other: {len(rc)} first {rc[0][0] if rc else '-'}")


def no_id_before_count(v, visit_start, id_conf=.3, near=.03, money_conf=.35):
    """C01 check for one client visit. Returns (t_violation, count_start) or None.
    Violation: the cashier counts in the machine after the visit started, and no ID-identification box was
    seen from the visit start up to the count start. Timecode = first time cash is at the cashier's wrist
    after the visit started (the moment the money is accepted), else the count start."""
    import collections
    cf = cash_flow(v)
    if not cf or cf.get("count_start") is None or cf["count_start"] < visit_start:
        return None
    cs = cf["count_start"]
    for f in v.frames:
        t = v.B[f]["t"]
        if visit_start <= t <= cs and any(d["c"] == "ID-identification" and d["conf"] >= id_conf for d in v.C.get(f, {}).get("custom", [])):
            return None
    persons = lambda f: [p for p in v.B[f].get("persons", []) if p["conf"] >= .4 and not v.outside(p)]
    seen = collections.Counter(p["id"] for f in v.frames for p in persons(f) if p["id"] >= 0)
    cid = seen.most_common(1)[0][0] if seen else None
    for f in v.frames:
        t = v.B[f]["t"]
        if t < visit_start or t > cs: continue
        cash = [d["box"] for d in v.C.get(f, {}).get("custom", []) if d["c"] == "money_stack" and d["conf"] >= money_conf]
        if any(p["id"] == cid and any(inside(w, b, near) for w in wrists(p) for b in cash) for p in persons(f)):
            return t, cs
    return cs, cs
