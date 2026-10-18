#!/usr/bin/env python3
"""Per-video summary features from the state timeline (longest run of each state, counts)."""
import csv, json, sys
from engine import Video, labels, runs
def longest(tl, cond):
    r = runs(tl, cond, 0.0)
    return round(max((b - a for a, b, _, _ in r), default=0.0), 1)
FEATS = {
    "safe_open": lambda s: s["safe_open"] and not s["safe_closed"],
    "door_open": lambda s: s["door_open"] and not s["door_closed"],
    "empty_room": lambda s: s["n_room"] == 0,
    "empty+money": lambda s: s["n_room"] == 0 and s["money_desk"] > 0,
    "empty+safe_open": lambda s: s["n_room"] == 0 and s["safe_open"],
    "empty+key_in": lambda s: s["n_room"] == 0 and s["key_in"],
    "key_out": lambda s: s["key_out"] > 0,
    "money_desk>=2": lambda s: s["money_desk"] >= 2,
    "money_desk": lambda s: s["money_desk"] >= 1,
    "personal": lambda s: bool(s["personal"]),
    "n_room>=2": lambda s: s["n_room"] >= 2,
    "n_room>=3": lambda s: s["n_room"] >= 3,
    "bag": lambda s: s["bag"] > 0,
    "id": lambda s: s["id"] > 0,
}
def table(base, cust):
    L = labels(); rows = []
    for stem, (y, t) in sorted(L.items(), key=lambda x: (x[1][0], x[0])):
        v = Video(stem, base, cust); tl = v.timeline()
        rows.append({"video": stem, "y": y, "dur": round(tl[-1]["t"], 0), **{k: longest(tl, f) for k, f in FEATS.items()}})
    return rows
if __name__ == "__main__":
    cust = sys.argv[1] if len(sys.argv) > 1 else "cust_s_v1"
    rows = table("base", cust)
    keys = list(rows[0])
    print(" ".join(f"{k[:9]:>9}" for k in keys))
    for r in rows: print(" ".join(f"{str(r[k])[:9]:>9}" for k in keys))
