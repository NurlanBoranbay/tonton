"""Evaluate raw/*.jsonl against labels.csv + violations.md types. Writes results.md and results.json."""
import csv, glob, json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import parse, ROOT, DS

# type labels from violations.md (C-codes via violation_classes.csv example_videos);
# Х-15 (bag left open, C09 not selected) and Х-18 (too many people, C15 not selected) mapped to nearest selected
GOLD = {"X-1": {"C05"}, "X-2": {"C01"}, "Х-4": {"C03"}, "Х-11": {"C11"}, "Х-13": {"C02"},
        "Х-15": {"C19", "C08", "C07"}, "Х-17": {"C14"}, "Х-18": {"C14"}, "Х-20": {"C10"}, "Х-22": {"C11"},
        "Х-24": {"C12"}, "Х-26": {"C08"}, "Х-28": {"C17"}, "Х-30": {"C14"}, "Х-32": {"C16"}, "Х-33": {"C16"},
        "Х-34": {"C01", "C06"}, "Х-38": {"C07"}, "Х-40": {"C12"}, "Х-44": {"C04"}, "Х-46": {"C12"}, "Х-47": {"C03"}}
LAB = {r["file"][:-4]: (int(r["violation"]), float(r["timestamp_sec"]) if r["timestamp_sec"] else None)
       for r in csv.DictReader(open(f"{DS}/labels.csv"))}

def events(rows, agg):
    """rows of one video -> list of (class, start) after aggregation."""
    ev = []
    for r in rows:
        for v in parse(r["content"]) or []:
            try:
                ev.append((str(v.get("class_id", "?")).strip()[:3], float(v.get("start_sec", r["t0"])), r["win"]))
            except (TypeError, ValueError):
                ev.append((str(v.get("class_id", "?"))[:3], float(r["t0"]), r["win"]))
    if agg == "noC14":  # what is left if the outsider/extra-person class were handled elsewhere (person counting)
        ev = [e for e in ev if e[0] != "C14"]
    if agg == "vote2":  # class must be reported in >=2 windows (sliding windows overlap 50%)
        n = Counter(c for c, _, w in set((c, 0, w) for c, _, w in ev))
        ev = [e for e in ev if n[e[0]] >= 2]
    return sorted((c, s) for c, s, _ in ev)

def evaluate(path, agg):
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        by[r["video"]].append(r)
    res = dict(tp=0, fp=0, fn=0, tn=0, err_first=[], err_best=[], type_first=0, type_any=0,
               fp_cls=Counter(), hit_cls=Counter(), pred_cls_pos=Counter(), per_video={})
    for v, (y, ts) in LAB.items():
        ev = events(by.get(v, []), agg)
        p = int(bool(ev))
        res["per_video"][v] = [c for c, _ in ev]
        if y and p:
            res["tp"] += 1
            first = min(ev, key=lambda e: e[1])
            best = min(ev, key=lambda e: abs(e[1] - ts))
            res["err_first"].append(abs(first[1] - ts)); res["err_best"].append(abs(best[1] - ts))
            res["type_first"] += first[0] in GOLD[v]
            res["type_any"] += bool({c for c, _ in ev} & GOLD[v])
        elif y:
            res["fn"] += 1
        elif p:
            res["fp"] += 1
        else:
            res["tn"] += 1
        for c in {c for c, _ in ev}:
            if not y:
                res["fp_cls"][c] += 1
            elif c in GOLD[v]:
                res["hit_cls"][c] += 1
            else:
                res["pred_cls_pos"][c] += 1
    calls = len(rows); sec = sum(r["sec"] for r in rows)
    toks = sum((r["usage"] or {}).get("completion_tokens", 0) for r in rows)
    res.update(calls_per_video=calls / len(LAB), sec_per_video=sec / len(LAB), out_tok_per_call=toks / max(calls, 1),
               parse_fail=sum(parse(r["content"]) is None for r in rows))
    return res

def main():
    table, out = [], {}
    for path in sorted(glob.glob(f"{ROOT}/raw/*.jsonl")):
        tag = os.path.basename(path)[:-6]
        for agg in (["any", "noC14"] if tag.startswith(("OV", "CAND")) else ["any", "vote2", "noC14"]):
            r = evaluate(path, agg)
            n = r["tp"] + r["fp"] + r["fn"] + r["tn"]
            prec = r["tp"] / max(r["tp"] + r["fp"], 1); rec = r["tp"] / (r["tp"] + r["fn"])
            e1, eb = r["err_first"], r["err_best"]
            w = lambda e, k: sum(x <= k for x in e)
            table.append(f"| {tag} | {agg} | {(r['tp'] + r['tn']) / n:.2f} | {prec:.2f} | {rec:.2f} | {r['fp']}/20 | "
                         f"{w(e1, 2)}/{w(e1, 5)} | {w(eb, 2)}/{w(eb, 5)} | {r['type_first']}/{r['type_any']} of {r['tp']} | "
                         f"{r['calls_per_video']:.1f} | {r['sec_per_video']:.0f} | {r['parse_fail']} |")
            out[f"{tag}/{agg}"] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in r.items()}
    hdr = ("| variant | agg | acc | prec | rec | FP clean | ts first ±2/±5 | ts best ±2/±5 | type first/any | "
           "calls/video | VLM s/video (serial) | parse fail |\n|" + "---|" * 12)
    md = "# VLM baseline results\n\n" + hdr + "\n" + "\n".join(table) + "\n"
    md += "\nts: |pred start - label| over true-positive videos; first = earliest reported event, best = event closest to label.\n"
    md += "type: first event's class in gold set / any reported class in gold set.\n\n## Per-class (clean videos with class = FP; hits = violation videos where gold class predicted)\n"
    for k, r in out.items():
        md += f"\n**{k}**: FP-by-class {dict(sorted(r['fp_cls'].items()))}; hits {dict(sorted(r['hit_cls'].items()))}; wrong-class-on-positive {dict(sorted(r['pred_cls_pos'].items()))}\n"
    open(f"{ROOT}/results.md", "w").write(md)
    json.dump(out, open(f"{ROOT}/results.json", "w"), ensure_ascii=False, indent=1)
    print(md)

if __name__ == "__main__":
    main()
