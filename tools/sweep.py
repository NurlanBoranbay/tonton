"""Quick param sweep: python3 tools/sweep.py  (prints FP/FN per detector for each variant)"""
import copy, csv, itertools
from app.config import load_settings
from app.jobs.batch import analyse
from app.pipeline.cache import load_rows, stems
L = {r["file"][:-4]: int(r["violation"]) for r in csv.DictReader(open("data/labels.csv", encoding="utf-8"))}
CUSTS = ["cust_prod", "cust_s_v1", "cust_cvb"]
ROWS = {c: {s: load_rows(s, c) for s in stems()} for c in CUSTS}
base = load_settings()
def run(settings):
    out = {}
    for c in CUSTS:
        fp, fn = [], []
        for s, rows in ROWS[c].items():
            _, _, al = analyse(rows, s, settings)
            p = bool(al)
            if p and not L[s]: fp.append(s)
            if L[s] and not p: fn.append(s)
        out[c] = (fp, fn)
    return out
variants = {"current": {}}
for mo in (5, 7, 9): variants[f"door_min_open={mo}"] = {"open_door_violation_intruder": {"min_open_sec": mo}}
for ps, mp in itertools.product((10, 15, 20), (10, 6)):
    variants[f"intermix ps={ps} mp={mp}"] = {"currency_intermix_violation": {"pile_samples": ps, "merge_px": mp}}
for name, ch in variants.items():
    st = copy.deepcopy(base)
    for rid, kv in ch.items(): st.setdefault("rules", {}).setdefault(rid, {}).update(kv)
    r = run(st)
    print(f"{name:28} " + "  ".join(f"{c}: FP{len(fp)} {fp} FN{len(fn)} {fn}" for c, (fp, fn) in r.items()), flush=True)
