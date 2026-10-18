"""Score the rule engine on the training videos against data/open_labels.xlsx (verdict + type + time).

  python3 -m app.jobs.batch [--cust cust_s_v1|cust_cvb] [--gt team] [--out DIR] [-v] [stem ...]
  GPU=0 ./run.sh python3 -m app.jobs.batch --live [stem ...]      # full pipeline on the raw videos
Offline (default): cached detections from tonton (app.pipeline.cache), no GPU. Writes <out>/summary.csv.
--gt team applies the team's later corrections (Х-19, Х-35 are violations: cash accepted before the ID check).
"""
import argparse, csv, os

from app.config import load_settings
from app.ground_truth import compare_ground_truth, load_gt
from app.pipeline.cache import VIDEOS, load_rows, stems
from app.pipeline.engine import analyse

TEAM_CORRECTIONS = {s: {"verdict": "violation", "timestamp_sec": 7.0, "type": "anonymous_cash_acceptance_violation"}
                    for s in ("Х-19", "Х-35")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cust", default="cust_s_v1"); ap.add_argument("--out"); ap.add_argument("-v", action="store_true")
    ap.add_argument("--gt", default="excel", choices=["excel", "team"]); ap.add_argument("--live", action="store_true")
    ap.add_argument("stems", nargs="*")
    a = ap.parse_args()
    tag = "live" if a.live else a.cust
    out = a.out or os.path.join("runs", f"batch_{tag}" + ("_team" if a.gt == "team" else ""))
    os.makedirs(out, exist_ok=True)
    gt, settings, res = load_gt(), load_settings(), []
    if a.gt == "team": gt.update(TEAM_CORRECTIONS)
    pipe = None
    if a.live:
        from app.pipeline.engine import Pipeline
        pipe = Pipeline(device=0)
    for s in a.stems or stems():
        rows = pipe.rows(os.path.join(VIDEOS, s + ".mov")) if pipe else load_rows(s, a.cust)
        _, _, alarms = analyse(rows, s, settings)
        g = gt.get(s, {"verdict": "no_violation", "timestamp_sec": None, "type": None})
        c = compare_ground_truth(g, alarms)
        res.append({"video": s, "gt_verdict": g["verdict"], "pred_verdict": c["predicted_verdict"],
                    "verdict_match": int(c["verdict_match"]), "gt_type": g["type"] or "",
                    "predicted_types": " ".join(c["predicted_types"]), "type_match": int(c["type_match"]),
                    "gt_t": g["timestamp_sec"], "timestamp_abs_error_sec": c["timestamp_abs_error_sec"],
                    "alarms": " ".join(f"{x.rule_id}@{x.timecode_sec:.1f}" for x in alarms)})
        if a.v:
            r = res[-1]
            mark = {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}[(g["verdict"] == "violation", bool(alarms))]
            print(f"{s:5} {mark} type={'ok' if r['type_match'] else '--'} gt={r['gt_type'] or '-'}@{r['gt_t']} "
                  f"err={r['timestamp_abs_error_sec']}  {r['alarms']}")
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    pos = [r for r in res if r["gt_verdict"] == "violation"]; neg = [r for r in res if r["gt_verdict"] != "violation"]
    tp = sum(r["pred_verdict"] == "violation" for r in pos); fp = sum(r["pred_verdict"] == "violation" for r in neg)
    typ = sum(r["type_match"] for r in pos); errs = [r["timestamp_abs_error_sec"] for r in pos if r["type_match"] and r["timestamp_abs_error_sec"] is not None]
    print(f"[{tag}, gt={a.gt}] videos {len(res)}  verdict acc {sum(r['verdict_match'] for r in res) / len(res):.3f}  "
          f"TP {tp}/{len(pos)}  FP {fp}/{len(neg)}  type match {typ}/{len(pos)}  "
          f"|dt|<=2s {sum(e <= 2 for e in errs)}/{len(errs)}  <=5s {sum(e <= 5 for e in errs)}/{len(errs)}  -> {out}/summary.csv")
    return res


if __name__ == "__main__":
    main()
