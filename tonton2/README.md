# tonton2 — cash-desk violations: scene state machine + named violation rules

Re-implementation of tonton's analysis as a per-video scene
compiler (roles, door/safe/ID/money/count/laptop state) and 16 named violation rules. Uses tonton's data, weights
and cached detections **read-only via symlinks**; tonton and its service (port 8105) are untouched.
No catalog.json text loading and no debug UI. Rule texts are hard-coded in the rule classes, and the
web page is tonton's.

## Layout

```
app/pipeline/timeline.py   FrameFact / Person / LaptopFact, rows -> Timeline (px boxes), screen_luma()
app/pipeline/extract.py    YOLO26x-pose+ByteTrack, YOLO26x COCO, custom 10-class detector, laptop luma per frame
app/pipeline/cache.py      offline rows from tonton's cached detections (feats/base + feats/<cust>) + cache/luma
app/pipeline/engine.py     Pipeline.analyze(video) -> API result (evidence JPEGs, legal citation per rule)
app/rules/scene.py         SceneCompiler -> SceneState (all thresholds in DEFAULTS)
app/rules/predicates.py    inter/iou/near/inside, hands (palm reach), head_in_bbox, standing, runs()
app/rules/base.py          Rule, Alarm, REGISTRY, evaluate_rules(timeline, scene, rule_settings)
app/rules/violations/      cash.py, transaction.py, door.py, unattended.py, belongings.py
app/config.py              settings {"scene", "rules"} -> runs/settings.json, validate_settings()
app/ground_truth.py        open_labels.xlsx (+ "тип нарушения" column), minute timestamps, compare_ground_truth()
app/jobs/batch.py          score all videos -> runs/batch_*/summary.csv (gt_type, predicted_types, type_match)
service/server.py          FastAPI, same API as tonton + GET/PUT /api/v1/settings; compose -> tonton2-api :8107
tools/scene_dump.py        print the compiled scene of videos (debugging)
tests/test_rules.py        synthetic timelines per rule, GT parsing/compare, settings merge
data/                      open_labels.xlsx (own copy with the type column); videos, violation_classes.csv -> tonton
weights/                   pose + COCO -> tonton; custom_y26s_v1.pt = copy of tonton y26s_v1 best.pt
```

## Run

```bash
cd /nlp/nurlan/tonton2
./run.sh python3 -m app.jobs.batch -v                    # offline, in-sample detector (cust_s_v1), no GPU
./run.sh python3 -m app.jobs.batch --cust cust_cvb -v    # offline, 4-fold CV detector (honest)
./run.sh python3 -m app.jobs.batch --gt team             # with the team's Х-19/Х-35 corrections
GPU=0 ./run.sh python3 -m app.jobs.batch --live X-1 Х-22 # full GPU pipeline on raw videos
./run.sh python3 tools/scene_dump.py Х-38                 # roles, door/safe/ID runs, counts, piles, departure
./run.sh python3 -m unittest discover -s tests
cd service && docker compose up -d                        # tonton2-api :8107 (GPU 0, ~2.5 GB) + tonton2-tunnel; key in service/.env
docker logs tonton2-tunnel 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1   # public URL (changes when the tunnel container is recreated)
docker compose restart api                                # after changing app/ (code is loaded at start)
```

`run.sh` runs the `tonton2` image (= tonton-service + openpyxl) with tonton mounted read-only.
API: `POST /api/v1/analyze` (file) → job_id; `GET /api/v1/jobs/{id}`; per violation `type`, `timecode_sec`,
`description`, `evidence_frames` (+ `rule` citation, `class_id`, `interval_sec`, `debug.rule_id`).

## Results (2026-10-18, 42 videos; type match = expected rule_id among the predicted ones)

| detections | labels | verdict acc | TP | FP | type match | \|dt\|≤2 s / ≤5 s (typed TPs) |
|---|---|---|---|---|---|---|
| y26s_v1, in-sample for 33 videos | Excel | .929 | 22/22 | 3/20 | 21/22 | 12/21, 17/21 |
| **4-fold CV detector (honest)** | Excel | **.857** | 19/22 | 3/20 | 18/22 | 10/18, 15/18 |
| y26s_v1 | team (Х-19, Х-35 = violation) | .976 | 24/24 | 1/18 | 22/24 | |
| 4-fold CV detector | team | .905 | 21/24 | 1/18 | 19/24 | |
| *tonton rule engine v3 (for comparison)* | team | *.88 in-sample, .81 CV* | | | | |

Live GPU pipeline = identical alarms to the cached path (checked on X-1, Х-22, Х-28, Х-29, Х-46). One job:
Х-15 (21 s) took 50 s with GPU 0 shared with two trainings.

Remaining errors (in-sample / CV):
- Rules and thresholds were designed looking at these same 42 videos (most types have 1–4 examples), so even
  the CV row is optimistic; it only removes the detector's leakage.
- Х-39 FP (both): the detector sees the safe open 29–50 s while counting and the door open 51–78 s (detector errors, as in tonton).
- Х-19 / Х-35 fire `anonymous_cash_acceptance` / `cash_return`: violations per the team, clean per the Excel/plan.
- CV only: Х-15 and Х-22 FN (bag / key_in_lock missed by the unseen-video detector), Х-44 FN, Х-14 FP
  (currency intermix: stacking bundles of one long collection count looks like mixing).
- Х-4 type: it is the same file as Х-47; we report `interrupted_transaction` at 51 s, not `currency_intermix` at 57 s.

## Rule notes and deviations from the plan

- **Objects365 → COCO**: there are no Objects365 weights locally; every class used (person, laptop, book, handbag,
  backpack, cup, bottle) is in COCO (YOLO26x). COCO calls the collection bag `handbag`, so belongings ignore handbags
  overlapping a `collection_bag` seen within ±2 s.
- **Ground truth** = original `open_labels.xlsx` + `тип нарушения` (plan's slugs). The minute timestamp parses (Х-47 = 68 s).
- **Departure** (all left-* rules): first door opening that a person passes through after which nobody who was in the room
  before stays and nobody works at machine/safe/laptop (≥5 samples). Needed because after leaving, the cashier
  is still visible behind the glass wall, so "room empty" alone almost never holds before the clip ends.
- **Key + lighting (Х-11)**: the lights switch at 16 s (frame brightness 137 → 90), the small key becomes invisible and
  the detector says key_out (brightening the frames does not bring it back). A key cannot leave the lock by itself:
  on a lighting jump (>15 % brightness change in 0.5 s) the key state from the 1.5 s before is carried if ≥70 % of
  those detections agree, until a wrist is ≤5 px from the lock for 3 samples. Only Х-11 and its clean pair Х-12 have a
  lighting jump. In Х-12 the key comes out before the jump, so the rule stays quiet; with the CV detector, Х-12's
  pre-jump state flickers (56 % agreement), so it is not carried. The 70 % threshold is set on these two videos only.
- **Laptop**: luma of the upper 60 % of the laptop box, on ≥130 / off ≤120. Separates Х-28 (≈150 after leaving)
  from Х-29 (≈100 after the lock). No VLM needed.
- **Intruder**: the glass door mirrors whoever stands in it. An "outside" person seen while an in-room person is in the
  doorway is treated as a reflection (this was the FP source in Х-35/Х-37). Door must be open ≥5 s ("persistent").
- **Transactions** open at a count that has money in it (not at any machine touch) and close when the safe opens
  (the plan says open+close; Х-39 counts while the safe is open, and a count right after closing reopened one).
- **anonymous_cash_acceptance**: "client in the cash path" = a non-cashier palm in the machine box within ±3 s of the
  money's first appearance, and the money appears before the first persistent ID. Depends on ID detection, which is
  weak on unseen videos.
- **cash_return**: wrists only (palm reach 0), cash only (`include_id` = 0). tonton found that ID hand-backs are legit.
- **uncounted_safe_deposit**: money pile on the desk ≥2 s before the safe opens, appeared during the clip (not at
  t<1 s), and no count since it appeared. Money that appears just before the safe opens is being taken out.
- **currency_intermix**: count → hand away ≥1 s → same person re-reaches the machine → within 6 s a pile that appeared
  after the count vanishes onto a pile that was already counted. Weakest rule (thin signal, 1 TP in-sample).
- **open_safe**: a count that starts ≥1 s after the safe opened and ≥0.5 s before it closes; timestamp = safe opening.
- Not done (plan, optional): stamp class, custom laptop active/inactive classes, role badges overlay, catalog.json, UI.
