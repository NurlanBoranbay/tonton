# tonton3: cash-desk violation detection (ForteBank Datathon 26)

Upload a `.mov` from the overhead cash-desk camera and get back a verdict (`violation` / `no_violation`). For each
violation the result gives the type, a timecode, 1–3 evidence frames, a description and the rule it breaks
(act, point, quote, link to adilet.zan.kz). Everything runs locally: no cloud AI, no internet at runtime and no face recognition.

```
video ─▶ detection (YOLO26, every 3rd frame)
           · people + skeleton + tracking   yolo26x-pose + ByteTrack
           · everyday objects               yolo26x (COCO: laptop, cup, bottle, book, handbag…)
           · our 11 classes                 money_stack, collection_bag, counting_machine, safe_open/closed,
                                            key_in/out_of_lock, door_open/closed, ID-identification, stamp
      ─▶ scene compiler   (app/rules/scene.py): roles, door / safe / key / ID / money / count / laptop state
      ─▶ violation rules  (app/rules/violations/*.py): one readable condition per violation
      ─▶ result JSON + evidence frames + legal citation
```

## Quick start

Requirements: an NVIDIA GPU (~3 GB of memory), Docker with the NVIDIA Container Toolkit.

```bash
cp .env.example .env              # set API_KEY
sh tools/download_weights.sh      # ~345 MB from huggingface.co/nurlanboranbay/tonton
docker compose up -d --build
curl localhost:8108/api/v1/health
```

The web UI (in Russian) is at http://localhost:8108/. It plays the video with the detections drawn on top (people with
skeletons and roles, cash-desk objects, personal items, scene zones), marks each violation on the timeline, and lists
the violations with evidence frames and the legal rule. Earlier jobs are in the history panel.
A video takes about half its length to process on an A100. The limit is 5 minutes per video.

## API

```bash
curl -X POST localhost:8108/api/v1/analyze -H "X-API-Key: $KEY" -F "file=@video.mov"   # → {"job_id": "..."}
curl localhost:8108/api/v1/jobs/<job_id> -H "X-API-Key: $KEY"                          # status: processing | done | failed
```

Result:
`{"job_id", "status", "verdict", "violations": [{"type", "timecode_sec", "description", "evidence_frames": [{"timestamp_sec", "image_base64"}], "class_id", "rule": {"act", "citation", "quote", "link"}, "interval_sec"}]}`.
A request without the right `X-API-Key` gets 401. Other endpoints: `GET /api/v1/jobs` (history),
`GET /api/v1/jobs/<id>/video` (H.264 MP4), `GET /api/v1/jobs/<id>/detections` (per-frame boxes and skeletons),
`GET/PUT /api/v1/settings` (rule thresholds).

### Testing the API

`tools/test_api.py` needs only Python 3. It checks the service, uploads each video, waits for the result and prints
the violations:

```bash
python3 tools/test_api.py --url https://<host> --key <API_KEY> video1.mov video2.mov
python3 tools/test_api.py --url https://<host> --key <API_KEY> --out results/ *.mov     # also save each result JSON
```

## Models

| file | what | size | licence |
|---|---|---|---|
| `yolo26x-pose.pt` | people + 17 keypoints | 126 MB | Ultralytics YOLO26, AGPL-3.0 |
| `yolo26x.pt` | COCO-80 objects | 119 MB | Ultralytics YOLO26, AGPL-3.0 |
| `custom_y26s_all_stamp.pt` | our 11-class detector (YOLO26-s, 960 px), trained on the team's annotations of all 42 videos | 100 MB | ours |

To use another custom detector, put its path in `service/custom_weights.txt` and run `docker compose restart api`.
Class names are read from the model.

## Violations

Citations are from НБРК resolution № 231 (V1900019680) unless noted; texts are in `data/violation_classes.csv`.

| rule | violation | legal basis |
|---|---|---|
| no_id_before_count, anonymous_cash_acceptance | cash accepted / counted without the client's ID (C01) | п.19 пп.1, п.31 пп.1 |
| uncounted_safe_deposit | money put in the safe without counting (C02) | п.19 пп.2 |
| currency_intermix | money of different clients mixed (C03) | п.20 ч.2, п.21 |
| cash_return | cash handed back right after counting, before the safe (C05) | п.31 пп.4 |
| interrupted_transaction | client leaves before the operation ends (C06) | п.35 |
| left_money, left_collection_bag | money / collection bag left when the cashier leaves (C08) | п.22, п.20 ч.2 |
| open_safe | safe left open (C10) | п.20 ч.2, п.80 |
| left_key | key left in the safe lock (C11) | п.76, п.80 |
| personal_belongings, beverage, left_commodities | personal items, drinks, stamps on the desk (C12) | п.12 |
| open_door_violation_intruder | outsider in the cash room (C14) | п.75 |
| open_door | cash-room door left open (C16) | V2000020075 п.8, п.16 пп.1, п.21 |
| left_laptop | computer left unlocked (C17) | no direct point in the acts; reported without a citation |
| anonymous_incassator | collection bag accepted without checking the collector's ID (C19) | п.24 пп.3–4, п.25 |

## Accuracy

On the 42 training videos (team labels: 26 violation, 16 clean):

| detector | found | false alarms | accuracy |
|---|---|---|---|
| 4-fold cross-validation (detector never saw the scored video) | 23/26 | 1/16 | **0.905** |
| deployed detector (trained on all 42 videos) | 26/26 | 1/16 (Х-45) | 0.976 |

The cross-validation figure is the estimate for new videos. The deployed row only shows that nothing broke. Per-video
results: `runs/batch_cust_cvb/summary.csv` and `runs/batch_cust_prod/summary.csv`. The thresholds were tuned on these
same videos, so the real figure may be lower. Rules use scene anchors found in each video (no fixed coordinates), so
they don't depend on the camera angle.

## Layout

```
app/pipeline/    detection (extract.py), per-frame facts (timeline.py), Pipeline.analyze → result (engine.py)
app/rules/       scene compiler, predicates, rule registry, violations/
app/jobs/        batch scoring against data/labels.csv
service/         FastAPI server + web UI
tests/           python -m unittest discover -s tests
data/            labels.csv, open_labels.xlsx, violation_classes.csv (legal citations)
```

## Offline deployment (bank network)

Run `docker save tonton3 | gzip > tonton3.tar.gz`, copy it together with this folder and `weights/`, then run
`docker load < tonton3.tar.gz` and `docker compose up -d` on the target machine. No internet access is needed.
