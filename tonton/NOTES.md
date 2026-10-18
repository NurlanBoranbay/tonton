# tonton — ForteBank Datathon 26: cash-discipline violations from video

Read this first. It is the shared notebook of every agent working on the case: keep it current,
add results as you get them (newest on top in "Experiment log"), and keep conclusions honest.

## The case (docs: `Кейс_ForteBank.docx.pdf`)

- Local AI service: user uploads a `.mov` (overhead camera, one cash desk, no audio) → verdict
  `violation` / `no_violation`; per violation: type, `timecode_sec`, 1–3 evidence JPEG frames (base64),
  description of what is visible, rule + normative source (act, point).
- API: `POST /api/v1/analyze` (field `file`) → `job_id`; `GET /api/v1/jobs/{id}` → status + result.
  `X-API-Key` header. Must finish in **5 min** per video.
- **Fully local** (no cloud AI, no sending frames anywhere). Open-source models with licences that
  allow commercial use. Face recognition is **forbidden**.
- Scoring (100): detection 25 (per-video verdict), localisation 15 (timecode + frame),
  classification 15 (type), rule grounding 15 (no invented citations), robustness 10 (**final test video
  is filmed from a different angle**), product 10, tech 10.
- FAQ: missing a violation is unacceptable, false alarms must be few. Test may contain violation types
  not in training data. Only RK acts (НБРК, adilet). Timecode: give a concrete frame.

## Data (`dataset/`, see `dataset/README.md`)

- `videos/*.mov`: 42 videos, 464×848, 30 fps, 11–142 s. **X-1, X-2 use Latin "X", the rest Cyrillic "Х".**
- `labels.csv`: file, violation 0/1, timestamp_sec (22 violation / 20 clean). No type labels.
- `violations.md/.csv`: team's description of every violation video + rule. **Use as type labels.**
- `violation_classes.csv`: 21 classes, `selected=1` for the 15 the team chose; verified quotes + links.
- `units.jsonl`, `txt/`: parsed text of 25 RK acts. Main act V1900019680 (НБРК № 231 cash rules).
- `yolo/`: live annotations from the web annotator (`annotator/`, served on :8097 + cloudflared
  tunnel). 10 custom classes: money_stack, collection_bag, counting_machine, safe_open, safe_closed,
  key_in_lock, key_out_of_lock, door_open, door_closed, ID-identification. ~3k frames (2026-10-17).

## Selected violation types (15)

C01 no ID before operation · C02 uncounted money into safe · C03 mixing clients' money ·
C04 mixing denominations · C05 not the whole counted sum issued · C06 client leaves before the end ·
C07 money left on desk after client leaves · C08 money unattended / cashier left · C10 safe open (no
cashier) · C11 key in safe lock (no cashier) · C12 foreign/personal items on desk · C14 outsider in cash
room · C16 door left open · C17 computer left unlocked · C19 collection bag accepted without check.

## Infrastructure

- gpu-server, A100 80GB ×6. **We may use GPU 0 and GPU 1.** GPU 1 was freed by stopping nailf's
  `vllm-gemma4-gpu1` (restore: `docker start vllm-gemma4-gpu1`, note in /nlp/nurlan/RESTORE_gemma4_gpu1.txt).
- Gemma 4 26B-A4B-it (vision) still available at `http://localhost:8887/v1` (LB over GPU 2 and 5),
  model name `/models/gemma-4-26B-A4B-it`, **max 1 image per request** (tile frames into one image).
- Docker image `tonton-train` (`docker/train/Dockerfile`): torch 2.11 cu130, ultralytics 8.4.180
  (YOLO26), transformers ≥5.18, sentence-transformers 6.1. Mount `/nlp/nurlan/tonton` at the same path.
  `tonton-embed`: EmbeddingGemma 2 (`models/embeddinggemma-2`).
- Licences: ultralytics is **AGPL-3.0** (open source, commercial use allowed but copyleft: service
  source must be offered). Apache alternatives if needed: RT-DETR / D-FINE, RTMPose.

## Code (`exp/`)

- `make_split.py <name>`: freeze annotations → `exp/data/<name>/` split **by video** (val videos chosen so
  every class is in val). v1 val: X-1, X-2, Х-16, Х-19, Х-20, Х-25, Х-29, Х-43, Х-48.
- `train_det.sh <gpu> <data> <model> <run>`: YOLO detector training → `exp/runs/<run>/`.
- `extract.py --tag T [--pose] [--coco] [--custom]`: per-frame facts → `exp/feats/T/<video>.jsonl`.

## Service (`service/`, see `service/README.md`)

- Deliverable web service: FastAPI (`service/app.py`) wrapping `exp/pipeline.py`, Russian UI at `/`.
  `cd service && docker compose up -d --build` → container `tonton-api` (GPU 0, host net, **port 8105**,
  restart unless-stopped) + `tonton-tunnel` (cloudflared quick tunnel, `--protocol http2`; QUIC was ~10× slower
  on upload). Public URL: `docker logs tonton-tunnel 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1`
  (changes whenever the tunnel container is recreated). API key in `service/.env` (mode 600).
- Models and `exp/*.py` are loaded once at start → **`docker compose restart api` after changing the pipeline/rules**.
  One worker thread, one job at a time; jobs on disk in `service/jobs/<id>/` (survive restarts).
- Timing (2026-10-17, stride 3): Х-14 (142 s) 252 s processing / 262 s end-to-end through the tunnel with GPU 0
  shared by 2 trainings + 1 extraction; the same video on an idle GPU 1 = 75 s. X-1 52 s, Х-22 34 s (contended).

## Field research (2026-10-17)

**Paper arXiv 2512.14698 = TimeLens (Tencent ARC + NJU, CVPR 2026)**: *video temporal grounding*
(text query → [start,end] s), not violation detection; no detectors/pose/explanations. What is useful:
- Fine-tuned Qwen2.5-VL-7B / Qwen3-VL-8B. **Best timestamp encoding = interleave a text prefix
  "12.3s" before each frame's visual tokens** (mIoU 48.3 vs 36.6 for position-embedding only, vs 46.3
  for burning the timestamp into the image). → When we prompt a VLM, send frames as
  `"t=12.3s" <img> "t=13.3s" <img> …` (or Qwen video input with fps so it inserts timestamps itself);
  do not just tile frames without times.
- Training: thinking-free RLVR (GRPO, reward = IoU) ≥ SFT; "thinking" hurt grounding and was slower.
  Data quality mattered more than method (noisy labels 35.6 → clean 48.3 mIoU).
- Numbers (mIoU Charades/ANet/QVH): Qwen3-VL-8B 48.3/46.8/59.4 → TimeLens-8B 55.2/53.2/65.5;
  Gemini-2.5-Pro 52.8/58.1/70.4. **Follow-up TimeLens2 (arXiv 2607.17423, Jul 2026)**: multi-interval
  output (JSON list of [start,end]), TimeLens2-8B 58.6/58.6/70.2, Ego4D-NLQ 19.0 vs Qwen3-VL-8B 5.3,
  Qwen3.5-35B-A3B 13.0. Even the best is ~20 mIoU on egocentric hand-object queries → **VLM grounding
  is coarse; use it to propose/verify a window, take the exact frame from our tracker/pose events.**
- Explanation/rule grounding ideas from VAD literature (AnomalyRuler, SlowFastVAD, AnyAnomaly WACV26,
  VERA CVPR25): fast detector path for everything + slow VLM only on uncertain windows, VLM gets the
  rule text and must answer per-rule yes/no with evidence frame — this is what we already do.

**Local VLMs already on disk** (`/home/nailf/vllm/models`, all vision-capable):
Qwen3.8-27B (qwen3_5 arch, 52 GB, **already served as `qwen-vl` on :8901**, container `tt-qwen-vl`,
16 images or 1 video per prompt), Qwen3.6-35B-A3B (MoE, 67 GB), gemma-4-26B-A4B-it (:8887, 1 image),
Qwen3.8-Flash-Next-FP8 (173 GB, too big for us), `/nlp/nurlan/models/qwen3.5-9b` (19 GB). HF cache
(`/nlp/nailf/cache-huggingface/hub`): gemma-4-31B-it, gemma-4-E2B, Qwen2.5-VL-3B, Qwen2-VL-7B,
siglip2-so400m, metaclip-2, jina-clip-v2. Not local: anything below marked "no". /nlp has ~100 GB free.

**Model shortlist (ranked for our use)**

| # | model | size | licence | HF id | local | why |
|---|---|---|---|---|---|---|
| 1 | Qwen3.8-27B (VL) | 27B | Apache-2.0 (Qwen3.x family) | local | **yes, :8901** | strongest local VLM; takes real video input with timestamps; use as verifier with per-rule questions |
| 2 | TimeLens2-8B / -4B | 8.8B / 4.4B, ~18/9 GB | Apache-2.0 | `MCG-NJU/TimeLens2-8B`, `-4B` | no | SOTA open temporal grounding, multi-interval JSON output, Qwen3-VL arch (vLLM ok); "when does the cashier put money into the safe?" → window |
| 3 | Molmo2-8B / -4B | 8.7B / 4.9B | Apache-2.0 (weights; some training data non-commercial — note it) | `allenai/Molmo2-8B` | no | video **pointing/tracking/counting**, beats Gemini 3 Pro on video tracking; good for "point to the banknotes / key" across frames. Needs transformers 4.57.1 / SGLang |
| 4 | V-JEPA 2 ViT-L (2.1 also out) | 0.3B | MIT (ViT-L), Apache-2.0 (ViT-g 384) | `facebook/vjepa2-vitl-fpc64-256`, `facebook/vjepa2-vitg-fpc64-384` | no (1.3 / 4 GB) | frozen clip encoder + attentive probe; SSv2 probe 73.7–77.3 % — strong on hand-object *motion*. With 40 videos: embed 2–4 s clips around events (desk ROI crop), train a tiny probe/logistic or kNN for "counting / handing over / putting in safe / ID shown"; also unsupervised pair-diff between staged pairs |
| 5 | RF-DETR (N…2XL, det + seg) | 30–130M | Apache-2.0 | `roboflow/rf-detr` (pip `rfdetr`) | no | Apache replacement for YOLO26 (AGPL), DINOv2 backbone → better few-shot fine-tune and cross-angle robustness; same YOLO/COCO labels |
| 6 | D-FINE / DEIM(v2) | 4–60M | Apache-2.0 (DEIMv2: check repo) | `ustc-community/dfine-*` | no | alternative Apache detectors, in transformers |
| 7 | OWLv2 / Grounding-DINO | 0.15 / 0.23B | Apache-2.0 | `google/owlv2-base-patch16-ensemble`, `IDEA-Research/grounding-dino-base` | no | zero-shot boxes for classes not in our 10 (phone, laptop screen, cup, notebook) and for auto-labelling the new camera angle |
| 8 | SAM 2.1 | 0.22B | Apache-2.0 | `facebook/sam2.1-hiera-large` | no | track a money stack / bag mask through the clip (where did it go?) |
| — | SAM 3 | 0.86B | SAM License (custom, commercial OK but ITAR/military restrictions, attribution) | `facebook/sam3` | no | text-prompted video seg+track ("banknotes") — best quality, licence acceptable but not OSI |
| — | RTMPose/RTMW (rtmlib) | — | Apache-2.0 | mmpose | no | Apache replacement for YOLO-pose; RTMW gives 21 hand keypoints/hand |
| — | InternVL3.5-8B, Qwen3-VL-8B | 8.5B | Apache-2.0 | `OpenGVLab/InternVL3_5-8B`, `Qwen/Qwen3-VL-8B-Instruct` | no | weaker than Qwen3.x on grounding; only if we need a small fast VLM |
| — | hand_object_detector (100DOH, ddshan) | Faster R-CNN | MIT code; data 100DOH research | github ddshan/hand_object_detector | no | hand state "in contact with portable object" — old but directly gives hand↔object contact |

Licence warnings: DINOv3 weights = custom licence; Molmo2 training data partly non-commercial; RetailAction (Standard AI, ICCV25), MERL Shopping, Something-Something v2, Ego4D, EPIC-Kitchens are **non-commercial/proprietary** — use only for reading/inspiration, not training a shipped model.

**Datasets**

| name | licence | size | URL | use |
|---|---|---|---|---|
| KZ-BD (Kazakhstan banknotes, 14 classes incl. 500–20000 ₸, YOLO boxes; note x,y may be top-left not centre) | CC BY 4.0 | 4200 imgs, **6.2 GB zip** (not downloaded) | https://data.mendeley.com/datasets/dny3dgvvw8/1 | close-up phone photos; at our 464×848 overhead resolution a note is ~20–40 px, so only useful as copy-paste augmentation for `money_stack` / denomination colour (C04) |
| Kazakh Banknote Image Dataset (7 classes + Mix, classification only) | CC BY 4.0 | ~1.6k+ imgs | https://data.mendeley.com/datasets/gyw87k93yb/1 | denomination classifier on crops (C04 mixing denominations) |
| BankNote-Net (Microsoft, 17 currencies, embeddings + encoder) | CDLA-Permissive-2.0 | small | https://github.com/microsoft/banknote-net | no KZT; low value |
| Roboflow Universe "banknote" (generic, 3.5k imgs) | CC BY 4.0 | small | https://universe.roboflow.com/fyp-leirw/banknote | generic money boxes for detector pretraining |
| RetailAction (take/put/touch, top-down fisheye, 21k clips) | proprietary (Standard AI) | large | HF `standard-cognition/RetailAction` | closest domain (overhead hand-object), but not licence-compatible |
| MERL Shopping (overhead, 106×2 min, hand-shelf actions) | non-commercial | ~ GB | merl.com/research/downloads/MERL_Shopping_Dataset | same: inspiration only |
| EgoOops / PREGO (procedural mistakes) | research | small | arXiv 2410.05343 | method ideas only |
No public cashier/teller cash-handling video dataset was found (checked retail checkout fraud, teller,
cash counting): **our 40 staged videos + synthetic augmentation are the only training data.**
Nothing downloaded to `external/`.

**Recommendation — try first (in this order):**
1. **Qwen3.8-27B on :8901 as the verifier/explainer with real video + timestamps** (TimeLens lesson):
   send the ±6 s window around each rule-engine candidate as video (fps 2–4) or as interleaved
   `t=..s` + frame images, crop to the desk/safe ROI, ask one yes/no question per candidate rule with
   the rule text, require JSON {answer, t_sec, evidence}. Cheap — already running, zero download.
   Also use it in "pair mode" (reference clean frame + candidate frame), which already fixed Gemma.
2. **V-JEPA 2 ViT-L (1.3 GB, MIT) frozen clip embeddings + tiny probe** for the transaction types the
   rule engine misses (C01 ID, C02 uncounted to safe, C03/C04 mixing, C05 partial sum): label 2–4 s
   event clips (~5–20 per class from the 40 videos), train a logistic/attentive probe with leave-pair-out
   CV. Expected to be less sensitive to camera angle than box-geometry rules (unverified), and it gives
   an event timeline for timecodes.
3. **TimeLens2-8B (18 GB, Apache)** only if (1) localises badly: one-shot query "when does the
   cashier hand money to the client / put money in the safe" → intervals, then rules check order.
   Side track: swap YOLO26 → **RF-DETR** (Apache) if the AGPL copyleft becomes a product issue, and
   use OWLv2 / Grounding-DINO to auto-label the new camera angle.

## Experiment log (newest first)

### 2026-10-18 — C01 "no ID before counting" rule; Х-37, Х-39 relabelled
- **Legal check:** cash deposit at the desk without an ID IS a violation: V1900019680 п.19 пп.1 (visual
  identification with an ID document **or** the digital-documents service, no amount threshold) + п.7
  (cash operations only after due diligence under the AML law). The AML law's ≤500,000 ₸ exemptions
  (ст.5) cover cash-accepting machines, not the teller. A digital ID on a phone is NOT a violation.
- **Х-37, Х-39 are violations** (team: client gave no ID). Labels now: 26 violation / 16 clean.
- Rule (engine rule 7, `cashflow.no_id_before_count`): in a client visit where the cashier counts in the
  machine, no `ID-identification` box (conf ≥.3) between the visit start and the count start → C01;
  timecode = first cash at the cashier's wrist after the visit started.

| detections | TP | FP | FN | TN | acc |
|---|---|---|---|---|---|
| y26s_v1 (in-sample) | 22/26 | 0/16 | 4 | 16 | .90 |
| **4-fold CV (honest)** | **19/26** | **0/16** | 7 | 16 | **.83** |

  X-2 caught at exactly 10.0 s (label 10). Misses (CV): Х-13, Х-15, Х-17, Х-20, Х-22, Х-40, Х-44.
- Detector with +3,500 synthetic MIDV-2020 ID images (`external/yolo_id`) reaches ID mAP50 .81 vs .53
  after 9 epochs (run `y26s_v1_id`, still training) → next: CV with it and re-evaluate C01.

### 2026-10-17 — open ID datasets
Goal: more `ID-identification` (class 9) training signal (mAP50 0.53 from 532 boxes / 9 videos).
Built **`external/yolo_id/`: 3500 copy-paste images, 4246 pasted IDs** (+ all original boxes of each frame
kept): real frames of our videos (v1-val videos excluded), document cut-outs shrunk to 11–34 px long side
(median 19; our real cards ~12–20 px), rotated, washed out, softened, placed near where IDs/money appear in
that video. `labels/` = loose boxes like our annotator, `labels_tight/` = tight; `manifest.csv` gives
source + bg video (filter by fold for CV). Builder `external/scripts/build_yolo_id.py`, notes in
`external/*/LICENCE_README.txt`, `external/yolo_id/{README,stats}.txt`. **Not trained yet.**

| dataset | licence | status | notes |
|---|---|---|---|
| MIDV-2020 (Smart Engines) | **CC BY-SA 2.5** (license.txt on FTP) | **used**: 1000 templates + 500 of 1000 photos (doc_quad polygons, hand-held/table) | ftp://smartengines.com/midv-2020/; photo.tar 4 GB → fetched subset by byte ranges; clips.tar (10 GB video frames, hand-held) not fetched. ShareAlike → derived set is CC BY-SA |
| lambdaWalker/creditCardDetectionDS (HF) | Apache-2.0 | **used**: 800 synthetic bank cards | tagged `creditcard` in manifest; drop if bank cards ≠ ID |
| MIDV-500 / MIDV-2019 | CC BY-SA 2.5 | usable, not downloaded | same docs as MIDV-2020; **FTP dir midv-500 is anonymously writable and contains planted `Photo.scr`/`info.zip` — do not download from it blindly** |
| IDNet (ASU, Zenodo) | CC0 | usable, not downloaded | synthetic flat docs, 20+ GB per country zip; only via remote-zip partial reads |
| SIDTD (CVC) | CC BY 4.0 (repo) / derived from MIDV-2020 | usable, not downloaded | forged-ID variants of MIDV-2020; no extra value for detection |
| Open Images V7 | CC BY 4.0 ann. / CC BY 2.0 imgs | not used | no passport/ID box class; "Mobile phone" exists but would teach phone = ID |
| DocXPand-25k | CC BY-NC-SA 4.0 | **not usable** | |
| UniDataPro / ud-biometrics synthetic passports & licences (HF) | CC BY-NC-ND 4.0 | **not usable** | |
| cloverx-id Indonesian KTP 24k (HF, CC BY 4.0) | CC BY 4.0 but faces morphed from UTKFace (non-commercial) | **not used (licence chain)** | also 12–22 GB tars |
| BID (Brazilian ID) | no licence stated | not usable | scans only |
| LRDE IDID | research only | not usable | |
| Roboflow Universe: royals `id-card-detection-1` (986 imgs, person+id_card), spoofsense `person-detection-id-card`, acc-3 `bdc` (card, passport), test-8qwew `holo`, ocr-id-card `id_card_detection` | CC BY 4.0 | **needs access** (Roboflow API key) | closest to "held in hand" photos |

### 2026-10-17 — Х-19 relabelled
- **Х-19 is a violation**: at 7–8 s the cashier accepts the cash and starts working before checking the ID → C01, label t=7 s. Now 24 violation /
  18 clean videos. Re-evaluated (with --vlm): in-sample 20/24 found, 1/18 FP (Х-39), acc .88;
  **4-fold CV 16/24 found, 0/18 FP, acc .81**.
- Note: the pipeline flags Х-19 as C05 (cash handed back) at ~22 s — right verdict for the wrong reason
  (wrong type, timecode 15 s off). C01 (missing ID) is still undetected in general (X-2, Х-17, Х-19).
- Labels corrected so far vs `open_labels.xlsx`: Х-35, Х-19 (both clean → violation), see dataset/README.md.

### 2026-10-17 — Х-35 relabelled + "cash return" rule (`exp/cashflow.py`, engine rule 6 → C05)
- **Х-35 is a violation** (team correction; no timestamp yet). Now 23 violation / 19 clean videos.
- Ideas tested (signals per video printed by `python3 exp/cashflow.py [cust_tag]`):
  - "machine used but safe never opened → violation": **rejected**, 8 clean videos also never open the safe.
  - "ID near another person's wrist after counting, before safe opens": **rejected**, the ID is
    legitimately handed back in clean videos (Х-3, Х-16, Х-19) and it adds no new TP.
  - "**cash** near another person's wrist after the **cashier** counted (cashier = longest-present track;
    wrist in machine box ≥0.5 s), before the safe opens, ≥10 samples (1 s)": **kept** → C05.
    New TPs: X-1 (first return 20.1 s, label 22), Х-35. FP: Х-19 (cash reaches the client at 22 s).
- Results (with --vlm):

| detections | TP | FP | FN | TN | acc |
|---|---|---|---|---|---|
| y26s_v1 (in-sample) | 19/23 | 2/19 | 4 | 17 | .86 |
| 4-fold CV (honest) | 15/23 | 1/19 | 8 | 18 | .79 |

Still missed (all detectors): X-2 (count before ID), Х-13 (uncounted into safe), Х-17 (collector without ID),
Х-44 (denominations); with CV additionally Х-15, Х-20, Х-22, Х-40 (detector misses on unseen videos).

### 2026-10-17 — honest evaluation of rule engine v3 (with --vlm)
| detections from | TP | FP | FN | TN | acc |
|---|---|---|---|---|---|
| y26s_v1 (in-sample for 33 videos) | 17 | 1 | 5 | 19 | .86 |
| same, all videos mirrored (`exp/aug/hflip`) | 17 | 1 | 5 | 19 | .86 |
| same, perspective warp + rotate 4° + zoom (`exp/aug/persp`) | 17 | 1 | 5 | 19 | .86 |
| **4-fold CV detectors (`feats/cust_cvb`, each video judged by a detector that never saw it)** | **13** | **0** | 9 | 20 | **.79** |

- Rules are geometry-robust (anchors found per video). The warp is milder than a real new viewpoint.
- CV loses Х-15 (bag), Х-20 (safe_open), Х-22 (key_in_lock), Х-40 (door_open): the bottleneck is now the
  detector on unseen videos, not the rules. Most valuable next data: more frames of safe_open,
  key_out_of_lock, collection_bag, ID-identification, door_open; and any footage from a second angle.
- Speed (service): Х-14 (142 s) = 75 s on an idle GPU, 252 s when GPU 0 is shared with training.

### 2026-10-17 — rule engine v3 + targeted VLM checks; negative results for end-to-end models
**Rule engine v3** (`exp/engine.py --eval --vlm`), same in-sample detector as v2 (CV numbers pending):

| version | TP | FP | FN | TN | acc | recall | precision |
|---|---|---|---|---|---|---|---|
| v2 | 10 | 1 | 12 | 19 | .69 | .45 | .91 |
| + C17 laptop unlocked (Qwen yes/no on laptop crop while the room is empty) | 12 | 1 | 10 | 19 | .74 | .55 | .92 |
| + C14 ≥3 people in the room ≥2 s | 13 | 1 | 9 | 19 | .76 | .59 | .93 |
| + C07/C03 client left, money stays on desk (≥60% of next 4 s, safe not opened); C03 if next client already present | **17** | **1** | 5 | 19 | **.86** | .77 | .94 |

Remaining misses: X-1 (partial sum handed over), X-2 (counting before ID), Х-13 (uncounted money into
safe), Х-17 (collector without ID), Х-44 (mixed denominations). Only FP: Х-39 (detector door_open error).
Caveat: thresholds were chosen looking at these 42 videos; detector in-sample for 33 of them.

**Negative results (do not repeat):**
- *Qwen3.8-27B whole-video judge* (video input 2 fps, 15-class prompt, `exp/qvl/run_checklist.py`):
  38 videos acc .53, recall .33, precision .64. ~60 s/video with 4 parallel.
- *V-JEPA 2 ViT-L window classifier* (`exp/vjepa/`, 2 s windows, logistic on PCA-64, leave-pair-out):
  verdict acc ≤.57 — it learns "cashier stands up / walks away", which happens in clean videos too.
- *Qwen "was an ID shown?"* on desk-crop video: answers "id_card" for every visit (clean and violation)
  → useless for C01/C14-collector. *Qwen "key in lock?"* on safe crop: always "no" (key too small);
  the detector's key_in_lock is better.
- *Laptop screen from pixels* (white share, change vs working frame): does not separate Х-28/Х-29.
**Positive:** *Qwen yes/no "laptop screen unlocked?"* on the laptop crop after the cashier leaves:
unlocked only for Х-28 (+Х-11, also a violation video), locked for all 5 clean videos → rule C17.

Infra: Qwen3.8-27B (vision, Apache-2.0, nailf's weights /home/nailf/vllm/models) served by us as
container `tt-qwen-vl` on GPU 1, port 8901, model name `qwen-vl` (video_url or ≤16 images per prompt).
GPUs 0/1 switched to Default compute mode (were Exclusive_Process) on 2026-10-17 with the user's OK.

### 2026-10-17 — VLM baseline (Gemma 4 grids)
Setup: `exp/vlm/` (`run.py` grids + calls, `eval.py` metrics, `run_all.sh`, raw replies `raw/*.jsonl`,
table `results.md`). Gemma 4 26B-A4B-it on :8887, temp 0, one grid image per call with
`mm_processor_kwargs {"max_soft_tokens": 1120}` (default is 280 image tokens; 1120 works and costs ~+0.5 s).
Frames ffmpeg 2 fps, timestamp burned into each tile. Designs: **W12** 12 tiles @1 fps (12 s window,
stride 6 s, 6×2, 232×424 tiles); **W12c** same, top 62 % of frame cropped (action area, bigger tiles);
**W16** 16 tiles @2 fps (8 s, stride 4 s); **OV** 24 tiles evenly over the whole video (1 call);
**CAND** one W12 window centred on the label time (clean videos: middle) = oracle candidate, tests the
VLM as a verifier. Prompts: P1 = 15 selected classes (visual_cue) + correct deposit sequence; P2 = without
the sequence; `_think` = `enable_thinking: true` (reasoning comes inline in `content`, parser takes the last
`{"violations":…}`). Aggregation: any = any window reports anything; vote2 = class in ≥2 windows;
noC14 = drop the outsider class. Baseline "always violation" = acc 0.52, prec 0.52, rec 1.0.

| variant | agg | acc | prec | rec | FP/20 clean | ts first ±2/±5 s | type ok (any) / TP | s per video (serial) |
|---|---|---|---|---|---|---|---|---|
| W12 P1 | any | 0.60 | 0.58 | 0.82 | 13 | 3/6 | 6/18 | 9 |
| W12 P1 | noC14 | 0.50 | 0.57 | 0.18 | 3 | 1/2 | 3/4 | 9 |
| W12 P2 | any | 0.57 | 0.56 | 0.82 | 14 | 2/6 | 5/18 | 8 |
| W12 P2 | vote2 | 0.55 | 0.57 | 0.59 | 10 | 1/5 | 4/13 | 8 |
| W12c P1 | any | 0.55 | 0.55 | 0.77 | 14 | 4/7 | 7/17 | 9 |
| W16 P1 | any | 0.48 | 0.50 | 0.68 | 15 | 2/4 | 5/15 | 13 |
| W12 P1 think | any | 0.60 | 0.67 | 0.45 | 5 | 3/5 | 5/10 | 108 |
| OV P1 | any | 0.43 | 0.40 | 0.18 | 6 | 0/0 | 1/4 | 2 |
| OV P2 | any | 0.43 | 0.42 | 0.23 | 7 | 0/0 | 1/5 | 1 |
| OV P1 think | any | 0.38 | 0.30 | 0.14 | 7 | 0/1 | 1/3 | 35 |
| CAND P1 (oracle window) | any | 0.76 | 0.83 | 0.68 | 3 | 5/10 | 4/15 | 2 |
| CAND P1 think | any | 0.50 | 0.57 | 0.18 | 3 | 0/3 | 3/4 | 19 |

Findings:
- **No variant beats chance as a detector.** Sliding windows reach recall 0.8 only by flagging 13–15/20
  clean videos; vote2 and thinking cut FPs but recall falls to the same degree. The whole-video overview is
  worse than chance (24 small tiles lose all the detail). Correct-sequence text (P1 vs P2) and the top crop
  change almost nothing; 2 fps (W16) is worse than 1 fps (more calls, more FPs).
- **C14 (outsider) is the main false alarm**: on almost every window with a client it reports "extra
  person", although the prompt says cashier + client is normal; it cannot tell cashier from client and
  swaps roles between windows. Without C14 the model almost never fires (recall 0.18).
- Classes it sees: C12 personal items (Х-24, Х-40, Х-46 hit in some variants, but also 1–3 FP),
  C14 for the real people cases (Х-17, Х-18, Х-30), occasionally C08 cashier away (Х-26) and C16 door
  (Х-32, only with thinking; thinking also adds C16 FPs). Never or almost never: C01 no ID, C02 uncounted to
  safe, C03/C04 mixing, C05 partial amount, C07 money left, C10 safe open (Х-20), C11 key in lock (Х-11,
  Х-22), C17 laptop unlocked (Х-28): the sequence/counting/small-object classes are invisible at grid size.
- Even with an oracle window (CAND) type agreement is 4/15 TP, and the "detection" is mostly C14 again
  (noC14 recall 0.09): the VLM is not a usable classifier of these violation types either.
- Thinking: ~20 s per call, 8/212 W12 calls looped to the 6000-token cap with no answer; more cautious
  (fewer FPs) but misses most violations. Not worth the cost.
- Cost: non-thinking ~1.8 s per call; W12 = 5 calls per video on average (max ~24 for the 142 s video) ≈ 9 s
  serial, ~3 s wall at concurrency 4. Thinking W12 ≈ 108 s serial per video.

Recommendation: do **not** use the VLM as primary detector or classifier. Use it only as a narrow
**yes/no verifier for one specific visual state** proposed by the rule engine (door open vs reference frame,
personal item on desk, "is anybody at the desk"), with a single targeted question and a crop or a pair of
frames, never the 15-class prompt, and never let it raise C14 by itself. A natural use is writing the
human-readable `description` for an event the engine already found, from the 1–3 evidence frames.

### 2026-10-17 — rule engine v2 (Nurlan's main agent)
Setup: `exp/engine.py` (states → rules), features `feats/base` (pose+COCO) + `feats/cust_s_v1`
(detector y26s_v1, **in-sample for 33 of 42 videos → optimistic**; honest numbers come from 4-fold CV,
`feats/cust_cv`, in progress). Run: `python3 exp/engine.py --eval`, `--timeline <stem>`.

| version | TP | FP | FN | TN | recall | precision |
|---|---|---|---|---|---|---|
| v1 naive states (≥3 people, any personal item, door open 3 s…) | 17 | 16 | 5 | 4 | .77 | .52 |
| v2 leave-state + long door/safe + drinks | 10 | 1 | 12 | 19 | .45 | .91 |

v2 catches: Х-15 (bag left), Х-20 (safe open 10 s), Х-22 (key left in lock), Х-24 (book left), Х-26
(money left), Х-30/32/33/40 (door open ≥8 s), Х-46 (cup). Only FP: Х-39 (detector says door_open at
60 s although it is closed — detector error). Misses = "transaction" types: X-1 partial sum,
X-2/Х-34 no ID, Х-13 uncounted money to safe, Х-17 collector without ID, Х-18 3 people, Х-28 laptop
unlocked, Х-38 money left after client, Х-4/47 mixing clients, Х-44 denominations, Х-11 (key left, but
the lights dim at 18 s and the detector then sees "key_out").


### 2026-10-17 — setup
- Detector y26s_v1 (YOLO26-s, 960 px, split v1) training on GPU 0.
- Feature extraction `feats/base` (YOLO26x-pose with ByteTrack + YOLO26x COCO, every 3rd frame) on GPU 1.
- Earlier finding: Gemma 4 judges door open/closed wrongly on a single frame (confident and wrong)
  but correctly when the frame is shown **side by side with a reference frame**. EmbeddingGemma 2
  zero-shot cannot tell door open/closed (margin ≤0.025 cosine).

## Conclusions so far
- **Data facts.** Х-4 = Х-47 and Х-5 = Х-48 are byte-identical files (40 unique videos). Х-4/Х-47 are
  labelled 57 s and 68 s → probably two violation moments. Keep duplicates in the same fold (done in
  `exp/data/folds.txt`; split v1 leaks Х-5/Х-48 → its val mAP is slightly optimistic).
- **Videos come in staged pairs** (same scenario done right / wrong): Х-20/21, Х-22/23, Х-24/25,
  Х-26/27, Х-28/29, Х-30/31, Х-10/11, Х-12/13, Х-14/15, Х-16/17, Х-18/19, Х-34/35, Х-37/38, Х-39/40,
  Х-43/44, Х-45/46 (adjacent numbers; verify per pair). The difference inside a pair IS the violation.
- **Almost every video ends with the cashier walking out**, clean or not. Judge *what is left behind*
  while the room is empty (key in lock, money, bag, items, safe), not the leaving itself.
- **Label timecode = the moment the cashier leaves the workplace** (stands up / walks away from the
  desk/safe), ~5–9 s before the room is empty. Use "last time hips within 0.17 desk-widths of the desk".
- **Х-20 "safe left open"**: the cashier does NOT leave; she takes money and sits down to count with the
  safe open behind her → rule "safe open ≥8 s" (clean videos ≤6.9 s), not only "safe open with no cashier".
- **People behind the glass wall** are detected as persons. "Outside" = box mostly in the door/glass
  area AND shorter than 0.38× door height AND ending above the top of the counting machine.
- COCO "cell phone" on the desk fires in many clean videos (ID cards, phones of clients): don't use it.
  Drinks (cup/bottle) and items left after the cashier leaves (book/handbag) are reliable.
- п.19 allows identification by ID document **or the digital documents service (phone)** — may explain
  why Х-35/Х-37 (no physical ID seen) are clean while X-2 is not. Unresolved.
