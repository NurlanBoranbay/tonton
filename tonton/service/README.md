# ForteBank cash-desk video analysis service

FastAPI wrapper around `exp/pipeline.py` (`Pipeline().analyze(video)`) + a one-page Russian web UI.
Pipeline logic lives in `exp/` (pipeline.py / engine.py / extract.py); this folder only serves it.

- `app.py` — API + job queue. Models are loaded once at startup; **one worker thread** runs one GPU job at a
  time. Jobs are stored on disk in `jobs/<job_id>/` (`input.mov`, `status.json`, `result.json`); jobs still
  `processing` when the container stops are re-queued on start.
- `static/index.html` — web UI at `/`.
- `Dockerfile` — `tonton-train` + fastapi/uvicorn/python-multipart → image `tonton-service`.
- `docker-compose.yml` — `api` (GPU 0, host network, port **8105**) and `tunnel` (cloudflared quick tunnel,
  `--protocol http2`; QUIC uploaded ~10× slower and a 29 MB upload hit Cloudflare's 100 s timeout).
- `.env` (mode 600, not to be shared) — `API_KEY=...`.

## Run / stop

```bash
cd /nlp/nurlan/tonton/service
docker compose up -d --build        # start (restart: unless-stopped)
docker compose logs -f api          # logs ("job <id> finished in N s")
docker compose restart api          # after changing exp/*.py (models + code are loaded at start)
docker compose down                 # stop both
docker logs tonton-tunnel 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1   # public URL
```

Model loading takes ~1–2 min after start; `GET /api/v1/health` shows `"status": "loading"` until then.
The public URL is a Cloudflare quick tunnel: it **changes every time the `tunnel` container is recreated**
(restart of the `api` container does not change it). The VLM check (laptop screen, C17) needs Qwen on
`localhost:8901` (container `tt-qwen-vl`); if it is down, the pipeline falls back to rules without it.

Config (environment in `docker-compose.yml`): `PORT` (8105), `PIPELINE_STRIDE` (3 = every 3rd frame),
`PIPELINE_DEVICE` (0), `MAX_UPLOAD_MB` (500).

## API

All `/api/v1/analyze` and `/api/v1/jobs/*` calls require header `X-API-Key` (401 otherwise).

```bash
URL=https://<tunnel>.trycloudflare.com      # or http://localhost:8105
KEY=$(cut -d= -f2 /nlp/nurlan/tonton/service/.env)

curl -X POST $URL/api/v1/analyze -H "X-API-Key: $KEY" -F "file=@video.mov"
# {"job_id":"9e327c00180c166b"}

curl -H "X-API-Key: $KEY" $URL/api/v1/jobs/9e327c00180c166b
# {"job_id":"...","status":"processing","started":true}
# {"job_id":"...","status":"done","verdict":"violation","violations":[
#    {"type":"Ключ оставлен в замке сейфа","timecode_sec":31.2,"description":"...",
#     "evidence_frames":[{"timestamp_sec":31.2,"image_base64":"/9j/..."}, ...],
#     "class_id":"C11","interval_sec":[..,..],
#     "rule":{"act":"...","citation":"V1900019680 п.…","quote":"...","link":"https://old.adilet.zan.kz/...","name_ru":"...","rule_match":"direct"}}],
#  "processing_time_sec":57.0,"filename":"video.mov","facts":{...}}
# {"job_id":"...","status":"failed","error":"..."}

curl $URL/api/v1/health      # no key: {"status":"ok","pipeline_loaded":true,"queue":0,"busy":false,...}
```

Required fields are exactly as in the case spec; `class_id`, `interval_sec`, `rule`, `filename`,
`processing_time_sec`, `facts` are extra.

## Web UI

Open `$URL/` → enter the API key once (kept in the browser's localStorage; the page calls the same API with
it, the server holds no unauthenticated proxy) → drop a video → status while processing → verdict
«Нарушение не обнаружено» / «Обнаружены признаки возможных нарушений» → per violation: type, clickable
timecode (seeks the embedded player, which plays the local file — the video is not downloaded back),
description, evidence frames (click to enlarge), act + point + verbatim quote + link to adilet.

## Measured timing (2026-10-17, stride 3, through the public tunnel)

| video | length | upload | processing | end-to-end | note |
|---|---|---|---|---|---|
| Х-14.mov (29 MB) | 142 s | 8.5 s | 252 s | 262 s | GPU 0 shared with 2 YOLO trainings + 1 extraction |
| Х-14.mov, CLI on idle GPU 1 | 142 s | — | 75 s | — | same pipeline, no contention |
| X-1.mov | 26 s | 2.9 s | 52 s | 56 s | GPU 0 contended |
| Х-22.mov | 21 s | 1.8 s | 34 s | 37 s | GPU 0 contended; C11 key in lock found |

Processing time is dominated by per-frame detection (3 YOLO models on every 3rd frame) and scales with GPU
load. If GPU 0 is busy at demo time, set `device_ids: ["1"]` in docker-compose.yml (the service needs
~2.3 GB of GPU memory) or raise `PIPELINE_STRIDE` (not evaluated for accuracy).
