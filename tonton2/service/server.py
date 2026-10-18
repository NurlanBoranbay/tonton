"""tonton2 cash-desk analysis service (same API as tonton, new rule engine).

POST /api/v1/analyze (multipart `file`) -> {"job_id"}
GET  /api/v1/jobs/{job_id}             -> {"job_id", "status", "verdict", "violations", ...}
GET  /api/v1/settings                  -> live settings {"scene": {...}, "rules": {rule_id: {...}}}
PUT  /api/v1/settings                  -> validate + save to runs/settings.json; the next job uses them
GET  /api/v1/health                    -> {"status": "ok", ...}   (no key)
GET  /                                 -> web UI

Jobs live on disk in jobs/<id>/ (input.mov, status.json, result.json); a restart re-queues unfinished jobs.
One worker thread runs the GPU pipeline, one job at a time.
"""
import json, os, queue, secrets, shutil, sys, threading, time, traceback

from fastapi import Body, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from app.config import load_settings, save_settings  # noqa: E402

API_KEY = os.environ.get("API_KEY", "")
JOBS = os.environ.get("JOBS_DIR", os.path.join(HERE, "jobs"))
STRIDE = int(os.environ.get("PIPELINE_STRIDE", "3"))
DEVICE = int(os.environ.get("PIPELINE_DEVICE", "0"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 2**20
if not API_KEY:
    sys.exit("API_KEY is not set")
os.makedirs(JOBS, exist_ok=True)

app = FastAPI(title="ForteBank cash-desk video analysis (tonton2)", version="2.0")
q: "queue.Queue[str]" = queue.Queue()
state = {"pipeline": None, "load_error": None, "current": None}


def jpath(jid, name): return os.path.join(JOBS, jid, name)


def read_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(p, obj):
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(p + ".tmp", p)


def set_status(jid, **kw):
    st = read_json(jpath(jid, "status.json"), {})
    st.update(kw)
    write_json(jpath(jid, "status.json"), st)


def check_key(key):
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def worker():
    try:
        from app.pipeline.engine import Pipeline
        state["pipeline"] = Pipeline(device=DEVICE, stride=STRIDE)
    except Exception as e:  # keep serving; jobs fail with the reason
        state["load_error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    while True:
        jid = q.get()
        state["current"] = jid
        t0 = time.time()
        set_status(jid, status="processing", started_at=t0)
        try:
            if state["pipeline"] is None:
                raise RuntimeError(f"pipeline not loaded: {state['load_error']}")
            res = state["pipeline"].analyze(jpath(jid, "input.mov"), load_settings())
            res["processing_time_sec"] = round(time.time() - t0, 1)
            write_json(jpath(jid, "result.json"), res)
            set_status(jid, status="done", finished_at=time.time())
        except Exception as e:
            traceback.print_exc()
            set_status(jid, status="failed", error=f"{type(e).__name__}: {e}", finished_at=time.time())
        finally:
            state["current"] = None
            print(f"job {jid} finished in {time.time() - t0:.1f} s", flush=True)


@app.on_event("startup")
def startup():
    pending = []
    for jid in os.listdir(JOBS):
        st = read_json(jpath(jid, "status.json"))
        if st and st.get("status") == "processing" and os.path.exists(jpath(jid, "input.mov")):
            pending.append((st.get("created_at", 0), jid))
    for _, jid in sorted(pending):
        q.put(jid)
    threading.Thread(target=worker, daemon=True).start()


@app.get("/api/v1/health")
def health():
    return {"status": "ok" if state["pipeline"] else ("error" if state["load_error"] else "loading"),
            "pipeline_loaded": state["pipeline"] is not None, "load_error": state["load_error"],
            "queue": q.qsize(), "busy": state["current"] is not None, "stride": STRIDE}


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...), x_api_key: str = Header(None)):
    check_key(x_api_key)
    jid = secrets.token_hex(8)
    os.makedirs(os.path.join(JOBS, jid))
    size = 0
    with open(jpath(jid, "input.mov"), "wb") as f:
        while chunk := await file.read(2**20):
            size += len(chunk)
            if size > MAX_BYTES:
                f.close(); shutil.rmtree(os.path.join(JOBS, jid))
                raise HTTPException(status_code=413, detail="file too large")
            f.write(chunk)
    if size == 0:
        shutil.rmtree(os.path.join(JOBS, jid))
        raise HTTPException(status_code=400, detail="empty file")
    write_json(jpath(jid, "status.json"), {"status": "processing", "filename": file.filename,
                                           "size_bytes": size, "created_at": time.time()})
    q.put(jid)
    return {"job_id": jid}


@app.get("/api/v1/jobs/{job_id}")
def job(job_id: str, x_api_key: str = Header(None)):
    check_key(x_api_key)
    if not job_id.isalnum():
        raise HTTPException(status_code=404, detail="job not found")
    st = read_json(jpath(job_id, "status.json"))
    if st is None:
        raise HTTPException(status_code=404, detail="job not found")
    out = {"job_id": job_id, "status": st["status"], "filename": st.get("filename")}
    if st["status"] == "done":
        res = read_json(jpath(job_id, "result.json"), {})
        out.update(verdict=res.get("verdict"), violations=res.get("violations", []),
                   processing_time_sec=res.get("processing_time_sec"), facts=res.get("facts"))
    elif st["status"] == "failed":
        out["error"] = st.get("error")
    else:
        out["started"] = "started_at" in st
    return out


@app.get("/api/v1/settings")
def get_settings(x_api_key: str = Header(None)):
    check_key(x_api_key)
    return load_settings()


@app.put("/api/v1/settings")
def put_settings(body: dict = Body(...), x_api_key: str = Header(None)):
    check_key(x_api_key)
    return save_settings(body)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))
