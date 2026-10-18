#!/usr/bin/env python3
"""Test the tonton3 API: upload videos, wait for the results, print the violations.

Needs only Python 3.8+ (no packages).

  python3 test_api.py --url https://<host> --key <API_KEY> video1.mov [video2.mov ...]
  python3 test_api.py --url https://<host> --key <API_KEY> --out results/ *.mov     # also save each result JSON

The key can also come from the API_KEY environment variable.
"""
import argparse, json, mimetypes, os, sys, time, uuid
import urllib.error, urllib.request


def request(method, url, key, body=None, headers=None, timeout=120):
    req = urllib.request.Request(url, data=body, method=method, headers={"X-API-Key": key, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"detail": e.read().decode(errors="replace")}


def upload(base, key, path):
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path)[0] or "video/quicktime"
    with open(path, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return request("POST", f"{base}/analyze", key, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=600)


def wait(base, key, job_id, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        code, j = request("GET", f"{base}/jobs/{job_id}", key)
        if code != 200:
            return code, j
        if j.get("status") in ("done", "failed"):
            return code, j
        time.sleep(2)
    return 0, {"status": "timeout"}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Cyrillic on Windows consoles
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--url", required=True, help="service address, e.g. https://xxx.trycloudflare.com")
    ap.add_argument("--key", default=os.environ.get("API_KEY"), help="API key (or set API_KEY)")
    ap.add_argument("--out", help="folder to save each result as <video>.json")
    a = ap.parse_args()
    if not a.key:
        sys.exit("give --key or set API_KEY")
    base = a.url.rstrip("/")
    base = base if base.endswith("/api/v1") else base + "/api/v1"

    try:
        with urllib.request.urlopen(f"{base}/health", timeout=30) as r:
            h = json.loads(r.read())
    except Exception as e:
        sys.exit(f"service not reachable: {e}")
    print(f"health: {h.get('status')} (pipeline loaded: {h.get('pipeline_loaded')}, queue: {h.get('queue')})")
    code, _ = request("GET", f"{base}/jobs/nonexistent", "wrong-key")
    print(f"security check: a request with a fake key is {'rejected (HTTP 401), OK' if code == 401 else f'NOT rejected (HTTP {code})'}\n")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
    summary = []
    for path in a.videos:
        name = os.path.basename(path)
        if not os.path.isfile(path):
            print(f"{name}: file not found ({os.path.abspath(path)})\n")
            summary.append((name, "not found", 0, 0))
            continue
        t0 = time.time()
        code, r = upload(base, a.key, path)
        if code != 200:
            print(f"{name}: upload failed, HTTP {code}: {r.get('detail')}\n")
            summary.append((name, "upload failed", 0, 0))
            continue
        code, j = wait(base, a.key, r["job_id"])
        dt = time.time() - t0
        if j.get("status") != "done":
            print(f"{name}: {j.get('status')} {j.get('error', '')}\n")
            summary.append((name, j.get("status"), 0, dt))
            continue
        vs = j.get("violations", [])
        print(f"{name}: {j['verdict'].upper()}  ({len(vs)} violation(s), {dt:.0f} s, job {r['job_id']})")
        for i, v in enumerate(vs, 1):
            rule = v.get("rule") or {}
            print(f"  {i}. {v['timecode_sec']:6.1f} s  [{v.get('class_id', '')}] {v['type']}")
            print(f"     {v['description']}")
            print(f"     frames: {len(v.get('evidence_frames', []))}  basis: {rule.get('citation') or '-'}")
        print()
        if a.out:
            with open(os.path.join(a.out, os.path.splitext(name)[0] + ".json"), "w", encoding="utf-8") as f:
                json.dump(j, f, ensure_ascii=False, indent=1)
        summary.append((name, j["verdict"], len(vs), dt))

    print("summary:")
    for name, verdict, n, dt in summary:
        print(f"  {name:30} {verdict:14} {n} violation(s)  {dt:5.0f} s")


if __name__ == "__main__":
    main()
