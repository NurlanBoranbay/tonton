#!/usr/bin/env python3
"""Call the local Qwen3.8-27B vision server (:8901) with a video clip (ffmpeg re-encode, fps/size control)."""
import base64, json, subprocess, sys, tempfile, time, urllib.request, os
URL = "http://localhost:8901/v1/chat/completions"
def clip(video, start=None, end=None, fps=2, height=640, crop=None):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    vf = [f"fps={fps}"] + ([f"crop={crop}"] if crop else []) + [f"scale=-2:{height}"]
    cmd = ["ffmpeg", "-v", "error", "-y"] + (["-ss", str(start)] if start is not None else []) + ["-i", video] + \
          (["-t", str(end - (start or 0))] if end is not None else []) + ["-vf", ",".join(vf), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp]
    subprocess.run(cmd, check=True)
    b = base64.b64encode(open(tmp, "rb").read()).decode(); os.remove(tmp); return b
def ask(video_b64, prompt, think=False, max_tokens=2000, fps=2):
    body = {"model": "qwen-vl", "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": think},
            "mm_processor_kwargs": {"fps": fps},
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + video_b64}},
                {"type": "text", "text": prompt}]}]}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    m = r["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or m.get("reasoning") or "", round(time.time() - t, 1), r.get("usage", {})
if __name__ == "__main__":
    v, prompt = sys.argv[1], sys.argv[2]
    c, rz, dt, u = ask(clip(v), prompt)
    print(dt, "s", u); print(c)
