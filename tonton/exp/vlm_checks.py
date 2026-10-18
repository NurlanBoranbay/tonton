#!/usr/bin/env python3
"""Narrow yes/no VLM checks on a crop (Qwen3.8-27B vision on :8901). Each returns (answer, raw text)."""
import base64, io, json, re, urllib.request
URL = "http://localhost:8901/v1/chat/completions"
def ask_image(img, prompt, max_tokens=200):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=92)
    body = {"model": "qwen-vl", "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
                                                      {"type": "text", "text": prompt}]}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120))
    return r["choices"][0]["message"]["content"] or ""
def crop(img, box, pad=.04, size=448):
    W, H = img.size
    c = img.crop((int(max(0, box[0] - pad) * W), int(max(0, box[1] - pad) * H), int(min(1, box[2] + pad) * W), int(min(1, box[3] + pad) * H)))
    s = size / max(c.size); return c.resize((int(c.size[0] * s), int(c.size[1] * s)))
def laptop_unlocked(img, box):
    t = ask_image(crop(img, box), 'Look at the laptop screen. Is it showing an open application/document window with work content (unlocked), '
                  'or a lock screen / login screen / wallpaper / dark or off screen (locked)? Answer JSON only: {"screen": "unlocked"|"locked"|"unclear", "evidence": "..."}')
    m = re.search(r'"screen"\s*:\s*"(\w+)"', t); return (m.group(1) if m else "unclear"), t
def key_in_lock(img, box):
    t = ask_image(crop(img, box, pad=.03), 'This is a safe (metal cabinet) in a bank cash room. Is a KEY inserted in the safe door lock '
                  '(a small key or key ring sticking out of the keyhole)? Answer JSON only: {"key_in_lock": "yes"|"no"|"unclear", "evidence": "..."}')
    m = re.search(r'"key_in_lock"\s*:\s*"(\w+)"', t); return (m.group(1) if m else "unclear"), t
