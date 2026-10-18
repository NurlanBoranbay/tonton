#!/usr/bin/env python3
"""V-JEPA 2 ViT-L clip embeddings: windows of WIN s (16 frames), stride 0.5 s -> emb/<stem>.npz {t: centers, e: [N, 1024]}"""
import os, sys, numpy as np, torch, cv2
from transformers import AutoModel, AutoVideoProcessor
M = "/nlp/nurlan/tonton/models/vjepa2-vitl-fpc64-256"; V = "/nlp/nurlan/tonton/dataset/videos"
OUT = "/nlp/nurlan/tonton/exp/vjepa/emb"; os.makedirs(OUT, exist_ok=True)
WIN, STRIDE, NF = 2.0, 0.5, 16
proc = AutoVideoProcessor.from_pretrained(M); model = AutoModel.from_pretrained(M, torch_dtype=torch.float16).cuda().eval()
for v in sorted(os.listdir(V)):
    out = os.path.join(OUT, v[:-4] + ".npz")
    if not v.endswith(".mov") or os.path.exists(out): continue
    cap = cv2.VideoCapture(os.path.join(V, v)); fps = cap.get(cv2.CAP_PROP_FPS); frames = []
    while True:
        ok, im = cap.read()
        if not ok: break
        frames.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (256, 256)))
    frames = np.stack(frames); dur = len(frames) / fps
    ts, es, batch, bt = [], [], [], []
    def flush():
        if not batch: return
        x = proc(batch, return_tensors="pt")["pixel_values_videos"].cuda().half()
        with torch.no_grad(): h = model.get_vision_features(x) if hasattr(model, "get_vision_features") else model(pixel_values_videos=x).last_hidden_state
        es.append(h.float().mean(1).cpu().numpy()); ts.extend(bt); batch.clear(); bt.clear()
    t = WIN / 2
    while t <= dur - WIN / 2 + 1e-6:
        idx = np.linspace((t - WIN / 2) * fps, (t + WIN / 2) * fps - 1, NF).astype(int).clip(0, len(frames) - 1)
        batch.append(list(frames[idx])); bt.append(round(t, 2))
        if len(batch) == 16: flush()
        t += STRIDE
    flush()
    np.savez(out, t=np.array(ts), e=np.concatenate(es)); print(v, len(ts), flush=True)
