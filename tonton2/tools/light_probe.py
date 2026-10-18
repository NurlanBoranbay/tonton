import cv2, numpy as np, sys
from ultralytics import YOLO
m = YOLO("weights/custom_y26s_v1.pt")
for stem in sys.argv[1:]:
    cap = cv2.VideoCapture(f"data/videos/{stem}.mov"); f = -1; ref = None; out = []
    while True:
        ok, img = cap.read(); f += 1
        if not ok: break
        if f % 15: continue
        lum = img.mean()
        ref = ref or lum
        def keys(im):
            r = m(im, imgsz=960, conf=.2, verbose=False)[0]
            return " ".join(f"{r.names[int(c)][4:7]}{s:.2f}" for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()) if r.names[int(c)].startswith("key"))
        g = ref / lum
        fixed = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8) if g > 1.08 else img
        out.append(f"{f/30:5.1f} luma={lum:5.1f} gain={g:4.2f} raw[{keys(img)}] bright[{keys(fixed) if g > 1.08 else '='}]")
    print("==", stem); print("\n".join(out))
