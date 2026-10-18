"""Detectors on every `stride`-th frame -> rows for Timeline (same format as tonton's feats/*.jsonl + laptop luma).

pose  : YOLO26x-pose + ByteTrack (persons, 17 keypoints, track ids)
coco  : YOLO26x COCO (laptop, book, handbag, backpack, cup, bottle, ...); persons come from the pose model
custom: 10-class cash-desk detector (money_stack, collection_bag, counting_machine, safe_*, key_*, door_*, ID)
"""
import cv2

from app.pipeline.timeline import CONF, screen_luma

r4 = lambda v: round(float(v), 4)


def _boxes(res, W, H):
    return [{"c": res.names[int(c)], "box": [r4(b[0] / W), r4(b[1] / H), r4(b[2] / W), r4(b[3] / H)], "conf": r4(s)}
            for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]


def extract_rows(path, pose, coco, cust, stride=3, imgsz=960, conf=.15, cust_conf=.2, luma_upper=.6):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows, f = [], -1
    pose.predictor = None                                  # fresh tracker per video
    while True:
        ok, img = cap.read(); f += 1
        if not ok: break
        if f % stride: continue
        H, W = img.shape[:2]
        row = {"f": f, "t": round(f / fps, 3), "W": W, "H": H, "frame_luma": round(float(img.mean()), 1)}
        r = pose.track(img, persist=True, imgsz=imgsz, conf=conf, classes=[0], verbose=False, tracker="bytetrack.yaml")[0]
        ids = r.boxes.id.tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        kp = r.keypoints.data.tolist() if r.keypoints is not None else [[]] * len(r.boxes)
        row["persons"] = [{"id": int(i) if i is not None else -1, "box": [r4(b[0] / W), r4(b[1] / H), r4(b[2] / W), r4(b[3] / H)],
                           "conf": r4(s), "kpts": [[r4(x / W), r4(y / H), r4(c)] for x, y, c in k]}
                          for b, s, i, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), ids, kp)]
        row["coco"] = [b for b in _boxes(coco(img, imgsz=imgsz, conf=conf, verbose=False)[0], W, H) if b["c"] != "person"]
        row["custom"] = _boxes(cust(img, imgsz=imgsz, conf=cust_conf, verbose=False)[0], W, H)
        row["laptops"] = [{"box": b["box"], "luma": screen_luma(img, b["box"], luma_upper)}
                          for b in row["coco"] if b["c"] == "laptop" and b["conf"] >= CONF["laptop"]]
        rows.append(row)
    cap.release()
    return rows
