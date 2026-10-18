#!/usr/bin/env python3
"""Laptop screen state from pixels: share of bright, low-saturation ('white app window') pixels in the upper
part of the laptop box. Unlocked app ≈ white spreadsheet; Windows lock screen ≈ dark blue photo."""
import subprocess, io
from PIL import Image
def frame(video, t):
    b = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", video, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return Image.open(io.BytesIO(b)).convert("RGB")
def white_share(img, box):
    W, H = img.size; x1, y1, x2, y2 = box
    crop = img.crop((int(x1 * W), int(y1 * H), int(x2 * W), int((y1 + .55 * (y2 - y1)) * H))).convert("HSV")
    px = list(crop.getdata())
    return sum(1 for h, s, v in px if v > 170 and s < 60) / max(len(px), 1)


def screen_change(img_a, img_b, box):
    """Mean absolute pixel difference (0..255) of the screen area between two frames."""
    W, H = img_a.size; x1, y1, x2, y2 = box
    r = (int(x1 * W), int(y1 * H), int(x2 * W), int((y1 + .55 * (y2 - y1)) * H))
    a = list(img_a.crop(r).convert("L").resize((48, 24)).getdata()); b = list(img_b.crop(r).convert("L").resize((48, 24)).getdata())
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)
