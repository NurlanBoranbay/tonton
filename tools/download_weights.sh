#!/bin/sh
# Download the model weights from Hugging Face into weights/.
set -e
cd "$(dirname "$0")/../weights"
HF=https://huggingface.co/nurlanboranbay/tonton/resolve/main/tonton3
for f in yolo26x-pose.pt yolo26x.pt custom_y26s_all_stamp.pt; do
  [ -s "$f" ] || curl -fL -o "$f" "$HF/$f"
done
ls -lh *.pt
