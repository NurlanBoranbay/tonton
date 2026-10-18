#!/bin/bash
# 4-fold CV detectors (YOLO26-s, 960) on one GPU, then out-of-fold detections -> feats/cust_cv/
GPU=${1:-1}; T=/nlp/nurlan/tonton
while docker ps --format '{{.Names}}' | grep -q tt-extract-cust; do sleep 10; done
for k in 0 1 2 3; do
  [ -f $T/exp/runs/y26s_cvb$k/weights/best.pt ] || ./train_det.sh $GPU cv$k yolo26s.pt y26s_cvb$k > runs/y26s_cvb$k.log 2>&1
  V=$(sed -n "$((k+1))p" data/folds.txt)
  docker run --rm --gpus "\"device=$GPU\"" --shm-size 8g --name tt-extract-cvb$k -v $T:$T -w $T/exp/weights tonton-train \
    python3 $T/exp/extract.py --tag cust_cvb --stride 3 --custom $T/exp/runs/y26s_cvb$k/weights/best.pt --conf 0.2 --videos "$V" >> runs/extract_cust_cvb.log 2>&1
done
echo CV DONE >> runs/extract_cust_cvb.log
