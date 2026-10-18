#!/bin/bash
# Usage: train_det.sh <gpu> <data_name> <model> <run_name> [extra yolo args]
# The tonton folder is mounted at its host path, so host paths in data.yaml work inside.
GPU=$1; DATA=$2; MODEL=$3; NAME=$4; shift 4
T=/nlp/nurlan/tonton
docker run --rm --gpus "\"device=$GPU\"" --shm-size 16g --name "tt-$NAME" -e YOLO_OFFLINE=0 \
  -v $T:$T -w $T/exp/weights tonton-train \
  yolo detect train data=$T/exp/data/$DATA/data.yaml model=$MODEL project=$T/exp/runs name=$NAME exist_ok=True \
  imgsz=960 epochs=100 patience=25 batch=16 workers=8 cos_lr=True close_mosaic=10 fliplr=0.5 degrees=5 plots=True "$@"
