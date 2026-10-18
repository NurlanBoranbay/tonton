#!/bin/bash
# Run a command in the tonton image with tonton (read-only) and tonton2 mounted. GPU only if GPU=<id> is set.
G=(); [ -n "$GPU" ] && G=(--gpus "\"device=$GPU\"")
exec docker run --rm -i "${G[@]}" --shm-size 4g -v /nlp/nurlan/tonton:/nlp/nurlan/tonton:ro -v /nlp/nurlan/tonton2:/nlp/nurlan/tonton2 \
  -w /nlp/nurlan/tonton2 -e PYTHONPATH=/nlp/nurlan/tonton2 -e PYTHONDONTWRITEBYTECODE=1 tonton2 "$@"
