#!/bin/sh
# all variants, sequential (each uses 4 concurrent requests)
cd "$(dirname "$0")"
python3 run.py --design W12 --prompt P1
python3 run.py --design W12 --prompt P2
python3 run.py --design W12c --prompt P1
python3 run.py --design W16 --prompt P1
python3 run.py --design OV --prompt P1
python3 run.py --design OV --prompt P2
python3 run.py --design OV --prompt P1 --think
python3 run.py --design W12 --prompt P1 --think
python3 run.py --design CAND --prompt P1
python3 run.py --design CAND --prompt P1 --think
