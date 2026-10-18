# Box annotator (YOLO)

Browser tool for drawing boxes of the 9 custom classes on frames of `../dataset/videos`.
Each save writes `../dataset/yolo/images/<stem>.jpg` + `labels/<stem>.txt` (YOLO format)
and a line in `annotations.jsonl`. `data.yaml` and `classes.txt` are written on start.

Run (on gpu-server):

    cd /nlp/nurlan/tonton/annotator
    nohup python3 serve.py --port 8097 > serve.log 2>&1 &                 # LAN: http://10.205.205.27:8097/
    nohup cloudflared tunnel --no-autoupdate --url http://localhost:8097 > tunnel.log 2>&1 &
    grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log          # public URL (changes on every restart)

No password by default (team decision). To require one: `ANNOTATOR_PASSWORD=... python3 serve.py`.
Stop: `pkill -f "serve.py --port 8097"; pkill -f "cloudflared tunnel --no-autoupdate --url http://localhost:8097"`.

Before training: split `images/` into train/val (by video, not by frame — neighbouring frames
are near-duplicates) and fix `train:`/`val:` in data.yaml.
