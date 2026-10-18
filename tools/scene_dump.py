"""Print the compiled scene of training videos: python3 tools/scene_dump.py [stem ...] [--cust cust_s_v1]"""
import sys
from app.pipeline.cache import load_rows, stems
from app.pipeline.timeline import Timeline
from app.rules.scene import SceneCompiler
from app.ground_truth import load_gt

args = [a for a in sys.argv[1:] if not a.startswith("--")]
cust = sys.argv[sys.argv.index("--cust") + 1] if "--cust" in sys.argv else "cust_s_v1"
args = [a for a in args if a != cust]
gt = load_gt()
for s in args or stems():
    tl = Timeline(load_rows(s, cust), s)
    S = SceneCompiler().compile(tl)
    g = gt.get(s, {})
    print(f"=== {s} dur={tl.duration:.0f}s GT={g.get('verdict')} t={g.get('timestamp_sec')} type={g.get('type')}")
    print(SceneCompiler.describe(S))
