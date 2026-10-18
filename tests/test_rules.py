"""Synthetic timelines per rule + ground truth + settings.  Run: ./run.sh python3 -m unittest discover -s tests"""
import os, tempfile, unittest

from app.config import defaults, load_settings, save_settings, validate_settings
from app.ground_truth import compare_ground_truth, parse_timestamp
from app.pipeline.engine import analyse
from app.rules.base import REGISTRY

DT = 0.1
DOOR = [0.0, 0.02, 0.30, 0.50]
MACHINE = [0.07, 0.47, 0.25, 0.57]
SAFE = [0.65, 0.41, 0.90, 0.58]


def person(pid, box, wrist=None, head=None, standing=False):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h = box[3] - box[1]
    k = [[cx, cy, .9] for _ in range(17)]
    k[0] = [*(head or (cx, box[1] + .05 * h)), .9]
    k[5] = k[6] = [cx, box[1] + .15 * h, .9]           # shoulders
    k[11] = k[12] = [cx, box[1] + .40 * h, .9]         # hips -> torso = .25 h
    for i in (9, 10):
        k[i] = [*(wrist or (cx, cy)), .9]
    for i in (13, 14, 15, 16):                         # knees / ankles: visible only when standing
        k[i] = [cx, box[3] - .05 * h, .9 if standing else 0.0]
    return {"id": pid, "box": box, "conf": .9, "kpts": k}


SEATED = [0.30, 0.45, 0.50, 0.62]                       # cashier in the chair, in front of the desk


def video(seconds, frame, luma=lambda t: 137.0):
    """frame(t) -> (persons, custom [(class, box)], coco [(class, box)]); static door/machine/safe are added."""
    rows = []
    for i in range(int(seconds / DT)):
        t = round(i * DT, 2)
        ps, cu, co = frame(t)
        base = [("counting_machine", MACHINE)]
        if not any(c in ("door_open", "door_closed") for c, _ in cu): base.append(("door_closed", DOOR))
        if not any(c in ("safe_open", "safe_closed") for c, _ in cu): base.append(("safe_closed", SAFE))
        rows.append({"f": i * 3, "t": t, "W": 464, "H": 848, "persons": ps, "frame_luma": luma(t),
                     "custom": [{"c": c, "box": b, "conf": .9} for c, b in cu + base],
                     "coco": [{"c": c, "box": b, "conf": .9} for c, b in co]})
    return rows


def rule_ids(rows):
    return {a.rule_id: a.timecode_sec for a in analyse(rows, "synthetic", defaults())[2]}


class GroundTruth(unittest.TestCase):
    def test_timestamps(self):
        self.assertEqual(parse_timestamp("1минута 8 секунда"), 68.0)
        self.assertEqual(parse_timestamp("22 секунда"), 22.0)
        self.assertEqual(parse_timestamp("2 минуты"), 120.0)
        self.assertIsNone(parse_timestamp(None))

    def test_compare(self):
        gt = {"verdict": "violation", "timestamp_sec": 20.0, "type": "interrupted_transaction"}
        al = [{"rule_id": "cash_return_violation", "timecode_sec": 5.0},
              {"rule_id": "interrupted_transaction", "timecode_sec": 18.0}]
        c = compare_ground_truth(gt, al)
        self.assertTrue(c["verdict_match"] and c["type_match"])
        self.assertEqual(c["timestamp_abs_error_sec"], 2.0)                  # error vs the alarm of the GT type
        c = compare_ground_truth({"verdict": "no_violation", "timestamp_sec": None, "type": None}, [])
        self.assertTrue(c["verdict_match"] and c["type_match"])
        c = compare_ground_truth(gt, al[:1])
        self.assertTrue(c["verdict_match"]); self.assertFalse(c["type_match"])


class Settings(unittest.TestCase):
    def test_merge_and_clamp(self):
        s = validate_settings({"scene": {"near_px": 55, "bogus": 1},
                               "rules": {"open_door_violation": {"min_open_sec": 999, "enabled": False}, "nope": {}}})
        self.assertEqual(s["scene"]["near_px"], 55); self.assertNotIn("bogus", s["scene"])
        self.assertEqual(s["rules"]["open_door_violation"]["min_open_sec"], 60)   # clamped to param_schema max
        self.assertFalse(s["rules"]["open_door_violation"]["enabled"]); self.assertNotIn("nope", s["rules"])
        self.assertEqual(set(s["rules"]), set(REGISTRY))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "settings.json")
            save_settings(s, p)
            self.assertEqual(load_settings(p), s)

    def test_disabled_rule_does_not_fire(self):
        rows = video(14, lambda t: ([person(1, SEATED)], [("door_open", DOOR)], []))
        s = defaults(); s["rules"]["open_door_violation"]["enabled"] = False
        self.assertNotIn("open_door_violation", {a.rule_id for a in analyse(rows, "x", s)[2]})


class Rules(unittest.TestCase):
    def test_open_door(self):
        self.assertIn("open_door_violation", rule_ids(video(14, lambda t: ([person(1, SEATED)], [("door_open", DOOR)], []))))
        short = lambda t: ([person(1, SEATED)], [("door_open", DOOR)] if 2 <= t < 5 else [], [])
        self.assertNotIn("open_door_violation", rule_ids(video(14, short)))

    def test_intruder_and_reflection(self):
        def frame(t, ghost=False):
            ps = [person(1, SEATED)]
            if 2 <= t < 9: ps.append(person(2, [0.05, 0.20, 0.20, 0.62], standing=True))   # first visitor in the door
            if 6 <= t < 9:
                if ghost:   # short "outside" figure while somebody stands in the doorway = mirror image
                    ps.append(person(3, [0.10, 0.18, 0.18, 0.32], standing=True))
                else:
                    ps.append(person(3, [0.12, 0.22, 0.28, 0.64], standing=True))
            return ps, [("door_open", DOOR)] if 1 <= t < 10 else [], []
        r = rule_ids(video(12, frame))
        self.assertAlmostEqual(r["open_door_violation_intruder"], 6.0, delta=.2)
        self.assertNotIn("open_door_violation_intruder", rule_ids(video(12, lambda t: frame(t, ghost=True))))

    def _leave(self, extra):
        """Cashier works at the machine, walks out through the door at 6 s; `extra` objects stay on the desk."""
        def frame(t):
            if t < 5: ps = [person(1, SEATED, wrist=(.15, .52))]
            elif t < 8: ps = [person(1, [0.05, 0.20, 0.25, 0.62], standing=True)]
            else: ps = []
            return ps, ([("door_open", DOOR)] if 6 <= t < 8.5 else []) + extra, []
        return rule_ids(video(14, frame))

    def test_left_key(self):
        r = self._leave([("key_in_lock", [0.66, 0.49, 0.71, 0.52])])
        self.assertAlmostEqual(r["left_key_violation"], 6.0, delta=.2)
        self.assertNotIn("left_key_violation", self._leave([("key_out_of_lock", [0.66, 0.49, 0.71, 0.52])]))

    def test_left_key_lights_dim(self):
        """Х-11: the lights dim, the key becomes invisible and the detector says key_out; nobody touched the lock."""
        lock = [0.66, 0.49, 0.71, 0.52]
        def frame(t, out_at):
            if t < 5: ps = [person(1, SEATED, wrist=(.15, .52))]
            elif t < 8: ps = [person(1, [0.05, 0.20, 0.25, 0.62], standing=True)]
            else: ps = []
            key = ("key_in_lock" if t < out_at else "key_out_of_lock", lock)
            return ps, ([("door_open", DOOR)] if 6 <= t < 8.5 else []) + [key], []
        dim = lambda t: 137.0 if t < 4 else 90.0
        r = rule_ids(video(14, lambda t: frame(t, 4.0), dim))                 # flips exactly at the light switch
        self.assertAlmostEqual(r["left_key_violation"], 6.0, delta=.2)
        r = rule_ids(video(14, lambda t: frame(t, 2.0), dim))                 # key taken out before, in normal light
        self.assertNotIn("left_key_violation", r)

    def test_left_money_and_bag(self):
        r = self._leave([("money_stack", [0.30, 0.50, 0.36, 0.53]), ("collection_bag", [0.40, 0.45, 0.48, 0.52])])
        self.assertIn("left_money_violation", r); self.assertIn("left_collection_bag_violation", r)

    def test_beverage(self):
        money = ("money_stack", [0.30, 0.50, 0.36, 0.53])
        near = lambda t: ([person(1, SEATED)], [money], [("cup", [0.37, 0.49, 0.40, 0.53])] if t >= 3 else [])
        far = lambda t: ([person(1, SEATED)], [money], [("cup", [0.80, 0.70, 0.84, 0.74])] if t >= 3 else [])
        self.assertAlmostEqual(rule_ids(video(8, near))["beverage_violation"], 3.0, delta=.2)
        self.assertNotIn("beverage_violation", rule_ids(video(8, far)))

    def test_open_safe_count(self):
        def frame(t, money):
            w = (.15, .52) if 4 <= t < 6 else (.4, .62)
            cu = [("safe_open", SAFE)] if t < 10 else []
            if money and 4 <= t < 6: cu.append(("money_stack", [0.12, 0.475, 0.18, 0.50]))   # in the machine's upper half
            return [person(1, SEATED, wrist=w)], cu, []
        self.assertAlmostEqual(rule_ids(video(12, lambda t: frame(t, True)))["open_safe_violation"], 0.0, delta=.2)
        self.assertNotIn("open_safe_violation", rule_ids(video(12, lambda t: frame(t, False))))   # tonton3: no money, no count

    def test_left_stamp(self):
        stamp = [("stamp", [0.48, 0.47, 0.53, 0.49])]
        self.assertIn("left_commodities_violation", self._leave(stamp))

    def test_uncounted_safe_deposit(self):
        def frame(t, count):
            w = (.15, .52) if count and 4 <= t < 5 else (.4, .62)
            cu = [("money_stack", [0.30, 0.50, 0.36, 0.53])] if 2 <= t < 9 else []
            if 9 <= t < 11: cu.append(("safe_open", SAFE))
            return [person(1, SEATED, wrist=w)], cu, []
        self.assertAlmostEqual(rule_ids(video(13, lambda t: frame(t, False)))["uncounted_safe_deposit"], 9.0, delta=.2)
        self.assertNotIn("uncounted_safe_deposit", rule_ids(video(13, lambda t: frame(t, True))))

    def test_track_relink(self):
        from app.pipeline.timeline import Timeline
        from app.rules.scene import SceneCompiler
        rows = video(6, lambda t: ([person(1 if t < 3 else 7, SEATED)], [], []))   # ByteTrack id switch
        S = SceneCompiler().compile(Timeline(rows))
        self.assertEqual({pid for row in S.people for pid, _ in row}, {1})


if __name__ == "__main__":
    unittest.main()
