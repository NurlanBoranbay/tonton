"""Per-frame facts of one video.

Rows come from the detectors (see extract.py), one per sampled frame:
  {"f", "t", "W", "H", "persons": [{"id", "box", "conf", "kpts"}], "coco": [{"c", "box", "conf"}],
   "custom": [{"c", "box", "conf"}], "laptops": [{"box", "luma"}]}
Boxes in rows are normalised (0..1); the Timeline converts everything to pixels.
"""
from dataclasses import dataclass, field

# COCO keypoint indices
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHO, R_SHO, L_WRI, R_WRI, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12
L_KNEE, R_KNEE, L_ANK, R_ANK = 13, 14, 15, 16

# minimum detector confidence per class (custom detector + COCO classes we use)
CONF = {"money_stack": .35, "collection_bag": .35, "counting_machine": .4, "safe_open": .4, "safe_closed": .4,
        "key_in_lock": .3, "stamp": .35, "key_out_of_lock": .35, "door_open": .4, "door_closed": .4, "ID-identification": .3,
        "laptop": .4, "book": .35, "handbag": .35, "backpack": .35, "cup": .35, "bottle": .35, "wine glass": .35,
        "cell phone": .35}
PERSON_CONF = .4


@dataclass
class Person:
    id: int
    box: list                 # x1, y1, x2, y2 px
    conf: float
    kpts: list                # 17 x [x, y, conf] px

    def kp(self, i, kmin=.3):
        k = self.kpts[i] if i < len(self.kpts) else None
        return (k[0], k[1]) if k and k[2] >= kmin else None

    def wrists(self, kmin=.3):
        return [p for p in (self.kp(L_WRI, kmin), self.kp(R_WRI, kmin)) if p]

    def head(self, kmin=.3):
        return [p for p in (self.kp(NOSE, kmin), self.kp(L_EYE, kmin), self.kp(R_EYE, kmin)) if p]


@dataclass
class LaptopFact:
    box: list
    luma: float               # mean luma (0..255) of the upper part of the box (the screen)
    active: bool = False      # set by the scene compiler (hysteresis on luma)


@dataclass
class FrameFact:
    f: int
    t: float
    persons: list = field(default_factory=list)
    objects: dict = field(default_factory=dict)     # class -> [(box px, conf)]
    laptops: list = field(default_factory=list)
    luma: float = None        # mean frame brightness (lighting changes)

    def boxes(self, c):
        return [b for b, _ in self.objects.get(c, [])]


class Timeline:
    def __init__(self, rows, name=""):
        self.name = name
        self.W = rows[0]["W"] if rows else 1
        self.H = rows[0]["H"] if rows else 1
        self.frames = [self._fact(r) for r in rows]
        self.dt = (self.frames[-1].t - self.frames[0].t) / max(len(self.frames) - 1, 1) if self.frames else .1

    def px(self, b):
        return [b[0] * self.W, b[1] * self.H, b[2] * self.W, b[3] * self.H]

    def _fact(self, r):
        ff = FrameFact(r["f"], r["t"], luma=r.get("frame_luma"))
        for p in r.get("persons", []):
            if p["conf"] >= PERSON_CONF:
                ff.persons.append(Person(p["id"], self.px(p["box"]), p["conf"],
                                         [[x * self.W, y * self.H, c] for x, y, c in p["kpts"]]))
        for d in r.get("custom", []) + r.get("coco", []):
            if d["conf"] >= CONF.get(d["c"], 1.1):
                ff.objects.setdefault(d["c"], []).append((self.px(d["box"]), d["conf"]))
        ff.laptops = [LaptopFact(self.px(x["box"]), x["luma"]) for x in r.get("laptops", [])]
        return ff

    def __len__(self):
        return len(self.frames)

    @property
    def duration(self):
        return self.frames[-1].t if self.frames else 0.0


def screen_luma(img_bgr, box_norm, upper=.6):
    """Mean luma of the upper `upper` fraction of a laptop box (the screen seen from behind the cashier)."""
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = int(box_norm[0] * W), int(box_norm[1] * H), int(box_norm[2] * W), int(box_norm[3] * H)
    y2 = y1 + max(1, int((y2 - y1) * upper))
    crop = img_bgr[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    b, g, r = crop[..., 0].mean(), crop[..., 1].mean(), crop[..., 2].mean()
    return round(float(.114 * b + .587 * g + .299 * r), 1)
