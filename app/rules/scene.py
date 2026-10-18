"""Scene compiler: one pass over a video's Timeline -> SceneState that every rule reads.

What it tracks (all per sampled frame, aligned with timeline.frames):
  people     in-room persons (people seen through the glass wall are dropped), relinked track ids (pid),
             a hysteresis people count and the empty-room flag
  roles      cashier (sticky: longest-present, seated, works at machine/safe/laptop; later tracks in the
             cashier's seat inherit the role), customer (everyone else), collector (customer next to the bag)
  door FSM   closed/open with gap tolerance; open runs; heads inside the open door
  safe FSM   closed/open/unknown; open runs; open->close cycles
  ID         persistent ID-identification (min hits, gap tolerant)
  money      money_stack tracks with last known position (kept after the track is lost)
  counting   machine reach log per person (palm in machine box), count events (>= K samples) + counted piles
  laptop     screen luma with hysteresis -> active
"""
import collections
from dataclasses import dataclass, field

from app.rules import predicates as P

DEFAULTS = {
    "people_win_sec": .5,            # median window for the people count (± s)
    "people_absent_sec": 1.0,        # room counts as empty after this long with nobody
    "relink_gap_sec": 2.0,           # join a new track to one that vanished this recently ...
    "relink_iou": .2,                # ... if their boxes overlap this much
    "cashier_min_presence_sec": 3.0,
    "seat_radius_px": 90,            # later tracks whose median position is this close to the cashier's seat are the cashier
    "seated_height_to_torso": 2.6,   # standing = box height >= this x torso length (and legs visible)
    "ankle_visibility_min": .3,
    "door_gap_sec": 1.0, "door_min_sec": .5,
    "safe_gap_sec": 1.0, "safe_min_sec": .5,
    "id_min_hits": 3, "id_gap_sec": 2.0,
    "palm_reach": .3,                # palm = wrist + reach x (wrist - elbow)
    "machine_margin_px": 4,
    "count_samples": 5,
    "count_money_upper": 0,          # Х-39: a count needs money in the upper half of the machine box. 1 globally: CV .905 -> .810, so off; used by open_safe only              # palm in the machine box for >= K samples (gap tolerant) = a count
    "reach_gap_sec": .4,             # gaps shorter than this inside one machine reach are bridged
    "money_link_px": 45,             # money detection joins the track whose last box is this close
    "laptop_luma_on": 130, "laptop_luma_off": 120,
    "near_px": 40,
    "door_pass_cover": .7,
    "light_jump": .15,               # frame brightness change vs 0.5 s earlier that counts as the lights switching
    "light_settle_sec": 1.5,         # detections this close to a lighting jump are not trusted
    "light_carry_agree": .7,         # carry the pre-jump key state only if this share of detections before it agree
    "lock_touch_px": 5,              # a wrist this close to the lock ...
    "lock_touch_samples": 3,         # ... for this many samples in a row = somebody handles the key           # a person is "in the door" if the head is in the door box or the box is this much inside it
}


@dataclass
class Track:
    pid: int
    first_t: float
    last_t: float
    n: int = 0
    standing: int = 0
    seat: tuple = (0.0, 0.0)          # median hip/box-bottom position
    work: int = 0                     # samples with a palm in machine / safe / laptop box
    bag: int = 0                      # samples with a palm next to the collection bag
    head_in_door: int = 0
    role: str = "customer"

    @property
    def presence(self):
        return self.last_t - self.first_t


@dataclass
class Count:
    t0: float
    t1: float
    i0: int
    i1: int
    pid: int                          # who had the palm in the machine
    piles: list = field(default_factory=list)   # money track ids at the machine / in that hand


@dataclass
class SceneState:
    ts: list
    params: dict
    door: list = None
    safe: list = None
    machine: list = None
    laptop: list = None
    desk: list = None
    people: list = field(default_factory=list)          # per frame [(pid, Person)]
    n_room: list = field(default_factory=list)
    empty: list = field(default_factory=list)
    tracks: dict = field(default_factory=dict)
    door_open: list = field(default_factory=list)
    door_runs: list = field(default_factory=list)
    safe_state: list = field(default_factory=list)      # "open" | "closed" | "unknown"
    safe_runs: list = field(default_factory=list)
    id_seen: list = field(default_factory=list)
    id_runs: list = field(default_factory=list)
    money: list = field(default_factory=list)           # per frame [(money track id, box)]
    money_tracks: dict = field(default_factory=dict)    # id -> {"first_t", "last_t", "last_box", "n"}
    reaches: dict = field(default_factory=dict)         # pid -> [(t0, t1, i0, i1)]
    counts: list = field(default_factory=list)
    laptop_active: list = field(default_factory=list)
    all_people: list = field(default_factory=list)      # per frame [(raw track id, Person, outside?)] incl. people behind the glass
    key_in: list = field(default_factory=list)          # key in the safe lock (lighting-aware, see _key)
    light_jumps: list = field(default_factory=list)     # times of sudden lighting changes
    alias: dict = field(default_factory=dict)           # raw track id -> pid
    departure: tuple = None                             # (t_door_open, t_door_close, i, j, leaver pid): staff left, room stays empty

    def role(self, pid):
        return self.tracks[pid].role if pid in self.tracks else "customer"

    def idx(self, t):
        """Index of the first sample at or after t."""
        for i, x in enumerate(self.ts):
            if x >= t - 1e-6: return i
        return len(self.ts) - 1

    def cashiers(self):
        return {p for p, tr in self.tracks.items() if tr.role == "cashier"}

    def others_at(self, i):
        return [(pid, p) for pid, p in self.people[i] if self.role(pid) != "cashier"]

    def safe_cycles(self):
        """[(t_open, t_close)] for every open run that is followed by a closed state."""
        out = []
        for a, b, i, j in self.safe_runs:
            if any(s == "closed" for s in self.safe_state[j + 1:]): out.append((a, b))
        return out


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _median_box(tl, classes):
    bs = [b for ff in tl.frames for c in classes for b in ff.boxes(c)]
    return [_median([b[k] for b in bs]) for k in range(4)] if bs else None


class SceneCompiler:
    def __init__(self, params=None):
        self.p = {**DEFAULTS, **(params or {})}

    def compile(self, tl):
        p = self.p
        S = SceneState(ts=[ff.t for ff in tl.frames], params=p)
        S.door = _median_box(tl, ("door_open", "door_closed"))
        S.safe = _median_box(tl, ("safe_open", "safe_closed"))
        S.machine = _median_box(tl, ("counting_machine",))
        S.laptop = _median_box(tl, ("laptop",))
        anchors = [b for b in (S.machine, S.laptop) if b]
        S.desk = P.expand([min(b[0] for b in anchors), min(b[1] for b in anchors), max(b[2] for b in anchors),
                           max(b[3] for b in anchors)], .08 * tl.H) if anchors else None
        self._people(tl, S)
        self._roles(tl, S)
        self._door(tl, S)
        self._safe(tl, S)
        self._id(tl, S)
        self._money(tl, S)
        self._counting(tl, S)
        self._laptop(tl, S)
        self._key(tl, S)
        self._departure(tl, S)
        return S

    # ---- people -------------------------------------------------------------------------------------
    def _outside(self, S, person):
        """Person behind the glass wall: box mostly in the door/glass area, short vs the door, and ending above
        the top of the counting machine (somebody inside always reaches down to desk level)."""
        b, d = person.box, S.door
        if d is None or P.cover(b, P.expand(d, .05 * (d[3] - d[1]))) <= .5:
            return False
        short = (b[3] - b[1]) < .38 * (d[3] - d[1])
        return short and S.machine is not None and b[3] < S.machine[1] + 8

    def _people(self, tl, S):
        p = self.p
        alias, last = {}, {}                      # raw track id -> pid; pid -> (t, box)
        for ff in tl.frames:
            row = []
            present = {alias.get(x.id) for x in ff.persons if x.id in alias}
            S.all_people.append([(x.id, x, self._outside(S, x)) for x in ff.persons])
            for x in ff.persons:
                if self._outside(S, x):
                    continue
                if x.id >= 0 and x.id not in alias:
                    cand = [(P.iou(box, x.box), pid) for pid, (t, box) in last.items()
                            if pid not in present and 0 < ff.t - t <= p["relink_gap_sec"]]
                    best = max(cand, default=(0, None))
                    alias[x.id] = best[1] if best[0] >= p["relink_iou"] else x.id
                    present.add(alias[x.id])
                pid = alias.get(x.id, -1)
                if pid >= 0:
                    last[pid] = (ff.t, x.box)
                row.append((pid, x))
            S.people.append(row)
        S.alias = alias
        raw = [len(r) for r in S.people]
        w = max(1, int(round(p["people_win_sec"] / tl.dt)))
        S.n_room = [_median(raw[max(0, i - w): i + w + 1]) for i in range(len(raw))]
        absent = P.runs([n == 0 for n in S.n_room], S.ts, min_dur=p["people_absent_sec"])
        S.empty = [False] * len(raw)
        for a, b, i, j in absent:
            for k in range(i, j + 1): S.empty[k] = True

    def _roles(self, tl, S):
        p = self.p
        seats = collections.defaultdict(list)
        work_boxes = [b for b in (S.machine, S.safe, S.laptop) if b]
        bags = [ff.boxes("collection_bag") for ff in tl.frames]
        for i, row in enumerate(S.people):
            for pid, x in row:
                if pid < 0: continue
                tr = S.tracks.setdefault(pid, Track(pid, S.ts[i], S.ts[i]))
                tr.last_t, tr.n = S.ts[i], tr.n + 1
                tr.standing += P.standing(x, p["seated_height_to_torso"], p["ankle_visibility_min"])
                seats[pid].append((P.center(x.box)[0], x.box[3]))
                tr.work += any(P.wrist_in_bbox(x, b, p["machine_margin_px"], p["palm_reach"]) for b in work_boxes)
                tr.bag += any(P.wrist_near(x, b, p["near_px"], p["palm_reach"]) for b in bags[i])
                tr.head_in_door += bool(S.door) and P.head_in_bbox(x, S.door)
        for pid, tr in S.tracks.items():
            tr.seat = (_median([s[0] for s in seats[pid]]), _median([s[1] for s in seats[pid]]))
        cand = [tr for tr in S.tracks.values() if tr.presence >= p["cashier_min_presence_sec"]]
        if not cand:
            return
        # score: time present, sitting, working at machine/safe/laptop
        main = max(cand, key=lambda tr: tr.n * (1.5 - tr.standing / tr.n) + 2 * tr.work)
        main.role = "cashier"
        for tr in S.tracks.values():                       # sticky: whoever sits in the cashier's seat later
            if tr is main: continue
            overlap = min(tr.last_t, main.last_t) - max(tr.first_t, main.first_t)
            d = ((tr.seat[0] - main.seat[0]) ** 2 + (tr.seat[1] - main.seat[1]) ** 2) ** .5
            if overlap <= .5 and d <= p["seat_radius_px"] and tr.standing / tr.n < .5:
                tr.role = "cashier"
        for tr in S.tracks.values():
            if tr.role != "cashier" and tr.bag >= 3:
                tr.role = "collector"

    # ---- door / safe / ID -----------------------------------------------------------------------------
    def _door(self, tl, S):
        p = self.p
        raw = []
        for ff in tl.frames:
            o = max((c for _, c in ff.objects.get("door_open", [])), default=0)
            c = max((c for _, c in ff.objects.get("door_closed", [])), default=0)
            raw.append(o > c)
        S.door_runs = P.runs(raw, S.ts, min_dur=p["door_min_sec"], max_gap=p["door_gap_sec"])
        S.door_open = [False] * len(raw)
        for a, b, i, j in S.door_runs:
            for k in range(i, j + 1): S.door_open[k] = True

    def _safe(self, tl, S):
        p = self.p
        raw = []
        for ff in tl.frames:
            o = max((c for _, c in ff.objects.get("safe_open", [])), default=0)
            c = max((c for _, c in ff.objects.get("safe_closed", [])), default=0)
            raw.append("open" if o > c else "closed" if c > 0 else "unknown")
        S.safe_runs = P.runs([s == "open" for s in raw], S.ts, min_dur=p["safe_min_sec"], max_gap=p["safe_gap_sec"])
        is_open = [False] * len(raw)
        for a, b, i, j in S.safe_runs:
            for k in range(i, j + 1): is_open[k] = True
        S.safe_state = ["open" if is_open[i] else ("unknown" if raw[i] == "unknown" else "closed") for i in range(len(raw))]

    def _id(self, tl, S):
        p = self.p
        hits = [bool(ff.boxes("ID-identification")) for ff in tl.frames]
        S.id_runs = [r for r in P.runs(hits, S.ts, max_gap=p["id_gap_sec"])
                     if sum(hits[r[2]: r[3] + 1]) >= p["id_min_hits"]]
        S.id_seen = [False] * len(hits)
        for a, b, i, j in S.id_runs:
            for k in range(i, j + 1): S.id_seen[k] = True

    # ---- money / counting / laptop --------------------------------------------------------------------
    def _money(self, tl, S):
        p, nxt = self.p, 0
        for i, ff in enumerate(tl.frames):
            row, used = [], set()
            for b in ff.boxes("money_stack"):
                live = [(P.gap(b, m["last_box"]), mid) for mid, m in S.money_tracks.items()
                        if mid not in used and ff.t - m["last_t"] <= 3.0]
                d, mid = min(live, default=(1e9, None))
                if d > p["money_link_px"]:
                    mid, nxt = nxt, nxt + 1
                    S.money_tracks[mid] = {"first_t": ff.t, "n": 0}
                m = S.money_tracks[mid]
                m.update(last_t=ff.t, last_box=b, n=m["n"] + 1)
                used.add(mid); row.append((mid, b))
            S.money.append(row)

    def _counting(self, tl, S):
        p = self.p
        if not S.machine:
            return
        m = S.machine
        hit = collections.defaultdict(lambda: [False] * len(S.ts))
        for i, row in enumerate(S.people):
            for pid, x in row:
                if pid >= 0 and P.wrist_in_bbox(x, m, p["machine_margin_px"], p["palm_reach"]):
                    hit[pid][i] = True
        for pid, flags in hit.items():
            S.reaches[pid] = P.runs(flags, S.ts, max_gap=p["reach_gap_sec"])
            for a, b, i, j in S.reaches[pid]:
                if sum(flags[i: j + 1]) < p["count_samples"]:
                    continue
                if p["count_money_upper"] and not self._money_upper(S, m, i, j):
                    continue
                piles = set()
                for k in range(i, j + 1):
                    hand = [x for q, x in S.people[k] if q == pid]
                    for mid, box in S.money[k]:
                        if P.cover(box, P.expand(m, p["near_px"] / 2)) > .3 or \
                                any(P.wrist_near(x, box, p["near_px"], p["palm_reach"]) for x in hand):
                            piles.add(mid)
                S.counts.append(Count(a, b, i, j, pid, sorted(piles)))
        S.counts.sort(key=lambda c: c.t0)

    @staticmethod
    def _money_upper(S, m, i, j):
        """Money in the counting machine's upper half (the feed tray) at some sample of the reach."""
        up = [m[0], m[1], m[2], (m[1] + m[3]) / 2]
        return any(P.inside(P.center(box), up) for k in range(i, j + 1) for _, box in S.money[k])

    def _laptop(self, tl, S):
        p, on = self.p, False
        for ff in tl.frames:
            lum = max((x.luma for x in ff.laptops if not S.laptop or P.iou(x.box, S.laptop) > .3), default=None)
            if lum is not None:
                on = lum >= p["laptop_luma_on"] or (on and lum > p["laptop_luma_off"])
            S.laptop_active.append(on)
            for x in ff.laptops: x.active = on

    def _key(self, tl, S):
        """Key in the safe lock. When the lights switch (Х-11: 137 -> 90 mean brightness) the small key becomes
        invisible and the detector flips to key_out. A key cannot leave the lock by itself, so across a lighting jump
        the state from before the jump is kept until somebody's hand comes to the lock."""
        p = self.p
        lum = [ff.luma for ff in tl.frames]
        back = max(1, int(round(.5 / tl.dt)))
        S.light_jumps = []
        for k in range(back, len(lum)):
            if lum[k] and lum[k - back] and abs(lum[k] - lum[k - back]) / lum[k - back] > p["light_jump"]:
                if not S.light_jumps or S.ts[k] - S.light_jumps[-1] > p["light_settle_sec"] * 2:
                    S.light_jumps.append(S.ts[k - back])
        state, lock, carry, touching, raw_hist, used = [], None, None, 0, [], set()
        for k, ff in enumerate(tl.frames):
            ki = max((c for _, c in ff.objects.get("key_in_lock", [])), default=0)
            ko = max((c for _, c in ff.objects.get("key_out_of_lock", [])), default=0)
            if ki or ko:
                lock = (ff.boxes("key_in_lock") or ff.boxes("key_out_of_lock"))[0]
            raw = "in" if ki > ko else "out" if ko else None
            t = ff.t
            jump = next((j for j in S.light_jumps if 0 <= t - j <= p["light_settle_sec"]), None)
            if carry is None and jump is not None and jump not in used:
                used.add(jump)
                before = [st for tt, st in raw_hist if jump - p["light_settle_sec"] <= tt <= jump and st]
                top = max(set(before), key=before.count) if before else None
                if top and before.count(top) / len(before) >= p["light_carry_agree"]:
                    carry = top                                # the stable state just before the lights changed
            raw_hist.append((t, raw))
            touch = lock is not None and any(P.wrist_near(x, lock, p["lock_touch_px"]) for _, x in S.people[k])
            touching = touching + 1 if touch else 0
            if carry is not None and touching >= p["lock_touch_samples"] and jump is None:
                carry = None                                   # somebody handles the key after the jump: trust the detector again
            cur = carry if carry is not None else raw          # outside a lighting jump: plain detections
            state.append((t, cur))
        S.key_in = [st == "in" for _, st in state]

    def in_door(self, S, x):
        return bool(S.door) and (P.head_in_bbox(x, S.door) or P.cover(x.box, S.door) >= self.p["door_pass_cover"])

    def working(self, S, x):
        return any(P.wrist_in_bbox(x, b, self.p["machine_margin_px"], self.p["palm_reach"])
                   for b in (S.machine, S.safe, S.laptop) if b)

    def _departure(self, tl, S):
        """First door opening after which nobody who was in the room before is left inside and nobody works
        at the desk/safe/laptop: the staff left. People who appear only after it (behind the glass) do not count.
        Fallback: a final empty-room run (not at the start), dated by the last door opening before it."""
        for a, b, i, j in S.door_runs:
            cnt = collections.Counter(pid for k in range(i, j + 1) for pid, x in S.people[k] if pid >= 0 and self.in_door(S, x))
            if not cnt or j + 1 >= len(S.ts):
                continue
            leaver = cnt.most_common(1)[0][0]
            if S.role(leaver) != "cashier" and any(pid in S.cashiers() for k in range(j + 1, len(S.ts)) for pid, _ in S.people[k]):
                continue
            after = [(pid, x) for k in range(j + 1, len(S.ts)) for pid, x in S.people[k]]
            stayed = any(pid >= 0 and pid != leaver and S.tracks[pid].first_t < a for pid, _ in after)
            if not stayed and sum(self.working(S, x) for _, x in after) < self.p["count_samples"]:
                S.departure = (a, b, i, j, leaver); return
        ends = [r for r in P.runs(S.empty, S.ts) if r[2] > 0 and r[3] == len(S.ts) - 1]
        if ends:
            a = ends[-1][0]
            door = [r for r in S.door_runs if a - 15 <= r[0] <= a]
            S.departure = (door[-1] + (None,)) if door else (a, a, ends[-1][2], ends[-1][2], None)

    # ---- debugging --------------------------------------------------------------------------------------
    @staticmethod
    def describe(S):
        f = lambda rs: " ".join(f"{a:.1f}-{b:.1f}" for a, b, *_ in rs) or "-"
        lines = [f"door {f(S.door_runs)}", f"safe_open {f(S.safe_runs)}", f"ID {f(S.id_runs)}",
                 f"empty {f(P.runs(S.empty, S.ts))}", f"laptop_active {f(P.runs(S.laptop_active, S.ts))}",
                 f"departure {S.departure[:2] if S.departure else '-'}",
                 f"light_jumps {[round(j, 1) for j in S.light_jumps]}  key_in {f(P.runs(S.key_in, S.ts))}"]
        for pid, tr in sorted(S.tracks.items(), key=lambda kv: kv[1].first_t):
            if tr.n < 5: continue
            lines.append(f"  pid {pid:3} {tr.role:9} {tr.first_t:5.1f}-{tr.last_t:5.1f} n={tr.n:3} stand={tr.standing / tr.n:.2f} "
                         f"work={tr.work} bag={tr.bag} door_head={tr.head_in_door} reaches {f(S.reaches.get(pid, []))}")
        lines.append("counts " + " ".join(f"{c.t0:.1f}-{c.t1:.1f}(pid {c.pid}, piles {c.piles})" for c in S.counts))
        lines.append("money " + " ".join(f"{k}:{m['first_t']:.1f}-{m['last_t']:.1f}/{m['n']}" for k, m in S.money_tracks.items() if m["n"] >= 5))
        return "\n".join(lines)
