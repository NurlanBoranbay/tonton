"""Small geometric predicates shared by the scene compiler and the rules. Boxes are [x1, y1, x2, y2] px."""


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b):
    i = inter(a, b)
    return i / max(area(a) + area(b) - i, 1e-9)


def cover(a, b):
    """Share of box a covered by box b."""
    return inter(a, b) / max(area(a), 1e-9)


def expand(b, m):
    return [b[0] - m, b[1] - m, b[2] + m, b[3] + m]


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def inside(pt, b, m=0.0):
    return b[0] - m <= pt[0] <= b[2] + m and b[1] - m <= pt[1] <= b[3] + m


def gap(a, b):
    """Distance between two boxes (0 if they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** .5


def near(a, b, px):
    return gap(a, b) <= px


def hands(person, reach=0.0):
    """Wrists, plus each palm extrapolated `reach` forearm-lengths past the wrist (the hand holding things)."""
    from app.pipeline.timeline import L_WRI, R_WRI
    out = []
    for w, e in ((L_WRI, 7), (R_WRI, 8)):
        pw, pe = person.kp(w), person.kp(e)
        if not pw: continue
        out.append(pw)
        if pe and reach:
            out.append((pw[0] + reach * (pw[0] - pe[0]), pw[1] + reach * (pw[1] - pe[1])))
    return out


def wrist_in_bbox(person, box, m=0.0, reach=0.0):
    return any(inside(w, box, m) for w in hands(person, reach))


def wrist_near(person, box, px, reach=0.0):
    return any(inside(w, box, px) for w in hands(person, reach))


def head_in_bbox(person, box):
    return any(inside(h, box) for h in person.head())


def torso(person):
    """Shoulder-to-hip length in px, None if the keypoints are not visible."""
    from app.pipeline.timeline import L_SHO, R_SHO, L_HIP, R_HIP
    sh = [p for p in (person.kp(L_SHO), person.kp(R_SHO)) if p]
    hp = [p for p in (person.kp(L_HIP), person.kp(R_HIP)) if p]
    if not sh or not hp:
        return None
    sy = sum(p[1] for p in sh) / len(sh); hy = sum(p[1] for p in hp) / len(hp)
    return abs(hy - sy) or None


def legs_visible(person, kmin):
    from app.pipeline.timeline import L_KNEE, R_KNEE, L_ANK, R_ANK
    return any(person.kp(i, kmin) for i in (L_ANK, R_ANK)) or sum(bool(person.kp(i, kmin)) for i in (L_KNEE, R_KNEE)) == 2


def standing(person, height_to_torso, kmin):
    """Standing (walking) person: legs visible and the box is tall relative to the torso.
    A seated cashier has the legs hidden by the desk/chair and a short box."""
    t = torso(person)
    h = person.box[3] - person.box[1]
    return legs_visible(person, kmin) and (t is None or h >= height_to_torso * t)


def runs(flags, ts, min_dur=0.0, max_gap=0.0):
    """Maximal [(t_start, t_end, i_start, i_end)] where flags[i] holds; gaps up to max_gap s are bridged."""
    out, s, last = [], None, None
    for i, ok in enumerate(flags):
        if ok:
            if s is not None and i > last + 1 and ts[i] - ts[last + 1] > max_gap + 1e-6:
                out.append((s, last)); s = None
            if s is None: s = i
            last = i
    if s is not None: out.append((s, last))
    return [(ts[a], ts[b], a, b) for a, b in out if ts[b] - ts[a] >= min_dur]
