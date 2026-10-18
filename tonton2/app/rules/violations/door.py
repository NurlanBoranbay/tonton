"""Door rules: door left open with nobody passing; a second person in a door that stays open."""
from app.rules import predicates as P
from app.rules.base import Rule, register


@register
class OpenDoor(Rule):
    rule_id, class_id = "open_door_violation", "C16"
    type = "Дверь кассового помещения оставлена открытой"
    description = "Дверь кассового помещения остаётся открытой, при этом через неё никто не проходит."
    params = {"min_open_sec": 8.0, "pass_samples": 3}
    param_schema = {"min_open_sec": ("door open without anybody passing, s", 2, 60, .5),
                    "pass_samples": ("samples in the door to count as passing", 1, 20, 1)}

    def evaluate(self, timeline, params, scene):
        sp = scene.params
        for a, b, i, j in scene.door_runs:
            if b - a < params["min_open_sec"]:
                continue
            # a standing person in the door = somebody passing; the seated cashier in front of the door is not
            passing = [any(P.standing(x, sp["seated_height_to_torso"], sp["ankle_visibility_min"]) and
                           (P.head_in_bbox(x, scene.door) or P.cover(x.box, scene.door) >= sp["door_pass_cover"])
                           for _, x, _ in scene.all_people[k]) for k in range(i, j + 1)]
            ts = scene.ts[i: j + 1]
            free = P.runs([not p for p in passing], ts, max_gap=(params["pass_samples"] - 1) * timeline.dt)
            longest = max((e - s for s, e, *_ in free), default=0)
            if longest >= params["min_open_sec"]:
                return [self.alarm(a, a, b, open_sec=round(b - a, 1), free_sec=round(longest, 1))]
        return []


@register
class OpenDoorIntruder(Rule):
    rule_id, class_id = "open_door_violation_intruder", "C14"
    type = "Постороннее лицо у открытой двери кассового помещения"
    description = "Дверь кассового помещения долго остаётся открытой, в дверном проёме появляется второй посторонний человек."
    params = {"min_open_sec": 5.0, "head_samples": 3, "body_cover": .85}
    param_schema = {"min_open_sec": ("door open at least, s", 1, 30, .5),
                    "head_samples": ("samples with the head in the door", 1, 20, 1),
                    "body_cover": ("or: share of the body box inside the door", .5, 1.0, .05)}

    def evaluate(self, timeline, params, scene):
        cashier = scene.cashiers()
        for a, b, i, j in scene.door_runs:
            if b - a < params["min_open_sec"]:
                continue
            hits = {}
            for k in range(i, j + 1):
                in_door = [(raw, x, out) for raw, x, out in scene.all_people[k]
                           if P.head_in_bbox(x, scene.door) or P.cover(x.box, scene.door) >= params["body_cover"]]
                # the glass door mirrors whoever stands in it: an "outside" person seen while somebody in the room
                # is in the doorway is that person's reflection
                ghosts = any(not out for _, _, out in in_door)
                for raw, x, out in in_door:
                    who = scene.alias.get(raw, raw)
                    if raw < 0 or who in cashier or (out and ghosts): continue
                    hits.setdefault(who, []).append(scene.ts[k])
            firsts = sorted(ts[0] for ts in hits.values() if len(ts) >= params["head_samples"])
            if len(firsts) >= 2:
                return [self.alarm(firsts[1], a, b, people_in_door=len(firsts), open_sec=round(b - a, 1))]
        return []
