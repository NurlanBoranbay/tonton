"""Personal belongings and drinks next to cash / work equipment."""
from app.rules import predicates as P
from app.rules.base import Rule, register


def _near_runs(timeline, scene, classes, targets, near_px, gap_sec, skip=None):
    flags = []
    for k, ff in enumerate(timeline.frames):
        items = [b for c in classes for b in ff.boxes(c) if not (skip and skip(k, b))]
        flags.append(any(P.near(b, t, near_px) for b in items for t in targets(k, ff)))
    return P.runs(flags, scene.ts, max_gap=gap_sec)


@register
class PersonalBelongings(Rule):
    rule_id, class_id = "personal_belongings_violation", "C12"
    type = "Личные вещи на рабочем месте кассира"
    description = "Личная сумка/рюкзак находится рядом с деньгами или рабочим оборудованием кассира."
    params = {"dwell_sec": 9.0, "near_px": 40, "gap_sec": 4.0, "bag_window_sec": 2.0}
    param_schema = {"dwell_sec": ("bag next to equipment for, s", 1, 60, .5),
                    "near_px": ("distance to equipment / money, px", 0, 150, 5),
                    "gap_sec": ("tolerated detection gap, s", 0, 5, .1),
                    "bag_window_sec": ("ignore handbags on a collection bag seen within, s", 0, 10, .5)}

    def evaluate(self, timeline, params, scene):
        fixed = [b for b in (scene.machine, scene.safe, scene.laptop) if b]
        bags = [(scene.ts[k], b) for k, ff in enumerate(timeline.frames) for b in ff.boxes("collection_bag")]

        def targets(k, ff):
            return fixed + [b for _, b in scene.money[k]] + ff.boxes("collection_bag")

        def is_collection_bag(k, b):             # COCO calls the collection bag a handbag
            t = scene.ts[k]
            return any(abs(t - tb) <= params["bag_window_sec"] and P.iou(b, bb) > .2 for tb, bb in bags)

        for a, b, i, j in _near_runs(timeline, scene, ("handbag", "backpack"), targets, params["near_px"],
                                     params["gap_sec"], is_collection_bag):
            if b - a >= params["dwell_sec"]:
                return [self.alarm(a + params["dwell_sec"], a, b, first_near=round(a, 1))]
        return []


@register
class Beverage(Rule):
    rule_id, class_id = "beverage_violation", "C12"
    type = "Напиток рядом с деньгами"
    description = "Чашка/бутылка с напитком стоит рядом с наличными деньгами на рабочем столе."
    params = {"min_sec": 1.5, "near_px": 40, "gap_sec": 1.0}
    param_schema = {"min_sec": ("drink next to money for, s", 0, 30, .5),
                    "near_px": ("distance to money, px", 0, 150, 5),
                    "gap_sec": ("tolerated detection gap, s", 0, 5, .1)}

    def evaluate(self, timeline, params, scene):
        for a, b, i, j in _near_runs(timeline, scene, ("cup", "bottle", "wine glass"),
                                     lambda k, ff: [m for _, m in scene.money[k]], params["near_px"], params["gap_sec"]):
            if b - a >= params["min_sec"]:
                return [self.alarm(a, a, b)]
        return []
