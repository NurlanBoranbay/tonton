"""Left-* rules: the staff left (scene.departure) and a hazard is still stably visible in the empty room.
One shared evaluator, so all five use the same timestamp policy: the door opening the cashier left through."""
from app.rules.base import Rule, register


class UnattendedHazard(Rule):
    params = {"min_share": .5}
    param_schema = {"min_share": ("share of samples after departure with the object", .1, 1.0, .05)}

    def present(self, ff, i, scene):
        raise NotImplementedError

    def evaluate(self, timeline, params, scene):
        if not scene.departure:
            return []
        a, b, i, j, _ = scene.departure
        idx = range(i, len(timeline.frames))
        share = sum(bool(self.present(timeline.frames[k], k, scene)) for k in idx) / max(len(idx), 1)
        if share < params["min_share"]:
            return []
        return [self.alarm(a, a, timeline.duration, share=round(share, 2), door_close=round(b, 1))]


@register
class LeftKey(UnattendedHazard):
    rule_id, class_id = "left_key_violation", "C11"
    type = "Ключ оставлен в замке сейфа"
    description = "Кассир покинул кассовое помещение, ключ остался в замке сейфа."

    def present(self, ff, i, scene):
        return scene.key_in[i]                       # lighting-aware key state (scene._key)


@register
class LeftCollectionBag(UnattendedHazard):
    rule_id, class_id = "left_collection_bag_violation", "C08"
    type = "Инкассаторская сумка оставлена без присмотра"
    description = "Кассир покинул кассовое помещение, инкассаторская сумка осталась на рабочем месте."

    def present(self, ff, i, scene):
        return ff.boxes("collection_bag")


@register
class LeftMoney(UnattendedHazard):
    rule_id, class_id = "left_money_violation", "C08"
    type = "Деньги оставлены без присмотра"
    description = "Кассир покинул кассовое помещение, на столе остались наличные деньги."

    def present(self, ff, i, scene):
        return ff.boxes("money_stack")


@register
class LeftLaptop(UnattendedHazard):
    rule_id, class_id = "left_laptop_violation", "C17"
    type = "Компьютер оставлен незаблокированным"
    description = "Кассир покинул кассовое помещение, экран ноутбука остался включённым (компьютер не заблокирован)."

    def present(self, ff, i, scene):
        return scene.laptop_active[i]


@register
class LeftCommodities(UnattendedHazard):
    rule_id, class_id = "left_commodities_violation", "C12"
    type = "Посторонние предметы оставлены на рабочем столе"
    description = "Кассир покинул кассовое помещение, на рабочем столе остались печати/штампы или посторонние предметы (тетрадь/книга)."
    # tonton3: custom class `stamp` (11-class detector, Х-24) counts as left items; `book` (COCO) is kept as a fallback
    # for the 10-class detector (Х-24's stamps/notebook are detected as a book). use_book=0 once stamps are detected.
    params = {**UnattendedHazard.params, "use_book": 0}
    param_schema = {**UnattendedHazard.param_schema, "use_book": ("also count COCO 'book' (0/1)", 0, 1, 1)}

    def evaluate(self, timeline, params, scene):
        self._use_book = params["use_book"]
        out = super().evaluate(timeline, params, scene)
        if out and scene.departure:
            idx = range(scene.departure[2], len(timeline.frames))
            seen = sorted({c for k in idx for c in ("stamp", "book") if timeline.frames[k].boxes(c)})
            out[0].evidence["items"] = seen
            if "stamp" in seen:
                out[0].description = "Кассир покинул кассовое помещение, на рабочем столе остались печати/штампы."
        return out

    def present(self, ff, i, scene):
        return ff.boxes("stamp") or (self._use_book and ff.boxes("book"))
