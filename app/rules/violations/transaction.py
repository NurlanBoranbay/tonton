"""Transaction-order rules: interrupted transaction, counting with the safe open, collector without ID."""
import collections

from app.rules import predicates as P
from app.rules.base import Rule, register
from app.rules.scene import SceneCompiler


def transactions(scene):
    """[(t_start, t_end)]: a transaction opens with a count that has money in it while the safe is closed and
    ends when the safe opens (the money goes in)."""
    starts = sorted(c.t0 for c in scene.counts if c.piles)
    opens = [a for a, *_ in scene.safe_runs]
    out, t = [], -1.0
    for s in starts:
        if s <= t or scene.safe_state[scene.idx(s)] == "open":
            continue
        end = next((a for a in opens if a > s), 1e9)
        out.append((s, end)); t = end
    return out


@register
class InterruptedTransaction(Rule):
    rule_id, class_id = "interrupted_transaction", "C06"
    type = "Прерванная операция"
    description = "Клиент уходит или входит новый клиент, пока операция не завершена: деньги не убраны в сейф."
    params = {"pass_samples": 2}
    param_schema = {"pass_samples": ("samples of a client in the door", 1, 20, 1)}

    def evaluate(self, timeline, params, scene):
        comp = scene.params
        for s, e in transactions(scene):
            for a, b, i, j in scene.door_runs:
                if not s < a < e:
                    continue
                passing = collections.Counter(
                    pid for k in range(i, j + 1) for pid, x in scene.others_at(k)
                    if pid >= 0 and (P.head_in_bbox(x, scene.door) or P.cover(x.box, scene.door) >= comp["door_pass_cover"]))
                who = [pid for pid, n in passing.items() if n >= params["pass_samples"]]
                if not who:
                    continue
                # a new client = a non-cashier track that first shows up around this door opening
                new = [scene.tracks[p].first_t for p in who if scene.tracks[p].first_t >= a - 2 and scene.tracks[p].first_t > s]
                t = min(new) if new else a
                return [self.alarm(t, s, b, transaction_start=round(s, 1), door_open=round(a, 1), new_client=bool(new))]
        return []


@register
class OpenSafe(Rule):
    rule_id, class_id = "open_safe_violation", "C10"
    type = "Пересчёт денег при открытом сейфе"
    description = "Кассир пересчитывает деньги на счётной машине, пока сейф остаётся открытым."
    params = {"lead_sec": 1.0, "tail_sec": .5, "money_upper": 1}
    param_schema = {"money_upper": ("count needs money in the machine's upper half (0/1)", 0, 1, 1),
                    "lead_sec": ("count starts at least this long after the safe opened, s", 0, 10, .5),
                    "tail_sec": ("... and this long before it closes, s", 0, 10, .5)}

    def evaluate(self, timeline, params, scene):
        for a, b, i, j in scene.safe_runs:
            # Х-39: a reach into the machine box while the safe is open is a count only if money is in
            # the machine's upper half (the feed tray); otherwise it is the cashier working at the open safe
            c = next((c for c in scene.counts if a + params["lead_sec"] <= c.t0 <= b - params["tail_sec"] and
                      (not params["money_upper"] or SceneCompiler._money_upper(scene, scene.machine, c.i0, c.i1))), None)
            if c:
                return [self.alarm(a, a, b, count_at=round(c.t0, 1))]
        return []


@register
class AnonymousIncassator(Rule):
    rule_id, class_id = "anonymous_incassator_violation", "C19"
    type = "Приём инкассаторской сумки без проверки документов"
    description = "Посетитель (инкассатор) входит, передаёт сумку и уходит; удостоверение личности не предъявлено."
    params = {"visitor_min_sec": 2.0, "bag_samples": 5, "leave_margin_sec": .5}
    param_schema = {"visitor_min_sec": ("visitor present at least, s", .5, 30, .5),
                    "bag_samples": ("samples with the collection bag", 1, 50, 1),
                    "leave_margin_sec": ("visitor gone this long before the end, s", 0, 10, .5)}

    def evaluate(self, timeline, params, scene):
        cashier = scene.cashiers()
        for tr in sorted(scene.tracks.values(), key=lambda t: t.first_t):
            if tr.role == "cashier" or tr.presence < params["visitor_min_sec"]:
                continue
            if tr.last_t > timeline.duration - params["leave_margin_sec"] or not scene.door_runs:
                continue                                       # visitor must come and go
            span = [k for k, t in enumerate(scene.ts) if tr.first_t <= t <= tr.last_t]
            bag = [k for k in span if timeline.frames[k].boxes("collection_bag")]
            if len(bag) < params["bag_samples"] or any(scene.id_seen[k] for k in span):
                continue
            hand = next((scene.ts[k] for k in bag for pid, x in scene.people[k] if pid in cashier and any(
                P.wrist_near(x, b, scene.params["near_px"], scene.params["palm_reach"]) for b in timeline.frames[k].boxes("collection_bag"))), None)
            t = hand if hand is not None else scene.ts[bag[0]]
            return [self.alarm(t, tr.first_t, tr.last_t, visitor=tr.pid)]
        return []
