"""Cash-handling rules built on the count events and money tracks of the scene."""
from app.rules import predicates as P
from app.rules.base import Rule, register


def new_piles(scene, min_samples, initial_sec, t_from=0.0, t_to=1e9):
    """Money tracks that appeared during the video (not already lying there at the start)."""
    return {mid: m for mid, m in scene.money_tracks.items()
            if m["n"] >= min_samples and m["first_t"] > initial_sec and t_from <= m["first_t"] < t_to}


@register
class CashReturn(Rule):
    rule_id, class_id = "cash_return_violation", "C05"
    type = "Возврат наличных клиенту после пересчёта"
    description = "После пересчёта на счётной машине наличные снова оказываются в руках клиента до того, как деньги убраны в сейф."
    params = {"min_samples": 10, "near_px": 15, "include_id": 0, "palm_reach": 0.0, "window_sec": 10.0}
    param_schema = {"window_sec": ("cash back in the client's hand within this long after the count ends, s (0 = until the safe opens)", 0, 120, .5),"min_samples": ("samples with cash at another person's hand", 1, 50, 1),
                    "near_px": ("hand-to-cash distance, px", 0, 80, 1),
                    "palm_reach": ("palm extrapolation for this rule (0 = wrist only)", 0, 1, .05),
                    "include_id": ("also count the ID document (0/1)", 0, 1, 1)}

    def evaluate(self, timeline, params, scene):
        cashier = scene.cashiers()
        count = next((c for c in scene.counts if c.pid in cashier), None)
        if not count:
            return []
        safe_t = next((a for a, *_ in scene.safe_runs if a >= count.t0), 1e9)
        if params["window_sec"]:            # Х-38/Х-47: a return follows the count, not 20-40 s later
            safe_t = min(safe_t, count.t1 + params["window_sec"])
        classes = ["money_stack"] + (["ID-identification"] if params["include_id"] else [])
        hits = []
        for k in range(count.i0, len(scene.ts)):
            if scene.ts[k] >= safe_t: break
            ff = timeline.frames[k]
            for pid, x in scene.others_at(k):
                if any(P.wrist_near(x, b, params["near_px"], params["palm_reach"]) for c in classes for b in ff.boxes(c)):
                    hits.append(scene.ts[k]); break
        if len(hits) < params["min_samples"]:
            return []
        # Х-35: the safe was opened and closed after the count with the counting person reaching
        # into it = the money went into the safe, the transaction is valid; hand-offs seen before are an angle effect
        for a, b in scene.safe_cycles():
            if a >= count.t0 and scene.safe and any(P.wrist_in_bbox(x, scene.safe, scene.params["machine_margin_px"], scene.params["palm_reach"])
                                                    for k in range(scene.idx(a), scene.idx(b) + 1) for pid, x in scene.people[k] if pid == count.pid):
                return []
        return [self.alarm(hits[0], hits[0], hits[-1], count_at=round(count.t0, 1), samples=len(hits))]


@register
class AnonymousCashAcceptance(Rule):
    rule_id, class_id = "anonymous_cash_acceptance_violation", "C01"
    type = "Приём наличных без идентификации клиента"
    description = "Клиент передаёт наличные и они принимаются в работу до того, как предъявлено удостоверение личности."
    params = {"reach_samples": 2, "reach_window_sec": 3.0, "money_samples": 10, "initial_sec": 1.0, "id_tolerance_sec": .5}
    param_schema = {"reach_samples": ("client palm in the machine box, samples", 1, 30, 1),
                    "reach_window_sec": ("client reach this close to the money's first appearance, s", .5, 20, .5),
                    "money_samples": ("min samples of a money pile", 1, 100, 1),
                    "initial_sec": ("money seen before this is pre-existing, s", 0, 10, .5),
                    "id_tolerance_sec": ("ID may follow the money by, s", 0, 10, .5)}

    def evaluate(self, timeline, params, scene):
        piles = new_piles(scene, params["money_samples"], params["initial_sec"])
        if not piles or not scene.counts:
            return []
        t_money = min(m["first_t"] for m in piles.values())
        # the client is in the cash path: a non-cashier's palm in the counting machine box when the money shows up
        reach = sum(1 for pid, rs in scene.reaches.items() if scene.role(pid) != "cashier"
                    for _, _, i, j in rs for k in range(i, j + 1)
                    if abs(scene.ts[k] - t_money) <= params["reach_window_sec"])
        if reach < params["reach_samples"]:
            return []
        t_id = scene.id_runs[0][0] if scene.id_runs else 1e9
        if t_money >= t_id - params["id_tolerance_sec"]:
            return []
        return [self.alarm(t_money, t_money, min(t_id, timeline.duration), id_first=None if t_id == 1e9 else round(t_id, 1))]


@register
class UncountedSafeDeposit(Rule):
    rule_id, class_id = "uncounted_safe_deposit", "C02"
    type = "Помещение в сейф непересчитанных денег"
    description = "Сейф открывается, чтобы убрать деньги, которые не были пересчитаны на счётной машине."
    params = {"money_samples": 10, "initial_sec": 1.0, "before_sec": 2.0}
    param_schema = {"money_samples": ("min samples of a money pile", 1, 100, 1),
                    "initial_sec": ("money seen before this is pre-existing, s", 0, 10, .5),
                    "before_sec": ("pile on the desk at least this long before the safe opens, s", 0, 30, .5)}

    def evaluate(self, timeline, params, scene):
        prev = 0.0
        for a, b, i, j in scene.safe_runs:
            # money that shows up just before the safe opens is being taken out, not deposited
            piles = new_piles(scene, params["money_samples"], params["initial_sec"], prev, a - params["before_sec"])
            if piles:
                t0 = min(m["first_t"] for m in piles.values())
                if not any(t0 - 1 <= c.t0 < a for c in scene.counts):
                    return [self.alarm(a, a, b, piles=sorted(piles), money_since=round(t0, 1))]
            prev = b
        return []


@register
class CurrencyIntermix(Rule):
    rule_id, class_id = "currency_intermix_violation", "C03"
    type = "Смешивание денег (разных клиентов / номиналов)"
    description = "После пересчёта кассир снова обращается к счётной машине, и новая пачка денег кладётся на уже пересчитанную."
    params = {"window_sec": 6.0, "leave_min_sec": 1.0, "merge_px": 10, "pile_samples": 10}
    param_schema = {"window_sec": ("merge within this long after the re-reach, s", 1, 60, .5),
                    "leave_min_sec": ("hand away from the machine at least, s", .2, 10, .1),
                    "merge_px": ("pile ends this close to another pile = merged", 0, 60, 1),
                    "pile_samples": ("min samples of a pile", 1, 50, 1)}

    def merges(self, scene, params, t_from, t_to):
        """(t, pile, onto): pile vanishes on top of another live pile (occupies its last position) in [t_from, t_to]."""
        ms = {k: m for k, m in scene.money_tracks.items() if m["n"] >= params["pile_samples"]}
        out = []
        for kb, mb in ms.items():
            if not t_from <= mb["last_t"] <= t_to: continue
            for ka, ma in ms.items():
                if ka != kb and ma["first_t"] < mb["last_t"] and ma["last_t"] > mb["last_t"] + 1 and \
                        P.gap(mb["last_box"], ma["last_box"]) <= params["merge_px"]:
                    out.append((mb["last_t"], kb, ka))
        return sorted(out)

    def evaluate(self, timeline, params, scene):
        # touch - leave - touch by the same person after a count, then a pile that appeared after that count is
        # put onto a pile that was already counted: the two lots are no longer kept apart
        for c in scene.counts:
            counted = {p for x in scene.counts if x.t0 <= c.t0 for p in x.piles}
            for a, b, i, j in scene.reaches.get(c.pid, []):
                if not params["leave_min_sec"] <= a - c.t1:
                    continue
                for t, pile, onto in self.merges(scene, params, a, a + params["window_sec"]):
                    if onto in counted and scene.money_tracks[pile]["first_t"] > c.t1:
                        return [self.alarm(a, c.t0, t, first_count=round(c.t0, 1), merged=[pile, onto])]
        return []


@register
class NoIdBeforeCount(Rule):
    """Ported from tonton (exp/cashflow.py no_id_before_count, engine rule 7)."""
    rule_id, class_id = "no_id_before_count_violation", "C01"
    type = "Операция без проверки удостоверения личности"
    description = ("Кассир принимает и пересчитывает наличные, но удостоверение личности клиента не было предъявлено "
                   "(V1900019680 п.19 пп.1: «проводит визуальную идентификацию клиента, вносящего наличные деньги, "
                   "с документом, удостоверяющим его личность»).")
    params = {"visit_sec": 3.0, "cash_margin": .03}
    param_schema = {"visit_sec": ("client visit = 2+ people in the room for at least, s", .5, 30, .5),
                    "cash_margin": ("wrist-to-cash margin, share of the frame", 0, .2, .01)}

    def evaluate(self, timeline, params, scene):
        cashier = scene.cashiers()
        # collection-bag visits (a collector role exists) are C19 (anonymous_incassator), not a client deposit
        if any(tr.role == "collector" for tr in scene.tracks.values()):
            return []
        count = next((c for c in scene.counts if c.pid in cashier), None)
        if not count:
            return []
        for a, b, i, j in P.runs([n >= 2 for n in scene.n_room], scene.ts, min_dur=params["visit_sec"]):
            if count.t0 < a:
                continue
            ks = [k for k in range(i, len(scene.ts)) if scene.ts[k] <= count.t0]
            if any(timeline.frames[k].boxes("ID-identification") for k in ks):
                return []
            mx, my = params["cash_margin"] * timeline.W, params["cash_margin"] * timeline.H
            t = next((scene.ts[k] for k in ks for pid, x in scene.people[k] if pid in cashier
                      for w in x.wrists() for bb in timeline.frames[k].boxes("money_stack")
                      if bb[0] - mx <= w[0] <= bb[2] + mx and bb[1] - my <= w[1] <= bb[3] + my), count.t0)
            return [self.alarm(t, a, count.t0, visit_start=round(a, 1), count_at=round(count.t0, 1))]
        return []
