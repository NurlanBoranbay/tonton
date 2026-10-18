# Build violation_classes.csv: team classes (from violations.md) + law-only classes.
# Every quote is checked to be a verbatim substring of the act text in units.jsonl.
import csv, json
U = {}
for l in open("units.jsonl"):
    x = json.loads(l)
    if x["path"]:
        U.setdefault((x["ngr"], x["number"]), x)  # first match = the rules, not the repealed-acts list at the end
BASE = "https://old.adilet.zan.kz/rus/docs/"
ACT = {"V1900019680": "Правила кассовых операций и инкассации (НБРК № 231)",
       "V2000020075": "Правила охраны и устройства помещений (НБРК № 14)",
       "V1900018545": "Правила обменных операций с наличной иностранной валютой (НБРК № 49)"}
K, P, E = "V1900019680", "V2000020075", "V1900018545"

# (id, name_en, name_ru, visual cue, [(ngr, point, sub, quote)], match, source, examples, detectability)
C = [
 ("C01", "no_id_before_operation", "Операция без проверки удостоверения личности",
  "Cashier counts or accepts the client's cash before an ID document is shown and examined",
  [(K, "19", "пп.1", "проводит визуальную идентификацию клиента, вносящего наличные деньги, с документом, удостоверяющим его личность"),
   (K, "31", "пп.1", "проводит визуальную идентификацию клиента по данным документа, удостоверяющего личность")],
  "direct", "team+law", "X-2;Х-34;Х-19;Х-35;Х-37;Х-39", "medium"),
 ("C02", "uncounted_money_to_safe", "Помещение в сейф непересчитанных денег",
  "Cash goes into the safe without passing through the counting machine / hand count",
  [(K, "19", "пп.2", "принимает банкноты с полистным пересчетом")],
  "closest", "team+law", "Х-13", "medium"),
 ("C03", "mixed_client_money", "Смешивание денег разных клиентов",
  "Cash of a new client is put on / stacked with cash from an earlier client at the workplace",
  [(K, "20", "ч.2", "На рабочем месте кассового работника находятся наличные деньги клиента. Все ранее принятые наличные деньги хранятся в специальных устройствах"),
   (K, "21", "", "кассовый работник принимает наличные деньги по каждому приходному кассовому документу отдельно")],
  "direct", "team+law", "Х-4;Х-47", "medium"),
 ("C04", "mixed_denominations", "Смешивание банкнот разных номиналов",
  "Banknotes of different denominations stacked or packed together",
  [(K, "43", "", "Сортировка наличных денег осуществляется по номиналам и по степени износа")],
  "closest", "team+law", "Х-44", "low"),
 ("C05", "wrong_amount_issued", "Выдача не всей пересчитанной суммы / без пересчёта при клиенте",
  "Cashier hands over only part of the amount counted on the machine, or issues cash without recounting it in front of the client",
  [(K, "31", "пп.4", "пересчитывает подготовленные к выдаче наличные деньги в присутствии клиента, выдает клиенту наличные деньги")],
  "closest", "team+law", "X-1", "low"),
 ("C06", "client_leaves_before_completion", "Клиент уходит до завершения операции / без пересчёта у кассы",
  "Client walks away while the operation is still running or without recounting at the desk",
  [(K, "35", "", "Клиент пересчитывает полученные наличные деньги, не отходя от кассы")],
  "direct", "team+law", "Х-34", "high"),
 ("C07", "money_left_after_client", "Деньги остаются на столе после ухода клиента",
  "After the client leaves, cash stays on the desk; the safe was not opened to store it",
  [(K, "20", "ч.2", "Все ранее принятые наличные деньги хранятся в специальных устройствах, находящихся на рабочем месте кассового работника, закрывающихся на ключ")],
  "closest", "team+law", "Х-38", "high"),
 ("C08", "cash_unattended", "Деньги оставлены без присмотра / кассир покинул рабочее место",
  "Cashier leaves the desk while cash lies on it or before the client is served",
  [(K, "22", "", "Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента"),
   (K, "20", "ч.2", "Все ранее принятые наличные деньги хранятся в специальных устройствах")],
  "direct", "team+law", "Х-26", "high"),
 ("C09", "collection_bag_unattended", "Инкассаторская сумка оставлена открыто",
  "A cash-collection bag is left lying in the open instead of being processed and stored",
  [(K, "20", "ч.2", "Все ранее принятые наличные деньги хранятся в специальных устройствах"),
   (K, "24", "пп.3–4", "проверяет целостность сумки")],
  "closest", "team+law", "Х-15", "medium"),
 ("C10", "safe_left_open", "Сейф оставлен открытым",
  "Safe door stays open with nobody working at it / when the cashier walks away",
  [(K, "20", "ч.2", "закрывающихся на ключ, предназначенных для хранения наличных денег (металлический шкаф, сейф, тележка, ящики стола, тумба)"),
   (K, "80", "", "обеспечивающими исключение их несанкционированного использования")],
  "closest", "team+law", "Х-20", "medium"),
 ("C11", "key_left_in_safe", "Ключ оставлен в замке сейфа",
  "Key left in the safe's keyhole when the cashier leaves",
  [(K, "76", "", "Ключи и (или) коды электронных и механических замков от двери кладовой и (или) сейфовой комнаты банка хранятся у лиц, ответственных за сохранность наличных денег"),
   (K, "80", "", "обеспечивающими исключение их несанкционированного использования")],
  "closest", "team+law", "Х-11;Х-22", "low"),
 ("C12", "foreign_items_on_desk", "Посторонние / личные предметы на рабочем столе",
  "Personal belongings, drinks or other non-work items on the cash desk (only working equipment allowed)",
  [(K, "12", "", "Хранение личных вещей, наличных денег и ценностей работников банка осуществляется вне кассового узла")],
  "direct", "team+law", "Х-24;Х-40;Х-46", "high"),
 ("C13", "phone_in_cash_room", "Телефон с камерой в кассовом помещении",
  "Cashier uses or keeps a smartphone (camera-capable) in the cash room",
  [(K, "12", "ч.2", "Допускается внос работниками банка в кассовый узел лекарственных средств, медицинских изделий и средств связи, не имеющих функции фото и видео фиксации")],
  "direct", "law", "", "high"),
 ("C14", "unauthorized_person", "Постороннее лицо в кассовом помещении",
  "A person who is not the cashier (client before the area is ready, collector who did not show ID, any extra person) is inside the cash room",
  [(K, "75", "", "Вход в кладовую и (или) сейфовую комнату разрешается лицам, ответственным за сохранность наличных денег, а также лицам на основании распорядительного документа руководителя банка при сопровождении лиц, ответственных за сохранность наличных денег")],
  "closest", "team+law", "Х-17;Х-30", "high"),
 ("C15", "too_many_people", "Лишние лица во время операции",
  "Three or more people in the room while money is being handled",
  [(K, "75", "", "Вход в кладовую и (или) сейфовую комнату разрешается лицам, ответственным за сохранность наличных денег")],
  "closest", "team+law", "Х-18", "high"),
 ("C16", "door_left_open", "Дверь кассового помещения оставлена открытой",
  "Door to the cash room left open after entering, so outsiders can enter or look in",
  [(P, "8", "", "Кассовый узел подлежит изолированию от других производственных, подсобных, подвальных и служебных помещений одним закрывающимся служебным входом"),
   (P, "16", "пп.1", "каждая касса имеет отдельный вход с закрывающейся на замок дверью"),
   (P, "21", "", "обеспечивается ограничение доступа и просмотра третьими лицами за осуществлением инкассации")],
  "closest", "team+law", "Х-32;Х-33", "medium"),
 ("C17", "workstation_unlocked", "Компьютер оставлен незаблокированным",
  "Laptop left logged in / unlocked when the cashier leaves",
  [], "none_found", "team", "Х-28", "medium"),
 ("C18", "counterfeit_returned", "Возврат клиенту банкнот с признаками подделки",
  "Cashier rejects a banknote and hands it back to the client instead of keeping it and drawing up an act",
  [(K, "61", "", "Банкноты и монеты, имеющие признаки подделки, клиенту не возвращаются"),
   (K, "59", "", "При обнаружении сомнительных банкнот и монет кассовый работник в присутствии клиента и заведующего кассой")],
  "direct", "law", "", "low"),
 ("C19", "collection_bag_not_checked", "Приём инкассаторской сумки без проверки",
  "Cashier opens or accepts a collection bag without checking its integrity and seal",
  [(K, "24", "пп.3–4", "проверяет целостность сумки"),
   (K, "25", "", "кассовый работник в присутствии работника подразделения инкассации банка или инкассаторской организации вскрывает и пересчитывает")],
  "direct", "law", "", "low"),
 ("C20", "recount_without_controller", "Пересчёт и сортировка без контролирующего работника",
  "Cashier recounts / sorts / packs cash (not a client operation) alone, with no second employee present",
  [(K, "42", "", "Кассовые операции по пересчету, сортировке и упаковке наличных денег осуществляются кассовым работником в присутствии контролирующего работника")],
  "direct", "law", "", "low"),
 ("C21", "exchange_desk_outsiders_or_foreign_money", "Посторонние лица или посторонние деньги в кассе обменного пункта",
  "Exchange-office cash desk: an outsider is inside, or money unrelated to the business is kept there",
  [(E, "41", "", "В помещениях операционных касс обменных пунктов не допускается присутствие посторонних лиц")],
  "direct", "law", "", "medium"),
]

# Team selection 2026-10-17 (Nurlan) and the signals each class needs from the
# perception models: pose = skeleton model, det:<cls> = custom box model, coco:<group> = COCO-80 model.
SELECTED = {
 "C01": "", "C02": "", "C03": "", "C04": "", "C05": "", "C06": "", "C07": "", "C08": "",
 "C10": "only when no cashier is present", "C11": "only when no cashier is present",
 "C12": "", "C14": "", "C16": "", "C17": "", "C19": "",
}
SIGNALS = {
 "C01": "det:ID-identification; pose (cashier/client hands, ID handover); det:money_stack; det:counting_machine; order of events",
 "C02": "det:money_stack; det:counting_machine; det:safe_open; order of events (money reaches safe without passing the machine)",
 "C03": "det:money_stack (count of separate stacks); pose (client change); tracking",
 "C04": "det:money_stack (+ denomination/colour cue, likely needs crop classifier or VLM)",
 "C05": "det:money_stack; det:counting_machine; pose (handover to client)",
 "C06": "pose (client leaves zone); det:money_stack; operation state",
 "C07": "pose (client gone); det:money_stack on desk; det:safe_open never seen",
 "C08": "pose (cashier leaves seat/zone); det:money_stack on desk",
 "C10": "det:safe_open; pose (no cashier in frame / near safe)",
 "C11": "det:key_in_lock; pose (no cashier in frame / near safe)",
 "C12": "coco:personal on desk zone",
 "C14": "pose (person count, role by position); det:collection_bag (collector); door crossing",
 "C16": "det:door_open / det:door_closed (+ VLM check against reference frame)",
 "C17": "coco:laptop; pose (cashier absent); screen state (VLM or screen crop classifier)",
 "C19": "det:collection_bag; pose (cashier hands on bag); order of events",
}

out = []
for cid, en, ru, cue, refs, match, src, ex, det in C:
    cites, quotes, links = [], [], []
    for ngr, pt, sub, q in refs:
        x = U[(ngr, pt)]
        assert " ".join(q.split()) in " ".join(x["text"].split()), (cid, ngr, pt, q[:60])
        cites.append(f"{ngr} п.{pt}{(' ' + sub) if sub else ''}")
        quotes.append(q)
        a = x["id"].split("#")[1]
        links.append(BASE + ngr + ("#" + a if a.startswith("z") else ""))
    out.append({"class_id": cid, "name_en": en, "name_ru": ru, "visual_cue": cue,
                "act": "; ".join(dict.fromkeys(ACT[r[0]] for r in refs)),
                "citation": "; ".join(cites), "quote": " | ".join(quotes),
                "link": " ; ".join(dict.fromkeys(links)), "rule_match": match,
                "source": src, "example_videos": ex, "detectability": det,
                "selected": int(cid in SELECTED), "selection_note": SELECTED.get(cid, ""),
                "required_signals": SIGNALS.get(cid, "")})
with open("violation_classes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out), "classes; all quotes verified")
