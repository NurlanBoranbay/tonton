# ForteBank: violations seen in the training videos

Team notes from watching the 22 violation videos (labels in `labels.csv`), with the rule each
one can cite. Rule text is in `units.jsonl` / `txt/`; links go to old.adilet.

**Rule match** — how safe the citation is in front of the jury:
- **direct**: the point says this almost word for word.
- **closest**: the point covers the situation, but the wording is broader or leaves details to the bank's internal documents.
- **none found**: nothing in the 25 collected acts; needs more research before the service may cite anything.

Main act unless stated: V1900019680, Правила осуществления кассовых операций и операций по инкассации
банкнот, монет и ценностей (НБРК № 231, 29.11.2019) — https://old.adilet.zan.kz/rus/docs/V1900019680
Premises act: V2000020075, Правила организации охраны и устройства помещений (НБРК № 14, 24.02.2020).

## Correct sequence (cash deposit)

1. Cashier examines the client's ID document.
2. Counts the cash through the counting machine.
3. Records the operation on the laptop.
4. Opens the safe and puts the money in.
5. Client leaves.

Also seen:
- Foreign objects and unrelated actions can be in frame.
- The money on the desk is not always a client's money.
- Without a client, several piles of money can lie on the desk; that alone is not a violation.

## Violations

| Video | t, s | Violation (team description) | Rule | Match |
|---|---|---|---|---|
| X-1 | 22 | Hands over only part of the amount the machine counted | п.31 пп.3–4 (prepare the amount, recount it in front of the client, issue it); п.35 | closest |
| X-2 | 10 | Counts the money before the client's ID was shown | п.19 пп.1 (visual identification by ID document before acceptance) | direct |
| Х-4 | 57 | Stacks money from different clients in one pile | п.20 ч.2 (only the current client's cash at the workplace; earlier cash in locked storage); п.21 | direct |
| Х-11 | 18 | Leaves the key in the safe's keyhole when leaving | п.80 (keys protected from unauthorised use); п.76 | closest |
| Х-13 | 47 | Puts money into the safe without counting it first | п.19 пп.2 (accept banknotes with sheet-by-sheet count) | closest |
| Х-15 | 15 | Leaves the cash-collection bag lying in the open | п.20 ч.2 (cash kept in locked storage); п.24 (bag accepted on the accompanying document, integrity and seal checked) | closest |
| Х-17 | 7 | Cash collector brings money (allowed) but must show ID; any second person in the room must show ID | п.75 (entry only for responsible persons, others by order and escorted); п.24 | closest |
| Х-18 | 20 | Three people in the room at once while money is being handled | п.75 | closest |
| Х-20 | 5 | Safe left open (team unsure) | п.20 ч.2 (locked storage); п.80 | closest |
| Х-22 | 14 | Key left in the keyhole (team unsure) | п.80; п.76 | closest |
| Х-24 | 8 | Items left on the desk other than working equipment | п.12 (personal belongings kept outside the cash unit); п.20 ч.2 | closest |
| Х-26 | 9 | Money left on the desk unattended | п.22 (cashier must not leave before the client is served); п.20 ч.2 | direct |
| Х-28 | 7 | Laptop left logged in when the cashier leaves | — | none found |
| Х-30 | 13 | Client enters the cash area before it is ready | п.75; V2000020075 п.8 (cash unit isolated, one lockable entrance), п.16 пп.1 | closest |
| Х-32 | 5 | Door left open after entering (hard to detect with fixed rules) | V2000020075 п.8, п.16 пп.1 (lockable door) | closest |
| Х-33 | 6 | Door left open, someone can look in | V2000020075 п.21 (limit third parties' access to and view of cash collection) | closest |
| Х-34 | 13 | Client leaves while the operation is still running; money counted without ID | п.35 (client recounts without leaving the cash desk); п.19 пп.1 | direct |
| Х-38 | 33 | Money still out when the client leaves; the safe must be opened at least once before the client leaves | п.20 ч.2 | closest |
| Х-40 | 17 | Personal belongings left on the desk | п.12 | direct |
| Х-44 | 16 | Banknotes of different denominations mixed | п.43 (sorting by denomination) | closest |
| Х-46 | 11 | Drinks or personal belongings put on the desk with money | п.12 | direct |
| Х-47 | 68 | Different clients' money mixed | п.20 ч.2; п.21 (each deposit document accepted separately) | direct |

## Violation types (for the classifier)

Grouping of the 22 videos above:

| Type | Videos | Main rule |
|---|---|---|
| Operation without ID / out of sequence | X-2, Х-34, Х-13 | п.19 |
| Mixing money (clients, denominations) | Х-4, Х-47, Х-44 | п.20 ч.2, п.21, п.43 |
| Wrong amount handed over / client leaves early | X-1, Х-34, Х-38 | п.31, п.35 |
| Money or bag left unattended / not in the safe | Х-26, Х-15, Х-38 | п.22, п.20 ч.2 |
| Safe or key left open | Х-11, Х-20, Х-22 | п.80, п.76 |
| Foreign objects / personal items on the desk | Х-24, Х-40, Х-46 | п.12 |
| Unauthorised people / too many people in the room | Х-17, Х-18, Х-30 | п.75 |
| Door left open | Х-32, Х-33 | V2000020075 п.8, п.16, п.21 |
| Workstation left logged in | Х-28 | none found |

## Open questions

1. **ID rule conflicts.** X-2 is a violation for counting without ID, yet Х-35 and Х-37 are labelled clean; Х-34 is listed as possibly the same case. Need to rewatch: was an ID shown off-camera, or is something else different?
2. **Can money go into the safe without an ID?** Not settled.
3. **Several piles, some counted, some not (Х-13).** How to tell them apart on video.
4. **Х-28 (laptop logged in).** No rule in the collected acts. Look for АРРФР/НБРК information-security requirements before citing anything; until then the service should not cite a rule for it.
5. **Х-17 (collector's ID).** The collected rules say who may enter (п.75) and that the bag comes with a document (п.24, п.106), but no point found that says the collector shows an ID to the cashier. Check further.
6. **Х-20 and Х-22** are marked uncertain by the team.
7. **Door rules.** V2000020075 requires a lockable door, but no point found that says the door must stay closed during work. Treat door cases as "closest".

## Point links

| Point | Link |
|---|---|
| п.12 | https://old.adilet.zan.kz/rus/docs/V1900019680#z81 |
| п.19 | https://old.adilet.zan.kz/rus/docs/V1900019680#z104 |
| п.20 | https://old.adilet.zan.kz/rus/docs/V1900019680#z111 |
| п.21 | https://old.adilet.zan.kz/rus/docs/V1900019680#z113 |
| п.22 | https://old.adilet.zan.kz/rus/docs/V1900019680#z114 |
| п.24 | https://old.adilet.zan.kz/rus/docs/V1900019680#z117 |
| п.31 | https://old.adilet.zan.kz/rus/docs/V1900019680#z144 |
| п.35 | https://old.adilet.zan.kz/rus/docs/V1900019680#z154 |
| п.43 | https://old.adilet.zan.kz/rus/docs/V1900019680#z164 |
| п.75 | https://old.adilet.zan.kz/rus/docs/V1900019680#z221 |
| п.76 | https://old.adilet.zan.kz/rus/docs/V1900019680#z222 |
| п.80 | https://old.adilet.zan.kz/rus/docs/V1900019680#z226 |
| V2000020075 п.8 | https://old.adilet.zan.kz/rus/docs/V2000020075#z49 |
| V2000020075 п.16 | https://old.adilet.zan.kz/rus/docs/V2000020075 (search «16. Устройство помещения кассы») |
| V2000020075 п.21 | https://old.adilet.zan.kz/rus/docs/V2000020075#z90 |
