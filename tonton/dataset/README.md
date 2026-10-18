# ForteBank case: RK legal acts

25 Kazakhstan acts relevant to cashier-video violations (cash operations, cash collection,
cash-room security, banknote validity, currency exchange, AML/ID checks, anti-terror).
Downloaded from old.adilet.zan.kz on 2026-10-17 and parsed by kz-law-retrieval/parser/parse_acts.py.

- `acts.jsonl`          list of acts: ngr, title_ru, group, tier (A = main source for video rules, B = supporting, C = background)
- `acts_research.json`  per act: issuer, status, latest amendment, key points for visible cashier behaviour, verified flag
- `units.jsonl`         one record per point (НБРК rules) or article (laws, codes): id, ngr, number, path, text, notes
- `docs.jsonl`          per-act parse stats
- `txt/<NGR>.txt`       readable text of each act, by chapter
- `html/<NGR>.html`     raw act page; `info/<NGR>.html` raw /info card (status, dates)

Tier A: V1900019680 (cash operations and cash collection), V2000020075 (guarding and fitting out
premises), V1700016120 (banknote validity), V1900018545 (currency exchange).
Note: Z950002444_ (1995 banking law) is repealed; cite Z2600000258.

## Videos (from ../open.zip)

- `videos/*.mov`       42 training videos (overhead view of one cash desk, no audio)
- `open_labels.xlsx`   original labels: file, timestamp, нарушение / без нарушения
- `labels.csv`         same labels: file, violation (0/1), timestamp_sec, raw_timestamp. 22 violation, 20 clean.
  Violation type is not labelled (by design of the case).
- Names: X-1 and X-2 use a Latin "X", all others a Cyrillic "Х". labels.csv uses the exact
  file names, so join on `file` as-is; don't retype the names.
- `violations.md` / `violations.csv`  team description of each violation video, cited rule, how strong the match is; open questions
- `violation_classes.csv`  21 violation classes (team + law): visual cue, act, citation, verbatim quote, link, rule_match, source, example videos, detectability. Rebuilt by `python3 make_violation_classes.py` (checks every quote against units.jsonl)
- `violation_classes.csv` also has `selected` (15 classes chosen by the team on 2026-10-17), `selection_note` and `required_signals` (which model outputs each class needs)
- `object_classes.csv`  detector classes: 10 custom box classes (money_stack, collection_bag, counting_machine, safe_open/closed, key_in/out_of_lock, door_open/closed, ID-identification) + all 80 COCO classes grouped into person / personal / work / laptop / ignore, with the violation classes that use them

- **Correction 2026-10-17 (team):** Х-35 is a violation: at 7–8 s the cashier accepts cash without checking the ID (C01). Labelled clean in `open_labels.xlsx`; changed in `labels.csv`, timestamp 7 s.
- **Correction 2026-10-17 (team):** Х-19 is a violation: at 7–8 s the cashier accepts cash and starts working before checking the ID (C01). Labelled clean in `open_labels.xlsx`; changed in `labels.csv`, timestamp 7 s.
- **Correction 2026-10-18 (team):** Х-37 (8 s) and Х-39 (7–8 s) are violations (client deposits cash without giving an ID, п.19 пп.1). Labelled clean in `open_labels.xlsx`.
