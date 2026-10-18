# VLM baseline results

| variant | agg | acc | prec | rec | FP clean | ts first ±2/±5 | ts best ±2/±5 | type first/any | calls/video | VLM s/video (serial) | parse fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAND_P1 | any | 0.76 | 0.83 | 0.68 | 3/20 | 5/10 | 5/10 | 4/4 of 15 | 1.0 | 2 | 0 |
| CAND_P1 | noC14 | 0.50 | 0.67 | 0.09 | 1/20 | 1/1 | 1/1 | 1/1 of 2 | 1.0 | 2 | 0 |
| CAND_P1_think | any | 0.50 | 0.57 | 0.18 | 3/20 | 0/3 | 1/4 | 3/3 of 4 | 1.0 | 19 | 0 |
| CAND_P1_think | noC14 | 0.48 | 0.50 | 0.14 | 3/20 | 0/2 | 1/3 | 2/2 of 3 | 1.0 | 19 | 0 |
| OV_P1 | any | 0.43 | 0.40 | 0.18 | 6/20 | 0/0 | 0/0 | 1/1 of 4 | 1.0 | 2 | 0 |
| OV_P1 | noC14 | 0.43 | 0.00 | 0.00 | 2/20 | 0/0 | 0/0 | 0/0 of 0 | 1.0 | 2 | 0 |
| OV_P1_think | any | 0.38 | 0.30 | 0.14 | 7/20 | 0/1 | 0/1 | 1/1 of 3 | 1.0 | 35 | 3 |
| OV_P1_think | noC14 | 0.38 | 0.30 | 0.14 | 7/20 | 0/1 | 0/1 | 1/1 of 3 | 1.0 | 35 | 3 |
| OV_P2 | any | 0.43 | 0.42 | 0.23 | 7/20 | 0/0 | 0/0 | 1/1 of 5 | 1.0 | 1 | 0 |
| OV_P2 | noC14 | 0.43 | 0.00 | 0.00 | 2/20 | 0/0 | 0/0 | 0/0 of 0 | 1.0 | 1 | 0 |
| W12_P1 | any | 0.60 | 0.58 | 0.82 | 13/20 | 3/6 | 7/14 | 5/6 of 18 | 5.0 | 9 | 0 |
| W12_P1 | vote2 | 0.52 | 0.56 | 0.45 | 8/20 | 1/3 | 5/9 | 3/3 of 10 | 5.0 | 9 | 0 |
| W12_P1 | noC14 | 0.50 | 0.57 | 0.18 | 3/20 | 1/2 | 1/2 | 3/3 of 4 | 5.0 | 9 | 0 |
| W12_P1_think | any | 0.60 | 0.67 | 0.45 | 5/20 | 3/5 | 4/6 | 5/5 of 10 | 5.0 | 108 | 9 |
| W12_P1_think | vote2 | 0.48 | 0.50 | 0.09 | 2/20 | 0/1 | 0/1 | 2/2 of 2 | 5.0 | 108 | 9 |
| W12_P1_think | noC14 | 0.57 | 0.67 | 0.36 | 4/20 | 3/4 | 4/5 | 4/4 of 8 | 5.0 | 108 | 9 |
| W12_P2 | any | 0.57 | 0.56 | 0.82 | 14/20 | 2/6 | 7/14 | 4/5 of 18 | 5.0 | 8 | 0 |
| W12_P2 | vote2 | 0.55 | 0.57 | 0.59 | 10/20 | 1/5 | 6/12 | 4/4 of 13 | 5.0 | 8 | 0 |
| W12_P2 | noC14 | 0.52 | 0.67 | 0.18 | 2/20 | 2/4 | 2/4 | 2/2 of 4 | 5.0 | 8 | 0 |
| W12c_P1 | any | 0.55 | 0.55 | 0.77 | 14/20 | 4/7 | 9/13 | 6/7 of 17 | 5.0 | 9 | 0 |
| W12c_P1 | vote2 | 0.48 | 0.50 | 0.32 | 7/20 | 0/1 | 3/6 | 3/3 of 7 | 5.0 | 9 | 0 |
| W12c_P1 | noC14 | 0.48 | 0.50 | 0.32 | 7/20 | 2/3 | 2/4 | 4/4 of 7 | 5.0 | 9 | 0 |
| W16_P1 | any | 0.48 | 0.50 | 0.68 | 15/20 | 2/4 | 7/11 | 4/5 of 15 | 7.8 | 13 | 0 |
| W16_P1 | vote2 | 0.50 | 0.53 | 0.45 | 9/20 | 1/1 | 6/8 | 2/3 of 10 | 7.8 | 13 | 0 |
| W16_P1 | noC14 | 0.50 | 0.56 | 0.23 | 4/20 | 1/2 | 2/2 | 2/2 of 5 | 7.8 | 13 | 0 |

ts: |pred start - label| over true-positive videos; first = earliest reported event, best = event closest to label.
type: first event's class in gold set / any reported class in gold set.

## Per-class (clean videos with class = FP; hits = violation videos where gold class predicted)

**CAND_P1/any**: FP-by-class {'C12': 1, 'C14': 2}; hits {'C12': 1, 'C14': 3}; wrong-class-on-positive {'C14': 10, 'C16': 1}

**CAND_P1/noC14**: FP-by-class {'C12': 1}; hits {'C12': 1}; wrong-class-on-positive {'C16': 1}

**CAND_P1_think/any**: FP-by-class {'C12': 2, 'C16': 1}; hits {'C12': 1, 'C14': 1, 'C16': 1}; wrong-class-on-positive {'C16': 2}

**CAND_P1_think/noC14**: FP-by-class {'C12': 2, 'C16': 1}; hits {'C12': 1, 'C16': 1}; wrong-class-on-positive {'C16': 2}

**OV_P1/any**: FP-by-class {'C10': 1, 'C14': 4, 'C16': 1}; hits {'C14': 1}; wrong-class-on-positive {'C14': 3}

**OV_P1/noC14**: FP-by-class {'C10': 1, 'C16': 1}; hits {}; wrong-class-on-positive {}

**OV_P1_think/any**: FP-by-class {'C14': 1, 'C16': 7}; hits {'C12': 1}; wrong-class-on-positive {'C16': 2}

**OV_P1_think/noC14**: FP-by-class {'C16': 7}; hits {'C12': 1}; wrong-class-on-positive {'C16': 2}

**OV_P2/any**: FP-by-class {'C14': 5, 'C16': 2}; hits {'C14': 1}; wrong-class-on-positive {'C14': 4}

**OV_P2/noC14**: FP-by-class {'C16': 2}; hits {}; wrong-class-on-positive {}

**W12_P1/any**: FP-by-class {'C01': 1, 'C10': 1, 'C12': 1, 'C14': 13}; hits {'C01': 1, 'C12': 2, 'C14': 3}; wrong-class-on-positive {'C14': 14, 'C16': 1}

**W12_P1/vote2**: FP-by-class {'C12': 1, 'C14': 8}; hits {'C14': 3}; wrong-class-on-positive {'C14': 7}

**W12_P1/noC14**: FP-by-class {'C01': 1, 'C10': 1, 'C12': 1}; hits {'C01': 1, 'C12': 2}; wrong-class-on-positive {'C16': 1}

**W12_P1_think/any**: FP-by-class {'C12': 1, 'C14': 1, 'C16': 3}; hits {'C08': 1, 'C12': 2, 'C14': 1, 'C16': 1}; wrong-class-on-positive {'C01': 1, 'C06': 1, 'C14': 1, 'C16': 4}

**W12_P1_think/vote2**: FP-by-class {'C12': 1, 'C16': 1}; hits {'C12': 1, 'C14': 1}; wrong-class-on-positive {}

**W12_P1_think/noC14**: FP-by-class {'C12': 1, 'C16': 3}; hits {'C08': 1, 'C12': 2, 'C16': 1}; wrong-class-on-positive {'C01': 1, 'C06': 1, 'C16': 4}

**W12_P2/any**: FP-by-class {'C10': 1, 'C12': 1, 'C14': 14}; hits {'C12': 1, 'C14': 3, 'C17': 1}; wrong-class-on-positive {'C12': 1, 'C14': 14, 'C16': 1}

**W12_P2/vote2**: FP-by-class {'C12': 1, 'C14': 9}; hits {'C12': 1, 'C14': 3}; wrong-class-on-positive {'C14': 9}

**W12_P2/noC14**: FP-by-class {'C10': 1, 'C12': 1}; hits {'C12': 1, 'C17': 1}; wrong-class-on-positive {'C12': 1, 'C16': 1}

**W12c_P1/any**: FP-by-class {'C01': 1, 'C08': 4, 'C14': 14, 'C16': 2}; hits {'C08': 2, 'C12': 2, 'C14': 3}; wrong-class-on-positive {'C01': 1, 'C14': 12, 'C16': 2}

**W12c_P1/vote2**: FP-by-class {'C14': 6, 'C16': 1}; hits {'C12': 1, 'C14': 2}; wrong-class-on-positive {'C14': 4}

**W12c_P1/noC14**: FP-by-class {'C01': 1, 'C08': 4, 'C16': 2}; hits {'C08': 2, 'C12': 2}; wrong-class-on-positive {'C01': 1, 'C16': 2}

**W16_P1/any**: FP-by-class {'C08': 1, 'C10': 1, 'C12': 3, 'C14': 12}; hits {'C08': 1, 'C12': 1, 'C14': 3}; wrong-class-on-positive {'C08': 2, 'C14': 10, 'C16': 1}

**W16_P1/vote2**: FP-by-class {'C12': 1, 'C14': 8}; hits {'C12': 1, 'C14': 2}; wrong-class-on-positive {'C14': 7, 'C16': 1}

**W16_P1/noC14**: FP-by-class {'C08': 1, 'C10': 1, 'C12': 3}; hits {'C08': 1, 'C12': 1}; wrong-class-on-positive {'C08': 2, 'C16': 1}
