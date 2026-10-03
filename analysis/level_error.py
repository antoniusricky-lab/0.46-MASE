"""Quantify the far-horizon level error in the submitted predictions.

MASE is L1, so the optimal constant per cell is the MEDIAN. Cell 13 shows the
submission predicting roughly the conditional MEAN at D8-D10, where the true
median is 0. Measure what that costs.
"""
import csv
import datetime as dt
import os
from collections import defaultdict

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
TEST_DOW = {2, 3, 4}

cell, ncl, nat, mdc = {}, defaultdict(lambda: defaultdict(int)), \
    defaultdict(lambda: defaultdict(int)), defaultdict(list)
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d = dt.date.fromisoformat(r["date_show"]); m = r["movie_title"]
        cell[(m, r["cinema_ids"], d)] = int(r["total_ticket"])
        ncl[m][d] += 1; nat[m][d] += int(r["total_ticket"])
        mdc[(m, d)].append(r["cinema_ids"])
TMIN = min(d for m in nat for d in nat[m]); TMAX = max(d for m in nat for d in nat[m])
first = {m: min(v) for m, v in nat.items()}
LEFT = {m for m in first if first[m] == TMIN}
release = {}
for m in first:
    if m in LEFT:
        continue
    wk = [d for d in sorted(nat[m]) if d <= first[m] + dt.timedelta(days=9)]
    pk = max(ncl[m][d] for d in wk)
    for d in wk:
        if ncl[m][d] >= 0.25 * pk:
            release[m] = d
            break

rows = defaultdict(list)
for m in sorted(release):
    d1 = release[m]
    if d1 + dt.timedelta(days=9) > TMAX or d1.weekday() not in TEST_DOW:
        continue
    obs = [d1 + dt.timedelta(days=i) for i in range(3)]
    cs = set()
    for dd in obs:
        cs.update(mdc.get((m, dd), ()))
    for c in cs:
        s = [cell.get((m, c, dd), 0) for dd in obs]
        if sum(s) == 0 or s[2] == 0:
            continue
        sc = max(sum(s) / 3.0, 1.0)
        for h in range(7):
            rows[h].append(cell.get((m, c, d1 + dt.timedelta(days=3 + h)), 0) / sc)


def med(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else .5 * (v[n // 2 - 1] + v[n // 2])


# what the two submissions predict, from the notebook cell-13 / 0.46641 outputs
MINE = {4: 0.941, 5: 0.645, 6: 0.507, 7: 0.490, 8: 0.401, 9: 0.375, 10: 0.358}
THEIRS = {4: 0.952, 5: 0.686, 6: 0.440, 7: 0.344, 8: 0.183, 9: 0.142, 10: 0.132}

print("=" * 82)
print("COST OF A CONSTANT PREDICTOR AT EACH HORIZON (proxy population)")
print("=" * 82)
print("  Not a model comparison - it isolates the LEVEL each submission sits at.")
print(f"\n  {'D':>4s} {'true med':>9s} {'true mean':>10s} {'zero%':>7s}"
      f" {'opt const':>10s} {'MASE opt':>9s}"
      f" {'mine':>7s} {'MASE':>8s} {'theirs':>7s} {'MASE':>8s}")
tot_opt = tot_mine = tot_theirs = 0.0
for h in range(7):
    v = rows[h]
    nn = len(v)
    mv = med(v)
    best, bq = None, None
    for q100 in range(0, 201, 1):
        q = q100 / 100.0
        c = sum(abs(x - q) for x in v) / nn
        if bq is None or c < bq:
            best, bq = q, c
    a = sum(abs(x - MINE[h + 4]) for x in v) / nn
    b = sum(abs(x - THEIRS[h + 4]) for x in v) / nn
    tot_opt += bq; tot_mine += a; tot_theirs += b
    print(f"  D{h + 4:<3d} {mv:9.3f} {sum(v) / nn:10.3f} "
          f"{100 * sum(1 for x in v if x == 0) / nn:6.1f}% {best:10.2f} {bq:9.4f}"
          f" {MINE[h + 4]:7.3f} {a:8.4f} {THEIRS[h + 4]:7.3f} {b:8.4f}")
print(f"  {'avg':>4s} {'':9s} {'':10s} {'':7s} {'':10s} {tot_opt / 7:9.4f}"
      f" {'':7s} {tot_mine / 7:8.4f} {'':7s} {tot_theirs / 7:8.4f}")

print()
print("=" * 82)
print("WHERE THE SUBMITTED LEVELS DIVERGE")
print("=" * 82)
print(f"  {'D':>4s} {'optimal':>9s} {'mine':>8s} {'excess':>8s}"
      f" {'theirs':>8s} {'excess':>8s} {'cost mine - theirs':>20s}")
for h in range(7):
    v = rows[h]
    nn = len(v)
    best, bq = None, None
    for q100 in range(0, 201, 1):
        q = q100 / 100.0
        c = sum(abs(x - q) for x in v) / nn
        if bq is None or c < bq:
            best, bq = q, c
    a = sum(abs(x - MINE[h + 4]) for x in v) / nn
    b = sum(abs(x - THEIRS[h + 4]) for x in v) / nn
    print(f"  D{h + 4:<3d} {best:9.2f} {MINE[h + 4]:8.3f} "
          f"{MINE[h + 4] - best:+8.3f} {THEIRS[h + 4]:8.3f} "
          f"{THEIRS[h + 4] - best:+8.3f} {(a - b) / 7:+20.4f}")
print(f"\n  total level-only penalty, mine vs theirs: "
      f"{(tot_mine - tot_theirs) / 7:+.4f} MASE")
print("  (constant-predictor proxy, so it overstates magnitude, but the SIGN")
print("   and the concentration in D8-D10 are what matter)")
