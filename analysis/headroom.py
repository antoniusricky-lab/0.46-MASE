"""Two cheap checks before committing effort.

1. Is the `001` bucket (2% of rows, ~15% of error, MASE 2.85) fixable at all,
   or is its spread irreducible? Compare against the best possible CONSTANT
   and against an oracle that knows the pair's own 7-day total.
2. Does extending the anchor-lag range past 24 still help?
"""
import csv
import datetime as dt
import os
import random
from collections import defaultdict

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
TEST_DOW = {2, 3, 4}
SEED = 2026

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


def med(v):
    v = sorted(v); n = len(v)
    return 0.0 if n == 0 else (v[n // 2] if n % 2 else .5 * (v[n // 2 - 1] + v[n // 2]))


def scbin(x):
    for i, b in enumerate((5, 10, 25, 50, 100, 250, 600)):
        if x <= b:
            return i
    return 7


def build(lags, dowf=None):
    out = []
    for m in sorted(release):
        for lag in lags:
            d1 = release[m] + dt.timedelta(days=lag)
            if d1 + dt.timedelta(days=9) > TMAX:
                continue
            if dowf and d1.weekday() not in dowf:
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
                out.append((m, d1, c, sc,
                            "".join("1" if x > 0 else "0" for x in s), scbin(sc),
                            [cell.get((m, c, d1 + dt.timedelta(days=3 + h)), 0)
                             for h in range(7)]))
    return out


EV = build((0,), TEST_DOW)
films = sorted({r[0] for r in EV})
random.Random(SEED).shuffle(films)
FOLD = {m: i % 5 for i, m in enumerate(films)}


def keys(r, h):
    return [(h, r[1].weekday(), r[4], r[5]), (h, r[1].weekday(), r[4]),
            (h, r[4], r[5]), (h, r[1].weekday()), (h, r[4]), (h,)]


def fit(rows, min_n=40):
    acc = [defaultdict(list) for _ in range(6)]
    for r in rows:
        for h in range(7):
            for j, k in enumerate(keys(r, h)):
                acc[j][k].append(r[6][h] / r[3])
    return ([{k: med(v) for k, v in a.items() if len(v) >= min_n} for a in acc],
            med([r[6][h] / r[3] for r in rows for h in range(7)]))


def pr(mdl, r, h):
    tabs, g = mdl
    for t, k in zip(tabs, keys(r, h)):
        if k in t:
            return t[k]
    return g


print("=" * 78)
print("1. IS THE `001` BUCKET FIXABLE?")
print("=" * 78)
TR24 = build(tuple(range(25)))
OOF = {}
for f in range(5):
    mdl = fit([r for r in TR24 if FOLD.get(r[0], -1) != f])
    for i, r in enumerate(EV):
        if FOLD.get(r[0], -1) == f:
            for h in range(7):
                OOF[(i, h)] = pr(mdl, r, h)

for pat in ("111", "011", "001", "101"):
    idx = [(i, h) for i, r in enumerate(EV) if r[4] == pat for h in range(7)]
    if not idx:
        continue
    ys = [EV[i][6][h] / EV[i][3] for i, h in idx]
    n = len(ys)
    cur = sum(abs(EV[i][6][h] / EV[i][3] - OOF[(i, h)]) for i, h in idx) / n
    # best single constant
    bc, bv = 0.0, None
    for q100 in range(0, 801, 5):
        q = q100 / 100.0
        v = sum(abs(y - q) for y in ys) / n
        if bv is None or v < bv:
            bc, bv = q, v
    # oracle: the pair's own 7-day total is known, shape from the model
    tot = 0.0
    for i, r in enumerate(EV):
        if r[4] != pat:
            continue
        T = sum(r[6]) / r[3]
        sh = sum(OOF[(i, k)] for k in range(7))
        for h in range(7):
            p = T * (OOF[(i, h)] / sh) if sh > 1e-9 else T / 7.0
            tot += abs(r[6][h] / r[3] - p)
    orc = tot / n
    sv = sorted(ys)
    print(f"  pattern {pat}  n={n:6d}  share={100 * n / (7 * len(EV)):5.2f}%")
    print(f"    true y/scale   median {sv[n // 2]:6.3f}  mean {sum(ys) / n:6.3f}  "
          f"p90 {sv[9 * n // 10]:6.3f}  zeros {100 * sum(1 for x in ys if x == 0) / n:4.1f}%")
    print(f"    model MASE          {cur:.4f}")
    print(f"    best CONSTANT       {bv:.4f}  (at {bc:.2f})")
    print(f"    oracle 7-day total  {orc:.4f}")
    print(f"    -> headroom vs best constant {bv - cur:+.4f}, "
          f"vs oracle {orc - cur:+.4f}")

print()
print("=" * 78)
print("2. DOES EXTENDING THE ANCHOR-LAG RANGE STILL HELP?")
print("=" * 78)
Nrows = 7 * len(EV)
for lags, lab in ((tuple(range(1)), "lag 0 only"),
                  (tuple(range(13)), "lags 0-12"),
                  (tuple(range(25)), "lags 0-24  (current)"),
                  (tuple(range(41)), "lags 0-40"),
                  (tuple(range(61)), "lags 0-60")):
    TRx = build(lags)
    tot = 0.0
    for f in range(5):
        mdl = fit([r for r in TRx if FOLD.get(r[0], -1) != f])
        for r in EV:
            if FOLD.get(r[0], -1) != f:
                continue
            for h in range(7):
                tot += abs(r[6][h] / r[3] - pr(mdl, r, h))
    print(f"  {lab:24s} pairs={len(TRx):7d}  MASE={tot / Nrows:.4f}")
