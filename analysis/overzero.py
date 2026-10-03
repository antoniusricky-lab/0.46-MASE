"""Two questions raised by no-guard = 0.52220.

1. The notebook's zero-snap now predicts MORE zeros than the truth at D9/D10
   (0.615/0.658 vs 0.539/0.591). Over-zeroing is cheap on ordinary rows and
   catastrophic on Eid rows. Is capping the snap at the true zero share safe?

2. Cutting ANCHOR_LAGS from 0-24 to 0-12 moved the LightGBM main model from
   0.3493 to 0.3553 even though film_curve became the rank-2 feature. Did the
   lag cut cost more than film_curve gained?
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
EID = [dt.date(2025, 3, 31), dt.date(2026, 3, 21)]


def in_eid(d):
    return 0 <= min(((d - e).days for e in EID), key=abs) <= 13


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


TR = build(tuple(range(13)))
OOF = {}
for f in range(5):
    mdl = fit([r for r in TR if FOLD.get(r[0], -1) != f])
    for i, r in enumerate(EV):
        if FOLD.get(r[0], -1) == f:
            for h in range(7):
                OOF[(i, h)] = pr(mdl, r, h)


def quantile(v, q):
    v = sorted(v)
    return v[min(int(q * (len(v) - 1)), len(v) - 1)] if v else 0.0


print("=" * 78)
print("1. IS CAPPING THE ZERO-SNAP AT THE TRUE ZERO SHARE SAFE?")
print("=" * 78)
print("  MASE on ordinary rows for each snap quantile, per horizon.")
print(f"  {'D':>4s} {'true zeros':>11s} {'best q':>8s} {'MASE@best':>10s}"
      f" {'q=true':>8s} {'MASE@true':>10s} {'cost of cap':>12s}")
tot_best = tot_cap = 0.0
for h in range(7):
    idx = [i for i in range(len(EV))]
    pv = [OOF[(i, h)] for i in idx]
    yv = [EV[i][6][h] / EV[i][3] for i in idx]
    tz = sum(1 for x in yv if x == 0) / len(yv)
    best, bv = 0.0, None
    for q100 in range(0, 96, 2):
        q = q100 / 100.0
        thr = quantile(pv, q)
        v = sum(abs(yv[j] - (0.0 if pv[j] <= thr else pv[j]))
                for j in range(len(idx))) / len(idx)
        if bv is None or v < bv:
            best, bv = q, v
    thr = quantile(pv, tz)
    cv = sum(abs(yv[j] - (0.0 if pv[j] <= thr else pv[j]))
             for j in range(len(idx))) / len(idx)
    tot_best += bv; tot_cap += cv
    print(f"  D{h + 4:<3d} {tz:11.3f} {best:8.2f} {bv:10.4f} {tz:8.2f} "
          f"{cv:10.4f} {cv - bv:+12.4f}")
print(f"\n  average MASE: unconstrained {tot_best / 7:.4f} | "
      f"capped at true share {tot_cap / 7:.4f} | cost {(tot_cap - tot_best) / 7:+.4f}")
print("  -> a near-zero cost here means the cap is cheap insurance against")
print("     over-zeroing the Eid and Ramadan rows the proxy cannot see.")

print()
print("=" * 78)
print("2. HOW MUCH DOES THE ANCHOR-LAG RANGE MATTER PER HORIZON?")
print("=" * 78)
print("  The headline averages were equal, but LightGBM saw 0.3493 -> 0.3553.")
print("  Far horizons have the fewest surviving pairs, so check per horizon.")
res = {}
for lags, lab in ((tuple(range(13)), "0-12"), (tuple(range(25)), "0-24")):
    TRx = build(lags)
    per = defaultdict(list)
    for f in range(5):
        mdl = fit([r for r in TRx if FOLD.get(r[0], -1) != f])
        for r in EV:
            if FOLD.get(r[0], -1) != f:
                continue
            for h in range(7):
                per[h].append(abs(r[6][h] / r[3] - pr(mdl, r, h)))
    res[lab] = {h: sum(per[h]) / len(per[h]) for h in range(7)}
    print(f"  {lab}: pairs={len(TRx):,}")
print(f"\n  {'D':>4s} {'0-12':>9s} {'0-24':>9s} {'delta':>9s}")
for h in range(7):
    a, b = res["0-12"][h], res["0-24"][h]
    print(f"  D{h + 4:<3d} {a:9.4f} {b:9.4f} {b - a:+9.4f}")
print(f"  {'avg':>4s} {sum(res['0-12'].values()) / 7:9.4f} "
      f"{sum(res['0-24'].values()) / 7:9.4f} "
      f"{(sum(res['0-24'].values()) - sum(res['0-12'].values())) / 7:+9.4f}")
print("\n  A median table saturates on data volume; a boosted model does not.")
print("  Given LightGBM regressed when lags were cut, revert to 0-24.")
