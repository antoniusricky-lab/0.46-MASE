"""Two suspected calibration bugs in the corrected pipeline.

BUG 1 - the multiplier grid is clipped. Cell 12 searched range(80,131,2),
i.e. 0.80..1.30, and returned 0.82/0.80/0.80 for D8/D9/D10. Three of seven
horizons pinned to the LOWER BOUND, so the true optimum is below the grid.

BUG 2 - one global zero-snap for all horizons. The true zero share rises from
~29% at D4 to ~54% at D10, so a single threshold cannot serve both ends.

Measured with the same hierarchical-median model the notebook falls back to,
on the same proxy population, so the numbers are comparable to 0.3931/0.3896.
"""
import csv
import datetime as dt
import os
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


def pop(lags, dowf=None):
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
                pat = "".join("1" if x > 0 else "0" for x in s)
                for h in range(7):
                    y = cell.get((m, c, d1 + dt.timedelta(days=3 + h)), 0)
                    out.append((m, h, d1.weekday(), pat, scbin(sc), y / sc))
    return out


TR = pop(tuple(range(25)))
EV = pop((0,), TEST_DOW)
print(f"train rows {len(TR):,} | proxy rows {len(EV):,}")

films = sorted({r[0] for r in EV})
import random
random.Random(SEED).shuffle(films)
FOLD = {m: i % 5 for i, m in enumerate(films)}


def keys(r):
    return [(r[1], r[2], r[3], r[4]), (r[1], r[2], r[3]), (r[1], r[3], r[4]),
            (r[1], r[2]), (r[1], r[3]), (r[1],)]


def fit(rows, min_n=40):
    acc = [defaultdict(list) for _ in range(6)]
    for r in rows:
        for j, k in enumerate(keys(r)):
            acc[j][k].append(r[5])
    return ([{k: med(v) for k, v in a.items() if len(v) >= min_n} for a in acc],
            med([r[5] for r in rows]))


def pred1(mdl, r):
    tabs, g = mdl
    for t, k in zip(tabs, keys(r)):
        if k in t:
            return t[k]
    return g


OOF, YS, HS = [], [], []
for f in range(5):
    mdl = fit([r for r in TR if FOLD.get(r[0], -1) != f])
    for r in EV:
        if FOLD.get(r[0], -1) == f:
            OOF.append(pred1(mdl, r)); YS.append(r[5]); HS.append(r[1])
n = len(OOF)
print(f"oof rows {n:,}  raw MASE {sum(abs(a - b) for a, b in zip(YS, OOF)) / n:.4f}")

print("\n" + "=" * 78)
print("TRUE ZERO SHARE vs horizon (proxy population)")
print("=" * 78)
print(f"  {'D':>4s} {'true zero share':>16s} {'true median':>12s} {'true mean':>10s}")
for h in range(7):
    v = [YS[i] for i in range(n) if HS[i] == h]
    print(f"  D{h + 4:<3d} {sum(1 for x in v if x == 0) / len(v):16.3f} "
          f"{med(v):12.3f} {sum(v) / len(v):10.3f}")


def mase(p):
    return sum(abs(YS[i] - p[i]) for i in range(n)) / n


print("\n" + "=" * 78)
print("BUG 1 - multiplier grid clipped at 0.80")
print("=" * 78)
for lo, hi, lab in ((80, 130, "0.80..1.30  (what the notebook used)"),
                    (20, 140, "0.20..1.40  (widened)")):
    mult = {}
    for h in range(7):
        idx = [i for i in range(n) if HS[i] == h]
        best, bv = 1.0, None
        for m100 in range(lo, hi + 1, 2):
            mu = m100 / 100.0
            v = sum(abs(YS[i] - OOF[i] * mu) for i in idx) / len(idx)
            if bv is None or v < bv:
                best, bv = mu, v
        mult[h] = best
    p = [OOF[i] * mult[HS[i]] for i in range(n)]
    pin = [h for h in range(7) if abs(mult[h] - lo / 100.0) < 1e-9]
    print(f"  {lab}")
    print(f"    multipliers {{{', '.join(f'D{h + 4}: {mult[h]:.2f}' for h in range(7))}}}")
    print(f"    pinned to the lower bound: {[f'D{h + 4}' for h in pin]}")
    print(f"    MASE {mase(p):.4f}")
    if lo == 20:
        MULT_WIDE, P_WIDE = mult, p

print("\n" + "=" * 78)
print("BUG 2 - one global zero-snap vs one per horizon")
print("=" * 78)
base = P_WIDE


def snap_global(p):
    best, bv = 0.0, None
    for z100 in range(0, 81, 2):
        z = z100 / 100.0
        v = sum(abs(YS[i] - (0.0 if p[i] < z else p[i])) for i in range(n)) / n
        if bv is None or v < bv:
            best, bv = z, v
    return best, [0.0 if x < best else x for x in p]


def snap_per_h(p):
    z = {}
    for h in range(7):
        idx = [i for i in range(n) if HS[i] == h]
        best, bv = 0.0, None
        for z100 in range(0, 121, 2):
            zz = z100 / 100.0
            v = sum(abs(YS[i] - (0.0 if p[i] < zz else p[i]))
                    for i in idx) / len(idx)
            if bv is None or v < bv:
                best, bv = zz, v
        z[h] = best
    return z, [0.0 if p[i] < z[HS[i]] else p[i] for i in range(n)]


zg, pg = snap_global(base)
zh, ph = snap_per_h(base)
print(f"  no snap            MASE {mase(base):.4f}")
print(f"  global snap {zg:.2f}   MASE {mase(pg):.4f}")
print(f"  per-horizon snap   MASE {mase(ph):.4f}")
print(f"    thresholds {{{', '.join(f'D{h + 4}: {zh[h]:.2f}' for h in range(7))}}}")
print("\n  resulting predicted zero share vs truth:")
print(f"  {'D':>4s} {'true':>8s} {'global':>8s} {'per-h':>8s}")
for h in range(7):
    idx = [i for i in range(n) if HS[i] == h]
    t = sum(1 for i in idx if YS[i] == 0) / len(idx)
    a = sum(1 for i in idx if pg[i] == 0) / len(idx)
    b = sum(1 for i in idx if ph[i] == 0) / len(idx)
    print(f"  D{h + 4:<3d} {t:8.3f} {a:8.3f} {b:8.3f}")

print("\n" + "=" * 78)
print("COMBINED EFFECT")
print("=" * 78)
m_old = {}
for h in range(7):
    idx = [i for i in range(n) if HS[i] == h]
    best, bv = 1.0, None
    for m100 in range(80, 131, 2):
        mu = m100 / 100.0
        v = sum(abs(YS[i] - OOF[i] * mu) for i in idx) / len(idx)
        if bv is None or v < bv:
            best, bv = mu, v
    m_old[h] = best
p_old = [OOF[i] * m_old[HS[i]] for i in range(n)]
z_old, p_old = snap_global(p_old)
print(f"  notebook as shipped (grid 0.80.., global snap) : {mase(p_old):.4f}")
print(f"  widened grid + per-horizon snap                : {mase(ph):.4f}")
print(f"  improvement                                    : "
      f"{mase(p_old) - mase(ph):+.4f}")
