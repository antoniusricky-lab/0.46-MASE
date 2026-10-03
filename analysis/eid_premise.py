"""Test the Idulfitri guard's premise against the only Eid analogue in train.

The guard assumes: when target days fall in an Eid window, y/scale runs high,
so flooring predictions at K * uplift helps.

Train has no window that STRADDLES into Eid, but it does have windows wholly
inside the Idulfitri 1446 aftermath (2025-04-01..04-13). If `uplift` predicts
y/scale well there, the guard's mechanism is sound. If not, it is a bad bet.
"""
import csv
import datetime as dt
import os
from collections import defaultdict

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOWN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
EID = [dt.date(2025, 3, 31), dt.date(2026, 3, 21)]

cell, ncl, nat, mdc = {}, defaultdict(lambda: defaultdict(int)), \
    defaultdict(lambda: defaultdict(int)), defaultdict(list)
natday = defaultdict(int)
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d = dt.date.fromisoformat(r["date_show"]); m = r["movie_title"]
        c = r["cinema_ids"]; t = int(r["total_ticket"])
        cell[(m, c, d)] = t
        ncl[m][d] += 1; nat[m][d] += t; mdc[(m, d)].append(c); natday[d] += t
TMIN, TMAX = min(natday), max(natday)
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

HOL = {}
with open(D + "holidays.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        HOL[dt.date.fromisoformat(r["date"])] = (r["day_tipe"], r["holiday_tipe"])


def is_hol(d):
    return HOL.get(d, ("weekday", "normal"))[1] == "holiday"


def med(v):
    v = sorted(v); n = len(v)
    return 0.0 if n == 0 else (v[n // 2] if n % 2 else .5 * (v[n // 2 - 1] + v[n // 2]))


# replicate the notebook's detrended dow / holiday factors
days = sorted(natday)
roll = {}
for i, d in enumerate(days):
    lo, hi = max(0, i - 3), min(len(days), i + 4)
    w = [natday[x] for x in days[lo:hi]]
    roll[d] = sum(w) / len(w)
rat = {d: natday[d] / roll[d] for d in days if roll[d] > 0}
DOWF = {w: med([rat[d] for d in rat if d.weekday() == w and not is_hol(d)])
        for w in range(7)}
HM = med([rat[d] / DOWF[d.weekday()] for d in rat if is_hol(d)])


def calf(d):
    f = DOWF[d.weekday()]
    return f * HM if is_hol(d) else f


def dse(d):
    return min(((d - e).days for e in EID), key=abs)


def in_eid(d):
    return 0 <= dse(d) <= 13


# ---------------------------------------------------------------- windows
rows = []
for m in sorted(release):
    for lag in range(25):
        d1 = release[m] + dt.timedelta(days=lag)
        if d1 + dt.timedelta(days=9) > TMAX:
            continue
        obs = [d1 + dt.timedelta(days=i) for i in range(3)]
        base = sum(calf(d) for d in obs) / 3.0
        cs = set()
        for dd in obs:
            cs.update(mdc.get((m, dd), ()))
        for c in cs:
            s = [cell.get((m, c, dd), 0) for dd in obs]
            if sum(s) == 0 or s[2] == 0:
                continue
            sc = max(sum(s) / 3.0, 1.0)
            nobs = sum(1 for d in obs if in_eid(d))
            for h in range(7):
                t = d1 + dt.timedelta(days=3 + h)
                y = cell.get((m, c, t), 0)
                rows.append((h, y / sc, calf(t) / base, in_eid(t), nobs,
                             dse(t), m))

print("=" * 78)
print("TRUE y/scale ON EID-WINDOW TARGET DAYS (train, Idulfitri 1446 aftermath)")
print("=" * 78)
eid_rows = [r for r in rows if r[3]]
oth = [r for r in rows if not r[3]]
print(f"  eid-window rows {len(eid_rows):,} | other rows {len(oth):,}")
for lab, sel in (("eid window", eid_rows), ("all other", oth)):
    v = sorted(x[1] for x in sel)
    n = len(v)
    print(f"  {lab:12s} y/scale  mean {sum(v) / n:6.3f}  median {v[n // 2]:6.3f}  "
          f"p25 {v[n // 4]:6.3f}  p75 {v[3 * n // 4]:6.3f}  p90 {v[9 * n // 10]:6.3f}  "
          f"zeros {100 * sum(1 for x in v if x == 0) / n:4.1f}%")

print()
print("=" * 78)
print("DOES `uplift` PREDICT y/scale ON EID DAYS?  (the guard's premise)")
print("=" * 78)
print("  uplift bucket -> median realised y/scale, eid-window rows only")
print(f"  {'uplift':>14s} {'n':>6s} {'median y/scale':>15s} {'mean':>8s} "
      f"{'ratio/uplift':>13s}")
bk = defaultdict(list)
for r in eid_rows:
    u = r[2]
    b = 0 if u <= .7 else 1 if u <= .9 else 2 if u <= 1.1 else 3 if u <= 1.5 else 4
    bk[b].append(r)
LAB = ["<=0.7", "0.7-0.9", "0.9-1.1", "1.1-1.5", ">1.5"]
for b in sorted(bk):
    v = bk[b]
    mv = med([x[1] for x in v])
    mu = sum(x[2] for x in v) / len(v)
    print(f"  {LAB[b]:>14s} {len(v):6d} {mv:15.3f} "
          f"{sum(x[1] for x in v) / len(v):8.3f} {mv / mu:13.3f}")

print()
print("  same for NON-eid rows, as the control:")
bk2 = defaultdict(list)
for r in oth:
    u = r[2]
    b = 0 if u <= .7 else 1 if u <= .9 else 2 if u <= 1.1 else 3 if u <= 1.5 else 4
    bk2[b].append(r)
for b in sorted(bk2):
    v = bk2[b]
    mv = med([x[1] for x in v])
    mu = sum(x[2] for x in v) / len(v)
    print(f"  {LAB[b]:>14s} {len(v):6d} {mv:15.3f} "
          f"{sum(x[1] for x in v) / len(v):8.3f} {mv / mu:13.3f}")

print()
print("=" * 78)
print("BY days_since_eid: what the cohort's D4-D10 (Eid+0..+6) would look like")
print("=" * 78)
print(f"  {'dse':>5s} {'n':>6s} {'median y/scale':>15s} {'mean':>8s} "
      f"{'median uplift':>14s} {'zeros':>7s}")
by = defaultdict(list)
for r in eid_rows:
    by[r[5]].append(r)
for k in sorted(by):
    v = by[k]
    ys = [x[1] for x in v]
    print(f"  {k:5d} {len(v):6d} {med(ys):15.3f} {sum(ys) / len(ys):8.3f} "
          f"{med([x[2] for x in v]):14.3f} "
          f"{100 * sum(1 for x in ys if x == 0) / len(ys):6.1f}%")

print()
print("=" * 78)
print("WHAT FLOOR WOULD HAVE BEEN OPTIMAL ON THESE ROWS?")
print("=" * 78)
print("  Using a plain horizon-median predictor as the stand-in baseline.")
base_pred = {h: med([r[1] for r in rows if r[0] == h]) for h in range(7)}
print(f"  horizon medians: { {h: round(base_pred[h], 3) for h in range(7)} }")
for mode in ("flat", "calendar"):
    print(f"\n  {mode} floor:")
    best, bv = None, None
    for k100 in range(0, 201, 10):
        k = k100 / 100.0
        tot = 0.0
        for r in eid_rows:
            p = base_pred[r[0]]
            fl = k if mode == "flat" else k * r[2]
            tot += abs(r[1] - max(p, fl))
        v = tot / len(eid_rows)
        if bv is None or v < bv:
            best, bv = k, v
        if k100 % 20 == 0:
            print(f"    K={k:4.2f}  MASE on eid rows = {v:.4f}")
    print(f"    best K = {best:.2f} -> {bv:.4f}")
no_floor = sum(abs(r[1] - base_pred[r[0]]) for r in eid_rows) / len(eid_rows)
print(f"\n  no floor at all: {no_floor:.4f}")
