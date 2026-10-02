"""Targeted test: does an observed date-level demand signal help on windows
whose target days are calendar ANOMALIES (holidays / the post-Idulfitri boom)?

That is the only part of train that resembles the Ramadan + Idulfitri +
Christmas stretch covering ~32% of the test rows.
"""
import csv
import datetime as dt
from collections import defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TEST_DOW = {2, 3, 4}
SEED = 2026

cell, ncl, nat, mdc = {}, defaultdict(lambda: defaultdict(int)), \
    defaultdict(lambda: defaultdict(int)), defaultdict(list)
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d = dt.date.fromisoformat(r["date_show"]); m = r["movie_title"]
        c = r["cinema_ids"]; t = int(r["total_ticket"])
        cell[(m, c, d)] = t; ncl[m][d] += 1; nat[m][d] += t; mdc[(m, d)].append(c)
TMIN = min(d for m in nat for d in nat[m]); TMAX = max(d for m in nat for d in nat[m])
first = {m: min(v) for m, v in nat.items()}
LEFT = {m for m in first if first[m] == TMIN}
release = {}
for m in first:
    if m in LEFT: continue
    wk = [d for d in sorted(nat[m]) if d <= first[m] + dt.timedelta(days=9)]
    pk = max(ncl[m][d] for d in wk)
    for d in wk:
        if ncl[m][d] >= 0.25 * pk:
            release[m] = d; break

hol = {}
with open(D + "holidays.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        hol[dt.date.fromisoformat(r["date"])] = (r["day_tipe"], r["holiday_tipe"])

# ---- the "anomaly" set in train: named holidays + post-Idulfitri boom (Apr 1-13)
BOOM_A, BOOM_B = dt.date(2025, 4, 1), dt.date(2025, 4, 13)


def anomalous(d):
    if BOOM_A <= d <= BOOM_B:
        return True
    return hol.get(d, ("weekday", "normal"))[1] == "holiday"


# true national daily total (full train visibility, for reference only)
nat_day = defaultdict(int)
for (m, c, d), t in cell.items():
    nat_day[d] += t

obs_films = defaultdict(set)
for m, r0 in release.items():
    for i in range(3):
        obs_films[r0 + dt.timedelta(days=i)].add(m)


def med(v):
    v = sorted(v); n = len(v)
    return 0.0 if n == 0 else (v[n//2] if n % 2 else 0.5*(v[n//2-1]+v[n//2]))


# observed date factor under the test availability rule
acc = defaultdict(list)
for m, r0 in release.items():
    ds = [r0 + dt.timedelta(days=i) for i in range(3)]
    v = [nat[m].get(d, 0) for d in ds]
    mu = sum(v) / 3.0
    if mu <= 0: continue
    for d, x in zip(ds, v):
        acc[d].append((x / mu, m))
DFAC = {d: med([x for x, _ in v]) for d, v in acc.items() if len(v) >= 3}


def dfac_ex(d, m):
    v = [x for x, mm in acc.get(d, []) if mm != m]
    return med(v) if len(v) >= 3 else None


def scbin(x):
    for i, b in enumerate((5, 10, 25, 50, 100, 250, 600)):
        if x <= b: return i
    return 7


def fbin(x):
    if x is None: return -1
    for i, b in enumerate((0.55, 0.75, 0.95, 1.15, 1.45)):
        if x <= b: return i
    return 5


def pop_lags(lags, dowf=None):
    out = []
    for m in sorted(release):
        for lag in lags:
            d1 = release[m] + dt.timedelta(days=lag)
            if d1 + dt.timedelta(days=9) > TMAX: continue
            if dowf and d1.weekday() not in dowf: continue
            obs = [d1 + dt.timedelta(days=i) for i in range(3)]
            tgt = [d1 + dt.timedelta(days=i) for i in range(3, 10)]
            cs = set()
            for dd in obs: cs.update(mdc.get((m, dd), ()))
            for c in cs:
                s = [cell.get((m, c, dd), 0) for dd in obs]
                if sum(s) == 0 or s[2] == 0: continue
                out.append((m, c, d1, s, [cell.get((m, c, dd), 0) for dd in tgt]))
    return out


def prep(rows):
    out = []
    for (m, c, d1, s, y) in rows:
        p = "".join("1" if x > 0 else "0" for x in s)
        sc = max(sum(s) / 3.0, 1.0)
        fx = []
        for h in range(7):
            t = d1 + dt.timedelta(days=3 + h)
            fx.append((dfac_ex(t, m), anomalous(t)))
        out.append((m, c, d1, sc, y, p, scbin(sc), fx))
    return out


def folds(rows, k=5):
    films = sorted({r[0] for r in rows}); st = SEED
    for i in range(len(films) - 1, 0, -1):
        st = (1103515245 * st + 12345) % (1 << 31)
        j = st % (i + 1); films[i], films[j] = films[j], films[i]
    return {m: i % k for i, m in enumerate(films)}


TR = prep(pop_lags(tuple(range(25))))
EV = prep(pop_lags((0,), TEST_DOW))
assign = folds(EV)

K_BASE = lambda r, h, fx: [(h, r[2].weekday(), r[5], r[6]),
                           (h, r[2].weekday(), r[5]), (h, r[5], r[6]),
                           (h, r[2].weekday()), (h, r[5]), (h,)]
K_DF = lambda r, h, fx: [(h, r[2].weekday(), r[5], r[6], fbin(fx[0])),
                         (h, r[2].weekday(), r[5], fbin(fx[0])),
                         (h, r[5], r[6], fbin(fx[0])), (h, r[5], fbin(fx[0])),
                         (h, r[2].weekday(), r[5], r[6]),
                         (h, r[2].weekday(), r[5]), (h, r[5], r[6]),
                         (h, r[2].weekday()), (h, r[5]), (h,)]
# multiplicative use of the date factor instead of a key
K_MULT = K_BASE


def cv(keyfn, mult=False, k=5, min_n=40):
    out = defaultdict(list)          # bucket -> errors
    for f in range(k):
        acc2 = None
        for r in TR:
            if assign.get(r[0], -1) == f: continue
            for h in range(7):
                ks = keyfn(r, h, r[7][h])
                if acc2 is None: acc2 = [defaultdict(list) for _ in ks]
                v = r[4][h] / r[3]
                if mult:
                    df = r[7][h][0]
                    v = v / df if df else v
                for i, kk in enumerate(ks):
                    acc2[i][kk].append(v)
        tabs = [{k2: med(v) for k2, v in a.items() if len(v) >= min_n} for a in acc2]
        glob = med([r[4][h] / r[3] for r in TR if assign.get(r[0], -1) != f
                    for h in range(7)])
        for r in EV:
            if assign.get(r[0], -1) != f: continue
            for h in range(7):
                p = glob
                for t2, kk in zip(tabs, keyfn(r, h, r[7][h])):
                    if kk in t2:
                        p = t2[kk]; break
                if mult:
                    df = r[7][h][0]
                    if df: p = p * df
                e = abs(r[4][h] / r[3] - p)
                out["ALL"].append(e)
                out["anomaly" if r[7][h][1] else "normal"].append(e)
    return out


print("=" * 80)
print("DOES AN OBSERVED DATE-DEMAND SIGNAL HELP ON CALENDAR ANOMALIES?")
print("=" * 80)
na = sum(1 for r in EV for h in range(7) if r[7][h][1])
nt = 7 * len(EV)
print(f"  eval rows {nt} | on a calendar anomaly: {na} ({100*na/nt:.1f}%)")
print("  (test-set equivalent: ~31.8% of rows in Ramadan / Idulfitri / Xmas)")
print()
runs = {"baseline                   ": cv(K_BASE),
        "date factor as a KEY       ": cv(K_DF),
        "date factor as a MULTIPLIER": cv(K_MULT, mult=True)}
print(f"  {'variant':28s} {'ALL':>9s} {'normal':>9s} {'anomaly':>9s}")
for lab, o in runs.items():
    f = lambda k: sum(o[k]) / len(o[k]) if o[k] else float("nan")
    print(f"  {lab:28s} {f('ALL'):9.4f} {f('normal'):9.4f} {f('anomaly'):9.4f}")

b = runs["baseline                   "]
print()
print(f"  {'variant':28s} {'delta ALL':>10s} {'delta normal':>13s} {'delta anomaly':>14s}")
for lab, o in runs.items():
    if lab.startswith("baseline"): continue
    f = lambda k: sum(o[k]) / len(o[k]) - sum(b[k]) / len(b[k])
    print(f"  {lab:28s} {f('ALL'):+10.4f} {f('normal'):+13.4f} {f('anomaly'):+14.4f}")

print()
print("=" * 80)
print("HOW BIG IS THE ANOMALY PENALTY AT ALL?")
print("=" * 80)
f = lambda k: sum(b[k]) / len(b[k])
print(f"  baseline MASE on normal target days : {f('normal'):.4f}")
print(f"  baseline MASE on anomaly target days: {f('anomaly'):.4f}")
print(f"  ratio                               : {f('anomaly')/f('normal'):.2f}x")
print()
print("  Train's anomalies are mild (single public holidays + the April boom).")
print("  Ramadan is a MONTH-LONG level shift with no train analogue at all,")
print("  so this is a lower bound on the test-set penalty.")
