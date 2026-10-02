"""Decisive experiment: is the cross-film target-date signal worth real MASE?

Simulated with the SAME availability structure as the test set:
for a target date t, only films m' with release(m') <= t <= release(m')+2
are treated as observable (that is exactly what test_history.csv gives).
The film being predicted is always excluded from its own features.
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
        cell[(m, c, d)] = t
        ncl[m][d] += 1
        nat[m][d] += t
        mdc[(m, d)].append(c)
TMIN = min(d for m in nat for d in nat[m])
TMAX = max(d for m in nat for d in nat[m])
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

# ---------------- which films are "observable" on each date (test-like rule)
obs_films = defaultdict(set)
for m, r0 in release.items():
    for i in range(3):
        obs_films[r0 + dt.timedelta(days=i)].add(m)

print("=" * 80)
print("TRAIN-SIDE COVERAGE UNDER THE TEST AVAILABILITY RULE")
print("=" * 80)
EVALPOP = []
for m in sorted(release):
    d1 = release[m]
    if d1 + dt.timedelta(days=9) > TMAX or d1.weekday() not in TEST_DOW:
        continue
    obs = [d1 + dt.timedelta(days=i) for i in range(3)]
    tgt = [d1 + dt.timedelta(days=i) for i in range(3, 10)]
    cs = set()
    for dd in obs:
        cs.update(mdc.get((m, dd), ()))
    for c in cs:
        s = [cell.get((m, c, dd), 0) for dd in obs]
        if sum(s) == 0 or s[2] == 0:
            continue
        EVALPOP.append((m, c, d1, s, [cell.get((m, c, dd), 0) for dd in tgt]))
nrow = 7 * len(EVALPOP)
cov = sum(1 for (m, c, d1, s, y) in EVALPOP for h in range(7)
          if obs_films.get(d1 + dt.timedelta(days=3 + h), set()) - {m})
print(f"  eval pairs {len(EVALPOP)}  rows {nrow}")
print(f"  rows with target date observed by another film: {cov} "
      f"({100*cov/nrow:.1f}%)   [test: 66.2%]")

# --------------------------------- national per-date factor (median polish)
def med(v):
    v = sorted(v); n = len(v)
    return 0.0 if n == 0 else (v[n//2] if n % 2 else 0.5*(v[n//2-1]+v[n//2]))


def build_date_factor(exclude_film=None):
    """mean over observable films of tickets(t)/film's own 3-day mean."""
    acc = defaultdict(list)
    for m, r0 in release.items():
        if m == exclude_film:
            continue
        ds = [r0 + dt.timedelta(days=i) for i in range(3)]
        v = [nat[m].get(d, 0) for d in ds]
        mu = sum(v) / 3.0
        if mu <= 0:
            continue
        for d, x in zip(ds, v):
            acc[d].append(x / mu)
    return {d: med(v) for d, v in acc.items() if len(v) >= 3}


# cluster-level observed demand per (cluster, date) from observable films
cl_day = defaultdict(lambda: defaultdict(int))     # (c,d) -> film -> tickets
for m, r0 in release.items():
    for i in range(3):
        d = r0 + dt.timedelta(days=i)
        for c in mdc.get((m, d), ()):
            cl_day[(c, d)][m] = cell[(m, c, d)]


def scbin(x):
    for i, b in enumerate((5, 10, 25, 50, 100, 250, 600)):
        if x <= b:
            return i
    return 7


def fbin(x):
    if x is None:
        return -1
    for i, b in enumerate((0.55, 0.75, 0.95, 1.15, 1.45)):
        if x <= b:
            return i
    return 5


DF_ALL = build_date_factor()


def feats(m, c, d1, s, h):
    """Cross-film features for one (pair, horizon), excluding film m."""
    t = d1 + dt.timedelta(days=3 + h)
    others = obs_films.get(t, set()) - {m}
    # national date factor
    df = DF_ALL.get(t)
    # cluster observed on target date by another film?
    cd = cl_day.get((c, t), {})
    cd_o = {k: v for k, v in cd.items() if k != m}
    cl_seen = 1 if cd_o else 0
    # cluster demand on target date relative to during D1-D3
    rel = None
    if cd_o:
        base = []
        for i in range(3):
            dd = d1 + dt.timedelta(days=i)
            b = {k: v for k, v in cl_day.get((c, dd), {}).items() if k != m}
            if b:
                base.append(sum(b.values()))
        if base:
            mu = sum(base) / len(base)
            if mu > 0:
                rel = sum(cd_o.values()) / mu
    return df, cl_seen, rel, len(others)


# --------------------------------------------------------- model + CV
def folds(rows, k=5):
    films = sorted({r[0] for r in rows})
    st = SEED
    for i in range(len(films) - 1, 0, -1):
        st = (1103515245 * st + 12345) % (1 << 31)
        j = st % (i + 1)
        films[i], films[j] = films[j], films[i]
    return {m: i % k for i, m in enumerate(films)}


def pop_lags(lags):
    out = []
    for m in sorted(release):
        for lag in lags:
            d1 = release[m] + dt.timedelta(days=lag)
            if d1 + dt.timedelta(days=9) > TMAX:
                continue
            obs = [d1 + dt.timedelta(days=i) for i in range(3)]
            tgt = [d1 + dt.timedelta(days=i) for i in range(3, 10)]
            cs = set()
            for dd in obs:
                cs.update(mdc.get((m, dd), ()))
            for c in cs:
                s = [cell.get((m, c, dd), 0) for dd in obs]
                if sum(s) == 0 or s[2] == 0:
                    continue
                out.append((m, c, d1, s, [cell.get((m, c, dd), 0) for dd in tgt]))
    return out


TRAINPOP = pop_lags(tuple(range(25)))
print(f"  training population (lags 0-24): {len(TRAINPOP)} pairs")

KEYSETS = {
    "baseline  (h, D1dow, pattern, scalebin)": lambda r, h, fx: [
        (h, r[2].weekday(), r[5], r[6]), (h, r[2].weekday(), r[5]),
        (h, r[5], r[6]), (h, r[2].weekday()), (h, r[5]), (h,)],
    "+ national date factor                 ": lambda r, h, fx: [
        (h, r[2].weekday(), r[5], r[6], fbin(fx[0])),
        (h, r[2].weekday(), r[5], fbin(fx[0])),
        (h, r[5], r[6], fbin(fx[0])), (h, r[5], fbin(fx[0])),
        (h, r[2].weekday(), r[5], r[6]), (h, r[2].weekday(), r[5]),
        (h, r[5], r[6]), (h, r[2].weekday()), (h, r[5]), (h,)],
    "+ cluster-seen flag                    ": lambda r, h, fx: [
        (h, r[2].weekday(), r[5], r[6], fx[1]),
        (h, r[2].weekday(), r[5], fx[1]), (h, r[5], r[6], fx[1]),
        (h, r[5], fx[1]), (h, r[2].weekday(), r[5], r[6]),
        (h, r[2].weekday(), r[5]), (h, r[5], r[6]), (h, r[5]), (h,)],
    "+ cluster relative demand              ": lambda r, h, fx: [
        (h, r[2].weekday(), r[5], r[6], fbin(fx[2])),
        (h, r[2].weekday(), r[5], fbin(fx[2])),
        (h, r[5], r[6], fbin(fx[2])), (h, r[5], fbin(fx[2])),
        (h, r[2].weekday(), r[5], r[6]), (h, r[2].weekday(), r[5]),
        (h, r[5], r[6]), (h, r[2].weekday()), (h, r[5]), (h,)],
    "+ date factor AND cluster rel          ": lambda r, h, fx: [
        (h, r[2].weekday(), r[5], r[6], fbin(fx[0]), fbin(fx[2])),
        (h, r[2].weekday(), r[5], fbin(fx[0]), fbin(fx[2])),
        (h, r[5], r[6], fbin(fx[0]), fbin(fx[2])),
        (h, r[5], fbin(fx[0]), fbin(fx[2])),
        (h, r[5], fbin(fx[2])), (h, r[5], fbin(fx[0])),
        (h, r[2].weekday(), r[5], r[6]), (h, r[2].weekday(), r[5]),
        (h, r[5], r[6]), (h, r[5]), (h,)],
}


def prep(rows):
    out = []
    for (m, c, d1, s, y) in rows:
        p = "".join("1" if x > 0 else "0" for x in s)
        sc = max(sum(s) / 3.0, 1.0)
        fx = [feats(m, c, d1, s, h) for h in range(7)]
        out.append((m, c, d1, sc, y, p, scbin(sc), fx))
    return out


print("  materialising features ...")
TR = prep(TRAINPOP)
EV = prep(EVALPOP)
assign = folds(EV)
print("  done.")


def cv(keyfn, k=5, min_n=40):
    tot = n = 0.0
    per_h = defaultdict(list)
    for f in range(k):
        tabs = None
        acc = None
        for r in TR:
            if assign.get(r[0], -1) == f:
                continue
            for h in range(7):
                ks = keyfn(r, h, r[7][h])
                if acc is None:
                    acc = [defaultdict(list) for _ in ks]
                v = r[4][h] / r[3]
                for i, kk in enumerate(ks):
                    acc[i][kk].append(v)
        tabs = [{k2: med(v) for k2, v in a.items() if len(v) >= min_n} for a in acc]
        glob = med([r[4][h] / r[3] for r in TR if assign.get(r[0], -1) != f
                    for h in range(7)])
        for r in EV:
            if assign.get(r[0], -1) != f:
                continue
            for h in range(7):
                p = glob
                for t2, kk in zip(tabs, keyfn(r, h, r[7][h])):
                    if kk in t2:
                        p = t2[kk]
                        break
                e = abs(r[4][h] / r[3] - p)
                tot += e; n += 1
                per_h[h].append(e)
    return tot / n, per_h


print()
print("=" * 80)
print("MASE ON THE TEST-PROXY POPULATION (release windows, Wed/Thu/Fri, s3>0)")
print("=" * 80)
res = {}
for lab, kf in KEYSETS.items():
    v, ph = cv(kf)
    res[lab] = (v, ph)
    print(f"  {lab}  MASE = {v:.4f}")

base = res["baseline  (h, D1dow, pattern, scalebin)"][0]
print()
print(f"  {'variant':42s} {'MASE':>8s} {'delta':>9s} {'rel':>8s}")
for lab, (v, ph) in res.items():
    print(f"  {lab:42s} {v:8.4f} {v-base:+9.4f} {100*(v-base)/base:+7.2f}%")

best = min(res, key=lambda k: res[k][0])
print()
print("=" * 80)
print(f"PER-HORIZON: baseline vs best ({best.strip()})")
print("=" * 80)
pb, pz = res["baseline  (h, D1dow, pattern, scalebin)"][1], res[best][1]
print(f"  {'D':>4s} {'baseline':>9s} {'best':>9s} {'delta':>9s}")
for h in range(7):
    a = sum(pb[h]) / len(pb[h]); b = sum(pz[h]) / len(pz[h])
    print(f"  D{h+4:<3d} {a:9.4f} {b:9.4f} {b-a:+9.4f}")
