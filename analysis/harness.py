"""Corrected replication harness + measured gains, pure stdlib.

Three structural corrections vs the 0.46641 notebook:
  C1  D1 = WIDE-RELEASE day (first day reaching >=25% of first-week peak
      cluster count), not the film's first-ever transaction (sneak previews
      shift that by 1-9 days for 44% of films).
  C2  keep only pairs with a transaction on D3  (the exact, proven test rule).
  C3  one window per film at lag 0 only, D1-dow restricted to Wed/Thu/Fri
      (96% of the test set), instead of 33 anchors per film at lag 0..47.

Evaluation: leave-one-film-out style GroupKFold on films, MASE on ratios
(identical to the official metric, since MASE == MAE(y/scale)).
"""
import csv
import datetime as dt
from collections import defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TEST_DOW = {2, 3, 4}
SEED = 2026


def load(p):
    with open(D + p, newline="", encoding="utf-8") as f:
        return [(dt.date.fromisoformat(r["date_show"]), r["cinema_ids"],
                 r["movie_title"], int(r["total_ticket"]),
                 float(r["occupation_rate"]), int(r["total_show"]))
                for r in csv.DictReader(f)]


tr = load("train.csv")
TMIN, TMAX = min(x[0] for x in tr), max(x[0] for x in tr)

cell, ncl, nat = {}, defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
mdc = defaultdict(list)
for d, c, m, t, o, s in tr:
    cell[(m, c, d)] = (t, o, s)
    ncl[m][d] += 1
    nat[m][d] += t
    mdc[(m, d)].append(c)

first = {m: min(v) for m, v in nat.items()}
LEFT = {m for m in first if first[m] == TMIN}

# ------------------------------------------------- C1: wide-release date
release = {}
for m in first:
    if m in LEFT:
        continue
    wk = [d for d in sorted(nat[m]) if d <= first[m] + dt.timedelta(days=9)]
    peak = max(ncl[m][d] for d in wk)
    for d in wk:
        if ncl[m][d] >= 0.25 * peak:
            release[m] = d
            break

# ------------------------------------------------------- holidays calendar
hol = {}
with open(D + "holidays.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        hol[dt.date.fromisoformat(r["date"])] = (r["day_tipe"], r["holiday_tipe"])


def nonwork(d):
    dy, ht = hol.get(d, ("weekday", "normal"))
    return 1 if (ht == "holiday" or dy == "weekend") else 0


def build(use_release=True, require_d3=True, dow_filter=None, lags=(0,)):
    """Return list of pair-windows: (film, cluster, d1, s[3], y[7])."""
    anchor = release if use_release else {m: first[m] for m in first if m not in LEFT}
    out = []
    for m in sorted(anchor):
        for lag in lags:
            d1 = anchor[m] + dt.timedelta(days=lag)
            if d1 + dt.timedelta(days=9) > TMAX:
                continue
            if dow_filter and d1.weekday() not in dow_filter:
                continue
            obs = [d1 + dt.timedelta(days=i) for i in range(3)]
            tgt = [d1 + dt.timedelta(days=i) for i in range(3, 10)]
            cs = set()
            for dd in obs:
                cs.update(mdc.get((m, dd), ()))
            for c in cs:
                s = [cell.get((m, c, dd), (0, 0, 0))[0] for dd in obs]
                if sum(s) == 0 or (require_d3 and s[2] == 0):
                    continue
                y = [cell.get((m, c, dd), (0, 0, 0))[0] for dd in tgt]
                out.append((m, c, d1, s, y))
    return out


def folds(rows, k=5):
    films = sorted({r[0] for r in rows})
    # deterministic shuffle
    st = SEED
    films = list(films)
    for i in range(len(films) - 1, 0, -1):
        st = (1103515245 * st + 12345) % (1 << 31)
        j = st % (i + 1)
        films[i], films[j] = films[j], films[i]
    assign = {m: i % k for i, m in enumerate(films)}
    return [[r for r in rows if assign[r[0]] != f] for f in range(k)], \
           [[r for r in rows if assign[r[0]] == f] for f in range(k)]


def med(v):
    v = sorted(v)
    n = len(v)
    return 0.0 if n == 0 else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def pat(s):
    return "".join("1" if x > 0 else "0" for x in s)


# =============================================== median-by-cell L1 model
# MASE == MAE on ratios, so the optimal constant per cell is the MEDIAN.
def fit(rows, keyfns, min_n=40):
    """Hierarchical median of y/scale over a list of key functions
    (most specific first); falls back when a cell is too small."""
    tabs = []
    for kf in keyfns:
        acc = defaultdict(list)
        for (m, c, d1, s, y) in rows:
            sc = max(sum(s) / 3.0, 1.0)
            for h in range(7):
                acc[kf(d1, s, sc, h)].append(y[h] / sc)
        tabs.append({k: med(v) for k, v in acc.items() if len(v) >= min_n})
    glob = med([y[h] / max(sum(s) / 3.0, 1.0)
                for (m, c, d1, s, y) in rows for h in range(7)])
    return tabs, glob


def predict(model, keyfns, d1, s, sc, h):
    tabs, glob = model
    for t, kf in zip(tabs, keyfns):
        k = kf(d1, s, sc, h)
        if k in t:
            return t[k]
    return glob


def scbin(sc):
    for i, b in enumerate((5, 10, 25, 50, 100, 250, 600)):
        if sc <= b:
            return i
    return 7


# key hierarchies to compare
K_H = [lambda d1, s, sc, h: (h,)]
K_HDOW = [lambda d1, s, sc, h: (h, d1.weekday()), lambda d1, s, sc, h: (h,)]
K_FULL = [
    lambda d1, s, sc, h: (h, d1.weekday(), pat(s), scbin(sc)),
    lambda d1, s, sc, h: (h, d1.weekday(), pat(s)),
    lambda d1, s, sc, h: (h, pat(s), scbin(sc)),
    lambda d1, s, sc, h: (h, d1.weekday()),
    lambda d1, s, sc, h: (h, pat(s)),
    lambda d1, s, sc, h: (h,),
]
K_NW = [   # non-working-day flag of the TARGET date instead of raw dow
    lambda d1, s, sc, h: (h, nonwork(d1 + dt.timedelta(days=3 + h)), pat(s), scbin(sc)),
    lambda d1, s, sc, h: (h, nonwork(d1 + dt.timedelta(days=3 + h)), pat(s)),
    lambda d1, s, sc, h: (h, nonwork(d1 + dt.timedelta(days=3 + h))),
    lambda d1, s, sc, h: (h,),
]


def cv(rows, keyfns, k=5):
    TRS, VAS = folds(rows, k)
    tot = n = 0.0
    for trs, vas in zip(TRS, VAS):
        mdl = fit(trs, keyfns)
        for (m, c, d1, s, y) in vas:
            sc = max(sum(s) / 3.0, 1.0)
            for h in range(7):
                tot += abs(y[h] / sc - predict(mdl, keyfns, d1, s, sc, h))
                n += 1
    return tot / n, int(n)


def naive(rows):
    tot = n = 0.0
    for (m, c, d1, s, y) in rows:
        sc = max(sum(s) / 3.0, 1.0)
        for h in range(7):
            tot += abs(y[h] / sc - 1.0)
            n += 1
    return tot / n


print("=" * 78)
print("CONSTRUCTION COMPARISON  (same evaluation population = corrected test-like)")
print("=" * 78)
EVAL = build(True, True, TEST_DOW, (0,))
print(f"  evaluation population: {len(EVAL)} pairs from "
      f"{len({r[0] for r in EVAL})} films, {7*len(EVAL)} rows")
print(f"  B0 naive (pred = scale)          MASE = {naive(EVAL):.4f}")
print()
print(f"  {'model (5-fold CV, grouped by film)':48s} {'MASE':>8s}")
for lab, kf in (("median by horizon", K_H),
                ("median by horizon x D1-dow", K_HDOW),
                ("median by horizon x nonworkday x pattern x scale", K_NW),
                ("median by horizon x D1-dow x pattern x scale", K_FULL)):
    v, n = cv(EVAL, kf)
    print(f"  {lab:48s} {v:8.4f}")

print()
print("=" * 78)
print("WHAT EACH CORRECTION IS WORTH")
print("=" * 78)
print("  Train on different populations, always evaluate on the corrected one.")
TRS, VAS = folds(EVAL, 5)
variants = {
    "C1+C2+C3  release D1, s3>0, lag 0 (correct)": build(True, True, TEST_DOW, (0,)),
    "  without C3 (lags 0..24 like notebook)    ": build(True, True, TEST_DOW, tuple(range(25))),
    "  without C2 (no s3>0 filter)              ": build(True, False, TEST_DOW, (0,)),
    "  without C1 (first-appearance D1)         ": build(False, True, TEST_DOW, (0,)),
    "  notebook-like (first-appear, no filter,  ": build(False, False, None, tuple(range(25))),
}
print(f"  {'training population':46s} {'pairs':>8s} {'MASE on correct eval':>22s}")
for lab, pop in variants.items():
    tot = n = 0.0
    for trs, vas in zip(TRS, VAS):
        vf = {r[0] for r in vas}
        sub = [r for r in pop if r[0] not in vf]          # no film leakage
        mdl = fit(sub, K_FULL)
        for (m, c, d1, s, y) in vas:
            sc = max(sum(s) / 3.0, 1.0)
            for h in range(7):
                tot += abs(y[h] / sc - predict(mdl, K_FULL, d1, s, sc, h))
                n += 1
    print(f"  {lab:46s} {len(pop):8d} {tot/n:22.4f}")

print()
print("=" * 78)
print("PER-HORIZON AND PER-PATTERN BREAKDOWN (best model)")
print("=" * 78)
TRS, VAS = folds(EVAL, 5)
eh, ep = defaultdict(list), defaultdict(list)
for trs, vas in zip(TRS, VAS):
    mdl = fit(trs, K_FULL)
    for (m, c, d1, s, y) in vas:
        sc = max(sum(s) / 3.0, 1.0)
        for h in range(7):
            e = abs(y[h] / sc - predict(mdl, K_FULL, d1, s, sc, h))
            eh[h].append(e)
            ep[pat(s)].append(e)
print(f"  {'D':>4s} {'n':>7s} {'MASE':>8s}")
for h in range(7):
    print(f"  D{h+4:<3d} {len(eh[h]):7d} {sum(eh[h])/len(eh[h]):8.4f}")
print(f"\n  {'pattern':>8s} {'n':>7s} {'MASE':>8s}")
for p in sorted(ep, reverse=True):
    print(f"  {p:>8s} {len(ep[p]):7d} {sum(ep[p])/len(ep[p]):8.4f}")

print()
print("=" * 78)
print("LEARNED RATIO TABLE  (median y/scale by D1-dow x horizon)")
print("=" * 78)
tabs, _glob = fit(EVAL, K_HDOW)
print(f"  {'D1 dow':8s}" + "".join(f"{'D' + str(h + 4):>8s}" for h in range(7)))
for w in sorted(TEST_DOW):
    row = "".join(f"{tabs[0].get((h, w), float('nan')):8.3f}" for h in range(7))
    print(f"  {DOW[w]:8s}{row}")
print("\n  The notebook's DECAY_PRIOR collapsed this to ONE curve over `off`")
print("  (0.689 0.542 0.403 0.272 0.109 0.000 0.000) averaged over a train mix")
print("  whose D1-dow distribution does not match the test set.")
