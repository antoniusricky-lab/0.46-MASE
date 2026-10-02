"""How did the organisers define D1?

Train "first appearance" is Friday-heavy, test D1 is Wed/Thu-heavy.
Either (a) D1 is not the release date, or (b) first-appearance in train.csv
is polluted by tiny sneak-preview days. Test (b).
"""
import csv
import datetime as dt
from collections import Counter, defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load(p):
    with open(D + p, newline="", encoding="utf-8") as f:
        return [(dt.date.fromisoformat(r["date_show"]), r["cinema_ids"],
                 r["movie_title"], int(r["total_ticket"])) for r in csv.DictReader(f)]


tr, hist = load("train.csv"), load("test_history.csv")
TMIN, TMAX = min(x[0] for x in tr), max(x[0] for x in tr)

nat = defaultdict(lambda: defaultdict(int))     # film -> date -> tickets
ncl = defaultdict(lambda: defaultdict(int))     # film -> date -> n clusters
for d, c, m, t in tr:
    nat[m][d] += t
    ncl[m][d] += 1

first = {m: min(v) for m, v in nat.items()}
left = {m for m in first if first[m] == TMIN}

print("=" * 78)
print("EARLY TRAJECTORY OF TRAIN FILMS AROUND FIRST APPEARANCE")
print("=" * 78)
print("  Is day 0 a tiny sneak preview relative to the following days?")
print(f"  {'dow of 1st':11s} {'n':>4s} {'tix_d0':>9s} {'tix_d1':>9s} {'tix_d2':>9s}"
      f" {'ncl_d0':>7s} {'ncl_d1':>7s}")
by = defaultdict(list)
for m in first:
    if m in left:
        continue
    f0 = first[m]
    if f0 + dt.timedelta(days=3) > TMAX:
        continue
    by[DOW[f0.weekday()]].append(m)
for k in DOW:
    ms = by.get(k)
    if not ms:
        continue
    t = [sum(nat[m].get(first[m] + dt.timedelta(days=i), 0) for m in ms) / len(ms)
         for i in range(3)]
    n = [sum(ncl[m].get(first[m] + dt.timedelta(days=i), 0) for m in ms) / len(ms)
         for i in range(2)]
    print(f"  {k:11s} {len(ms):4d} {t[0]:9.0f} {t[1]:9.0f} {t[2]:9.0f}"
          f" {n[0]:7.1f} {n[1]:7.1f}")

# ---------------------------------------------- "real" release = first big day
print()
print("=" * 78)
print("DEFINE D1 = first day reaching >=25% of the film's first-week peak clusters")
print("=" * 78)
real = {}
for m in first:
    if m in left:
        continue
    ds = sorted(nat[m])
    wk = [d for d in ds if d <= first[m] + dt.timedelta(days=9)]
    peak = max(ncl[m][d] for d in wk) if wk else 0
    for d in wk:
        if ncl[m][d] >= 0.25 * peak:
            real[m] = d
            break
c1 = Counter(DOW[first[m].weekday()] for m in real)
c2 = Counter(DOW[real[m].weekday()] for m in real)
shift = Counter((real[m] - first[m]).days for m in real)
print(f"  {'dow':6s} {'first-appearance':>18s} {'>=25% clusters':>16s}")
for k in DOW:
    print(f"  {k:6s} {c1[k]:18d} {c2[k]:16d}")
print(f"  shift in days (real - first): {dict(sorted(shift.items()))}")

# ------------------------------------------------- test side, same statistic
print()
print("=" * 78)
print("TEST SIDE: is D1 already the 'big' day, or is D1 small?")
print("=" * 78)
hn = defaultdict(lambda: defaultdict(int))
hc = defaultdict(lambda: defaultdict(int))
for d, c, m, t in hist:
    hn[m][d] += t
    hc[m][d] += 1
hd1 = {m: min(v) for m, v in hn.items()}
print(f"  {'D1 dow':8s} {'n':>4s} {'ncl_D1':>8s} {'ncl_D2':>8s} {'ncl_D3':>8s}"
      f" {'tix_D1':>9s} {'tix_D2':>9s} {'tix_D3':>9s}")
g = defaultdict(list)
for m in hd1:
    g[DOW[hd1[m].weekday()]].append(m)
for k in DOW:
    ms = g.get(k)
    if not ms:
        continue
    n = [sum(hc[m].get(hd1[m] + dt.timedelta(days=i), 0) for m in ms) / len(ms) for i in range(3)]
    t = [sum(hn[m].get(hd1[m] + dt.timedelta(days=i), 0) for m in ms) / len(ms) for i in range(3)]
    print(f"  {k:8s} {len(ms):4d} {n[0]:8.1f} {n[1]:8.1f} {n[2]:8.1f}"
          f" {t[0]:9.0f} {t[1]:9.0f} {t[2]:9.0f}")

print()
print("  D1 dates equal to the very first day of the test period (2025-10-01):",
      sum(1 for m in hd1 if hd1[m] == dt.date(2025, 10, 1)))
print("  -> a spike here would mean running films were clipped at the period start.")

# --------------------------------- how many test films look like true openings
print()
print("=" * 78)
print("DO TEST FILMS LOOK LIKE OPENINGS?  (cluster count vs its own later peak)")
print("=" * 78)
print("  For a true opening, D1 cluster count is at/near the film's maximum,")
print("  because a wide release opens everywhere at once.")
import statistics

ratios = []
for m in hd1:
    mx = max(hc[m].values())
    ratios.append(hc[m][hd1[m]] / mx)
ratios.sort()
print(f"  n={len(ratios)} ncl(D1)/max(ncl over D1-D3): "
      f"p10={ratios[len(ratios)//10]:.2f} p50={statistics.median(ratios):.2f} "
      f"mean={sum(ratios)/len(ratios):.2f}")
print(f"  films where D1 has the max cluster count: "
      f"{sum(1 for r in ratios if r >= 0.999)}/{len(ratios)}")
