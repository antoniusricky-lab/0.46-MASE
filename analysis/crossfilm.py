"""The biggest miss: cross-film demand observation on the TARGET date.

The notebook (cell 10) states it deliberately builds every calendar/cluster
table from train.csv only, refusing test_history.csv "so no prediction-window
information leaks".

But test_history.csv gives D1-D3 for 163 films, and those windows are at
DIFFERENT calendar dates per film. So for film X's target date t, other films
Y often have t inside THEIR observation window. Reading demand on t from
those films is not leakage -- it is observed data the organisers handed over.

This script measures how much of the test set is covered that way.
"""
import csv
import datetime as dt
from collections import defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

hist, test = [], []
with open(D + "test_history.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        hist.append((dt.date.fromisoformat(r["date_show"]), r["cinema_ids"],
                     r["movie_title"], int(r["total_ticket"]), int(r["total_show"])))
with open(D + "test.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        test.append((dt.date.fromisoformat(r["date_show"]), r["movie_title"],
                     r["cinema_ids"]))

d1 = {}
for d, c, m, t, s in hist:
    if m not in d1 or d < d1[m]:
        d1[m] = d

# observed national / per-cluster demand per date, from ALL films' D1-D3
nat_obs = defaultdict(int)
nat_films = defaultdict(set)
cl_obs = defaultdict(int)
for d, c, m, t, s in hist:
    nat_obs[d] += t
    nat_films[d].add(m)
    cl_obs[(c, d)] += t

print("=" * 78)
print("COVERAGE OF TARGET DATES BY OTHER FILMS' OBSERVATION WINDOWS")
print("=" * 78)
cov_rows = 0
cov_rows_excl_self = 0
cl_cov = 0
for d, m, c in test:
    others = nat_films.get(d, set()) - {m}
    if nat_films.get(d):
        cov_rows += 1
    if others:
        cov_rows_excl_self += 1
    if (c, d) in cl_obs:
        cl_cov += 1
tot = len(test)
print(f"  test rows                                        : {tot:7d}")
print(f"  target date observed by >=1 film (any)           : {cov_rows:7d} "
      f"({100*cov_rows/tot:5.1f}%)")
print(f"  target date observed by >=1 OTHER film           : {cov_rows_excl_self:7d} "
      f"({100*cov_rows_excl_self/tot:5.1f}%)")
print(f"  target (cluster, date) directly observed         : {cl_cov:7d} "
      f"({100*cl_cov/tot:5.1f}%)")

print("\n  coverage by number of observing films:")
buck = defaultdict(int)
for d, m, c in test:
    n = len(nat_films.get(d, set()) - {m})
    buck[min(n, 10)] += 1
for k in sorted(buck):
    lab = f"{k}" if k < 10 else "10+"
    print(f"    {lab:>4s} other films observe the target date : {buck[k]:7d} "
          f"({100*buck[k]/tot:5.1f}%)")

print("\n  coverage by horizon:")
print(f"  {'D':>4s} {'rows':>7s} {'covered':>9s} {'%':>7s} {'cluster-level':>14s}")
for h in range(4, 11):
    sel = [(d, m, c) for d, m, c in test if (d - d1[m]).days == h - 1]
    cv = sum(1 for d, m, c in sel if nat_films.get(d, set()) - {m})
    cc = sum(1 for d, m, c in sel if (c, d) in cl_obs)
    print(f"  D{h:<3d} {len(sel):7d} {cv:9d} {100*cv/max(len(sel),1):6.1f}% "
          f"{100*cc/max(len(sel),1):13.1f}%")

print("\n  coverage by month (the Idulfitri tail is the uncovered part):")
bm = defaultdict(lambda: [0, 0])
for d, m, c in test:
    k = (d.year, d.month)
    bm[k][0] += 1
    if nat_films.get(d, set()) - {m}:
        bm[k][1] += 1
for k in sorted(bm):
    a, b = bm[k]
    print(f"    {k[0]}-{k[1]:02d}  {b:6d}/{a:6d} = {100*b/a:5.1f}%")

print("\n" + "=" * 78)
print("WHAT THE OBSERVED DATES LOOK LIKE (national tickets from test_history)")
print("=" * 78)
print("  This is a partial but real demand index for the TEST period --")
print("  something the notebook never builds.")
ds = sorted(nat_obs)
print(f"  observed dates: {len(ds)}  {ds[0]} .. {ds[-1]}")
print(f"  {'date':12s} {'dow':4s} {'films':>6s} {'tickets':>12s}")
for d in ds[:8]:
    print(f"  {d!s:12s} {DOW[d.weekday()]:4s} {len(nat_films[d]):6d} {nat_obs[d]:12,}")
print("  ...")
for d in ds[-14:]:
    print(f"  {d!s:12s} {DOW[d.weekday()]:4s} {len(nat_films[d]):6d} {nat_obs[d]:12,}")

print("\n" + "=" * 78)
print("CAVEAT: the index is composition-biased")
print("=" * 78)
print("  nat_obs[d] only sums films whose D1-D3 covers d, so it mixes 'how")
print("  strong was demand on d' with 'how many films happened to open near d'.")
print("  The usable form is a RATIO against the same films' own window mean,")
print("  i.e. a per-date multiplier estimated from films observing that date.")

# build a proper multiplicative date factor via per-film normalisation
num, den = defaultdict(float), defaultdict(float)
film_mean = {}
for m in d1:
    v = [nat_obs_f for nat_obs_f in []]
tot_f = defaultdict(lambda: defaultdict(int))
for d, c, m, t, s in hist:
    tot_f[m][d] += t
for m, dd in tot_f.items():
    mu = sum(dd.values()) / len(dd)
    if mu <= 0:
        continue
    for d, v in dd.items():
        num[d] += v / mu
        den[d] += 1
fac = {d: num[d] / den[d] for d in num if den[d] >= 3}
print(f"\n  per-date factor estimated from >=3 films: {len(fac)} dates")
print(f"  {'date':12s} {'dow':4s} {'n':>3s} {'factor':>8s}")
for d in sorted(fac)[-20:]:
    print(f"  {d!s:12s} {DOW[d.weekday()]:4s} {int(den[d]):3d} {fac[d]:8.3f}")
