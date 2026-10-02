"""Harden the two key claims:
  A) test.csv pair set == exactly the pairs with a transaction on D3
  B) D1 == the film's release date (test set is 100% "opening" regime)
"""
import csv
import datetime as dt
from collections import Counter, defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def rd(p):
    with open(D + p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


hist, test = rd("test_history.csv"), rd("test.csv")
for r in hist:
    r["d"] = dt.date.fromisoformat(r["date_show"]); r["t"] = int(r["total_ticket"])
for r in test:
    r["d"] = dt.date.fromisoformat(r["date_show"])

d1 = {}
for r in hist:
    m = r["movie_title"]
    if m not in d1 or r["d"] < d1[m]:
        d1[m] = r["d"]

# ============================================================== CLAIM A
print("=" * 78)
print("CLAIM A: test.csv pairs == pairs active on D3")
print("=" * 78)
act = defaultdict(set)                      # pair -> set of offsets present
for r in hist:
    o = (r["d"] - d1[r["movie_title"]]).days
    if 0 <= o <= 2:
        act[(r["movie_title"], r["cinema_ids"])].add(o)
tp = set((r["movie_title"], r["cinema_ids"]) for r in test)

active_d3 = {p for p, s in act.items() if 2 in s}
print(f"  pairs with history in D1-D3      : {len(act)}")
print(f"  pairs active on D3               : {len(active_d3)}")
print(f"  pairs in test.csv                : {len(tp)}")
print(f"  active-on-D3  == test.csv pairs  : {active_d3 == tp}")
print(f"    in test.csv but NOT active D3  : {len(tp - active_d3)}")
print(f"    active D3 but NOT in test.csv  : {len(active_d3 - tp)}")

print("\n  activity pattern (D1,D2,D3 present?) -> predicted / not:")
pat = defaultdict(lambda: [0, 0])
for p, s in act.items():
    key = "".join("1" if i in s else "0" for i in range(3))
    pat[key][0 if p in tp else 1] += 1
print(f"    {'D1D2D3':8s} {'predicted':>10s} {'excluded':>10s}")
for k in sorted(pat, reverse=True):
    print(f"    {k:8s} {pat[k][0]:10d} {pat[k][1]:10d}")

# ============================================================== CLAIM B
print()
print("=" * 78)
print("CLAIM B: D1 == release date")
print("=" * 78)
dates = Counter(d1.values())
weeks = defaultdict(set)
for d in dates:
    weeks[d.isocalendar()[:2]].add(d)
print(f"  distinct D1 dates : {len(dates)} for {len(d1)} films")
print(f"  distinct ISO weeks: {len(weeks)}")
print(f"  mean distinct D1 dates per week: {len(dates)/len(weeks):.2f}")
print("  (a random per-film window would give ~1 unique date per film,")
print("   spread over all 7 weekdays; weekly release slots give ~2-3/week)")

print("\n  films per D1 date, histogram:")
for n, c in sorted(Counter(dates.values()).items()):
    print(f"    {n} film(s) share a D1 date : {c} dates")

print("\n  every ISO week covered in the span?")
span = sorted(dates)
allw = set()
cur = min(span)
while cur <= max(span):
    allw.add(cur.isocalendar()[:2]); cur += dt.timedelta(days=7)
print(f"    weeks in span {len(allw)} | weeks with a release {len(weeks)} "
      f"| missing {len(allw - set(weeks))}")

# national demand on each film's D1 vs the weekday pattern
print("\n  per-film D1-D3 ticket shape, Wed-D1 films only (release-day effect):")
tot = defaultdict(lambda: defaultdict(int))
for r in hist:
    o = (r["d"] - d1[r["movie_title"]]).days
    if 0 <= o <= 2:
        tot[r["movie_title"]][o] += r["t"]
wed = [m for m in d1 if d1[m].weekday() == 2]
thu = [m for m in d1 if d1[m].weekday() == 3]
for lab, ms in (("Wed-D1", wed), ("Thu-D1", thu)):
    s = [sum(tot[m][i] for m in ms) for i in range(3)]
    print(f"    {lab} (n={len(ms):3d}): D1={s[0]:,} D2={s[1]:,} D3={s[2]:,}")

# ---- do films ever EXPAND to new clusters after D3?
print()
print("=" * 78)
print("EXPANSION CHECK: new clusters appearing in D4-D10")
print("=" * 78)
hp = set((r["movie_title"], r["cinema_ids"]) for r in hist)
new = [p for p in tp if p not in hp]
print(f"  test pairs with NO D1-D3 history at all: {len(new)}")
print("  -> 0 means the organisers never ask you to forecast a cluster the film")
print("     expanded into after D3. Training rows for such pairs are OUT OF")
print("     DISTRIBUTION and should be dropped.")

# ---- how concentrated is scale?
print()
print("=" * 78)
print("TEST SCALE DISTRIBUTION (drives every MASE denominator)")
print("=" * 78)
s3 = defaultdict(lambda: [0, 0, 0])
for r in hist:
    o = (r["d"] - d1[r["movie_title"]]).days
    if 0 <= o <= 2:
        s3[(r["movie_title"], r["cinema_ids"])][o] += r["t"]
sc = sorted(max(sum(s3[p]) / 3.0, 1.0) for p in tp)
n = len(sc)
print(f"  n={n}  min={sc[0]:.2f} p1={sc[n//100]:.2f} p5={sc[n//20]:.2f} "
      f"p25={sc[n//4]:.1f} p50={sc[n//2]:.1f} p75={sc[3*n//4]:.1f} max={sc[-1]:.0f}")
for thr in (1.5, 2, 3, 5, 10, 25):
    k = sum(1 for x in sc if x <= thr)
    print(f"  scale <= {thr:5}: {k:6d} pairs ({100*k/n:5.2f}% of pairs, "
          f"{100*k/n:5.2f}% of rows) -- each ticket of error costs "
          f"{1/thr:.2f} MASE/row")
