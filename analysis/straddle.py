"""MASE is scale-relative, so a uniform demand level shift is absorbed by
`scale` (D1-D3 sits in the same regime as D4-D10). What is NOT absorbed is a
window that STRADDLES a regime boundary: D1-D3 in one regime, D4-D10 in
another. Quantify that exposure in the test set.
"""
import csv
import datetime as dt
from collections import Counter, defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

RAM_A, RAM_B = dt.date(2026, 2, 17), dt.date(2026, 3, 19)   # Ramadan 1447 H
EID = dt.date(2026, 3, 21)                                   # Idulfitri 1447 H
XMAS_A, XMAS_B = dt.date(2025, 12, 20), dt.date(2026, 1, 4)

hist, test = [], []
with open(D + "test_history.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        hist.append((dt.date.fromisoformat(r["date_show"]), r["cinema_ids"],
                     r["movie_title"], int(r["total_ticket"])))
with open(D + "test.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        test.append((dt.date.fromisoformat(r["date_show"]), r["movie_title"],
                     r["cinema_ids"]))

d1 = {}
for d, c, m, t in hist:
    if m not in d1 or d < d1[m]:
        d1[m] = d


def regime(d):
    if XMAS_A <= d <= XMAS_B:
        return "xmas/NY"
    if RAM_A <= d <= RAM_B:
        return "ramadan"
    if d >= EID - dt.timedelta(days=1):
        return "idulfitri"
    return "normal"


print("=" * 78)
print("PER-FILM WINDOW vs CALENDAR REGIME")
print("=" * 78)
rows_by = Counter()
films_by = Counter()
for m in sorted(d1):
    a = d1[m]
    obs = [a + dt.timedelta(days=i) for i in range(3)]
    tgt = [a + dt.timedelta(days=i) for i in range(3, 10)]
    ro = {regime(x) for x in obs}
    rt = {regime(x) for x in tgt}
    key = ("+".join(sorted(ro)), "+".join(sorted(rt)))
    films_by[key] += 1
print(f"  {'D1-D3 regime':22s} {'D4-D10 regime':26s} {'films':>6s}")
for k, v in films_by.most_common():
    flag = "  <-- STRADDLES" if k[0] != k[1] else ""
    print(f"  {k[0]:22s} {k[1]:26s} {v:6d}{flag}")

straddle = sum(v for k, v in films_by.items() if k[0] != k[1])
print(f"\n  films whose window straddles a regime boundary: {straddle}/{len(d1)} "
      f"({100*straddle/len(d1):.1f}%)")

# ------------------------------------------------- row-level exposure
print("\n" + "=" * 78)
print("ROW-LEVEL EXPOSURE IN test.csv")
print("=" * 78)
cnt = Counter()
for d, m, c in test:
    cnt[regime(d)] += 1
tot = len(test)
for k, v in cnt.most_common():
    print(f"  {k:12s} {v:7d} rows ({100*v/tot:5.1f}%)")

# straddling rows: target regime != observation regime
sr = 0
for d, m, c in test:
    a = d1[m]
    ro = regime(a + dt.timedelta(days=1))
    if regime(d) != ro:
        sr += 1
print(f"\n  rows whose target regime differs from their D1-D3 regime: "
      f"{sr} ({100*sr/tot:.1f}%)")
print("  -> only these rows need a regime CORRECTION; the rest are absorbed")
print("     by `scale`. This is the honest size of the calendar problem.")

# ------------------------------------- which films cross into Idulfitri
print("\n" + "=" * 78)
print("FILMS CROSSING INTO IDULFITRI (the largest uplift in the Indonesian year)")
print("=" * 78)
n = 0
for m in sorted(d1):
    a = d1[m]
    tgt = [a + dt.timedelta(days=i) for i in range(3, 10)]
    if any(regime(x) == "idulfitri" for x in tgt) and regime(a) != "idulfitri":
        n += 1
        days = sum(1 for x in tgt if regime(x) == "idulfitri")
        print(f"  {m[:46]:46s} D1={a} {DOW[a.weekday()]:3s} "
              f"idulfitri days in D4-D10: {days}")
print(f"  total: {n} films")

print("\n" + "=" * 78)
print("FILMS RELEASING JUST BEFORE RAMADAN STARTS (demand collapse mid-window)")
print("=" * 78)
n = 0
for m in sorted(d1):
    a = d1[m]
    tgt = [a + dt.timedelta(days=i) for i in range(3, 10)]
    if regime(a) == "normal" and any(regime(x) == "ramadan" for x in tgt):
        n += 1
        days = sum(1 for x in tgt if regime(x) == "ramadan")
        print(f"  {m[:46]:46s} D1={a} {DOW[a.weekday()]:3s} "
              f"ramadan days in D4-D10: {days}")
print(f"  total: {n} films")

print("\n" + "=" * 78)
print("TRAIN ANALOGUE FOR THE IDULFITRI SPIKE")
print("=" * 78)
nat = defaultdict(int)
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        nat[dt.date.fromisoformat(r["date_show"])] += int(r["total_ticket"])
print("  national daily tickets, first 3 weeks of train (post-Idulfitri 1446 boom):")
for d in sorted(nat)[:21]:
    print(f"    {d} {DOW[d.weekday()]:3s} {nat[d]:10,}")
base = sum(v for d, v in nat.items() if d >= dt.date(2025, 5, 15)) / \
    sum(1 for d in nat if d >= dt.date(2025, 5, 15))
peak = max(v for d, v in nat.items() if d <= dt.date(2025, 4, 10))
print(f"\n  post-Idulfitri peak / later baseline = {peak/base:.2f}x")
print("  -> train DOES contain the Idulfitri AFTERMATH (April 2025), labelled")
print("     'Idulfitri 1446 H' in holidays.csv, so holiday_name can bridge to")
print("     'Idulfitri 1447 H' in the test period. Ramadan itself is absent.")
