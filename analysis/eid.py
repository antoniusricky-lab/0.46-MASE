"""The Idulfitri-crossing films: the largest single concentrated error source.

These films observe D1-D3 just before Eid and must be forecast across Eid and
the week after -- the biggest moviegoing week of the Indonesian year. The
notebook's submission profile (mean ratio 0.952 at D4 falling to 0.132 at D10,
71.7% zeros at D10) is pointed the wrong way for them.
"""
import csv
import datetime as dt
from collections import defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

test, hist = [], []
with open(D + "test.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        test.append((int(r["id"]), dt.date.fromisoformat(r["date_show"]),
                     r["movie_title"], r["cinema_ids"]))
with open(D + "test_history.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        hist.append((dt.date.fromisoformat(r["date_show"]), r["cinema_ids"],
                     r["movie_title"], int(r["total_ticket"])))
d1 = {}
for d, c, m, t in hist:
    if m not in d1 or d < d1[m]:
        d1[m] = d

EID = dt.date(2026, 3, 21)
late = sorted({d1[m] for m in d1 if d1[m] >= dt.date(2026, 3, 1)})
print("=" * 80)
print("TEST FILMS RELEASING IN MARCH 2026 (the Lebaran corridor)")
print("=" * 80)
for a in late:
    ms = [m for m in d1 if d1[m] == a]
    rows = sum(1 for i, d, m, c in test if m in ms)
    tgt = [a + dt.timedelta(days=i) for i in range(3, 10)]
    neid = sum(1 for x in tgt if x >= EID)
    print(f"  D1={a} {DOW[a.weekday()]:3s}  films={len(ms):2d}  test_rows={rows:5d}"
          f"  target days on/after Eid: {neid}")
    for m in sorted(ms):
        print(f"      {m}")

MAR18 = dt.date(2026, 3, 18)
ms18 = [m for m in d1 if d1[m] == MAR18]
rows18 = [(i, d, m, c) for i, d, m, c in test if m in ms18]
print()
print("=" * 80)
print("THE 2026-03-18 COHORT: ENTIRE D4-D10 WINDOW IS THE EID HOLIDAY WEEK")
print("=" * 80)
print(f"  films      : {len(ms18)}")
print(f"  test rows  : {len(rows18)}  = {100*len(rows18)/len(test):.2f}% of the test set")
print(f"  pairs      : {len({(m,c) for i,d,m,c in rows18})}")
print("  D1-D3      : 2026-03-18..20 (pre-Eid, end of Ramadan)")
print("  D4-D10     : 2026-03-21..27 (Idulfitri 1447 H + the week after)")

# their observed D1-D3 level
tot = defaultdict(int)
for d, c, m, t in hist:
    if m in ms18:
        tot[d] += t
print("\n  their observed national tickets:")
for d in sorted(tot):
    print(f"    {d} {DOW[d.weekday()]:3s} {tot[d]:10,}")

# ---------------- train analogue: the post-Idulfitri 1446 boom (April 2025)
nat = defaultdict(int)
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        nat[dt.date.fromisoformat(r["date_show"])] += int(r["total_ticket"])

print()
print("=" * 80)
print("TRAIN ANALOGUE: Idulfitri 1446 H fell 2025-03-31 (day before train)")
print("=" * 80)
print("  Mapping Eid-relative day onto the 2026 window:")
print(f"  {'Eid+k':>7s} {'2025 date':12s} {'dow':4s} {'national':>10s} "
      f"{'vs Eid+1..3':>12s}  {'2026 date':12s}")
base = sum(nat[dt.date(2025, 4, 1) + dt.timedelta(days=k)] for k in range(3)) / 3
for k in range(10):
    d25 = dt.date(2025, 4, 1) + dt.timedelta(days=k)     # Eid+1 == 2025-04-01
    d26 = dt.date(2026, 3, 22) + dt.timedelta(days=k)    # Eid+1 == 2026-03-22
    if d25 not in nat:
        continue
    print(f"  Eid+{k+1:<3d} {d25!s:12s} {DOW[d25.weekday()]:4s} "
          f"{nat[d25]:10,} {nat[d25]/base:12.2f}  {d26!s:12s}")

print()
print("  The 2025 pattern: the week AFTER Eid runs 1.3-1.5x the Eid weekend")
print("  itself, and ~3x the pre-Eid baseline. So for the 2026-03-18 cohort the")
print("  true y/scale across D4-D10 should sit ABOVE 1, not decay toward 0.")
print()
print("  Notebook's submitted profile (cell 17 output):")
prof = {4: 0.952, 5: 0.686, 6: 0.440, 7: 0.344, 8: 0.183, 9: 0.142, 10: 0.132}
zer = {4: .106, 5: .173, 6: .254, 7: .380, 8: .588, 9: .689, 10: .717}
print(f"  {'D':>4s} {'mean ratio':>11s} {'share zero':>11s}")
for h in range(4, 11):
    print(f"  D{h:<3d} {prof[h]:11.3f} {zer[h]:11.1%}")
avg = sum(prof.values()) / 7
print(f"\n  mean predicted ratio across D4-D10 = {avg:.3f}")
for truth in (1.2, 1.5, 2.0):
    err = abs(truth - avg)
    print(f"  if the true mean ratio is {truth:.1f} -> per-row error ~{err:.2f} MASE, "
          f"cohort cost ~{err*len(rows18)/len(test):.4f} MASE overall")
