"""Calendar regime overlap between train (Apr-Sep 2025) and test (Oct 2025-Mar 2026).

Indonesian cinema demand is dominated by Ramadan (collapse) and Idulfitri
(explosion), plus Christmas/New Year. If the test window covers regimes that
train barely contains, generic dow/holiday features cannot learn them.
"""
import csv
import datetime as dt
from collections import Counter, defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

hol = []
with open(D + "holidays.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        hol.append((dt.date.fromisoformat(r["date"]), r["day_tipe"],
                    r["holiday_tipe"], r["holiday_name"]))
hol.sort()
print(f"holidays.csv: {len(hol)} days  {hol[0][0]} .. {hol[-1][0]}")

print("\n" + "=" * 78)
print("NAMED HOLIDAYS BY PERIOD")
print("=" * 78)
TRAIN_END = dt.date(2025, 9, 30)
for lab, sel in (("TRAIN  (Apr 1 - Sep 30 2025)", lambda d: d <= TRAIN_END),
                 ("TEST   (Oct 1 2025 - Mar 27 2026)", lambda d: d > TRAIN_END)):
    print(f"\n  {lab}")
    named = [(d, n) for d, dy, ht, n in hol if ht == "holiday" and sel(d)]
    print(f"    holiday days: {len(named)}")
    for d, n in named:
        print(f"      {d} {DOW[d.weekday()]:3s}  {n}")

# ---------------------------------------------------- test row exposure
test = []
with open(D + "test.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        test.append((dt.date.fromisoformat(r["date_show"]), r["movie_title"]))

print("\n" + "=" * 78)
print("TEST ROWS BY MONTH")
print("=" * 78)
c = Counter((d.year, d.month) for d, m in test)
tot = len(test)
for k in sorted(c):
    print(f"  {k[0]}-{k[1]:02d}  {c[k]:7d} rows ({100*c[k]/tot:5.1f}%)")

# Ramadan 1447 H ~ 2026-02-17 .. 2026-03-19 ; Idulfitri 1447 H ~ 2026-03-20
RAM_A, RAM_B = dt.date(2026, 2, 17), dt.date(2026, 3, 19)
nr = sum(1 for d, m in test if RAM_A <= d <= RAM_B)
ni = sum(1 for d, m in test if d > RAM_B)
nx = sum(1 for d, m in test if dt.date(2025, 12, 20) <= d <= dt.date(2026, 1, 4))
print("\n" + "=" * 78)
print("EXPOSURE TO SPECIAL DEMAND REGIMES")
print("=" * 78)
print(f"  Ramadan window  2026-02-17..03-19 : {nr:6d} rows ({100*nr/tot:5.1f}%)")
print(f"  post-Idulfitri  after 2026-03-19  : {ni:6d} rows ({100*ni/tot:5.1f}%)")
print(f"  Christmas/NY    2025-12-20..01-04 : {nx:6d} rows ({100*nx/tot:5.1f}%)")
print(f"  -> {100*(nr+ni+nx)/tot:.1f}% of test rows sit in a regime that train")
print("     (Apr-Sep 2025) contains little or none of.")

# what does train contain by way of comparison?
print("\n  Does train contain a comparable Ramadan/Idulfitri stretch?")
print("    Idulfitri 1446 H fell on 2025-03-31 -- ONE DAY BEFORE train starts.")
print("    Ramadan 1446 H ran Mar 1-30 2025 -- entirely BEFORE train starts.")
print("    So train has NO Ramadan and NO Idulfitri coverage at all.")

# national demand seasonality within train, for reference
tr = []
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        tr.append((dt.date.fromisoformat(r["date_show"]), int(r["total_ticket"])))
nat = defaultdict(int)
for d, t in tr:
    nat[d] += t
print("\n" + "=" * 78)
print("NATIONAL DAILY TICKETS IN TRAIN, BY MONTH (seasonal level the model sees)")
print("=" * 78)
bm = defaultdict(list)
for d, v in nat.items():
    bm[(d.year, d.month)].append(v)
for k in sorted(bm):
    v = bm[k]
    print(f"  {k[0]}-{k[1]:02d}  mean/day {sum(v)/len(v):10,.0f}  days {len(v)}")

print("\n" + "=" * 78)
print("DAY-OF-WEEK FACTORS IN TRAIN (national, normalised)")
print("=" * 78)
bd = defaultdict(list)
for d, v in nat.items():
    bd[d.weekday()].append(v)
g = sum(sum(v) for v in bd.values()) / sum(len(v) for v in bd.values())
for w in range(7):
    v = bd[w]
    print(f"  {DOW[w]:4s} factor {sum(v)/len(v)/g:5.2f}")
hw = [v for d, v in nat.items() if hol.__class__ and
      (lambda x: x)(dict((a, (b, c)) for a, b, c, _ in hol).get(d, ("weekday", "normal"))[1]) == "holiday"]
print(f"  named holidays factor {sum(hw)/len(hw)/g:5.2f}  (n={len(hw)} days)")
