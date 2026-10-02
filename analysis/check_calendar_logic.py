"""Verify the calendar math in fixes.py independently, in pure stdlib.

Mirrors islamic_calendar_features() exactly so the Eid / Ramadan indices can be
checked without pandas (unavailable in this sandbox).
"""
import csv
import datetime as dt
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep

EID = [dt.date(2025, 3, 31), dt.date(2026, 3, 21)]
RAM = 30


def feats(d):
    diffs = [(d - e).days for e in EID]
    signed = min(diffs, key=abs)
    in_eid = 0 <= signed <= 13
    in_ram = -RAM <= signed < 0
    month, dom = d.month, d.day
    return dict(
        days_since_eid=max(-45, min(45, signed)),
        is_eid_window=int(in_eid),
        eid_week=(signed // 7 + 1) if in_eid else 0,
        in_ramadan=int(in_ram),
        ramadan_day=(signed + RAM + 1) if in_ram else 0,
        ramadan_late=int(in_ram and signed >= -10),
        in_xmas_ny=int((month == 12 and dom >= 20) or (month == 1 and dom <= 4)),
    )


print("=" * 78)
print("1. KEY DATES")
print("=" * 78)
hdr = ["days_since_eid", "is_eid_window", "eid_week", "in_ramadan",
       "ramadan_day", "ramadan_late", "in_xmas_ny"]
print(f"  {'date':12s} {'dow':4s} " + "".join(f"{h[:9]:>10s}" for h in hdr))
for s in ["2025-03-31", "2025-04-01", "2025-04-07", "2025-04-13", "2025-04-14",
          "2025-06-15", "2025-12-25", "2026-01-01", "2026-02-17", "2026-02-18",
          "2026-03-11", "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-21",
          "2026-03-22", "2026-03-27"]:
    d = dt.date.fromisoformat(s)
    f = feats(d)
    print(f"  {s:12s} {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d.weekday()]:4s} "
          + "".join(f"{f[h]:>10d}" for h in hdr))

print()
print("=" * 78)
print("2. CRITICAL CHECK: does train actually cover the Eid+ range the")
print("   2026-03-18 cohort needs?")
print("=" * 78)
cohort_tgt = [dt.date(2026, 3, 21) + dt.timedelta(days=i) for i in range(7)]
need = sorted({feats(d)["days_since_eid"] for d in cohort_tgt})
print(f"  cohort D4-D10 dates      : {cohort_tgt[0]} .. {cohort_tgt[-1]}")
print(f"  days_since_eid they need : {need}")

TR_A, TR_B = dt.date(2025, 4, 1), dt.date(2025, 9, 30)
have = sorted({feats(TR_A + dt.timedelta(days=i))["days_since_eid"]
               for i in range((TR_B - TR_A).days + 1)
               if feats(TR_A + dt.timedelta(days=i))["is_eid_window"]})
print(f"  days_since_eid in train  : {have}")
cov = [x for x in need if x in have]
print(f"  covered by train         : {cov}  ({len(cov)}/{len(need)})")
print(f"  NOT covered              : {[x for x in need if x not in have]}")

print()
print("=" * 78)
print("3. HOW MANY TRAIN ROWS SUPPORT EACH days_since_eid VALUE?")
print("=" * 78)
from collections import defaultdict

cnt = defaultdict(int)
tix = defaultdict(int)
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d = dt.date.fromisoformat(r["date_show"])
        k = feats(d)["days_since_eid"]
        if feats(d)["is_eid_window"]:
            cnt[k] += 1
            tix[k] += int(r["total_ticket"])
print(f"  {'days_since_eid':>15s} {'train rows':>11s} {'tickets':>12s}")
for k in sorted(cnt):
    mark = "  <- cohort needs" if k in need else ""
    print(f"  {k:>15d} {cnt[k]:>11,} {tix[k]:>12,}{mark}")
print(f"\n  total train rows inside an Eid window: {sum(cnt.values()):,}")

print()
print("=" * 78)
print("4. RAMADAN COVERAGE IN TRAIN (expected: zero)")
print("=" * 78)
nram = 0
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if feats(dt.date.fromisoformat(r["date_show"]))["in_ramadan"]:
            nram += 1
print(f"  train rows flagged in_ramadan: {nram}")
print("  -> confirms Ramadan is unlearnable from train; only the Eid aftermath is.")

print()
print("=" * 78)
print("5. TEST ROWS BY REGIME FLAG (what the new features will actually mark)")
print("=" * 78)
tc = defaultdict(int)
with open(D + "test.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    f2 = feats(dt.date.fromisoformat(r["date_show"]))
    if f2["is_eid_window"]:
        tc["eid_window"] += 1
    elif f2["in_ramadan"]:
        tc["ramadan"] += 1
    elif f2["in_xmas_ny"]:
        tc["xmas_ny"] += 1
    else:
        tc["ordinary"] += 1
tot = len(rows)
for k in ("ordinary", "ramadan", "eid_window", "xmas_ny"):
    print(f"  {k:12s} {tc[k]:7d} ({100*tc[k]/tot:5.1f}%)")
print(f"  {'TOTAL':12s} {tot:7d}")
print()
print("  eid_window rows are the ones eid_cohort_guard() will floor.")
