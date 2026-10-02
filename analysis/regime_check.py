"""Settle the regime question using only test_history.csv + test.csv (pure stdlib).

Hypothesis to test: for test films, D1-D3 is the film's OPENING window
(D1 = release date), not a random mid-run window.
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


def day(s):
    return dt.date.fromisoformat(s)


hist = rd("test_history.csv")
test = rd("test.csv")
for r in hist:
    r["d"] = day(r["date_show"])
    r["t"] = int(r["total_ticket"])
    r["occ"] = float(r["occupation_rate"])
    r["sh"] = int(r["total_show"])
for r in test:
    r["d"] = day(r["date_show"])

# ---------------------------------------------------------------- D1 per film
d1 = {}
for r in hist:
    m = r["movie_title"]
    if m not in d1 or r["d"] < d1[m]:
        d1[m] = r["d"]

print("=" * 78)
print("1. DAY-OF-WEEK OF D1  (per test film)")
print("=" * 78)
c = Counter(DOW[d1[m].weekday()] for m in d1)
tot = sum(c.values())
for k in DOW:
    if c[k]:
        print(f"  {k}: {c[k]:4d}  ({100*c[k]/tot:5.1f}%)")
print(f"  total films: {tot}")
print("  -> If D1 were a RANDOM mid-run date, this would be ~uniform (~14% each).")

# ------------------------------------------- how many distinct D1 dates/weeks
print()
print("=" * 78)
print("2. ARE D1 DATES CLUSTERED ON WEEKLY RELEASE SLOTS?")
print("=" * 78)
dates = Counter(d1.values())
print(f"  distinct D1 dates: {len(dates)} for {tot} films")
print(f"  date range: {min(dates)} .. {max(dates)}")
span_weeks = (max(dates) - min(dates)).days / 7
print(f"  span: {span_weeks:.1f} weeks -> {tot/span_weeks:.1f} films per week")
print("  films sharing the same D1 date (top 12):")
for d, n in dates.most_common(12):
    print(f"    {d} {DOW[d.weekday()]}  {n} films")

# ------------------------------------------------- cluster count D1 -> D3 -> test
print()
print("=" * 78)
print("3. CLUSTER FOOTPRINT: D1 -> D2 -> D3  (opening = flat/growing wide release)")
print("=" * 78)
per = defaultdict(lambda: defaultdict(set))
for r in hist:
    per[r["movie_title"]][r["d"]].add(r["cinema_ids"])
grow = Counter()
rows = []
for m, dd in per.items():
    ds = sorted(dd)
    n = [len(dd[x]) for x in ds]
    if len(n) == 3:
        rows.append((m, n))
        if n[2] > n[0]:
            grow["grew"] += 1
        elif n[2] < n[0]:
            grow["shrank"] += 1
        else:
            grow["flat"] += 1
print(f"  films with 3 history days: {len(rows)}")
for k, v in grow.most_common():
    print(f"    {k}: {v}  ({100*v/len(rows):.1f}%)")
avg = [sum(n[i] for _, n in rows) / len(rows) for i in range(3)]
print(f"  mean clusters per film: D1={avg[0]:.1f} D2={avg[1]:.1f} D3={avg[2]:.1f}")

# ------------------------------------------------ national ticket trajectory
print()
print("=" * 78)
print("4. TICKET TRAJECTORY D1->D2->D3, SPLIT BY D1 DAY-OF-WEEK")
print("=" * 78)
bym = defaultdict(lambda: [0, 0, 0])
occm = defaultdict(lambda: [0.0, 0.0, 0.0])
cnt = defaultdict(lambda: [0, 0, 0])
for r in hist:
    m = r["movie_title"]
    o = (r["d"] - d1[m]).days
    if 0 <= o <= 2:
        bym[m][o] += r["t"]
        occm[m][o] += r["occ"]
        cnt[m][o] += 1
byd1 = defaultdict(list)
for m, v in bym.items():
    byd1[DOW[d1[m].weekday()]].append(v)
print(f"  {'D1dow':6s} {'n':>4s} {'s1':>9s} {'s2':>9s} {'s3':>9s}   {'s2/s1':>6s} {'s3/s1':>6s}")
for k in DOW:
    if k not in byd1:
        continue
    v = byd1[k]
    s = [sum(x[i] for x in v) / len(v) for i in range(3)]
    print(f"  {k:6s} {len(v):4d} {s[0]:9.0f} {s[1]:9.0f} {s[2]:9.0f}   "
          f"{s[1]/max(s[0],1):6.2f} {s[2]/max(s[0],1):6.2f}")
print("  -> A Wednesday RELEASE should ramp UP into Fri/Sat (s3/s1 > 1).")
print("     A random mid-run Wednesday window would look much flatter.")

# --------------------------------------- occupancy level: openings sell well
print()
print("=" * 78)
print("5. MEAN OCCUPATION RATE ON D1 BY D1 DAY-OF-WEEK")
print("=" * 78)
for k in DOW:
    ms = [m for m in d1 if DOW[d1[m].weekday()] == k]
    if not ms:
        continue
    o = [occm[m][0] / max(cnt[m][0], 1) for m in ms]
    print(f"  {k:6s} n={len(ms):4d}  mean_occ_D1={sum(o)/len(o):6.2f}")

# ------------------------------------------------- pairs in history not in test
print()
print("=" * 78)
print("6. THE 1450 PAIRS WITH HISTORY THAT ARE *NOT* PREDICTED")
print("=" * 78)
hp = defaultdict(lambda: [0, 0, 0])
for r in hist:
    m = r["movie_title"]
    o = (r["d"] - d1[m]).days
    if 0 <= o <= 2:
        hp[(m, r["cinema_ids"])][o] += r["t"]
tp = set((r["movie_title"], r["cinema_ids"]) for r in test)
inc = [v for k, v in hp.items() if k in tp]
exc = [v for k, v in hp.items() if k not in tp]
print(f"  pairs in history: {len(hp)} | predicted: {len(inc)} | NOT predicted: {len(exc)}")


def stat(v, lab):
    sc = sorted(max(sum(x) / 3.0, 1.0) for x in v)
    n = len(sc)
    print(f"  {lab:16s} n={n:6d}  scale p10={sc[n//10]:8.1f} p50={sc[n//2]:8.1f} "
          f"p90={sc[9*n//10]:8.1f} mean={sum(sc)/n:8.1f}")
    z = sum(1 for x in v if sum(x) == 0)
    d3z = sum(1 for x in v if x[2] == 0)
    print(f"  {'':16s}   all-zero D1-D3: {z} ({100*z/n:.1f}%) | zero on D3: {d3z} ({100*d3z/n:.1f}%)")


stat(inc, "PREDICTED")
stat(exc, "NOT PREDICTED")
print()
print("  -> If excluded pairs are simply 'pair died after D3', the organisers")
print("     filtered test.csv to pairs with activity in D4-D10. That would mean")
print("     EVERY predicted pair has >=1 non-zero day in D4-D10 (exploitable).")

# ------------------------------------------- off -> dow mapping in the test set
print()
print("=" * 78)
print("7. HORIZON -> DAY-OF-WEEK MAPPING IN test.csv  (collinearity check)")
print("=" * 78)
mp = defaultdict(Counter)
for r in test:
    mp[(r["d"] - d1[r["movie_title"]]).days][DOW[r["d"].weekday()]] += 1
for o in sorted(mp):
    tt = sum(mp[o].values())
    top = ", ".join(f"{k} {100*v/tt:.0f}%" for k, v in mp[o].most_common(3))
    print(f"  D{o:<3d} n={tt:6d}  {top}")
print("  -> horizon and day-of-week are nearly collinear in the test set,")
print("     because 81% of films share the same two D1 weekdays.")
