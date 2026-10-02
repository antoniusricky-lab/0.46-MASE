"""Replicate the EXACT test-set construction on train.csv and measure MASE.

Test construction, now proven from the data:
  * D1 = the film's release date (weekly Wed/Thu/Fri slot)
  * D1-D3 = the first three calendar days
  * a (film, cluster) pair is included IFF it had a transaction on D3
  * targets are D4-D10, zero when no transaction exists

The notebook instead trains on anchors at lag 0..47 and only filters
sum3 > 0. This script measures how much that matters.
"""
import csv
import datetime as dt
from collections import Counter, defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TEST_D1_DOW = {2, 3, 4}          # Wed/Thu/Fri = 96% of test films


def load(p):
    out = []
    with open(D + p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append((dt.date.fromisoformat(r["date_show"]), r["cinema_ids"],
                        r["movie_title"], int(r["total_ticket"]),
                        float(r["occupation_rate"]), int(r["total_show"])))
    return out


tr = load("train.csv")
TMIN = min(x[0] for x in tr)
TMAX = max(x[0] for x in tr)
print(f"train span {TMIN} .. {TMAX}  rows={len(tr):,}")

# ------------------------------------------------------- per (film, cluster, date)
cell = {}
for d, c, m, t, o, s in tr:
    cell[(m, c, d)] = (t, o, s)
film_dates = defaultdict(set)
for d, c, m, t, o, s in tr:
    film_dates[m].add(d)

release = {m: min(ds) for m, ds in film_dates.items()}
left_cens = {m for m in release if release[m] == TMIN}
print(f"films={len(release)} | left-censored (release==train start, unusable)={len(left_cens)}")

print("\nrelease day-of-week in train (all films):")
c = Counter(DOW[release[m].weekday()] for m in release if m not in left_cens)
for k in DOW:
    if c[k]:
        print(f"  {k}: {c[k]:3d}")

# =========================================================== build test-like windows
def build(anchor_lag=0, require_d3=True, dow_filter=None):
    """One window per film at `anchor_lag` days after its release."""
    rows = []
    films = 0
    for m in sorted(release):
        if m in left_cens:
            continue
        d1 = release[m] + dt.timedelta(days=anchor_lag)
        if d1 + dt.timedelta(days=9) > TMAX:
            continue
        if dow_filter and d1.weekday() not in dow_filter:
            continue
        films += 1
        obs = [d1 + dt.timedelta(days=i) for i in range(3)]
        tgt = [d1 + dt.timedelta(days=i) for i in range(3, 10)]
        clusters = {c for (mm, c, d) in cell if mm == m and d in obs} \
            if False else None
        # gather clusters cheaply
        clusters = set()
        for i, dd in enumerate(obs):
            for key in idx_md.get((m, dd), ()):
                clusters.add(key)
        for c in clusters:
            s = [cell.get((m, c, dd), (0, 0, 0))[0] for dd in obs]
            if require_d3 and s[2] == 0:
                continue
            if sum(s) == 0:
                continue
            y = [cell.get((m, c, dd), (0, 0, 0))[0] for dd in tgt]
            rows.append((m, c, d1, s, y))
    return rows, films


idx_md = defaultdict(list)
for d, c, m, t, o, s in tr:
    idx_md[(m, d)].append(c)


def mase(rows, predfn):
    tot = n = 0.0
    for (m, c, d1, s, y) in rows:
        sc = max(sum(s) / 3.0, 1.0)
        p = predfn(m, c, d1, s, sc)
        for h in range(7):
            tot += abs(y[h] - p[h]) / sc
            n += 1
    return tot / n, int(n)


# ------------------------------------------------------------------ baselines
def b0(m, c, d1, s, sc):
    return [sc] * 7


print("\n" + "=" * 78)
print("EFFECT OF THE  s3 > 0  FILTER  (release windows, lag 0)")
print("=" * 78)
for req in (False, True):
    rows, nf = build(0, require_d3=req)
    v, n = mase(rows, b0)
    print(f"  require_d3={req!s:5s} films={nf:3d} pairs={len(rows):6d} rows={n:7d}"
          f"  B0_naive_MASE={v:.4f}")

rows_all, _ = build(0, require_d3=False)
rows_d3, _ = build(0, require_d3=True)
dropped = [r for r in rows_all if r[3][2] == 0]
print(f"\n  pairs dropped by the filter: {len(dropped)} "
      f"({100*len(dropped)/len(rows_all):.1f}% of lag-0 pairs)")
if dropped:
    v, n = mase(dropped, b0)
    sc = sorted(max(sum(r[3]) / 3.0, 1.0) for r in dropped)
    tgt0 = sum(1 for r in dropped if sum(r[4]) == 0)
    print(f"  their B0 MASE = {v:.4f}  (vs {mase(rows_d3,b0)[0]:.4f} for kept pairs)")
    print(f"  their scale   : p50={sc[len(sc)//2]:.1f} mean={sum(sc)/len(sc):.1f}")
    print(f"  fully dead in D4-D10: {tgt0}/{len(dropped)} = {100*tgt0/len(dropped):.1f}%")
    print("  -> these pairs are UNREACHABLE in test, yet the notebook trains on them.")

# ------------------------------------------------------- MASE by anchor_lag
print("\n" + "=" * 78)
print("B0 MASE BY ANCHOR LAG, WITH THE CORRECT FILTER APPLIED")
print("=" * 78)
print(f"  {'lag':>4s} {'films':>6s} {'pairs':>7s} {'B0_MASE':>9s}")
for lag in (0, 1, 2, 3, 5, 7, 10, 14, 21, 28):
    rows, nf = build(lag, require_d3=True)
    if not rows:
        continue
    v, n = mase(rows, b0)
    print(f"  {lag:4d} {nf:6d} {len(rows):7d} {v:9.4f}")

# -------------------------------------- the real shape of y/scale at release
print("\n" + "=" * 78)
print("TRUE y/scale PROFILE ON RELEASE WINDOWS (the thing to predict)")
print("=" * 78)
rows, _ = build(0, require_d3=True, dow_filter=TEST_D1_DOW)
print(f"  films matching test D1-dow (Wed/Thu/Fri): pairs={len(rows)}")
prof = defaultdict(list)
for (m, c, d1, s, y) in rows:
    sc = max(sum(s) / 3.0, 1.0)
    for h in range(7):
        prof[h].append(y[h] / sc)
print(f"  {'D':>3s} {'mean':>7s} {'median':>7s} {'share_zero':>11s}")
for h in range(7):
    v = sorted(prof[h])
    z = sum(1 for x in v if x == 0) / len(v)
    print(f"  D{h+4:<2d} {sum(v)/len(v):7.3f} {v[len(v)//2]:7.3f} {z:11.3f}")

# ------------------------------- partial-activity pairs: deflated scale
print("\n" + "=" * 78)
print("PARTIAL-ACTIVITY PAIRS (s1 and/or s2 == 0): scale is DEFLATED")
print("=" * 78)
rows, _ = build(0, require_d3=True)
grp = defaultdict(list)
for (m, c, d1, s, y) in rows:
    pat = "".join("1" if x > 0 else "0" for x in s)
    sc = max(sum(s) / 3.0, 1.0)
    grp[pat].append((sum(y) / 7.0 / sc, sc, y, s))
print(f"  {'D1D2D3':8s} {'pairs':>7s} {'mean y/scale':>13s} {'median scale':>13s}")
for pat in sorted(grp, reverse=True):
    v = grp[pat]
    r = sum(x[0] for x in v) / len(v)
    sc = sorted(x[1] for x in v)
    print(f"  {pat:8s} {len(v):7d} {r:13.3f} {sc[len(sc)//2]:13.1f}")
print("  -> pattern 001 / 011 pairs have scale computed over days the pair was")
print("     not even showing, so the correct y/scale is far ABOVE 1.")
print("     Test has 189 pairs at '001' and 684 at '011' = 8.4% of all test pairs.")
