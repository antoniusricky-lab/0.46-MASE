"""Why does model Z win the proxy and lose the leaderboard?

Scoreboard:
  v3  no Z,  quantile snap, film_curve, lags 0-12   proxy ~0.337   LB 0.46795
  v4  +Z                                            proxy  0.3377  LB 0.52220 (no guard)
  v7  +Z, capped snap, lags 0-24, stage-1 uplift     proxy  0.3348  LB 0.53421

Z improves the proxy every time and destroys the leaderboard every time. Two
candidate mechanisms, both testable here:

  M1 the decile EDGES are absolute. TP_EDGES comes from the out-of-fold p0
     distribution and is applied to test p0. If test p0 shifts up, more rows
     land in the zeroing deciles. This is the same class of bug as the
     absolute zero-snap threshold fixed earlier.
  M2 forcing exact zeros is one-sided. A wrongly zeroed row loses the entire
     true ratio, so the loss is unbounded in one direction while the gain is
     capped. Optimal under matched distributions, reckless under shift.

Tested against a TIME-BASED holdout - train on earlier films, validate on the
latest ones - which the original 0.46641 notebook had (its cell 11) and this
pipeline dropped.
"""
import csv
import datetime as dt
import os
import random
from collections import defaultdict

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
TEST_DOW = {2, 3, 4}
SEED = 2026

cell, ncl, nat, mdc = {}, defaultdict(lambda: defaultdict(int)), \
    defaultdict(lambda: defaultdict(int)), defaultdict(list)
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d = dt.date.fromisoformat(r["date_show"]); m = r["movie_title"]
        cell[(m, r["cinema_ids"], d)] = int(r["total_ticket"])
        ncl[m][d] += 1; nat[m][d] += int(r["total_ticket"])
        mdc[(m, d)].append(r["cinema_ids"])
TMIN = min(d for m in nat for d in nat[m]); TMAX = max(d for m in nat for d in nat[m])
first = {m: min(v) for m, v in nat.items()}
LEFT = {m for m in first if first[m] == TMIN}
release = {}
for m in first:
    if m in LEFT:
        continue
    wk = [d for d in sorted(nat[m]) if d <= first[m] + dt.timedelta(days=9)]
    pk = max(ncl[m][d] for d in wk)
    for d in wk:
        if ncl[m][d] >= 0.25 * pk:
            release[m] = d
            break


def med(v):
    v = sorted(v); n = len(v)
    return 0.0 if n == 0 else (v[n // 2] if n % 2 else .5 * (v[n // 2 - 1] + v[n // 2]))


def scb(x):
    for i, b in enumerate((5, 10, 25, 50, 100, 250, 600)):
        if x <= b:
            return i
    return 7


def build(lags, dowf=None):
    out = []
    for m in sorted(release):
        for lag in lags:
            d1 = release[m] + dt.timedelta(days=lag)
            if d1 + dt.timedelta(days=9) > TMAX:
                continue
            if dowf and d1.weekday() not in dowf:
                continue
            obs = [d1 + dt.timedelta(days=i) for i in range(3)]
            cs = set()
            for dd in obs:
                cs.update(mdc.get((m, dd), ()))
            for c in cs:
                s = [cell.get((m, c, dd), 0) for dd in obs]
                if sum(s) == 0 or s[2] == 0:
                    continue
                sc = max(sum(s) / 3.0, 1.0)
                out.append((m, d1, c, sc,
                            "".join("1" if x > 0 else "0" for x in s), scb(sc),
                            [cell.get((m, c, d1 + dt.timedelta(days=3 + h)), 0)
                             for h in range(7)]))
    return out


TR = build(tuple(range(25)))
EV = build((0,), TEST_DOW)


def keys(r, h):
    return [(h, r[1].weekday(), r[4], r[5]), (h, r[1].weekday(), r[4]),
            (h, r[4], r[5]), (h, r[1].weekday()), (h, r[4]), (h,)]


def fit_main(rows, min_n=40):
    acc = [defaultdict(list) for _ in range(6)]
    allv = []
    for r in rows:
        for h in range(7):
            v = r[6][h] / r[3]
            allv.append(v)
            for j, k in enumerate(keys(r, h)):
                acc[j][k].append(v)
    return ([{k: med(v) for k, v in a.items() if len(v) >= min_n} for a in acc],
            med(allv))


def fit_zero(rows, min_n=40):
    acc = [defaultdict(lambda: [0, 0]) for _ in range(6)]
    n0 = n = 0
    for r in rows:
        for h in range(7):
            z = 1 if r[6][h] == 0 else 0
            n0 += z; n += 1
            for j, k in enumerate(keys(r, h)):
                acc[j][k][0] += z; acc[j][k][1] += 1
    return ([{k: v[0] / v[1] for k, v in a.items() if v[1] >= min_n}
             for a in acc], n0 / n)


def fit_pos(rows, min_n=20):
    acc = [defaultdict(list) for _ in range(6)]
    allv = []
    for r in rows:
        for h in range(7):
            v = r[6][h] / r[3]
            if v <= 0:
                continue
            allv.append(v)
            for j, k in enumerate(keys(r, h)):
                acc[j][k].append(v)
    return ([{k: med(v) for k, v in a.items() if len(v) >= min_n} for a in acc],
            med(allv))


def pr(mdl, r, h):
    tabs, g = mdl
    for t, k in zip(tabs, keys(r, h)):
        if k in t:
            return t[k]
    return g


def quantile(v, q):
    v = sorted(v)
    return v[min(int(q * (len(v) - 1)), len(v) - 1)] if v else 0.0


def evaluate(split_name, tr_rows, ev_rows):
    mm = fit_main(tr_rows)
    mz = fit_zero(tr_rows)
    mp = fit_pos(tr_rows)
    # fit decile multipliers on the TRAIN side (as the notebook does)
    p0t = [pr(mz, r, h) for r in tr_rows for h in range(7)]
    post = [pr(mp, r, h) for r in tr_rows for h in range(7)]
    yt = [r[6][h] / r[3] for r in tr_rows for h in range(7)]
    edges = [quantile(p0t, q / 10.0) for q in range(1, 10)]

    def dec_abs(v):
        return sum(1 for e in edges if v > e)

    mult = {}
    for d in range(10):
        sel = [i for i in range(len(yt)) if dec_abs(p0t[i]) == d]
        if not sel:
            mult[d] = 1.0
            continue
        best, bv = 1.0, None
        for k100 in range(0, 141, 5):
            k = k100 / 100.0
            v = sum(abs(yt[i] - post[i] * k) for i in sel) / len(sel)
            if bv is None or v < bv:
                best, bv = k, v
        mult[d] = best

    # evaluate: main, Z with ABSOLUTE edges, Z with RANK-based deciles
    p0e = [pr(mz, r, h) for r in ev_rows for h in range(7)]
    pose = [pr(mp, r, h) for r in ev_rows for h in range(7)]
    ye = [r[6][h] / r[3] for r in ev_rows for h in range(7)]
    maine = [pr(mm, r, h) for r in ev_rows for h in range(7)]
    n = len(ye)
    e_edges = [quantile(p0e, q / 10.0) for q in range(1, 10)]

    def dec_rank(v):
        return sum(1 for e in e_edges if v > e)

    m_main = sum(abs(ye[i] - maine[i]) for i in range(n)) / n
    m_zabs = sum(abs(ye[i] - pose[i] * mult[dec_abs(p0e[i])])
                 for i in range(n)) / n
    m_zrnk = sum(abs(ye[i] - pose[i] * mult[dec_rank(p0e[i])])
                 for i in range(n)) / n
    z_abs = sum(1 for i in range(n) if mult[dec_abs(p0e[i])] == 0) / n
    z_rnk = sum(1 for i in range(n) if mult[dec_rank(p0e[i])] == 0) / n
    true_z = sum(1 for v in ye if v == 0) / n
    print(f"  {split_name}")
    print(f"    main model                  {m_main:.4f}")
    print(f"    Z, absolute decile edges    {m_zabs:.4f}   "
          f"forces zero on {z_abs:.1%}")
    print(f"    Z, rank-based deciles       {m_zrnk:.4f}   "
          f"forces zero on {z_rnk:.1%}")
    print(f"    true zero share             {true_z:.1%}")
    return m_main, m_zabs, m_zrnk


print("=" * 78)
print("RANDOM FILM SPLIT (what the notebook's proxy does)")
print("=" * 78)
films = sorted({r[0] for r in EV})
random.Random(SEED).shuffle(films)
F5 = {m: i % 5 for i, m in enumerate(films)}
tr = [r for r in TR if F5.get(r[0], -1) != 0]
ev = [r for r in EV if F5.get(r[0], -1) == 0]
evaluate("fold 0", tr, ev)

print()
print("=" * 78)
print("TIME-BASED HOLDOUT (train on earlier films, test on the latest)")
print("=" * 78)
print("  This is the split the 0.46641 notebook had and this pipeline dropped.")
rel = sorted(release.items(), key=lambda kv: kv[1])
cut = rel[int(0.75 * len(rel))][1]
late = {m for m, d in release.items() if d >= cut}
print(f"  cutoff {cut} | early films {len(release) - len(late)} | "
      f"late films {len(late)}")
tr = [r for r in TR if r[0] not in late]
ev = [r for r in EV if r[0] in late]
print(f"  train pairs {len(tr):,} | holdout pairs {len(ev):,}")
evaluate("time holdout", tr, ev)

print()
print("=" * 78)
print("INTERPRETATION")
print("=" * 78)
print("  If Z beats main on the random split but loses on the time holdout,")
print("  the proxy was never able to see this and the time holdout must")
print("  become a gating metric.")
print("  If rank-based deciles beat absolute edges, mechanism M1 is real and")
print("  the decile mapping needs to be rank-based like the zero-snap.")
