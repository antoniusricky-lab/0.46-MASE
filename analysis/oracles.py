"""Oracle study: rank the remaining levers before building anything.

Each oracle hands the predictor perfect knowledge of ONE quantity and
measures the resulting MASE on the proxy population. The gap between an
oracle and the baseline is the ceiling on what predicting that quantity well
could ever buy. Cheap way to avoid building the wrong feature.
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
natday = defaultdict(int)
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d = dt.date.fromisoformat(r["date_show"]); m = r["movie_title"]
        c = r["cinema_ids"]; t = int(r["total_ticket"])
        cell[(m, c, d)] = t
        ncl[m][d] += 1; nat[m][d] += t; mdc[(m, d)].append(c); natday[d] += t
TMIN = min(natday); TMAX = max(natday)
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


def scbin(x):
    for i, b in enumerate((5, 10, 25, 50, 100, 250, 600)):
        if x <= b:
            return i
    return 7


# ------------------------------------------------------------ build windows
def build(lags, dowf=None):
    """-> per-pair records with per-horizon truth, plus film-level totals."""
    pairs, film = [], {}
    for m in sorted(release):
        for lag in lags:
            d1 = release[m] + dt.timedelta(days=lag)
            if d1 + dt.timedelta(days=9) > TMAX:
                continue
            if dowf and d1.weekday() not in dowf:
                continue
            obs = [d1 + dt.timedelta(days=i) for i in range(3)]
            tgt = [d1 + dt.timedelta(days=i) for i in range(3, 10)]
            cs = set()
            for dd in obs:
                cs.update(mdc.get((m, dd), ()))
            keep = []
            for c in cs:
                s = [cell.get((m, c, dd), 0) for dd in obs]
                if sum(s) == 0 or s[2] == 0:
                    continue
                y = [cell.get((m, c, dd), 0) for dd in tgt]
                keep.append((c, s, y))
            if not keep:
                continue
            N_obs = sum(sum(s) for _, s, _ in keep)
            N_tgt = [sum(y[h] for _, _, y in keep) for h in range(7)]
            film[(m, d1)] = (N_obs, N_tgt)
            for c, s, y in keep:
                pairs.append((m, d1, c, s, y))
    return pairs, film


TR, TRF = build(tuple(range(25)))
EV, EVF = build((0,), TEST_DOW)
print(f"train pairs {len(TR):,} | proxy pairs {len(EV):,}")

films = sorted({p[0] for p in EV})
random.Random(SEED).shuffle(films)
FOLD = {m: i % 5 for i, m in enumerate(films)}


def rec(p):
    m, d1, c, s, y = p
    sc = max(sum(s) / 3.0, 1.0)
    pat = "".join("1" if x > 0 else "0" for x in s)
    return m, d1, c, sc, pat, scbin(sc), s, y


TRr = [rec(p) for p in TR]
EVr = [rec(p) for p in EV]


def keys(r, h):
    return [(h, r[1].weekday(), r[4], r[5]), (h, r[1].weekday(), r[4]),
            (h, r[4], r[5]), (h, r[1].weekday()), (h, r[4]), (h,)]


def fit(rows, min_n=40):
    acc = [defaultdict(list) for _ in range(6)]
    for r in rows:
        for h in range(7):
            v = r[7][h] / r[3]
            for j, k in enumerate(keys(r, h)):
                acc[j][k].append(v)
    return ([{k: med(v) for k, v in a.items() if len(v) >= min_n} for a in acc],
            med([r[7][h] / r[3] for r in rows for h in range(7)]))


def pred1(mdl, r, h):
    tabs, g = mdl
    for t, k in zip(tabs, keys(r, h)):
        if k in t:
            return t[k]
    return g


# ------------------------------------------------------------- baseline OOF
OOF = {}
for f in range(5):
    mdl = fit([r for r in TRr if FOLD.get(r[0], -1) != f])
    for i, r in enumerate(EVr):
        if FOLD.get(r[0], -1) == f:
            for h in range(7):
                OOF[(i, h)] = pred1(mdl, r, h)

N = 7 * len(EVr)


def score(fn, lab):
    tot = 0.0
    for i, r in enumerate(EVr):
        for h in range(7):
            tot += abs(r[7][h] / r[3] - fn(i, r, h))
    print(f"  {lab:52s} {tot / N:.4f}")
    return tot / N


print()
print("=" * 78)
print("ORACLE STUDY (proxy population, 5-fold grouped by film)")
print("=" * 78)
base = score(lambda i, r, h: OOF[(i, h)], "baseline: median(h, D1dow, pattern, scalebin)")

# --- oracle A: the film's own national ratio at each horizon is known
def orA(i, r, h):
    N_obs, N_tgt = EVF[(r[0], r[1])]
    return N_tgt[h] / max(N_obs / 3.0, 1.0)


a = score(orA, "A  film's national ratio known (no pair info)")

# --- oracle A2: film ratio known, scaled by the pair's own observed share
def orA2(i, r, h):
    N_obs, N_tgt = EVF[(r[0], r[1])]
    nr = N_tgt[h] / max(N_obs / 3.0, 1.0)
    return nr


# --- oracle B: whether the pair is alive at h is known; level from baseline
def orB(i, r, h):
    return 0.0 if r[7][h] == 0 else max(OOF[(i, h)], 1e-9)


b = score(orB, "B  zero / non-zero known, level from baseline")

# --- oracle C: the national date factor is known (cross-film signal ceiling)
roll = {}
ds = sorted(natday)
for j, d in enumerate(ds):
    lo, hi = max(0, j - 3), min(len(ds), j + 4)
    w = [natday[x] for x in ds[lo:hi]]
    roll[d] = sum(w) / len(w)


def orC(i, r, h):
    t = r[1] + dt.timedelta(days=3 + h)
    obs = [r[1] + dt.timedelta(days=k) for k in range(3)]
    bo = sum(natday.get(x, 0) for x in obs) / 3.0
    f = natday.get(t, 0) / bo if bo > 0 else 1.0
    return OOF[(i, h)] * f / max(med([1.0]), 1e-9) if False else OOF[(i, h)] * f


c = score(orC, "C  target-date national demand known (multiplier)")

# --- oracle D: pair's TOTAL over D4-D10 known, shape from baseline
def orD(i, r, h):
    tot = sum(r[7]) / r[3]
    sh = sum(OOF[(i, k)] for k in range(7))
    return tot * (OOF[(i, h)] / sh) if sh > 1e-9 else tot / 7.0


d_ = score(orD, "D  pair's 7-day total known, shape from baseline")

print()
print("=" * 78)
print("WHAT EACH ORACLE IS WORTH")
print("=" * 78)
for lab, v in (("A film national curve", a), ("B zero/non-zero flag", b),
               ("C target-date demand", c), ("D pair 7-day total", d_)):
    print(f"  {lab:28s} {v:.4f}   gain {base - v:+.4f}  "
          f"({100 * (base - v) / base:+.1f}%)")

print()
print("=" * 78)
print("WHY ARE WEDNESDAY RELEASES SO MUCH HARDER?  (41% of the test set)")
print("=" * 78)
print(f"  {'D1 dow':8s} {'pairs':>7s}" + "".join(f"{'D' + str(h + 4):>8s}" for h in range(7)))
for w in sorted(TEST_DOW):
    sel = [i for i, r in enumerate(EVr) if r[1].weekday() == w]
    cells = []
    for h in range(7):
        cells.append(sum(abs(EVr[i][7][h] / EVr[i][3] - OOF[(i, h)])
                         for i in sel) / len(sel))
    print(f"  {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][w]:8s} {len(sel):7d}"
          + "".join(f"{x:8.3f}" for x in cells))
print("\n  true median y/scale for comparison:")
for w in sorted(TEST_DOW):
    sel = [i for i, r in enumerate(EVr) if r[1].weekday() == w]
    cells = [med([EVr[i][7][h] / EVr[i][3] for i in sel]) for h in range(7)]
    print(f"  {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][w]:8s} {len(sel):7d}"
          + "".join(f"{x:8.3f}" for x in cells))

print()
print("=" * 78)
print("FORMAT VARIANTS: same film, different screen format")
print("=" * 78)
TAGS = ("(IMAX 2D)", "(IMAX)", "(3D)", "(2D)", "(DUB)", "(DUBBING)",
        "(RE-RELEASE)", "(PREMIUM)", "(4DX)", "(GOLD CLASS)")


def base_title(t):
    u = t.upper()
    for g in TAGS:
        u = u.replace(g, "")
    return " ".join(u.split())


test_films = set()
with open(D + "test_history.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        test_films.add(r["movie_title"])
grp = defaultdict(list)
for t in test_films:
    grp[base_title(t)].append(t)
multi = {k: v for k, v in grp.items() if len(v) > 1}
print(f"  test films {len(test_films)} -> {len(grp)} base titles, "
      f"{len(multi)} with >1 format")
nrows = 0
with open(D + "test.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
inmulti = {t for v in multi.values() for t in v}
nrows = sum(1 for r in rows if r["movie_title"] in inmulti)
print(f"  test rows belonging to a multi-format title: {nrows:,} "
      f"({100 * nrows / len(rows):.1f}%)")
for k in sorted(multi)[:8]:
    print(f"    {k[:40]:42s} -> {len(multi[k])} formats")

# do format variants share a trajectory? measure on train
tgrp = defaultdict(list)
for m in release:
    tgrp[base_title(m)].append(m)
tmulti = {k: v for k, v in tgrp.items() if len(v) > 1}
print(f"\n  train base titles with >1 format: {len(tmulti)}")
pairs_by = defaultdict(dict)
for i, r in enumerate(EVr):
    pairs_by[(base_title(r[0]), r[1], r[2])][r[0]] = i
shared = [v for v in pairs_by.values() if len(v) > 1]
print(f"  proxy (cluster, date) slots where 2+ formats of the same title "
      f"both play: {len(shared)}")
if shared:
    agree = []
    for v in shared:
        ii = list(v.values())
        for h in range(7):
            rr = [EVr[i][7][h] / EVr[i][3] for i in ii]
            agree.append(max(rr) - min(rr))
    print(f"  spread in y/scale between formats, same slot: "
          f"median {med(agree):.3f} mean {sum(agree) / len(agree):.3f}")
    print("  -> small spread would mean formats are near-duplicates and can")
    print("     be pooled; large spread means they must stay separate.")
