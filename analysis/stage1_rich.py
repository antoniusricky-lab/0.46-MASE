"""Can a RICHER pure-Python stage 1 capture more of the film-curve ceiling?

Current stage 1 keys on (horizon, D1-dow, trend bin, size bin) and banks
+0.0116 of a +0.0771 ceiling. Moving it to LightGBM is the obvious next step
but means new untested code. First check whether simply giving the existing
median table better features closes more of the gap for free.
"""
import csv
import datetime as dt
import os
import random
from collections import defaultdict

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
TEST_DOW = {2, 3, 4}
SEED = 2026

cell, occ_of, sh_of = {}, {}, {}
ncl, nat, mdc = defaultdict(lambda: defaultdict(int)), \
    defaultdict(lambda: defaultdict(int)), defaultdict(list)
natday = defaultdict(int)
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d = dt.date.fromisoformat(r["date_show"]); m = r["movie_title"]
        c = r["cinema_ids"]; t = int(r["total_ticket"])
        cell[(m, c, d)] = t
        occ_of[(m, c, d)] = float(r["occupation_rate"])
        sh_of[(m, c, d)] = int(r["total_show"])
        ncl[m][d] += 1; nat[m][d] += t; mdc[(m, d)].append(c); natday[d] += t
TMIN, TMAX = min(natday), max(natday)
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

HOL = {}
with open(D + "holidays.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        HOL[dt.date.fromisoformat(r["date"])] = (r["day_tipe"], r["holiday_tipe"])
GEN = {}
with open(D + "movies.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        GEN[r["original_title"].strip().upper()] = (r["genre"] or "").split(",")[0].strip()


def is_hol(d):
    return HOL.get(d, ("weekday", "normal"))[1] == "holiday"


def med(v):
    v = sorted(v); n = len(v)
    return 0.0 if n == 0 else (v[n // 2] if n % 2 else .5 * (v[n // 2 - 1] + v[n // 2]))


days = sorted(natday)
roll = {}
for j, d in enumerate(days):
    lo, hi = max(0, j - 3), min(len(days), j + 4)
    w = [natday[x] for x in days[lo:hi]]
    roll[d] = sum(w) / len(w)
rat = {d: natday[d] / roll[d] for d in days if roll[d] > 0}
DOWF = {w: med([rat[d] for d in rat if d.weekday() == w and not is_hol(d)])
        for w in range(7)}
HM = med([rat[d] / DOWF[d.weekday()] for d in rat if is_hol(d)])


def calf(d):
    f = DOWF[d.weekday()]
    return f * HM if is_hol(d) else f


def b(x, edges):
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)


def scb(x):
    return b(x, (5, 10, 25, 50, 100, 250, 600))


def trb(x):
    return b(x, (0.5, 0.75, 0.95, 1.15, 1.5))


def build(lags, dowf=None):
    pairs, film = [], {}
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
            keep = []
            for c in cs:
                s = [cell.get((m, c, dd), 0) for dd in obs]
                if sum(s) == 0 or s[2] == 0:
                    continue
                keep.append((c, s,
                             [cell.get((m, c, d1 + dt.timedelta(days=3 + h)), 0)
                              for h in range(7)],
                             [occ_of.get((m, c, dd), 0.0) for dd in obs],
                             [sh_of.get((m, c, dd), 0) for dd in obs]))
            if not keep:
                continue
            n123 = [sum(s[i] for _, s, _, _, _ in keep) for i in range(3)]
            sh123 = [sum(sh[i] for _, _, _, _, sh in keep) for i in range(3)]
            film[(m, d1)] = (sum(n123), [sum(y[h] for _, _, y, _, _ in keep)
                                         for h in range(7)],
                             n123, len(keep), sh123,
                             sum(sum(o) for _, _, _, o, _ in keep) / (3 * len(keep)))
            for c, s, y, o, sh in keep:
                pairs.append((m, d1, c, max(sum(s) / 3.0, 1.0),
                              "".join("1" if x > 0 else "0" for x in s),
                              scb(max(sum(s) / 3.0, 1.0)), y))
    return pairs, film


TR, TRF = build(tuple(range(25)))
EV, EVF = build((0,), TEST_DOW)
films = sorted({p[0] for p in EV})
random.Random(SEED).shuffle(films)
FOLD = {m: i % 5 for i, m in enumerate(films)}
N = 7 * len(EV)
print(f"train {len(TR):,} pairs / {len(TRF):,} film-windows | proxy {len(EV):,}")


def p_keys(r, h):
    return [(h, r[1].weekday(), r[4], r[5]), (h, r[1].weekday(), r[4]),
            (h, r[4], r[5]), (h, r[1].weekday()), (h, r[4]), (h,)]


# ---------------- stage-1 feature variants
def f_basic(key, F):
    m, d1 = key
    N_obs, _, n123, npairs, sh123, occ = F[key]
    return (d1.weekday(), trb(n123[2] / max(n123[0], 1)),
            scb(max(N_obs, 1) / 1000.0), min(npairs // 20, 5))


def f_rich(key, F):
    m, d1 = key
    N_obs, _, n123, npairs, sh123, occ = F[key]
    return (d1.weekday(), trb(n123[2] / max(n123[0], 1)),
            scb(max(N_obs, 1) / 1000.0), min(npairs // 20, 5),
            trb(n123[1] / max(n123[0], 1)),
            b(occ, (8, 15, 25, 40)),
            trb(sh123[2] / max(sh123[0], 1)),
            b(N_obs / max(sum(sh123), 1), (30, 60, 100)))


VARIANTS = {
    "basic  (h, dow, trend, size)": (
        f_basic,
        lambda ff, h: [(h,) + ff[:3], (h, ff[0], ff[1]), (h, ff[0], ff[2]),
                       (h, ff[0]), (h, ff[1]), (h,)]),
    "+ uplift of the target day   ": (
        f_basic,
        lambda ff, h: [(h,) + ff[:3] + (ff[4],), (h, ff[0], ff[1], ff[4]),
                       (h, ff[0], ff[1]), (h, ff[0], ff[4]), (h, ff[0]), (h,)]),
    "rich   (+occ, shows, tps)    ": (
        f_rich,
        lambda ff, h: [(h, ff[0], ff[1], ff[2]), (h, ff[0], ff[1], ff[5]),
                       (h, ff[0], ff[1]), (h, ff[0], ff[2]), (h, ff[0], ff[5]),
                       (h, ff[0]), (h, ff[1]), (h,)]),
    "rich + uplift                ": (
        f_rich,
        lambda ff, h: [(h, ff[0], ff[1], ff[2], ff[8]),
                       (h, ff[0], ff[1], ff[8]), (h, ff[0], ff[1], ff[5]),
                       (h, ff[0], ff[1]), (h, ff[0], ff[8]), (h, ff[0], ff[2]),
                       (h, ff[0]), (h,)]),
}


def upb(key, h):
    m, d1 = key
    base = sum(calf(d1 + dt.timedelta(days=i)) for i in range(3)) / 3.0
    return trb(calf(d1 + dt.timedelta(days=3 + h)) / max(base, 1e-6))


def run(ffun, kfun, use_oracle=False):
    tot = 0.0
    for f in range(5):
        trk = [k for k in TRF if FOLD.get(k[0], -1) != f]
        trr = [r for r in TR if FOLD.get(r[0], -1) != f]
        # stage 1
        acc = [defaultdict(list) for _ in range(9)]
        gl = defaultdict(list)
        for k in trk:
            N_obs, N_tgt, *_ = TRF[k]
            mu = max(N_obs / 3.0, 1.0)
            ffb = ffun(k, TRF)
            for h in range(7):
                ff = ffb + (upb(k, h),) * 5
                v = N_tgt[h] / mu
                gl[h].append(v)
                for j, kk in enumerate(kfun(ff, h)):
                    acc[j][kk].append(v)
        s1 = [{kk: med(v) for kk, v in a.items() if len(v) >= 8} for a in acc]
        s1g = {h: med(gl[h]) for h in range(7)}

        def p1(key, F, h):
            ff = ffun(key, F) + (upb(key, h),) * 5
            for t, kk in zip(s1, kfun(ff, h)):
                if kk in t:
                    return t[kk]
            return s1g[h]

        # stage 2 on the ratio to stage 1
        acc2 = [defaultdict(list) for _ in range(6)]
        base2 = []
        for r in trr:
            for h in range(7):
                nr = p1((r[0], r[1]), TRF, h)
                t = r[6][h] / r[3]
                base2.append(t)
                t = t / nr if nr > 1e-6 else 0.0
                for j, kk in enumerate(p_keys(r, h)):
                    acc2[j][kk].append(t)
        s2 = [{kk: med(v) for kk, v in a.items() if len(v) >= 40} for a in acc2]
        s2g = med(base2)

        def p2(r, h):
            for t, kk in zip(s2, p_keys(r, h)):
                if kk in t:
                    return t[kk]
            return s2g

        for r in EV:
            if FOLD.get(r[0], -1) != f:
                continue
            for h in range(7):
                if use_oracle:
                    N_obs, N_tgt, *_ = EVF[(r[0], r[1])]
                    nr = N_tgt[h] / max(N_obs / 3.0, 1.0)
                else:
                    nr = p1((r[0], r[1]), EVF, h)
                tot += abs(r[6][h] / r[3] - max(0.0, p2(r, h) * nr))
    return tot / N


print()
print("=" * 78)
print("STAGE-1 FEATURE VARIANTS (two-stage proxy MASE)")
print("=" * 78)
base_single = None
# single-stage reference
acc = [defaultdict(list) for _ in range(6)]
tot = 0.0
for f in range(5):
    trr = [r for r in TR if FOLD.get(r[0], -1) != f]
    a2 = [defaultdict(list) for _ in range(6)]
    allv = []
    for r in trr:
        for h in range(7):
            v = r[6][h] / r[3]
            allv.append(v)
            for j, kk in enumerate(p_keys(r, h)):
                a2[j][kk].append(v)
    t2 = [{kk: med(v) for kk, v in x.items() if len(v) >= 40} for x in a2]
    g2 = med(allv)
    for r in EV:
        if FOLD.get(r[0], -1) != f:
            continue
        for h in range(7):
            p = g2
            for t, kk in zip(t2, p_keys(r, h)):
                if kk in t:
                    p = t[kk]
                    break
            tot += abs(r[6][h] / r[3] - p)
base_single = tot / N
print(f"  {'single-stage reference':34s} {base_single:.4f}")
for lab, (ffun, kfun) in VARIANTS.items():
    v = run(ffun, kfun)
    print(f"  {lab:34s} {v:.4f}   gain {base_single - v:+.4f}")
orc = run(f_basic, VARIANTS["basic  (h, dow, trend, size)"][1], use_oracle=True)
print(f"  {'stage-1 ORACLE (ceiling)':34s} {orc:.4f}   gain {base_single - orc:+.4f}")
