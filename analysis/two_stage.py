"""Is the film-level national curve PREDICTABLE, not just valuable?

Oracle A showed that knowing a film's own national D4-D10 ratio curve - with
no pair-level information at all - scores 0.3039 against a 0.3944 baseline.
That is only a ceiling. This script asks whether the curve can be predicted
well enough to bank any of it, via a two-stage decomposition:

    y[c,h] / scale[c]  ~  nat_r[h]  x  pair_factor[c,h]

Stage 1 predicts the film's national ratio from film-level features only.
Stage 2 learns the pair's multiplicative deviation from it. Both stages are
fitted inside the same film-grouped folds, so nothing leaks.
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
occ_of, sh_of = {}, {}
with open(D + "train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d = dt.date.fromisoformat(r["date_show"]); m = r["movie_title"]
        c = r["cinema_ids"]; t = int(r["total_ticket"])
        cell[(m, c, d)] = t
        occ_of[(m, c, d)] = float(r["occupation_rate"])
        sh_of[(m, c, d)] = int(r["total_show"])
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

HOL = {}
with open(D + "holidays.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        HOL[dt.date.fromisoformat(r["date"])] = (r["day_tipe"], r["holiday_tipe"])


def is_hol(d):
    return HOL.get(d, ("weekday", "normal"))[1] == "holiday"


def med(v):
    v = sorted(v); n = len(v)
    return 0.0 if n == 0 else (v[n // 2] if n % 2 else .5 * (v[n // 2 - 1] + v[n // 2]))


def scbin(x):
    for i, b in enumerate((5, 10, 25, 50, 100, 250, 600)):
        if x <= b:
            return i
    return 7


def tbin(x):
    for i, b in enumerate((0.5, 0.75, 0.95, 1.15, 1.5)):
        if x <= b:
            return i
    return 5


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
            tgt = [d1 + dt.timedelta(days=i) for i in range(3, 10)]
            cs = set()
            for dd in obs:
                cs.update(mdc.get((m, dd), ()))
            keep = []
            for c in cs:
                s = [cell.get((m, c, dd), 0) for dd in obs]
                if sum(s) == 0 or s[2] == 0:
                    continue
                keep.append((c, s, [cell.get((m, c, dd), 0) for dd in tgt],
                             [occ_of.get((m, c, dd), 0.0) for dd in obs],
                             [sh_of.get((m, c, dd), 0) for dd in obs]))
            if not keep:
                continue
            n123 = [sum(s[i] for _, s, _, _, _ in keep) for i in range(3)]
            N_obs = sum(n123)
            film[(m, d1)] = (
                N_obs,
                [sum(y[h] for _, _, y, _, _ in keep) for h in range(7)],
                n123, len(keep),
                sum(sum(sh) for _, _, _, _, sh in keep),
                sum(sum(o) for _, _, _, o, _ in keep) / (3 * len(keep)),
            )
            for c, s, y, o, sh in keep:
                pairs.append((m, d1, c, s, y, o, sh))
    return pairs, film


TR, TRF = build(tuple(range(25)))
EV, EVF = build((0,), TEST_DOW)
print(f"train pairs {len(TR):,} ({len(TRF):,} film-windows) | "
      f"proxy pairs {len(EV):,} ({len(EVF):,} film-windows)")

films = sorted({p[0] for p in EV})
random.Random(SEED).shuffle(films)
FOLD = {m: i % 5 for i, m in enumerate(films)}


def rec(p):
    m, d1, c, s, y, o, sh = p
    sc = max(sum(s) / 3.0, 1.0)
    return (m, d1, c, sc, "".join("1" if x > 0 else "0" for x in s),
            scbin(sc), s, y)


TRr, EVr = [rec(p) for p in TR], [rec(p) for p in EV]
N = 7 * len(EVr)


# ======================================================= stage 1: film curve
def film_feats(key, F):
    N_obs, N_tgt, n123, npairs, shows, occ = F[key]
    m, d1 = key
    mu = max(N_obs / 3.0, 1.0)
    trend = n123[2] / max(n123[0], 1)
    return (d1.weekday(), tbin(trend),
            scbin(N_obs / 1000.0), min(npairs // 20, 5),
            1 if occ > 20 else 0)


def f_keys(ff, h):
    dow, tr, sz, npb, occb = ff
    return [(h, dow, tr, sz), (h, dow, tr), (h, dow, sz), (h, dow), (h, tr), (h,)]


def fit_stage1(keys_list, F, min_n=8):
    acc = [defaultdict(list) for _ in range(6)]
    for key in keys_list:
        N_obs, N_tgt, *_ = F[key]
        mu = max(N_obs / 3.0, 1.0)
        ff = film_feats(key, F)
        for h in range(7):
            v = N_tgt[h] / mu
            for j, k in enumerate(f_keys(ff, h)):
                acc[j][k].append(v)
    return ([{k: med(v) for k, v in a.items() if len(v) >= min_n} for a in acc],
            {h: med([F[k][1][h] / max(F[k][0] / 3.0, 1.0) for k in keys_list])
             for h in range(7)})


def pred_stage1(mdl, ff, h):
    tabs, g = mdl
    for t, k in zip(tabs, f_keys(ff, h)):
        if k in t:
            return t[k]
    return g[h]


# ======================================================= stage 2: pair factor
def p_keys(r, h):
    return [(h, r[1].weekday(), r[4], r[5]), (h, r[1].weekday(), r[4]),
            (h, r[4], r[5]), (h, r[1].weekday()), (h, r[4]), (h,)]


def fit_stage2(rows, F, s1, min_n=40, ratio=True):
    """If ratio, learn y/(scale*nat_r); else learn y/scale (= baseline)."""
    acc = [defaultdict(list) for _ in range(6)]
    for r in rows:
        ff = film_feats((r[0], r[1]), F)
        for h in range(7):
            t = r[7][h] / r[3]
            if ratio:
                nr = pred_stage1(s1, ff, h)
                t = t / nr if nr > 1e-6 else 0.0
            for j, k in enumerate(p_keys(r, h)):
                acc[j][k].append(t)
    base = []
    for r in rows:
        for h in range(7):
            base.append(r[7][h] / r[3])
    return ([{k: med(v) for k, v in a.items() if len(v) >= min_n} for a in acc],
            med(base))


def pred_stage2(mdl, r, h):
    tabs, g = mdl
    for t, k in zip(tabs, p_keys(r, h)):
        if k in t:
            return t[k]
    return g


def run(mode):
    """mode: 'baseline' | 'oracle' | 'two_stage'"""
    tot = 0.0
    for f in range(5):
        tr_rows = [r for r in TRr if FOLD.get(r[0], -1) != f]
        tr_keys = [k for k in TRF if FOLD.get(k[0], -1) != f]
        s1 = fit_stage1(tr_keys, TRF)
        s2 = fit_stage2(tr_rows, TRF, s1, ratio=(mode != "baseline"))
        for i, r in enumerate(EVr):
            if FOLD.get(r[0], -1) != f:
                continue
            ff = film_feats((r[0], r[1]), EVF)
            for h in range(7):
                if mode == "baseline":
                    p = pred_stage2(s2, r, h)
                else:
                    if mode == "oracle":
                        N_obs, N_tgt, *_ = EVF[(r[0], r[1])]
                        nr = N_tgt[h] / max(N_obs / 3.0, 1.0)
                    else:
                        nr = pred_stage1(s1, ff, h)
                    p = max(0.0, pred_stage2(s2, r, h) * nr)
                tot += abs(r[7][h] / r[3] - p)
    return tot / N


print()
print("=" * 78)
print("TWO-STAGE DECOMPOSITION")
print("=" * 78)
b = run("baseline")
o = run("oracle")
t = run("two_stage")
print(f"  {'single-stage baseline':44s} {b:.4f}")
print(f"  {'two-stage, stage 1 ORACLE (ceiling)':44s} {o:.4f}  "
      f"gain {b - o:+.4f}")
print(f"  {'two-stage, stage 1 PREDICTED':44s} {t:.4f}  "
      f"gain {b - t:+.4f}")
print(f"\n  share of the ceiling actually captured: "
      f"{100 * (b - t) / max(b - o, 1e-9):.1f}%")

# how well is the film curve predicted at all?
print()
print("=" * 78)
print("STAGE-1 ACCURACY (film national ratio, out of fold)")
print("=" * 78)
err, errb = defaultdict(list), defaultdict(list)
for f in range(5):
    tr_keys = [k for k in TRF if FOLD.get(k[0], -1) != f]
    s1 = fit_stage1(tr_keys, TRF)
    gl = {h: med([TRF[k][1][h] / max(TRF[k][0] / 3.0, 1.0) for k in tr_keys])
          for h in range(7)}
    for k in EVF:
        if FOLD.get(k[0], -1) != f:
            continue
        N_obs, N_tgt, *_ = EVF[k]
        mu = max(N_obs / 3.0, 1.0)
        ff = film_feats(k, EVF)
        for h in range(7):
            tv = N_tgt[h] / mu
            err[h].append(abs(tv - pred_stage1(s1, ff, h)))
            errb[h].append(abs(tv - gl[h]))
print(f"  {'D':>4s} {'MAE(model)':>11s} {'MAE(global med)':>16s} {'skill':>8s}")
for h in range(7):
    a, c = sum(err[h]) / len(err[h]), sum(errb[h]) / len(errb[h])
    print(f"  D{h + 4:<3d} {a:11.4f} {c:16.4f} {100 * (c - a) / c:+7.1f}%")

print()
print("=" * 78)
print("WHERE THE TWO-STAGE GAIN LANDS (per D1-dow x horizon)")
print("=" * 78)
for mode in ("baseline", "two_stage"):
    cells = defaultdict(lambda: defaultdict(list))
    for f in range(5):
        tr_rows = [r for r in TRr if FOLD.get(r[0], -1) != f]
        tr_keys = [k for k in TRF if FOLD.get(k[0], -1) != f]
        s1 = fit_stage1(tr_keys, TRF)
        s2 = fit_stage2(tr_rows, TRF, s1, ratio=(mode != "baseline"))
        for r in EVr:
            if FOLD.get(r[0], -1) != f:
                continue
            ff = film_feats((r[0], r[1]), EVF)
            for h in range(7):
                p = (pred_stage2(s2, r, h) if mode == "baseline"
                     else max(0.0, pred_stage2(s2, r, h) * pred_stage1(s1, ff, h)))
                cells[r[1].weekday()][h].append(abs(r[7][h] / r[3] - p))
    print(f"  {mode}:")
    for w in sorted(cells):
        row = "".join(f"{sum(cells[w][h]) / len(cells[w][h]):8.3f}"
                      for h in range(7))
        print(f"    {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][w]:5s}{row}")
