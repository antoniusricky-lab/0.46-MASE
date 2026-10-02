"""The partial-activity fix.

Official scale = max(sum(s1,s2,s3)/3, 1). When a pair was not showing on D1
and/or D2, that average is taken over days with no screenings, so `scale`
UNDERSTATES the pair's real daily run-rate by a factor of 3/n_active.

The notebook predicts  y_hat = ratio_model * scale, forcing the tree to learn
that 3/n_active factor internally from very few rows.

Fix: predict on a RATE basis -- y_hat = g(...) * rate, where
     rate = sum(s) / n_active_days  -- and let `scale` stay purely as the
metric denominator. Everything else identical.
"""
import csv
import datetime as dt
from collections import defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TEST_DOW = {2, 3, 4}
SEED = 2026


def load(p):
    with open(D + p, newline="", encoding="utf-8") as f:
        return [(dt.date.fromisoformat(r["date_show"]), r["cinema_ids"],
                 r["movie_title"], int(r["total_ticket"])) for r in csv.DictReader(f)]


tr = load("train.csv")
TMIN, TMAX = min(x[0] for x in tr), max(x[0] for x in tr)
cell, ncl, nat, mdc = {}, defaultdict(lambda: defaultdict(int)), \
    defaultdict(lambda: defaultdict(int)), defaultdict(list)
for d, c, m, t in tr:
    cell[(m, c, d)] = t
    ncl[m][d] += 1
    nat[m][d] += t
    mdc[(m, d)].append(c)
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

hol = {}
with open(D + "holidays.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        hol[dt.date.fromisoformat(r["date"])] = (r["day_tipe"], r["holiday_tipe"])


def nonwork(d):
    dy, ht = hol.get(d, ("weekday", "normal"))
    return 1 if (ht == "holiday" or dy == "weekend") else 0


def build(lags, require_d3=True, dow_filter=None):
    out = []
    for m in sorted(release):
        for lag in lags:
            d1 = release[m] + dt.timedelta(days=lag)
            if d1 + dt.timedelta(days=9) > TMAX:
                continue
            if dow_filter and d1.weekday() not in dow_filter:
                continue
            obs = [d1 + dt.timedelta(days=i) for i in range(3)]
            tgt = [d1 + dt.timedelta(days=i) for i in range(3, 10)]
            cs = set()
            for dd in obs:
                cs.update(mdc.get((m, dd), ()))
            for c in cs:
                s = [cell.get((m, c, dd), 0) for dd in obs]
                if sum(s) == 0 or (require_d3 and s[2] == 0):
                    continue
                out.append((m, c, d1, s, [cell.get((m, c, dd), 0) for dd in tgt]))
    return out


def med(v):
    v = sorted(v)
    n = len(v)
    return 0.0 if n == 0 else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def wmed(pairs):
    """Weighted median: pairs of (value, weight). L1-optimal under weights."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return 0.0
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2:
            return v
    return pairs[-1][0]


def scbin(x):
    for i, b in enumerate((5, 10, 25, 50, 100, 250, 600)):
        if x <= b:
            return i
    return 7


def prep(rows):
    """Materialise rows with both bases precomputed."""
    out = []
    for (m, c, d1, s, y) in rows:
        na = sum(1 for x in s if x > 0)
        sc = max(sum(s) / 3.0, 1.0)
        rate = max(sum(s) / na, 1.0)
        p = "".join("1" if x > 0 else "0" for x in s)
        out.append((m, d1.weekday(), p, na, sc, rate, scbin(sc),
                    [nonwork(d1 + dt.timedelta(days=3 + h)) for h in range(7)], y))
    return out


def folds(rows, k=5):
    films = sorted({r[0] for r in rows})
    st = SEED
    for i in range(len(films) - 1, 0, -1):
        st = (1103515245 * st + 12345) % (1 << 31)
        j = st % (i + 1)
        films[i], films[j] = films[j], films[i]
    a = {m: i % k for i, m in enumerate(films)}
    return a


# ---------------------------------------------------------------- models
def keys_scale(r, h):
    _, w, p, na, sc, rate, sb, nw, y = r
    return [(h, w, p, sb), (h, w, p), (h, p, sb), (h, w), (h, p), (h,)]


def keys_rate(r, h):
    _, w, p, na, sc, rate, sb, nw, y = r
    rb = scbin(rate)
    return [(h, w, na, rb), (h, w, na), (h, na, rb), (h, w), (h, na), (h,)]


def fit(rows, keyfn, basis, min_n=40, weighted=False):
    tabs = [defaultdict(list) for _ in range(6)]
    for r in rows:
        b = r[4] if basis == "scale" else r[5]
        sc = r[4]
        for h in range(7):
            v = r[8][h] / b
            w = b / sc            # weight so the fit minimises MASE, not MAE/basis
            for i, k in enumerate(keyfn(r, h)):
                tabs[i][k].append((v, w) if weighted else v)
    out = []
    for t in tabs:
        if weighted:
            out.append({k: wmed(v) for k, v in t.items() if len(v) >= min_n})
        else:
            out.append({k: med(v) for k, v in t.items() if len(v) >= min_n})
    return out


def pred(model, keyfn, r, h):
    for t, k in zip(model, keyfn(r, h)):
        if k in t:
            return t[k]
    return 0.0


def run(train_rows, eval_rows, assign, keyfn, basis, weighted=False, k=5):
    tot = n = 0.0
    per = defaultdict(list)
    for f in range(k):
        sub = [r for r in train_rows if assign.get(r[0], -1) != f]
        mdl = fit(sub, keyfn, basis, weighted=weighted)
        for r in eval_rows:
            if assign.get(r[0], -1) != f:
                continue
            b = r[4] if basis == "scale" else r[5]
            sc = r[4]
            for h in range(7):
                e = abs(r[8][h] - pred(mdl, keyfn, r, h) * b) / sc
                tot += e
                n += 1
                per[r[2]].append(e)
    return tot / n, per


EVAL = prep(build((0,), True, TEST_DOW))
assign = folds(EVAL)
print(f"evaluation population: {len(EVAL)} pairs, {7*len(EVAL)} rows, "
      f"{len({r[0] for r in EVAL})} films")

POPS = {
    "lag 0 only          ": prep(build((0,), True, TEST_DOW)),
    "lag 0-6             ": prep(build(tuple(range(7)), True, TEST_DOW)),
    "lag 0-24            ": prep(build(tuple(range(25)), True, TEST_DOW)),
    "lag 0-24, any D1-dow": prep(build(tuple(range(25)), True, None)),
}

print()
print("=" * 86)
print("BASIS COMPARISON:  predict ratio x scale   vs   ratio x run-rate")
print("=" * 86)
print(f"  {'training population':22s} {'scale basis':>13s} {'rate basis':>13s} "
      f"{'rate+wmedian':>13s}")
best = None
for lab, pop in POPS.items():
    a, _ = run(pop, EVAL, assign, keys_scale, "scale")
    b, pb = run(pop, EVAL, assign, keys_rate, "rate")
    c, pc = run(pop, EVAL, assign, keys_rate, "rate", weighted=True)
    print(f"  {lab:22s} {a:13.4f} {b:13.4f} {c:13.4f}")
    for v, p in ((a, None), (b, pb), (c, pc)):
        if best is None or v < best[0]:
            best = (v, lab, p)

print()
print("=" * 86)
print("PER-PATTERN MASE: scale basis vs rate basis  (best training population)")
print("=" * 86)
pop = POPS["lag 0-24, any D1-dow"]
_, ps = run(pop, EVAL, assign, keys_scale, "scale")
_, pr = run(pop, EVAL, assign, keys_rate, "rate", weighted=True)
print(f"  {'pattern':>8s} {'rows':>7s} {'scale basis':>13s} {'rate basis':>13s} {'delta':>9s}")
tot_s = tot_r = n_all = 0.0
for p in sorted(ps, reverse=True):
    a = sum(ps[p]) / len(ps[p])
    b = sum(pr[p]) / len(pr[p])
    print(f"  {p:>8s} {len(ps[p]):7d} {a:13.4f} {b:13.4f} {b-a:+9.4f}")
    tot_s += sum(ps[p]); tot_r += sum(pr[p]); n_all += len(ps[p])
print(f"  {'TOTAL':>8s} {int(n_all):7d} {tot_s/n_all:13.4f} {tot_r/n_all:13.4f} "
      f"{(tot_r-tot_s)/n_all:+9.4f}")

print()
print("=" * 86)
print("CONTRIBUTION TO TOTAL MASE (rate basis)")
print("=" * 86)
for p in sorted(pr, reverse=True):
    sh = len(pr[p]) / n_all
    print(f"  {p:>8s} share={sh:6.2%}  MASE={sum(pr[p])/len(pr[p]):7.4f}  "
          f"contributes {sum(pr[p])/n_all:7.4f}")
