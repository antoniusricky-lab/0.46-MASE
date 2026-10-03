# %% [markdown]
# # Cinema ticket D4-D10 forecast - corrected pipeline
#
# Rebuild of the 0.46641 submission with the fixes from `FINDINGS.md`.
#
# **Design note.** Every data-handling and feature-engineering step below uses
# only the Python standard library. `numpy` and `lightgbm` are touched in one
# place (the training cell). That is deliberate: it keeps the pipeline
# inspectable, and it means the whole thing still runs - with a weaker fallback
# model - if LightGBM is unavailable.
#
# What changed versus the 0.46641 notebook:
#
# | # | Fix | Finding |
# |---|---|---|
# | 1 | Islamic-calendar features (`days_since_eid`, Ramadan, Xmas) | §1, §2 |
# | 2 | Floor the ratio profile for Idulfitri-window rows | §2 |
# | 3 | D1 = wide-release date, not first transaction | §5 |
# | 4 | Explicit partial-activity (`001`/`011`) features | §6 |
# | 5 | Decay learned per D1-day-of-week, no pooled `DECAY_PRIOR` | §7 |
# | 6 | No regime reweighting - the test set is ~100% opening | §4, §5 |
#
# Outputs `submission.csv` plus `submission_no_eid_guard.csv` so the single
# largest bet (fix 2) can be A/B tested on the leaderboard.

# %%
import csv
import datetime as dt
import math
import os
import random
from collections import Counter, defaultdict

SEED = 2026
random.seed(SEED)

DATA_DIR = "."                 # folder holding the competition CSVs
ANCHOR_LAGS = tuple(range(25))  # 0-24: LightGBM regressed on 0-12
TEST_D1_DOW = {2, 3, 4}         # Wed/Thu/Fri = 96% of test films
N_FOLDS = 5
LABEL_CLIP_Q = 0.999
EID_GUARD = True                # fix 2
EID_GUARD_MODE = "flat"         # "calendar" | "flat" | "none"
EID_GUARD_K = 1.0               # calendar: floor = K * uplift ; flat: floor = K

print(f"SEED={SEED}  ANCHOR_LAGS={ANCHOR_LAGS[0]}..{ANCHOR_LAGS[-1]}  "
      f"folds={N_FOLDS}  eid_guard={EID_GUARD} ({EID_GUARD_MODE} K={EID_GUARD_K})")


def rd(name):
    with open(os.path.join(DATA_DIR, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def D(s):
    return dt.date.fromisoformat(s)


train_raw = rd("train.csv")
hist_raw = rd("test_history.csv")
test_raw = rd("test.csv")
movies_raw = rd("movies.csv")
hol_raw = rd("holidays.csv")
price_raw = rd("ticket_prices.csv")
sub_raw = rd("sample_submission.csv")

print(f"train {len(train_raw):,} | test_history {len(hist_raw):,} | "
      f"test {len(test_raw):,} | movies {len(movies_raw)} | "
      f"holidays {len(hol_raw)} | prices {len(price_raw)} | "
      f"sample_sub {len(sub_raw):,}")

# %% [markdown]
# ## 1. Index the transaction tables
#
# `CELL_TR` / `CELL_TE` map `(movie, cluster, date) -> (tickets, occupancy,
# shows)`. Training windows read from train.csv, test windows from
# test_history.csv, so the two never mix.

# %%
CELL_TR, CELL_TE = {}, {}
NCL_TR, NAT_TR = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
MDC_TR, MDC_TE = defaultdict(list), defaultdict(list)
NAT_DAY = defaultdict(int)          # national tickets per date (train only)

for r in train_raw:
    d, m, c = D(r["date_show"]), r["movie_title"], r["cinema_ids"]
    t = int(r["total_ticket"])
    CELL_TR[(m, c, d)] = (t, float(r["occupation_rate"]), int(r["total_show"]))
    NCL_TR[m][d] += 1
    NAT_TR[m][d] += t
    MDC_TR[(m, d)].append(c)
    NAT_DAY[d] += t

for r in hist_raw:
    d, m, c = D(r["date_show"]), r["movie_title"], r["cinema_ids"]
    CELL_TE[(m, c, d)] = (int(r["total_ticket"]), float(r["occupation_rate"]),
                          int(r["total_show"]))
    MDC_TE[(m, d)].append(c)

CITY_OF = {}
for r in train_raw:
    CITY_OF[r["cinema_ids"]] = r["city_name"]
for r in hist_raw:
    CITY_OF.setdefault(r["cinema_ids"], r["city_name"])

TMIN, TMAX = min(NAT_DAY), max(NAT_DAY)
print(f"train dates {TMIN} .. {TMAX}")
print(f"clusters: train {len({r['cinema_ids'] for r in train_raw})} | "
      f"test {len({r['cinema_ids'] for r in test_raw})}")

# %% [markdown]
# ## 2. Fix 3 - D1 is the wide-release date
#
# FINDINGS §5: 44% of train films first appear as a 1-4 cluster sneak preview
# 1-9 days before the real wide release. Using that as the anchor origin
# mislabels the whole regime axis.
#
# Definition: the first day inside the film's first 10 days that reaches >=25%
# of its peak cluster count in that span.
#
# For test films this reduces to `min(date)` in test_history by construction,
# since only three days are given.

# %%
_first = {m: min(v) for m, v in NAT_TR.items()}
LEFT_CENSORED = {m for m in _first if _first[m] == TMIN}

RELEASE_TR = {}
for m, f0 in _first.items():
    if m in LEFT_CENSORED:
        continue                      # already running when train starts
    wk = [d for d in sorted(NAT_TR[m]) if d <= f0 + dt.timedelta(days=9)]
    peak = max(NCL_TR[m][d] for d in wk)
    for d in wk:
        if NCL_TR[m][d] >= 0.25 * peak:
            RELEASE_TR[m] = d
            break

D1_TE = {}
for r in hist_raw:
    m, d = r["movie_title"], D(r["date_show"])
    if m not in D1_TE or d < D1_TE[m]:
        D1_TE[m] = d

_shift = Counter((RELEASE_TR[m] - _first[m]).days for m in RELEASE_TR)
_dow_first = Counter(_first[m].weekday() for m in RELEASE_TR)
_dow_rel = Counter(RELEASE_TR[m].weekday() for m in RELEASE_TR)
DOWN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print(f"train films {len(_first)} | left-censored {len(LEFT_CENSORED)} | "
      f"usable {len(RELEASE_TR)}")
print(f"shift (release - first appearance) in days: {dict(sorted(_shift.items()))}")
print(f"{'dow':6s}{'first-appearance':>18s}{'wide-release':>14s}")
for w in range(7):
    print(f"{DOWN[w]:6s}{_dow_first[w]:18d}{_dow_rel[w]:14d}")
print("-> release dow should now be Wed/Thu/Fri heavy, matching the test set")
print(f"test D1 dow: {dict(sorted(Counter(d.weekday() for d in D1_TE.values()).items()))}")

# %% [markdown]
# ## 3. Verify the two proven structural rules
#
# FINDINGS §3: a pair is forecast **iff** it sold a ticket on D3.
# FINDINGS §4: D1 is a weekly release slot, so the test set is ~100% opening
# windows. If either assertion fails, the rest of this notebook's reasoning
# does not apply to your files.

# %%
_act = defaultdict(set)
for r in hist_raw:
    m, c, d = r["movie_title"], r["cinema_ids"], D(r["date_show"])
    off = (d - D1_TE[m]).days
    if 0 <= off <= 2:
        _act[(m, c)].add(off)
_test_pairs = {(r["movie_title"], r["cinema_ids"]) for r in test_raw}
_active_d3 = {p for p, s in _act.items() if 2 in s}

print(f"pairs with D1-D3 history      : {len(_act)}")
print(f"pairs active on D3            : {len(_active_d3)}")
print(f"pairs in test.csv             : {len(_test_pairs)}")
print(f"active-on-D3 == test.csv pairs: {_active_d3 == _test_pairs}")
assert _active_d3 == _test_pairs, "the D3 selection rule no longer holds"
assert not (_test_pairs - set(_act)), "some test pair has no D1-D3 history"

_d1dates = set(D1_TE.values())
print(f"distinct D1 dates             : {len(_d1dates)} for {len(D1_TE)} films")
assert len(_d1dates) < len(D1_TE) / 2, "D1 does not look like a weekly release slot"

_offs = Counter((D(r["date_show"]) - D1_TE[r["movie_title"]]).days for r in test_raw)
print(f"test offsets (expect 3..9 = D4..D10): {dict(sorted(_offs.items()))}")
assert set(_offs) == {3, 4, 5, 6, 7, 8, 9}
print("all structural assertions passed")

# %% [markdown]
# ## 4. Fix 1 - calendar features including the Islamic calendar
#
# FINDINGS §1: MASE on calendar-anomaly days is 0.849 versus 0.359 on ordinary
# days (2.37x). Anomalies are 7.1% of train rows but **31.8% of test rows**.
# Train contains no Ramadan at all and `holidays.csv` never labels it, so
# 18.2% of the test set currently has no feature that can represent it.
#
# `days_since_eid` is the bridge that *is* learnable: April 2025 is the
# Idulfitri 1446 aftermath, giving 800+ train rows for each of Eid+1..Eid+13 -
# which covers 6 of the 7 target days of the 2026-03-18 cohort.
#
# Eid dates are the first day of Syawal as observed in Indonesia. They are
# hardcoded because `holidays.csv` starts on 2025-04-01 and therefore only
# labels Eid+1 for 1446 H, which would misalign the two years by a day.

# %%
EID_DATES = [dt.date(2025, 3, 31), dt.date(2026, 3, 21)]
RAMADAN_LEN = 30

HOL = {}
for r in hol_raw:
    HOL[D(r["date"])] = (r["day_tipe"], r["holiday_tipe"], r["holiday_name"] or "")


def is_hol(d):
    return HOL.get(d, ("weekday", "normal", ""))[1] == "holiday"


def is_nonwork(d):
    dy, ht, _ = HOL.get(d, ("weekday", "normal", ""))
    return 1 if (ht == "holiday" or dy == "weekend") else 0


# ---- detrended national day-of-week / holiday factors (train only) ----------
_days = sorted(NAT_DAY)
_roll = {}
for i, d in enumerate(_days):
    lo, hi = max(0, i - 3), min(len(_days), i + 4)
    win = [NAT_DAY[x] for x in _days[lo:hi]]
    _roll[d] = sum(win) / len(win)


def _median(v):
    v = sorted(v)
    n = len(v)
    return 0.0 if n == 0 else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


_rat = {d: NAT_DAY[d] / _roll[d] for d in _days if _roll[d] > 0}
DOW_FAC = {}
for w in range(7):
    v = [_rat[d] for d in _rat if d.weekday() == w and not is_hol(d)]
    DOW_FAC[w] = _median(v) if v else 1.0
_hv = [_rat[d] / DOW_FAC[d.weekday()] for d in _rat if is_hol(d)]
HOL_MULT = _median(_hv) if _hv else 1.0
print("detrended dow factors:", {DOWN[w]: round(DOW_FAC[w], 3) for w in range(7)})
print(f"holiday multiplier   : {HOL_MULT:.3f}")


def cal_factor(d):
    f = DOW_FAC[d.weekday()]
    return f * HOL_MULT if is_hol(d) else f


# ---- per-date calendar bundle, precomputed once -----------------------------
_ALL = sorted({d for d in HOL} | set(NAT_DAY) |
              {D1_TE[m] + dt.timedelta(days=i) for m in D1_TE for i in range(10)} |
              {D(r["date_show"]) for r in test_raw})
_LO, _HI = min(_ALL) - dt.timedelta(days=40), max(_ALL) + dt.timedelta(days=40)
_SPAN = [_LO + dt.timedelta(days=i) for i in range((_HI - _LO).days + 1)]
_HOLSET = sorted(d for d in _SPAN if is_hol(d))


def _near_hol(d):
    nxt = min((x - d).days for x in _HOLSET if x >= d) if any(x >= d for x in _HOLSET) else 99
    prv = min((d - x).days for x in _HOLSET if x <= d) if any(x <= d for x in _HOLSET) else 99
    return min(nxt, 30), min(prv, 30)


_nw = {d: is_nonwork(d) for d in _SPAN}
_runlen = {}
for d in _SPAN:
    if not _nw[d]:
        _runlen[d] = 0
        continue
    n, x = 0, d
    while x in _nw and _nw[x]:
        n += 1
        x -= dt.timedelta(days=1)
    m2, x = 0, d + dt.timedelta(days=1)
    while x in _nw and _nw[x]:
        m2 += 1
        x += dt.timedelta(days=1)
    _runlen[d] = n + m2

CAL = {}
for d in _SPAN:
    signed = min(((d - e).days for e in EID_DATES), key=abs)
    in_eid = 1 if 0 <= signed <= 13 else 0
    in_ram = 1 if -RAMADAN_LEN <= signed < 0 else 0
    nxt, prv = _near_hol(d)
    in_xmas = 1 if ((d.month == 12 and d.day >= 20) or
                    (d.month == 1 and d.day <= 4)) else 0
    d2x = (25 - d.day) if d.month == 12 else (-(d.day + 6) if d.month == 1 else 99)
    CAL[d] = (
        d.weekday(),                              # 0 t_dow
        1 if d.weekday() >= 5 else 0,             # 1 is_weekend
        1 if d.weekday() == 4 else 0,             # 2 is_friday
        1 if is_hol(d) else 0,                    # 3 is_hol
        _nw[d],                                   # 4 is_nonwork
        _runlen[d],                               # 5 nonwork_run
        nxt,                                      # 6 days_to_next_hol
        prv,                                      # 7 days_from_prev_hol
        max(-45, min(45, signed)),                # 8 days_since_eid
        in_eid,                                   # 9 is_eid_window
        (signed // 7 + 1) if in_eid else 0,       # 10 eid_week
        in_ram,                                   # 11 in_ramadan
        (signed + RAMADAN_LEN + 1) if in_ram else 0,   # 12 ramadan_day
        1 if (in_ram and signed >= -10) else 0,   # 13 ramadan_late
        in_xmas,                                  # 14 in_xmas_ny
        max(-20, min(99, d2x)),                   # 15 days_to_xmas
        cal_factor(d),                            # 16 cal_factor
    )


def regime_of(d):
    c = CAL[d]
    if c[9]:
        return 1          # eid window
    if c[11]:
        return 2          # ramadan
    if c[14]:
        return 3          # xmas / NY
    return 0


_tc = Counter(regime_of(D(r["date_show"])) for r in test_raw)
_tot = len(test_raw)
print("\ntest rows by calendar regime:")
for k, lab in ((0, "ordinary"), (2, "ramadan"), (1, "eid window"), (3, "xmas/NY")):
    print(f"  {lab:12s} {_tc[k]:7d} ({100 * _tc[k] / _tot:5.1f}%)")

# %% [markdown]
# ## 5. Static reference tables
#
# Cluster behaviour is always taken from `train.csv` (clusters are shared
# between periods). Film metadata, city prices and title format flags come from
# the reference files.

# %%
_cl_daily = defaultdict(list)
_cl_shows = defaultdict(list)
_cl_occ = defaultdict(list)
_cl_mov = defaultdict(set)
for (m, c, d), (t, o, s) in CELL_TR.items():
    _cl_daily[c].append(t)
    _cl_shows[c].append(s)
    _cl_occ[c].append(o)
    _cl_mov[c].add(m)

CL = {}
for c, v in _cl_daily.items():
    v2 = sorted(v)
    CL[c] = (
        sum(v) / len(v),                       # mean daily
        _median(v),                            # median daily
        v2[int(0.9 * (len(v2) - 1))],          # p90 daily
        sum(_cl_shows[c]) / len(_cl_shows[c]),
        len(_cl_mov[c]),
        sum(_cl_occ[c]) / len(_cl_occ[c]),
        sum(v) / max(sum(_cl_shows[c]), 1),
    )
_glob_cl = tuple(_median([CL[c][i] for c in CL]) for i in range(7))
print(f"cluster stats for {len(CL)} clusters | fallback = global median")

PRICE = defaultdict(dict)
for r in price_raw:
    PRICE[r["city_name"]][r["price_day"]] = float(r["ceil"])
_pg = (_median([PRICE[c].get("Weekday", 0) for c in PRICE]),
       _median([PRICE[c].get("Weekend", 0) for c in PRICE]))

MOV = {}
_genres, _ages = {}, {}
for r in movies_raw:
    g = (r["genre"] or "").split(",")[0].strip()
    a = (r["age_rating"] or "").strip()
    _genres.setdefault(g, len(_genres))
    _ages.setdefault(a, len(_ages))
    MOV[r["original_title"].strip().upper()] = (
        _genres[g], _ages[a],
        len([x for x in (r["genre"] or "").split(",") if x.strip()]),
        len([x for x in (r["casts"] or "").split(",") if x.strip()]),
    )

CINE_CODE, CITY_CODE = {}, {}
for c in sorted(CITY_OF):
    CINE_CODE[c] = len(CINE_CODE)
for c in sorted(set(CITY_OF.values())):
    CITY_CODE[c] = len(CITY_CODE)


def title_feats(m):
    u = m.upper()
    base = u
    for tag in ("(IMAX 2D)", "(IMAX)", "(3D)", "(2D)", "(DUB)", "(DUBBING)",
                "(RE-RELEASE)", "(PREMIUM)"):
        base = base.replace(tag, "")
    base = " ".join(base.split())
    meta = MOV.get(base, None)
    if meta is None:
        meta = MOV.get(u, None)
    return (
        1 if "IMAX" in u else 0,
        1 if "3D" in u else 0,
        1 if ("DUB" in u) else 0,
        1 if "PREMIUM" in u else 0,
        len(m), len(m.split()),
        meta if meta else (-1, -1, 0, 0),
        1 if meta else 0,
    )


TITLE = {}
for m in set(list(NAT_TR) + list(D1_TE)):
    TITLE[m] = title_feats(m)
print(f"title metadata matched for "
      f"{sum(1 for m in TITLE if TITLE[m][7])}/{len(TITLE)} films")

# %% [markdown]
# ## 6. Window construction
#
# One window per (film, anchor lag). Three observation days, seven target days.
# Pairs are kept only when they sold a ticket on D3, replicating the proven
# test rule exactly (FINDINGS §3).
#
# Training uses lags 0..24. Restricting to lag 0 was **tested and is worse**
# (0.4024 vs 0.3932) - volume wins, so the multi-anchor idea from the original
# notebook is kept.

# %%
def build_windows(films, d1_of, cell, mdc, lags, require_d3=True, dow_filter=None):
    """-> list of (film, cluster, d1, s[3], occ[3], sh[3], y[7] or None)"""
    out = []
    for m in sorted(films):
        for lag in lags:
            d1 = d1_of[m] + dt.timedelta(days=lag)
            if dow_filter and d1.weekday() not in dow_filter:
                continue
            obs = [d1 + dt.timedelta(days=i) for i in range(3)]
            if cell is CELL_TR and obs[0] + dt.timedelta(days=9) > TMAX:
                continue
            cs = set()
            for dd in obs:
                cs.update(mdc.get((m, dd), ()))
            for c in cs:
                tri = [cell.get((m, c, dd), (0, 0.0, 0)) for dd in obs]
                s = [x[0] for x in tri]
                if sum(s) == 0 or (require_d3 and s[2] == 0):
                    continue
                y = None
                if cell is CELL_TR:
                    y = [cell.get((m, c, d1 + dt.timedelta(days=i)), (0, 0.0, 0))[0]
                         for i in range(3, 10)]
                out.append((m, c, d1, s, [x[1] for x in tri], [x[2] for x in tri], y))
    return out


TRAIN_W = build_windows(RELEASE_TR, RELEASE_TR, CELL_TR, MDC_TR, ANCHOR_LAGS)
PROXY_W = build_windows(RELEASE_TR, RELEASE_TR, CELL_TR, MDC_TR, (0,),
                        dow_filter=TEST_D1_DOW)
TEST_W = build_windows(D1_TE, D1_TE, CELL_TE, MDC_TE, (0,))

print(f"training windows : {len(TRAIN_W):,} pairs -> {7 * len(TRAIN_W):,} rows")
print(f"proxy  windows   : {len(PROXY_W):,} pairs -> {7 * len(PROXY_W):,} rows "
      f"(release windows, Wed/Thu/Fri, s3>0 = the honest test stand-in)")
print(f"test   windows   : {len(TEST_W):,} pairs -> {7 * len(TEST_W):,} rows")
assert 7 * len(TEST_W) == len(test_raw), (
    f"test windows {7 * len(TEST_W)} != test rows {len(test_raw)}")

_proxy_keys = {(w[0], w[1], w[2]) for w in PROXY_W}

# %% [markdown]
# ## 7. Film-level aggregates over the observation window
#
# Computed from whichever history the window came from, so train and test
# definitions stay identical.

# %%
def scbin(x):
    for i, b in enumerate((5, 10, 25, 50, 100, 250, 600)):
        if x <= b:
            return i
    return 7


def film_aggs(windows):
    acc = defaultdict(lambda: [0, 0, 0, 0, 0.0, 0])   # s1,s2,s3,npairs,occ,shows
    for (m, c, d1, s, occ, sh, y) in windows:
        a = acc[(m, d1)]
        a[0] += s[0]
        a[1] += s[1]
        a[2] += s[2]
        a[3] += 1
        a[4] += sum(occ)
        a[5] += sum(sh)
    return acc


def pair_ranks(windows):
    by = defaultdict(list)
    for i, (m, c, d1, s, occ, sh, y) in enumerate(windows):
        by[(m, d1)].append((sum(s), i))
    rank = [0.0] * len(windows)
    for k, v in by.items():
        v.sort()
        n = len(v)
        for j, (_, i) in enumerate(v):
            rank[i] = (j + 1) / n
    return rank


AGG_TRAIN, RANK_TRAIN = film_aggs(TRAIN_W), pair_ranks(TRAIN_W)
AGG_TEST, RANK_TEST = film_aggs(TEST_W), pair_ranks(TEST_W)
print(f"film-window aggregates: train {len(AGG_TRAIN):,} | test {len(AGG_TEST):,}")

# Folds are assigned HERE, before any feature is built, because the film-curve
# feature below is a target encoding and must be computed out of fold.
_uniq = sorted({w[0] for w in TRAIN_W})
random.Random(SEED).shuffle(_uniq)
FOLD_OF = {m: i % N_FOLDS for i, m in enumerate(_uniq)}
print(f"film folds assigned for {len(FOLD_OF)} films")


# ===================================================== stage 1: film curve
# Oracle study (analysis/oracles.py): knowing a film's own national D4-D10
# ratio curve - with NO pair-level information - scores 0.3039 against a
# 0.3944 baseline, the largest single lever found. analysis/two_stage.py
# banks +0.0116 of that with a crude 5-feature median table, so the curve is
# partly predictable. Exposed here as a FEATURE rather than a hard
# multiplicative decomposition, which let the tree ignore it where the
# decomposition regressed (Wednesday D4).
FILM_TGT = {}
for (m, c, d1, s, occ, sh, y) in TRAIN_W:
    if y is None:
        continue
    t = FILM_TGT.setdefault((m, d1), [0] * 7)
    for h in range(7):
        t[h] += y[h]


def _tbin(x):
    for i, b in enumerate((0.5, 0.75, 0.95, 1.15, 1.5)):
        if x <= b:
            return i
    return 5


def film_key(key, agg):
    m, d1 = key
    a = agg[key]
    n_obs = max(a[0] + a[1] + a[2], 1)
    trend = a[2] / max(a[0], 1)
    return (d1.weekday(), _tbin(trend), scbin(n_obs / 1000.0),
            min(a[3] // 20, 5))


def _fc_keys(fk, h):
    dow, tr, sz, npb = fk
    return [(h, dow, tr, sz), (h, dow, tr), (h, dow, sz), (h, dow), (h, tr), (h,)]


def fit_film_curve(keys, agg, min_n=8):
    acc = [defaultdict(list) for _ in range(6)]
    glob = defaultdict(list)
    for key in keys:
        a = agg[key]
        mu = max((a[0] + a[1] + a[2]) / 3.0, 1.0)
        fk = film_key(key, agg)
        for h in range(7):
            v = FILM_TGT[key][h] / mu
            glob[h].append(v)
            for j, k in enumerate(_fc_keys(fk, h)):
                acc[j][k].append(v)
    return ([{k: _median(v) for k, v in a.items() if len(v) >= min_n}
             for a in acc],
            {h: _median(glob[h]) for h in range(7)})


def pred_film_curve(mdl, fk, h):
    tabs, g = mdl
    for t, k in zip(tabs, _fc_keys(fk, h)):
        if k in t:
            return t[k]
    return g[h]


_all_keys = [k for k in AGG_TRAIN if k in FILM_TGT]
FC = {}
for f in range(N_FOLDS):                      # out-of-fold for training rows
    mdl = fit_film_curve([k for k in _all_keys if FOLD_OF.get(k[0], -1) != f],
                         AGG_TRAIN)
    for k in _all_keys:
        if FOLD_OF.get(k[0], -1) == f:
            fk = film_key(k, AGG_TRAIN)
            for h in range(7):
                FC[(k[0], k[1], h)] = pred_film_curve(mdl, fk, h)
_full = fit_film_curve(_all_keys, AGG_TRAIN)  # full fit for test rows
for k in AGG_TEST:
    fk = film_key(k, AGG_TEST)
    for h in range(7):
        FC[(k[0], k[1], h)] = pred_film_curve(_full, fk, h)
print(f"film-curve feature: {len(FC):,} (film, window, horizon) entries")
print("  predicted national ratio by horizon (test windows):")
for h in range(7):
    v = [FC[(k[0], k[1], h)] for k in AGG_TEST]
    print(f"    D{h + 4:<3d} median {_median(v):.3f}")

# %% [markdown]
# ## 8. Feature builder
#
# Fix 4 (FINDINGS §6): the `001` / `011` activity patterns are 8% of rows but
# 25% of the error, because `scale` averages over days the pair was not
# screening. `scale_deflation`, `run_rate` and `last_over_scale` hand the model
# that correction directly instead of making a tree rediscover it.
#
# Fix 5 (FINDINGS §7): no pooled `DECAY_PRIOR`. `d1_dow` is a feature and
# `off` is a feature, so the tree learns a separate decay per release weekday -
# the median ratio at D4 is 1.114 for Wednesday releases but 0.454 for Friday
# ones, a 2.5x spread the single pooled curve destroyed.

# %%
PAIR_NAMES = [
    "s1", "s2", "s3", "occ1", "occ2", "occ3", "sh1", "sh2", "sh3",
    "sum3", "scale", "log_scale",
    "n_active", "act_pattern", "starts_late", "only_d3", "has_gap",
    "run_rate", "scale_deflation", "rate_over_scale", "last_level", "last_over_scale",
    "r1", "r2", "r3", "slope31", "lr21", "lr32", "lr31",
    "max3_s", "min3_s", "rng3_s", "argmax3", "r3_over_max",
    "tickets_per_show", "occ_mean", "occ_last", "occ_trend", "sh_trend", "sh_last",
    "nat_sum3", "nat_npairs", "nat_log_scale", "nat_r1", "nat_r2", "nat_r3",
    "nat_slope31", "pair_share", "pair_rank_pct",
    "cl_mean_daily", "cl_med_daily", "cl_p90_daily", "cl_mean_shows",
    "cl_n_movies", "cl_occ_mean", "cl_tickets_per_show",
    "scale_vs_cluster", "log_scale_vs_cluster", "occ_vs_cluster",
    "cinema_code", "city_code",
    "d1_dow", "d1_days_since_eid", "d1_in_ramadan",
    "n_nonwork_obs", "n_hol_obs", "n_ramadan_obs", "n_eid_obs", "n_xmas_obs",
    "base_cal_factor",
    "genre_code", "age_code", "n_genres", "n_casts", "meta_matched",
    "fmt_imax", "fmt_3d", "fmt_dub", "fmt_premium",
    "price_weekday", "price_weekend", "title_len", "title_words",
]
ROW_NAMES = [
    "off", "t_dow", "is_weekend", "is_friday", "is_holiday", "is_nonwork",
    "nonwork_run", "days_to_next_hol", "days_from_prev_hol",
    "days_since_eid", "is_eid_window", "eid_week",
    "in_ramadan", "ramadan_day", "ramadan_late", "in_xmas_ny", "days_to_xmas",
    "cal_factor", "uplift", "regime_change", "film_curve",
]
FEATS = PAIR_NAMES + ROW_NAMES
CAT_FEATS = ["cinema_code", "city_code", "d1_dow", "t_dow", "genre_code",
             "age_code", "act_pattern"]
CAT_IDX = [FEATS.index(c) for c in CAT_FEATS]
print(f"{len(FEATS)} features ({len(PAIR_NAMES)} pair-level, "
      f"{len(ROW_NAMES)} row-level), {len(CAT_IDX)} categorical")


def _lg(x):
    return math.log1p(max(x, 0.0))


def pair_features(w, agg, rank, idx):
    m, c, d1, s, occ, sh, y = w
    sum3 = float(sum(s))
    scale = max(sum3 / 3.0, 1.0)
    act = [1 if x > 0 else 0 for x in s]
    n_act = sum(act)
    pattern = act[0] * 4 + act[1] * 2 + act[2]
    rate = sum3 / n_act if n_act else 0.0
    last = s[2] if act[2] else (s[1] if act[1] else s[0])
    mx, mn = float(max(s)), float(min(s))
    shs = float(sum(sh))
    a = agg[(m, d1)]
    nat3 = float(a[0] + a[1] + a[2])
    nat_scale = max(nat3 / 3.0, 1.0)
    cl = CL.get(c, _glob_cl)
    city = CITY_OF.get(c, "")
    pw = PRICE.get(city, {}).get("Weekday", _pg[0])
    pe = PRICE.get(city, {}).get("Weekend", _pg[1])
    ti = TITLE.get(m) or title_feats(m)
    meta = ti[6]
    obs = [d1 + dt.timedelta(days=i) for i in range(3)]
    co = [CAL[d] for d in obs]
    base_cf = sum(x[16] for x in co) / 3.0
    c0 = CAL[d1]
    return [
        float(s[0]), float(s[1]), float(s[2]),
        occ[0], occ[1], occ[2], float(sh[0]), float(sh[1]), float(sh[2]),
        sum3, scale, _lg(scale),
        float(n_act), float(pattern), float(1 - act[0]),
        float(1 - act[0] and 1 - act[1]), float(act[0] and not act[1] and act[2]),
        rate, 3.0 / max(n_act, 1), rate / scale, float(last), last / scale,
        s[0] / scale, s[1] / scale, s[2] / scale, (s[2] - s[0]) / scale,
        _lg(s[1]) - _lg(s[0]), _lg(s[2]) - _lg(s[1]), _lg(s[2]) - _lg(s[0]),
        mx / scale, mn / scale, (mx - mn) / scale, float(s.index(max(s))),
        s[2] / mx if mx > 0 else 0.0,
        sum3 / max(shs, 1.0), sum(occ) / 3.0, occ[2], occ[2] - occ[0],
        float(sh[2] - sh[0]), float(sh[2]),
        nat3, float(a[3]), _lg(nat_scale),
        a[0] / nat_scale, a[1] / nat_scale, a[2] / nat_scale,
        (a[2] - a[0]) / nat_scale,
        sum3 / nat3 if nat3 > 0 else 0.0, rank[idx],
        cl[0], cl[1], float(cl[2]), cl[3], float(cl[4]), cl[5], cl[6],
        scale / max(cl[0], 1e-6), _lg(scale) - _lg(cl[0]),
        (sum(occ) / 3.0) / max(cl[5], 1e-6),
        float(CINE_CODE.get(c, 0)), float(CITY_CODE.get(city, 0)),
        float(d1.weekday()), float(c0[8]), float(c0[11]),
        float(sum(x[4] for x in co)), float(sum(x[3] for x in co)),
        float(sum(x[11] for x in co)), float(sum(x[9] for x in co)),
        float(sum(x[14] for x in co)), base_cf,
        float(meta[0]), float(meta[1]), float(meta[2]), float(meta[3]),
        float(ti[7]),
        float(ti[0]), float(ti[1]), float(ti[2]), float(ti[3]),
        pw, pe, float(ti[4]), float(ti[5]),
    ]


def row_features(d1, h, base_cf, obs_regime, fc=0.0):
    t = d1 + dt.timedelta(days=3 + h)
    c = CAL[t]
    return [
        float(h + 4), float(c[0]), float(c[1]), float(c[2]), float(c[3]),
        float(c[4]), float(c[5]), float(c[6]), float(c[7]),
        float(c[8]), float(c[9]), float(c[10]),
        float(c[11]), float(c[12]), float(c[13]), float(c[14]), float(c[15]),
        c[16], c[16] / max(base_cf, 1e-6),
        float(1 if regime_of(t) != obs_regime else 0), fc,
    ]


_BCF = FEATS.index("base_cal_factor")


def iter_rows(windows, agg, rank):
    """Yield (film, d1, cluster, off, scale, y_or_None, feature_list)."""
    for idx, w in enumerate(windows):
        m, c, d1, s, occ, sh, y = w
        pf = pair_features(w, agg, rank, idx)
        scale = pf[FEATS.index("scale")]
        base_cf = pf[_BCF]
        obs_reg = regime_of(d1 + dt.timedelta(days=1))
        for h in range(7):
            yield (m, d1, c, h, scale,
                   (float(y[h]) if y is not None else None),
                   pf + row_features(d1, h, base_cf, obs_reg,
                                     FC.get((m, d1, h), 0.0)))


_probe = next(iter_rows(TRAIN_W[:1], AGG_TRAIN, RANK_TRAIN))
assert len(_probe[6]) == len(FEATS), f"{len(_probe[6])} values vs {len(FEATS)} names"
print(f"feature vector length verified: {len(_probe[6])}")
print("sample:", {k: round(v, 3) for k, v in list(zip(FEATS, _probe[6]))[:8]})


# %% [markdown]
# ## 9. One streaming pass: row metadata (+ the feature matrix if numpy exists)
#
# `meta` carries everything the calibration, diagnostics, Eid guard and
# submission writer need, as plain Python tuples. Those stages therefore do not
# depend on numpy at all. The dense matrix is built in the same pass, only when
# numpy is available, and rows go straight into a preallocated array so the
# features never exist as Python objects.

# %%
try:
    import gc

    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

I_OFF = FEATS.index("off")
I_PAT = FEATS.index("act_pattern")
I_D1DOW = FEATS.index("d1_dow")
I_EID = FEATS.index("is_eid_window")
I_RAM = FEATS.index("in_ramadan")
I_XMAS = FEATS.index("in_xmas_ny")
I_REGCH = FEATS.index("regime_change")
I_SCALE = FEATS.index("scale")

I_NEIDOBS = FEATS.index("n_eid_obs")

# meta tuple layout
M_FILM, M_D1, M_CLU, M_OFF, M_SCALE, M_Y = 0, 1, 2, 3, 4, 5
M_PAT, M_EID, M_RAM, M_XMAS, M_REGCH, M_DOW, M_PROXY = 6, 7, 8, 9, 10, 11, 12
M_OBSEID = 13

FB_COLS = ["off", "d1_dow", "act_pattern", "scale", "is_eid_window",
           "in_ramadan", "in_xmas_ny", "n_active", "last_over_scale",
           "uplift", "cal_factor", "n_eid_obs"]
FB_IDX = [FEATS.index(c) for c in FB_COLS]


def stream(windows, agg, rank, release_of=None, labeled=True, want_matrix=True):
    """Single pass -> (X or small column dict, meta list)."""
    n, nf = 7 * len(windows), len(FEATS)
    X = np.empty((n, nf), dtype=np.float32) if (HAS_NP and want_matrix) else None
    cols = {c: [0.0] * n for c in FB_COLS} if X is None else None
    meta = [None] * n
    for i, (m, d1, c, h, scale, y, fv) in enumerate(iter_rows(windows, agg, rank)):
        if X is not None:
            X[i] = fv
        else:
            for c2, j in zip(FB_COLS, FB_IDX):
                cols[c2][i] = fv[j]
        proxy = (release_of is not None and release_of.get(m) == d1
                 and d1.weekday() in TEST_D1_DOW)
        meta[i] = (m, d1, c, int(fv[I_OFF]), scale,
                   (y / scale if labeled else None),
                   int(fv[I_PAT]), int(fv[I_EID]), int(fv[I_RAM]),
                   int(fv[I_XMAS]), int(fv[I_REGCH]), int(fv[I_D1DOW]),
                   1 if proxy else 0, int(fv[I_NEIDOBS]))
    return (X if X is not None else cols), meta


print("streaming training rows ...")
TR_X, TR_META = stream(TRAIN_W, AGG_TRAIN, RANK_TRAIN, release_of=RELEASE_TR)
print("streaming test rows ...")
TE_X, TE_META = stream(TEST_W, AGG_TEST, RANK_TEST, labeled=False)

def col(X, name, i):
    """Read one feature from either the numpy matrix or the fallback dict."""
    return float(X[i, FEATS.index(name)]) if HAS_NP else X[name][i]


PROXY_MASK = [r[M_PROXY] == 1 for r in TR_META]
Y_TR = [r[M_Y] for r in TR_META]
print(f"training rows {len(TR_META):,} | proxy rows {sum(PROXY_MASK):,} "
      f"| test rows {len(TE_META):,}")
assert len(TE_META) == len(test_raw)

_ys = sorted(Y_TR)
Y_CLIP = _ys[min(int(LABEL_CLIP_Q * len(_ys)), len(_ys) - 1)]
Y_TR_C = [min(v, Y_CLIP) for v in Y_TR]
print(f"y/scale: mean {sum(Y_TR) / len(Y_TR):.3f} median {_ys[len(_ys) // 2]:.3f} "
      f"zeros {100 * sum(1 for v in Y_TR if v == 0) / len(Y_TR):.1f}%")
print(f"label clip at q{LABEL_CLIP_Q} = {Y_CLIP:.2f} "
      f"({sum(1 for v in Y_TR if v > Y_CLIP):,} rows affected)")
if HAS_NP:
    assert np.isfinite(TR_X).all(), "non-finite value in the training matrix"
    assert np.isfinite(TE_X).all(), "non-finite value in the test matrix"
    print(f"matrices: train {TR_X.shape} test {TE_X.shape} (finite check passed)")

# %% [markdown]
# ## 10. Model
#
# L1 objective on the ratio target, because MASE is exactly MAE on `y/scale`.
# Folds are grouped by film, so no film is ever split across a fold boundary.
#
# Fix 6: **no regime reweighting.** The 0.46641 notebook reweighted training
# toward an estimated 13.7% "opening" mix and then chose its feature set and
# blend weights by that weighted MASE. FINDINGS §4 proves the test set is
# ~100% opening windows, so the weighting optimised the wrong quantity.
#
# If LightGBM is missing the notebook falls back to a hierarchical median
# lookup on `(off, d1_dow, pattern, scale bin)`. That scores about 0.393 on the
# proxy population, so it still produces a usable submission - just a weaker
# one than the boosted model.

# %%
LGB_PARAMS = dict(
    objective="l1", metric="l1",
    learning_rate=0.04, num_leaves=160, min_data_in_leaf=120,
    feature_fraction=0.70, bagging_fraction=0.80, bagging_freq=1,
    lambda_l2=3.0, max_bin=255, cat_smooth=20, min_data_per_group=100,
    num_threads=-1, verbosity=-1, force_row_wise=True, deterministic=True,
    seed=SEED, bagging_seed=SEED + 1, feature_fraction_seed=SEED + 2,
    data_random_seed=SEED + 3, extra_seed=SEED + 4,
)
NUM_ROUNDS, EARLY_STOP = 3000, 150
# Cause 3 of the 0.47789 regression: calibration was fitted on single-model
# out-of-fold predictions but applied to a seed-averaged refit. Use ONE seed
# list for both so the two distributions match.
MODEL_SEEDS = [SEED, SEED + 101]
CAT_IDX_LGB = CAT_IDX

try:
    import lightgbm as lgb
    HAS_LGB = HAS_NP
except ImportError:
    HAS_LGB = False
print(f"numpy {HAS_NP} | lightgbm {HAS_LGB}"
      f"{'' if HAS_LGB else '  -> using the median-table fallback'}")


def quantile(v, q):
    v = sorted(v)
    if not v:
        return 0.0
    return v[min(int(q * (len(v) - 1)), len(v) - 1)]


def film_folds(meta, k):
    uniq = sorted({r[M_FILM] for r in meta})
    rng = random.Random(SEED)
    rng.shuffle(uniq)
    return {m: i % k for i, m in enumerate(uniq)}


FOLD = [FOLD_OF[r[M_FILM]] for r in TR_META]


# ---------------------------------------------------------------- fallback
def _fb_keys(r):
    sb = scbin(r[M_SCALE])
    return [(r[M_OFF], r[M_DOW], r[M_PAT], sb),
            (r[M_OFF], r[M_DOW], r[M_PAT]),
            (r[M_OFF], r[M_PAT], sb),
            (r[M_OFF], r[M_DOW]),
            (r[M_OFF], r[M_PAT]),
            (r[M_OFF],)]


def fb_fit(meta, y, keep, min_n=40):
    acc = [defaultdict(list) for _ in range(6)]
    allv = []
    for i, r in enumerate(meta):
        if not keep[i]:
            continue
        allv.append(y[i])
        for j, k in enumerate(_fb_keys(r)):
            acc[j][k].append(y[i])
    tabs = [{k: _median(v) for k, v in a.items() if len(v) >= min_n} for a in acc]
    return tabs, (_median(allv) if allv else 0.0)


def fb_predict(model, r):
    tabs, glob = model
    for t, k in zip(tabs, _fb_keys(r)):
        if k in t:
            return t[k]
    return glob


# ---------------------------------------------------------------- lightgbm
def lgb_fit(Xa, ya, Xb, yb, seed=SEED):
    params = dict(LGB_PARAMS, seed=seed, bagging_seed=seed + 1,
                  feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    dtr = lgb.Dataset(Xa, label=ya, categorical_feature=CAT_IDX_LGB,
                      feature_name=list(FEATS), free_raw_data=False)
    dva = lgb.Dataset(Xb, label=yb, reference=dtr)
    try:
        mdl = lgb.train(params, dtr, num_boost_round=NUM_ROUNDS,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                   lgb.log_evaluation(0)])
    except (TypeError, AttributeError):
        mdl = lgb.train(params, dtr, num_boost_round=NUM_ROUNDS,
                        valid_sets=[dva], early_stopping_rounds=EARLY_STOP,
                        verbose_eval=False)
    return mdl, (getattr(mdl, "best_iteration", None) or NUM_ROUNDS)


def mase(y, pred, mask):
    tot = n = 0
    for i, m in enumerate(mask):
        if m:
            tot += abs(y[i] - pred[i])
            n += 1
    return tot / n if n else float("nan")


OOF = [0.0] * len(TR_META)
ITERS = []
for f in range(N_FOLDS):
    keep = [x != f for x in FOLD]
    if HAS_LGB:
        tr_i = np.flatnonzero(np.array(FOLD) != f)
        va_i = np.flatnonzero(np.array(FOLD) == f)
        ya = np.asarray(Y_TR_C, dtype=np.float64)
        pv = np.zeros(len(va_i))
        it = 0
        mdl_keep = None
        for sd in MODEL_SEEDS:
            mdl, it_s = lgb_fit(TR_X[tr_i], ya[tr_i], TR_X[va_i], ya[va_i], sd)
            pv += np.clip(mdl.predict(TR_X[va_i], num_iteration=it_s), 0,
                          None) / len(MODEL_SEEDS)
            it = max(it, it_s)
            mdl_keep = mdl if mdl_keep is None else mdl_keep
            if sd != MODEL_SEEDS[0]:
                del mdl
        for j, i in enumerate(va_i):
            OOF[int(i)] = float(pv[j])
        ITERS.append(it)
        if f == 0:
            LAST_IMP = list(zip(FEATS, mdl_keep.feature_importance("gain")))
    else:
        mdl = fb_fit(TR_META, Y_TR_C, keep)
        it = 0
        for i, r in enumerate(TR_META):
            if FOLD[i] == f:
                OOF[i] = fb_predict(mdl, r)
    vm = [FOLD[i] == f and PROXY_MASK[i] for i in range(len(TR_META))]
    print(f"  fold {f + 1}/{N_FOLDS}: iter={it:4d} "
          f"proxy MASE={mase(Y_TR, OOF, vm):.4f} (n={sum(vm):,})")

LAST_IMP = globals().get("LAST_IMP")
OOF_ALL = mase(Y_TR, OOF, [True] * len(TR_META))
OOF_PROXY = mase(Y_TR, OOF, PROXY_MASK)
print(f"\nOOF MASE, all anchors : {OOF_ALL:.4f}")
print(f"OOF MASE, proxy only  : {OOF_PROXY:.4f}  <- the honest estimate")
if ITERS:
    print(f"mean best_iteration   : {sum(ITERS) / len(ITERS):.0f}")

# %% [markdown]
# ## 10b. Model Z - two-part zero model
#
# Oracle B (`analysis/oracles.py`) says perfect knowledge of which rows are
# zero is worth **+0.0595** - the second largest lever after the film curve.
# The per-horizon quantile snap in cell 12 is a crude proxy for it; this is
# the real thing, and it is what the 0.46641 notebook's model Z did at blend
# weight 0.75.
#
# Two heads on the same feature matrix:
#
# * a binary classifier for `P(y_ratio == 0)`
# * an L1 regressor fitted on the **positive rows only**
#
# Combined through a per-decile multiplier on `p0`, fitted on out-of-fold
# MASE. Under L1 the optimal action is to predict 0 once `P(zero) > 0.5`, and
# the multiplier learns that boundary from data rather than assuming it.
#
# The blend weight against the main model is also fitted on the proxy rows.

# %%
PARAMS_CLF = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.05, num_leaves=127, min_data_in_leaf=150,
    feature_fraction=0.70, bagging_fraction=0.80, bagging_freq=1,
    lambda_l2=3.0, num_threads=-1, verbosity=-1, force_row_wise=True,
    deterministic=True, seed=SEED + 303, bagging_seed=SEED + 304,
    feature_fraction_seed=SEED + 305, data_random_seed=SEED + 306,
)
PARAMS_POS = dict(LGB_PARAMS, seed=SEED + 404, bagging_seed=SEED + 405,
                  feature_fraction_seed=SEED + 406, data_random_seed=SEED + 407)
RUN_Z = True


def lgb_fit_generic(params, Xa, ya, Xb, yb, rounds=NUM_ROUNDS):
    dtr = lgb.Dataset(Xa, label=ya, categorical_feature=CAT_IDX_LGB,
                      feature_name=list(FEATS), free_raw_data=False)
    dva = lgb.Dataset(Xb, label=yb, reference=dtr)
    try:
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(0)])
    except (TypeError, AttributeError):
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    return m, (getattr(m, "best_iteration", None) or rounds)


def fit_decile_mult(p0, pos, y, mask):
    """Per-decile multiplier on the positive prediction, fitted on MASE."""
    idx = [i for i, m in enumerate(mask) if m]
    if not idx:
        return {d: 1.0 for d in range(10)}
    edges = [quantile([p0[i] for i in idx], q / 10.0) for q in range(1, 10)]

    def dec(v):
        d = 0
        for e in edges:
            if v > e:
                d += 1
        return d

    out = {}
    for d in range(10):
        sel = [i for i in idx if dec(p0[i]) == d]
        if not sel:
            out[d] = 1.0
            continue
        best, bv = 1.0, None
        for k100 in range(0, 141, 5):
            k = k100 / 100.0
            v = sum(abs(y[i] - pos[i] * k) for i in sel) / len(sel)
            if bv is None or v < bv:
                best, bv = k, v
        out[d] = best
    return out, edges, dec


def fb_fit_zero(meta, y, keep, min_n=40):
    """P(y == 0) as a hierarchical frequency table (fallback classifier)."""
    acc = [defaultdict(lambda: [0, 0]) for _ in range(6)]
    n0 = n = 0
    for i, r in enumerate(meta):
        if not keep[i]:
            continue
        z = 1 if y[i] == 0 else 0
        n0 += z
        n += 1
        for j, k in enumerate(_fb_keys(r)):
            acc[j][k][0] += z
            acc[j][k][1] += 1
    tabs = [{k: v[0] / v[1] for k, v in a.items() if v[1] >= min_n}
            for a in acc]
    return tabs, (n0 / n if n else 0.0)


def fb_fit_pos(meta, y, keep, min_n=20):
    """Median of the POSITIVE rows only (fallback positive regressor)."""
    acc = [defaultdict(list) for _ in range(6)]
    allv = []
    for i, r in enumerate(meta):
        if not keep[i] or y[i] <= 0:
            continue
        allv.append(y[i])
        for j, k in enumerate(_fb_keys(r)):
            acc[j][k].append(y[i])
    tabs = [{k: _median(v) for k, v in a.items() if len(v) >= min_n}
            for a in acc]
    return tabs, (_median(allv) if allv else 0.0)


OOF_MAIN = list(OOF)
OOF_Z = [0.0] * len(TR_META)
ITERS_CLF, ITERS_POS = [], []
Z_OK = False

if RUN_Z and not HAS_LGB:
    # Fallback path: same two-part structure, hierarchical tables instead of
    # boosted trees. Exercises the decile multiplier, the blend search and the
    # refit combination so that plumbing is not shipped untested.
    P0 = [0.0] * len(TR_META)
    POS = [0.0] * len(TR_META)
    for f in range(N_FOLDS):
        keep = [x != f for x in FOLD]
        mz = fb_fit_zero(TR_META, Y_TR_C, keep)
        mp = fb_fit_pos(TR_META, Y_TR_C, keep)
        for i, r in enumerate(TR_META):
            if FOLD[i] != f:
                continue
            P0[i] = fb_predict(mz, r)
            POS[i] = fb_predict(mp, r)
    TP_MULT, TP_EDGES, TP_DEC = fit_decile_mult(P0, POS, Y_TR, PROXY_MASK)
    print("  [Z fallback] per-decile multipliers on p0:")
    print(f"    { {d: TP_MULT[d] for d in range(10)} }")
    OOF_Z = [max(0.0, POS[i] * TP_MULT[TP_DEC(P0[i])])
             for i in range(len(TR_META))]
    print(f"  [Z fallback] proxy MASE = {mase(Y_TR, OOF_Z, PROXY_MASK):.4f}  "
          f"(main {OOF_PROXY:.4f})")
    Z_OK = True

if RUN_Z and HAS_LGB:
    P0 = [0.0] * len(TR_META)
    POS = [0.0] * len(TR_META)
    ya = np.asarray(Y_TR_C, dtype=np.float64)
    is_zero = (ya == 0).astype(np.int8)
    fa = np.array(FOLD)
    sd = MODEL_SEEDS[0]
    for f in range(N_FOLDS):
        tr_i = np.flatnonzero(fa != f)
        va_i = np.flatnonzero(fa == f)
        mc, ic = lgb_fit_generic(PARAMS_CLF, TR_X[tr_i], is_zero[tr_i],
                                 TR_X[va_i], is_zero[va_i])
        pv = np.clip(mc.predict(TR_X[va_i], num_iteration=ic), 0, 1)
        for j, i in enumerate(va_i):
            P0[int(i)] = float(pv[j])
        ITERS_CLF.append(ic)
        del mc
        pos_tr = tr_i[ya[tr_i] > 0]
        pos_va = va_i[ya[va_i] > 0]
        if len(pos_tr) < 500 or len(pos_va) < 50:
            for i in va_i:
                POS[int(i)] = OOF_MAIN[int(i)]
            continue
        mr, ir = lgb_fit_generic(PARAMS_POS, TR_X[pos_tr], ya[pos_tr],
                                 TR_X[pos_va], ya[pos_va])
        rv = np.clip(mr.predict(TR_X[va_i], num_iteration=ir), 0, None)
        for j, i in enumerate(va_i):
            POS[int(i)] = float(rv[j])
        ITERS_POS.append(ir)
        del mr
        gc.collect()
        print(f"  [Z] fold {f + 1}/{N_FOLDS}: clf iter={ic} pos iter={ir}")

    auc_rows = [(P0[i], 1 if Y_TR[i] == 0 else 0) for i in range(len(TR_META))
                if PROXY_MASK[i]]
    _pos = sum(1 for _, t in auc_rows if t)
    print(f"\n  zero-rate on proxy rows: {_pos / len(auc_rows):.1%}")
    TP_MULT, TP_EDGES, TP_DEC = fit_decile_mult(P0, POS, Y_TR, PROXY_MASK)
    print(f"  per-decile multipliers on p0 (0 = least likely zero):")
    print(f"    { {d: TP_MULT[d] for d in range(10)} }")
    OOF_Z = [max(0.0, POS[i] * TP_MULT[TP_DEC(P0[i])])
             for i in range(len(TR_META))]
    print(f"  [Z] proxy MASE = {mase(Y_TR, OOF_Z, PROXY_MASK):.4f}  "
          f"(main model {OOF_PROXY:.4f})")
    Z_OK = True

if Z_OK:
    best_w, bv = 1.0, None
    for w100 in range(0, 101, 5):
        w = w100 / 100.0
        cand = [w * OOF_MAIN[i] + (1 - w) * OOF_Z[i] for i in range(len(TR_META))]
        v = mase(Y_TR, cand, PROXY_MASK)
        if bv is None or v < bv:
            best_w, bv = w, v
    BLEND_W = best_w
    print("  blend curve (main weight -> proxy MASE):")
    for w100 in range(0, 101, 10):
        w = w100 / 100.0
        cand = [w * OOF_MAIN[i] + (1 - w) * OOF_Z[i] for i in range(len(TR_META))]
        mark = "  <- chosen" if abs(w - BLEND_W) < 1e-9 else ""
        print(f"    main={w:.1f}  {mase(Y_TR, cand, PROXY_MASK):.4f}{mark}")
    if BLEND_W < 0.10:
        print("  NOTE: the blend discards the main model. If the curve above is")
        print("  flat below main=0.2, prefer a small non-zero weight for")
        print("  diversification - the proxy has only 144 films.")
    OOF = [BLEND_W * OOF_MAIN[i] + (1 - BLEND_W) * OOF_Z[i]
           for i in range(len(TR_META))]
    OOF_PROXY = mase(Y_TR, OOF, PROXY_MASK)
    print(f"\n  blend weight main={BLEND_W:.2f} Z={1 - BLEND_W:.2f}"
          f"  -> proxy MASE {OOF_PROXY:.4f}")
else:
    BLEND_W = 1.0
    print("model Z skipped (needs LightGBM); using the main model alone")

# %% [markdown]
# ## 11. The diagnostics that actually matter
#
# Overall OOF hid the problem that cost the 0.46641 submission its score, so
# the breakdown by calendar regime and by activity pattern is printed
# explicitly. Compare the regime rows against FINDINGS §1 (ordinary 0.359 vs
# anomaly 0.849) and the pattern rows against §6.

# %%
def report(pred, title):
    print(f"--- {title} ---")

    def g(sel, lab):
        n = sum(sel)
        if n:
            print(f"  {lab:30s} n={n:7d}  MASE={mase(Y_TR, pred, sel):.4f}")

    P = PROXY_MASK
    anom = [P[i] and (r[M_EID] or r[M_RAM] or r[M_XMAS])
            for i, r in enumerate(TR_META)]
    ordn = [P[i] and not (r[M_EID] or r[M_RAM] or r[M_XMAS])
            for i, r in enumerate(TR_META)]
    print(" by calendar regime:")
    g(ordn, "ordinary target day")
    g(anom, "anomaly target day")
    g([P[i] and r[M_EID] == 1 for i, r in enumerate(TR_META)], "  eid window")
    g([P[i] and r[M_XMAS] == 1 for i, r in enumerate(TR_META)], "  xmas / NY")
    g([P[i] and r[M_REGCH] == 1 for i, r in enumerate(TR_META)],
      "regime changes mid-window")
    print(" by activity pattern:")
    for code, lab in ((7, "111"), (5, "101"), (3, "011"), (1, "001")):
        g([P[i] and r[M_PAT] == code for i, r in enumerate(TR_META)], lab)
    print(" by horizon:")
    for h in range(4, 11):
        g([P[i] and r[M_OFF] == h for i, r in enumerate(TR_META)], f"D{h}")
    print(" by D1 day-of-week:")
    for w in sorted(TEST_D1_DOW):
        g([P[i] and r[M_DOW] == w for i, r in enumerate(TR_META)], DOWN[w])


report(OOF, "OOF, uncalibrated")

if LAST_IMP:
    tot = sum(g for _, g in LAST_IMP) or 1.0
    top = sorted(LAST_IMP, key=lambda x: -x[1])[:20]
    print("\ntop 20 features by gain (fold 1):")
    for nm, g in top:
        print(f"  {nm:24s} {100 * g / tot:6.2f}%")
    fc = dict(LAST_IMP).get("film_curve", 0.0)
    rank = 1 + sum(1 for _, g in LAST_IMP if g > fc)
    print(f"\n  film_curve: {100 * fc / tot:.2f}% of gain, rank {rank}"
          f" of {len(LAST_IMP)}")
    print("  -> if this is near zero the stage-1 curve is not being used and")
    print("     cell 9's film-curve block needs richer features.")

# %% [markdown]
# ## 12. Post-hoc calibration
#
# **Rewritten after the first leaderboard result (0.47789 vs 0.46641).**
# The failure was here, not in the features. Cell 11 was healthy -
# ordinary-day MASE 0.3175 and pattern `111` at 0.2662 both beat the previous
# submission - but the far horizons shipped at roughly the conditional MEAN
# when MASE is L1 and rewards the MEDIAN:
#
# | horizon | true median | shipped | the 0.46641 notebook |
# |---|---|---|---|
# | D8  | 0.117 | 0.401 | 0.183 |
# | D9  | 0.000 | 0.375 | 0.142 |
# | D10 | 0.000 | 0.358 | 0.132 |
#
# That alone accounts for roughly +0.034 on a constant-predictor proxy,
# against a measured gap of +0.016. Three causes, all here:
#
# 1. the multiplier grid ran 0.80..1.30, and D9/D10 **pinned to the lower
#    bound** - the optimum was outside the grid
# 2. the zero-snap was a single global threshold searched over 0..0.40, but
#    LightGBM emits ~0.4-0.5 at D8-D10, so it snapped almost nothing: 21-23%
#    zeros against a true 46-59%
# 3. calibration was fitted on single-model out-of-fold predictions and then
#    applied to a 3-seed average, which is a smoother distribution
#
# The snap is now a **quantile** per horizon rather than an absolute
# threshold. If out-of-fold says 55% of D9 rows should be zero, the lowest 55%
# of D9 test predictions are zeroed - scale-free, so it transfers even when
# the refit model's output distribution shifts. Fix 3 is handled in cell 10 by
# using one seed list for both CV and refit.

# %%
MULT_GRID = [x / 100.0 for x in range(20, 141, 2)]
SNAP_QGRID = [x / 100.0 for x in range(0, 91, 2)]

CAL_MULT, CAL_SNAPQ = {}, {}
for h in range(4, 11):
    idx = [i for i, r in enumerate(TR_META) if PROXY_MASK[i] and r[M_OFF] == h]
    best, bv = 1.0, None
    for mu in MULT_GRID:
        v = sum(abs(Y_TR[i] - OOF[i] * mu) for i in idx) / len(idx)
        if bv is None or v < bv:
            best, bv = mu, v
    CAL_MULT[h] = best
    sp = [OOF[i] * best for i in idx]
    # Cap: never zero out MORE rows than are actually zero. Over-zeroing is
    # nearly free on ordinary rows but catastrophic on Eid rows, which is how
    # submission_no_eid_guard reached 0.52220. Costs ~0.0005 on the proxy.
    true_zero = sum(1 for i in idx if Y_TR[i] == 0) / len(idx)
    bq, bvq = 0.0, None
    for q in SNAP_QGRID:
        if q > true_zero:
            continue
        thr = quantile(sp, q)
        v = sum(abs(Y_TR[i] - (0.0 if OOF[i] * best <= thr else OOF[i] * best))
                for i in idx) / len(idx)
        if bvq is None or v < bvq:
            bq, bvq = q, v
    CAL_SNAPQ[h] = bq

_pin = [h for h in range(4, 11) if CAL_MULT[h] <= MULT_GRID[0] + 1e-9]
print("per-horizon multipliers:",
      {h: round(CAL_MULT[h], 2) for h in range(4, 11)})
print(f"  pinned to the grid floor: {_pin or 'none'}"
      f"{'   <-- WIDEN MULT_GRID' if _pin else ''}")
print("per-horizon zero-snap quantiles (capped at the true zero share):",
      {h: round(CAL_SNAPQ[h], 2) for h in range(4, 11)})


def apply_cal(pred, meta):
    """Multiplier then per-horizon quantile snap, computed within `pred`."""
    out = [max(0.0, pred[i] * CAL_MULT[r[M_OFF]]) for i, r in enumerate(meta)]
    for h in range(4, 11):
        idx = [i for i, r in enumerate(meta) if r[M_OFF] == h]
        if not idx:
            continue
        thr = quantile([out[i] for i in idx], CAL_SNAPQ[h])
        for i in idx:
            if out[i] <= thr:
                out[i] = 0.0
    return out


OOF_C = apply_cal(OOF, TR_META)
print(f"\nproxy MASE  raw {OOF_PROXY:.4f} -> calibrated "
      f"{mase(Y_TR, OOF_C, PROXY_MASK):.4f}")

print("\nzero share by horizon, proxy rows (the thing that was wrong):")
print(f"  {'D':>4s} {'true':>8s} {'predicted':>10s} {'true median':>12s} "
      f"{'pred median':>12s}")
for h in range(4, 11):
    idx = [i for i, r in enumerate(TR_META) if PROXY_MASK[i] and r[M_OFF] == h]
    yv = [Y_TR[i] for i in idx]
    pv = [OOF_C[i] for i in idx]
    print(f"  D{h:<3d} {sum(1 for x in yv if x == 0) / len(yv):8.3f} "
          f"{sum(1 for x in pv if x == 0) / len(pv):10.3f} "
          f"{_median(yv):12.3f} {_median(pv):12.3f}")

report(OOF_C, "OOF, calibrated")

# %% [markdown]
# ## 13. Refit on all rows and predict the test set
#
# Refit on 100% of the training rows, rounds = mean best iteration x 1.1 to
# compensate for the extra data, averaged over three seeds to cut variance.

# %%
PRED = [0.0] * len(TE_META)
if HAS_LGB:
    SEEDS = MODEL_SEEDS
    n_round = max(int(sum(ITERS) / len(ITERS) * 1.1), 200)
    ya = np.asarray(Y_TR_C, dtype=np.float64)
    acc = np.zeros(len(TE_META))
    for sd in SEEDS:
        p = dict(LGB_PARAMS, seed=sd, bagging_seed=sd + 1,
                 feature_fraction_seed=sd + 2, data_random_seed=sd + 3)
        d = lgb.Dataset(TR_X, label=ya, categorical_feature=CAT_IDX_LGB,
                        feature_name=list(FEATS), free_raw_data=False)
        mm = lgb.train(p, d, num_boost_round=n_round)
        acc += np.clip(mm.predict(TE_X), 0, None) / len(SEEDS)
        print(f"  seed {sd}: {n_round} rounds done")
        del d, mm
    PRED_MAIN = [float(x) for x in acc]
    PRED = list(PRED_MAIN)
    if Z_OK:
        nc = max(int(sum(ITERS_CLF) / len(ITERS_CLF) * 1.1), 200)
        nr = (max(int(sum(ITERS_POS) / len(ITERS_POS) * 1.1), 200)
              if ITERS_POS else n_round)
        is_zero_full = (ya == 0).astype(np.int8)
        dz = lgb.Dataset(TR_X, label=is_zero_full,
                         categorical_feature=CAT_IDX_LGB,
                         feature_name=list(FEATS), free_raw_data=False)
        mc = lgb.train(PARAMS_CLF, dz, num_boost_round=nc)
        te_p0 = np.clip(mc.predict(TE_X), 0, 1)
        del dz, mc
        gc.collect()
        pos_idx = np.flatnonzero(ya > 0)
        dz = lgb.Dataset(TR_X[pos_idx], label=ya[pos_idx],
                         categorical_feature=CAT_IDX_LGB,
                         feature_name=list(FEATS), free_raw_data=False)
        mr = lgb.train(PARAMS_POS, dz, num_boost_round=nr)
        te_pos = np.clip(mr.predict(TE_X), 0, None)
        del dz, mr
        gc.collect()
        pz = [max(0.0, float(te_pos[i]) * TP_MULT[TP_DEC(float(te_p0[i]))])
              for i in range(len(TE_META))]
        PRED = [BLEND_W * PRED_MAIN[i] + (1 - BLEND_W) * pz[i]
                for i in range(len(TE_META))]
        print(f"  Z refit: clf {nc} rounds, positive regressor {nr} rounds")
        print(f"  Z predicts exact zero on "
              f"{100 * sum(1 for x in pz if x == 0) / len(pz):.1f}% of test rows")
else:
    allk = [True] * len(TR_META)
    mdl = fb_fit(TR_META, Y_TR_C, allk)
    PRED_MAIN = [fb_predict(mdl, r) for r in TE_META]
    PRED = list(PRED_MAIN)
    if Z_OK:
        mz = fb_fit_zero(TR_META, Y_TR_C, allk)
        mp = fb_fit_pos(TR_META, Y_TR_C, allk)
        pz = [max(0.0, fb_predict(mp, r)
                  * TP_MULT[TP_DEC(fb_predict(mz, r))]) for r in TE_META]
        PRED = [BLEND_W * PRED_MAIN[i] + (1 - BLEND_W) * pz[i]
                for i in range(len(TE_META))]
        print(f"  Z predicts exact zero on "
              f"{100 * sum(1 for x in pz if x == 0) / len(pz):.1f}% of test rows")

PRED = apply_cal(PRED, TE_META)
PRED_RAW = list(PRED)

_byoff = defaultdict(list)
for i, r in enumerate(TE_META):
    _byoff[r[M_OFF]].append(PRED[i])
print("\nmean predicted ratio by horizon:")
for h in range(4, 11):
    v = _byoff[h]
    print(f"  D{h:<3d} mean {sum(v) / len(v):.3f}  "
          f"zeros {100 * sum(1 for x in v if x == 0) / len(v):5.1f}%")

# %% [markdown]
# ## 14. Fix 2 - the Idulfitri guard
#
# FINDINGS §2. Seven films release 2026-03-18 and their **entire** D4-D10
# window is the Idulfitri 1447 H holiday week: 631 pairs, 4,417 rows, 6.08% of
# the test set. It sits at the end of the period, so it is likely concentrated
# in the private leaderboard.
#
# The 0.46641 submission has that cohort decaying from 0.952 to 0.132, while
# the April 2025 analogue (Idulfitri 1446 aftermath) shows the week after Eid
# running at or above the Eid weekend itself. Train has 800+ rows for each of
# Eid+1..Eid+6, which covers six of this cohort's seven target days.
#
# This is a **floor**, not a multiplier - it can only raise rows whose target
# day falls inside an Eid window, and leaves every other row untouched.

# %%
# Rows that straddle INTO an Eid window: target day inside it, observation
# window outside it. That is exactly the 2026-03-18 cohort's situation, and it
# is the only case where `scale` is measured in one demand regime and the
# target in another.
def straddles_eid(r):
    return r[M_EID] == 1 and r[M_OBSEID] == 0


tr_eid = [i for i, r in enumerate(TR_META) if straddles_eid(r)]
print(f"training rows straddling into an Eid window: {len(tr_eid):,}")
print("""
This is expected to be ZERO, and that is the whole difficulty. train.csv
starts 2025-04-01, which is already Eid+1, so no training window can have its
observation days before Eid and its target days after it. The straddle cannot
be fitted - not with more features, not with a bigger model. Fix 1
(`days_since_eid`) teaches the model what an Eid day looks like, but every
training example of one has its `scale` measured inside the Eid window too, so
the model still never sees a pre-Eid denominator.

The floor below is therefore an explicit PRIOR, not a fitted value. The
sensitivity table shows exactly what is being risked.
""".strip())

te_eid = [i for i, r in enumerate(TE_META) if straddles_eid(r)]
print(f"\ntest rows straddling into an Eid window: {len(te_eid):,} "
      f"({100 * len(te_eid) / len(TE_META):.2f}%)")
print("affected films:")
for f in sorted({TE_META[i][M_FILM] for i in te_eid}):
    print(f"  {f}")

# What the calendar features alone already imply for these rows. `uplift` is
# cal_factor(target) / mean cal_factor(D1-D3), built only from train, so it is
# evidence rather than assumption - but it only knows about LABELLED holidays.
# In 2025 the Lebaran block (Apr 2-13) ran at 560-710k tickets/day against a
# ~266k May baseline while holidays.csv labelled none of those days, which is
# why the labelled-holiday uplift understates the real effect.
if te_eid:
    bo = defaultdict(list)
    for i in te_eid:
        bo[TE_META[i][M_OFF]].append(i)
    print(f"\n  {'D':>4s} {'rows':>6s} {'model ratio':>12s} {'calendar uplift':>16s}")
    for h in sorted(bo):
        ii = bo[h]
        up = sum(col(TE_X, "uplift", i) for i in ii) / len(ii)
        print(f"  D{h:<3d} {len(ii):6d} "
              f"{sum(PRED_RAW[i] for i in ii) / len(ii):12.3f} {up:16.3f}")

# ---------------------------------------------------------------- strategies
# A flat floor is crude. `uplift` already encodes the calendar shape from
# train data - 1.95 on the Eid weekend, 0.64-0.93 on the following weekdays -
# so a floor proportional to it raises the collapsed far horizons without
# over-committing on the near ones.
UP = [col(TE_X, "uplift", i) for i in range(len(TE_META))]


def floors_for(mode, k):
    """-> dict row index -> floor, for straddling rows only."""
    if mode == "none":
        return {}
    if mode == "flat":
        return {i: k for i in te_eid}
    return {i: k * UP[i] for i in te_eid}          # "calendar"


STRATS = [("none", 0.0), ("flat", 0.6), ("flat", 0.8), ("flat", 1.0),
          ("flat", 1.2), ("calendar", 0.5), ("calendar", 0.6),
          ("calendar", 0.8), ("calendar", 1.0)]

# Truth scenarios. Flat ones are blunt; the calendar-shaped ones model the film
# holding its level while the calendar drives the day-to-day shape, which is
# what Lebaran week actually looks like.
SCEN = ([(f"flat {t}", [t] * len(TE_META)) for t in (0.3, 0.6, 1.0, 1.5)] +
        [(f"{k}xuplift", [k * u for u in UP]) for k in (0.6, 0.9, 1.2)])


def cost(fl_map, truth):
    """Contribution to TOTAL test MASE from the straddling rows."""
    tot = 0.0
    for i in te_eid:
        tot += abs(truth[i] - max(PRED_RAW[i], fl_map.get(i, 0.0)))
    return tot / len(TE_META)


print(f"\n  cost contribution to TOTAL MASE from the {len(te_eid):,} "
      f"straddling rows ({100 * len(te_eid) / len(TE_META):.2f}% of test)")
hdr = "".join(f"{n:>12s}" for n, _ in SCEN)
print(f"  {'strategy':>18s}{hdr}{'worst vs none':>15s}")
base = None
best_minimax = None
for mode, k in STRATS:
    fm = floors_for(mode, k)
    cells = [cost(fm, tv) for _, tv in SCEN]
    if base is None:
        base = cells
    worst = max(c - b for c, b in zip(cells, base))
    lab = "none" if mode == "none" else f"{mode} {k}"
    print(f"  {lab:>18s}" + "".join(f"{c:12.4f}" for c in cells) +
          f"{worst:+15.4f}")
    if mode != "none" and (best_minimax is None or worst < best_minimax[0]):
        best_minimax = (worst, mode, k)

print(f"\n  lowest worst-case regret: {best_minimax[1]} {best_minimax[2]} "
      f"({best_minimax[0]:+.4f})")
print(f"  configured             : {EID_GUARD_MODE} {EID_GUARD_K}")

PRED_GUARD = list(PRED_RAW)
_fm = floors_for(EID_GUARD_MODE, EID_GUARD_K) if EID_GUARD else {}
for i, fl in _fm.items():
    PRED_GUARD[i] = max(PRED_GUARD[i], fl)
if _fm:
    raised = sum(1 for i in te_eid if PRED_GUARD[i] != PRED_RAW[i])
    print(f"\n  guard applied: mean ratio "
          f"{sum(PRED_RAW[i] for i in te_eid) / len(te_eid):.3f} -> "
          f"{sum(PRED_GUARD[i] for i in te_eid) / len(te_eid):.3f} "
          f"({raised:,} of {len(te_eid):,} rows raised)")
    print(f"  {'D':>4s} {'before':>9s} {'after':>9s} {'floor':>9s}")
    _bo = defaultdict(list)
    for i in te_eid:
        _bo[TE_META[i][M_OFF]].append(i)
    for h in sorted(_bo):
        ii = _bo[h]
        print(f"  D{h:<3d} {sum(PRED_RAW[i] for i in ii) / len(ii):9.3f} "
              f"{sum(PRED_GUARD[i] for i in ii) / len(ii):9.3f} "
              f"{sum(_fm[i] for i in ii) / len(ii):9.3f}")
else:
    print("\n  guard disabled")

# %% [markdown]
# ## 15. Write the submissions
#
# Two files, so the single largest bet can be isolated on the leaderboard:
# `submission.csv` has every fix including the Idulfitri guard,
# `submission_no_eid_guard.csv` has everything except it. Submit both and the
# difference tells you whether the prior was right.

# %%
TEST_ID = {}
for r in test_raw:
    TEST_ID[(r["movie_title"], r["cinema_ids"], D(r["date_show"]))] = r["id"]
SUB_IDS = [r["id"] for r in sub_raw]


def write_sub(path, ratio):
    by_id = {}
    for i, r in enumerate(TE_META):
        k = (r[M_FILM], r[M_CLU], r[M_D1] + dt.timedelta(days=3 + r[M_OFF] - 4))
        rid = TEST_ID.get(k)
        assert rid is not None, f"no test id for {k}"
        by_id[rid] = max(0, int(round(ratio[i] * r[M_SCALE])))
    assert len(by_id) == len(TE_META), "duplicate id collision"
    missing = [i for i in SUB_IDS if i not in by_id]
    assert not missing, f"{len(missing)} sample_submission ids unpredicted"
    out = [(i, by_id[i]) for i in SUB_IDS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "total_ticket"])
        w.writerows(out)
    v = [x[1] for x in out]
    print(f"{path}: {len(out):,} rows | total {sum(v):,} tickets | "
          f"mean {sum(v) / len(v):.1f} | "
          f"zeros {100 * sum(1 for x in v if x == 0) / len(v):.1f}%")
    return out


# The guard floor cannot be fitted (train has no straddling window), so emit
# several and let the leaderboard choose. Writing them all here costs one run
# instead of three.
FLOOR_VARIANTS = [1.0, 1.3, 1.6]
_a = write_sub("submission.csv", PRED_GUARD if EID_GUARD else PRED_RAW)
_b = write_sub("submission_no_eid_guard.csv", PRED_RAW)
_d = sum(1 for x, y in zip(_a, _b) if x[1] != y[1])
print(f"rows differing, main vs no-guard: {_d:,}")

for _fl in FLOOR_VARIANTS:
    if abs(_fl - EID_GUARD_K) < 1e-9 and EID_GUARD_MODE == "flat":
        continue
    _pv = list(PRED_RAW)
    for _i in te_eid:
        _pv[_i] = max(_pv[_i], _fl)
    write_sub(f"submission_floor_{_fl:.1f}.csv", _pv)

print("\nSubmission order, given a 3-per-day limit:")
print("  1. submission.csv              (floor 1.0, the measured-safe choice)")
print("  2. submission_floor_1.3.csv    (tests whether the floor should rise)")
print("  3. hold one back until 1 and 2 are known")

print("\n" + "=" * 70)
print(f"model                      : {'LightGBM' if HAS_LGB else 'median-table fallback'}")
print(f"OOF MASE, all anchors      : {OOF_ALL:.4f}")
print(f"OOF MASE, proxy raw        : {OOF_PROXY:.4f}")
print(f"OOF MASE, proxy calibrated : {mase(Y_TR, OOF_C, PROXY_MASK):.4f}")
print("=" * 70)
print("The proxy population contains NO Ramadan, because train has none, so")
print("the leaderboard will read higher than this number - see FINDINGS §1.")
print("A/B the two submission files to isolate the Idulfitri guard.")
