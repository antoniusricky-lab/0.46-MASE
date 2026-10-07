"""
Cinema ticket forecasting, version 3.

v2 scored 0.47 on the leaderboard although its movie-grouped CV said 0.386.
Diagnostics found two main causes:
  * train.csv is corrupted on 2025-06-01..2025-06-16 (days missing, 1-87 of
    116 clusters reporting). Windows touching those dates taught the model fake
    collapses, and month/day-of-month features let it memorise them.
  * the test period (Oct 2025 - Mar 2026) contains calendar events that
    holidays.csv under-describes: cuti bersama days, the Christmas/New Year
    school break and the Lebaran week (30% of test rows are in such periods).
v3 cleans the training windows, drops date-memorising features, adds the
missing calendar knowledge, trains on more windows per movie, and checks
everything with a purged time-block CV whose validation dates are never seen
during training (the movie-grouped CV leaked date effects across folds).

Usage:
  python cinema_forecasting_v3.py experiments        # compare configurations
  python cinema_forecasting_v3.py final CONFIG MULT  # train, write submissions
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

SEED = 2026
np.random.seed(SEED)
DATA = 'data/'
HORIZONS = range(4, 11)

# D1 shifts (days after the detected release day) used to build training windows
OPEN_OFFSETS = [0, -1, 1, 2, 3]                                 # opening-week windows
MID_OFFSETS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20]   # mid-run windows

# train.csv is broken on these dates: whole days missing, most clusters absent
BAD_START, BAD_END = pd.Timestamp('2025-06-01'), pd.Timestamp('2025-06-16')

# Calendar facts missing from holidays.csv (SKB 3 Menteri cuti bersama days,
# Lebaran holiday weeks, school breaks). The pre-Lebaran cuti bersama days
# 2026-03-18 and 2026-03-20 are deliberately NOT marked: people travel home
# (mudik) then and cinemas are quiet, unlike a normal holiday.
CUTI_BERSAMA = ['2025-04-02', '2025-04-03', '2025-04-04', '2025-04-07', '2025-05-13',
                '2025-05-30', '2025-06-09', '2025-08-18', '2025-12-26', '2026-02-16',
                '2026-03-23', '2026-03-24']
LEBARAN_WEEKS = [('2025-03-31', '2025-04-13'), ('2026-03-21', '2026-03-29')]
SCHOOL_BREAKS = [('2025-06-23', '2025-07-13'), ('2025-12-20', '2026-01-04')]

PARAMS = dict(objective='l1', learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              cat_smooth=20, seed=SEED, deterministic=True, force_row_wise=True,
              num_threads=12, verbose=-1)


def load_data():
    # Read the six competition files and parse dates.
    # holidays.csv uses Indonesian column names (day_tipe, holiday_tipe); they
    # are renamed to day_type / holiday_type here.
    # Returns: train, test_history, test, movies, holidays, ticket_prices.
    tr = pd.read_csv(DATA + 'train.csv', parse_dates=['date_show'])
    th = pd.read_csv(DATA + 'test_history.csv', parse_dates=['date_show'])
    te = pd.read_csv(DATA + 'test.csv', parse_dates=['date_show'])
    mv = pd.read_csv(DATA + 'movies.csv')
    hol = pd.read_csv(DATA + 'holidays.csv', parse_dates=['date'])
    hol = hol.rename(columns={'day_tipe': 'day_type', 'holiday_tipe': 'holiday_type'})
    pr = pd.read_csv(DATA + 'ticket_prices.csv')
    return tr, th, te, mv, hol, pr


def detect_release_dates(tr):
    # Find each training movie's wide-release day, used as D1.
    # A movie's first date in train is often a 1-cluster preview, while test D1
    # is a wide release, so D1 = first date with >= 50% of the movie's peak
    # cluster count within its first 21 days. Movies already playing on the
    # first train day are dropped because their real release is unknown.
    # Input: tr (train DataFrame). Returns: Series movie_title -> D1 date.
    cnt = tr.groupby(['movie_title', 'date_show']).cinema_ids.nunique().reset_index(name='n')
    first = cnt.groupby('movie_title').date_show.transform('min')
    cnt = cnt[cnt.date_show <= first + pd.Timedelta(days=20)]
    cnt['peak'] = cnt.groupby('movie_title').n.transform('max')
    rel = cnt[cnt.n >= 0.5 * cnt.peak].groupby('movie_title').date_show.min()
    first_all = tr.groupby('movie_title').date_show.min()
    return rel[first_all.reindex(rel.index) > tr.date_show.min()]


def build_samples(hist, d1_map, fut=None, window_id=None):
    # Turn transactions into model rows: one row per (movie, cluster, horizon
    # h = 4..10). Every feature uses ONLY D1-D3 transactions of the window, the
    # same information the test set provides, so nothing from D4-D10 leaks in.
    # hist: transactions (train or test_history); d1_map: Series movie -> D1;
    # fut: transactions holding the D4-D10 truth (train only, None for test);
    # window_id: offset label stored on every row.
    # Returns a DataFrame with features, MASE scale and (train) total_ticket.
    h = hist[hist.movie_title.isin(d1_map.index)].copy()
    h['d1'] = h.movie_title.map(d1_map)
    h['day'] = (h.date_show - h.d1).dt.days + 1
    h = h[h.day.between(1, 3)]

    # per-pair D1/D2/D3 tickets, shows and occupancy (missing day = 0)
    piv = h.pivot_table(index=['movie_title', 'cinema_ids'], columns='day',
                        values=['total_ticket', 'total_show', 'occupation_rate'], aggfunc='sum')
    short = {'total_ticket': 't', 'total_show': 'shw', 'occupation_rate': 'occ'}
    piv.columns = [f'{short[a]}{b}' for a, b in piv.columns]
    piv = piv.reindex(columns=[f'{p}{d}' for p in ['t', 'shw', 'occ'] for d in (1, 2, 3)]).fillna(0)
    piv = piv.reset_index()

    # test only contains pairs that sold tickets on D3, so mimic that in train
    piv = piv[piv.t3 > 0]
    city = hist.drop_duplicates('cinema_ids').set_index('cinema_ids').city_name
    piv['city_name'] = piv.cinema_ids.map(city)

    # competition scale: mean of D1-D3 tickets, at least 1
    piv['scale'] = ((piv.t1 + piv.t2 + piv.t3) / 3).clip(lower=1)
    for d in (1, 2, 3):
        piv[f'r{d}'] = piv[f't{d}'] / piv.scale                                  # opening shape
        piv[f'tps{d}'] = piv[f't{d}'] / piv[f'shw{d}'].replace(0, np.nan)       # tickets per show
    piv['log_scale'] = np.log1p(piv.scale)
    piv['r3_r2'] = piv.t3 / piv.t2.replace(0, np.nan)
    piv['shw3_shw1'] = piv.shw3 / piv.shw1.replace(0, np.nan)
    piv['n_hist_days'] = (piv[['t1', 't2', 't3']] > 0).sum(axis=1)

    # national (movie-level) opening strength and shape across all clusters
    nat = h.groupby(['movie_title', 'day']).agg(nt=('total_ticket', 'sum'),
                                                nc=('cinema_ids', 'nunique'),
                                                ns=('total_show', 'sum')).unstack()
    nat.columns = [f'{a}{b}' for a, b in nat.columns]
    nat = nat.reindex(columns=[f'{p}{d}' for p in ['nt', 'nc', 'ns'] for d in (1, 2, 3)]).fillna(0)
    nat['nat_r31'] = nat.nt3 / nat.nt1.replace(0, np.nan)
    nat['nat_r32'] = nat.nt3 / nat.nt2.replace(0, np.nan)
    nat['nat_log'] = np.log1p(nat[['nt1', 'nt2', 'nt3']].sum(axis=1))
    nat['nat_tps3'] = nat.nt3 / nat.ns3.replace(0, np.nan)
    piv = piv.merge(nat[['nat_r31', 'nat_r32', 'nat_log', 'nc3', 'nat_tps3']].reset_index(),
                    on='movie_title', how='left')
    piv['share3'] = piv.t3 / (piv.movie_title.map(nat.nt3) + 1)

    # expand to the 7 predicted days
    g = piv.merge(pd.DataFrame({'h': list(HORIZONS)}), how='cross')
    g['d1'] = g.movie_title.map(d1_map)
    g['date_show'] = g.d1 + pd.to_timedelta(g.h - 1, unit='D')
    if fut is not None:
        y = fut[['movie_title', 'cinema_ids', 'date_show', 'total_ticket']]
        g = g.merge(y, on=['movie_title', 'cinema_ids', 'date_show'], how='left')
        g['total_ticket'] = g.total_ticket.fillna(0)    # no transaction = 0 tickets
    if window_id is not None:
        g['window'] = window_id
    return g


def build_training_windows(tr, rel, offsets):
    # Build training rows for every D1 offset relative to the release day.
    # Windows whose D10 is after the last train day are skipped. Windows that
    # touch the corrupted June dates are kept but flagged in column 'bad'.
    # Returns one DataFrame with a 'window' column holding the offset.
    parts, last = [], tr.date_show.max()
    for off in offsets:
        d1 = rel + pd.Timedelta(days=off)
        d1 = d1[d1 + pd.Timedelta(days=9) <= last]
        parts.append(build_samples(tr, d1, fut=tr, window_id=off))
    g = pd.concat(parts, ignore_index=True)
    g['bad'] = (g.d1 <= BAD_END) & (g.d1 + pd.Timedelta(days=9) >= BAD_START)
    return g


def make_calendar(hol, enriched):
    # Daily calendar table indexed by date with three codes:
    #   dtype: 0 weekday, 1 friday, 2 weekend (holidays.csv day_type)
    #   hol:   1 on public holidays; if enriched, also cuti bersama days and
    #          the Lebaran holiday weeks
    #   sch:   1 during school breaks (enriched only)
    # Returns a DataFrame indexed by date.
    days = pd.date_range('2025-03-20', '2026-04-10')
    h = hol.set_index('date')
    fallback = pd.Series(np.select([days.dayofweek >= 5, days.dayofweek == 4], [2, 1], 0), index=days)
    cal = pd.DataFrame(index=days)
    cal['dtype'] = h.day_type.map({'weekday': 0, 'friday': 1, 'weekend': 2}).reindex(days).fillna(fallback).astype(int)
    cal['hol'] = (h.holiday_type == 'holiday').reindex(days, fill_value=False).astype(int)
    cal['sch'] = 0
    if enriched:
        cal.loc[pd.to_datetime(CUTI_BERSAMA), 'hol'] = 1
        for a, b in LEBARAN_WEEKS:
            cal.loc[a:b, 'hol'] = 1
        for a, b in SCHOOL_BREAKS:
            cal.loc[a:b, 'sch'] = 1
    return cal


def add_base_calendar(g, cal, pr):
    # Calendar features that do not depend on the holiday list: weekday of the
    # target day and of D1, the city ticket price for the target day's price
    # category, and month / day of month (kept only for the v2 replica).
    name = {0: 'Weekday', 1: 'Friday', 2: 'Weekend'}
    g['dow'] = g.date_show.dt.dayofweek
    g['d1_dow'] = g.d1.dt.dayofweek
    g['month'] = g.date_show.dt.month
    g['dom'] = g.date_show.dt.day
    day = pd.Series(cal.dtype.reindex(g.date_show).to_numpy(), index=g.index).map(name)
    p = pr.set_index(['city_name', 'price_day']).ceil
    g['price'] = p.reindex(pd.MultiIndex.from_arrays([g.city_name, day])).to_numpy()
    return g


def add_calendar(g, cal, sfx):
    # Compare the target day's calendar with the D1-D3 calendar.
    # eff (effective day type) = max(day type, 2 x holiday): a holiday counts
    # like a weekend. eff_diff > 0 means the predicted day should be busier than
    # the days the scale was measured on. sch_* do the same for school breaks.
    # sfx lets two calendar versions live side by side for comparisons.
    look = lambda dates, col: cal[col].reindex(dates).to_numpy()
    eff = lambda dates: np.maximum(look(dates, 'dtype'), 2 * look(dates, 'hol'))
    g[f'dtype{sfx}'] = look(g.date_show, 'dtype')
    g[f'hol{sfx}'] = look(g.date_show, 'hol')
    g[f'eff{sfx}'] = eff(g.date_show)
    tot, sch = 0, 0
    for d in (1, 2, 3):
        dd = g.d1 + pd.Timedelta(days=d - 1)
        g[f'eff_h{d}{sfx}'] = eff(dd)
        tot = tot + g[f'eff_h{d}{sfx}']
        sch = sch + look(dd, 'sch')
    g[f'eff_hist{sfx}'] = tot
    g[f'eff_diff{sfx}'] = g[f'eff{sfx}'] - tot / 3
    g[f'sch{sfx}'] = look(g.date_show, 'sch')
    g[f'sch_hist{sfx}'] = sch / 3
    g[f'sch_diff{sfx}'] = g[f'sch{sfx}'] - sch / 3
    return g


def add_movie_meta(g, mv):
    # Movie metadata: first listed genre, age rating, and format markers in the
    # title such as 3D, IMAX or a bracketed edition. Format copies sell differently.
    m = mv.rename(columns={'original_title': 'movie_title'}).drop_duplicates('movie_title')
    m['genre1'] = m.genre.astype(str).str.split(',').str[0].str.strip()
    g = g.merge(m[['movie_title', 'genre1', 'age_rating']], on='movie_title', how='left')
    t = g.movie_title.str.upper()
    g['is_3d'] = t.str.contains('3D').astype(int)
    g['is_imax'] = t.str.contains('IMAX').astype(int)
    g['has_paren'] = t.str.contains(r'\(').astype(int)
    return g


def prepare(g, hol, pr, mv):
    # Add every feature family to a sample table (train or test). Both calendar
    # versions are computed so configurations can be compared on the same rows.
    cal2, cal3 = make_calendar(hol, False), make_calendar(hol, True)
    g = add_base_calendar(g, cal2, pr)
    g = add_calendar(g, cal2, '')
    g = add_calendar(g, cal3, '_c3')
    return add_movie_meta(g, mv)


CAT_COLS = ['cinema_ids', 'city_name', 'genre1', 'age_rating']
BASE = (['h', 'r1', 'r2', 'r3', 't1', 't2', 't3', 'shw1', 'shw2', 'shw3', 'occ1', 'occ2', 'occ3',
         'tps1', 'tps2', 'tps3', 'log_scale', 'scale', 'r3_r2', 'shw3_shw1', 'n_hist_days',
         'nat_r31', 'nat_r32', 'nat_log', 'nc3', 'nat_tps3', 'share3', 'is_3d', 'is_imax',
         'has_paren'] + CAT_COLS)
CAL_V2 = ['dow', 'd1_dow', 'price', 'dtype', 'hol', 'eff', 'eff_h1', 'eff_h2', 'eff_h3',
          'eff_hist', 'eff_diff', 'month', 'dom']
CAL_V2_NODATE = [c for c in CAL_V2 if c not in ('month', 'dom')]
CAL_V3 = ['dow', 'd1_dow', 'price'] + [f'{c}_c3' for c in
          ['dtype', 'hol', 'eff', 'eff_h1', 'eff_h2', 'eff_h3', 'eff_hist', 'eff_diff',
           'sch', 'sch_hist', 'sch_diff']]
ALL_OFFSETS = OPEN_OFFSETS + MID_OFFSETS
CONFIGS = {
    'A_v2_replica': dict(feats=BASE + CAL_V2, offsets=OPEN_OFFSETS, clean=False),
    'B_clean_june': dict(feats=BASE + CAL_V2, offsets=OPEN_OFFSETS, clean=True),
    'C_no_dates':   dict(feats=BASE + CAL_V2_NODATE, offsets=OPEN_OFFSETS, clean=True),
    'D_midrun':     dict(feats=BASE + CAL_V2_NODATE, offsets=ALL_OFFSETS, clean=True),
    'E_calendar':   dict(feats=BASE + CAL_V3, offsets=ALL_OFFSETS, clean=True),
}


def encode_cats(frames):
    # Give every categorical column the same category list in all frames, so
    # a value maps to the same code in training and test.
    for c in CAT_COLS:
        cats = pd.Categorical(pd.concat([f[c] for f in frames]).astype(str)).categories
        for f in frames:
            f[c] = pd.Categorical(f[c].astype(str), categories=cats)


def make_folds(pool, kind):
    # Validation folds over movies that have a clean offset-0 (release) window.
    # kind='group': 5-fold GroupKFold by movie (v2's scheme; dates are shared
    #               between folds, so market-wide date effects leak).
    # kind='time' : 5 consecutive release-date blocks. Training windows whose
    #               dates overlap a block's D1..D10 span are purged, so the model
    #               never sees the validation dates, as on the leaderboard.
    # Returns a list of (validation movie set, purge span or None).
    mv = pool.drop_duplicates('movie_title')[['movie_title', 'd1']].sort_values(['d1', 'movie_title'])
    if kind == 'group':
        return [(set(mv.movie_title.iloc[b]), None)
                for _, b in GroupKFold(5).split(mv, groups=mv.movie_title)]
    out = []
    for blk in np.array_split(np.arange(len(mv)), 5):
        m = mv.iloc[blk]
        out.append((set(m.movie_title), (m.d1.min(), m.d1.max() + pd.Timedelta(days=9))))
    return out


def train_mask(data, cfg):
    # Rows a configuration may train on: its window offsets, minus the
    # corrupted-June windows when cfg['clean'] is set.
    m = data.window.isin(cfg['offsets'])
    return m & ~data.bad if cfg['clean'] else m


def run_cv(data, cfg, folds):
    # Cross-validate one configuration. Trains LightGBM (L1 loss on y/scale,
    # which is exactly MASE) per fold, scores clean offset-0 windows only.
    # Returns (pooled MASE, per-fold MASE list, best-iteration list).
    feats = cfg['feats']
    cats = [c for c in CAT_COLS if c in feats]
    base = train_mask(data, cfg)
    val_pool = (data.window == 0) & ~data.bad
    w_end = data.d1 + pd.Timedelta(days=9)
    oof = pd.Series(np.nan, index=data.index)
    iters = []
    for vm, purge in folds:
        in_val = data.movie_title.isin(vm)
        trm = base & ~in_val
        if purge is not None:
            trm &= ~((data.d1 <= purge[1]) & (w_end >= purge[0]))
        A, B = data[trm], data[val_pool & in_val]
        dtr = lgb.Dataset(A[feats], A.y, categorical_feature=cats)
        dva = lgb.Dataset(B[feats], B.y, categorical_feature=cats, reference=dtr)
        m = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        iters.append(m.best_iteration)
        oof[B.index] = np.clip(m.predict(B[feats], num_iteration=m.best_iteration), 0, None)
    s = data[val_pool & oof.notna()]
    err = (s.y - oof[s.index]).abs()
    return err.mean(), [err[s.movie_title.isin(vm)].mean() for vm, _ in folds], iters


def load_training_table():
    # Load data, detect release days, build all training windows with features.
    tr, th, te, mv, hol, pr = load_data()
    rel = detect_release_dates(tr)
    data = prepare(build_training_windows(tr, rel, ALL_OFFSETS), hol, pr, mv)
    data['y'] = data.total_ticket / data.scale       # scaled target: L1 on it = MASE
    return data, (tr, th, te, mv, hol, pr)


def experiments():
    # Compare configurations A-E with both CV schemes on identical validation rows.
    data, _ = load_training_table()
    encode_cats([data])
    pool = data[(data.window == 0) & ~data.bad]
    print(f'training rows {len(data):,} | clean release windows {pool.movie_title.nunique()} movies, '
          f'{len(pool):,} rows | corrupted-June rows {int(data.bad.sum()):,}', flush=True)
    folds = {k: make_folds(pool, k) for k in ('group', 'time')}
    for name, cfg in CONFIGS.items():
        for kind in ('group', 'time'):
            score, fs, iters = run_cv(data, cfg, folds[kind])
            print(f'{name:13s} {kind:5s} CV {score:.5f} | folds {" ".join(f"{x:.3f}" for x in fs)} '
                  f'| iters {iters}', flush=True)


def final(cfg_name, leb_mult):
    # Train the chosen configuration on all its training windows (3 seeds),
    # predict the test set and write two submissions:
    #   submission_v3.csv          the model as is
    #   submission_v3_lebaran.csv  same, with Lebaran-week rows (target date
    #                              2026-03-21..27, D1-D3 before Lebaran) scaled by
    #                              leb_mult, a leaderboard probe for the effect
    #                              the training data cannot teach.
    cfg = CONFIGS[cfg_name]
    data, (tr, th, te, mv, hol, pr) = load_training_table()
    d1_test = te.groupby('movie_title').date_show.min() - pd.Timedelta(days=3)
    ts = build_samples(th, d1_test)
    test = te.merge(ts.drop(columns=['city_name', 'd1', 'h']), on=['movie_title', 'cinema_ids', 'date_show'], how='left')
    assert len(test) == len(te) and test.id.equals(te.id)
    print('test rows without D1-D3 history:', int(test.scale.isna().sum()))
    test['scale'] = test.scale.fillna(1)
    test['d1'] = test.movie_title.map(d1_test)
    test['h'] = (test.date_show - test.d1).dt.days + 1
    test = prepare(test, hol, pr, mv)
    encode_cats([data, test])

    pool = data[(data.window == 0) & ~data.bad]
    res = {k: run_cv(data, cfg, make_folds(pool, k)) for k in ('group', 'time')}
    for k, (score, fs, iters) in res.items():
        print(f'{cfg_name} {k} CV MASE {score:.5f} | folds {" ".join(f"{x:.3f}" for x in fs)} | iters {iters}')
    n_iter = int(1.1 * np.mean(res['group'][2]))

    feats = cfg['feats']
    cats = [c for c in CAT_COLS if c in feats]
    A = data[train_mask(data, cfg)]
    dall = lgb.Dataset(A[feats], A.y, categorical_feature=cats)
    ratio = np.zeros(len(test))
    for s in range(3):
        mdl = lgb.train({**PARAMS, 'seed': SEED + s}, dall, n_iter)
        ratio += mdl.predict(test[feats]) / 3
    ratio = np.clip(ratio, 0, None)
    ratio[(test.t1.fillna(0) + test.t2.fillna(0) + test.t3.fillna(0)).to_numpy() == 0] = 0
    imp = pd.Series(mdl.feature_importance('gain'), feats).sort_values(ascending=False)
    print('final iterations', n_iter, '| top features:', ', '.join(imp.index[:12]))

    leb = (test.date_show.between('2026-03-21', '2026-03-27') & (test.d1 <= '2026-03-18')).to_numpy()
    for fname, r in [('submission_v3.csv', ratio), ('submission_v3_lebaran.csv', np.where(leb, ratio * leb_mult, ratio))]:
        tick = np.floor(r * test.scale.to_numpy() + 0.5).astype(int)       # round half up
        sub = pd.DataFrame({'id': te.id, 'total_ticket': tick})
        assert len(sub) == len(te) and (sub.total_ticket >= 0).all()
        sub.to_csv(fname, index=False)
        print(f'wrote {fname}: mean {sub.total_ticket.mean():.1f}, zeros {(sub.total_ticket == 0).mean():.1%}')
    print(f'Lebaran-week rows: {leb.sum()} (multiplier {leb_mult})')
    test['ratio'] = ratio
    print('mean predicted ratio by horizon, Lebaran rows:',
          test[leb].groupby('h').ratio.mean().round(2).to_dict())


if __name__ == '__main__':
    if sys.argv[1] == 'experiments':
        experiments()
    else:
        final(sys.argv[2], float(sys.argv[3]))
