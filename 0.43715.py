"""
Cinema ticket forecasting, version 4 (extends cinema_forecasting_v3.py).

v3 scored 0.45 on the leaderboard. v4 tests these additions with v3's two
cross-validation schemes (movie-grouped folds and purged time blocks) and keeps
only what lowers the CV MASE:
  1. competition: opening strength of films released after a window's D3,
     nationally and at the same cluster (new releases take screens)
  2. each cluster's own weekday profile
  3. extra weight on opening-week training windows (the test only has those)
  4. tuned LightGBM settings
  5. a second, differently configured LightGBM blended in
  6. a classifier for "no tickets that day" combined with the ratio model
  7. per-horizon calibration multipliers
  8. 5-seed averaging in the final model
Steps 5-7 are fitted cross-fitted (on the other folds' out-of-fold
predictions), so their CV gains are honest.

Usage:
  python cinema_forecasting_v4.py experiments
  python cinema_forecasting_v4.py final FEATSET W_OPEN PARAMSET
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import cinema_forecasting_v3 as v3

SEED = v3.SEED
HORIZONS = list(v3.HORIZONS)
DAYS = pd.date_range('2025-03-01', '2026-04-30')
RAMADAN = ('2026-02-19', '2026-03-20')   # 1 Ramadan 1447 H .. the day before Idulfitri

FEAT_E = v3.BASE + v3.CAL_V3                       # v3's final feature set
COMP = ['comp_nat', 'comp_nat_rel', 'comp_cl', 'comp_cl_rel']
CLDOW = ['cl_dow_t', 'cl_dow_h', 'cl_dow_r']
FEATSETS = {'E': FEAT_E, 'F': FEAT_E + COMP, 'G': FEAT_E + CLDOW, 'H': FEAT_E + COMP + CLDOW}
PARAMSETS = {'P0': {},                              # v3 settings
             'P1': dict(num_leaves=31, min_data_in_leaf=100),
             'P2': dict(num_leaves=127, min_data_in_leaf=100),
             'P3': dict(min_data_in_leaf=200, lambda_l2=5.0),
             'P4': dict(feature_fraction=0.5, min_data_in_leaf=80)}
# second model for blending: bigger, randomised trees so its errors differ from model A
PARAMS_B = dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.5,
                lambda_l2=10.0, extra_trees=True)
POST_OPTIONS = [(), ('zero',), ('h',), ('zero', 'h'), ('blend',), ('blend', 'zero'),
                ('blend', 'zero', 'h')]
W_GRID = np.round(np.arange(0, 1.01, 0.1), 1)
ZERO_GRID = np.round(np.arange(0, 1.31, 0.05), 2)
H_GRID = np.round(np.arange(0.85, 1.155, 0.01), 2)


# -----------------------------------------------------------------------------
# New features
# -----------------------------------------------------------------------------
def release_pressure(hist, d1_map):
    # Cumulative opening strength of new releases by date.
    # Every film adds its mean daily D1-D3 tickets on its release day, so
    # P(t) - P(D3) is the opening strength of all films released after a
    # window's D3 and on or before day t. The film's own release always falls
    # on or before its D3, so it never counts as its own competition.
    # hist: transactions; d1_map: Series movie_title -> release date.
    # Returns (national Series indexed by date,
    #          per-cluster Series indexed by (date, cinema_ids)).
    h = hist[hist.movie_title.isin(d1_map.index)].copy()
    h['d1c'] = h.movie_title.map(d1_map)
    h = h[(h.date_show >= h.d1c) & (h.date_show <= h.d1c + pd.Timedelta(days=2))]
    n_days = h.groupby('movie_title').date_show.nunique()
    pair = h.groupby(['movie_title', 'cinema_ids'], as_index=False).total_ticket.sum()
    pair['avg'] = pair.total_ticket / pair.movie_title.map(n_days)
    pair['d1c'] = pair.movie_title.map(d1_map)
    cl = (pair.pivot_table(index='d1c', columns='cinema_ids', values='avg', aggfunc='sum')
              .reindex(DAYS).fillna(0).cumsum())
    return cl.sum(axis=1), cl.stack()


def _at(series, dates, ids=None):
    # Look up a date-indexed (or (date, cluster)-indexed) Series for many rows.
    key = dates if ids is None else pd.MultiIndex.from_arrays([dates, ids])
    return series.reindex(key).to_numpy(dtype=float)


def add_competition(g, nat, cl):
    # Competition a sample faces on its target day from films released after
    # its D3 (see release_pressure):
    #   comp_nat, comp_cl: log of the newcomers' opening strength, nationally
    #                      and at this cluster
    #   comp_*_rel:        the same relative to this film's own D1-D3 level,
    #                      i.e. how big the newcomers are compared with it
    d3 = g.d1 + pd.Timedelta(days=2)
    cid = g.cinema_ids.astype(str)
    cn = np.clip(_at(nat, g.date_show) - _at(nat, d3), 0, None)
    cc = np.clip(np.nan_to_num(_at(cl, g.date_show, cid) - _at(cl, d3, cid)), 0, None)
    own = np.expm1(g.nat_log.to_numpy(dtype=float)) / 3
    g['comp_nat'] = np.log1p(cn)
    g['comp_nat_rel'] = np.log1p(cn) - np.log1p(own)
    g['comp_cl'] = np.log1p(cc)
    g['comp_cl_rel'] = np.log1p(cc) - np.log1p(g.scale.to_numpy(dtype=float))
    return g


def cluster_dow(tr, cal):
    # Each cluster's own weekday profile on ordinary days (no holiday, no
    # school break, outside the corrupted June dates): mean daily tickets on a
    # weekday divided by the cluster's mean over all ordinary days.
    # Returns a Series indexed by (cinema_ids, weekday 0=Mon..6=Sun).
    day = tr.groupby(['cinema_ids', 'date_show'], as_index=False).total_ticket.sum()
    d = day.date_show
    ok = ((cal.hol.reindex(d).to_numpy() == 0) & (cal.sch.reindex(d).to_numpy() == 0)
          & ~d.between(v3.BAD_START, v3.BAD_END).to_numpy())
    day = day[ok].assign(dow=d[ok].dt.dayofweek)
    by_dow = day.groupby(['cinema_ids', 'dow']).total_ticket.mean()
    return by_dow.div(day.groupby('cinema_ids').total_ticket.mean(), level='cinema_ids')


def add_cluster_dow(g, prof):
    # Cluster weekday profile of the target day (cl_dow_t), its mean over the
    # D1-D3 days (cl_dow_h) and their ratio (cl_dow_r): how much busier this
    # cluster usually is on the target weekday than on the opening weekdays.
    cid = g.cinema_ids.astype(str)
    look = lambda dates: prof.reindex(pd.MultiIndex.from_arrays([cid, dates.dt.dayofweek])).to_numpy(dtype=float)
    t = look(g.date_show)
    hist = sum(look(g.d1 + pd.Timedelta(days=k)) for k in range(3)) / 3
    g['cl_dow_t'], g['cl_dow_h'], g['cl_dow_r'] = t, hist, t / hist
    return g


# -----------------------------------------------------------------------------
# Data tables
# -----------------------------------------------------------------------------
def load_table():
    # Training rows: v3's windows (opening offsets -1..3 and mid-run 5..20) with
    # all v3 features plus the competition and cluster-weekday features.
    # Returns (data, ctx); ctx keeps the raw tables needed for the test set.
    tr, th, te, mv, hol, pr = v3.load_data()
    rel = v3.detect_release_dates(tr)
    prof = cluster_dow(tr, v3.make_calendar(hol, True))
    data = v3.prepare(v3.build_training_windows(tr, rel, v3.ALL_OFFSETS), hol, pr, mv)
    data = add_cluster_dow(add_competition(data, *release_pressure(tr, rel)), prof)
    data['y'] = data.total_ticket / data.scale         # scaled target: L1 on it = MASE
    # typical relative competition of release windows, by horizon and D1 weekday
    rw = data[(data.window == 0) & ~data.bad]
    comp_med = rw.groupby(['h', 'd1_dow'])[COMP].median()
    return data, dict(tr=tr, th=th, te=te, mv=mv, hol=hol, pr=pr, prof=prof, rel=rel, comp_med=comp_med)


def load_test(ctx):
    # Test rows in test.csv order with the same features. Competition comes
    # from the openings recorded in test_history.csv (every test-period film).
    th, te = ctx['th'], ctx['te']
    d1_test = te.groupby('movie_title').date_show.min() - pd.Timedelta(days=3)
    ts = v3.build_samples(th, d1_test)
    test = te.merge(ts.drop(columns=['city_name', 'd1', 'h']),
                    on=['movie_title', 'cinema_ids', 'date_show'], how='left')
    print('test rows without D1-D3 history:', int(test.scale.isna().sum()))
    test['scale'] = test.scale.fillna(1)
    test['d1'] = test.movie_title.map(d1_test)
    test['h'] = (test.date_show - test.d1).dt.days + 1
    test = v3.prepare(test, ctx['hol'], ctx['pr'], ctx['mv'])
    nat, cl = release_pressure(th, th.groupby('movie_title').date_show.min())
    test = add_cluster_dow(add_competition(test, nat, cl), ctx['prof'])
    # test_history.csv ends on 2026-03-20, so films released after that are
    # unknown and later target days would show zero competition. For those
    # rows use the typical relative competition of training release windows
    # with the same horizon and D1 weekday, but only where such windows
    # usually do face new releases (median competition above zero).
    late = (test.date_show > th.date_show.max()).to_numpy()
    med = ctx['comp_med'].reindex(pd.MultiIndex.from_arrays([test.h, test.d1_dow]))
    base = {'comp_nat': np.log1p(np.expm1(test.nat_log.to_numpy(dtype=float)) / 3),
            'comp_cl': np.log1p(test.scale.to_numpy(dtype=float))}
    for col, b in base.items():
        rel, typical = med[col + '_rel'].to_numpy(), med[col].to_numpy()
        use = late & (np.nan_to_num(typical) > 0)
        test.loc[use, col + '_rel'] = rel[use]
        test.loc[use, col] = np.clip(rel[use] + b[use], 0, None)
    print(f'test rows after the last known release date (competition imputed): {int(late.sum())}')
    assert len(test) == len(te) and test.id.equals(te.id)
    return test


def release_check(ctx):
    # The competition features assume new releases are recorded equally well in
    # train.csv and test_history.csv. Compare weekly new-release volume.
    nat_tr, _ = release_pressure(ctx['tr'], ctx['rel'])
    nat_te, _ = release_pressure(ctx['th'], ctx['th'].groupby('movie_title').date_show.min())
    per_week = lambda s, a, b: (s[pd.Timestamp(b)] - s[pd.Timestamp(a)]) / ((pd.Timestamp(b) - pd.Timestamp(a)).days / 7)
    print(f'new-release opening tickets per week: train period {per_week(nat_tr, "2025-04-07", "2025-09-21"):,.0f}'
          f' | test period {per_week(nat_te, "2025-10-01", "2026-03-18"):,.0f}', flush=True)


# -----------------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------------
def run_cv(data, feats, folds, params, w_open=1.0, kind='reg'):
    # Cross-validate one LightGBM setup with v3's folds.
    # kind='reg': L1 regression on y = tickets / scale (exactly the MASE)
    # kind='clf': binary classifier for "no tickets that day" (y == 0)
    # Training rows: clean windows of training-fold movies, with opening-week
    # windows weighted w_open; for time folds, windows touching the validation
    # dates are purged. Validation rows: clean release (offset 0) windows.
    # Returns (out-of-fold predictions as a Series, list of best iterations).
    cats = [c for c in v3.CAT_COLS if c in feats]
    p = {**v3.PARAMS, **params}
    if kind == 'clf':
        p.update(objective='binary', metric='binary_logloss')
    target = (data.y == 0).astype(int) if kind == 'clf' else data.y
    weight = np.where(data.window.isin(v3.OPEN_OFFSETS), w_open, 1.0)
    val_pool = (data.window == 0) & ~data.bad
    w_end = data.d1 + pd.Timedelta(days=9)
    oof, iters = pd.Series(np.nan, index=data.index), []
    for vm, purge in folds:
        in_val = data.movie_title.isin(vm)
        trm = ~data.bad & ~in_val
        if purge is not None:
            trm &= ~((data.d1 <= purge[1]) & (w_end >= purge[0]))
        vam = val_pool & in_val
        dtr = lgb.Dataset(data.loc[trm, feats], target[trm], weight=weight[trm.to_numpy()],
                          categorical_feature=cats)
        dva = lgb.Dataset(data.loc[vam, feats], target[vam], categorical_feature=cats, reference=dtr)
        m = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        iters.append(m.best_iteration)
        oof[vam] = m.predict(data.loc[vam, feats], num_iteration=m.best_iteration)
    return oof.dropna(), iters


def mase(data, oof):
    # Unrounded MASE of out-of-fold ratio predictions.
    return float(np.abs(data.y.loc[oof.index] - oof.clip(lower=0)).mean())


def experiments():
    # Stage 1: feature sets E (v3), F (+competition), G (+cluster weekday), H (both).
    # Stage 2: double weight on opening-week windows for the best set.
    # Stage 3: LightGBM settings P1-P4 (grouped CV; the winner confirmed on time CV).
    # Selection criterion: mean of grouped and time CV MASE.
    data, ctx = load_table()
    release_check(ctx)
    v3.encode_cats([data])
    pool = data[(data.window == 0) & ~data.bad]
    folds = {k: v3.make_folds(pool, k) for k in ('group', 'time')}
    print(f'training rows {len(data):,} | validation rows {len(pool):,} ({pool.movie_title.nunique()} movies)', flush=True)

    def cv(fs, w, ps, kinds=('group', 'time')):
        out = {}
        for k in kinds:
            oof, iters = run_cv(data, FEATSETS[fs], folds[k], PARAMSETS[ps], w)
            out[k] = mase(data, oof)
            print(f'  {fs} w_open={w} {ps} {k:5s} CV {out[k]:.5f} | iters {iters}', flush=True)
        return out

    avg = lambda r: (r['group'] + r['time']) / 2
    res = {fs: cv(fs, 1.0, 'P0') for fs in FEATSETS}
    best_fs = min(res, key=lambda f: avg(res[f]))
    ref = res[best_fs]
    print(f'stage 1 best: {best_fs} (mean {avg(ref):.5f})', flush=True)
    rw = cv(best_fs, 2.0, 'P0')
    best_w = 2.0 if avg(rw) < avg(ref) else 1.0
    if best_w == 2.0:
        ref = rw
    print(f'stage 2 opening weight: {best_w} (mean {avg(ref):.5f})', flush=True)
    grp = {'P0': ref['group']}
    for ps in ['P1', 'P2', 'P3', 'P4']:
        grp[ps] = cv(best_fs, best_w, ps, ('group',))['group']
    best_ps = min(grp, key=grp.get)
    if best_ps != 'P0':
        t = cv(best_fs, best_w, best_ps, ('time',))['time']
        if (grp[best_ps] + t) / 2 >= avg(ref):
            print(f'  {best_ps} wins grouped CV but not the mean -> keep P0')
            best_ps = 'P0'
    print(f'CHOICE: featureset {best_fs} | w_open {best_w} | params {best_ps}', flush=True)


# -----------------------------------------------------------------------------
# Post-processing (fitted cross-fitted on out-of-fold predictions)
# -----------------------------------------------------------------------------
def zero_bin(p0):
    # Tenth of P(no sales) a row falls in, 0..9.
    return np.clip((np.asarray(p0) * 10).astype(int), 0, 9)


def apply_post(f, p):
    # Apply fitted post-processing to a frame with columns A, B, p0, h:
    # blend A and B with weight w, multiply by the P(no sales) bin multiplier,
    # then by the horizon multiplier.
    r = p['w'] * f.A.to_numpy() + (1 - p['w']) * f.B.to_numpy()
    if p['zm'] is not None:
        r = r * p['zm'][zero_bin(f.p0.to_numpy())]
    if p['hm'] is not None:
        r = r * pd.Series(f.h.to_numpy()).map(p['hm']).to_numpy()
    return r


def fit_post(f, steps):
    # Fit the chosen post-processing steps in order on frame f by grid search
    # minimising absolute error on the scaled target (= MASE):
    #   'blend': weight of model A vs model B
    #   'zero':  multiplier per tenth of P(no sales); 0 means "predict nothing"
    #   'h':     multiplier per horizon D4..D10
    # Returns a parameter dict for apply_post.
    y, p = f.y.to_numpy(), {'w': 1.0, 'zm': None, 'hm': None}
    best = lambda grid, err: grid[int(np.argmin([err(g) for g in grid]))]
    if 'blend' in steps:
        a, b = f.A.to_numpy(), f.B.to_numpy()
        p['w'] = best(W_GRID, lambda w: np.abs(y - w * a - (1 - w) * b).sum())
    r = apply_post(f, p)
    if 'zero' in steps:
        bins, zm = zero_bin(f.p0.to_numpy()), np.ones(10)
        for k in range(10):
            s = bins == k
            if s.sum() >= 200:
                zm[k] = best(ZERO_GRID, lambda m: np.abs(y[s] - m * r[s]).sum())
        p['zm'] = zm
        r = apply_post(f, p)
    if 'h' in steps:
        hh, hm = f.h.to_numpy(), {}
        for k in HORIZONS:
            s = hh == k
            hm[k] = best(H_GRID, lambda m: np.abs(y[s] - m * r[s]).sum())
        p['hm'] = hm
    return p


def crossfit(f, steps):
    # Honest score of a post-processing recipe: for each fold, fit it on the
    # other folds' out-of-fold rows and apply it to this fold.
    # Returns (unrounded MASE, MASE after rounding to whole tickets).
    pred = np.zeros(len(f))
    for k in np.unique(f.fold):
        m = (f.fold == k).to_numpy()
        pred[m] = apply_post(f[m], fit_post(f[~m], steps))
    pred = np.clip(pred, 0, None)
    raw = np.abs(f.y.to_numpy() - pred).mean()
    rnd = (np.abs(f.t.to_numpy() - np.floor(pred * f.scale.to_numpy() + 0.5)) / f.scale.to_numpy()).mean()
    return raw, rnd


def oof_frame(data, oofs, folds):
    # Out-of-fold predictions of models A, B and the classifier on the same
    # validation rows, with target, horizon, scale, tickets and fold number.
    idx = oofs['A'].index
    f = pd.DataFrame({k: v.reindex(idx).to_numpy() for k, v in oofs.items()}, index=idx)
    f['A'], f['B'] = f.A.clip(lower=0), f.B.clip(lower=0)
    f['y'], f['h'] = data.y.loc[idx].to_numpy(), data.h.loc[idx].to_numpy()
    f['scale'], f['t'] = data.scale.loc[idx].to_numpy(), data.total_ticket.loc[idx].to_numpy()
    fold_of = {m: i for i, (vm, _) in enumerate(folds) for m in vm}
    f['fold'] = data.movie_title.loc[idx].map(fold_of).to_numpy()
    return f


# -----------------------------------------------------------------------------
# Final model and submissions
# -----------------------------------------------------------------------------
def train_full(data, feats, params, n_iter, seeds, w_open, kind='reg'):
    # Train one model per seed on every clean training window (no hold-out).
    cats = [c for c in v3.CAT_COLS if c in feats]
    p = {**v3.PARAMS, **params}
    if kind == 'clf':
        p.update(objective='binary', metric='binary_logloss')
    trm = ~data.bad
    target = (data.y == 0).astype(int) if kind == 'clf' else data.y
    weight = np.where(data.window.isin(v3.OPEN_OFFSETS), w_open, 1.0)[trm.to_numpy()]
    ds = lgb.Dataset(data.loc[trm, feats], target[trm], weight=weight, categorical_feature=cats)
    return [lgb.train({**p, 'seed': s}, ds, n_iter) for s in seeds]


def write_sub(fname, te, ratio, scale):
    # Convert ratios to whole tickets (round half up) and write id,total_ticket.
    tick = np.floor(np.clip(ratio, 0, None) * scale + 0.5).astype(int)
    sub = pd.DataFrame({'id': te.id, 'total_ticket': tick})
    assert len(sub) == len(te) and sub.id.equals(te.id) and (sub.total_ticket >= 0).all()
    sub.to_csv(fname, index=False)
    print(f'wrote {fname}: mean {sub.total_ticket.mean():.1f} tickets, zeros {(sub.total_ticket == 0).mean():.1%}')


def final(fs, w_open, ps):
    # 1. out-of-fold predictions for model A (chosen settings), model B and the
    #    zero classifier, on both CV schemes
    # 2. pick the post-processing recipe with the best cross-fitted CV
    # 3. train the final models on all windows (A: 5 seeds, B/classifier: 3)
    # 4. predict the test set and write the submission plus probe variants
    feats, params = FEATSETS[fs], PARAMSETS[ps]
    data, ctx = load_table()
    test = load_test(ctx)
    v3.encode_cats([data, test])
    te = ctx['te']
    pool = data[(data.window == 0) & ~data.bad]
    print(f'competition comp_nat median: train release windows {pool.comp_nat.median():.2f} | test {test.comp_nat.median():.2f}')
    folds = {k: v3.make_folds(pool, k) for k in ('group', 'time')}
    specs = {'A': (params, 'reg'), 'B': (PARAMS_B, 'reg'), 'p0': (params, 'clf')}
    oofs, iters = {k: {} for k in folds}, {}
    for name, (p, kind) in specs.items():
        for k in folds:
            o, it = run_cv(data, feats, folds[k], p, w_open, kind)
            oofs[k][name] = o
            iters.setdefault(name, it)
            if kind == 'reg':
                print(f'model {name} {k:5s} CV MASE {mase(data, o):.5f} | iters {it}', flush=True)
            else:
                z = (data.y.loc[o.index] == 0).astype(int)
                print(f'zero classifier {k:5s} AUC {roc_auc_score(z, o):.4f} | actual zero share {z.mean():.3f} '
                      f'| mean P {o.mean():.3f} | iters {it}', flush=True)

    frames = {k: oof_frame(data, oofs[k], folds[k]) for k in folds}
    scores = {}
    for steps in POST_OPTIONS:
        scores[steps] = {k: crossfit(frames[k], steps) for k in frames}
        r = scores[steps]
        print(f'post {"+".join(steps) or "model A only":18s} grouped {r["group"][0]:.5f} '
              f'(rounded {r["group"][1]:.5f}) | time {r["time"][0]:.5f} (rounded {r["time"][1]:.5f})', flush=True)
    steps = min(scores, key=lambda s: scores[s]['group'][1] + scores[s]['time'][1])
    post = fit_post(pd.concat([frames['group'], frames['time']]), steps)
    print('chosen post-processing:', steps or 'none', '|', post, flush=True)

    n = {k: max(1, int(1.1 * np.mean(v))) for k, v in iters.items()}
    use_b = 'blend' in steps and post['w'] < 1
    use_z = 'zero' in steps
    print('final iterations', n, '| use B', use_b, '| use classifier', use_z, flush=True)
    mA = train_full(data, feats, params, n['A'], [SEED + i for i in range(5)], w_open)
    rA = np.clip(np.mean([m.predict(test[feats]) for m in mA], axis=0), 0, None)
    rB = (np.clip(np.mean([m.predict(test[feats]) for m in
                           train_full(data, feats, PARAMS_B, n['B'], [SEED + 100 + i for i in range(3)], w_open)], axis=0), 0, None)
          if use_b else np.zeros(len(test)))
    p0 = (np.mean([m.predict(test[feats]) for m in
                   train_full(data, feats, params, n['p0'], [SEED + 200 + i for i in range(3)], w_open, 'clf')], axis=0)
          if use_z else np.zeros(len(test)))
    imp = pd.Series(np.mean([m.feature_importance('gain') for m in mA], axis=0), feats).sort_values(ascending=False)
    print('top features:', ', '.join(imp.index[:15]))

    ratio = np.clip(apply_post(pd.DataFrame({'A': rA, 'B': rB, 'p0': p0, 'h': test.h.to_numpy()}), post), 0, None)
    hist_sum = (test.t1.fillna(0) + test.t2.fillna(0) + test.t3.fillna(0)).to_numpy()
    ratio[hist_sum == 0] = 0

    # Probe variants for effects the training data cannot teach (no Lebaran
    # run-in or Ramadan in train.csv). Upload them to measure the effects.
    dates, d1 = test.date_show, test.d1
    leb = (dates.between('2026-03-21', '2026-03-27') & (d1 <= '2026-03-18')).to_numpy()
    in_ram = lambda d: d.between(*RAMADAN).to_numpy().astype(float)
    ram_hist = sum(in_ram(d1 + pd.Timedelta(days=k)) for k in range(3)) / 3
    ram_exp = in_ram(dates) * (1 - ram_hist)       # 1 = target in Ramadan, opening days before it
    leb_ratio = np.where(leb, ratio * 1.6, ratio)
    scale = test.scale.to_numpy()
    write_sub('submission_v4.csv', te, ratio, scale)
    write_sub('submission_v4_lebaran.csv', te, leb_ratio, scale)
    write_sub('submission_v4_lebaran_ramadan.csv', te, leb_ratio * 0.75 ** ram_exp, scale)
    print(f'Lebaran-week rows: {int(leb.sum())} | Ramadan-transition rows: {int((ram_exp > 0).sum())}')


if __name__ == '__main__':
    if sys.argv[1] == 'experiments':
        experiments()
    else:
        final(sys.argv[2], float(sys.argv[3]), sys.argv[4])
