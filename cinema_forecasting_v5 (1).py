"""
Cinema ticket forecasting, version 5 (extends cinema_forecasting_v3.py and v4).

Leaderboard so far: v2 0.47, v3 0.45, v4 0.43. Local CV gains have carried
over to the leaderboard about one to one, so v5 tests more ideas with v3's two
CV schemes and keeps only what lowers the mean of the movie-grouped and the
purged time-block CV MASE:
  S  opening shape: national cluster and show trends over D1-D3, the pair's
     tickets per show relative to the national level, and the cluster's size
     and screening capacity (typical day in train.csv)
  P  incumbents: opening strength of films released in the 14 days before D1
  R  long weekends: length of the run of days off that contains each date
  B  more training windows (D1 shifted 4, 15, 17, 19 and 22-30 days)
  C  a lower learning rate (0.03 instead of 0.05)
Lebaran and Ramadan probe files are written from saved test predictions.

Usage:
  python cinema_forecasting_v5.py                        # no arguments: same as 'final v4'
  python cinema_forecasting_v5.py experiments            # compare configurations, writes v5_choice.json
  python cinema_forecasting_v5.py final [v4|choice]      # writes submission_v5.csv (default v4)
  python cinema_forecasting_v5.py variant NAME LEB RAM   # writes submission_v5_NAME.csv
final v4 is v4's model plus the Nyepi calendar fix (it made the current
submission_v5.csv); final choice is what experiments picked (v4 plus extra
training windows). Both read v5_choice.json. variant multiplies the saved
final predictions: Lebaran-week rows by LEB and Ramadan rows by RAM
(e.g. variant lebaran 1.6 1.0).
"""
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import cinema_forecasting_v3 as v3
import cinema_forecasting_v4 as v4

SEED = v3.SEED
EXTRA_OFFSETS = [4, 15, 17, 19, 22, 24, 26, 28, 30]
FEAT_F = v4.FEATSETS['F']                       # v4's final feature set
P2 = dict(v4.PARAMSETS['P2'])                   # v4's tuned LightGBM settings
SHAPE = ['nat_nc31', 'nat_ns31', 'nat_ns32', 'tps_rel3', 'cap_share3', 'cl_size']
PREV = ['prev_nat', 'prev_nat_rel', 'prev_cl', 'prev_cl_rel']
RUNS = ['run_t', 'run_h']
BUNDLES = {'S': SHAPE, 'P': PREV, 'R': RUNS}

# Test-period dates that holidays.csv lists as public holidays but on which
# cinemas saw no holiday crowds. Nyepi (2026-03-19) fell in the Lebaran travel
# days: the seven films that opened on 2026-03-18 sold a median 0.68x their
# opening-day tickets on it, like an ordinary Thursday after a Wednesday
# opening (0.67-0.75), while Wednesday openers whose second day was a public
# holiday in train.csv sold about 1.3x. These dates are treated as normal days.
NOT_BOOSTED = ['2026-03-19']


def cluster_static(tr, cal):
    # Typical size of each cinema cluster: mean daily tickets and mean daily
    # shows over all its films, on ordinary train days (no holiday, no school
    # break, outside the corrupted June dates). Daily shows approximate the
    # cluster's screening capacity.
    # Returns a DataFrame indexed by cinema_ids with cl_tickets and cl_shows.
    day = tr.groupby(['cinema_ids', 'date_show'], as_index=False)[['total_ticket', 'total_show']].sum()
    d = day.date_show
    ok = ((cal.hol.reindex(d).to_numpy() == 0) & (cal.sch.reindex(d).to_numpy() == 0)
          & ~d.between(v3.BAD_START, v3.BAD_END).to_numpy())
    st = day[ok].groupby('cinema_ids')[['total_ticket', 'total_show']].mean()
    return st.rename(columns={'total_ticket': 'cl_tickets', 'total_show': 'cl_shows'})


def add_shape(g, hist, stat):
    # Opening-shape features, from D1-D3 transactions only:
    #   nat_nc31     clusters showing the film on D3 / on D1 (widening or narrowing)
    #   nat_ns31/32  national shows on D3 / on D1 and on D2 (cinemas adding or
    #                cutting screenings already during the opening)
    #   tps_rel3     the pair's D3 tickets per show / the national value
    #                (does the film sell better at this cluster than elsewhere?)
    #   cap_share3   the pair's D3 shows / the cluster's typical daily shows
    #                (how much of the cinema's programme the film holds)
    #   cl_size      log of the cluster's typical daily tickets
    # g: sample table (movie_title, d1, cinema_ids, shw3, tps3, nat_tps3);
    # hist: transactions of the same period; stat: output of cluster_static.
    # Returns g with the six new columns (row order kept).
    keys = g[['movie_title', 'd1']].drop_duplicates()
    h = hist[['movie_title', 'cinema_ids', 'date_show', 'total_show']].merge(keys, on='movie_title')
    h['day'] = (h.date_show - h.d1).dt.days + 1
    h = h[h.day.between(1, 3)]
    a = h.groupby(['movie_title', 'd1', 'day']).agg(nc=('cinema_ids', 'nunique'),
                                                   ns=('total_show', 'sum')).unstack('day')
    a.columns = [f'{x}{d}' for x, d in a.columns]
    a = a.reindex(columns=['nc1', 'nc2', 'nc3', 'ns1', 'ns2', 'ns3']).fillna(0)
    s = pd.DataFrame({'nat_nc31': a.nc3 / a.nc1.replace(0, np.nan),
                      'nat_ns31': a.ns3 / a.ns1.replace(0, np.nan),
                      'nat_ns32': a.ns3 / a.ns2.replace(0, np.nan)}, index=a.index).reset_index()
    g = g.merge(s, on=['movie_title', 'd1'], how='left')
    g['tps_rel3'] = g.tps3 / g.nat_tps3
    cid = g.cinema_ids.astype(str)
    g['cap_share3'] = g.shw3.to_numpy(dtype=float) / cid.map(stat.cl_shows).to_numpy(dtype=float)
    g['cl_size'] = np.log1p(cid.map(stat.cl_tickets).to_numpy(dtype=float))
    return g


def add_incumbents(g, nat, cl, first_ok):
    # Opening strength of the films released in the 14 days before this
    # window's D1 (the incumbents it shares screens with), nationally and at
    # this cluster, also relative to the film's own opening, like v4's
    # competition features. nat / cl: cumulative release strength from
    # v4.release_pressure. NaN when those 14 days start before first_ok, the
    # earliest date from which every release is known.
    start, a, b = (g.d1 - pd.Timedelta(days=k) for k in (14, 15, 1))
    cid = g.cinema_ids.astype(str)
    pn = np.clip(v4._at(nat, b) - v4._at(nat, a), 0, None)
    pc = np.clip(np.nan_to_num(v4._at(cl, b, cid) - v4._at(cl, a, cid)), 0, None)
    own = np.expm1(g.nat_log.to_numpy(dtype=float)) / 3
    unknown = (start < first_ok).to_numpy()
    nan_if = lambda x: np.where(unknown, np.nan, x)
    g['prev_nat'] = nan_if(np.log1p(pn))
    g['prev_nat_rel'] = nan_if(np.log1p(pn) - np.log1p(own))
    g['prev_cl'] = nan_if(np.log1p(pc))
    g['prev_cl_rel'] = nan_if(np.log1p(pc) - np.log1p(g.scale.to_numpy(dtype=float)))
    return g


def add_runs(g, cal):
    # Long weekends: length of the run of consecutive days off (weekend,
    # public holiday, cuti bersama, Lebaran week) that contains the target day,
    # 0 on working days (run_t), and the mean of the same over D1-D3 (run_h).
    off = ((cal.dtype == 2) | (cal.hol == 1)).astype(int)
    run = off.groupby((off != off.shift()).cumsum()).transform('sum') * off
    look = lambda dates: run.reindex(dates).to_numpy(dtype=float)
    g['run_t'] = look(g.date_show)
    g['run_h'] = sum(look(g.d1 + pd.Timedelta(days=k)) for k in range(3)) / 3
    return g


def load_table(offsets):
    # Training rows for the given D1 offsets with v3 features, v4 competition
    # and cluster-weekday features, and the new S, P and R features.
    # Returns (data, ctx); ctx holds what load_test needs.
    tr, th, te, mv, hol, pr = v3.load_data()
    rel = v3.detect_release_dates(tr)
    cal = v3.make_calendar(hol, True)
    prof, stat = v4.cluster_dow(tr, cal), cluster_static(tr, cal)
    nat, cl = v4.release_pressure(tr, rel)
    data = v3.prepare(v3.build_training_windows(tr, rel, offsets), hol, pr, mv)
    data = v4.add_cluster_dow(v4.add_competition(data, nat, cl), prof)
    data = add_runs(add_incumbents(add_shape(data, tr, stat), nat, cl, rel.min()), cal)
    data['y'] = data.total_ticket / data.scale        # scaled target: L1 on it = MASE
    rw = data[(data.window == 0) & ~data.bad]
    comp_med = rw.groupby(['h', 'd1_dow'])[v4.COMP].median()
    return data, dict(tr=tr, th=th, te=te, mv=mv, hol=hol, pr=pr, prof=prof, rel=rel,
                      comp_med=comp_med, stat=stat, cal=cal)


def load_test(ctx):
    # Test rows (test.csv order) with every feature. The NOT_BOOSTED dates are
    # removed from the holiday list first. Incumbents come from the openings
    # in train.csv and test_history.csv together, so test windows in early
    # October also see the late-September releases.
    tr, th = ctx['tr'], ctx['th']
    hol = ctx['hol'].copy()
    hol.loc[hol.date.isin(pd.to_datetime(NOT_BOOSTED)), 'holiday_type'] = 'normal'
    ctx = {**ctx, 'hol': hol, 'cal': v3.make_calendar(hol, True)}
    test = v4.load_test(ctx)
    d1_all = pd.concat([ctx['rel'], th.groupby('movie_title').date_show.min()])
    d1_all = d1_all[~d1_all.index.duplicated(keep='last')]
    nat, cl = v4.release_pressure(pd.concat([tr, th], ignore_index=True), d1_all)
    test = add_runs(add_incumbents(add_shape(test, th, ctx['stat']), nat, cl, ctx['rel'].min()), ctx['cal'])
    assert len(test) == len(ctx['te']) and test.id.equals(ctx['te'].id)
    return test


def run_cv(data, feats, folds, params, rows):
    # Cross-validate one LightGBM setup (L1 loss on y = tickets / scale, which
    # is the MASE) with v3's folds. Trains on the clean windows selected by
    # rows from the training-fold movies (time folds also purge windows that
    # touch the validation dates) and scores the validation movies' clean
    # release windows. Returns (out-of-fold predictions, best iterations).
    cats = [c for c in v3.CAT_COLS if c in feats]
    p = {**v3.PARAMS, **params}
    val_pool = (data.window == 0) & ~data.bad
    w_end = data.d1 + pd.Timedelta(days=9)
    oof, iters = pd.Series(np.nan, index=data.index), []
    for vm, purge in folds:
        in_val = data.movie_title.isin(vm)
        trm = rows & ~data.bad & ~in_val
        if purge is not None:
            trm &= ~((data.d1 <= purge[1]) & (w_end >= purge[0]))
        vam = val_pool & in_val
        dtr = lgb.Dataset(data.loc[trm, feats], data.y[trm], categorical_feature=cats)
        dva = lgb.Dataset(data.loc[vam, feats], data.y[vam], categorical_feature=cats, reference=dtr)
        m = lgb.train(p, dtr, 6000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        iters.append(m.best_iteration)
        oof[vam] = m.predict(data.loc[vam, feats], num_iteration=m.best_iteration)
    return oof.dropna(), iters


def experiments():
    # Stage A: add each new feature bundle (S, P, R) to v4's setup on its own;
    #          if several help, also try them together; keep the best.
    # Stage B: add the extra training windows.
    # Stage C: learning rate 0.03.
    # A step is kept only if it lowers the mean of grouped and time CV MASE.
    # Writes the choice to v5_choice.json.
    data, ctx = load_table(v3.ALL_OFFSETS + EXTRA_OFFSETS)
    test = load_test(ctx)
    pool = data[(data.window == 0) & ~data.bad]
    print('new features, median (NaN share): train release windows | test')
    for c in SHAPE + PREV + RUNS:
        print(f'  {c:12s} {pool[c].median():8.3f} ({pool[c].isna().mean():.2f}) | '
              f'{test[c].median():8.3f} ({test[c].isna().mean():.2f})', flush=True)
    del test
    v3.encode_cats([data])
    folds = {k: v3.make_folds(pool, k) for k in ('group', 'time')}
    base_rows = data.window.isin(v3.ALL_OFFSETS)
    all_rows = pd.Series(True, index=data.index)
    print(f'clean training rows: v4 windows {int((base_rows & ~data.bad).sum()):,} | '
          f'with extra windows {int((~data.bad).sum()):,}', flush=True)
    log = {}

    def cv(name, feats, params, rows):
        r = {}
        for k in ('group', 'time'):
            oof, iters = run_cv(data, feats, folds[k], params, rows)
            r[k], r[k + '_iters'] = v4.mase(data, oof), iters
            print(f'  {name:26s} {k:5s} CV {r[k]:.5f} | iters {iters}', flush=True)
        r['mean'] = (r['group'] + r['time']) / 2
        print(f'  {name:26s} mean     {r["mean"]:.5f}', flush=True)
        log[name] = r
        return r

    best_name, best_feats = 'v4 (F, P2)', FEAT_F
    best = cv(best_name, FEAT_F, P2, base_rows)
    single = {n: cv(f'F + {n}', FEAT_F + b, P2, base_rows) for n, b in BUNDLES.items()}
    wins = [n for n in BUNDLES if single[n]['mean'] < best['mean']]
    cands = {f'F + {n}': (FEAT_F + BUNDLES[n], single[n]) for n in wins}
    if len(wins) > 1:
        name = 'F + ' + ' + '.join(wins)
        feats = FEAT_F + [c for n in wins for c in BUNDLES[n]]
        cands[name] = (feats, cv(name, feats, P2, base_rows))
    if cands:
        best_name = min(cands, key=lambda n: cands[n][1]['mean'])
        best_feats, best = cands[best_name]
    print(f'stage A: {best_name} (mean {best["mean"]:.5f})', flush=True)

    rows = 'base'
    r = cv(best_name + ' + windows', best_feats, P2, all_rows)
    if r['mean'] < best['mean']:
        rows, best = 'all', r
    print(f'stage B: windows {rows} (mean {best["mean"]:.5f})', flush=True)

    params = dict(P2)
    r = cv('learning rate 0.03', best_feats, {**P2, 'learning_rate': 0.03},
           all_rows if rows == 'all' else base_rows)
    if r['mean'] < best['mean']:
        params, best = {**P2, 'learning_rate': 0.03}, r
    print(f'stage C: params {params} (mean {best["mean"]:.5f})', flush=True)

    with open('v5_choice.json', 'w') as f:
        json.dump(dict(name=best_name, feats=best_feats, rows=rows, params=params,
                       iters_group=best['group_iters'], cv_group=best['group'],
                       cv_time=best['time'], log=log), f, indent=1)
    print('CHOICE:', best_name, '| windows', rows, '| params', params, flush=True)


def final(mode):
    # Train a configuration on all of its clean training windows with 5 seeds,
    # predict the test set, save the raw predictions (for probe variants) and
    # write submission_v5.csv.
    # mode 'choice': the configuration chosen by experiments()
    # mode 'v4':     v4's windows, features, settings and rounds, so the only
    #                difference from submission_v4.csv is the calendar fix
    with open('v5_choice.json') as f:
        ch = json.load(f)
    if mode == 'v4':
        b = ch['log']['v4 (F, P2)']
        ch = dict(name='v4 (F, P2)', feats=FEAT_F, rows='base', params=P2,
                  iters_group=b['group_iters'], cv_group=b['group'], cv_time=b['time'])
    offsets = v3.ALL_OFFSETS + (EXTRA_OFFSETS if ch['rows'] == 'all' else [])
    data, ctx = load_table(offsets)
    test = load_test(ctx)
    v3.encode_cats([data, test])
    feats, params = ch['feats'], ch['params']
    cats = [c for c in v3.CAT_COLS if c in feats]
    n_iter = max(1, int(1.1 * np.mean(ch['iters_group'])))
    print(f'chosen: {ch["name"]} | windows {ch["rows"]} | params {params} | CV grouped '
          f'{ch["cv_group"]:.5f} time {ch["cv_time"]:.5f} | {n_iter} rounds', flush=True)
    trm = ~data.bad
    ds = lgb.Dataset(data.loc[trm, feats], data.y[trm], categorical_feature=cats)
    models = [lgb.train({**v3.PARAMS, **params, 'seed': SEED + i}, ds, n_iter) for i in range(5)]
    ratio = np.clip(np.mean([m.predict(test[feats]) for m in models], axis=0), 0, None)
    hist_sum = (test.t1.fillna(0) + test.t2.fillna(0) + test.t3.fillna(0)).to_numpy()
    ratio[hist_sum == 0] = 0
    imp = pd.Series(np.mean([m.feature_importance('gain') for m in models], axis=0), feats)
    print('top features:', ', '.join(imp.sort_values(ascending=False).index[:15]))
    dates, d1 = test.date_show, test.d1
    leb = (dates.between('2026-03-21', '2026-03-27') & (d1 <= '2026-03-18')).to_numpy()
    in_ram = lambda d: d.between(*v4.RAMADAN).to_numpy().astype(float)
    ram_exp = in_ram(dates) * (1 - sum(in_ram(d1 + pd.Timedelta(days=k)) for k in range(3)) / 3)
    pd.DataFrame({'id': test.id, 'ratio': ratio, 'scale': test.scale, 'h': test.h,
                  'leb': leb.astype(int), 'ram_exp': ram_exp}).to_csv('v5_test_pred.csv', index=False)
    v4.write_sub('submission_v5.csv', ctx['te'], ratio, test.scale.to_numpy())


def variant(name, leb_mult, ram_mult):
    # Probe submission from the saved v5 predictions: Lebaran-week rows
    # (target 2026-03-21..27, film opened by 2026-03-18) times leb_mult, and
    # Ramadan rows of films that opened before Ramadan times ram_mult (less
    # when some opening days were already in Ramadan).
    p = pd.read_csv('v5_test_pred.csv')
    te = pd.read_csv(v3.DATA + 'test.csv')
    r = p.ratio.to_numpy() * np.where(p.leb == 1, leb_mult, 1.0) * ram_mult ** p.ram_exp.to_numpy()
    v4.write_sub(f'submission_v5_{name}.csv', te, r, p.scale.to_numpy())


USAGE = __doc__[__doc__.index('Usage:'):].rstrip()


def main(args):
    # Command-line entry point; args are the words after the script name.
    # With no arguments (the editor's Run button passes none) it runs
    # 'final v4'. Unknown modes or wrong values stop with the usage text
    # instead of a traceback.
    if not args:
        print("No mode given, so running 'final v4' (takes a few minutes).\n\n" + USAGE + '\n', flush=True)
        args = ['final', 'v4']
    mode, rest = args[0], args[1:]
    if mode == 'experiments' and not rest:
        experiments()
    elif mode == 'final' and rest in ([], ['v4'], ['choice']):
        final(rest[0] if rest else 'v4')
    elif mode == 'variant' and len(rest) == 3:
        try:
            leb_mult, ram_mult = float(rest[1]), float(rest[2])
        except ValueError:
            sys.exit(f'LEB and RAM must be numbers, got {rest[1]!r} and {rest[2]!r}\n\n{USAGE}')
        variant(rest[0], leb_mult, ram_mult)
    else:
        sys.exit(f'Unrecognised arguments: {" ".join(args)}\n\n{USAGE}')


if __name__ == '__main__':
    main(sys.argv[1:])
