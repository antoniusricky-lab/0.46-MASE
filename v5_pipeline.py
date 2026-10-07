"""Shared setup for cinema_v5.ipynb.

Every heavy cell imports this instead of relying on state left behind by an earlier cell,
so cells can be run in any order, or re-run after a kernel restart, without surprises.

It also hides three things that are easy to get wrong:

1. `v3.load_data()` hard-codes `DATA = 'data/'` and the canonical six filenames, but the real
   file is often `train (1).csv`. `stage_data()` copies correctly named files into
   `cache/v3data/` and `load_modules()` repoints `v3.DATA` at them.
2. `v3.encode_cats()` must be called with the training frame AND the test frame TOGETHER, or
   the category codes do not align and every test prediction is silently wrong.
3. Featsets F/G/H need v4's own competition and cluster-weekday columns, so the table has to
   come from `v4.load_table()`, not `v3.prepare()`.
"""
import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 3          # bump when the public API changes; CELL 2 checks it
CACHE_DIR = Path("cache")
SCALE_EDGES = [0, 2, 5, 10, 25, 50, 100, 250, 1e9]
SCALE_LABELS = ["<=2", "2-5", "5-10", "10-25", "25-50", "50-100", "100-250", ">250"]
STEMS = ("train", "test_history", "test", "movies", "holidays", "ticket_prices")


def find_file(stem, data_dir="."):
    """Locate `<stem>.csv`, tolerating a Windows-style " (1)" duplicate suffix."""
    d = Path(data_dir)
    exact = d / f"{stem}.csv"
    if exact.exists() and exact.stat().st_size > 2:
        return exact
    pat = re.compile(r"^" + re.escape(stem) + r"( \(\d+\))?\.csv$")
    hits = sorted((p for p in d.glob("*.csv") if pat.match(p.name)),
                  key=lambda p: -p.stat().st_size)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{stem}.csv not found in {d.resolve()}")


def stage_data(data_dir=".", cache_dir=CACHE_DIR, verbose=True):
    """Copy the six competition files into one directory under their canonical names."""
    stage = Path(cache_dir) / "v3data"
    stage.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        src, dst = find_file(stem, data_dir), stage / f"{stem}.csv"
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copyfile(src, dst)
        if verbose:
            print(f"  staged {stem + '.csv':20s} <- {src.name:24s} "
                  f"{dst.stat().st_size / 1e6:7.2f} MB")
    return stage


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)          # both files guard real work behind __main__
    return mod


def load_modules(data_dir=".", cache_dir=CACHE_DIR, verbose=True):
    """Import cinema_forecasting_v3 and 0.43715.py, with data paths repointed."""
    v3_path = Path(data_dir) / "cinema_forecasting_v3.py"
    if not v3_path.exists():
        alt = sorted(Path(data_dir).glob("cinema_forecasting_v3*.py"))
        if not alt:
            raise FileNotFoundError("cinema_forecasting_v3.py not found")
        shutil.copyfile(alt[0], v3_path)
        if verbose:
            print(f"  copied {alt[0].name} -> cinema_forecasting_v3.py")
    v4_path = Path(data_dir) / "0.43715.py"
    if not v4_path.exists():
        raise FileNotFoundError("0.43715.py not found")

    stage = stage_data(data_dir, cache_dir, verbose)
    v3 = _load_module(v3_path, "cinema_forecasting_v3")
    v4 = _load_module(v4_path, "v4_script")
    v3.DATA = str(stage) + os.sep
    v4.v3.DATA = v3.DATA                  # v4 reaches data through its own reference to v3
    if verbose:
        print(f"  v3.DATA -> {v3.DATA}")
    return v3, v4


def load_modules_v5(data_dir=".", cache_dir=CACHE_DIR, verbose=True):
    """Also import Sam's cinema_forecasting_v5.py. Returns (v3, v4, v5).

    v5 does `import cinema_forecasting_v4 as v4`, so 0.43715.py has to be importable
    under that exact name; this stages a copy rather than renaming the original.
    """
    v3, v4 = load_modules(data_dir, cache_dir, verbose)
    d = Path(data_dir)
    v4_alias = d / "cinema_forecasting_v4.py"
    if not v4_alias.exists():
        shutil.copyfile(d / "0.43715.py", v4_alias)
        if verbose:
            print("  copied 0.43715.py -> cinema_forecasting_v4.py (v5 imports that name)")
    sys.modules["cinema_forecasting_v4"] = v4
    v5_path = d / "cinema_forecasting_v5.py"
    if not v5_path.exists():
        alt = sorted(d.glob("cinema_forecasting_v5*.py"))
        if not alt:
            raise FileNotFoundError("cinema_forecasting_v5.py not found")
        shutil.copyfile(alt[0], v5_path)
        if verbose:
            print(f"  copied {alt[0].name} -> cinema_forecasting_v5.py")
    v5 = _load_module(v5_path, "cinema_forecasting_v5")
    if verbose:
        print(f"  v5 imported | EXTRA_OFFSETS {v5.EXTRA_OFFSETS} | "
              f"bundles {list(v5.BUNDLES)} | NOT_BOOSTED {v5.NOT_BOOSTED}")
    return v3, v4, v5


def get_tables_v5(v3, v4, v5, offsets=None, cache_dir=CACHE_DIR, tag="v5all",
                  rebuild=False, verbose=True):
    """Build (or load) Sam v5's training-window and test tables.

    v5's test table carries the Nyepi calendar fix (NOT_BOOSTED) and builds the
    incumbent features from train.csv and test_history.csv together, neither of which
    v4's tables have.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    offsets = list(v3.ALL_OFFSETS) + list(v5.EXTRA_OFFSETS) if offsets is None else list(offsets)
    f_data, f_test = cache_dir / f"{tag}_window.pkl", cache_dir / f"{tag}_test.pkl"
    if not rebuild and f_data.exists() and f_test.exists():
        data, test = pd.read_pickle(f_data), pd.read_pickle(f_test)
        if verbose:
            print(f"  v5 tables from cache: train {data.shape} | test {test.shape}")
    else:
        if verbose:
            print(f"  building v5 tables for {len(offsets)} offsets "
                  f"(this is the slow part) ...", flush=True)
        data, ctx = v5.load_table(offsets)
        test = v5.load_test(ctx)
        v3.encode_cats([data, test])      # one call, both frames
        data.to_pickle(f_data)
        test.to_pickle(f_test)
        if verbose:
            print(f"  v5 tables built: train {data.shape} | test {test.shape}")
    for c in v3.CAT_COLS:
        assert list(data[c].cat.categories) == list(test[c].cat.categories), (
            f"category codes for {c} differ - delete {f_data} and {f_test} and rebuild")
    return data, test


def get_tables(v3, v4, cache_dir=CACHE_DIR, rebuild=False, verbose=True):
    """Build (or load) the 545k training-window table and the aligned test table.

    Returns (data, test). Categoricals are encoded across BOTH frames together, which is
    required for the codes to line up; encoding them separately silently corrupts every
    test prediction.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    f_data, f_test = cache_dir / "v4_window_table.pkl", cache_dir / "v4_test_table.pkl"
    if not rebuild and f_data.exists() and f_test.exists():
        data, test = pd.read_pickle(f_data), pd.read_pickle(f_test)
        if verbose:
            print(f"  tables from cache: train {data.shape} | test {test.shape}")
    else:
        if verbose:
            print("  building tables via v4.load_table() / v4.load_test() ...", flush=True)
        data, ctx = v4.load_table()
        test = v4.load_test(ctx)
        v3.encode_cats([data, test])      # MUST be both frames in one call
        data.to_pickle(f_data)
        test.to_pickle(f_test)
        if verbose:
            print(f"  tables built: train {data.shape} | test {test.shape}")
    assert str(data[v3.CAT_COLS[0]].dtype) == "category", "categoricals not encoded"
    for c in v3.CAT_COLS:
        assert list(data[c].cat.categories) == list(test[c].cat.categories), (
            f"category codes for {c} differ between train and test - delete {f_data} "
            f"and {f_test} and rebuild")
    return data, test


def release_mask(data):
    """v4's validation population: clean release (offset 0) windows."""
    return (data.window == 0) & ~data.bad


def scale_weights(data, test, verbose=True):
    """Importance weights that turn validation MASE into a test-composition estimate.

    Returns (w_eval, row_weights, bin_index) where w_eval is per scale bin.
    """
    sb = pd.cut(data.scale, SCALE_EDGES, labels=False).fillna(7).astype(int)
    sbt = pd.cut(test.scale, SCALE_EDGES, labels=False).fillna(7).astype(int)
    vp = release_mask(data)
    te_share = sbt.value_counts(normalize=True).reindex(range(8)).fillna(0.0)
    va_share = sb[vp].value_counts(normalize=True).reindex(range(8)).fillna(0.0)
    w_eval = (te_share / va_share.replace(0, np.nan)).fillna(1.0).to_numpy()
    if verbose:
        print(pd.DataFrame({"bin": SCALE_LABELS, "valid_share": va_share.to_numpy(),
                            "test_share": te_share.to_numpy(),
                            "weight": w_eval}).round(4).to_string(index=False))
    return w_eval, w_eval[sb.to_numpy()], sb


def make_wmase(data, row_weights):
    """Return wmase(oof) -> (plain MASE, test-mix-weighted MASE). Select on the second."""
    def wmase(oof):
        s = data.loc[oof.index]
        err = np.abs(s.y - oof.clip(lower=0))
        w = row_weights[data.index.get_indexer(oof.index)]
        return float(err.mean()), float((err * w).sum() / w.sum())
    return wmase


def fit_post_weighted(frame, v3, v4, weighted=True):
    """v4's fit_post, but minimising the test-mix-weighted objective.

    `frame` needs columns y, A, B, p0, h and (when weighted) w.
    Returns dict(w=blend weight, zm=per-p0-decile multipliers, hm=per-horizon multipliers).
    """
    y = frame.y.to_numpy()
    ww = frame.w.to_numpy() if weighted else np.ones(len(frame))
    best = lambda grid, err: grid[int(np.argmin([err(g) for g in grid]))]
    a, b = frame.A.to_numpy(), frame.B.to_numpy()
    w0 = best(v4.W_GRID, lambda w: (ww * np.abs(y - w * a - (1 - w) * b)).sum())
    r = w0 * a + (1 - w0) * b
    bins, zm = v4.zero_bin(frame.p0.to_numpy()), np.ones(10)
    for k in range(10):
        m = bins == k
        if m.sum() >= 200:
            zm[k] = best(v4.ZERO_GRID, lambda g: (ww[m] * np.abs(y[m] - g * r[m])).sum())
    r = r * zm[bins]
    hm = {}
    for k in v3.HORIZONS:
        m = frame.h.to_numpy() == k
        if m.sum():
            hm[k] = best(v4.H_GRID, lambda g: (ww[m] * np.abs(y[m] - g * r[m])).sum())
    return {"w": float(w0), "zm": zm, "hm": hm}


def apply_post(frame, post, v4):
    """Apply fit_post_weighted output to a frame with columns A, B, p0, h."""
    r = post["w"] * frame.A.to_numpy() + (1 - post["w"]) * frame.B.to_numpy()
    r = r * post["zm"][v4.zero_bin(frame.p0.to_numpy())]
    return r * pd.Series(frame.h.to_numpy()).map(post["hm"]).fillna(1.0).to_numpy()
