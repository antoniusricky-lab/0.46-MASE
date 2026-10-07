"""Unit tests for the new logic in CELLS 15 and 16.

Those cells only run under RUN_HEAVY, which is reserved for the local machine, so their
bodies are never exercised by smoke_test.py. These tests pin down the pure functions
(absorbing-zero rule, weighted median, monotone calibration curve) and the pivot/cross-fit
index alignment on small synthetic arrays, so a bug does not surface only after a long
local run.

Run:  python test_v5_logic.py
"""
import numpy as np
import pandas as pd

HORIZONS = range(4, 11)
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


# ----------------------------------------------------------- absorbing-zero rule
def absorb(mat, tau):
    below = np.nan_to_num(mat, nan=np.inf) < tau
    return np.where(np.logical_or.accumulate(below, axis=1), 0.0, mat)


print("\nabsorb()")
m = np.array([[0.9, 0.5, 0.01, 0.4, 0.8]])
out = absorb(m, 0.1)
check("zeros the trigger horizon and everything after",
      np.allclose(out, [[0.9, 0.5, 0.0, 0.0, 0.0]]), str(out))
check("tau=0 is a no-op", np.allclose(absorb(m, 0.0), m), str(absorb(m, 0.0)))
check("tau above everything zeros the whole row",
      np.allclose(absorb(m, 10.0), np.zeros_like(m)))
m2 = np.array([[0.5, np.nan, 0.6]])
check("NaN does not trigger absorption (treated as +inf)",
      np.allclose(np.nan_to_num(absorb(m2, 0.1), nan=-1), [[0.5, -1, 0.6]]), str(absorb(m2, 0.1)))
m3 = np.array([[0.2, 0.3], [0.05, 0.9]])
check("rows are independent",
      np.allclose(absorb(m3, 0.1), [[0.2, 0.3], [0.0, 0.0]]), str(absorb(m3, 0.1)))


# ------------------------------------------------------------- weighted median
def wmedian(v, w):
    if len(v) == 0:
        return np.nan
    o = np.argsort(v)
    v, w = np.asarray(v)[o], np.asarray(w)[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


print("\nwmedian()")
check("uniform weights match np.median on odd length",
      wmedian([1, 5, 3], [1, 1, 1]) == 3.0, str(wmedian([1, 5, 3], [1, 1, 1])))
check("weight concentrated on one value returns it",
      wmedian([1.0, 9.0], [0.001, 1000.0]) == 9.0)
check("empty input gives NaN", np.isnan(wmedian([], [])))
check("zero total weight falls back to the plain median",
      wmedian([2.0, 4.0, 6.0], [0.0, 0.0, 0.0]) == 4.0)
check("unsorted input handled", wmedian([7, 1, 2, 9], [1, 1, 1, 1]) in (2.0, 7.0))


# --------------------------------------------------- monotone calibration curve
def fit_curve(r, y, w, nbins=40, min_n=80):
    q = np.unique(np.quantile(r, np.linspace(0, 1, nbins + 1)))
    if len(q) < 3:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    idx = np.clip(np.searchsorted(q, r, side="right") - 1, 0, len(q) - 2)
    xs, ys = [], []
    for b in range(len(q) - 1):
        m = idx == b
        if m.sum() >= min_n:
            xs.append(float(np.median(r[m])))
            ys.append(wmedian(y[m], w[m]))
    if len(xs) < 3:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    xs, ys = np.array(xs), np.maximum.accumulate(np.array(ys))
    return xs, ys


print("\nfit_curve()")
rng = np.random.default_rng(0)
r = rng.uniform(0, 2, 20000)
y = 1.5 * r + rng.normal(0, 0.05, 20000)          # truth is 1.5x the prediction
w = np.ones_like(r)
xs, ys = fit_curve(r, y, w)
check("knots are non-decreasing in y", bool((np.diff(ys) >= -1e-12).all()))
check("knots are increasing in x", bool((np.diff(xs) > 0).all()))
mapped = np.interp([0.5, 1.0, 1.5], xs, ys)
check("recovers the 1.5x relationship", np.allclose(mapped, [0.75, 1.5, 2.25], atol=0.08),
      str(mapped))
xs2, ys2 = fit_curve(np.ones(500), np.ones(500), np.ones(500))
check("degenerate input returns identity", np.allclose(xs2, [0, 1]) and np.allclose(ys2, [0, 1]))
r3 = rng.uniform(0, 1, 100)
xs3, ys3 = fit_curve(r3, r3, np.ones(100), min_n=80)
check("too few points per bin returns identity",
      np.allclose(xs3, [0, 1]) and np.allclose(ys3, [0, 1]))
# a deliberately non-monotone truth must still produce a monotone curve
y4 = np.where(r < 1.0, 2.0, 0.5) + rng.normal(0, 0.01, 20000)
_, ys4 = fit_curve(r, y4, w)
check("non-monotone truth is forced monotone", bool((np.diff(ys4) >= -1e-12).all()))


# ------------------------------------------- pivot alignment used by Z4 / R1
print("\npivot alignment")
rows = []
for mv in ("A", "B", "C"):
    for ci in ("c1", "c2"):
        for h in HORIZONS:
            rows.append(dict(mv=mv, ci=ci, h=h, y=rng.uniform(0, 2),
                             r=rng.uniform(0, 2), w=rng.uniform(0.5, 3)))
fr = pd.DataFrame(rows)
fr = fr.drop(index=fr.index[3])                       # puncture one (pair, h) cell
cols = list(HORIZONS)
Yp = fr.pivot_table(index=["mv", "ci"], columns="h", values="y")
Rp = fr.pivot_table(index=["mv", "ci"], columns="h", values="r")
Wp = fr.pivot_table(index=["mv", "ci"], columns="h", values="w")
ix = Yp.index.union(Rp.index).union(Wp.index)
Yp, Rp, Wp = (x.reindex(index=ix, columns=cols) for x in (Yp, Rp, Wp))
check("pivots share an index after reindex",
      Yp.index.equals(Rp.index) and Rp.index.equals(Wp.index))
check("pivots share column order", list(Yp.columns) == cols == list(Rp.columns))
check("the punctured cell is NaN in every pivot",
      bool(np.isnan(Yp.to_numpy()).sum() == 1 and np.isnan(Rp.to_numpy()).sum() == 1))
ok = ~np.isnan(Rp.to_numpy()) & ~np.isnan(Yp.to_numpy())
check("the ok mask drops exactly the punctured cell", int((~ok).sum()) == 1)
Wa = np.nan_to_num(Wp.to_numpy())
num = float((Wa[ok] * np.abs(Yp.to_numpy()[ok] - Rp.to_numpy()[ok])).sum())
check("weighted error over the ok mask is finite and positive", np.isfinite(num) and num > 0)
mv_of_row = Yp.index.get_level_values(0).to_numpy()
in_val = np.isin(mv_of_row, ["A"])
check("fold mask selects the right rows", int(in_val.sum()) == 2)
va_mask = np.repeat(in_val[:, None], len(cols), axis=1)
check("broadcast fold mask has the pivot's shape", va_mask.shape == Yp.to_numpy().shape)

# --------------------------------------------- the tested code is the shipped code
# These tests are worthless if the notebook's copies have drifted, so compare bodies.
print("\nnotebook parity")
import json
import re
import textwrap


def body_from_notebook(name, path="cinema_v5.ipynb"):
    nb = json.load(open(path, encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    m = re.search(rf"^([ \t]*)def {name}\(.*?\n(?:\1[ \t]+.*\n|[ \t]*\n)*", src, re.M)
    if not m:
        return None
    return textwrap.dedent(m.group(0)).strip()


def norm(s):
    # compare executable content only: drop blank lines, whole-line comments and
    # trailing inline comments (none of these functions contain a '#' inside a string)
    out = []
    for ln in s.splitlines():
        code = ln.split("#", 1)[0].rstrip()
        if code.strip():
            out.append(code)
    return out


try:
    for fn in (absorb, wmedian, fit_curve):
        nb_src = body_from_notebook(fn.__name__)
        check(f"{fn.__name__}() matches the notebook", nb_src is not None
              and norm(nb_src) == norm(textwrap.dedent(
                  __import__("inspect").getsource(fn)).strip()),
              "notebook copy differs - re-sync before trusting these tests")
except FileNotFoundError:
    print("  SKIP  cinema_v5.ipynb not found")

print(f"\n{'=' * 60}")
print("ALL PASS" if not fails else f"FAILURES: {fails}")
print(f"{'=' * 60}")
raise SystemExit(1 if fails else 0)
