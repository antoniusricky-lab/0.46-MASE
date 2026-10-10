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


# ------------------------------------ CELL 25 P5: convexity of the cohort multiplier
# A cohort multiplier m scales the predicted ratio r, so with u = y/scale the cohort's
# MASE contribution is S(m) = (1/N) sum |u - m*r|. That is a sum of absolute values of
# functions AFFINE in m, hence CONVEX in m, minimised at the r-weighted median of u/r.
#
# The consequence that matters: two probes at equal height sit on opposite walls of the
# V and BRACKET an interior optimum. v23 read S(0.5) ~ S(1.0) as "no effect, cohort
# closed", which is the opposite of what convexity implies, and shipped it.
#
# The v23 estimator f = (1 + G/D)/2 assumed every displaced row gains or loses the FULL
# displacement. That holds only when the truth lies OUTSIDE the interval between the two
# predictions. Its fixture placed truth only at k = 0 and k = 3, so it never built a row
# in the 0.5 < k < 1 band where the assumption fails -- and agreement to 1e-9 therefore
# proved only that the formula is exact where it is exact. These tests sweep k ACROSS
# that band, which is the regime that actually decides the recommendation.
def S(m, u, r, n_rows):
    return float(np.abs(u - m * r).sum() / n_rows)


# -- 1. the exact break-even: at k = (1 + m^a)/2 the two probes score identically
for _a in (1.0, 0.75, 0.5, 0.25):
    _r, _m = 0.41, 0.5
    _k_be = (1 + _m ** _a) / 2
    _u = _k_be * _r
    check(f"break-even k=(1+m^a)/2 ties the probes at a={_a}",
          abs(abs(_u - _r) - abs(_u - _r * _m ** _a)) < 1e-12,
          "the two multipliers should score this row identically")

# the multiplier that predicts the break-even row exactly is k^(1/a), ~0.72-0.75
_imp = [((1 + 0.5 ** a) / 2) ** (1 / a) for a in (1.0, 0.75, 0.5, 0.25)]
check("implied optimum is 0.70-0.76 across the whole exponent range",
      all(0.70 <= v <= 0.76 for v in _imp),
      f"implied multipliers {[round(v, 4) for v in _imp]}")

# -- 2. v23's estimator is WRONG inside the band, and right outside it ----------------
# This is the test the v23 fixture structurally could not perform.
_r = 0.41
_bad_in_band, _ok_outside = [], []
for _k in (0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.5, 3.0):
    _u = _k * _r
    _true_gain = abs(_u - _r) - abs(_u - 0.5 * _r)      # + == halving improves
    _d = 0.5 * _r                                        # the displacement ceiling
    _binary = _d if abs(_u - 0.5 * _r) < abs(_u - _r) else -_d   # v23's assumption
    (_bad_in_band if 0.5 < _k < 1.0 else _ok_outside).append(
        abs(_true_gain - _binary))
check("v23 estimator is exact OUTSIDE the 0.5<k<1 band",
      max(_ok_outside) < 1e-12, f"max error {max(_ok_outside):.2e}")
check("v23 estimator is WRONG INSIDE the 0.5<k<1 band (the untested regime)",
      max(_bad_in_band) > 0.1 * 0.5 * _r,
      f"max error {max(_bad_in_band):.4f} -- if this is small the test has no power")

# -- 3. S(m) is convex, so equal endpoints bracket an interior optimum ----------------
_rng = np.random.default_rng(11)
_n, _N = 2776, 72611
for _sig in (0.25, 0.6, 1.0, 1.4):
    _r = np.clip(_rng.lognormal(np.log(0.41), 0.6, _n), 0.01, None)
    _k = _rng.lognormal(0, _sig, _n) / np.exp(_sig ** 2 / 2)
    # centre k so that S(1.0) - S(0.5) == +0.00005, the measured null
    _lo, _hi = 0.05, 5.0
    for _ in range(100):
        _mid = (_lo + _hi) / 2
        _u = _mid * _k * _r
        if S(1.0, _u, _r, _N) - S(0.5, _u, _r, _N) > 0.00005:
            _lo = _mid
        else:
            _hi = _mid
    _u = (_lo + _hi) / 2 * _k * _r
    _grid = np.linspace(0.2, 1.3, 1101)
    _vals = np.array([S(m, _u, _r, _N) for m in _grid])
    _mstar = float(_grid[int(_vals.argmin())])
    _gain = S(1.0, _u, _r, _N) - _vals.min()
    check(f"null brackets an interior optimum, not m=1, at k-spread {_sig}",
          0.70 <= _mstar <= 0.80 and _gain > 0,
          f"m*={_mstar:.3f} gain={_gain:.5f} -- v23 claimed the optimum was m=1")
    # midpoint convexity: S(0.75) can never exceed the mean of the two endpoints
    check(f"convexity caps the m=0.75 downside at k-spread {_sig}",
          S(0.75, _u, _r, _N) <= (S(1.0, _u, _r, _N) + S(0.5, _u, _r, _N)) / 2 + 1e-12,
          "S(0.75) exceeded the endpoint mean, so S is not convex as claimed")

# -- 4. displacement really does bound the gain, and scales with |m - 1| -------------
_r = np.clip(_rng.lognormal(np.log(0.41), 0.6, _n), 0.01, None)
_u = 0.5 * _r                                   # truth well below both predictions
_D50 = float((np.abs(0.50 - 1.0) * _r).sum() / _N)
_D75 = float((np.abs(0.75 - 1.0) * _r).sum() / _N)
check("displacement ceiling is proportional to |m - 1|",
      abs(_D75 / _D50 - 0.5) < 1e-12, f"ratio {_D75 / _D50:.6f}, expected 0.5")
check("no truth arrangement beats the displacement ceiling",
      S(1.0, _u, _r, _N) - S(0.5, _u, _r, _N) <= _D50 + 1e-12,
      "gain exceeded the ceiling")

# -- 5. the conclusion v23 drew is ruled out by the magnitude it was drawn from -------
# If predictions were already optimal (k ~ 1), halving forfeits the WHOLE ceiling.
_u = 1.0 * _r
check("already-optimal predictions would have LOST the full ceiling",
      abs((S(0.5, _u, _r, _N) - S(1.0, _u, _r, _N)) - _D50) < 1e-12,
      "halving an optimal prediction should cost exactly D, so a null rules it out")


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
