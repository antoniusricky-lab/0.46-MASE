"""Generates cinema_v5.ipynb. Run:  python build_notebook.py"""
import json

NB = "cinema_v5.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# ======================================================================= intro
md(r"""
# Cinema ticket forecasting — v5 working notebook

**Goal:** close the gap from LB 0.43715 to the 0.34456 leader.

**How to use:** every cell is independent once `CELL 1`-`CELL 3` have run. Cells are named
`# CELL n — NAME` on their first line. Run them in the order I give you, paste the printed
output back, and do not run anything else in between. Expensive steps cache to `cache/` and
`results/`, so a kernel restart never loses work.

### Round 1 (Phase 0) — run CELL 1, 2, 3, 4, 5, 6. No model training, ~2 minutes total.

Round 1 deliberately skips the error anatomy, because that needs OOF predictions from the v4
model and **`cinema_forecasting_v3.py` is missing from the repo** (`0.43715.py` is v4 and starts
with `import cinema_forecasting_v3 as v3`). Instead these cells measure the thing that the
evidence says actually costs us the score: the calendar.
""")

# ====================================================================== CELL 1
md("## CELL 1 — CONFIG")
code(r"""
# CELL 1 — CONFIG
import os, sys, glob, json, time, pickle, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 120)

SEED           = 2026
N_JOBS         = max(1, (os.cpu_count() or 4) - 1)
SUBSAMPLE_ROWS = None      # e.g. 200_000 -> subsample training windows for a quick run
RUN_HEAVY      = False     # gate for anything that takes more than ~3 minutes

# ---- paths: set DATA_DIR if the csv files live somewhere else ----------------
DATA_DIR  = Path(os.environ.get("DATA_DIR", "."))
OUT_DIR   = Path("results");     OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache");       CACHE_DIR.mkdir(exist_ok=True)
SUB_DIR   = Path("submissions"); SUB_DIR.mkdir(exist_ok=True)

HORIZONS = [4, 5, 6, 7, 8, 9, 10]

# corrupted stretch inside train.csv (v3 finding)
BAD_START, BAD_END = pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-16")

# ---- Islamic calendar anchors -----------------------------------------------
# holidays.csv labels 2026-03-21 AND 2026-03-22 as "Idulfitri 1447 H", so Eid is a
# two-day holiday and day 0 is the first of them. By the same convention Idulfitri
# 1446 H was 2025-03-31 + 2025-04-01, i.e. train.csv starts on Eid day 1.
EID_1446 = pd.Timestamp("2025-03-31")   # train.csv begins the next day
EID_1447 = pd.Timestamp("2026-03-21")   # inside the test period
RAMADAN_1447 = (pd.Timestamp("2026-02-19"), pd.Timestamp("2026-03-20"))
RAMADAN_1446 = (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-30"))  # entirely before train
SCHOOL_BREAKS = [(pd.Timestamp("2025-06-21"), pd.Timestamp("2025-07-13")),
                 (pd.Timestamp("2025-12-20"), pd.Timestamp("2026-01-04"))]
# "normal" reference stretches used to normalise market level in each year
REF_2025 = (pd.Timestamp("2025-05-05"), pd.Timestamp("2025-05-31"))
REF_2026 = (pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-14"))


def find_file(stem):
    # Locate <stem>.csv, tolerating browser-style duplicates like 'train (1).csv'.
    exact = DATA_DIR / (stem + ".csv")
    if exact.exists() and exact.stat().st_size > 1000:
        return exact
    pat = re.compile(r"^" + re.escape(stem) + r"( \(\d+\))?\.csv$")
    hits = sorted((p for p in DATA_DIR.glob("*.csv") if pat.match(p.name)),
                  key=lambda p: -p.stat().st_size)
    if hits:
        return hits[0]
    raise FileNotFoundError(stem + ".csv not found in " + str(DATA_DIR.resolve()))


STEMS = ["train", "test_history", "test", "sample_submission", "movies", "holidays", "ticket_prices"]
print("CONFIG ok")
print("  DATA_DIR =", DATA_DIR.resolve())
for _s in STEMS:
    try:
        _p = find_file(_s)
        print(f"  {_s + '.csv':22s} -> {_p.name:24s} {_p.stat().st_size / 1e6:8.2f} MB")
    except FileNotFoundError:
        print(f"  {_s + '.csv':22s} -> *** MISSING ***")
print(f"  SEED={SEED}  N_JOBS={N_JOBS}  SUBSAMPLE_ROWS={SUBSAMPLE_ROWS}  RUN_HEAVY={RUN_HEAVY}")
""")

# ====================================================================== CELL 2
md("## CELL 2 — environment (CPU / RAM / versions) and the v4 reproducibility check")
code(r"""
# CELL 2 — ENVIRONMENT + v3 MODULE CHECK
t0 = time.time()
import platform, importlib

print("python   :", sys.version.split()[0], "|", platform.platform())
print("cpu count:", os.cpu_count())
try:
    import psutil
    _vm = psutil.virtual_memory()
    print(f"RAM      : total {_vm.total / 2**30:.1f} GiB | available {_vm.available / 2**30:.1f} GiB")
except Exception:
    try:
        _mi = {}
        for _l in open("/proc/meminfo"):
            _mi[_l.split(":")[0]] = int(_l.split()[1])
        print(f"RAM      : total {_mi['MemTotal'] / 2**20:.1f} GiB | available {_mi['MemAvailable'] / 2**20:.1f} GiB")
    except Exception:
        print("RAM      : unknown (pip install psutil)")

for _m in ["numpy", "pandas", "sklearn", "lightgbm", "catboost", "pyarrow", "scipy", "tqdm"]:
    try:
        print(f"{_m:9s}:", importlib.import_module(_m).__version__)
    except Exception:
        print(f"{_m:9s}: NOT INSTALLED")

# --- can we reproduce v4? 0.43715.py is v4 and does `import cinema_forecasting_v3 as v3`
V3_SYMBOLS = ["SEED", "HORIZONS", "BASE", "CAL_V3", "CAT_COLS", "PARAMS", "OPEN_OFFSETS",
              "ALL_OFFSETS", "BAD_START", "BAD_END", "load_data", "detect_release_dates",
              "make_calendar", "prepare", "build_training_windows", "build_samples",
              "encode_cats", "make_folds"]
try:
    import cinema_forecasting_v3 as v3
    _missing = [s for s in V3_SYMBOLS if not hasattr(v3, s)]
    print("\nv3 module : FOUND ->", v3.__file__)
    print("  missing symbols:", _missing or "none")
    HAVE_V3 = not _missing
except Exception as e:
    print("\nv3 module : NOT IMPORTABLE ->", type(e).__name__, e)
    print("  v4 cannot be re-run until cinema_forecasting_v3.py sits next to this notebook.")
    print("  symbols v4 needs from it:", ", ".join(V3_SYMBOLS))
    HAVE_V3 = False
print("HAVE_V3 =", HAVE_V3)
print(f"\n[CELL 2] {time.time() - t0:.1f}s")
""")

# ====================================================================== CELL 3
md("## CELL 3 — load all tables and cache them")
code(r"""
# CELL 3 — LOAD + CACHE
t0 = time.time()


def _read(stem, dates=()):
    df = pd.read_csv(find_file(stem))
    for c in dates:
        df[c] = pd.to_datetime(df[c])
    for c in ("cinema_ids", "movie_title", "city_name"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


train    = _read("train", ["date_show"])
th       = _read("test_history", ["date_show"])
test     = _read("test", ["date_show"])
sample   = _read("sample_submission")
movies   = _read("movies")
holidays = _read("holidays", ["date"])
prices   = _read("ticket_prices")

for _n, _d in [("train", train), ("test_history", th), ("test", test),
               ("sample_submission", sample), ("movies", movies),
               ("holidays", holidays), ("ticket_prices", prices)]:
    _rng = ""
    for _c in ("date_show", "date"):
        if _c in _d.columns:
            _rng = f" | {_c} {_d[_c].min().date()} -> {_d[_c].max().date()}"
    print(f"{_n:20s} {str(_d.shape):16s}{_rng}")

assert len(train) > 1000, f"train.csv looks destroyed ({len(train)} rows) - on `main` it is 2 bytes"
if len(train) < 100_000:
    print(f"\n*** WARNING: train.csv has only {len(train):,} rows, expected 138,959 ***")
_nt, _nh = train.isna().sum(), th.isna().sum()
print("\nnulls train       :", dict(_nt[_nt > 0]) or "none")
print("nulls test_history:", dict(_nh[_nh > 0]) or "none")
print("total_ticket <= 0 -> train:", int((train.total_ticket <= 0).sum()),
      "| test_history:", int((th.total_ticket <= 0).sum()))
print("duplicate (date,cluster,movie) -> train:",
      int(train.duplicated(["date_show", "cinema_ids", "movie_title"]).sum()),
      "| test_history:", int(th.duplicated(["date_show", "cinema_ids", "movie_title"]).sum()))
print("sample_submission id == test id:", bool(sample.id.equals(test.id)))

# D1 per test film, used by every later cell
d1_test = (test.groupby("movie_title").date_show.min() - pd.Timedelta(days=3)).rename("d1")
test["d1"] = test.movie_title.map(d1_test)
test["off"] = (test.date_show - test.d1).dt.days + 1
print("\nd1_test built for", len(d1_test), "films | test offsets:",
      test.off.value_counts().sort_index().to_dict())

for _n, _d in [("train", train), ("test_history", th), ("test", test)]:
    try:
        _d.to_parquet(CACHE_DIR / f"{_n}.parquet", index=False)
    except Exception:
        _d.to_pickle(CACHE_DIR / f"{_n}.pkl")
print(f"\n[CELL 3] cached to {CACHE_DIR}/ in {time.time() - t0:.1f}s")
""")

# ====================================================================== CELL 4
md(r"""
## CELL 4 — AUDIT A: re-verify the structural rules, then size the leverage

A previous analysis pass (PR #1 `FINDINGS.md`) **proved** two structural rules against the raw
files with a pure-stdlib reimplementation. This cell re-checks them in pandas so we are certain
they hold, then measures the thing that follows from them: which rows can actually move the score.

* **R1** a pair is forecast *iff* it sold at least one ticket on D3 (no expansions are ever asked for)
* **R2** `D1` is the film's wide-release date, so **every test window is an opening window**

`001`/`011` = the pair's activity pattern over D1,D2,D3. These were measured at 8% of rows
carrying 25% of the error, because `scale` divides by 3 even when the pair screened on fewer days.
""")
code(r"""
# CELL 4 — AUDIT A: STRUCTURAL RULES + LEVERAGE
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

# ---------------------------------------------- A1 pair activity patterns in test
line("A1  R1: is a pair forecast exactly when it was active on D3?")
_th = th.copy()
_th["d1"] = _th.movie_title.map(d1_test)
_th["off"] = (_th.date_show - _th.d1).dt.days + 1
print("test_history offsets relative to D1 (expect only 1,2,3):",
      _th.off.value_counts().sort_index().to_dict())
_w3 = _th[_th.off.between(1, 3)]
act = (_w3.assign(on=lambda d: (d.total_ticket > 0).astype(int))
       .pivot_table(index=["movie_title", "cinema_ids"], columns="off", values="on", aggfunc="max")
       .reindex(columns=[1, 2, 3]).fillna(0).astype(int))
act["pattern"] = act[1].astype(str) + act[2].astype(str) + act[3].astype(str)
pairs_te = set(map(tuple, test[["movie_title", "cinema_ids"]].drop_duplicates().to_numpy()))
act["in_test"] = [tuple(i) in pairs_te for i in act.index]
print("\npairs with any D1-D3 history:", len(act), "| pairs in test.csv:", len(pairs_te))
print("\npattern x in_test:")
print(pd.crosstab(act.pattern, act.in_test).to_string())
_d3_on = act[3] == 1
print("\nR1 'active on D3' == 'in test.csv' ->", bool((_d3_on == act.in_test).all()))
print("test pairs with NO D1-D3 history at all:", len(pairs_te - set(map(tuple, act.index))))

# ------------------------------------------------- A2 pattern mix of scored rows
line("A2  pattern mix of the scored rows, and 1/scale leverage")


def hitung_skala(history):
    total = history.groupby(["movie_title", "cinema_ids"]).total_ticket.sum()
    return (total / 3).clip(lower=1).rename("scale")


scale_off = hitung_skala(th)
test_scale = test.join(scale_off, on=["movie_title", "cinema_ids"]).scale
print("test rows whose OFFICIAL scale is NaN:", int(test_scale.isna().sum()))
pat = act.pattern.rename("pattern")
test_pat = test.join(pat, on=["movie_title", "cinema_ids"]).pattern
_n_act = test_pat.str.count("1")
tab = pd.DataFrame({"pattern": test_pat, "scale": test_scale, "inv": 1.0 / test_scale})
g = tab.groupby("pattern").agg(rows=("scale", "size"), med_scale=("scale", "median"),
                               leverage=("inv", "sum"))
g["row_share"] = g.rows / len(tab)
g["leverage_share"] = g.leverage / tab.inv.sum()
print("\nby D1-D3 activity pattern:")
print(g[["rows", "row_share", "med_scale", "leverage_share"]].round(4).to_string())
print("\nreading: leverage_share = share of the final MASE that one ticket of error in that")
print("bucket produces. scale divides by 3 even for a pair that screened 1 day, so 001 pairs")
print("have an understated scale and a true y/scale well above 1.")

print("\nofficial scale distribution over test rows:")
print(test_scale.describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .99]).round(2).to_string())
_s, _inv = test_scale.dropna(), 1.0 / test_scale.dropna()
print("\n  scale<=T   share of rows   share of total leverage")
for _t in (1, 2, 3, 5, 10, 25, 50):
    _m = _s <= _t
    print(f"      <={_t:<4d}   {_m.mean():12.2%}   {_inv[_m].sum() / _inv.sum():22.2%}")

# ----------------------------------------------- A3 R2: D1 is the release date
line("A3  R2: is D1 the film's wide-release date?")
print("films:", test.movie_title.nunique(), "| distinct D1 dates:", d1_test.nunique(),
      "| D1 range", d1_test.min().date(), "->", d1_test.max().date())
_dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_dc = d1_test.dt.dayofweek.value_counts().sort_index()
print("D1 weekday:", {_dow[i]: f"{v} ({v / len(d1_test):.0%})" for i, v in _dc.items()})
print("films sharing one D1 date:", d1_test.value_counts().value_counts().sort_index().to_dict())
_ncl = _w3.groupby(["movie_title", "off"]).cinema_ids.nunique().unstack()
print("mean distinct clusters on D1/D2/D3:", _ncl.mean().round(1).to_dict())
print("films whose D1 has the max cluster count:",
      int((_ncl.idxmax(axis=1) == 1).sum()), "/", len(_ncl))

# ------------------------------------------------------- A4 overlap with train
line("A4  overlap of test films / clusters with train.csv")
_ov = sorted(set(test.movie_title) & set(train.movie_title))
_rows_ov = int(test.movie_title.isin(_ov).sum())
print(f"test films also in train.csv: {len(_ov)} / {test.movie_title.nunique()}"
      f" -> {_rows_ov:,} rows ({_rows_ov / len(test):.2%})")
print("  examples:", _ov[:6])
print("clusters in test:", test.cinema_ids.nunique(),
      "| in train:", len(set(test.cinema_ids) & set(train.cinema_ids)),
      "| UNSEEN:", len(set(test.cinema_ids) - set(train.cinema_ids)))
act.to_pickle(CACHE_DIR / "test_pair_patterns.pkl")
print(f"\n[CELL 4] {time.time() - t0:.1f}s")
""")

# ====================================================================== CELL 5
md(r"""
## CELL 5 — AUDIT B: calendar regime composition and straddle exposure

`FINDINGS.md` §1 localised the CV-to-LB gap here: anomaly days were 7.1% of train rows but
31.8% of test rows, and MASE on them was 2.37x the ordinary-day MASE.

Because MASE divides by the pair's own `scale`, a *uniform* level shift cancels out. What does
not cancel is a window whose **target days sit in a different regime from its own D1-D3**. This
cell measures that straddle exposure and lists the release cohorts that carry it.
""")
code(r"""
# CELL 5 — AUDIT B: REGIME COMPOSITION + STRADDLE EXPOSURE
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

_hol = (holidays.drop_duplicates("date")
        .assign(holiday_tipe=lambda d: d.holiday_tipe.astype(str).str.strip().str.lower())
        .set_index("date"))


def regime(dates):
    # One row of calendar facts per date. 'eid_k' is days since Idulfitri day 0.
    d = pd.DatetimeIndex(pd.Series(np.asarray(dates)).to_numpy())
    f = pd.DataFrame(index=np.arange(len(d)))
    f["dow"] = d.dayofweek
    f["weekend"] = np.isin(d.dayofweek, [5, 6]).astype(int)
    f["holiday"] = (pd.Series(_hol.holiday_tipe.reindex(d).to_numpy())
                    .fillna("normal").eq("holiday").astype(int).to_numpy())
    k46 = (d - EID_1446).days.to_numpy()
    k47 = (d - EID_1447).days.to_numpy()
    f["eid_k"] = np.where(np.abs(k46) <= 45, k46, np.where(np.abs(k47) <= 45, k47, np.nan))
    f["ramadan"] = (((d >= RAMADAN_1447[0]) & (d <= RAMADAN_1447[1])) |
                    ((d >= RAMADAN_1446[0]) & (d <= RAMADAN_1446[1]))).astype(int)
    f["eid_week"] = ((f.eid_k >= 0) & (f.eid_k <= 6)).astype(int)
    _sch = np.zeros(len(d), dtype=int)
    for a, b in SCHOOL_BREAKS:
        _sch = _sch | ((d >= a) & (d <= b)).astype(int)
    f["school"] = _sch
    f["label"] = np.where(f.eid_week == 1, "eid_week",
                 np.where(f.ramadan == 1, "ramadan",
                 np.where(f.school == 1, "school_break",
                 np.where(f.holiday == 1, "holiday", "ordinary"))))
    return f


# ------------------------------------------------------ B1 test composition
line("B1  calendar regime of the scored test rows")
rt = regime(test.date_show)
_c = rt.label.value_counts()
print(pd.DataFrame({"rows": _c, "share": (_c / len(rt)).round(4)}).to_string())
print("\nweekend share of test rows:", f"{rt.weekend.mean():.2%}")
print("\ntest rows per month:")
print(test.date_show.dt.to_period("M").value_counts().sort_index().to_string())

# --------------------------------------------------------- B2 straddle rows
line("B2  straddle exposure: target-day regime != the window's own D1-D3 regime")
r1, r2, r3 = regime(test.d1), regime(test.d1 + pd.Timedelta(days=1)), regime(test.d1 + pd.Timedelta(days=2))
hist_lab = pd.DataFrame({"a": r1.label, "b": r2.label, "c": r3.label})
same = (hist_lab.a == rt.label) | (hist_lab.b == rt.label) | (hist_lab.c == rt.label)
print(f"rows whose target regime appears nowhere in its own D1-D3: {int((~same).sum()):,}"
      f" ({(~same).mean():.2%})")
print("\nbreakdown of those straddle rows by target regime:")
print(rt.label[~same].value_counts().to_string())
print("\ncross-tab: D1 regime (rows) x target regime (cols), share of all test rows")
print((pd.crosstab(hist_lab.a, rt.label) / len(rt)).round(4).to_string())

# --------------------------------------- B3 release cohorts carrying the risk
line("B3  release cohorts (by D1 date) ranked by anomaly exposure")
coh = pd.DataFrame({"d1": test.d1, "label": rt.label, "eid_k": rt.eid_k})
_t = (coh.groupby("d1").agg(rows=("label", "size"),
                            eid=("label", lambda v: float((v == "eid_week").mean())),
                            ram=("label", lambda v: float((v == "ramadan").mean())),
                            sch=("label", lambda v: float((v == "school_break").mean()))))
_t["films"] = test.groupby("d1").movie_title.nunique()
_t["row_share"] = _t.rows / len(test)
_t["anomaly"] = _t.eid + _t.ram + _t.sch
print(_t.sort_values("anomaly", ascending=False).head(14).round(4).to_string())
print("\nthe Eid cohort in detail (films whose targets land in Idulfitri 1447 H week):")
_eid_d1 = _t[_t.eid > 0].index
_m = test.d1.isin(_eid_d1)
print("  D1 dates:", [str(d.date()) for d in _eid_d1], "| films:",
      test.loc[_m, "movie_title"].nunique(), "| pairs:",
      len(test.loc[_m].groupby(["movie_title", "cinema_ids"])), "| rows:", int(_m.sum()),
      f"({_m.mean():.2%} of test)")
print("  target days by eid_k:", rt.eid_k[_m].value_counts().sort_index().to_dict())
print("  films:", sorted(test.loc[_m, "movie_title"].unique())[:12])
np.save(CACHE_DIR / "test_regime_label.npy", rt.label.to_numpy())
print(f"\n[CELL 5] {time.time() - t0:.1f}s")
""")

# ====================================================================== CELL 6
md(r"""
## CELL 6 — AUDIT C: estimate the market-level factors we are missing

This is the cell that matters. The decomposition I want to act on is

```
y / scale  ~=  normal_decay(h, D1-dow)  x  market_level(target day) / market_level(D1..D3)
```

The first term is what the model already learns from ordinary days. The second term is what
train.csv cannot teach, and it is exactly what breaks on the Eid cohort: those films observe
D1-D3 in the **last days of Ramadan** (depressed) and are forecast across **Idulfitri week**
(elevated), so the ratio is pushed up twice over.

Both halves are measurable from data we already have:

* `market_level(Eid+k)` — from **train.csv**, since 2025-04-01 is Eid+1 of Idulfitri 1446 H
* `market_level(late Ramadan 2026)` — from **test_history.csv**, which runs to 2026-03-20

It also checks §8 of `FINDINGS.md`: how often another film in `test_history.csv` observes the
exact cluster and date we are asked to forecast.
""")
code(r"""
# CELL 6 — AUDIT C: MARKET LEVEL (EID 2025, RAMADAN 2026) + CROSS-FILM COVERAGE
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)


def daily_level(df, label):
    # Scale-free market level per date. tps = tickets per show, occ = mean occupancy.
    g = df.groupby("date_show").agg(tickets=("total_ticket", "sum"),
                                    shows=("total_show", "sum"),
                                    occ=("occupation_rate", "mean"),
                                    rows=("total_ticket", "size"),
                                    films=("movie_title", "nunique"),
                                    clusters=("cinema_ids", "nunique"))
    g["tps"] = g.tickets / g.shows.replace(0, np.nan)
    g["src"] = label
    return g


def dow_factor(g, ref_window, col):
    # level(date) / median level over the same weekday inside a normal reference window.
    ref = g[(g.index >= ref_window[0]) & (g.index <= ref_window[1])]
    base = ref.groupby(ref.index.dayofweek)[col].median()
    return g[col] / g.index.dayofweek.map(base).to_numpy()


lvl_tr = daily_level(train, "train")
lvl_th = daily_level(th, "test_history")

# ------------------------------------------------- C1 Eid 1446 profile in train
line("C1  market level around Idulfitri 1446 H, measured in train.csv")
lvl_tr["eid_k"] = (lvl_tr.index - EID_1446).days
lvl_tr["f_tps"] = dow_factor(lvl_tr, REF_2025, "tps")
lvl_tr["f_occ"] = dow_factor(lvl_tr, REF_2025, "occ")
_downame = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
_e = lvl_tr[lvl_tr.eid_k.between(1, 16)].copy()
_e["dow"] = _downame[_e.index.dayofweek]
print(f"reference window for 'normal' = {REF_2025[0].date()} .. {REF_2025[1].date()}")
print("\nf_tps / f_occ are multiples of a normal same-weekday level:")
print(_e[["eid_k", "dow", "tickets", "tps", "f_tps", "occ", "f_occ", "films", "clusters"]]
      .round(3).to_string())
print("\nmean factor over Eid+1..Eid+6 (the window the 2026 cohort needs):")
_w = lvl_tr[lvl_tr.eid_k.between(1, 6)]
print(f"  f_tps {_w.f_tps.mean():.3f} | f_occ {_w.f_occ.mean():.3f}")
print("CAVEAT: train.csv starts at Eid+1, so there is no pre-Eid baseline inside the file and")
print("the reference window is 5 weeks later. Treat these as levels, not as a straddle ratio.")

# --------------------------------- C2 Ramadan 1447 depression from test_history
line("C2  market level through Ramadan 1447 H, measured in test_history.csv")
lvl_th["f_tps"] = dow_factor(lvl_th, REF_2026, "tps")
lvl_th["f_occ"] = dow_factor(lvl_th, REF_2026, "occ")
lvl_th["ram_day"] = (lvl_th.index - RAMADAN_1447[0]).days + 1
_mo = lvl_th.assign(month=lvl_th.index.to_period("M")).groupby("month").agg(
    days=("tps", "size"), tps=("tps", "mean"), f_tps=("f_tps", "mean"),
    occ=("occ", "mean"), f_occ=("f_occ", "mean"), films=("films", "sum"))
print(f"reference window for 'normal' = {REF_2026[0].date()} .. {REF_2026[1].date()}")
print("\nby month (test_history holds only D1-D3 rows, so composition is at least consistent):")
print(_mo.round(3).to_string())
_r = lvl_th[lvl_th.ram_day.between(1, 31)]
print("\ninside Ramadan 1447 H, by week:")
print(_r.assign(wk=((_r.ram_day - 1) // 7 + 1)).groupby("wk").agg(
    days=("tps", "size"), f_tps=("f_tps", "mean"), f_occ=("f_occ", "mean"),
    films=("films", "sum")).round(3).to_string())
print("\nthe last three days before Eid (= the cohort's own D1-D3):")
print(lvl_th[(lvl_th.index >= pd.Timestamp("2026-03-18")) &
             (lvl_th.index <= pd.Timestamp("2026-03-20"))]
      [["tickets", "tps", "f_tps", "occ", "f_occ", "films", "clusters"]].round(3).to_string())

# ------------------------------------------- C3 the implied cohort multiplier
line("C3  implied market multiplier for the Eid cohort")
_num = lvl_tr[lvl_tr.eid_k.between(0, 6)].groupby("eid_k")[["f_tps", "f_occ"]].mean()
_den = lvl_th[(lvl_th.index >= pd.Timestamp("2026-03-18")) &
              (lvl_th.index <= pd.Timestamp("2026-03-20"))][["f_tps", "f_occ"]].mean()
print("numerator   = normal-adjusted level on Eid+k, from train.csv (Apr 2025)")
print("denominator = normal-adjusted level on 2026-03-18..20, from test_history.csv")
print(f"denominator: f_tps {_den.f_tps:.3f} | f_occ {_den.f_occ:.3f}\n")
imp = (_num.reindex(range(0, 7)) / _den).rename(columns={"f_tps": "mult_tps", "f_occ": "mult_occ"})
# the cohort's D1 is 2026-03-18, so its D4 lands on Eid+0 and its D10 on Eid+6
imp["horizon"] = ["D" + str(int(k) + 4) for k in imp.index]
imp["in_train"] = _num.reindex(range(0, 7)).f_tps.notna().to_numpy()
print(imp.round(3).to_string())
if not bool(imp.in_train.iloc[0]):
    print("\nNOTE: Eid+0 (= the cohort's D4) is absent from train.csv, which starts at Eid+1.")
    print("That single horizon has no training analogue and must be extrapolated or probed.")
print("\nHOW TO READ THIS: the multiplier is the MARKET term only. The ratio we must predict is")
print("roughly normal_decay(h, D1-dow) x multiplier. With a normal D4-D10 decay around 0.4-0.7,")
print("a multiplier of ~3 implies a true y/scale of roughly 1.2-2.0 for this cohort, which is")
print("what FINDINGS.md estimated independently. v4 ships x1.6 on top of a profile that decays")
print("to ~0.13 by D10, so it is very likely still far too low at the long horizons.")

# ---------------------------------------- C4 cross-film coverage of target days
line("C4  do other films observe our target (cluster, date)?")
_cd = set(map(tuple, th[["cinema_ids", "date_show"]].drop_duplicates().to_numpy()))
_dates = set(th.date_show.unique())
_tc = [tuple(x) for x in test[["cinema_ids", "date_show"]].to_numpy()]
cov_cl = np.fromiter((x in _cd for x in _tc), bool, len(test))
cov_dt = test.date_show.isin(_dates).to_numpy()
print(f"target date observed by >=1 other film : {cov_dt.sum():,} ({cov_dt.mean():.2%})")
print(f"exact (cluster, date) observed         : {cov_cl.sum():,} ({cov_cl.mean():.2%})")
_cv = pd.DataFrame({"h": test.off.to_numpy(), "date": cov_dt, "cluster": cov_cl,
                    "label": regime(test.date_show).label.to_numpy()})
print("\nby horizon:")
print(_cv[_cv.h.between(4, 10)].groupby("h")[["date", "cluster"]].mean().round(3).to_string())
print("\nby target regime (this is where the feature would have to earn its keep):")
print(_cv.groupby("label")[["date", "cluster"]].agg(["mean", "size"]).round(3).to_string())
lvl_tr.to_pickle(CACHE_DIR / "level_train.pkl")
lvl_th.to_pickle(CACHE_DIR / "level_test_history.pkl")
print(f"\n[CELL 6] {time.time() - t0:.1f}s  (level tables cached)")
""")

# ====================================================================== CELL 7
md(r"""
## CELL 7 — diagnose an existing submission (needs `submission_v4.csv` next to the notebook)

No model. This reads a finished submission back, converts it to the ratio space the metric
actually scores, and shows exactly how wrong the Eid cohort's profile is.

The thing I most want to see: **how many cohort rows are predicted exactly 0.** A multiplier
(v4's `x1.6`) leaves a zero at zero, so if most of the cohort's late horizons are zeros then the
Lebaran probe was structurally incapable of fixing them and we need a *floor*, not a scale.
""")
code(r"""
# CELL 7 — DIAGNOSE AN EXISTING SUBMISSION
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

SUB_PATH = None          # set to "path/to/file.csv" to diagnose a specific submission

_hol7 = (holidays.drop_duplicates("date")
         .assign(holiday_tipe=lambda d: d.holiday_tipe.astype(str).str.strip().str.lower())
         .set_index("date"))


def regime(dates):
    # Calendar facts per date. Reused by CELL 8.
    d = pd.DatetimeIndex(pd.Series(np.asarray(dates)).to_numpy())
    f = pd.DataFrame(index=np.arange(len(d)))
    f["dow"] = d.dayofweek
    f["holiday"] = (pd.Series(_hol7.holiday_tipe.reindex(d).to_numpy())
                    .fillna("normal").eq("holiday").astype(int).to_numpy())
    k46 = (d - EID_1446).days.to_numpy()
    k47 = (d - EID_1447).days.to_numpy()
    f["eid_k"] = np.where(np.abs(k46) <= 45, k46, np.where(np.abs(k47) <= 45, k47, np.nan))
    f["ramadan"] = (((d >= RAMADAN_1447[0]) & (d <= RAMADAN_1447[1])) |
                    ((d >= RAMADAN_1446[0]) & (d <= RAMADAN_1446[1]))).astype(int)
    f["eid_week"] = ((f.eid_k >= 0) & (f.eid_k <= 6)).astype(int)
    _s = np.zeros(len(d), dtype=int)
    for a, b in SCHOOL_BREAKS:
        _s = _s | ((d >= a) & (d <= b)).astype(int)
    f["school"] = _s
    f["label"] = np.where(f.eid_week == 1, "eid_week",
                 np.where(f.ramadan == 1, "ramadan",
                 np.where(f.school == 1, "school_break",
                 np.where(f.holiday == 1, "holiday", "ordinary"))))
    return f


def hitung_skala(history):
    total = history.groupby(["movie_title", "cinema_ids"]).total_ticket.sum()
    return (total / 3).clip(lower=1).rename("scale")


def find_sub(path=None):
    cands = ([Path(path)] if path else
             [SUB_DIR / "submission_v4.csv", Path("submission_v4.csv"),
              DATA_DIR / "submission_v4.csv"])
    for c in cands:
        if c.exists():
            return c
    hits = sorted(Path(".").glob("submission*.csv")) + sorted(SUB_DIR.glob("submission*.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError("put submission_v4.csv beside the notebook, or set SUB_PATH")


_p = find_sub(SUB_PATH)
sub = pd.read_csv(_p)
print("diagnosing:", _p.resolve(), "|", sub.shape)
assert sub.id.equals(test.id), "submission id order does not match test.csv"

scale_off = hitung_skala(th)
D = pd.DataFrame({"h": test.off.to_numpy(), "d1": test.d1.to_numpy(),
                  "date": test.date_show.to_numpy(),
                  "scale": test.join(scale_off, on=["movie_title", "cinema_ids"]).scale.to_numpy(),
                  "pred": sub.total_ticket.to_numpy().astype(float)})
D["ratio"] = D.pred / D.scale
_rg = regime(D.date)
D["label"], D["eid_k"] = _rg.label.to_numpy(), _rg.eid_k.to_numpy()
_ap = CACHE_DIR / "test_pair_patterns.pkl"
if _ap.exists():
    D["pattern"] = test.join(pd.read_pickle(_ap).pattern,
                             on=["movie_title", "cinema_ids"]).pattern.to_numpy()
else:
    D["pattern"] = "?"
_zs = lambda v: float((v == 0).mean())

# ------------------------------------------------------- S1 profile by horizon
line("S1  predicted profile by horizon")
print(D.groupby("h").agg(rows=("ratio", "size"), mean_ratio=("ratio", "mean"),
                         median_ratio=("ratio", "median"), zero_share=("pred", _zs),
                         mean_tickets=("pred", "mean")).round(4).to_string())
print(f"\noverall mean ratio {D.ratio.mean():.4f} | zero share {(D.pred == 0).mean():.2%}"
      f" | total tickets {int(D.pred.sum()):,}")

# ------------------------------------------------------- S2 profile by regime
line("S2  predicted profile by target regime")
print(D.groupby("label").agg(rows=("ratio", "size"), mean_ratio=("ratio", "mean"),
                             median_ratio=("ratio", "median"),
                             zero_share=("pred", _zs)).round(4).to_string())

# ----------------------------------------------------------- S3 the Eid cohort
line("S3  THE EID COHORT (D1 = 2026-03-18)")
coh = D[D.label == "eid_week"]
print("cohort D1 dates:", sorted({str(x.date()) for x in coh.d1}))
print(f"cohort rows {len(coh):,} ({len(coh) / len(D):.2%} of test) | pairs {len(coh) // 7}")
print(coh.groupby("h").agg(eid_k=("eid_k", "first"), rows=("ratio", "size"),
                           mean_ratio=("ratio", "mean"), median_ratio=("ratio", "median"),
                           zero_share=("pred", _zs), mean_tickets=("pred", "mean"))
      .round(4).to_string())
print(f"\ncohort mean predicted ratio over D4-D10 : {coh.ratio.mean():.4f}")
print(f"cohort rows predicted exactly 0         : {int((coh.pred == 0).sum()):,}"
      f" ({(coh.pred == 0).mean():.2%})")
print("\nMASE recoverable if the true cohort ratio is a flat r  (= sum|r - pred| / 72611):")
for _r in (1.0, 1.25, 1.5, 2.0, 2.5):
    print(f"   r = {_r:4.2f}  ->  {np.abs(_r - coh.ratio).sum() / len(D):.4f}")
_z = coh[coh.pred == 0]
print(f"\n...of which the {len(_z):,} rows currently predicted 0 alone account for:")
for _r in (1.0, 1.5, 2.0):
    print(f"   r = {_r:4.2f}  ->  {_r * len(_z) / len(D):.4f}")

# -------------------------------------------- S4 why a multiplier cannot work
line("S4  what a pure multiplier does to the cohort (v4 shipped x1.6)")
for _m in (1.0, 1.6, 2.0, 3.0):
    _p2 = np.floor(coh.ratio.to_numpy() * _m * coh.scale.to_numpy() + 0.5)
    print(f"  x{_m:<4} -> cohort mean ratio {(_p2 / coh.scale.to_numpy()).mean():.4f}"
          f" | zeros {(_p2 == 0).mean():.2%}")
print("\nreading: zeros are fixed points of multiplication. Whatever share of the cohort is")
print("already 0 cannot be moved by scaling, so the x1.6 probe could never reach a ratio > 1.")

# ------------------------------------------------ S5/S6 leverage concentration
line("S5  by D1-D3 activity pattern")
print(D.groupby("pattern").agg(rows=("ratio", "size"), med_scale=("scale", "median"),
                               mean_ratio=("ratio", "mean"),
                               zero_share=("pred", _zs)).round(4).to_string())
line("S6  by scale bucket, with the share of the metric each bucket controls")
D["sb"] = pd.cut(D.scale, [0, 2, 5, 10, 25, 50, 100, 1e9],
                 labels=["<=2", "<=5", "<=10", "<=25", "<=50", "<=100", ">100"])
_g6 = D.groupby("sb", observed=False).agg(rows=("ratio", "size"), mean_ratio=("ratio", "mean"),
                                          zero_share=("pred", _zs),
                                          lev=("scale", lambda v: float((1.0 / v).sum())))
_g6["leverage_share"] = _g6.lev / float((1.0 / D.scale).sum())
print(_g6.drop(columns="lev").round(4).to_string())
D.to_pickle(CACHE_DIR / "sub_diag.pkl")
print(f"\n[CELL 7] {time.time() - t0:.1f}s")
""")

# ====================================================================== CELL 8
md(r"""
## CELL 8 — the market-level index and the per-row market ratio (needs CELL 7)

Builds one number per calendar date: a **day-of-week-adjusted market level** `L(date)`, measured
from `test_history.csv` (every row there is a D1-D3 row, so film age is already controlled) and
normalised against a normal reference window. Because the day-of-week effect is divided out, the
model's existing weekday features are not double-counted.

Then, per test row:

```
market_ratio = L(target day) / mean( L(D1), L(D2), L(D3) )
```

For ordinary-to-ordinary rows this should land near 1.0 — that is the sanity check. It departs
from 1 exactly where the window straddles a regime boundary, which is the 13.34% of rows that
CELL 5 flagged. Idulfitri week 2026 is unobserved, so those seven days are filled from the
Idulfitri 1446 H profile in `train.csv`.
""")
code(r"""
# CELL 8 — MARKET-LEVEL INDEX + PER-ROW MARKET RATIO
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

# ---------------------------------- daily dow-adjusted level from test_history
_g = th.groupby("date_show").agg(tickets=("total_ticket", "sum"), shows=("total_show", "sum"),
                                 occ=("occupation_rate", "mean"), films=("movie_title", "nunique"),
                                 rows=("total_ticket", "size"))
_g["tps"] = _g.tickets / _g.shows.replace(0, np.nan)
_ref = _g[(_g.index >= REF_2026[0]) & (_g.index <= REF_2026[1])]
_base = _ref.groupby(_ref.index.dayofweek).tps.median()
_g["f"] = _g.tps / _g.index.dayofweek.map(_base).to_numpy()
print(f"reference window {REF_2026[0].date()} .. {REF_2026[1].date()}"
      f" | observed dates {len(_g)} | median f {_g.f.median():.3f}")

# weekly smoothing: full coverage and far less noise (dow is already divided out)
_wk = (_g.assign(w=_g.index.to_period("W")).groupby("w")
       .agg(days=("f", "size"), f=("f", "mean"), films=("films", "sum")))
print("\nweekly market level (f = multiple of a normal same-weekday level):")
print(_wk.round(3).to_string())

# ------------------------------------------- assemble L(date) over every date
_all = pd.date_range(min(test.d1.min(), th.date_show.min()), test.date_show.max())
_wmap = {p: v for p, v in _wk.f.items()}
L = pd.Series([_wmap.get(p, np.nan) for p in pd.PeriodIndex(_all, freq="W")],
              index=_all, dtype=float, name="L")

# Idulfitri 1447 H week is unobserved -> borrow the 1446 H profile from train.csv
_ltr = pd.read_pickle(CACHE_DIR / "level_train.pkl")
_prof = _ltr[_ltr.eid_k.between(1, 6)].set_index("eid_k").f_tps
_prof = _prof[~_prof.index.duplicated()].reindex(range(1, 7)).interpolate().bfill().ffill()
if _prof.isna().all():
    print("*** WARNING: no Idulfitri 1446 H profile in train.csv - falling back to 1.0 ***")
    _prof = pd.Series(1.0, index=range(1, 7))
_vals = [float(_prof.loc[1])] + [float(_prof.loc[k]) for k in range(1, 7)]   # k=0 copies k=1
for _k, _v in enumerate(_vals):
    L.loc[EID_1447 + pd.Timedelta(days=_k)] = _v
print(f"\nEid week fill (from Apr 2025): " +
      " ".join(f"k{_k}={_v:.2f}" for _k, _v in enumerate(_vals)))
print("CAVEAT: k0 (2026-03-21, the cohort's D4) is a copy of k1 - train.csv starts at Eid+1.")
_miss = int(L.isna().sum())
L = L.interpolate().bfill().ffill()
print(f"dates with no observed week, filled by interpolation: {_miss}")
print("L coverage over all needed dates:", f"{L.notna().mean():.1%}")


def Lof(dates):
    return L.reindex(pd.DatetimeIndex(pd.Series(np.asarray(dates)).to_numpy())).to_numpy()


_num = Lof(test.date_show)
_den = sum(Lof(test.d1 + pd.Timedelta(days=k)) for k in range(3)) / 3.0
M = pd.DataFrame({"h": test.off.to_numpy(), "d1": test.d1.to_numpy(),
                  "label": regime(test.date_show).label.to_numpy(),
                  "hist_label": regime(test.d1).label.to_numpy(),
                  "L_t": _num, "L_h": _den})
M["mr"] = M.L_t / M.L_h
# NOTE: the column is `hist_label`, never `hist` -- `df.hist` is the pandas histogram
# METHOD, so `M.hist == "ordinary"` silently compares a bound method and returns all-False.
# That bug made M1 print "rows 0 | mean mr nan" in the previous run.

# ------------------------------------------------------- sanity + distribution
line("M1  sanity: ordinary -> ordinary rows must sit near mr = 1.0")
_ord = M[(M["label"] == "ordinary") & (M["hist_label"] == "ordinary")]
assert len(_ord) > 0, "M1 cohort is empty -- selection bug, do not trust the index"
print(f"rows {len(_ord):,} | mean mr {_ord.mr.mean():.3f} | median {_ord.mr.median():.3f}"
      f" | 10-90% {_ord.mr.quantile(.1):.3f}-{_ord.mr.quantile(.9):.3f}")
print("if this is far from 1.0 the index is biased and the correction must not be trusted.")

line("M2  market ratio by target regime")
print(M.groupby("label").mr.describe(percentiles=[.1, .5, .9]).round(3).to_string())
line("M3  market ratio by (D1-D3 regime -> target regime), the straddle cells")
_pv = M.pivot_table(index="hist_label", columns="label", values="mr", aggfunc="median")
_cn = M.pivot_table(index="hist_label", columns="label", values="mr", aggfunc="size")
print("median mr:");  print(_pv.round(3).to_string())
print("\nrows:");     print(_cn.fillna(0).astype(int).to_string())

line("M4  the Eid cohort's implied market ratio by horizon")
_c = M[M.label == "eid_week"]
print(_c.groupby("h").agg(rows=("mr", "size"), L_target=("L_t", "first"),
                          L_hist=("L_h", "first"), mr=("mr", "first")).round(3).to_string())
print("\nthis is the MARKET term only. The ratio to predict is roughly")
print("   normal_decay(h, D1-dow) x mr,  and separately the drop-out hazard collapses during")
print("   Eid week, so the zero share should be near zero rather than rising with h.")

line("M5  how many rows a correction would touch")
for _grp in (["ramadan", "eid_week"], ["ramadan", "eid_week", "school_break"]):
    _m = M.label.isin(_grp).to_numpy()
    print(f"  target in {str(_grp):46s} rows {int(_m.sum()):6,} ({_m.mean():6.2%})"
          f" | median mr {np.nanmedian(M.mr.to_numpy()[_m]):.3f}")
print(f"  rows with a missing market ratio                       {int(M.mr.isna().sum()):6,}")
_far = (M.mr > 1.5) | (M.mr < 0.67)
print(f"  rows whose market ratio is beyond x1.5 / x0.67        {int(_far.sum()):6,}"
      f" ({_far.mean():6.2%})")
M.to_pickle(CACHE_DIR / "market_ratio.pkl")
L.to_pickle(CACHE_DIR / "market_level.pkl")
print(f"\n[CELL 8] {time.time() - t0:.1f}s  (market ratio cached)")
""")

md(r"""
## CELL 9 — the honest target shape: release-anchored ratio profile from `train.csv`

Mined from `0.46641.ipynb`'s stored outputs, v1's own error anatomy says the error is concentrated
in **opening-phase windows (MASE 0.9000, 25.9% of estimated error)**, while scale is a weak and
non-monotone driver. And R2 already proved the test is ~100% opening windows: D1 *is* the
wide-release date.

So the population that matters is "window anchored at the wide release". v1 trained mostly on
mid-run windows, and its `buka (0-2)` bucket is polluted by sneak previews (D1-D3 = a 1-4 cinema
preview, D4-D10 = the wide release exploding) — the opposite shape to the test.

This cell measures the **true** `y / scale` profile on release-anchored train windows built to the
test's exact geometry, so it is directly comparable to CELL 7's predicted profile. No model
training. Guards applied:

- wide release detected by cluster count, so preview days are skipped;
- `d1 + 9 <= 2025-09-30` so no horizon is right-censored into a false zero;
- the film's first appearance must be after the start of `train.csv` (no left-censoring);
- windows overlapping the corrupted 2025-06-01..06-16 span are dropped;
- only pairs **active on D3** are kept (R1: that is exactly the test's inclusion rule);
- absent future dates are filled with 0, which is what "zero" means in this data;
- `ordinary`-only variant isolates normal decay from the calendar effects of CELL 8;
- the profile is reweighted to the test's D1-weekday mix, because R2 showed D1-dow dominates.
""")
code(r"""
# CELL 9 — RELEASE-ANCHORED TRUE RATIO PROFILE (no model training)
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)
MIN_CLUSTERS, WIDE_FRAC = 5, 0.50
TR_MAX = train.date_show.max()


def regime(dates):
    d = pd.DatetimeIndex(dates)
    lab = np.full(len(d), "ordinary", dtype=object)
    for a, b in SCHOOL_BREAKS:
        lab[(d >= a) & (d <= b)] = "school_break"
    for a, b in (RAMADAN_1446, RAMADAN_1447):
        lab[(d >= a) & (d <= b)] = "ramadan"
    for e in (EID_1446, EID_1447):
        lab[(d >= e) & (d <= e + pd.Timedelta(days=6))] = "eid_week"
    return pd.Series(lab, index=range(len(d)))


# ------------------------------------------------- 1. preview-corrected release date
_dc = train.groupby(["movie_title", "date_show"], as_index=False).agg(
    clusters=("cinema_ids", "nunique"), tickets=("total_ticket", "sum"))
_mx = _dc.groupby("movie_title").clusters.max().rename("mx")
_dc = _dc.merge(_mx, on="movie_title")
_wide = _dc[(_dc.clusters >= np.maximum(MIN_CLUSTERS, WIDE_FRAC * _dc.mx))]
rel = _wide.groupby("movie_title").date_show.min().rename("d1").reset_index()
_first = train.groupby("movie_title").date_show.min().rename("first")
rel = rel.merge(_first, on="movie_title")
rel["preview_days"] = (rel.d1 - rel["first"]).dt.days
print(f"films in train {train.movie_title.nunique()} | wide release detected {len(rel)}")
print(f"films whose wide release is LATER than first appearance (sneak preview): "
      f"{int((rel.preview_days > 0).sum())} ({(rel.preview_days > 0).mean():.1%})")
print("preview-day distribution:",
      rel.preview_days.value_counts().sort_index().head(12).to_dict())

# ------------------------------------------------------------- 2. eligibility guards
_ok = rel[(rel.d1 + pd.Timedelta(days=9) <= TR_MAX) & (rel["first"] > train.date_show.min())].copy()
_bad = (_ok.d1 + pd.Timedelta(days=9) >= BAD_START) & (_ok.d1 <= BAD_END)
_ok = _ok[~_bad]
print(f"\neligible films: {len(_ok)}  (fully observed D1-D10, release inside train, "
      f"not touching {BAD_START.date()}..{BAD_END.date()})")
_need = 20 if len(train) > 100_000 else 3      # real train.csv vs a synthetic smoke fixture
assert len(_ok) >= _need, (f"only {len(_ok)} eligible films - loosen the guards or check the "
                           f"release detector before trusting this profile")

# ------------------------------------------------- 3. build windows to test geometry
w = train.merge(_ok[["movie_title", "d1"]], on="movie_title", how="inner")
w["off"] = (w.date_show - w.d1).dt.days + 1
w = w[(w.off >= 1) & (w.off <= 10)]
h3 = w[w.off <= 3]
piv = (h3.pivot_table(index=["movie_title", "cinema_ids"], columns="off",
                      values="total_ticket", aggfunc="sum")
       .reindex(columns=[1, 2, 3]))
piv.columns = ["s1", "s2", "s3"]
piv = piv.fillna(0.0).reset_index()
piv = piv[piv.s3 > 0].copy()                      # R1: test contains only pairs active on D3
piv["scale"] = ((piv.s1 + piv.s2 + piv.s3) / 3.0).clip(lower=1.0)
print(f"\nrelease-anchored pairs active on D3: {len(piv):,}")

piv = piv.merge(_ok[["movie_title", "d1"]], on="movie_title", how="left")
assert piv.d1.notna().all(), "a release-anchored pair lost its d1 in the merge"
grid = piv[["movie_title", "cinema_ids", "d1", "scale"]].merge(
    pd.DataFrame({"off": np.arange(4, 11)}), how="cross")
grid["date_show"] = grid.d1 + pd.to_timedelta(grid.off - 1, unit="D")
_fut = (w[w.off >= 4].groupby(["movie_title", "cinema_ids", "off"], as_index=False)
        .total_ticket.sum())
grid = grid.merge(_fut, on=["movie_title", "cinema_ids", "off"], how="left")
grid["total_ticket"] = grid.total_ticket.fillna(0.0)
grid["ratio"] = grid.total_ticket / grid.scale
grid["d1_dow"] = grid.d1.dt.dayofweek
grid["lab"] = regime(grid.date_show).to_numpy()
grid["lab_d1"] = regime(grid.d1).to_numpy()
print(f"scored rows built: {len(grid):,}  (pairs x 7 horizons)")

# --------------------------------------------------------------- 4. the true profile
line("T1  TRUE ratio profile by horizon (release-anchored, all calendar days)")
_t1 = grid.groupby("off").agg(rows=("ratio", "size"), mean_ratio=("ratio", "mean"),
                              median_ratio=("ratio", "median"),
                              zero_share=("total_ticket", lambda s: float((s == 0).mean())))
print(_t1.round(4).to_string())

line("T2  TRUE profile on ORDINARY days only (D1 and target both ordinary)")
_pure = grid[(grid.lab == "ordinary") & (grid.lab_d1 == "ordinary")]
_t2 = _pure.groupby("off").agg(rows=("ratio", "size"), mean_ratio=("ratio", "mean"),
                               median_ratio=("ratio", "median"),
                               zero_share=("total_ticket", lambda s: float((s == 0).mean())))
print(_t2.round(4).to_string())
print(f"ordinary-only rows: {len(_pure):,} of {len(grid):,} ({len(_pure) / len(grid):.1%})")

line("T3  TRUE profile by D1 weekday (ordinary days) - R2 said this dominates")
_t3 = _pure.pivot_table(index="d1_dow", columns="off", values="ratio", aggfunc="mean")
_n3 = _pure.pivot_table(index="d1_dow", columns="off", values="ratio", aggfunc="size")
_DOW = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
print("mean ratio:");  print(_t3.rename(index=_DOW).round(3).to_string())
print("\nrows:");      print(_n3.rename(index=_DOW).fillna(0).astype(int).to_string())

# ------------------------- 5. reweight to the test's D1-dow mix, then face the model
line("T4  TRUE profile reweighted to the test's D1-weekday mix  vs  v4's PREDICTED profile")
_td = (test.drop_duplicates("movie_title").d1.dt.dayofweek.value_counts(normalize=True)
       .rename("w_test"))
print("test D1-dow mix:", {_DOW[k]: round(v, 3) for k, v in _td.sort_index().items()})
_pz = _pure.pivot_table(index="d1_dow", columns="off", values="total_ticket",
                        aggfunc=lambda s: float((s == 0).mean()))
_w = _td.reindex(_t3.index).fillna(0.0)
_cover = float(_w.sum())
_w = _w / _cover if _cover > 0 else _w


def _wmean(tab, wts):
    # weighted mean down the dow axis, renormalised per column over non-missing cells
    num = (tab.mul(wts, axis=0)).sum(skipna=True)
    den = (tab.notna().mul(wts, axis=0)).sum()
    return num / den.replace(0, np.nan)


true_mean = _wmean(_t3, _w)
true_zero = _wmean(_pz, _w)
print(f"\n(these weights cover {_cover:.1%} of test films; a D1-dow that never occurs in the"
      f" eligible train films is dropped and the rest renormalised)")

_sp = CACHE_DIR / "sub_diag.pkl"
if _sp.exists():
    D = pd.read_pickle(_sp)
    _po = D[D.label == "ordinary"] if "label" in D.columns else D
    pred = _po.groupby("h").agg(pred_mean=("ratio", "mean"),
                                pred_zero=("pred", lambda s: float((s == 0).mean())))
    cmp = pd.DataFrame({"true_mean": true_mean, "pred_mean": pred.pred_mean,
                        "true_zero": true_zero, "pred_zero": pred.pred_zero})
    cmp["mean_mult"] = cmp.true_mean / cmp.pred_mean
    cmp["zero_gap_pp"] = (cmp.true_zero - cmp.pred_zero) * 100
    print(cmp.round(4).to_string())
    print("\nmean_mult > 1 => v4 UNDER-predicts that horizon; < 1 => it OVER-predicts.")
    print("zero_gap_pp > 0 => v4 emits too FEW zeros at that horizon.")
    print("CAVEAT: a mean-matching multiplier is not MAE-optimal (the median is). This table")
    print("        sizes the bias; it is not yet the correction.")
else:
    print("cache/sub_diag.pkl missing - run CELL 7 first to get the predicted profile.")
    print("true (reweighted) mean ratio by horizon:", true_mean.round(4).to_dict())
    print("true (reweighted) zero share by horizon:", true_zero.round(4).to_dict())

line("T5  how much of the test population this profile actually speaks for")
print(f"eligible train films {len(_ok)} vs test films {test.movie_title.nunique()}")
print("pure-ordinary share of the TRAIN profile rows:", f"{len(_pure) / len(grid):.1%}")
_rl = np.load(CACHE_DIR / "test_regime_label.npy", allow_pickle=True)
print("ordinary share of TEST rows:", f"{float((_rl == 'ordinary').mean()):.1%}")
print("=> the remaining test rows need the CELL 8 market ratio on top of this shape.")
grid.to_pickle(CACHE_DIR / "release_anchored_grid.pkl")
_t1.to_pickle(CACHE_DIR / "true_profile_all.pkl")
_t2.to_pickle(CACHE_DIR / "true_profile_ordinary.pkl")
print(f"\n[CELL 9] {time.time() - t0:.1f}s  (release-anchored grid cached)")
""")

# =================================================================== decision log
md(r"""
## DECISION LOG

| ver | what changed | grouped CV | hidden-date CV | LB | keep? |
|-----|--------------|-----------|----------------|----|-------|
| v1 | ratio target + L1 objective, multi-anchor windows | 0.4328 OOF | 0.3978 holdout | 0.46641 | superseded |
| v2 | - | 0.374 | - | ~0.47 | dropped |
| v3 | dropped corrupted Jun 1-16; removed month/day-of-month; mid-run windows (545k rows); extra calendar facts; hidden-date CV; Lebaran x1.6 probe; round half up | 0.359 | - | 0.43715 | kept |
| v4 | competition features (4); Lebaran competition imputation; 127 leaves / min 100; 5 seeds | 0.344 | 0.354 | *(need from you)* | pending |
| v5 | *(this notebook)* | | | | |

### ROUND 2 — Phase 1 error anatomy, recovered for free from `0.46641.ipynb`

The v1 notebook still carries its **stored outputs**, and it is not merely the 0.46641 baseline:
it is a complete pipeline that already ran the two-part model, per-horizon calibration,
regime-weighted MASE, adversarial validation and a blend. So Phase 1 needed no compute.

**v1's own OOF error anatomy** (`r` = film-age bucket from `anchor_lag`, not calendar):

| regime | MASE | n | est. test share | error contribution |
|--------|------|---|-----------------|--------------------|
| buka (0-2)    | **0.9000** | 96,649 | 13.7% | 25.9% |
| awal (3-5)    | 0.5861 | 115,913 | 14.1% | 17.3% |
| tengah (6-15) | 0.3755 | 496,671 | 43.7% | 34.4% |
| akhir (16+)   | 0.3742 | 255,360 | 28.6% | 22.4% |

- Reweighted 4-regime estimate **0.4766** (plain OOF 0.4530, time holdout 0.4133).
- After the two-part model + calibration: OOF 0.4328, **estimate 0.4566**, holdout 0.3978.
  Actual v1 LB was **0.46641** — so the reweighted estimator was accurate to 0.010 and the
  CV→LB "gap" is not a mystery, it is the known age-composition reweighting.
- **Adversarial AUC = 0.9997 (0.9999 without identity features).** Train and test are almost
  perfectly separable. This is the quantitative proof of the population mismatch.
- **True zero share = 43.5%** (confirms the ~40% figure). v1's zero classifier reached AUC
  0.9339 and its optimal decile multipliers were `{0:1.05, 1:1.0, 2:1.0, 3:0.9, 4:0.7, 5:0.4,
  6-9: 0.0}` — i.e. zero out the top 40% by `p0`. That emitted 38.7% zeros (41.53% after
  blending) and moved plain OOF 0.4530 → 0.4359.
- **Per-horizon calibration is already spent**: the fitted multipliers were `{4:1.01 … 10:1.00}`,
  and blend+calibration+two-part together bought only −0.0074 on the weighted objective. The
  friend's quoted +0.012 is therefore *not* additional headroom for us.
- **MASE by (age × scale) is non-monotone**, so the Σ(1/scale) "leverage" framing from round 1
  overstated small-scale rows. Mid-scale *opening* rows are the worst cells (10-25 → 1.304,
  25-50 → 1.555), while `>50` is the best (0.650) within the same bucket:

| regime | ≤2 | 2-5 | 5-10 | 10-25 | 25-50 | >50 |
|--------|----|-----|------|-------|-------|-----|
| buka (0-2)    | 0.533 | 0.672 | 0.995 | 1.304 | 1.555 | 0.650 |
| awal (3-5)    | 0.587 | 0.335 | 0.640 | 0.740 | 0.962 | 0.492 |
| tengah (6-15) | 0.544 | 0.401 | 0.586 | 0.401 | 0.540 | 0.339 |
| akhir (16+)   | 1.572 | 0.740 | 0.457 | 0.456 | 0.413 | 0.319 |

**Consequence — the round-1 Eid thesis is dead and the ranking changes.** CELL 7 showed v4
already predicts the Eid cohort at mean ratio 1.0943 with **0.00% zeros**, so the ×1.6 hypothesis
was already applied; the whole cohort is worth at most 0.0255 (at true r=1.5) and 0.0058 if v4 is
already right. The dominant, *measurable* error source is the **opening-window population**, and
R2 proved the test is ~100% opening windows while v1's `buka` bucket is polluted by sneak
previews (D1-D3 = tiny preview, D4-D10 = wide release exploding — the inverse of the test shape).
That also explains why "restrict training to lag-0 windows" was recorded as a negative result:
with a corrupted `anchor_lag`, lag-0 selected *previews*, not releases. CELL 9 measures the true
release-anchored shape with preview-corrected detection.

**v1's pipeline is extractable** — `build_train_samples(hist_all, anchors, ...)` takes an explicit
anchor set, so the missing `cinema_forecasting_v3.py` is no longer a hard blocker for retraining
or for building an honest, release-anchored validation split.

### Bug fixed this round

`M.hist` resolved to the pandas `DataFrame.hist` *method*, so `M.hist == "ordinary"` compared a
bound method and matched nothing — CELL 8's M1 sanity check printed `rows 0 | mean mr nan`. The
column is now `hist_label`, access is `M["hist_label"]`, M1 asserts non-empty, and
`build_notebook.py` fails the build if any generated cell accesses a column attribute-style when
that name collides with the pandas API.

### The headline problem: CV gains are not transferring

| step | CV moved | LB moved | transfer |
|------|---------|---------|----------|
| v1 -> v3 | 0.4328 -> 0.359 (-0.074) | 0.46641 -> 0.43715 (-0.029) | ~39% |
| v3 -> v4 | 0.359 -> 0.344 (-0.015) | ? | ? |

The CV-to-LB gap **grew** from 0.033 (v1) to 0.078 (v3) as CV improved. That is the signature of
optimising a population that is not the scored population, not of a weak model. The leader at
0.34456 is roughly where our *CV* already sits.

### Round 1 measurements

Environment: 16 cores, **11.7 GiB RAM but only 0.1 GiB free**, Python 3.12.5, pandas 3.0.6,
lightgbm 4.7.0. catboost / pyarrow / tqdm absent. `cinema_forecasting_v3.py` absent, so **v4
still cannot be re-run**. v4 == the `0.43715.py` script, LB **0.43715**.

Both structural rules re-verified in pandas. `R1` exact: active-on-D3 == in test.csv, and 0 test
pairs lack D1-D3 history. `R2`: 160 films, 67 distinct D1 dates, Wed 40% / Thu 41% / Fri 15%.

**The metric is controlled by a handful of small pairs.** Share of rows vs share of total
`1/scale` leverage: scale<=2 is 0.24% of rows but **6.6%** of leverage; <=5 is 1.54% / **20.1%**;
<=10 is 4.97% / **37.2%**; <=25 is 17.04% / **64.6%**; <=50 is 32.78% / **81.2%**. Official scale
median 91.67, 1st percentile 4.0, max 24,117. By D1-D3 pattern, `001`+`011`+`101` are 8.9% of
rows but **24.6% of leverage** (median scale 15 / 53 / 25, vs 99 for `111`).

**Calendar composition of the 72,611 scored rows:** ordinary 68.39%, Ramadan 17.09%, school break
7.21%, Eid week 6.08%, other holiday 1.22%. **Straddle rows (target regime absent from the
window's own D1-D3): 9,683 = 13.34%** — Eid 4,417 + ordinary 1,627 + Ramadan 1,460 + school 1,290
+ holiday 889. Every Eid-week row comes from a Ramadan D1-D3.

**The two market factors, measured** (day-of-week-adjusted, so a weekday effect is already
divided out):

| period | source | level vs normal |
|---|---|---|
| Idulfitri 1446 H, Eid+1..+6 (Apr 2025) | `train.csv` | **2.41x** tps / 2.52x occ |
| Ramadan 1447 H weeks 1 / 2 / 3 / 4 / 5 | `test_history.csv` | 0.44 / 0.41 / **0.29** / 0.64 / 0.92 |
| 2026-03-18..20 (the cohort's own D1-D3) | `test_history.csv` | 1.17x |

Implied **market multiplier for the Eid cohort: 1.52x (D10) to 2.41x (D5)**, mean ~2.05 on
tickets-per-show, ~2.3 on occupancy. Eid+0 (the cohort's D4) has no analogue at all, because
`train.csv` begins at Eid+1.

**Cross-film coverage of our target days** (another film's D1-D3 observed that date): 66.19% by
date, 58.38% at exact cluster+date — matching PR #1 exactly. By regime this splits the calendar
problem in two: Ramadan **72.8%** covered, holiday 100%, school break 53.8%, but Eid week
**0.0%**, because `test_history.csv` stops on 2026-03-20. Ramadan is measurable from 2026 data;
Eid week is pure extrapolation from April 2025.

### Already proven, do not re-measure (PR #1 `FINDINGS.md`)

- **R1** a pair is forecast iff it sold >=1 ticket on D3; zero test pairs lack D1-D3 history;
  no cluster ever appears after D3, so expansions are never forecast.
- **R2** D1 is the film's wide-release date (67 distinct D1 dates for 163 films, Wed 41% / Thu 40%),
  so the test is ~100% opening windows. v4 validating on `window == 0` is therefore *right*.
- **Calendar, not film age, explains the gap**: anomaly days are 7.1% of train rows vs 31.8% of
  test rows, at 2.37x the MASE (0.8485 vs 0.3587). Reweighting to the test mix predicted 0.5144.
- **Train has no Ramadan at all** (Ramadan 1446 H ended 2025-03-30; train starts 2025-04-01) and
  `holidays.csv` never labels Ramadan, so 18.2% of test rows have no representable feature.
- **The 2026-03-18 Eid cohort** is 7 films / 631 pairs / 4,417 rows = 6.08% of test, with all
  seven target days inside Idulfitri 1447 H week. v1 predicted a mean ratio of 0.411 decaying to
  0.132 by D10, when the truth should exceed 1. Worth 0.05-0.10 MASE on its own.
- **`anchor_lag` is corrupted by sneak previews**: 44% of train films first appear as a 1-4
  cluster preview 1-9 days before wide release, so "release window" is mislabelled in training.
- **D1-dow matters more than horizon**: median y/scale at D4 is 1.114 (Wed), 0.906 (Thu),
  0.454 (Fri). A pooled per-horizon curve sits below the 81% of test that is Wed/Thu.
- **8% of rows carry 25% of the error**: pattern `001` is 1.99% of rows at MASE 2.88.
- **66% of test rows have their target date observed by another film** in test_history.csv
  (58% at exact cluster+date, 88-93% at D8-D10).

### Negative results, do not repeat

| idea | result |
|------|--------|
| restrict training to lag-0 release windows only | worse (0.4024 vs 0.3932) |
| apply the `s3 > 0` filter to *training* data | neutral |
| predict on run-rate basis instead of `scale` | slightly worse |
| national date factor as a model key | worse (+0.0067) |
| national date factor as a multiplier | worse (+0.0166) |

Caveat on the last two: they were tested with a median-lookup evaluator on train only, where
there are no calendar anomalies to learn from. Structurally the idea is still sound; it has to be
judged on anomaly rows, not on pooled OOF.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

# --------------------------------------------------------------------- guard rail
# `M.hist` silently resolved to DataFrame.hist (the plotting method) and made a whole
# sanity check return 0 rows. Fail the build if any generated cell accesses a column
# attribute-style when that name collides with the pandas API.
def _audit(cells):
    import re
    try:
        import pandas as _pd
    except ImportError:
        print("audit skipped: pandas unavailable"); return
    api = set(dir(_pd.DataFrame)) | set(dir(_pd.Series))
    src = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assigned = set(re.findall(r'\[\s*"([a-zA-Z_]\w*)"\s*\]\s*=', src))
    assigned |= set(re.findall(r'^\s*"([a-zA-Z_]\w*)"\s*:', src, re.M))
    bad = []
    for col in sorted(assigned & api):
        for obj in set(re.findall(r'\b([A-Za-z_]\w*)\.' + col + r'\b(?!\s*\()', src)):
            bad.append(f"{obj}.{col}")
    if bad:
        raise SystemExit("AUDIT FAIL - attribute access collides with pandas API: "
                         + ", ".join(sorted(set(bad)))
                         + "\n  use df[\"col\"] instead, or rename the column.")
    print(f"audit ok: {len(assigned)} column names, none shadowed by attribute access")


_audit(cells)

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {NB}: {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code)")
