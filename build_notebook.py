"""Generates cinema_v5.ipynb. Run:  python build_notebook.py"""
import datetime
import json
import subprocess

NB = "cinema_v5.ipynb"

# Bump NB_VERSION on any change the user must re-run. The stamp goes into the notebook
# header AND is printed by CELL 1, so a stale notebook is obvious in two seconds.
NB_VERSION = 10
NEEDS_PIPELINE = 3
_d = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
try:
    _c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                        text=True, timeout=10).stdout.strip() or "unknown"
except Exception:
    _c = "unknown"
STAMP = f"v{NB_VERSION} | built {_d} | parent commit {_c}"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# ======================================================================= intro
md("# Cinema ticket forecasting — v5 working notebook\n\n"
   "> ## NOTEBOOK BUILD " + STAMP + "\n"
   "> Requires `v5_pipeline.py` VERSION >= " + str(NEEDS_PIPELINE) + ". CELL 1 reprints this\n"
   "> stamp and CELL 2 verifies the helper version. **If the stamp does not match what I told\n"
   "> you to run, pull again and reopen the notebook — VS Code caches the file and will not\n"
   "> reread it on its own.**\n>\n"
   "> `RUN_HEAVY` is read from `local_config.json` (gitignored), so this notebook is never\n"
   "> hand-edited and `git pull` stays clean. Enable heavy cells with `{\"run_heavy\": true}`.\n")

md(r"""
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
FOLD_KINDS     = ("group", "time")   # v3's two CV schemes; "time" is the gate
import json as _json                 # cells 11-20 write resumable state as json
SUBSAMPLE_ROWS = None      # e.g. 200_000 -> subsample training windows for a quick run
# ---- RUN_HEAVY lives in local_config.json, NOT in this notebook ----------------
# Editing the notebook to flip this makes `git pull` refuse to update it, which is how
# a stale cell survives a pull and then fails with a confusing AttributeError.
# To enable the heavy cells, create local_config.json beside the notebook:
#     {"run_heavy": true}
# That file is gitignored, so pulling is always clean and you never edit a cell again.
_LOCAL = Path("local_config.json")
_cfg = json.loads(_LOCAL.read_text()) if _LOCAL.exists() else {}
RUN_HEAVY = bool(_cfg.get("run_heavy", False))
if not _LOCAL.exists():
    print('NOTE: no local_config.json -> RUN_HEAVY = False. Create it with'
          ' {"run_heavy": true} to enable cells 11-20.')

# ---- v4's invocation: 0.43715.py was run as  final(featset, w_open, paramset) ----
# Reconstructed from its docstring ("competition features", "127 leaves / min 100", "extra
# weight on opening-week windows"); reproduces the reported CV to ~0.005. V5_FS is the
# featset that won under the test-mix-weighted metric (CELL 14), which is what to ship.
V4_FS, V4_WOPEN, V4_PS = "H", 2.0, "P2"
V5_FS = "F"

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
print("=" * 78)
print("NOTEBOOK BUILD __NB_STAMP__")
print("  requires v5_pipeline.VERSION >= __NB_NEEDS__   (CELL 2 verifies this)")
print("=" * 78)
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

# ---- staleness guard --------------------------------------------------------
# If the notebook and v5_pipeline.py are out of step (a pull that did not reach the
# kernel, or an editor still holding the old file), say so here instead of letting a
# later cell die on AttributeError.
try:
    import importlib, v5_pipeline
    _P = importlib.reload(v5_pipeline)
    _need = 3
    _have = getattr(_P, "VERSION", 0)
    print(f"v5_pipeline VERSION {_have} (this notebook wants >= {_need})")
    assert _have >= _need, (
        f"v5_pipeline.py is version {_have} but this notebook needs {_need}. "
        "Pull again, then close and reopen the notebook so the editor rereads it.")
    for _fn in ("load_modules", "load_modules_v5", "get_tables", "get_tables_v5",
                "scale_weights", "make_wmase", "fit_post_weighted", "apply_post"):
        assert hasattr(_P, _fn), f"v5_pipeline.py is missing {_fn}() - stale copy"
    print("v5_pipeline: all required functions present")
except ImportError:
    print("v5_pipeline.py not found - it must sit beside this notebook")
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
    return hits[0] if hits else None


_p = find_sub(SUB_PATH)
HAVE_SUB = _p is not None
if not HAVE_SUB:
    print("No submission csv found, so this cell has nothing to diagnose.")
    print("Put submission_v4.csv beside the notebook (or set SUB_PATH in CELL 1) and re-run.")
    print("CELLS 8 and 11-14 do not need it; CELL 10 does.")
sub = pd.read_csv(_p) if HAVE_SUB else None
if HAVE_SUB:
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

md(r"""
## CELL 10 — is post-processing closed? the MAE-correct version of T4

T4 compared **means**. MASE is L1, so the optimal point forecast is the conditional **median**,
and a mean-matching multiplier pushes predictions *above* the median on a right-skewed
distribution. That is precisely the documented `+0.0163` regression in build 1 ("far horizons
shipped at the mean, not the median"). So T4's `mean_mult` of 1.26 at D4 and 1.45 at D5 is **not**
evidence of a 26-45% under-prediction — it is mostly skew.

This cell redoes it properly, per `(horizon x D1-weekday)` cell on ordinary rows:

1. **Quantile comparison** — true vs predicted q25/q50/q75. Assumption-light: if v4 is
   distributionally calibrated the medians agree, and any gap at the median is the real bias.
2. **MAE-optimal multiplier** — for each cell, scan `m` and evaluate the true expected MAE of
   `m x pred` against the empirical true ratio distribution, via sorted prefix sums so it is
   exact rather than sampled.

The scan's assumption is stated explicitly because it matters: it treats the true ratio as
**independent of v4's prediction within a cell**, i.e. it credits v4 with no within-cell
discrimination. That makes the *level* correction it finds trustworthy and the *gain* it reports
an over-estimate. Read the sign and the size, not the decimal.

Everything here is measured on Apr-Sep 2025 films and applied to an Oct-Mar test, which is the
exact time shift that made model Z lose 0.065 on the leaderboard. Treat a positive result as a
hypothesis needing one probe, not as a win.
""")
code(r"""
# CELL 10 — MAE-OPTIMAL PER-HORIZON LEVEL CHECK (medians, not means)
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)
_DOW = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

if not (CACHE_DIR / "sub_diag.pkl").exists():
    print("cache/sub_diag.pkl missing -> run CELL 7 with submission_v4.csv present.")
    print("This cell compares that submission against the release-anchored truth.")
else:
    G = pd.read_pickle(CACHE_DIR / "release_anchored_grid.pkl")
    D = pd.read_pickle(CACHE_DIR / "sub_diag.pkl")
    # real data gives ~1000-3200 true rows per (horizon x dow); a synthetic fixture gives single digits
    MIN_TRUE = MIN_PRED = 150 if len(G) > 10_000 else 5
    D["d1_dow"] = pd.DatetimeIndex(D.d1).dayofweek
    Gp = G[(G.lab == "ordinary") & (G.lab_d1 == "ordinary")]
    Dp = D[D.label == "ordinary"]
    print(f"true grid (ordinary) {len(Gp):,} rows | v4 preds (ordinary) {len(Dp):,} rows")


    def mad_to(c, ys, cs):
        # exact mean |y - c| for every c against sorted ys with prefix sums cs
        c = np.asarray(c, dtype=float)
        k = np.searchsorted(ys, c)
        left = k * c - cs[k]
        right = (cs[-1] - cs[k]) - (len(ys) - k) * c
        return (left + right) / len(ys)


    line("Q1  true vs predicted quantiles per (horizon x D1-dow), ordinary rows")
    rows, grid_m = [], np.round(np.arange(0.40, 2.01, 0.02), 2)
    _i1 = int(np.where(grid_m == 1.0)[0][0])
    _drop = 0
    for (h, dw), gt in Gp.groupby(["off", "d1_dow"]):
        pr = Dp[(Dp.h == h) & (Dp.d1_dow == dw)]
        y = gt.ratio.to_numpy(dtype=float)
        p = pr.ratio.to_numpy(dtype=float)
        y, p = y[np.isfinite(y)], p[np.isfinite(p)]      # a NaN scale must not poison a whole cell
        _drop += int((~np.isfinite(gt.ratio.to_numpy(dtype=float))).sum())
        if len(y) < MIN_TRUE or len(p) < MIN_PRED:
            continue
        y = np.sort(y)
        cs = np.concatenate([[0.0], np.cumsum(y)])
        curve = np.array([mad_to(m * p, y, cs).mean() for m in grid_m])
        assert np.isfinite(curve).all(), f"non-finite MAE curve at h={h} dow={dw}"
        j = int(curve.argmin())
        _tm, _pm = float(np.median(y)), float(np.median(p))
        rows.append(dict(h=h, dow=_DOW[dw], n_true=len(y), n_pred=len(p),
                         true_q25=np.quantile(y, .25), true_med=_tm,
                         true_q75=np.quantile(y, .75),
                         pred_q25=np.quantile(p, .25), pred_med=_pm,
                         pred_q75=np.quantile(p, .75),
                         med_mult=(_tm / _pm) if _pm > 0 else np.nan,
                         med_agree=(abs(_tm - _pm) < 1e-9) or (_pm > 0 and abs(_tm / _pm - 1) < 0.10),
                         best_m=grid_m[j], mae_at_1=curve[_i1], mae_best=curve[j]))
    Q = pd.DataFrame(rows)
    assert len(Q) > 0, "no (horizon x dow) cell had enough rows on both sides"
    if _drop:
        print(f"note: dropped {_drop:,} non-finite true ratios before any statistic")
    Q["gain"] = Q.mae_at_1 - Q.mae_best
    print(Q[["h", "dow", "n_true", "n_pred", "true_q25", "true_med", "true_q75",
             "pred_q25", "pred_med", "pred_q75"]].round(3).to_string(index=False))

    line("Q2  MAE-optimal multiplier per cell  (med_mult = median-matching, best_m = MAE-optimal)")
    print(Q[["h", "dow", "n_pred", "med_mult", "best_m", "mae_at_1", "mae_best", "gain"]]
          .round(4).to_string(index=False))
    print("\nbest_m pinned to a grid edge (0.40 / 2.00) means the optimum is outside the scan:",
          sorted(set(Q.best_m[(Q.best_m <= 0.40) | (Q.best_m >= 2.00)])) or "none")

    line("Q3  what this is worth over the whole submission, if it transfers")
    _wt = Q.n_pred / len(Dp)
    _ord_share = len(Dp) / len(D)
    _cell = float((Q.gain * _wt).sum())
    print(f"cells covered: {len(Q)} | {int(Q.n_pred.sum()):,} of {len(Dp):,} ordinary rows "
          f"({Q.n_pred.sum() / len(Dp):.1%}) | ordinary = {_ord_share:.1%} of all test rows")
    print(f"weighted MAE gain on the covered ordinary rows : {_cell:.4f}")
    print(f"=> upper bound on total MASE gain              : {_cell * _ord_share * (Q.n_pred.sum() / len(Dp)):.4f}")
    print("\nThis is an UPPER bound and almost certainly optimistic: the scan assumes v4 has no")
    print("within-cell discrimination, and it is fitted on Apr-Sep films then applied to Oct-Mar.")

    line("Q4  the honest read on the median")
    print(f"cells where the predicted median already matches the true median: "
          f"{int(Q.med_agree.sum())} of {len(Q)}")
    print("  (counts a both-zero median as agreement - at D9/D10 the true median IS 0)")
    _mm = Q[(Q.pred_med > 0) & (Q.true_med > 0)]
    if len(_mm):
        print(f"across the {len(_mm)} cells with both medians non-zero, med_mult"
              f" median {_mm.med_mult.median():.3f} | range {_mm.med_mult.min():.3f}"
              f"-{_mm.med_mult.max():.3f}")
    print("if med_mult sits near 1.0, v4's LEVEL is right and the T4 mean gap was skew,")
    print("which closes level post-processing and leaves only the film-curve (oracle A) lever.")
    Q.to_pickle(CACHE_DIR / "horizon_level_check.pkl")
print(f"\n[CELL 10] {time.time() - t0:.1f}s")
""")

md(r"""
## CELL 11 — the decisive test: v4's own OOF, and whether `H_GRID` is clipped

`cinema_forecasting_v3.py` arrived, and reading it **overturns the population-mismatch thesis**:

- `run_cv` already scores `val_pool = (data.window == 0) & ~data.bad` — v4's CV is **already
  release-anchored**, so its 0.344 / 0.354 is measured on the same opening-window population as
  the test. My round-2 claim that v4 validates on mid-run windows was wrong.
- `make_folds(pool, 'time')` already builds **purged** date-blocked folds.
- v4 already up-weights opening windows (`w_open`), already fits a zero-snap (`ZERO_GRID`
  0.00-1.30) and already fits a per-horizon multiplier (`H_GRID`).

Which means CELL 10's Q1 gap is **confounded**: it compares *train* truth (Apr-Sep films) against
*test* predictions (Oct-Mar films). A gap can mean v4 is biased, or simply that the two film
populations differ. It cannot distinguish them, so it must not be acted on.

But one thing in v4 is checkable and concrete:

```
H_GRID = np.round(np.arange(0.85, 1.155, 0.01), 2)      # floor 0.85
```

CELL 10's scan wanted 0.40-0.68 at D6-D10. If the optimum on v4's **own OOF** also sits below
0.85, the grid is pinned at its floor and widening it is a free, properly-validated win — the same
clipped-grid bug the other lineage already paid for. If the optimum sits inside the grid, then
CELL 10's gap was a population artifact and level post-processing is closed.

This cell settles it on data where the truth is known. It fits the multipliers **cross-fitted**
(parameters from four folds, scored on the fifth), because fitting and scoring on the same OOF is
leakage and would manufacture a win. It reports both CV schemes, and the `time` scheme is the gate.

Heavy: builds ~545k windows and runs 5-fold LightGBM. Gated behind `RUN_HEAVY`, resumable, and it
caches the window table and the OOF so later cells are instant.
""")
code(r"""
# CELL 11 — REPRODUCE v4 OOF, THEN TEST THE CLIPPED H_GRID (heavy, resumable)
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)
import json as _json
import importlib, v5_pipeline
P = importlib.reload(v5_pipeline)  # never trust Jupyter's cached copy          # shared setup; keeps every heavy cell independently runnable

FOLD_KINDS = ("group", "time")

if not RUN_HEAVY:
    print("RUN_HEAVY is False -> skipping. Set RUN_HEAVY = True in CELL 1.")
else:
    # v5_pipeline handles the three things that are easy to get wrong:
    #   * v3.load_data() hard-codes DATA='data/' and canonical names ("train (1).csv" here)
    #   * v3.encode_cats() must see the train AND test frames in ONE call or codes misalign
    #   * featsets F/G/H need v4.load_table(), not v3.prepare()
    v3, v4 = P.load_modules(".")
    data, test = P.get_tables(v3, v4)
    print(f"\nfeatset {V4_FS} ({len(v4.FEATSETS[V4_FS])} feats) | w_open {V4_WOPEN} "
          f"| paramset {V4_PS}")
    print(f"v4 H_GRID: {v4.H_GRID.min():.2f} .. {v4.H_GRID.max():.3f}   <-- floor under test")
    _vp = P.release_mask(data)
    print(f"rows {len(data):,} | clean release (offset 0) rows {int(_vp.sum()):,} "
          f"| opening-offset rows {int(data.window.isin(v3.OPEN_OFFSETS).sum()):,}")

    # --------------------------------------------- 2. OOF per fold scheme (cached)
    pool = data[_vp]
    OOF = {}
    for kind in FOLD_KINDS:
        _op = CACHE_DIR / f"oof_{V4_FS}_{V4_WOPEN}_{V4_PS}_{kind}.pkl"
        if _op.exists():
            OOF[kind] = pd.read_pickle(_op)
            print(f"[{kind}] OOF from cache ({int(OOF[kind].notna().sum()):,} rows)")
        else:
            print(f"[{kind}] running 5-fold CV ...", flush=True)
            _f = v3.make_folds(pool, kind)
            _o, _it = v4.run_cv(data, v4.FEATSETS[V4_FS], _f, v4.PARAMSETS[V4_PS],
                                w_open=V4_WOPEN)
            OOF[kind] = _o
            pd.to_pickle(_o, _op)
            print(f"[{kind}] done, best_iters {_it}")
        _s = data[_vp & OOF[kind].notna()]
        _m = (_s.y - OOF[kind][_s.index]).abs().mean()
        print(f"[{kind}] pooled MASE on clean release windows = {_m:.4f}  "
              f"({len(_s):,} rows)")
        (OUT_DIR / f"cv_{V4_FS}_{V4_WOPEN}_{V4_PS}_{kind}.json").write_text(
            _json.dumps({"mase": float(_m), "rows": int(len(_s))}))

    # ------------------------- 3. the real question: where does the optimum sit?
    def opt_mult(y, p, grid):
        # MAE-optimal scalar multiplier on p against known y
        return grid[int(np.argmin([np.abs(y - g * p).mean() for g in grid]))]

    WIDE = np.round(np.arange(0.20, 1.401, 0.01), 2)
    for kind in FOLD_KINDS:
        line(f"H1  [{kind}] MAE-optimal per-horizon multiplier: v4's grid vs a wide grid")
        s = data[_vp & OOF[kind].notna()].copy()
        s["p"] = OOF[kind][s.index].to_numpy()
        s["dow"] = s.d1.dt.dayofweek
        rows = []
        for h, g in s.groupby("h"):
            y, p = g.y.to_numpy(), g.p.to_numpy()
            m_v4, m_wide = opt_mult(y, p, v4.H_GRID), opt_mult(y, p, WIDE)
            rows.append(dict(h=h, rows=len(g), mase_at_1=np.abs(y - p).mean(),
                             m_v4grid=m_v4, mase_v4grid=np.abs(y - m_v4 * p).mean(),
                             m_wide=m_wide, mase_wide=np.abs(y - m_wide * p).mean(),
                             pinned=bool(m_v4 <= v4.H_GRID.min() + 1e-9)))
        H = pd.DataFrame(rows)
        H["extra_gain"] = H.mase_v4grid - H.mase_wide
        print(H.round(4).to_string(index=False))
        _np = int(H.pinned.sum())
        print(f"\nhorizons pinned to v4's 0.85 floor: {_np} of {len(H)}")
        print(f"extra MASE from widening the grid (row-weighted): "
              f"{float((H.extra_gain * H.rows).sum() / H.rows.sum()):.4f}")
        if _np == 0:
            print("NOT pinned -> v4's grid is adequate and CELL 10's gap was a population")
            print("artifact. Level post-processing is closed; do not spend a submission on it.")

        line(f"H2  [{kind}] cross-fitted: does widening actually generalise?")
        _folds = v3.make_folds(pool, kind)
        for tag, keys, grid in (("per-horizon, v4 grid", ["h"], v4.H_GRID),
                                ("per-horizon, wide", ["h"], WIDE),
                                ("per-horizon x D1-dow, wide", ["h", "dow"], WIDE)):
            base, corr, n = 0.0, 0.0, 0
            for vm, _ in _folds:
                va = s[s.movie_title.isin(vm)]
                trn = s[~s.movie_title.isin(vm)]
                if not len(va) or not len(trn):
                    continue
                mt = {k: opt_mult(g.y.to_numpy(), g.p.to_numpy(), grid)
                      for k, g in trn.groupby(keys)}
                mm = np.array([mt.get(k, 1.0) for k in
                               (zip(*[va[c] for c in keys]) if len(keys) > 1 else va[keys[0]])])
                base += np.abs(va.y - va.p).sum()
                corr += np.abs(va.y.to_numpy() - mm * va.p.to_numpy()).sum()
                n += len(va)
            print(f"  {tag:30s} MASE {base / n:.4f} -> {corr / n:.4f}  "
                  f"(gain {(base - corr) / n:+.4f})")
        print("\nA gain here is measured the honest way: multipliers fitted on four folds and")
        print("scored on the fifth. Only act if BOTH schemes agree, and trust 'time' more.")
        s[["h", "dow", "y", "p", "movie_title", "d1", "scale"]].to_pickle(
            CACHE_DIR / f"oof_frame_{kind}.pkl")
print(f"\n[CELL 11] {time.time() - t0:.1f}s")
""")

md(r"""
## CELL 12 — candidate feature experiments against v4's own CV

Two candidates, both run through v4's unmodified `run_cv` on **both** fold schemes, so the
both-schemes gate applies. Results are already recorded in the DECISION LOG; this cell exists so
they are reproducible rather than taken on trust.

1. **`film_curve` (oracle A).** A stage-1 LightGBM predicts each film-window's aggregate
   `sum(tickets)/sum(scale)` curve, out of fold against the same film folds, and the prediction is
   exposed as a feature. Oracle A's ceiling was quoted at +0.0905.
2. **Sibling format variants.** 16 base titles ship more than one format (IMAX/3D/dubbed). For
   each `(base_title, cluster, window, horizon)` the sibling's D1-D3 strength is added, excluding
   the row itself. Listed as unexplored in `HANDOVER.md` §10.5.

Both were **rejected**. Keep this cell for the harness: swapping in a new feature list is a
two-line change, and it is the only place a candidate can be judged honestly.
""")
code(r"""
# CELL 12 — CANDIDATE FEATURES vs v4 CV (heavy, gated)
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

if not RUN_HEAVY:
    print("RUN_HEAVY is False -> skipping. Set RUN_HEAVY = True in CELL 1.")
else:
    import lightgbm as lgb
    import importlib, v5_pipeline
    P = importlib.reload(v5_pipeline)  # never trust Jupyter's cached copy
    v3, v4 = P.load_modules(".", verbose=False)
    data, test = P.get_tables(v3, v4, verbose=False)
    vp = (data.window == 0) & ~data.bad
    pool = data[vp]
    BASE_FEATS = v4.FEATSETS[V4_FS]

    def bench(frame, label_feats, kinds=("group", "time")):
        # one row per (scheme, variant): v4's own run_cv, nothing reweighted
        for kind in kinds:
            folds = v3.make_folds(pool, kind)
            out = {}
            for tag, feats in label_feats:
                _t = time.time()
                o, _ = v4.run_cv(frame, feats, folds, v4.PARAMSETS[V4_PS], w_open=V4_WOPEN)
                s = frame[vp & o.notna()]
                out[tag] = float((s.y - o[s.index]).abs().mean())
                print(f"  [{kind}] {tag:22s} MASE {out[tag]:.4f}  ({time.time() - _t:.0f}s)",
                      flush=True)
            _b = out[label_feats[0][0]]
            for tag in list(out)[1:]:
                _d = out[tag] - _b
                print(f"  [{kind}] --> {tag} delta {_d:+.4f}"
                      f"  {'IMPROVES' if _d < 0 else 'WORSE'}")
            yield kind, out

    # ---------------------------------------------------- 1. film_curve (oracle A)
    line("E1  film_curve: a stage-1 film-window curve exposed as a feature")
    K = ["movie_title", "window"]
    S1F = ["h", "nat_log", "nat_r31", "nat_r32", "nc3", "nat_tps3", "d1_dow", "dow", "price",
           "genre1", "age_rating"]
    agg = (data.groupby(K + ["h"], as_index=False)
           .agg(tick=("total_ticket", "sum"), sc=("scale", "sum"),
                **{c: (c, "first") for c in S1F if c != "h"}))
    agg["fy"] = agg.tick / agg.sc
    S1P = dict(objective="l1", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED)
    _md = float(np.abs(agg.fy - agg.groupby("h").fy.transform("median")).mean())
    for kind in ("group", "time"):
        agg["fc"] = np.nan
        for vm, _ in v3.make_folds(pool, kind):
            trm = ~agg.movie_title.isin(vm)
            if trm.all():
                continue
            _m = lgb.train(S1P, lgb.Dataset(agg.loc[trm, S1F], agg.loc[trm, "fy"]), 400)
            agg.loc[~trm, "fc"] = _m.predict(agg.loc[~trm, S1F])
        d2 = data.merge(agg[K + ["h", "fc"]], on=K + ["h"], how="left")
        d2.index = data.index
        print(f"[{kind}] stage-1 OOF MAE {float(np.abs(agg.fy - agg.fc).mean()):.4f} vs "
              f"per-horizon median {_md:.4f} | coverage {d2.fc.notna().mean():.1%}"
              f"  (NaN = mid-run-only films, which are never validated)")
        list(bench(d2, [("v4 baseline", BASE_FEATS),
                        ("v4 + film_curve", BASE_FEATS + ["fc"])], kinds=(kind,)))
    print("\nstage 1 HAS skill, yet the feature hurts: v4 already carries the film-level D1-D3")
    print("aggregates (nat_log/nat_r31/nat_r32/nc3/nat_tps3/share3) that stage 1 is built from,")
    print("so `fc` is a strictly less expressive function of inputs the main model already sees.")

    # ------------------------------------------------- 2. sibling format variants
    line("E2  sibling format variants (IMAX / 3D / dubbed siblings of the same base title)")
    data["base_title"] = (data.movie_title.astype(str)
                          .str.replace(r"\s*\((?![^)]*\b(?:PART|CHAPTER|VOL)\b)[^)]*\)\s*$",
                                       "", regex=True).str.strip())
    _nb = data.groupby("base_title").movie_title.nunique()
    _multi = set(_nb[_nb > 1].index)
    print(f"base titles {len(_nb)} | with >1 format {len(_multi)} | rows in a multi-format "
          f"family {data.base_title.isin(_multi).mean():.1%}")
    _g = data.groupby(["base_title", "cinema_ids", "window", "h"], as_index=False).agg(
        fam_t3=("t3", "sum"), fam_t1=("t1", "sum"), fam_n=("t3", "size"), fam_sc=("scale", "sum"))
    d3 = data.merge(_g, on=["base_title", "cinema_ids", "window", "h"], how="left")
    d3.index = data.index
    d3["sib_n"] = d3.fam_n - 1
    d3["sib_t3"] = d3.fam_t3 - d3.t3
    d3["sib_t1"] = d3.fam_t1 - d3.t1
    d3["sib_sc"] = d3.fam_sc - d3.scale
    d3["sib_share"] = np.where(d3.fam_t3 > 0, d3.t3 / d3.fam_t3, 1.0)
    d3["sib_rel"] = np.where(d3.scale > 0, d3.sib_sc / d3.scale, 0.0)
    SIB = ["sib_n", "sib_t3", "sib_t1", "sib_sc", "sib_share", "sib_rel"]
    print(f"rows with at least one sibling: {(d3.sib_n > 0).mean():.1%}")
    list(bench(d3, [("v4 baseline", BASE_FEATS), ("v4 + sibling", BASE_FEATS + SIB)]))
    print("\nNote the split verdict: this helps the grouped scheme and hurts the purged time")
    print("scheme. That is the model-Z signature, so the gate rejects it.")
print(f"\n[CELL 12] {time.time() - t0:.1f}s")
""")

md(r"""
## CELL 13 — is model diversity worth anything here? (the ensemble / neural-net question)

v4 already blends two LightGBMs plus a zero classifier. This cell asks whether *any* additional
model class can help, by adding a deliberately alien third class — a hierarchical median lookup
table over `h x d1_dow x scale bin`, with no feature interactions and no boosting — and measuring
**absolute-error correlation** and **cross-fitted blend gain** on both fold schemes.

The point is not the median table itself. It is the correlation: if a model this structurally
different still tracks LightGBM's errors, then the residual is irreducible given these features
and no architecture — neural net included — changes that. Run it before spending days on a new
model class.

Results are in the DECISION LOG (round 5). Reuses CELL 11's cached window table.
""")
code(r"""
# CELL 13 — MODEL DIVERSITY CEILING (moderate; needs CELL 11's cache)
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)
import itertools

if not RUN_HEAVY:
    print("RUN_HEAVY is False -> skipping. Set RUN_HEAVY = True in CELL 1.")
else:
    import importlib, v5_pipeline
    P = importlib.reload(v5_pipeline)  # never trust Jupyter's cached copy
    v3, v4 = P.load_modules(".", verbose=False)
    data, test = P.get_tables(v3, v4, verbose=False)
    vp = (data.window == 0) & ~data.bad
    pool, s = data[vp], data[vp]
    yy = s.y.to_numpy()
    F = v4.FEATSETS[V4_FS]

    def med_oof(kind):
        # deliberately alien third class: hierarchical median table, no interactions
        folds = v3.make_folds(pool, kind)
        o = pd.Series(np.nan, index=data.index)
        d = data.assign(sb=pd.cut(data.scale, [0, 5, 15, 40, 100, 1e9], labels=False))
        base = d.window.isin(v3.ALL_OFFSETS) & ~d.bad
        we = d.d1 + pd.Timedelta(days=9)
        for vm, pu in folds:
            iv = d.movie_title.isin(vm)
            trm = base & ~iv
            if pu is not None:
                trm &= ~((d.d1 <= pu[1]) & (we >= pu[0]))     # same purge as v3
            T = d[trm]
            t3 = T.groupby(["h", "d1_dow", "sb"], observed=True).y.median()
            t2 = T.groupby(["h", "d1_dow"], observed=True).y.median()
            gm = T.y.median()
            o[vp & iv] = [t3.get((a, b, c), t2.get((a, b), gm)) for a, b, c in
                          zip(d.loc[vp & iv, "h"], d.loc[vp & iv, "d1_dow"],
                              d.loc[vp & iv, "sb"])]
        return o

    def simp(n, st=0.1):
        g = np.round(np.arange(0, 1 + 1e-9, st), 2)
        for c in itertools.product(g, repeat=n - 1):
            if sum(c) <= 1 + 1e-9:
                yield np.array(list(c) + [round(1 - sum(c), 2)])

    for kind in ("group", "time"):
        line(f"D1  [{kind}] standalone accuracy and absolute-error correlation")
        O = {}
        _ap = CACHE_DIR / f"oof_{V4_FS}_{V4_WOPEN}_{V4_PS}_{kind}.pkl"
        assert _ap.exists(), f"run CELL 11 first to produce {_ap}"
        O["A"] = pd.read_pickle(_ap)
        for nm, mk in (("B", lambda: v4.run_cv(data, F, v3.make_folds(pool, kind),
                                               v4.PARAMS_B, w_open=V4_WOPEN)[0]),
                       ("M", lambda: med_oof(kind))):
            _p = CACHE_DIR / f"oof_{nm}_{kind}.pkl"
            if _p.exists():
                O[nm] = pd.read_pickle(_p)
            else:
                _t = time.time()
                O[nm] = mk()
                pd.to_pickle(O[nm], _p)
                print(f"  trained {nm} in {time.time() - _t:.0f}s", flush=True)
        eA = np.abs(yy - O["A"][s.index].to_numpy())
        for nm, lab in (("A", "A  LightGBM P2"), ("B", "B  LightGBM extra_trees"),
                        ("M", "M  median table (alien class)")):
            e = np.abs(yy - O[nm][s.index].to_numpy())
            print(f"  {lab:32s} MASE {e.mean():.4f} | abs-err corr vs A "
                  f"{np.corrcoef(e, eA)[0, 1]:.3f}")

        line(f"D2  [{kind}] cross-fitted blend gains")
        folds = v3.make_folds(pool, kind)
        for names in (["A", "B"], ["A", "M"], ["A", "B", "M"]):
            b = c = n = 0
            for vm, _ in folds:
                m = s.movie_title.isin(vm)
                va, tr = s[m], s[~m]
                if not len(va) or not len(tr):
                    continue
                Ptr = np.column_stack([O[x][tr.index].to_numpy() for x in names])
                Pva = np.column_stack([O[x][va.index].to_numpy() for x in names])
                ytr = tr.y.to_numpy()
                bw = min(simp(len(names)), key=lambda wv: np.abs(ytr - Ptr @ wv).mean())
                b += np.abs(va.y.to_numpy() - Pva[:, 0]).sum()
                c += np.abs(va.y.to_numpy() - Pva @ bw).sum()
                n += len(va)
            print(f"  A+{'+'.join(names[1:]):6s}  {b / n:.4f} -> {c / n:.4f}"
                  f"   gain {(b - c) / n:+.4f}")

        line(f"D3  [{kind}] why: the error is aleatoric and concentrated")
        P = np.column_stack([O[x][s.index].to_numpy() for x in ("A", "B", "M")])
        E = np.abs(yy[:, None] - P)
        print(f"  oracle per-row model choice {E.min(1).mean():.4f} vs best single "
              f"{E.mean(0).min():.4f}  (hindsight only, no feature selects the winner)")
        e = E[:, 0]
        q = np.argsort(-e)
        for f in (0.01, 0.05, 0.10, 0.25):
            k = int(f * len(e))
            print(f"  worst {f:4.0%} of rows carry {e[q[:k]].sum() / e.sum():6.1%} of all error"
                  f" | their mean true y/scale {np.abs(yy[q[:k]]).mean():7.2f}")
        print("\n  A fifth of the error sits in 1% of rows whose truth is ~8x their own D1-D3")
        print("  baseline. That is unpredictable from these features, and MAE forbids chasing it")
        print("  because the optimum is the median, which v4 already matches.")
print(f"\n[CELL 13] {time.time() - t0:.1f}s")
""")

md(r"""
## CELL 14 — the test-mix-weighted metric, and the two changes that survived

**This is the most useful cell in the notebook.** `W1` builds the scale-bin importance weights
and shows that reweighting v4's CV to the test's scale distribution reproduces the leaderboard
(0.4482 / 0.4551 against an actual 0.43715). Use `wmase()` as the selection metric for every
future experiment — selecting on plain CV is what made six rounds of gains fail to transfer.

`W2` then applies the only two changes that passed the both-schemes gate and writes
`submission_v5a.csv`:

1. **featset `F`** (`FEAT_E + COMP`) instead of `H` — dropping the `CLDOW` block. -0.0017 time /
   -0.0022 group.
2. **post-processing fitted on the weighted metric** rather than plain MASE. -0.0023 time /
   -0.0004 group.

Expect roughly **0.437 -> 0.433**. Modest and verified. Set `RUN_HEAVY = True`.
""")
code(r"""
# CELL 14 — WEIGHTED METRIC + submission_v5a.csv
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

if not RUN_HEAVY:
    print("RUN_HEAVY is False -> skipping. Set RUN_HEAVY = True in CELL 1.")
else:
    import importlib, v5_pipeline
    P = importlib.reload(v5_pipeline)  # never trust Jupyter's cached copy
    v3, v4 = P.load_modules(".", verbose=False)
    data, test = P.get_tables(v3, v4, verbose=False)
    vp = P.release_mask(data)

    line("W1  the gap is scale composition - build the weighted metric")
    w_eval, EW, sb = P.scale_weights(data, test)
    wmase = P.make_wmase(data, EW)
    for kind in FOLD_KINDS:
        _p = CACHE_DIR / f"oof_{V4_FS}_{V4_WOPEN}_{V4_PS}_{kind}.pkl"
        if _p.exists():
            a, b = wmase(pd.read_pickle(_p))
            print(f"  [{kind}] v4 pooled CV {a:.4f} -> REWEIGHTED {b:.4f}   (actual LB 0.43715)")
    print("\n  ALWAYS select on the weighted number. Selecting on plain CV is what made six")
    print("  rounds of apparent gains fail to transfer to the leaderboard.")

    line(f"W2  OOF for featset {V5_FS} on both schemes (cached)")
    feats = v4.FEATSETS[V5_FS]
    folds = {k: v3.make_folds(data[vp], k) for k in FOLD_KINDS}
    SPEC = {"A": (v4.PARAMSETS[V4_PS], "reg", 5), "B": (v4.PARAMS_B, "reg", 3),
            "p0": (v4.PARAMSETS[V4_PS], "clf", 3)}
    OO, ITERS = {}, {}
    for nm, (pp, kd, _) in SPEC.items():
        OO[nm], it = {}, []
        for kind in FOLD_KINDS:
            _c = CACHE_DIR / f"oofF_{nm}_{kind}.pkl"
            _ci = OUT_DIR / f"iters_{nm}_{kind}.json"
            if _c.exists() and _ci.exists():
                OO[nm][kind] = pd.read_pickle(_c)
                it += _json.loads(_ci.read_text())
            else:
                _t = time.time()
                o, i = v4.run_cv(data, feats, folds[kind], pp, w_open=V4_WOPEN, kind=kd)
                OO[nm][kind] = o
                pd.to_pickle(o, _c)
                _ci.write_text(_json.dumps(i))
                it += i
                print(f"  trained {nm} [{kind}] in {time.time() - _t:.0f}s", flush=True)
        ITERS[nm] = max(50, int(np.mean(it)))
    for kind in FOLD_KINDS:
        a, b = wmase(OO["A"][kind])
        print(f"  [{kind}] featset {V5_FS} raw model A: plain {a:.4f} | WEIGHTED {b:.4f}")
    print(f"  mean best_iteration per model: {ITERS}")

    line("W3  fit post-processing on the WEIGHTED objective, cross-fitted to verify")
    frames = {}
    for kind in FOLD_KINDS:
        idx = OO["A"][kind].index
        fr = pd.DataFrame({"y": data.y.loc[idx], "A": OO["A"][kind].clip(lower=0),
                           "B": OO["B"][kind].clip(lower=0), "p0": OO["p0"][kind].loc[idx],
                           "h": data.h.loc[idx], "sb": sb.loc[idx],
                           "mv": data.movie_title.loc[idx]})
        fr["w"] = w_eval[fr.sb.to_numpy()]
        frames[kind] = fr
    for kind in FOLD_KINDS:
        fr = frames[kind]
        raw = float((fr.w * np.abs(fr.y - fr.A)).sum() / fr.w.sum())
        print(f"  [{kind}] raw A                        WEIGHTED {raw:.4f}")
        for wt, lab in ((False, "post fit on PLAIN (v4 as-is)"), (True, "post fit on WEIGHTED")):
            num = den = pn = 0.0
            for vm, _ in folds[kind]:
                m = fr.mv.isin(vm)
                va, tr = fr[m], fr[~m]
                if not len(va) or not len(tr):
                    continue
                r = P.apply_post(va, P.fit_post_weighted(tr, v3, v4, wt), v4)
                e = np.abs(va.y.to_numpy() - r)
                num += (va.w.to_numpy() * e).sum()
                den += va.w.to_numpy().sum()
                pn += e.sum()
            print(f"  [{kind}] {lab:28s} plain {pn / len(fr):.4f} | WEIGHTED {num / den:.4f}")
    # ship the version fitted on both schemes pooled, weighted objective
    post = P.fit_post_weighted(pd.concat(frames.values()), v3, v4, weighted=True)
    print(f"\n  shipped post: blend A weight {post['w']:.2f}")
    print(f"    zero multipliers per p0 decile {np.round(post['zm'], 2).tolist()}")
    print(f"    horizon multipliers {post['hm']}")

    line("W4  train on all clean windows and write submission_v5a.csv")
    preds = {}
    for nm, (pp, kd, nseed) in SPEC.items():
        seeds = [SEED + {"A": 0, "B": 100, "p0": 200}[nm] + i for i in range(nseed)]
        boosters = v4.train_full(data, feats, pp, ITERS[nm], seeds, V4_WOPEN, kd)
        preds[nm] = np.mean([m.predict(test[feats]) for m in boosters], axis=0)
        print(f"  {nm}: {len(boosters)} seeds x {ITERS[nm]} rounds", flush=True)
    tf = pd.DataFrame({"A": np.clip(preds["A"], 0, None), "B": np.clip(preds["B"], 0, None),
                       "p0": preds["p0"], "h": test.h.to_numpy()})
    ratio = np.clip(P.apply_post(tf, post, v4), 0, None)
    # v4's final() rule: a pair with no D1-D3 tickets cannot sell any
    _hist = (test.t1.fillna(0) + test.t2.fillna(0) + test.t3.fillna(0)).to_numpy()
    ratio[_hist == 0] = 0
    print(f"  rows forced to zero by the no-history rule: {int((_hist == 0).sum()):,}")
    v4.write_sub("submission_v5a.csv", test, ratio, test.scale.to_numpy())
    _old = Path("submission_v4.csv")
    if _old.exists():
        _o = pd.read_csv(_old)
        _n = pd.read_csv("submission_v5a.csv")
        _d = (_o.total_ticket.to_numpy() != _n.total_ticket.to_numpy())
        print(f"  rows differing from submission_v4.csv: {int(_d.sum()):,} ({_d.mean():.1%})"
              f" | total tickets {_o.total_ticket.sum():,} -> {_n.total_ticket.sum():,}")
    print("\n  SUBMIT submission_v5a.csv. Expected: 0.43715 -> about 0.433.")
    print("  The two changes are featset F instead of H, and post-processing fitted on the")
    print("  test-mix-weighted objective. Both passed the group AND time schemes.")
print(f"\n[CELL 14] {time.time() - t0:.1f}s")
""")

md(r"""
## CELL 15 — is zero an absorbing state, and can we enforce it?

v4 predicts every `(pair, day)` **independently**. Nothing stops it emitting `0.5, 0, 0.3` across
D8-D10, which is physically odd: once a cinema drops a film it rarely brings it back. If zero is
close to absorbing in the data, enforcing that on predictions is free information v4 cannot
express.

This is cheap and it is the first experiment here that exploits structure *across* horizons rather
than treating rows independently. Three steps:

- **Z1/Z2** measure the real resurrection rate on release-anchored train windows: given `y = 0` at
  horizon `h`, how often is any later horizon positive? That is the honest ceiling on the idea --
  if films come back often, enforcing absorption destroys those rows.
- **Z3** counts how often v4's own post-processed predictions already violate the pattern.
- **Z4** fits a threshold `tau`: once the prediction drops below `tau`, zero that horizon and
  every later one. Fitted on four folds, scored on the fifth, on the **weighted** objective, on
  both fold schemes.

Needs CELL 14's cached OOF (`cache/oofF_*.pkl`). No model training, so it runs in seconds.
""")
code(r"""
# CELL 15 — ZERO AS AN ABSORBING STATE (cheap; needs CELL 14's cached OOF)
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)
PAIR = ["movie_title", "cinema_ids"]

if not RUN_HEAVY:
    print("RUN_HEAVY is False -> skipping. Set RUN_HEAVY = True in CELL 1.")
else:
    import importlib, v5_pipeline
    P = importlib.reload(v5_pipeline)  # never trust Jupyter's cached copy
    v3, v4 = P.load_modules(".", verbose=False)
    data, test = P.get_tables(v3, v4, verbose=False)
    vp = P.release_mask(data)
    w_eval, EW, sb = P.scale_weights(data, test, verbose=False)
    s = data[vp]
    assert not s.duplicated(PAIR + ["h"]).any(), "pair x horizon is not unique in release windows"

    line("Z1  resurrection: given y == 0 at horizon h, is any LATER horizon positive?")
    Y = s.pivot_table(index=PAIR, columns="h", values="total_ticket", aggfunc="sum")
    Y = Y.reindex(columns=list(v3.HORIZONS))
    A = Y.to_numpy()
    rows = []
    for j, h in enumerate(list(v3.HORIZONS)[:-1]):
        zero_now = A[:, j] == 0
        later_pos = (np.nan_to_num(A[:, j + 1:]) > 0).any(axis=1)
        n = int(zero_now.sum())
        rows.append(dict(h=h, pairs_zero_at_h=n,
                         resurrect=float(later_pos[zero_now].mean()) if n else np.nan,
                         absorbing=float(1 - later_pos[zero_now].mean()) if n else np.nan))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\npairs in release windows: {len(Y):,}")
    print("`absorbing` near 1.0 means a dropped pair stays dropped and the constraint is safe;")
    print("near 0.5 means films come back often and enforcing it would destroy those rows.")

    line("Z2  first-zero onward: of pairs with a zero, share whose remaining days are all zero")
    first_zero = np.argmax(np.nan_to_num(A) == 0, axis=1)
    has_zero = (np.nan_to_num(A) == 0).any(axis=1)
    tail_all_zero = np.array([
        bool((np.nan_to_num(A[i, first_zero[i]:]) == 0).all()) if has_zero[i] else False
        for i in range(len(A))])
    print(f"pairs with at least one zero in D4-D10 : {int(has_zero.sum()):,} "
          f"({has_zero.mean():.1%})")
    print(f"  of those, every day from the first zero onward is zero: "
          f"{tail_all_zero[has_zero].mean():.1%}")

    line("Z3  does v4 already violate the pattern? (zero then positive in its own predictions)")
    frames, folds = {}, {k: v3.make_folds(data[vp], k) for k in FOLD_KINDS}
    for kind in FOLD_KINDS:
        _c = {n: CACHE_DIR / f"oofF_{n}_{kind}.pkl" for n in ("A", "B", "p0")}
        if not all(p.exists() for p in _c.values()):
            print(f"  [{kind}] missing {[p.name for p in _c.values() if not p.exists()]}"
                  f" -> run CELL 14 first")
            continue
        O = {n: pd.read_pickle(p) for n, p in _c.items()}
        idx = O["A"].index
        fr = pd.DataFrame({"y": data.y.loc[idx], "A": O["A"].clip(lower=0),
                           "B": O["B"].clip(lower=0), "p0": O["p0"].loc[idx],
                           "h": data.h.loc[idx], "sb": sb.loc[idx],
                           "mv": data.movie_title.loc[idx],
                           "ci": data.cinema_ids.loc[idx].astype(str)})
        fr["w"] = w_eval[fr.sb.to_numpy()]
        frames[kind] = fr
    if not frames:
        print("no cached OOF -> nothing further to do in this cell")
    else:
        for kind, fr in frames.items():
            post = P.fit_post_weighted(fr, v3, v4, weighted=True)
            fr["r"] = np.clip(P.apply_post(fr, post, v4), 0, None)
            Rp = fr.pivot_table(index=["mv", "ci"], columns="h", values="r")
            Rp = Rp.reindex(columns=list(v3.HORIZONS)).to_numpy()
            iszero = np.nan_to_num(Rp) <= 0
            viol = (iszero[:, :-1] & (np.nan_to_num(Rp[:, 1:]) > 0)).any(axis=1)
            print(f"  [{kind}] pairs whose predictions go zero then positive again: "
                  f"{int(viol.sum()):,} of {len(Rp):,} ({viol.mean():.1%})")

        line("Z4  fit the absorbing threshold tau, cross-fitted, on the WEIGHTED objective")
        TAU = np.round(np.concatenate([[0.0], np.arange(0.02, 0.405, 0.02)]), 3)

        def absorb(mat, tau):
            # once the prediction is below tau at some horizon, zero that one and all later
            below = np.nan_to_num(mat, nan=np.inf) < tau
            return np.where(np.logical_or.accumulate(below, axis=1), 0.0, mat)

        for kind, fr in frames.items():
            cols = list(v3.HORIZONS)
            Yp = fr.pivot_table(index=["mv", "ci"], columns="h", values="y")
            Rp = fr.pivot_table(index=["mv", "ci"], columns="h", values="r")
            Wp = fr.pivot_table(index=["mv", "ci"], columns="h", values="w")
            # align all three on one index/column order - never assume pivots agree
            _ix = Yp.index.union(Rp.index).union(Wp.index)
            Yp, Rp, Wp = (x.reindex(index=_ix, columns=cols) for x in (Yp, Rp, Wp))
            mv_of_row = Yp.index.get_level_values(0).to_numpy()
            ok = ~np.isnan(Rp.to_numpy()) & ~np.isnan(Yp.to_numpy())
            Ya, Ra, Wa = Yp.to_numpy(), Rp.to_numpy(), np.nan_to_num(Wp.to_numpy())

            def wmae(mask, mat):
                m = ok & mask
                return float((Wa[m] * np.abs(Ya[m] - mat[m])).sum()), float(Wa[m].sum())

            base_n = base_d = corr_n = corr_d = 0.0
            picks = []
            for vm, _ in folds[kind]:
                in_val = np.isin(mv_of_row, list(vm))
                tr_mask = np.repeat(~in_val[:, None], len(cols), axis=1)
                va_mask = np.repeat(in_val[:, None], len(cols), axis=1)
                if not (ok & tr_mask).any() or not (ok & va_mask).any():
                    continue
                best, bt = None, 0.0
                for t in TAU:
                    n_, d_ = wmae(tr_mask, absorb(Ra, t))
                    v = n_ / d_ if d_ else np.inf
                    if best is None or v < best:
                        best, bt = v, t
                picks.append(bt)
                n_, d_ = wmae(va_mask, Ra)
                base_n += n_
                base_d += d_
                n_, d_ = wmae(va_mask, absorb(Ra, bt))
                corr_n += n_
                corr_d += d_
            print(f"  [{kind}] tau chosen per fold {picks} | WEIGHTED "
                  f"{base_n / base_d:.4f} -> {corr_n / corr_d:.4f}"
                  f"   gain {(base_n / base_d) - (corr_n / corr_d):+.4f}")
        print("\nA gain on BOTH schemes means ship it; a split verdict means drop it.")
        print("tau = 0.0 chosen means the data prefers no absorption at all.")
print(f"\n[CELL 15] {time.time() - t0:.1f}s")
""")

md(r"""
## CELL 16 — monotone median recalibration, and the `total_show` oracle

Two independent ideas.

**R1 — recalibration.** v4's `H_GRID` step is one scalar per horizon. That is the crudest possible
calibration. A **monotone median curve** is strictly more expressive and still exactly right for
MAE: bin predictions by weighted quantile, take each bin's weighted *median* of the truth (the
MAE-optimal constant for that bin), force the curve non-decreasing, then interpolate. Because it
is monotone it preserves v4's within-bin ranking, and because it uses medians it cannot repeat the
mean-matching mistake. Fitted on four folds, scored on the fifth, weighted objective.

**R2 — the `total_show` oracle.** `tickets = shows x tickets-per-show`. Showtimes are a cinema
*scheduling decision* with strong persistence, plausibly far more predictable than demand, and
`total_show` is in `train.csv` but v4 only uses it for D1-D3 (`shw1..shw3`). R2 asks what we would
score if future showtimes were known, holding tickets-per-show at its D3 level. R3 then asks how
much of that survives when the showtime path is *predicted* instead of known -- the same
total-vs-shape question that killed the two-stage model, so expect R3 to be far below R2.

Both reuse CELL 14's cached OOF. Minutes, not hours.
""")
code(r"""
# CELL 16 — MONOTONE MEDIAN RECALIBRATION + total_show ORACLE
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

if not RUN_HEAVY:
    print("RUN_HEAVY is False -> skipping. Set RUN_HEAVY = True in CELL 1.")
else:
    import importlib, v5_pipeline
    P = importlib.reload(v5_pipeline)  # never trust Jupyter's cached copy
    v3, v4 = P.load_modules(".", verbose=False)
    data, test = P.get_tables(v3, v4, verbose=False)
    vp = P.release_mask(data)
    w_eval, EW, sb = P.scale_weights(data, test, verbose=False)
    folds = {k: v3.make_folds(data[vp], k) for k in FOLD_KINDS}

    def wmedian(v, w):
        if len(v) == 0:
            return np.nan
        o = np.argsort(v)
        v, w = np.asarray(v)[o], np.asarray(w)[o]
        c = np.cumsum(w)
        if c[-1] <= 0:
            return float(np.median(v))
        return float(v[np.searchsorted(c, 0.5 * c[-1])])

    def fit_curve(r, y, w, nbins=40, min_n=80):
        # monotone median calibration curve: returns (x knots, y knots) for np.interp
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
        xs, ys = np.array(xs), np.maximum.accumulate(np.array(ys))   # enforce non-decreasing
        return xs, ys

    frames = {}
    for kind in FOLD_KINDS:
        _c = {n: CACHE_DIR / f"oofF_{n}_{kind}.pkl" for n in ("A", "B", "p0")}
        if not all(p.exists() for p in _c.values()):
            print(f"[{kind}] missing cached OOF -> run CELL 14 first")
            continue
        O = {n: pd.read_pickle(p) for n, p in _c.items()}
        idx = O["A"].index
        fr = pd.DataFrame({"y": data.y.loc[idx], "A": O["A"].clip(lower=0),
                           "B": O["B"].clip(lower=0), "p0": O["p0"].loc[idx],
                           "h": data.h.loc[idx], "sb": sb.loc[idx],
                           "mv": data.movie_title.loc[idx]})
        fr["w"] = w_eval[fr.sb.to_numpy()]
        post = P.fit_post_weighted(fr, v3, v4, weighted=True)
        fr["r"] = np.clip(P.apply_post(fr, post, v4), 0, None)
        frames[kind] = fr

    if frames:
        line("R1  monotone median recalibration on top of v4's post-processing")
        for kind, fr in frames.items():
            res = {}
            for tag in ("global", "per-horizon"):
                bn = bd = cn = cd = 0.0
                for vm, _ in folds[kind]:
                    m = fr.mv.isin(vm)
                    va, tr = fr[m], fr[~m]
                    if not len(va) or not len(tr):
                        continue
                    out = va.r.to_numpy().copy()
                    if tag == "global":
                        xs, ys = fit_curve(tr.r.to_numpy(), tr.y.to_numpy(), tr.w.to_numpy())
                        out = np.interp(va.r.to_numpy(), xs, ys)
                    else:
                        for hh in v3.HORIZONS:
                            mt, mv2 = tr.h.to_numpy() == hh, va.h.to_numpy() == hh
                            if mt.sum() < 400 or not mv2.any():
                                continue
                            xs, ys = fit_curve(tr.r.to_numpy()[mt], tr.y.to_numpy()[mt],
                                               tr.w.to_numpy()[mt], nbins=20)
                            out[mv2] = np.interp(va.r.to_numpy()[mv2], xs, ys)
                    yv, wv = va.y.to_numpy(), va.w.to_numpy()
                    bn += (wv * np.abs(yv - va.r.to_numpy())).sum()
                    bd += wv.sum()
                    cn += (wv * np.abs(yv - np.clip(out, 0, None))).sum()
                    cd += wv.sum()
                res[tag] = (bn / bd, cn / cd)
            for tag, (b, c) in res.items():
                print(f"  [{kind}] {tag:12s} WEIGHTED {b:.4f} -> {c:.4f}   gain {b - c:+.4f}")

    line("R2  oracle: what if future total_show were known?")
    _tr = pd.read_csv(Path(v3.DATA) / "train.csv", parse_dates=["date_show"])
    _shows = _tr.groupby(["movie_title", "cinema_ids", "date_show"], as_index=False).agg(
        shw_future=("total_show", "sum"), tix_future=("total_ticket", "sum"))
    s = data[vp].copy()
    s["_mt"] = s.movie_title.astype(str)
    s["_ci"] = s.cinema_ids.astype(str)
    _shows["_mt"] = _shows.movie_title.astype(str)
    _shows["_ci"] = _shows.cinema_ids.astype(str)
    s = s.merge(_shows[["_mt", "_ci", "date_show", "shw_future"]],
                on=["_mt", "_ci", "date_show"], how="left")
    s["shw_future"] = s.shw_future.fillna(0.0)
    _tps3 = np.where(s.shw3.to_numpy() > 0, s.t3.to_numpy() / np.maximum(s.shw3.to_numpy(), 1e-9),
                     0.0)
    s["oracle_shows"] = s.shw_future.to_numpy() * _tps3 / s.scale.to_numpy()
    _w = w_eval[pd.cut(s.scale, P.SCALE_EDGES, labels=False).fillna(7).astype(int).to_numpy()]
    _ix = data[vp].index
    for kind, fr in frames.items():
        base = float((fr.w * np.abs(fr.y - fr.r)).sum() / fr.w.sum())
        print(f"  [{kind}] v4 post-processed            WEIGHTED {base:.4f}")
        break
    _e = np.abs(s.y.to_numpy() - s.oracle_shows.to_numpy())
    print(f"  oracle: known shows x D3 tickets-per-show  WEIGHTED "
          f"{float((_w * _e).sum() / _w.sum()):.4f}")
    print(f"  (coverage: rows whose future date is in train.csv "
          f"{float((s.shw_future > 0).mean()):.1%}; a zero means the pair did not screen)")

    line("R3  how much survives if the showtime path is PREDICTED, not known?")
    s["shw_ratio"] = s.shw_future.to_numpy() / np.maximum(s.shw3.to_numpy(), 1e-9)
    s["mv"] = s.movie_title
    for kind in frames:
        bn = bd = 0.0
        for vm, _ in folds[kind]:
            m = s.mv.isin(vm)
            va, tr = s[m], s[~m]
            if not len(va) or not len(tr):
                continue
            tab = tr.groupby(["h", "d1_dow"], observed=True).shw_ratio.median()
            glob = tr.groupby("h", observed=True).shw_ratio.median()
            pred_ratio = np.array([tab.get((a, b), glob.get(a, 1.0)) for a, b in
                                   zip(va.h, va.d1_dow)])
            pr = pred_ratio * va.shw3.to_numpy() * (
                np.where(va.shw3.to_numpy() > 0,
                         va.t3.to_numpy() / np.maximum(va.shw3.to_numpy(), 1e-9), 0.0)
            ) / va.scale.to_numpy()
            wv = w_eval[pd.cut(va.scale, P.SCALE_EDGES, labels=False).fillna(7)
                        .astype(int).to_numpy()]
            bn += (wv * np.abs(va.y.to_numpy() - np.clip(pr, 0, None))).sum()
            bd += wv.sum()
        print(f"  [{kind}] predicted-showtime path alone   WEIGHTED {bn / bd:.4f}")
    print("\nIf R2 is far below v4 but R3 is far above it, showtimes are informative yet")
    print("unpredictable - the same verdict as the two-stage pair-total model. If R3 lands")
    print("NEAR v4, a showtime sub-model is worth building as a feature.")
print(f"\n[CELL 16] {time.time() - t0:.1f}s")
""")

md(r"""
## CELL 17 — the p0 classifier, and what the showtime oracle really measures

CELL 16's R2 looked like the best remaining lever (-0.041 from knowing future showtimes), but it
is **confounded**: 31.4% of rows have `shw_future = 0`, and the oracle predicts exactly 0 there.
So an unknown share of that gain is simply *knowing which rows are zero* — Oracle B in disguise,
and model Z already proved chasing zero-knowledge costs 0.05-0.07 on the leaderboard. **P1**
splits the oracle into its zero-state part and its level part. If it is mostly the former, the
showtime idea is dead and we stop.

The rest targets the component doing the most work. Your W3 output fitted zero multipliers of
`[1.05, 1.1, 1.2, 1.2, 0.95, 0.4, 0.0, 0.0, 0.0, 0.0]`, i.e. the top four `p0` deciles are zeroed
outright. That classifier decides roughly 29% of the submission, yet it is trained with the
*regressor's* featset and paramset and early-stops at ~116 rounds. It has never been tuned.

- **P2** tunes it across v4's paramsets, scoring the **post-processed weighted** objective, which
  is what actually matters, rather than AUC.
- **P3** replaces the ten decile multipliers with one fitted threshold (`predict 0 iff p0 > t`).
  R1 just showed this pipeline punishes over-parameterisation, so fewer knobs may generalise
  better.
- **P4** checks whether the final model under-trains. `train_full` uses the CV mean
  `best_iteration` (332 for A) but trains on ~25% more rows than any fold, so the optimum should
  be higher. Rather than apply the usual 1.25x rule blindly, P4 measures how `best_iteration`
  grows with training-set size and extrapolates.

P2 retrains classifiers, so budget ~10 min. P1, P3 and P4 reuse CELL 14's caches.
""")
code(r"""
# CELL 17 — p0 CLASSIFIER TUNING + ORACLE DECOMPOSITION
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

if not RUN_HEAVY:
    print("RUN_HEAVY is False -> skipping. Set RUN_HEAVY = True in CELL 1.")
else:
    import importlib, v5_pipeline
    P = importlib.reload(v5_pipeline)  # never trust Jupyter's cached copy
    v3, v4 = P.load_modules(".", verbose=False)
    data, test = P.get_tables(v3, v4, verbose=False)
    vp = P.release_mask(data)
    w_eval, EW, sb = P.scale_weights(data, test, verbose=False)
    folds = {k: v3.make_folds(data[vp], k) for k in FOLD_KINDS}
    feats = v4.FEATSETS[V5_FS]

    def load_frame(kind, p0_name="p0"):
        need = {"A": f"oofF_A_{kind}.pkl", "B": f"oofF_B_{kind}.pkl",
                "p0": f"oofF_{p0_name}_{kind}.pkl"}
        if not all((CACHE_DIR / f).exists() for f in need.values()):
            return None
        O = {n: pd.read_pickle(CACHE_DIR / f) for n, f in need.items()}
        idx = O["A"].index
        fr = pd.DataFrame({"y": data.y.loc[idx], "A": O["A"].clip(lower=0),
                           "B": O["B"].clip(lower=0), "p0": O["p0"].loc[idx],
                           "h": data.h.loc[idx], "sb": sb.loc[idx],
                           "mv": data.movie_title.loc[idx], "scale": data.scale.loc[idx]})
        fr["w"] = w_eval[fr.sb.to_numpy()]
        return fr

    def cf_post(fr, kind, weighted=True):
        # cross-fitted weighted MASE of v4's post-processing recipe
        num = den = 0.0
        for vm, _ in folds[kind]:
            m = fr.mv.isin(vm)
            va, tr = fr[m], fr[~m]
            if not len(va) or not len(tr):
                continue
            r = P.apply_post(va, P.fit_post_weighted(tr, v3, v4, weighted), v4)
            e = np.abs(va.y.to_numpy() - np.clip(r, 0, None))
            num += (va.w.to_numpy() * e).sum()
            den += va.w.to_numpy().sum()
        return num / den

    base = {}
    for kind in FOLD_KINDS:
        fr = load_frame(kind)
        if fr is None:
            print(f"[{kind}] missing cached OOF -> run CELL 14 first")
            continue
        base[kind] = (fr, cf_post(fr, kind))
        print(f"  [{kind}] baseline post-processed WEIGHTED {base[kind][1]:.4f}")
    if not base:
        print("nothing cached; stop here")
    else:
        # ---------------------------------------------- P1 decompose the oracle
        line("P1  the showtime oracle: how much is just knowing the zero state?")
        _tr = pd.read_csv(Path(v3.DATA) / "train.csv", parse_dates=["date_show"])
        _sh = _tr.groupby(["movie_title", "cinema_ids", "date_show"], as_index=False).agg(
            shw_future=("total_show", "sum"))
        _sh["_mt"] = _sh.movie_title.astype(str)
        _sh["_ci"] = _sh.cinema_ids.astype(str)
        kind0 = next(iter(base))
        fr0 = base[kind0][0]
        s = data.loc[fr0.index].copy()
        s["_mt"] = s.movie_title.astype(str)
        s["_ci"] = s.cinema_ids.astype(str)
        s = s.merge(_sh[["_mt", "_ci", "date_show", "shw_future"]],
                    on=["_mt", "_ci", "date_show"], how="left")
        s["shw_future"] = s.shw_future.fillna(0.0)
        s.index = fr0.index
        # v4's own post-processed prediction, cross-fitted, as the reference level
        rv4 = np.full(len(fr0), np.nan)
        pos = {ix: i for i, ix in enumerate(fr0.index)}
        for vm, _ in folds[kind0]:
            m = fr0.mv.isin(vm)
            va, tr = fr0[m], fr0[~m]
            if not len(va) or not len(tr):
                continue
            r = np.clip(P.apply_post(va, P.fit_post_weighted(tr, v3, v4, True), v4), 0, None)
            for ix, val in zip(va.index, r):
                rv4[pos[ix]] = val
        ok = ~np.isnan(rv4)
        y = fr0.y.to_numpy()
        w = fr0.w.to_numpy()
        tps3 = np.where(s.shw3.to_numpy() > 0,
                        s.t3.to_numpy() / np.maximum(s.shw3.to_numpy(), 1e-9), 0.0)
        lvl = s.shw_future.to_numpy() * tps3 / s.scale.to_numpy()
        screens = s.shw_future.to_numpy() > 0
        truly_zero = y == 0

        def wm(pred):
            m = ok
            return float((w[m] * np.abs(y[m] - pred[m])).sum() / w[m].sum())

        variants = {
            "v4 post-processed (reference)": rv4,
            "ORACLE full: shows x D3 tps": lvl,
            "ORACLE zero-state only: v4 level, zeroed where it did not screen":
                np.where(screens, rv4, 0.0),
            "ORACLE level only: oracle where it screened, v4 elsewhere":
                np.where(screens, lvl, rv4),
            "ORACLE B (knows y == 0 exactly), v4 level elsewhere":
                np.where(truly_zero, 0.0, rv4),
        }
        for nm, pr in variants.items():
            print(f"  {nm:62s} {wm(np.asarray(pr, dtype=float)):.4f}")
        print("\n  Compare 'zero-state only' with 'level only'. If zero-state captures most of")
        print("  the full oracle's gain, the showtime idea is really the zero oracle, which")
        print("  model Z already proved is unsafe under time shift. If 'level only' carries it,")
        print("  a showtime sub-model is worth building.")

        # -------------------------------------------- P2 tune the p0 classifier
        line("P2  is the p0 classifier under-tuned? (scored on the post-processed objective)")
        P0_CFGS = {"P2 (current)": v4.PARAMSETS["P2"], "P1 shallow": v4.PARAMSETS["P1"],
                   "P3 regularised": v4.PARAMSETS["P3"], "P4 feat-frac": v4.PARAMSETS["P4"],
                   "P0 v3 default": v4.PARAMSETS["P0"]}
        for kind in base:
            print(f"  [{kind}]")
            for nm, pp in P0_CFGS.items():
                tag = nm.split()[0]
                _c = CACHE_DIR / f"oofF_p0{tag}_{kind}.pkl"
                if _c.exists():
                    o = pd.read_pickle(_c)
                else:
                    _t = time.time()
                    o, _ = v4.run_cv(data, feats, folds[kind], pp, w_open=V4_WOPEN, kind="clf")
                    pd.to_pickle(o, _c)
                    print(f"    (trained {nm} in {time.time() - _t:.0f}s)", flush=True)
                fr = base[kind][0].copy()
                fr["p0"] = o.loc[fr.index].to_numpy()
                sc = cf_post(fr, kind)
                print(f"    {nm:16s} WEIGHTED {sc:.4f}   "
                      f"{'*** better' if sc < base[kind][1] - 1e-6 else ''}")

        # ------------------------- P3 one threshold instead of ten multipliers
        line("P3  single zero threshold vs v4's ten decile multipliers")
        TH = np.round(np.arange(0.30, 0.901, 0.02), 2)
        for kind in base:
            fr, b = base[kind]
            num = den = 0.0
            picks = []
            for vm, _ in folds[kind]:
                m = fr.mv.isin(vm)
                va, tr = fr[m], fr[~m]
                if not len(va) or not len(tr):
                    continue
                # blend + horizon multipliers come from the usual fit; only the zero step changes
                pt = P.fit_post_weighted(tr, v3, v4, True)
                pt_nozero = {"w": pt["w"], "zm": np.ones(10), "hm": pt["hm"]}
                rt = np.clip(P.apply_post(tr, pt_nozero, v4), 0, None)
                best, bt = None, 0.5
                for t in TH:
                    cand = np.where(tr.p0.to_numpy() > t, 0.0, rt)
                    v = float((tr.w.to_numpy() * np.abs(tr.y.to_numpy() - cand)).sum())
                    if best is None or v < best:
                        best, bt = v, t
                picks.append(bt)
                rv = np.clip(P.apply_post(va, pt_nozero, v4), 0, None)
                cand = np.where(va.p0.to_numpy() > bt, 0.0, rv)
                num += (va.w.to_numpy() * np.abs(va.y.to_numpy() - cand)).sum()
                den += va.w.to_numpy().sum()
            print(f"  [{kind}] thresholds {picks} | WEIGHTED {num / den:.4f} vs "
                  f"v4 deciles {b:.4f}   gain {b - num / den:+.4f}")

        # --------------------- P4 does the final model train for enough rounds?
        line("P4  best_iteration vs training-set size - does train_full under-train?")
        print("  v4 passes the CV mean best_iteration straight to train_full, but train_full")
        print("  fits ~25% more rows than any single fold, so the optimum should be higher.")
        pool = data[vp]
        for nfold in (3, 5, 10):
            _c = OUT_DIR / f"iters_sizecurve_{nfold}.json"
            if _c.exists():
                mi = _json.loads(_c.read_text())
            else:
                from sklearn.model_selection import GroupKFold
                mvs = pool.drop_duplicates("movie_title")[["movie_title"]]
                fl = [(set(mvs.movie_title.iloc[b]), None) for _, b in
                      GroupKFold(nfold).split(mvs, groups=mvs.movie_title)]
                _, it = v4.run_cv(data, feats, fl, v4.PARAMSETS[V4_PS], w_open=V4_WOPEN)
                mi = [int(np.mean(it)), float(1 - 1 / nfold)]
                _c.write_text(_json.dumps(mi))
            print(f"    {nfold:2d}-fold: trains on {mi[1]:.0%} of movies -> "
                  f"mean best_iteration {mi[0]}", flush=True)
        print("\n  If best_iteration rises with training size, scale the final round count by")
        print("  the same trend extrapolated to 100% instead of using the 5-fold mean.")
print(f"\n[CELL 17] {time.time() - t0:.1f}s")
""")

md(r"""
## CELL 18 — blend an external submission

The one source of error decorrelation never tested: a **separately built pipeline**. CELL 13's
0.972 error correlation was measured across model *classes sharing one feature set*, which is a
much weaker form of diversity than two people building different pipelines.

There are no test labels, so this cannot be validated offline — it is an LB probe. What the cell
*can* do is report whether the two submissions disagree enough for a blend to be worth a
submission at all: if they agree on almost every row, skip it.

For MAE with two predictors the average is the natural combiner (the median of two points is
their mean). Set `BLEND_FILES` to the submissions you want to combine.
""")
code(r"""
# CELL 18 — BLEND EXTERNAL SUBMISSIONS (writes submission_v5b.csv)
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

# edit this: the files to blend, and the weight on the FIRST one
BLEND_FILES = ["submission_v5a.csv", "submission_v4_sam.csv"]
BLEND_W = 0.5

_have = [f for f in BLEND_FILES if Path(f).exists()]
if len(_have) < 2:
    print(f"need at least two submissions; found {_have}")
    print(f"missing: {[f for f in BLEND_FILES if f not in _have]}")
    print("Put the other team member's submission beside the notebook and set BLEND_FILES.")
else:
    subs = {f: pd.read_csv(f) for f in _have}
    ids = None
    for f, d in subs.items():
        assert {"id", "total_ticket"} <= set(d.columns), f"{f} is not a submission csv"
        if ids is None:
            ids = d.id
        assert d.id.equals(ids), f"{f} has a different id order"
    line("B1  how much do they actually disagree?")
    import itertools
    for a, b in itertools.combinations(_have, 2):
        x, y = subs[a].total_ticket.to_numpy(), subs[b].total_ticket.to_numpy()
        diff = x != y
        rel = np.abs(x - y) / np.maximum((x + y) / 2, 1)
        print(f"  {Path(a).name} vs {Path(b).name}")
        print(f"    rows differing {diff.mean():6.1%} | median relative gap on those "
              f"{np.median(rel[diff]) if diff.any() else 0:.3f}")
        print(f"    totals {x.sum():,} vs {y.sum():,} | zeros {(x == 0).mean():.1%} vs "
              f"{(y == 0).mean():.1%} | corr {np.corrcoef(x, y)[0, 1]:.4f}")
    print("\n  Rows differing below ~20% means the blend cannot move the score much; skip it.")

    line("B2  write the blend")
    f0, f1 = _have[0], _have[1]
    x, y = subs[f0].total_ticket.to_numpy(float), subs[f1].total_ticket.to_numpy(float)
    blend = np.floor(BLEND_W * x + (1 - BLEND_W) * y + 0.5).astype(int)
    out = pd.DataFrame({"id": ids, "total_ticket": np.clip(blend, 0, None)})
    out.to_csv("submission_v5b.csv", index=False)
    print(f"  wrote submission_v5b.csv: {BLEND_W:.2f} x {Path(f0).name} + "
          f"{1 - BLEND_W:.2f} x {Path(f1).name}")
    print(f"  mean {out.total_ticket.mean():.1f} tickets | zeros "
          f"{(out.total_ticket == 0).mean():.1%} | total {out.total_ticket.sum():,}")
    print(f"  differs from {Path(f0).name} on "
          f"{(out.total_ticket.to_numpy() != x).mean():.1%} of rows")
    print("\n  This is an LB probe: there are no labels to validate it offline. If the blend")
    print("  beats both inputs, the two pipelines are genuinely decorrelated and a weighted")
    print("  blend is worth one more probe to tune BLEND_W.")
print(f"\n[CELL 18] {time.time() - t0:.1f}s")
""")

md(r"""
## CELL 19 — re-select Sam's v5 under the weighted metric

Reading `cinema_forecasting_v5.py` corrected two of my assumptions and opened the best
opportunity so far.

**What I had wrong.** `FEAT_F = v4.FEATSETS['F']  # v4's final feature set` — v4 already used
featset **F**, not H. My `V4_FS = "H"` reconstruction was wrong, so round 6's "featset F beats H"
was me fixing my own error, not improving on v4.

**What v5 adds that my `submission_v5a.csv` lacks:** the **Nyepi fix** (`NOT_BOOSTED =
['2026-03-19']` — `holidays.csv` calls it a public holiday, but Balinese Nyepi closes cinemas, so
boosting it was backwards), incumbent features built from `train.csv` **and**
`test_history.csv` together, and three new feature bundles (S shape, P incumbents, R long
weekends) plus extra training windows and a lower learning rate.

**What v5 dropped that my `submission_v5a.csv` has:** v5's `final()` is *five seeds of model A and
nothing else* — no A/B blend, no `p0` zero classifier, no horizon multipliers. It threw away
v4's entire post-processing stack, which is worth about 0.005 on the weighted proxy.

That explains the scoreboard: `submission_v5.csv` 0.43255 is raw-model-plus-Nyepi, and
`submission_v5a.csv` 0.43072 is post-processing-without-Nyepi. **They are complementary.**

And critically, `experiments()` selects on `mean = (group + time) / 2` of **plain** MASE. We know
plain CV under-weights the small-scale rows that are 32.8% of the test set, so Sam's bundle
choices were made on the wrong objective. This cell re-runs that selection on the **weighted**
metric and reports both, so we can see whether the choice actually changes.

Every configuration is cached to `results/`, so the cell is resumable — stop and restart it
freely. Budget 1-2 hours for a cold run, and note the table is larger than v4's (27 offsets
instead of 18), so expect roughly 800k rows.
""")
code(r"""
# CELL 19 — v5 BUNDLE SELECTION ON THE WEIGHTED METRIC (heavy, resumable)
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

if not RUN_HEAVY:
    print("RUN_HEAVY is False -> skipping. Set RUN_HEAVY = True in CELL 1.")
else:
    import lightgbm as lgb
    import importlib, v5_pipeline
    P = importlib.reload(v5_pipeline)  # never trust Jupyter's cached copy
    v3, v4, v5 = P.load_modules_v5(".")
    data, test = P.get_tables_v5(v3, v4, v5)
    vp = P.release_mask(data)
    w_eval, EW, sb = P.scale_weights(data, test, verbose=False)
    folds = {k: v3.make_folds(data[vp], k) for k in FOLD_KINDS}
    BASE_ROWS = data.window.isin(v3.ALL_OFFSETS)
    ALL_ROWS = pd.Series(True, index=data.index)
    print(f"  clean rows: v4 windows {int((BASE_ROWS & ~data.bad).sum()):,} | "
          f"with extra windows {int((~data.bad).sum()):,}")

    line("V0  sanity: are the new v5 features comparable on train vs test?")
    _pool = data[vp]
    for c in v5.SHAPE + v5.PREV + v5.RUNS:
        print(f"  {c:12s} train med {_pool[c].median():9.3f} (NaN {_pool[c].isna().mean():.2f})"
              f" | test med {test[c].median():9.3f} (NaN {test[c].isna().mean():.2f})")

    CATS = [c for c in v3.CAT_COLS]

    def rcv(feats, params, rows, w_open, kind, folds_k):
        # v5's run_cv plus v4's opening-window weighting and classifier mode
        cats = [c for c in CATS if c in feats]
        p = {**v3.PARAMS, **params}
        if kind == "clf":
            p.update(objective="binary", metric="binary_logloss")
        tgt = (data.y == 0).astype(int) if kind == "clf" else data.y
        wt = np.where(data.window.isin(v3.OPEN_OFFSETS), w_open, 1.0)
        w_end = data.d1 + pd.Timedelta(days=9)
        oof, iters = pd.Series(np.nan, index=data.index), []
        for vm, purge in folds_k:
            in_val = data.movie_title.isin(vm)
            trm = rows & ~data.bad & ~in_val
            if purge is not None:
                trm &= ~((data.d1 <= purge[1]) & (w_end >= purge[0]))
            vam = vp & in_val
            dtr = lgb.Dataset(data.loc[trm, feats], tgt[trm], weight=wt[trm.to_numpy()],
                              categorical_feature=cats)
            dva = lgb.Dataset(data.loc[vam, feats], tgt[vam], categorical_feature=cats,
                              reference=dtr)
            m = lgb.train(p, dtr, 6000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            iters.append(m.best_iteration)
            oof[vam] = m.predict(data.loc[vam, feats], num_iteration=m.best_iteration)
        return oof.dropna(), iters

    def score(oof):
        s = data.loc[oof.index]
        e = np.abs(s.y - oof.clip(lower=0))
        w = EW[data.index.get_indexer(oof.index)]
        return float(e.mean()), float((e * w).sum() / w.sum())

    LOG_PATH = OUT_DIR / "v5_weighted_log.json"
    LOG = _json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else {}

    def sig(feats, params, rows, w_open):
        # cache key over the FULL configuration. Keying on the display name alone let a
        # re-run with a different w_open silently return the previous w_open's result.
        import hashlib
        raw = _json.dumps([sorted(feats), sorted(params.items()),
                           "all" if rows is ALL_ROWS else "base", w_open], default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def cv(name, feats, params, rows, w_open):
        key = sig(feats, params, rows, w_open)
        if key in LOG:
            r = LOG[key]
            print(f"  {name:32s} (cached) mean plain {r['mean_plain']:.5f} | "
                  f"mean WEIGHTED {r['mean_weighted']:.5f}")
            return r
        else:
            r = {}
            for k in FOLD_KINDS:
                _t = time.time()
                o, it = rcv(feats, params, rows, w_open, "reg", folds[k])
                pl, wg = score(o)
                r[k] = dict(plain=pl, weighted=wg, iters=it)
                pd.to_pickle(o, CACHE_DIR / f"v5oof_{name.replace(' ', '_')}_{k}.pkl")
                print(f"    {name:30s} {k:5s} plain {pl:.5f} | WEIGHTED {wg:.5f} "
                      f"({time.time() - _t:.0f}s)", flush=True)
            r["mean_plain"] = np.mean([r[k]["plain"] for k in FOLD_KINDS])
            r["mean_weighted"] = np.mean([r[k]["weighted"] for k in FOLD_KINDS])
            r["feats"] = feats
            r["params"] = params
            r["rows"] = "all" if rows is ALL_ROWS else "base"
            r["w_open"] = w_open
            r["cfg_name"] = name
            LOG[key] = r
            LOG_PATH.write_text(_json.dumps(LOG, indent=1, default=float))
        print(f"  {name:32s} mean plain {r['mean_plain']:.5f} | "
              f"mean WEIGHTED {r['mean_weighted']:.5f}")
        return r

    # The round-4 rule, now actually enforced: a change ships only if it does not make
    # EITHER scheme worse, and improves at least one by more than MARGIN. Selecting on the
    # mean let `windows all` through on a 0.00004 mean gain that was really -0.0015 on group
    # (the scheme the LB offset is calibrated against) and +0.0016 on time.
    MARGIN = 0.0005

    def better(cand, base):
        d = {k: base[k]["weighted"] - cand[k]["weighted"] for k in FOLD_KINDS}   # >0 = better
        no_regress = all(v > -1e-9 for v in d.values())
        real_gain = max(d.values()) > MARGIN
        return no_regress and real_gain, d

    def verdict(name, cand, base):
        ok, d = better(cand, base)
        bits = " | ".join(f"{k} {d[k]:+.5f}" for k in FOLD_KINDS)
        print(f"    gate: {bits}  -> {'ACCEPT' if ok else 'REJECT'}"
              f"{'' if ok else '  (needs no regression on either scheme and >'
                              f'{MARGIN:.4f} on one)'}")
        return ok

    line("V1a  stage 0: pick w_open FIRST, on the baseline")
    print("  v5's run_cv passes no sample weight at all (w_open = 1.0), while v4 used 2.0.")
    print("  Testing bundles at the wrong w_open is what invalidated the first run of this")
    print("  cell, so the weighting is settled before anything else.")
    wo_runs = {wo: cv(f"v4 (F, P2) wopen{wo}", v5.FEAT_F, v5.P2, BASE_ROWS, wo)
               for wo in (1.0, 2.0, 3.0)}
    # w_open is a baseline property, not an incremental change, so pick the one that is
    # best on BOTH schemes if such a value exists, else the best on group.
    _cands = [w for w in wo_runs
              if all(wo_runs[w][k]["weighted"] <= min(wo_runs[x][k]["weighted"] for x in wo_runs)
                     + 1e-9 for k in FOLD_KINDS)]
    wo_sel = _cands[0] if _cands else min(wo_runs,
                                          key=lambda w: wo_runs[w]["group"]["weighted"])
    for w in sorted(wo_runs):
        print(f"    w_open {w}: " + " | ".join(
            f"{k} {wo_runs[w][k]['weighted']:.5f}" for k in FOLD_KINDS))
    print(f"  stage 0 choice: w_open = {wo_sel}"
          f"{' (best on both schemes)' if _cands else ' (best on group; schemes disagree)'}")

    line("V1b  stage A: each new bundle, at the chosen w_open, selected on WEIGHTED")
    best_name, best_feats = f"v4 (F, P2) wopen{wo_sel}", v5.FEAT_F
    best = wo_runs[wo_sel]
    single = {n: cv(f"F + {n} wopen{wo_sel}", v5.FEAT_F + b, v5.P2, BASE_ROWS, wo_sel)
              for n, b in v5.BUNDLES.items()}
    wins_w = []
    for n in v5.BUNDLES:
        print(f"  bundle {n}:")
        if verdict(n, single[n], best):
            wins_w.append(n)
    wins_p = [n for n in v5.BUNDLES if single[n]["mean_plain"] < best["mean_plain"]]
    print(f"\n  bundles that help on WEIGHTED: {wins_w or 'none'}")
    print(f"  bundles that help on PLAIN   : {wins_p or 'none'}"
          f"{'   <-- the two metrics disagree' if set(wins_w) != set(wins_p) else ''}")
    cands = {f"F + {n}": single[n] for n in wins_w}
    if len(wins_w) > 1:
        nm = "F + " + " + ".join(wins_w) + f" wopen{wo_sel}"
        ft = v5.FEAT_F + [c for n in wins_w for c in v5.BUNDLES[n]]
        cands[nm] = cv(nm, ft, v5.P2, BASE_ROWS, wo_sel)
    if cands:
        best_name = min(cands, key=lambda n: cands[n]["mean_weighted"])
        best = cands[best_name]
        best_feats = best["feats"]
    print(f"  stage A choice: {best_name} (WEIGHTED {best['mean_weighted']:.5f})")

    line("V2  stage B: extra training windows, at the chosen w_open")
    r = cv(best_name + " + windows", best_feats, v5.P2, ALL_ROWS, wo_sel)
    _ok = verdict("windows", r, best)
    rows_sel = ALL_ROWS if _ok else BASE_ROWS
    if _ok:
        best, best_name = r, best_name + " + windows"
    print(f"  stage B choice: windows "
          f"{'all' if rows_sel is ALL_ROWS else 'base'} (WEIGHTED {best['mean_weighted']:.5f})")

    line("V3  stage C: learning rate, at the chosen w_open and windows")
    params_sel = dict(v5.P2)
    for lr in (0.03, 0.08):
        r = cv(f"{best_name} + lr{lr}", best_feats, {**v5.P2, "learning_rate": lr},
               rows_sel, wo_sel)
        if verdict(f"lr{lr}", r, best):
            params_sel, best = {**v5.P2, "learning_rate": lr}, r
            best_name = f"{best_name} + lr{lr}"
    print(f"  final: {best_name}")
    print(f"    params {params_sel} | windows {'all' if rows_sel is ALL_ROWS else 'base'} "
          f"| w_open {wo_sel}")
    print(f"    mean plain {best['mean_plain']:.5f} | mean WEIGHTED "
          f"{best['mean_weighted']:.5f}")
    print(f"    group WEIGHTED {best['group']['weighted']:.5f} | "
          f"time WEIGHTED {best['time']['weighted']:.5f}")
    print("    (these are RAW model A. The LB estimate comes from CELL 20, after"
          " post-processing: expected LB = group post-processed - 0.0087)")

    _choice = dict(name=best_name, feats=best_feats, params=params_sel,
                   rows="all" if rows_sel is ALL_ROWS else "base", w_open=wo_sel,
                   iters_group=best["group"]["iters"], mean_weighted=best["mean_weighted"],
                   mean_plain=best["mean_plain"])
    (OUT_DIR / "v5_choice_weighted.json").write_text(_json.dumps(_choice, indent=1, default=float))
    print(f"\n  wrote results/v5_choice_weighted.json -> CELL 20 reads this")
print(f"\n[CELL 19] {time.time() - t0:.1f}s")
""")

md(r"""
## CELL 20 — the combined submission: v5's table + v4's post-processing

`submission_v5.csv` (0.43255) is raw model A plus the Nyepi fix, with no post-processing.
`submission_v5a.csv` (0.43072) is the full post-processing stack without the Nyepi fix. This cell
puts the two halves together on whatever CELL 19 selected, and adds the one post-processing
change that passed the gate in CELL 17:

- **P3's single zero threshold** (`predict 0 iff p0 > t`) instead of v4's ten per-decile
  multipliers, worth +0.0011 group / +0.0012 time;
- the blend weight and horizon multipliers fitted on the **weighted** objective;
- Nyepi, concatenated incumbents and any winning bundles, inherited from v5's tables.

Writes `submission_v5c.csv`. Expected LB is printed from the calibrated offset, but treat it as a
hypothesis: offline deltas have transferred at roughly 55%.
""")
code(r"""
# CELL 20 — submission_v5c.csv (needs CELL 19's choice)
t0 = time.time()
line = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)

if not RUN_HEAVY:
    print("RUN_HEAVY is False -> skipping. Set RUN_HEAVY = True in CELL 1.")
else:
    import lightgbm as lgb
    import importlib, v5_pipeline
    P = importlib.reload(v5_pipeline)  # never trust Jupyter's cached copy
    # FORCE_CHOICE lets you rebuild any configuration without re-running CELL 19.
    # Set it to reproduce an earlier submission, e.g. the windows="base" model that
    # scored 0.42677:
    #     FORCE_CHOICE = {"rows": "base"}
    # Any key given here overrides results/v5_choice_weighted.json.
    FORCE_CHOICE = {}

    _cp = OUT_DIR / "v5_choice_weighted.json"
    if not _cp.exists():
        print("results/v5_choice_weighted.json missing -> run CELL 19 first")
    else:
        ch = _json.loads(_cp.read_text())
        if FORCE_CHOICE:
            ch = {**ch, **FORCE_CHOICE}
            ch["cfg_label"] = str(ch.get("name", "?")) + " [FORCED " + str(FORCE_CHOICE) + "]"
            print(f"  FORCE_CHOICE applied: {FORCE_CHOICE}")
        v3, v4, v5 = P.load_modules_v5(".", verbose=False)
        data, test = P.get_tables_v5(v3, v4, v5, verbose=False)
        vp = P.release_mask(data)
        w_eval, EW, sb = P.scale_weights(data, test, verbose=False)
        folds = {k: v3.make_folds(data[vp], k) for k in FOLD_KINDS}
        feats, params = ch["feats"], ch["params"]
        rows = (pd.Series(True, index=data.index) if ch["rows"] == "all"
                else data.window.isin(v3.ALL_OFFSETS))
        wo = ch["w_open"]
        cats = [c for c in v3.CAT_COLS if c in feats]
        print(f"  chosen: {ch['name']}")
        print(f"    {len(feats)} feats | params {params} | windows {ch['rows']} | w_open {wo}")
        print(f"    mean WEIGHTED {ch['mean_weighted']:.5f}")
        # Cache files must be keyed on the CONFIGURATION, not just the model name.
        # Keyed on the name alone, re-running after CELL 19 picks a different config
        # would silently reuse the previous config's OOF and write a wrong submission.
        import hashlib
        CFG = hashlib.md5(_json.dumps(
            [sorted(feats), sorted(params.items()), ch["rows"], wo], default=str
        ).encode()).hexdigest()[:12]
        print(f"    config fingerprint {CFG} (cache files are tagged with it)")

        line("C1  OOF for A, B and the zero classifier on the chosen configuration")

        def rcv(pp, kind, folds_k):
            p = {**v3.PARAMS, **pp}
            if kind == "clf":
                p.update(objective="binary", metric="binary_logloss")
            tgt = (data.y == 0).astype(int) if kind == "clf" else data.y
            wt = np.where(data.window.isin(v3.OPEN_OFFSETS), wo, 1.0)
            w_end = data.d1 + pd.Timedelta(days=9)
            oof, iters = pd.Series(np.nan, index=data.index), []
            for vm, purge in folds_k:
                in_val = data.movie_title.isin(vm)
                trm = rows & ~data.bad & ~in_val
                if purge is not None:
                    trm &= ~((data.d1 <= purge[1]) & (w_end >= purge[0]))
                vam = vp & in_val
                dtr = lgb.Dataset(data.loc[trm, feats], tgt[trm], weight=wt[trm.to_numpy()],
                                  categorical_feature=cats)
                dva = lgb.Dataset(data.loc[vam, feats], tgt[vam], categorical_feature=cats,
                                  reference=dtr)
                m = lgb.train(p, dtr, 6000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                iters.append(m.best_iteration)
                oof[vam] = m.predict(data.loc[vam, feats], num_iteration=m.best_iteration)
            return oof.dropna(), iters

        SPEC = {"A": (params, "reg", 5), "B": (v4.PARAMS_B, "reg", 3),
                "p0": (params, "clf", 3)}
        OO, ITERS = {}, {}
        for nm, (pp, kd, _) in SPEC.items():
            OO[nm], it = {}, []
            for kind in FOLD_KINDS:
                _c = CACHE_DIR / f"v5c_{CFG}_{nm}_{kind}.pkl"
                _ci = OUT_DIR / f"v5c_iters_{CFG}_{nm}_{kind}.json"
                if _c.exists() and _ci.exists():
                    OO[nm][kind] = pd.read_pickle(_c)
                    it += _json.loads(_ci.read_text())
                else:
                    _t = time.time()
                    o, i = rcv(pp, kd, folds[kind])
                    OO[nm][kind] = o
                    pd.to_pickle(o, _c)
                    _ci.write_text(_json.dumps(i))
                    it += i
                    print(f"    {nm} [{kind}] {time.time() - _t:.0f}s", flush=True)
            ITERS[nm] = max(50, int(1.1 * np.mean(it)))   # v5 scales rounds by 1.1x
        print(f"  rounds for the full-data refit (1.1x the CV mean): {ITERS}")

        line("C2  fit post-processing: weighted objective + P3's single zero threshold")
        TH = np.round(np.arange(0.30, 0.901, 0.02), 2)

        def build(kind):
            idx = OO["A"][kind].index
            fr = pd.DataFrame({"y": data.y.loc[idx], "A": OO["A"][kind].clip(lower=0),
                               "B": OO["B"][kind].clip(lower=0), "p0": OO["p0"][kind].loc[idx],
                               "h": data.h.loc[idx], "sb": sb.loc[idx],
                               "mv": data.movie_title.loc[idx]})
            fr["w"] = w_eval[fr.sb.to_numpy()]
            return fr

        frames = {k: build(k) for k in FOLD_KINDS}

        def fit_v5c(fr):
            # blend + horizon multipliers on the weighted objective, then ONE zero threshold
            y, ww = fr.y.to_numpy(), fr.w.to_numpy()
            bf = lambda g, er: g[int(np.argmin([er(x) for x in g]))]
            a, b = fr.A.to_numpy(), fr.B.to_numpy()
            w0 = bf(v4.W_GRID, lambda w: (ww * np.abs(y - w * a - (1 - w) * b)).sum())
            r = w0 * a + (1 - w0) * b
            hm = {}
            for k in v3.HORIZONS:
                m = fr.h.to_numpy() == k
                if m.sum():
                    hm[k] = bf(v4.H_GRID, lambda g: (ww[m] * np.abs(y[m] - g * r[m])).sum())
            r = r * pd.Series(fr.h.to_numpy()).map(hm).fillna(1.0).to_numpy()
            th = bf(TH, lambda t: (ww * np.abs(
                y - np.where(fr.p0.to_numpy() > t, 0.0, r))).sum())
            return dict(w=float(w0), hm=hm, th=float(th))

        def apply_v5c(fr, pp):
            r = pp["w"] * fr.A.to_numpy() + (1 - pp["w"]) * fr.B.to_numpy()
            r = r * pd.Series(fr.h.to_numpy()).map(pp["hm"]).fillna(1.0).to_numpy()
            return np.clip(np.where(fr.p0.to_numpy() > pp["th"], 0.0, r), 0, None)

        for kind, fr in frames.items():
            raw = float((fr.w * np.abs(fr.y - fr.A)).sum() / fr.w.sum())
            num = den = 0.0
            for vm, _ in folds[kind]:
                m = fr.mv.isin(vm)
                va, tr = fr[m], fr[~m]
                if not len(va) or not len(tr):
                    continue
                e = np.abs(va.y.to_numpy() - apply_v5c(va, fit_v5c(tr)))
                num += (va.w.to_numpy() * e).sum()
                den += va.w.to_numpy().sum()
            print(f"  [{kind}] raw A {raw:.4f} -> post-processed {num / den:.4f}"
                  f"   gain {raw - num / den:+.4f}")
            if kind == "group":
                post_group = num / den          # the LB-calibrated figure
                print(f"  [group] => expected LB {post_group - 0.0087:.4f} "
                      f"(calibrated offset 0.0087 from two paired readings)")
        post = fit_v5c(pd.concat(frames.values()))
        print(f"\n  shipped: blend A {post['w']:.2f} | zero threshold p0 > {post['th']:.2f}")
        print(f"    horizon multipliers {post['hm']}")

        line("C3  train on all clean windows and write submission_v5c.csv")
        trm = rows & ~data.bad
        wt = np.where(data.window.isin(v3.OPEN_OFFSETS), wo, 1.0)[trm.to_numpy()]
        preds = {}
        for nm, (pp, kd, nseed) in SPEC.items():
            p = {**v3.PARAMS, **pp}
            if kd == "clf":
                p.update(objective="binary", metric="binary_logloss")
            tgt = (data.y == 0).astype(int) if kd == "clf" else data.y
            ds = lgb.Dataset(data.loc[trm, feats], tgt[trm], weight=wt,
                             categorical_feature=cats)
            off = {"A": 0, "B": 100, "p0": 200}[nm]
            ms = [lgb.train({**p, "seed": SEED + off + i}, ds, ITERS[nm]) for i in range(nseed)]
            preds[nm] = np.mean([m.predict(test[feats]) for m in ms], axis=0)
            print(f"    {nm}: {nseed} seeds x {ITERS[nm]} rounds", flush=True)
        tf = pd.DataFrame({"A": np.clip(preds["A"], 0, None),
                           "B": np.clip(preds["B"], 0, None),
                           "p0": preds["p0"], "h": test.h.to_numpy()})
        ratio = apply_v5c(tf, post)
        _hist = (test.t1.fillna(0) + test.t2.fillna(0) + test.t3.fillna(0)).to_numpy()
        ratio[_hist == 0] = 0
        # Write a fingerprinted file FIRST so a run can never destroy a good submission,
        # then promote it to submission_v5c.csv only if its proxy is at least as good as
        # whatever that name currently holds (tracked in results/submission_ledger.json).
        _cfg_label = ch.get("cfg_label", ch.get("name", "?"))
        _fn = f"submission_v5c_{CFG}.csv"
        v4.write_sub(_fn, test, ratio, test.scale.to_numpy())
        _ledger_p = OUT_DIR / "submission_ledger.json"
        _ledger = _json.loads(_ledger_p.read_text()) if _ledger_p.exists() else {}
        _this = float(post_group)
        _cur = _ledger.get("submission_v5c.csv", {}).get("group_post")
        _ledger[_fn] = dict(group_post=_this, config=_cfg_label, fingerprint=CFG)
        if _cur is None or _this <= _cur + 1e-9:
            import shutil as _sh
            _sh.copyfile(_fn, "submission_v5c.csv")
            _ledger["submission_v5c.csv"] = dict(group_post=_this, config=_cfg_label,
                                                 fingerprint=CFG)
            print(f"  promoted {_fn} -> submission_v5c.csv "
                  f"(group post-processed {_this:.4f}"
                  f"{f', previous {_cur:.4f}' if _cur is not None else ''})")
        else:
            print(f"  NOT promoted: this config's group post-processed {_this:.4f} is WORSE "
                  f"than the {_cur:.4f} already behind submission_v5c.csv.")
            print(f"  The better file is untouched. This run is saved as {_fn} only.")
        _ledger_p.write_text(_json.dumps(_ledger, indent=1, default=float))
        for _ref in ("submission_v5a.csv", "submission_v5.csv"):
            if Path(_ref).exists():
                _o = pd.read_csv(_ref).total_ticket.to_numpy()
                _n = pd.read_csv("submission_v5c.csv").total_ticket.to_numpy()
                print(f"  vs {_ref}: differs on {(_o != _n).mean():.1%} of rows | "
                      f"totals {_o.sum():,} -> {_n.sum():,} | zeros "
                      f"{(_o == 0).mean():.1%} -> {(_n == 0).mean():.1%}")
        print(f"\n  expected LB for THIS config: {post_group - 0.0087:.4f}")
        print("  Submit submission_v5c.csv only if the promotion line above says it was")
        print("  promoted; otherwise the previous file is better and is still in place.")
print(f"\n[CELL 20] {time.time() - t0:.1f}s")
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

### ROUND 3 — two separate lineages, and the leaderboard has already killed three ideas

**Critical: `FINDINGS.md` / `HANDOVER.md` describe a DIFFERENT lineage from `0.43715.py`.**
That branch rewrote the pipeline and never beat the baseline:

| build | proxy | LB | note |
|-------|-------|----|------|
| `0.46641.ipynb` (v1, ancestor of v4) | 0.4328 OOF | **0.46641** | |
| rewrite, no Eid guard | 0.3429 | 0.48267 | |
| rewrite, Eid guard `calendar 0.6` | 0.3429 | 0.47789 | |
| quantile snap, guard `flat 1.0`, film_curve, lags 0-12 | ~0.337 | **0.46795** | best of that branch |
| + model Z | 0.3377 | **0.52220** | |
| + Z at 100%, capped snap, lags 0-24 | **0.3348** (best proxy) | **0.53421** (worst LB) | |
| **v3/v4 `0.43715.py`** (team's line) | 0.344 / 0.354 | **0.43715** | **current best** |

So the rewrite's best is 0.03 *behind* v4, and its proxy is anti-correlated with the LB. Its
*negative* results measured on that proxy do not automatically transfer to v4 — but the three
measured on the **leaderboard itself** do, and they are decisive.

**1. The two-part zero model is dead.** 0.46795 → 0.52220 → 0.53421. Model Z improved the proxy on
every build and lost 0.05-0.07 on the LB every time. Not a threshold-transfer bug — rank-based
deciles score the same (`z_shift.py`). Forcing exact zeros is **one-sided**: a wrongly zeroed row
forfeits the entire true ratio, so the downside is unbounded while the upside is capped at the
prediction it replaced. Optimal under a matched distribution, reckless under the Apr-Sep → Oct-Mar
shift. This kills the friend's reported +0.008 as a direction for us.

**2. CELL 9's T4 says v4 has no zero-share problem at all.** On the matched release-anchored
population the zero gaps are +2.7, -0.1, +2.1, +1.7, -1.9, -1.6, -5.0 pp across D4-D10 — already
calibrated. Round 2's "30.57% predicted vs 43.5% true" was a population artifact: the 43.5% came
from a different window construction, and the 30.57% was pooled over Ramadan rows. Both the
leaderboard and the matched measurement now point the same way. **Stop pursuing zeros.**

**3. The Eid cohort is worth ~0.005, confirmed on the leaderboard.** The clean A/B (0.48267 no
guard → 0.47789 with `calendar 0.6`) is **-0.00478** on 6.08% of rows. That lands inside CELL 7's
estimate of 0.0058-0.0255 and v4 already predicts that cohort at mean ratio 1.0943 with 0.00%
zeros. Essentially spent.

**4. T4's `mean_mult` of 1.26 (D4) and 1.45 (D5) is mostly skew, not bias.** The true ordinary
D4 distribution has median 0.9146 but mean 1.0612. Matching the *mean* pushes predictions above
the median and *costs* MAE — this is the documented `+0.0163` regression in build 1 ("far horizons
shipped at the mean, not the median"). CELL 10 redoes the comparison on medians and MAE-optimal
multipliers instead. Do not act on T4 directly.

### The oracle study — the ranking that should drive effort

Each oracle gets perfect knowledge of one quantity; the gap is the ceiling on predicting it well.

| oracle | MASE | ceiling |
|--------|------|---------|
| baseline | 0.3944 | — |
| **A — the film's own national D4-D10 ratio curve** | **0.3039** | **+0.0905** |
| D — the pair's own 7-day total | 0.3214 | +0.0731 |
| B — whether each row is zero | 0.3350 | +0.0595 |
| C — the target date's national demand | 0.3867 | +0.0078 |

Oracle A uses **no pair-level information** and still beats the full per-pair baseline by 23%.
That branch banks only +0.0209 of it via a median table. It is the one lever large enough to close
our 0.093 gap, and it requires **retraining** — so `cinema_forecasting_v3.py` is the critical path.

**Oracle C's +0.0078 is contested, and it is the ceiling on CELL 8's market index.** `FINDINGS.md`
dismisses the cross-film/national-date direction on the strength of it. But the oracle was measured
on the **train proxy, which has almost no calendar variance** — the same structural objection that
document raises against its own §8, then forgets when reading its own oracle. Train anomaly rows
are 7.1% against 31.8% in test, and CELL 8 measures market ratios from 0.295 to 10.0 inside
Ramadan 1447 H, a spread train simply does not contain. An oracle handed a near-constant quantity
will always show a near-zero ceiling. Independent sizing: MASE divides by each pair's own `scale`,
so a uniform level shift is absorbed and only **straddles** bite — 13.34% of rows (CELL 5) at
market ratios of 0.56-2.41, worth roughly **0.005-0.02**, and CELL 7's S2 suggests v4 already
captures most of it (predicted Ramadan ratio 0.4116 against 0.519 x 0.811 = 0.421 expected).
Verdict: real but small, and not the 0.09 we need.

### Mandatory validation gate, adopted

> Ship a change only if it improves **both** the random-film proxy **and** a time-based holdout
> (train on the earliest 75% of films by release date, score the latest 25%).

Model Z passed the proxy and failed the holdout (0.3922 → 0.3999) before it failed the LB twice.
v1 had this holdout; the rewrite dropped it, and that single omission cost two submissions.

### Correction to a reported inference

"The same submission file gives the same score, so the split is not random" does **not** follow.
Every Kaggle public/private split is *fixed* once chosen, so re-submitting an identical file always
reproduces its score regardless of whether the 30% was drawn at random or cut by date. That test
has no power here. The question is still open, and it matters: if the split is by date, the
2026-03-18 Eid cohort sits at the very end of the test period and may be **entirely absent from the
public 30%**, which would make every Eid probe uninformative.

### ROUND 4 — v3 arrived, v4 reproduced, and every remaining lever measured out at zero

`cinema_forecasting_v3.py` is complete (all 18 symbols; `BAD_START, BAD_END` is a tuple
assignment). v4 reproduced end to end as `final(fs='H', w_open=2.0, ps='P2')`:

| scheme | reproduced MASE | reported |
|--------|-----------------|----------|
| `group` (5-fold by movie) | **0.3497** | 0.344 |
| `time` (5 purged date blocks) | **0.3584** | 0.354 |

Within ~0.005, so the reconstruction is faithful. Window table 545,153 rows, 47,761 clean release
rows, builds in ~3 s; each CV scheme runs in ~2.5 min on 8 threads.

Two practical gotchas, both handled in CELL 11: `v3.load_data()` hard-codes `DATA = 'data/'` and
the canonical six filenames (the real file is `train (1).csv`), so the files must be staged and
`v3.DATA` repointed; and `v3.encode_cats([data])` is mandatory or LightGBM rejects the four
`str`-dtype categoricals under pandas 3.

**Reading v3 retracts my round-2 thesis.** `run_cv` scores `val_pool = (data.window == 0) &
~data.bad` — v4's CV was **already release-anchored**, `make_folds(pool,'time')` already purges,
and v4 already up-weights opening windows. v4's CV is honest about the population. CELL 10's Q1
gap was therefore **confounded** — it compared *train* truth against *test* predictions, two
different film populations — and must not be acted on.

#### Measured on v4's own OOF, where the truth is known

| test | group | time | verdict |
|------|-------|------|---------|
| horizons pinned to v4's `H_GRID` floor of 0.85 | **0 of 7** | **0 of 7** | grid is *not* clipped |
| per-horizon multiplier, cross-fitted | **+0.0000** | **+0.0000** | no gain |
| per-horizon x D1-dow multiplier, cross-fitted | **-0.0055** | **-0.0148** | harmful |
| `film_curve` / stage-1 oracle A | **+0.0022** | **+0.0034** | harmful |
| sibling format variants | -0.0028 | **+0.0039** | split -> rejected |

Optimal per-horizon multipliers land at 0.88-1.05, comfortably inside v4's grid. So the clipped-
grid bug that cost the other lineage does **not** exist in v4, and CELL 10's wish for 0.40-0.68
was the population artifact, not a bias.

v4's calibration on the matched population is genuinely good — true vs predicted medians:

| h | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|----|----|----|----|----|----|-----|
| true median | 0.931 | 0.655 | 0.493 | 0.385 | 0.184 | 0.000 | 0.000 |
| v4 OOF median | 0.920 | 0.623 | 0.476 | 0.380 | 0.225 | 0.150 | 0.150 |
| true zero share | 9.6% | 14.1% | 20.2% | 28.6% | 42.6% | 50.4% | 54.6% |

and v4's shipped zero shares (CELL 7: 51.2% at D9, 57.1% at D10) track the true 50.4% / 54.6%
almost exactly, so its `ZERO_GRID` snap is already well fitted.

**`film_curve` failed for an instructive reason.** Stage 1 has real skill — OOF MAE 0.2278 against
0.3715 for a per-horizon median, 39% better. But v4's `BASE` already contains the film-level
D1-D3 aggregates (`nat_r31`, `nat_r32`, `nat_log`, `nc3`, `nat_tps3`, `share3`) that stage 1 is
built from, so `fc` is a strictly less expressive function of inputs the main model already sees.
Oracle A's +0.0905 ceiling was measured against a baseline that **lacked** those features; v4 has
already banked the accessible part. This retracts round-3's "only lever large enough".

**Sibling features show the gate earning its keep**: -0.0028 on grouped, +0.0039 on purged time.
That split is the model-Z signature, so it is rejected rather than shipped.

#### Where this leaves the gap

v4 `time` CV 0.3584 vs LB 0.43715 = **+0.079**, and nothing above explains it. Solving
`0.684 x 0.3584 + 0.316 x X = 0.43715` puts the non-ordinary 31.6% of test rows at **X = 0.608**,
1.70x the ordinary rows. The leader's 0.34456 is *below* v4's own ordinary-population CV, which is
only reachable by making the calendar-anomaly rows behave roughly like ordinary ones.

So the entire remaining gap is the calendar regime, and it is **structurally unlearnable**: train
holds no Ramadan at all and starts at Eid+1, so no training window can straddle into Eid. Any
correction there is a **prior**, and the leaderboard is the only instrument that can read it.
Which also means Oracle C's +0.0078 "ceiling" remains unusable as evidence — it was measured on a
population containing none of the phenomenon.

**One caveat against over-correcting Ramadan.** CELL 8 measures the Ramadan market ratio at 0.811
and CELL 7 shows v4 predicting a 0.4116 mean ratio there against roughly 0.519 x 0.811 = 0.421
expected — about 2% low. v4 appears to have *already* absorbed the Ramadan level, so multiplying
those 12,408 rows down would likely hurt. The regime-average comparison is confounded by horizon
mix, so it is suggestive, not proof — but it argues against Ramadan as the first probe.

### ROUND 5 — would another model class help? measured: no, and here is why

v4 is **already an ensemble**: model A (LightGBM `P2`, 5 seeds) + model B (`extra_trees`, 255
leaves, `feature_fraction` 0.5, 3 seeds) + a zero classifier, with blend weight, per-`p0`-decile
zero multipliers and per-horizon multipliers all fitted by `crossfit` over both schemes.

So the real question is whether *any* model class is decorrelated enough to add value. Measured on
v4's own folds, with a hierarchical median lookup table as a deliberately alien third class — no
feature interactions, no boosting, pure non-parametric:

| model | group MASE | time MASE | abs-error correlation with A |
|-------|-----------|-----------|------------------------------|
| A — LightGBM `P2` | 0.3497 | 0.3584 | 1.000 |
| B — LightGBM `extra_trees` | 0.3511 | 0.3604 | **0.996** |
| median table (`h` x `d1_dow` x scale bin) | 0.4315 | 0.4332 | **0.972** |

Cross-fitted blend gains:

| blend | group | time |
|-------|-------|------|
| A + B | +0.0017 | **+0.0004** |
| A + median table | +0.0000 | +0.0000 |
| A + B + median table | +0.0017 | +0.0004 |

**A completely different model class still correlates 0.972 on absolute error.** That is the whole
answer: the residual is not model-specific, so it is not epistemic error that capacity or a new
architecture could reduce. It is **aleatoric** — irreducible given these features.

The error distribution confirms it:

| rows | share of total error | their mean true `y/scale` |
|------|---------------------|---------------------------|
| worst 1% | **20.9%** | 8.11 |
| worst 5% | 40.3% | 3.42 |
| worst 10% | 52.7% | 2.31 |
| worst 25% | 73.9% | 1.35 |

A fifth of all error sits in 1% of rows whose true ratio averages **8.1x** their own D1-D3
baseline — pairs that sold almost nothing in their first three days and then exploded. Nothing in
the data predicts that: it is cinema scheduling decisions, local events and word of mouth. And
chasing the tail is forbidden anyway, because MAE's optimum is the conditional **median** and
v4's medians are already correct (round 4).

For reference, oracle per-row routing between the three classes scores 0.2769 / 0.2834 — a 0.073
"gain" that is pure hindsight and unreachable by any blend or stacker, since no feature
distinguishes which model will happen to be closer on a given row.

**Verdict on deep learning / neural networks / another supervised model: do not.** On ~259k
tabular training rows with high-cardinality categoricals and an L1 objective, a neural net would
at best match a GBDT and would land at the same ~0.97 error correlation, so the blend gain stays
in the 0.000-0.002 range. More decisively, it does not touch the actual bottleneck: **31.6% of
test rows sit in a calendar regime with zero training examples**, and no architecture learns
Ramadan from a training set containing no Ramadan.

The one strictly-safe model-side action left is **more seeds** on A and B. It reduces variance
without introducing a systematic bet, but expect ~0.001-0.003, not 0.08.

### ROUND 6 — the gap is SCALE COMPOSITION, and we finally have an honest offline LB proxy

**This supersedes the round-4 calendar conclusion.** Reweighting v4's OOF MASE to the test's
**scale** distribution predicts the leaderboard almost exactly:

| | pooled CV | reweighted to test scale mix | actual LB |
|---|---|---|---|
| group | 0.3497 | **0.4482** | **0.43715** |
| time | 0.3584 | **0.4551** | **0.43715** |

The proxy slightly over-predicts, so it is conservative. The cause:

| | train (all offsets) | train (opening) | validation | **TEST** |
|---|---|---|---|---|
| scale <= 25 | 5.6% | 7.9% | 5.4% | **17.0%** |
| scale <= 50 | 14.2% | 18.8% | 14.8% | **32.8%** |

The test set holds **3x more small / partial-activity pairs**, and MASE divides by each pair's own
`scale`, so those rows are intrinsically harder. v4's OOF MASE by bin: `2-5` -> **2.80**,
`5-10` -> **1.72**, `>250` -> 0.27. Scale <= 50 is 32.8% of test rows and **53% of all error**.

**Use the test-mix-weighted MASE as the selection metric from now on.** Every "CV gain did not
transfer" episode since v1 traces back to selecting on a validation mix that does not match the
test. This is the fix.

#### The small bins are irreducible — confirmed, not assumed

| bin | model | best constant | best constant per horizon |
|-----|-------|---------------|---------------------------|
| 2-5 | 2.8003 | 2.8546 | 2.8381 |
| 5-10 | **1.7202** | **1.7124** | 1.7078 |

At `5-10` the model is *worse than a constant*. True median is 0 with mean `|y|` of 2.85 — pairs
that barely screened in D1-D3 then exploded. `FINDINGS.md` was right that this is irreducible;
it is 21% of the weighted error and it is not recoverable.

#### Oracles, reweighted to the test mix

| | ceiling | headroom |
|---|---|---|
| oracle A — film-window curve known | 0.4397 | +0.0146 |
| **oracle D — the pair's own 7-day total known** | **0.3389** | **+0.1153** |

Oracle D keeps v4's per-day *shape* and replaces only the per-pair *level*, and lands below the
leader. So the shape is right and the level is the problem. But the level is **not predictable**:
a stage-1 model on the pair total reaches OOF MAE 2.2278 against a median target of 3.225, and
using it scores **0.4720** vs 0.4543. Oracle A's headroom is also entirely in the two largest
bins and *negative* in the small ones — which is exactly why `film_curve` hurt in round 4.

#### v4's pipeline is verified correct

- `scale` matches the competition definition on all 10,373 test pairs, **max abs diff 0**
  (`(sum of D1-D3)/3`, clipped at 1; 7 pairs clipped).
- `write_sub` is `floor(ratio * scale + 0.5)` — round half up, correct.
- `test_history.csv` cannot supply a cluster-weekday profile: it is D1-D3 only, so Monday reads
  0.02 and Tuesday is `NaN`. Structurally biased by the release calendar. Do not try it.

#### Full experiment ledger (time scheme; WEIGHTED is the honest proxy)

| experiment | plain | weighted | verdict |
|---|---|---|---|
| v4 baseline `H/P2/2.0` | 0.3577 | 0.4543 | — |
| **featset `F` instead of `H`** | **0.3549** | **0.4526** | **KEEP, passes both schemes** |
| **post-processing fitted on the weighted metric** | 0.3563 | **0.4499** | **KEEP, passes both schemes** |
| `F/P4`, `H/P2/w_open=1` | ~0.356 | 0.4537 / 0.4546 | marginal |
| featsets `E`, `G`; `w_open=3` | 0.362-0.366 | 0.459-0.467 | worse |
| per-horizon separate models (v1's model H) | 0.3577 | 0.4547 | worse |
| drop raw absolute level features | 0.3607 | 0.4588 | worse |
| drop `scale` + `log_scale` | 0.3575 | 0.4542 | worse |
| scale-reweighted *training* | 0.3621 | 0.4593 | worse |
| weighted early stopping / `min_data_in_leaf=20` | worse | worse | worse |
| two-stage pair-total model | 0.3764 | 0.4720 | much worse |
| cross-film cluster x date in LightGBM | 0.3587 | 0.4572 | gate fail (helps group) |
| sibling format variants | 0.3622 | — | gate fail |
| `film_curve` stage 1 | 0.3618 | — | worse |
| third model class in the blend | — | — | +0.0004 |
| per-horizon multiplier, wide grid | +0.0000 | — | nothing |
| per-scale-bin x p0-decile zero snap | 0.3582 | 0.4530 | worse, over-parameterised |

Gate-passing total: **featset `F` (-0.0017 time / -0.0022 group) + weighted post-fit (-0.0023 time
/ -0.0004 group) ~= -0.004**, i.e. roughly 0.437 -> **0.433** on the leaderboard. Real, verified,
and far short of the 0.40 target.

**Honest conclusion.** Sixteen experiments against a now-trustworthy proxy yield about -0.004.
Reaching 0.375 needs something structurally different that is not in v4's formulation, and the
cluster of seven teams inside 0.371-0.379 looks like a shared public approach rather than seven
independent discoveries. That is the next thing to investigate, not another feature.

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


for _cell in cells:
    _cell["source"] = [ln.replace("__NB_STAMP__", STAMP).replace("__NB_NEEDS__", str(NEEDS_PIPELINE))
                       for ln in _cell["source"]]

_audit(cells)

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {NB}: {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code)")
