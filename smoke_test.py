"""Runtime smoke test for cinema_v5.ipynb cells 1-5.

Builds SYNTHETIC csv files with the same schema as the competition data in a temp
directory, then executes the notebook's code cells against them. This only checks
that the code runs; the numbers it prints are meaningless.

Run:  python smoke_test.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
SMOKE = Path("/tmp/smoke")
if SMOKE.exists():
    shutil.rmtree(SMOKE)
SMOKE.mkdir(parents=True)

CITIES = ["SURABAYA", "JAKARTA", "MEDAN"]
CLUSTERS = [f"cl{i:03d}" for i in range(8)]
CL_CITY = {c: CITIES[i % len(CITIES)] for i, c in enumerate(CLUSTERS)}


def make_history(films, start, end, n_clusters_rng=(2, 7), run_rng=(4, 30)):
    """Rows of a transaction history file."""
    rows = []
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    for f in films:
        d1 = pd.Timestamp(start) + pd.Timedelta(days=int(RNG.integers(0, max(1, days - 2))))
        run = int(RNG.integers(*run_rng))
        cls = RNG.choice(CLUSTERS, size=int(RNG.integers(*n_clusters_rng)), replace=False)
        for c in cls:
            life = max(1, run - int(RNG.integers(0, run)))
            for k in range(life):
                d = d1 + pd.Timedelta(days=k)
                if d > pd.Timestamp(end):
                    break
                if RNG.random() < 0.12:          # some days simply have no transaction row
                    continue
                rows.append((d, c, CL_CITY[c], f, int(RNG.integers(0, 400)),
                             round(float(RNG.uniform(1, 60)), 2), int(RNG.integers(1, 20))))
    return pd.DataFrame(rows, columns=["date_show", "cinema_ids", "city_name", "movie_title",
                                       "total_ticket", "occupation_rate", "total_show"])


# ---------------------------------------------------------------- train.csv
train = make_history([f"TRAIN FILM {i}" for i in range(40)], "2025-04-01", "2025-09-30")
# force a few films to start exactly at the file start (left-censored) so the
# drop_censored branch in opening_rows() is exercised
train.loc[train.movie_title == "TRAIN FILM 0", "date_show"] = pd.Timestamp("2025-04-01")
train.to_csv(SMOKE / "train.csv", index=False)

# ------------------------------------------------- test_history.csv + test.csv
test_films = [f"TEST FILM {i}" for i in range(18)]
th_rows, te_rows = [], []
for f in test_films:
    d1 = pd.Timestamp("2025-10-08") + pd.Timedelta(days=int(RNG.integers(0, 160)))
    cls = RNG.choice(CLUSTERS, size=int(RNG.integers(2, 7)), replace=False)
    for c in cls:
        for k in range(3):                                   # D1..D3 history
            if RNG.random() < 0.1:
                continue
            th_rows.append((d1 + pd.Timedelta(days=k), c, CL_CITY[c], f,
                            int(RNG.integers(0, 300)), round(float(RNG.uniform(1, 60)), 2),
                            int(RNG.integers(1, 15))))
        for k in range(3, 10):                               # D4..D10 targets
            te_rows.append((f, c, CL_CITY[c], d1 + pd.Timedelta(days=k)))
# a pair asked about in test.csv with NO history at all (the scale == NaN case)
for k in range(3, 10):
    te_rows.append(("TEST FILM 0", "cl007", CL_CITY["cl007"],
                    pd.Timestamp("2025-10-08") + pd.Timedelta(days=k)))
# make one test film also appear in train.csv (overlap case in A5/A6)
_ov = train[train.movie_title == "TRAIN FILM 1"].copy()
_ov["movie_title"] = "TEST FILM 1"
train = pd.concat([train, _ov], ignore_index=True)
train.to_csv(SMOKE / "train.csv", index=False)

th = pd.DataFrame(th_rows, columns=["date_show", "cinema_ids", "city_name", "movie_title",
                                    "total_ticket", "occupation_rate", "total_show"])
th.to_csv(SMOKE / "test_history.csv", index=False)

test = pd.DataFrame(te_rows, columns=["movie_title", "cinema_ids", "city_name", "date_show"])
test.insert(0, "id", np.arange(1, len(test) + 1))
test.to_csv(SMOKE / "test.csv", index=False)
pd.DataFrame({"id": test.id, "total_ticket": 100.0}).to_csv(SMOKE / "sample_submission.csv", index=False)

# ------------------------------------------- reference tables (real calendar)
shutil.copy("holidays.csv", SMOKE / "holidays.csv")
shutil.copy("ticket_prices.csv", SMOKE / "ticket_prices.csv")
pd.DataFrame({"original_title": [f"TRAIN FILM {i}" for i in range(40)] + test_films,
              "age_rating": "Remaja", "genre": "Drama", "producer": "p",
              "director": "d", "writer": "w", "casts": "c"}).to_csv(SMOKE / "movies.csv", index=False)

print(f"synthetic data in {SMOKE}: train {train.shape}, test_history {th.shape}, test {test.shape}")
print("=" * 78)

# --------------------------------------------------------------- run the cells
nb = json.load(open("cinema_v5.ipynb"))
srcs = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]

import os
os.chdir(SMOKE)
env = {"__name__": "__main__"}
failed = 0
for src in srcs:
    name = src.splitlines()[0]
    try:
        exec(compile(src, name, "exec"), env)
        print(f"\n>>> PASS {name}\n")
    except Exception as e:
        failed += 1
        print(f"\n>>> FAIL {name}: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        break
print("=" * 78)
print("SMOKE TEST:", "ALL CELLS RAN" if not failed else "FAILURE - fix before shipping")
sys.exit(1 if failed else 0)
