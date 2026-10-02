"""Drop-in fixes for the cinema-ticket MASE notebook.

Paste these as new notebook cells on Kaggle (pandas + lightgbm available there).
Each function is independent and documented with the finding it addresses.
See FINDINGS.md for the evidence behind every choice.

Ordered by measured impact:
  1. islamic_calendar_features   -> FINDINGS §1, §2   (~32% of test rows)
  2. eid_cohort_guard            -> FINDINGS §2       (6.08% of test rows)
  3. wide_release_date           -> FINDINGS §5       (fixes the regime axis)
  4. pair_activity_features      -> FINDINGS §6       (8% of rows, 25% of error)
  5. cross_film_features         -> FINDINGS §8       (66% coverage; unproven)
  6. verify_test_construction    -> FINDINGS §3, §4   (assertions)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 2026

# ---------------------------------------------------------------------------
# Islamic calendar anchors for the two Hijri years spanned by this dataset.
# Taken as the FIRST day of Idulfitri (1 Syawal) as observed in Indonesia.
#
#   1446 H : Idulfitri 2025-03-31  ->  train 2025-04-01 is Eid+1
#   1447 H : Idulfitri 2026-03-21  ->  holidays.csv marks 03-21 and 03-22
#
# holidays.csv only labels 2025-04-01 (Eid+1) because the file starts there,
# so these are hardcoded rather than derived, to keep the two years aligned
# on the same Eid-relative index.
# ---------------------------------------------------------------------------
EID_DATES = [pd.Timestamp("2025-03-31"), pd.Timestamp("2026-03-21")]
RAMADAN_LEN = 30


def islamic_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """FINDINGS §1, §2.

    Train (Apr-Sep 2025) contains no Ramadan, and holidays.csv never labels
    Ramadan, so 18.2% of test rows currently have no representable feature.
    But train DOES contain the Idulfitri 1446 aftermath (2025-04-01..04-13 =
    Eid+1..Eid+13), which is exactly what the 2026-03-18 cohort needs.

    `days_since_eid` is therefore a genuinely learnable bridge between the two
    periods, not an extrapolation.

    Returns one row per input date. Safe to compute for observation dates and
    target dates alike -- it depends only on the calendar.
    """
    # normalise to a day-resolution numpy array (version-robust: avoids
    # Series.view / .astype('int64') differences across pandas versions)
    d = pd.to_datetime(np.asarray(pd.Series(dates).to_numpy()))
    day = np.asarray(d, dtype="datetime64[D]")
    out = pd.DataFrame(index=range(len(day)))

    # signed distance in days to the nearest Eid
    eid = np.array([np.datetime64(e.date(), "D") for e in EID_DATES])
    diff = (day[:, None] - eid[None, :]).astype("int64")
    nearest = np.argmin(np.abs(diff), axis=1)
    signed = diff[np.arange(len(day)), nearest]

    out["days_since_eid"] = np.clip(signed, -45, 45).astype(np.int16)
    in_eid = (signed >= 0) & (signed <= 13)
    out["is_eid_window"] = in_eid.astype(np.int8)
    out["eid_week"] = np.where(in_eid, (signed // 7) + 1, 0).astype(np.int8)

    # Ramadan = the 30 days before 1 Syawal
    in_ram = (signed >= -RAMADAN_LEN) & (signed < 0)
    out["in_ramadan"] = in_ram.astype(np.int8)
    out["ramadan_day"] = np.where(in_ram, signed + RAMADAN_LEN + 1, 0).astype(np.int8)
    # last 10 nights: attendance collapses hardest, then rebounds into Eid
    out["ramadan_late"] = (in_ram & (signed >= -10)).astype(np.int8)

    # Christmas / New Year block
    month = day.astype("datetime64[M]").astype(int) % 12 + 1
    dom = (day - day.astype("datetime64[M]")).astype("int64") + 1
    out["in_xmas_ny"] = (((month == 12) & (dom >= 20)) |
                         ((month == 1) & (dom <= 4))).astype(np.int8)
    out["days_to_xmas"] = np.clip(
        np.where(month == 12, 25 - dom,
                 np.where(month == 1, -(dom + 6), 99)), -20, 99).astype(np.int16)
    return out


def add_calendar_regime(df: pd.DataFrame, date_col: str,
                        prefix: str = "") -> pd.DataFrame:
    """Attach islamic_calendar_features for `date_col`, with an optional prefix."""
    f = islamic_calendar_features(df[date_col])
    f.index = df.index
    if prefix:
        f = f.add_prefix(prefix)
    return pd.concat([df, f], axis=1)


def observation_window_calendar(pairs: pd.DataFrame,
                                d1_col: str = "d1") -> pd.DataFrame:
    """FINDINGS §1. Composition of the D1-D3 window itself.

    `scale` is the mean of D1-D3, so WHICH days those were sets the denominator.
    A window of three Ramadan weekdays produces a much smaller scale than three
    ordinary weekend days, and the model must know that to interpret the ratio.
    """
    out = pd.DataFrame(index=pairs.index)
    d1 = pd.to_datetime(pairs[d1_col])
    for i in range(3):
        f = islamic_calendar_features(d1 + pd.Timedelta(days=i))
        f.index = pairs.index
        for c in ("in_ramadan", "is_eid_window", "in_xmas_ny"):
            out[f"obs{i+1}_{c}"] = f[c].values
    out["n_ramadan_obs"] = sum(out[f"obs{i+1}_in_ramadan"] for i in range(3))
    out["n_eid_obs"] = sum(out[f"obs{i+1}_is_eid_window"] for i in range(3))
    out["n_xmas_obs"] = sum(out[f"obs{i+1}_in_xmas_ny"] for i in range(3))
    f1 = islamic_calendar_features(d1)
    f1.index = pairs.index
    out["d1_days_since_eid"] = f1["days_since_eid"].values
    out["d1_in_ramadan"] = f1["in_ramadan"].values
    return out


# ---------------------------------------------------------------------------
def eid_cohort_guard(X_te: pd.DataFrame, pred_ratio: np.ndarray,
                     min_profile: float = 1.0,
                     verbose: bool = True) -> np.ndarray:
    """FINDINGS §2. Floor the ratio profile for Eid-crossing windows.

    The 2026-03-18 cohort (7 films, 631 pairs, 4,417 rows = 6.08% of test) has
    its entire D4-D10 window inside the Idulfitri 1447 holiday week. The
    0.46641 submission predicts a mean ratio of 0.411 decaying to 0.132 by D10;
    the April 2025 analogue says the week after Eid runs at or above the Eid
    weekend and ~3x the later baseline, so the truth is above 1.

    This is a floor, not a multiplier: it only raises predictions on rows whose
    TARGET day is in the Eid window, and leaves everything else untouched.

    Expected value at min_profile=1.0 and a true ratio of 1.2: about -0.048 MASE.
    Raise `min_profile` only if you are willing to bet on a larger uplift.

    Parameters
    ----------
    X_te       test frame; must contain `date_show` and `off` (4..10)
    pred_ratio predicted y/scale, same length and order as X_te
    """
    pred = np.asarray(pred_ratio, dtype=np.float64).copy()
    f = islamic_calendar_features(X_te["date_show"])
    in_eid = f["is_eid_window"].values.astype(bool)

    if verbose:
        n = int(in_eid.sum())
        print(f"[eid_guard] rows with target day in an Eid window: {n:,} "
              f"({100*n/len(pred):.2f}% of test)")
        if n:
            print(f"[eid_guard] mean predicted ratio there before: "
                  f"{pred[in_eid].mean():.3f}")

    # Flat floor across the horizon: the holiday week does not decay.
    pred[in_eid] = np.maximum(pred[in_eid], min_profile)

    if verbose and in_eid.any():
        print(f"[eid_guard] mean predicted ratio there after : "
              f"{pred[in_eid].mean():.3f}")
        print(f"[eid_guard] rows actually raised: "
              f"{int((pred != np.asarray(pred_ratio)).sum()):,}")
    return pred


# ---------------------------------------------------------------------------
def wide_release_date(hist: pd.DataFrame, cluster_frac: float = 0.25,
                      window: int = 9) -> pd.Series:
    """FINDINGS §5. Replacement for 'first appearance in train.csv'.

    44% of train films first appear as a 1-4 cluster sneak preview 1-9 days
    before the real wide release, which corrupts `anchor_lag` and therefore
    every regime label, the regime classifier, the 13.7% opening estimate and
    the weights derived from it.

    Definition: the first day within the film's first `window` days that reaches
    at least `cluster_frac` of the film's peak cluster count in that span.

    Validation: applying this to train moves the release day-of-week from
    Fri 55 / Wed 44 / Thu 32 (first-appearance) to Wed 68 / Thu 58 / Fri 51,
    which matches the test set's Wed 41% / Thu 40% / Fri 15%.

    Returns a Series indexed by movie_title.
    """
    g = (hist.groupby(["movie_title", "date_show"])["cinema_ids"]
         .nunique().rename("ncl").reset_index())
    first = g.groupby("movie_title")["date_show"].min().rename("first")
    g = g.merge(first, on="movie_title")
    g = g[g["date_show"] <= g["first"] + pd.Timedelta(days=window)]
    peak = g.groupby("movie_title")["ncl"].transform("max")
    ok = g[g["ncl"] >= cluster_frac * peak]
    return ok.groupby("movie_title")["date_show"].min().rename("d1")


# ---------------------------------------------------------------------------
def pair_activity_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """FINDINGS §6. Make the partial-activity structure explicit.

    pattern `001` is 1.99% of rows at MASE 2.88 and `011` is 5.91% at 0.71 --
    together 8% of rows and 25% of the total error. The test set has the same
    mix (189 + 684 pairs = 8.4%).

    Cause: `scale = max((s1+s2+s3)/3, 1)` averages over days the pair was not
    screening, understating the real run-rate by 3/n_active. Expect the model
    to need `scale_deflation` to recover a ratio near 1.8-3 on `001` rows.

    NOTE: reparameterising the target onto a run-rate basis does NOT help
    (tested: 0.3950 vs 0.3932). Supply these as features instead.

    `pairs` must contain s1, s2, s3.
    """
    p = pd.DataFrame(index=pairs.index)
    s = pairs[["s1", "s2", "s3"]].to_numpy(dtype=np.float64)
    act = (s > 0)
    n_act = act.sum(axis=1)

    p["n_active_days"] = n_act.astype(np.int8)
    p["act_pattern"] = (act[:, 0] * 4 + act[:, 1] * 2 + act[:, 2]).astype(np.int8)
    p["starts_late"] = (~act[:, 0]).astype(np.int8)          # s1 == 0
    p["only_d3"] = ((~act[:, 0]) & (~act[:, 1])).astype(np.int8)
    p["has_gap"] = (act[:, 0] & ~act[:, 1] & act[:, 2]).astype(np.int8)

    sum3 = s.sum(axis=1)
    scale = np.maximum(sum3 / 3.0, 1.0)
    rate = np.where(n_act > 0, sum3 / np.maximum(n_act, 1), 0.0)

    p["run_rate"] = rate.astype(np.float32)
    # the multiplicative correction the tree would otherwise have to discover
    p["scale_deflation"] = (3.0 / np.maximum(n_act, 1)).astype(np.float32)
    p["rate_over_scale"] = (rate / scale).astype(np.float32)
    p["log_run_rate"] = np.log1p(rate).astype(np.float32)
    # level on the most recent observed day, which for `001` is the only signal
    p["last_active_level"] = np.where(act[:, 2], s[:, 2],
                                      np.where(act[:, 1], s[:, 1], s[:, 0])
                                      ).astype(np.float32)
    p["last_over_scale"] = (p["last_active_level"] / scale).astype(np.float32)
    return p


# ---------------------------------------------------------------------------
def cross_film_features(test_history: pd.DataFrame, target: pd.DataFrame,
                        d1_map: pd.Series, min_films: int = 3) -> pd.DataFrame:
    """FINDINGS §8. Demand on the target date, observed from OTHER films.

    test_history.csv holds D1-D3 for 163 films at different calendar dates, so
    66.2% of test rows have their target date observed by another film and 58.4%
    have the exact (cluster, date) observed. Coverage is highest at D8-D10
    (88-93%), exactly where a pure decay model is blindest.

    This is provided data, not D4-D10 labels. The film being predicted is always
    excluded from its own features.

    HONEST STATUS: I could not measure a gain from this with a median-table
    evaluator (best -0.0021; cruder forms hurt). A national date factor is
    nearly collinear with day-of-week, which the model already has; its unique
    content is calendar anomalies, and train has almost none to learn from.
    Evaluate it on Ramadan/Eid/Christmas rows specifically, not on overall OOF.

    Parameters
    ----------
    test_history  D1-D3 transactions (date_show, cinema_ids, movie_title,
                  total_ticket)
    target        rows to featurise; needs movie_title, cinema_ids, date_show
    d1_map        Series movie_title -> D1
    """
    h = test_history
    # per-film window mean, used to strip film-level scale before averaging
    fm = h.groupby(["movie_title", "date_show"])["total_ticket"].sum().rename("t")
    fm = fm.reset_index()
    mu = fm.groupby("movie_title")["t"].mean().rename("mu")
    fm = fm.merge(mu, on="movie_title")
    fm = fm[fm["mu"] > 0]
    fm["rel"] = fm["t"] / fm["mu"]

    # national date factor: leave-one-film-out via sum/count bookkeeping
    agg = fm.groupby("date_show")["rel"].agg(["sum", "count"])
    out = pd.DataFrame(index=target.index)
    j = target[["movie_title", "date_show"]].merge(
        agg, left_on="date_show", right_index=True, how="left")
    own = target[["movie_title", "date_show"]].merge(
        fm[["movie_title", "date_show", "rel"]],
        on=["movie_title", "date_show"], how="left")["rel"].fillna(0.0).to_numpy()
    own_n = target[["movie_title", "date_show"]].merge(
        fm[["movie_title", "date_show"]].assign(_x=1),
        on=["movie_title", "date_show"], how="left")["_x"].fillna(0.0).to_numpy()
    s = j["sum"].fillna(0.0).to_numpy() - own
    n = j["count"].fillna(0.0).to_numpy() - own_n
    out["xf_date_factor"] = np.where(n >= min_films, s / np.maximum(n, 1),
                                     np.nan).astype(np.float32)
    out["xf_n_films_on_date"] = n.astype(np.int16)

    # cluster-level: was this cluster trading on the target date at all?
    cd = (h.groupby(["cinema_ids", "date_show"])
          .agg(cl_t=("total_ticket", "sum"), cl_m=("movie_title", "nunique"))
          .reset_index())
    k = target[["cinema_ids", "date_show"]].merge(
        cd, on=["cinema_ids", "date_show"], how="left")
    out["xf_cluster_seen"] = k["cl_t"].notna().astype(np.int8).to_numpy()
    out["xf_cluster_tickets"] = k["cl_t"].fillna(0.0).astype(np.float32).to_numpy()
    out["xf_cluster_nfilms"] = k["cl_m"].fillna(0).astype(np.int16).to_numpy()

    # cluster demand on the target date relative to during this film's D1-D3
    d1 = pd.to_datetime(target["movie_title"].map(d1_map))
    base = np.zeros(len(target), dtype=np.float64)
    cnt = np.zeros(len(target), dtype=np.float64)
    for i in range(3):
        bb = target[["cinema_ids"]].copy()
        bb["date_show"] = d1 + pd.Timedelta(days=i)
        m = bb.merge(cd, on=["cinema_ids", "date_show"], how="left")["cl_t"]
        v = m.fillna(0.0).to_numpy()
        base += v
        cnt += (m.notna().to_numpy()).astype(np.float64)
    mu_cl = np.where(cnt > 0, base / np.maximum(cnt, 1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["xf_cluster_rel"] = (out["xf_cluster_tickets"].to_numpy() /
                                 mu_cl).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
def verify_test_construction(test: pd.DataFrame,
                             test_history: pd.DataFrame) -> dict:
    """FINDINGS §3, §4. Assert the two proven structural rules still hold.

    Run this once. If an assertion fails, the rest of the reasoning in
    FINDINGS.md does not apply to your files.
    """
    h = test_history.copy()
    d1 = h.groupby("movie_title")["date_show"].min().rename("d1")
    h = h.merge(d1, on="movie_title")
    h["off"] = (h["date_show"] - h["d1"]).dt.days

    obs = h[h["off"].between(0, 2)]
    active_d3 = set(map(tuple, obs.loc[obs["off"] == 2,
                                       ["movie_title", "cinema_ids"]].to_numpy()))
    test_pairs = set(map(tuple, test[["movie_title", "cinema_ids"]]
                         .drop_duplicates().to_numpy()))
    all_pairs = set(map(tuple, obs[["movie_title", "cinema_ids"]]
                        .drop_duplicates().to_numpy()))

    res = {
        "pairs_with_history": len(all_pairs),
        "pairs_active_d3": len(active_d3),
        "pairs_in_test": len(test_pairs),
        "rule_active_d3_matches_test": active_d3 == test_pairs,
        "pairs_without_history": len(test_pairs - all_pairs),
        "n_distinct_d1_dates": int(d1.nunique()),
        "n_films": int(d1.size),
        "d1_dow_counts": d1.dt.dayofweek.value_counts().sort_index().to_dict(),
    }

    print("FINDINGS §3 -- test.csv pairs == pairs active on D3")
    print(f"  pairs with D1-D3 history : {res['pairs_with_history']}")
    print(f"  pairs active on D3       : {res['pairs_active_d3']}")
    print(f"  pairs in test.csv        : {res['pairs_in_test']}")
    print(f"  rule holds exactly       : {res['rule_active_d3_matches_test']}")
    print(f"  test pairs with no D1-D3 : {res['pairs_without_history']}")
    print()
    print("FINDINGS §4 -- D1 is a weekly release slot, so test is 100% opening")
    print(f"  distinct D1 dates        : {res['n_distinct_d1_dates']} "
          f"for {res['n_films']} films")
    print(f"  D1 weekday counts (0=Mon): {res['d1_dow_counts']}")
    print("  -> a per-film random window would give ~1 distinct date per film,")
    print("     spread over all 7 weekdays.")

    assert res["rule_active_d3_matches_test"], "the D3 rule no longer holds"
    assert res["pairs_without_history"] == 0
    return res


# ---------------------------------------------------------------------------
def decay_prior_by_d1dow(X_tr: pd.DataFrame) -> pd.DataFrame:
    """FINDINGS §7. Replace the single DECAY_PRIOR curve over `off`.

    81% of test films share two D1 weekdays, so horizon and target weekday are
    nearly collinear in the test set (D4 is Mon 45% / Sun 40%, D10 is Sat 45% /
    Fri 40%). One curve over `off` averages over a train D1-dow mix that does
    not match test.

    Measured: median by horizon alone 0.4506 -> by horizon x D1-dow 0.4259.

    `X_tr` needs `off`, `d1`, `y_ratio`. Returns a frame indexed by
    (d1_dow, off); reindex and ffill/bfill before use.
    """
    t = X_tr[["off", "y_ratio"]].copy()
    t["d1_dow"] = pd.to_datetime(X_tr["d1"]).dt.dayofweek
    out = (t.groupby(["d1_dow", "off"])["y_ratio"]
           .agg(prior="median", n="size").reset_index())
    # fall back to the pooled curve wherever a cell is thin
    pooled = t.groupby("off")["y_ratio"].median().rename("pooled")
    out = out.merge(pooled, on="off")
    out["prior"] = np.where(out["n"] >= 200, out["prior"], out["pooled"])
    return out.set_index(["d1_dow", "off"])[["prior", "n"]]
