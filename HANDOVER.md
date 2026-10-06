# Handover

Everything checked so far on this competition, so the next person does not
repeat it. `FINDINGS.md` has the full evidence; this is the index and the
scoreboard.

Metric: MASE = MAE on `y / scale`, where
`scale = max((s1+s2+s3)/3, 1)` per (film, cluster) pair.

---

## 1. Leaderboard history

| # | build | proxy MASE | time holdout | **LB** |
|---|---|---|---|---|
| 0 | original notebook (`0.46641.ipynb`), blend A+B+H+Z, 140 feats | 0.4328 OOF | 0.3978 | **0.46641** |
| 1 | rewrite, no Eid guard | 0.3429 | — | 0.48267 |
| 2 | rewrite, Eid guard `calendar 0.6` | 0.3429 | — | 0.47789 |
| 3 | quantile snap, guard `flat 1.0`, film_curve, lags 0–12 | ~0.337 | — | **0.46795** |
| 4 | +model Z (no-guard variant submitted) | 0.3377 | — | 0.52220 |
| 5 | +Z at 100%, capped snap, lags 0–24, stage-1 uplift | **0.3348** | — | **0.53421** |
| 6 | Z disabled, everything else kept | 0.3901 fb | 0.3922 fb | *pending* |

Best so far is **0.46795** (build 3). The original 0.46641 is still ahead.

"proxy" = random 5-fold split over films, evaluated on release windows with a
Wed/Thu/Fri D1. "fb" = the pure-Python fallback model, not LightGBM, so those
two numbers are not comparable to the LightGBM rows above.

**Build 5 had the best proxy of any build and the worst leaderboard of any
build.** Read §6 before trusting the proxy.

---

## 2. Repo hygiene issue, fixed

`train.csv` on `main` is **2 bytes**. Commit `0709022` ("Rename train (1).csv
to train.csv") replaced the real file with an empty one, so the notebook could
not run from a fresh clone. Recovered from the dangling blob:

```bash
git cat-file blob 2627d8f3fbe231bae8fe052381f6ffff019775ff > train.csv
```

138,959 rows. Already restored on this branch.

---

## 3. Proven structural facts

Deterministic properties of the data, not model estimates. Re-verify with
`analysis/verify.py`, which the notebook also asserts in cell 3.

1. **A pair is forecast iff it sold a ticket on D3.** Exact, both directions:
   11,823 pairs have D1–D3 history, 10,373 were active on D3, and that set
   equals the test pair set exactly. All 1,450 excluded pairs have zero D3.
2. **No test pair lacks D1–D3 history, and no cluster appears after D3.** You
   are never asked to forecast an expansion.
3. **D1 is each film's wide-release date.** 67 distinct D1 dates for 163 films,
   up to 7 films sharing one, every one of the 25 weeks in span covered, D1
   weekday Wed 41% / Thu 40% / Fri 15%. So the test set is **~100% opening
   windows** — not the 13.7% the original notebook's regime classifier
   estimated.
4. **`anchor_lag` from "first appearance in train.csv" is corrupted.** 44% of
   train films first appear as a 1–4 cluster sneak preview 1–9 days before
   wide release. Redefining D1 as the first day reaching ≥25% of the film's
   first-week peak cluster count moves the weekday distribution from
   Fri-heavy to Wed 68 / Thu 58 / Fri 51, matching test.
5. **Horizon and day-of-week are nearly collinear in test.** D4 is Mon 45% /
   Sun 40%, D10 is Sat 45% / Fri 40%. A single decay curve over `off` is
   therefore badly mis-specified — the median `y/scale` at D4 is 1.114 for
   Wednesday releases and 0.454 for Friday ones.
6. **31.8% of test rows sit in a calendar regime train does not contain.**
   Ramadan 18.2%, Idulfitri 6.3%, Christmas/NY 7.2%. Ramadan 1446 H ended
   2025-03-30 and train starts 2025-04-01, so train has **zero** Ramadan rows,
   and `holidays.csv` never labels Ramadan at all.
7. **Train has zero windows straddling into an Eid window.** Train starts at
   Eid+1, so no training window can have observations before Eid and targets
   after. The 2026-03-18 cohort's situation is **structurally unfittable** —
   not a feature gap. Cell 14 asserts and prints this.

---

## 4. Where the error actually is

Measured on the proxy population (`analysis/headroom.py`, `exp2.py`).

| activity pattern (D1,D2,D3) | share of rows | MASE | share of total error |
|---|---|---|---|
| `111` | 90.58% | 0.3200 | **73.5%** |
| `001` | 1.99% | 2.8731 | 14.5% |
| `011` | 5.91% | 0.7070 | 10.6% |
| `101` | 1.52% | 0.3672 | 1.4% |

Also concentrated: **Wednesday releases at D4/D5** (the opening weekend) are
~12% of rows and ~22% of error. Partly mechanical — those cells have the
largest true ratios, and MASE is an absolute scaled error.

---

## 5. Oracle study — the ranking that should drive effort

Each oracle is handed perfect knowledge of **one** quantity
(`analysis/oracles.py`). The gap is the ceiling on predicting it well.

| oracle | MASE | ceiling |
|---|---|---|
| baseline | 0.3944 | — |
| **A — the film's own national D4–D10 ratio curve** | **0.3039** | **+0.0905** |
| D — the pair's own 7-day total | 0.3214 | +0.0731 |
| B — whether each row is zero | 0.3350 | +0.0595 |
| C — the target date's national demand | 0.3867 | +0.0078 |

Oracle A uses **no pair-level information at all** and still beats the full
per-pair baseline by 23%. The film-level curve is where the signal is.

Current capture of A: **+0.0209** of +0.0771 via a pure-Python stage-1 median
table (`film_curve`, rank 2 of 104 features at 12.89% of LightGBM gain).

---

## 6. The process lesson — read this first

**The proxy is a random split over films.** It measures "unseen film, same
period". The leaderboard measures "unseen film, **later** period". Those
differ, and model Z exploited the difference:

| split | main model | model Z |
|---|---|---|
| random film split | 0.4381 | 0.4427 |
| **time holdout** | **0.3922** | **0.3999** |

Z improved the proxy on every build and cost 0.05–0.07 on the leaderboard
every time. The original `0.46641.ipynb` **had** the right validation — its
cell 11, a time-based holdout — and the rewrite dropped it. That single
omission cost two submissions.

**Cell 10c now provides it.** Train on the earliest 75% of films by release
date, score the latest 25%.

> **Rule: ship a change only if it improves BOTH the proxy and the time
> holdout.** Treat proxy-only gains as exploitation of the random split until
> the holdout agrees.

Two further process notes:

* **The pure-Python fallback masked a LightGBM-specific bug.** The fallback
  emits hard zeros; LightGBM never does. So the absolute zero-snap threshold
  looked fine offline (+0.0008) while costing ~0.016 on the leaderboard. When
  a fallback exists, check whether the thing being validated behaves the same
  in both.
* **Avoid bundling.** Builds 3 and 5 each changed four things at once, which
  made attribution guesswork and wasted submissions. Submissions are limited
  to 3/day.

---

## 7. What worked

| change | evidence |
|---|---|
| **Idulfitri guard** (floor the ratio on Eid-window rows) | isolated A/B: 0.48267 → 0.47789 at `calendar 0.6` |
| **Per-horizon quantile zero-snap** instead of one absolute threshold | fixed D8–D10 predicting 22% zeros against a true 46–59% |
| **Capping the snap at the true zero share** | ~0.0005 proxy cost, removes the over-zeroing tail risk |
| **`film_curve` stage-1 feature** | rank 2 of 104, 12.89% of gain |
| **Target-day uplift in stage 1** | +0.0116 → **+0.0209** (`analysis/stage1_rich.py`) |
| **Wide-release D1 definition** | weekday distribution matches test (§3.4) |
| **Explicit calendar features** (`days_since_eid`, Ramadan, Xmas) | −0.0097 overall, **−0.045 on anomaly rows** (`exp5.py`) |
| Anchor lags 0–24 | LightGBM main model 0.3553 → 0.3497 |

---

## 8. What failed — do not redo these

| idea | result | script |
|---|---|---|
| **Model Z / two-part zero model** | **catastrophic: 0.46795 → 0.53421.** Forcing exact zeros is one-sided; a wrongly zeroed row loses the whole true ratio. Not a decile-mapping bug — rank-based deciles score the same | `z_shift.py` |
| **Cross-film target-date demand** | ceiling only **+0.0078**. 66% of test rows do have their target date observed by another film, but a national date factor is nearly collinear with day-of-week | `oracles.py`, `crossfilm.py` |
| **Dedicated `001` handling** | **irreducible.** Model 2.8731 already beats the best constant 3.1073; an oracle knowing the pair's 7-day total only reaches 2.5387. Whole-bucket headroom ~0.007 | `headroom.py` |
| Training only on lag-0 release windows | worse: 0.4024 vs 0.3932 | `harness.py` |
| Applying the `s3 > 0` filter to training data | neutral (0.3932 → 0.3932) | `replicate.py` |
| Predicting on a run-rate basis instead of `scale` | slightly worse (0.3950 vs 0.3932) | `exp2.py` |
| Anchor lags beyond 24 | worse (0–40: 0.3958) | `headroom.py` |
| Anchor lags cut to 0–12 | neutral for a median table, **worse for LightGBM** | `overzero.py` |
| Richer stage-1 aggregates (occupancy, shows, tickets/show) | no gain over basic (+0.0119 vs +0.0116) | `stage1_rich.py` |
| Replacing the ensemble with one model | the original's blend was worth ~0.037 OOF on its own trace | — |

Two claims from earlier in this work were **retracted** on measurement:
cross-film features (was "the biggest miss") and `001` handling (was "8% of
rows, 25% of error"). Both are in §8 above.

---

## 9. The 2026-03-18 cohort

Seven films — `DANUR: THE LAST CHAPTER` (+IMAX), `NA WILLA`,
`PELANGI DI MARS`, `SENIN HARGA NAIK`, `SUZZANNA: SANTET DOSA DI ATAS DOSA`,
`TUNGGU AKU SUKSES NANTI`.

* D1–D3 = 2026-03-18…20, the end of Ramadan
* D4–D10 = 2026-03-21…27, **Idulfitri 1447 H plus the whole week after**
* 631 pairs, **4,417 rows = 6.08% of the test set**, at the end of the period
  so likely concentrated in the private leaderboard

Train's only Eid analogue (1,470 rows in the Idulfitri 1446 aftermath) has
median `y/scale` **1.880** against **0.357** on ordinary days, with an optimal
floor around `calendar K≈1.5` or `flat K≈1.9` (`analysis/eid_premise.py`).

The floor **cannot be fitted** (§3.7), so the leaderboard has to choose it.
The notebook writes `submission.csv` (floor 1.0),
`submission_floor_1.3.csv`, `submission_floor_1.6.csv` and
`submission_no_eid_guard.csv` in one run to avoid wasting runs.

Current setting: `EID_GUARD_MODE = "flat"`, `EID_GUARD_K = 1.0`. Untested
above 1.0.

---

## 10. Open leads, best first

1. **Move stage 1 to LightGBM.** Ceiling +0.0771, currently banking +0.0209.
   Largest remaining lever by a wide margin. ~17.7k film-window rows, so watch
   for overfitting on a small table.
2. **Test the Eid floor at 1.3 and 1.6.** Files are already written. Worth up
   to ~0.01–0.02 on 6.08% of rows, and the analogue argues the floor is low.
3. **Port the fixes into `0.46641.ipynb` rather than replacing it.** That
   notebook's blend (A+B+H) is worth ~0.037 OOF on its own trace and the
   rewrite has no blend at all. Keep its ensemble, add the calendar features,
   the wide-release D1, and `film_curve`; drop its regime reweighting (§3.3
   shows the 13.7% estimate it relies on is wrong).
4. **Wednesday D4/D5.** ~12% of rows, ~22% of error. Oracle A should attack it
   since the film curve carries the opening-weekend magnitude, but the
   multiplicative two-stage form regressed it (0.692 → 0.741) while helping
   Thu/Fri — hence exposing the curve as a feature instead.
5. **Format variants.** 17 base titles have >1 screen format (IMAX/3D/etc),
   covering 16.0% of test rows. Spread in `y/scale` between formats in the
   same (cluster, date) slot is median 0.265 — correlated but not duplicates,
   so a sibling feature rather than pooling. Unexplored.
6. `101` pattern: predicting zero scores 0.3288 against the model's 0.3672.
   Trivial (+0.0006) but free.

---

## 11. Files

| path | what |
|---|---|
| `FINDINGS.md` | full evidence, in the order it was discovered, including both post-mortems |
| `corrected_pipeline.ipynb` | runnable pipeline, 36 cells. Needs numpy + lightgbm; pandas not required |
| `pipeline.py` | same code as a flat script; the notebook is generated from it |
| `fixes.py` | standalone drop-in functions for porting into another notebook |
| `0.46641.ipynb` | the original 0.46641 submission, kept for reference |
| `analysis/*.py` | 23 pure-stdlib scripts, one per experiment. Every number quoted anywhere is reproduced by one of these |

The pipeline deliberately uses only the standard library for all data handling
and feature engineering; numpy and lightgbm are touched in three cells. It
carries a hierarchical-median fallback so the whole thing runs without
LightGBM — useful for testing, but see the warning in §6 about what that
fallback can hide.

### Key analysis scripts

```
verify.py          the two proven structural rules (D3 rule, release calendar)
d1_def.py          sneak-preview corruption of anchor_lag
calendar.py        train/test calendar regime overlap
straddle.py        window-straddle exposure
eid.py             the 2026-03-18 cohort
eid_premise.py     does uplift predict Eid-day ratios
oracles.py         the oracle ranking  <- start here
two_stage.py       is the film curve predictable
stage1_rich.py     stage-1 feature variants
headroom.py        is 001 fixable; anchor-lag sweep
z_shift.py         why model Z fails under time shift
level_error.py     far-horizon level error vs the 0.46641 profile
calib_bug.py       clipped multiplier grid, global vs per-horizon snap
overzero.py        is capping the snap safe; lag range per horizon
harness.py         construction comparison
```
