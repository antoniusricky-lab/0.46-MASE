# Diagnosis of the 0.46641 submission

All numbers below were reproduced locally from the competition files. Scripts
are in `analysis/`. Where I claim something is *proven*, it is a deterministic
property of the data, not a model estimate.

---

## 0. Repo problem: `train.csv` was destroyed

`train.csv` in `main` is **2 bytes** (`\r\n`). Commit `0709022` ("Rename
train (1).csv to train.csv") replaced the real file with an empty one. The
notebook cannot run from a fresh clone.

The real file survives as a dangling blob and is recovered:

```bash
git cat-file blob 2627d8f3fbe231bae8fe052381f6ffff019775ff > train.csv   # 138,959 rows
```

Already restored in this branch.

---

## 1. The leaderboard gap is a CALENDAR problem, not a film-age problem

This is the headline. Your notebook spends cells 13–14 estimating a *film-age*
regime mix ("buka / awal / tengah / akhir") and reweights training to it. It
never checks the **calendar** composition — and that is what explains the gap.

Measured on train release windows (`analysis/exp5.py`):

| target day type | MASE |
|---|---|
| ordinary day | **0.3587** |
| public holiday or post-Idulfitri boom | **0.8485** (2.37×) |

Anomaly share: **7.1% of train rows** vs **31.8% of test rows**.

Reweighting the measured per-regime MASE to the test calendar mix predicts
**0.5144** — versus your OOF 0.4328 and holdout 0.3978. Your LB is 0.46641.
The calendar mix alone accounts for essentially the whole discrepancy.

Test row exposure (`analysis/calendar.py`, `straddle.py`):

| regime | rows | share |
|---|---|---|
| ordinary | 49,547 | 68.2% |
| Ramadan 1447 H (2026-02-17 … 03-19) | 13,231 | 18.2% |
| Idulfitri 1447 H and after | 4,598 | 6.3% |
| Christmas / New Year | 5,235 | 7.2% |

**Train contains no Ramadan at all.** Ramadan 1446 H ran 2025-03-01…03-30 and
Idulfitri 1446 H fell 2025-03-31 — both *before* `train.csv` starts on
2025-04-01. `holidays.csv` does not label Ramadan either, so your model has no
feature that can even represent 18.2% of the test set.

Because MASE divides by `scale` (the pair's own D1–D3 mean), a *uniform* level
shift is absorbed. What is not absorbed is a window that **straddles** a
boundary: **14.8% of test rows have a target-day regime different from their
own D1–D3 regime.** That is the honest size of the problem.

---

## 2. Biggest single concentrated win: the 2026-03-18 cohort

Seven films release 2026-03-18 (`analysis/eid.py`):

```
DANUR: THE LAST CHAPTER          NA WILLA            SENIN HARGA NAIK
DANUR: THE LAST CHAPTER (IMAX)   PELANGI DI MARS     TUNGGU AKU SUKSES NANTI
SUZZANNA: SANTET DOSA DI ATAS DOSA
```

* D1–D3 = 2026-03-18…20 — the last days of Ramadan
* D4–D10 = 2026-03-21…27 — **Idulfitri 1447 H plus the whole week after**
* 631 pairs, **4,417 rows = 6.08% of the test set**, all 7 target days in the
  Eid holiday week

Train's analogue (Idulfitri 1446 H aftermath, April 2025) shows the week after
Eid running **1.0–1.26×** the Eid weekend and ~3× the later baseline; national
daily tickets peak at 693k on Eid+7 versus a 160k June baseline.

Your submitted profile (cell 17 output) is pointed the wrong way for them:

| horizon | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|
| mean predicted ratio | 0.952 | 0.686 | 0.440 | 0.344 | 0.183 | 0.142 | 0.132 |
| share predicted zero | 10.6% | 17.3% | 25.4% | 38.0% | 58.8% | 68.9% | 71.7% |

Mean predicted ratio over D4–D10 is **0.411**. For this cohort the true ratio
should sit *above* 1.

| if true mean ratio is | per-row error | cost to your total MASE |
|---|---|---|
| 1.2 | 0.79 | **0.048** |
| 1.5 | 1.09 | **0.066** |
| 2.0 | 1.59 | **0.097** |

Note this cohort sits at the very end of the test period, so it is likely
concentrated in the **private** leaderboard.

**The fix is learnable, not guesswork.** `days_since_eid` is directly
observable in train: 2025-04-01…04-13 is Eid+1…Eid+13. Six of this cohort's
seven target days (Eid+1…Eid+6) are covered by real training data.

---

## 3. PROVEN: test pairs are exactly the pairs active on D3

`analysis/verify.py`:

```
pairs with D1-D3 history      : 11823
pairs active on D3            : 10373
pairs in test.csv             : 10373
active-on-D3 == test.csv pairs: True      <- exact, both directions
```

| D1,D2,D3 active | predicted | excluded |
|---|---|---|
| `111` | 9454 | 0 |
| `101` | 46 | 0 |
| `011` | 684 | 0 |
| `001` | 189 | 0 |
| `110` | 0 | 862 |
| `100` | 0 | 514 |
| `010` | 0 | 74 |

A pair is forecast **iff** it sold at least one ticket on D3. Also: zero test
pairs lack D1–D3 history, and **zero clusters appear after D3** — you are never
asked to forecast an expansion.

Your training set does not replicate this. It filters `sum3 > 0` only, so
**28.6% of your lag-0 pairs are of a type that cannot occur in test**, and 52.7%
of those are completely dead in D4–D10 (`analysis/replicate.py`).

Honest caveat: I tested removing them and it did **not** help training
(0.3932 either way) — extra volume compensates. The damage is to your
**diagnostics**: these unreachable pairs are what inflate your "buka" regime
MASE to 0.889, which is what drove the wrong reweighting decision in §5.

---

## 4. PROVEN: D1 is the film's wide-release date, so test is 100% "opening"

`analysis/regime_check.py`, `verify.py`:

* **67 distinct D1 dates for 163 films** — if D1 were a per-film random window
  you would see ~160 distinct dates
* 2.68 distinct D1 dates per week, and **every one of the 25 weeks in the span
  has a release**
* up to **7 films share a single D1 date** — these are release cohorts
* D1 day-of-week: **Wed 41%, Thu 40%, Fri 15%**, Sat 3%, Tue 1% (not uniform)
* D1 is the film's maximum cluster count for 95/163 films; median
  `ncl(D1)/max(ncl)` = 1.00

That is a weekly release calendar. **Every test window is an opening window.**

Your cell 13 regime classifier estimated **13.7% opening** and you reweighted
training to that number. The true value is ~100%.

---

## 5. PROVEN: your `anchor_lag` origin is corrupted by sneak previews

This is *why* the regime estimate went wrong. You define `anchor_lag` from the
film's first appearance in `train.csv`. For 44% of films that is a tiny
preview, 1–9 days before the wide release (`analysis/d1_def.py`):

| first-appearance dow | films | tickets on day 0 | clusters on day 0 | clusters on day 1 |
|---|---|---|---|---|
| Mon | 30 | 592 | 1.3 | 0.4 |
| Tue | 28 | 1,384 | 1.6 | 33.5 |
| Sun | 10 | 602 | 3.7 | 0.8 |
| Wed | 44 | 11,574 | 24.2 | 21.0 |

Monday/Tuesday/Sunday "releases" are 1–4 cluster previews. Redefining D1 as the
first day reaching ≥25% of the film's first-week peak cluster count moves the
distribution to **Wed 68, Thu 58, Fri 51** — matching the test set. 97 of 221
films shift by 1–9 days.

Consequence: your `anchor_lag=0` is often *not* a release window, so the regime
labels, the regime classifier (AUC 0.853), the 13.7% estimate, the weights
(1.366 / 1.172 / 0.848 / 1.078), **and the `MASE_tertimbang` objective you used
to select the feature set and blend weights** are all computed against a
mislabelled axis.

That matters concretely: in cell 14 you picked `tanpa identitas + bobot rezim`
and in cell 15 you set blend weights `{A:0, B:0, H:0.25, Z:0.75}` by minimising
weighted MASE. Those decisions were made against a mis-specified objective.

---

## 6. Error is concentrated in partial-activity pairs

Measured on the corrected test-proxy population (`analysis/exp2.py`):

| pattern | share of rows | MASE | contribution |
|---|---|---|---|
| `111` | 90.58% | 0.3197 | 0.2896 |
| `011` | 5.91% | 0.7070 | 0.0418 |
| `001` | 1.99% | **2.8776** | 0.0573 |
| `101` | 1.52% | 0.4148 | 0.0063 |

**8% of rows carry 25% of the error.** The test set has the same mix: 189 pairs
at `001` and 684 at `011` = 8.4% of pairs.

Cause: `scale = max(sum(s1,s2,s3)/3, 1)` averages over days the pair was not
screening, so it understates the real run-rate by `3/n_active`. A `001` pair
has a true `y/scale` around 1.8–3, not 0.6.

I tested predicting on a run-rate basis (`sum(s)/n_active`) instead of `scale`
— **no gain** (0.3950 vs 0.3932), because conditioning on the pattern already
captures it. So the fix is not a reparameterisation; it is giving the model
enough capacity and training signal on these rows specifically.

---

## 7. Horizon and day-of-week are nearly collinear in the test set

Because 81% of films share two D1 weekdays (`analysis/regime_check.py`):

| horizon | dominant target weekdays |
|---|---|
| D4 | Mon 45%, Sun 40% |
| D7 | Thu 45%, Wed 40% |
| D10 | Sat 45%, Fri 40% |

Your `DECAY_PRIOR` collapses this to one curve over `off`
(0.689, 0.542, 0.403, 0.272, 0.109, 0.000, 0.000), and your calibration
multipliers `CAL_MULT` are also per-`off` only — both averaged over a train
D1-dow mix that does not match test.

That single curve is hiding an enormous spread. Measured median `y/scale` by
D1-dow on train release windows (`analysis/harness.py`):

| D1 dow | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|
| **Wed** (41% of test) | **1.114** | 0.977 | 0.528 | 0.446 | 0.281 | 0.171 | 0.027 |
| **Thu** (40% of test) | 0.906 | 0.521 | 0.447 | 0.302 | 0.024 | 0.000 | 0.000 |
| **Fri** (15% of test) | 0.454 | 0.317 | 0.102 | 0.000 | 0.000 | 0.000 | 0.000 |
| your pooled `DECAY_PRIOR` | 0.689 | 0.542 | 0.403 | 0.272 | 0.109 | 0.000 | 0.000 |

The D4 ratio differs by **2.5×** between a Wednesday and a Friday release, and
your pooled prior sits *below* both Wed and Thu — the 81% of the test set that
matters most.

The mechanism is `scale` composition. A Friday release observes D1–D3 =
Fri/Sat/Sun (the weekend, so a large `scale`) and is then forecast across
Mon–Sun, giving a low ratio. A Wednesday release observes Wed/Thu/Fri (a small
`scale`) and its D4 lands on Saturday, giving a ratio above 1.

Conditioning on D1-dow is worth a lot:

| model | MASE |
|---|---|
| median by horizon | 0.4506 |
| median by horizon × D1-dow | **0.4259** |
| + pattern + scale bin | **0.4024** |

A 5.5% relative gain from the D1-dow split alone.

---

## 8. Unused data: cross-film observation of your target dates

Your cell 10 says it builds every calendar/cluster table from `train.csv` only,
deliberately refusing `test_history.csv`. But `test_history.csv` gives D1–D3 for
163 films at **different calendar dates**, so other films routinely observe
*your* target dates (`analysis/crossfilm.py`):

| | rows | share |
|---|---|---|
| target date observed by ≥1 other film | 48,058 | **66.2%** |
| exact (cluster, date) directly observed | 42,388 | **58.4%** |

By horizon — coverage is *best* exactly where your model is blindest:

| | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|
| date covered | 68.7% | 37.9% | 22.6% | 60.3% | **93.0%** | **92.3%** | **88.6%** |
| cluster-level | 50.6% | 19.6% | 16.2% | 56.6% | **90.2%** | **91.1%** | **84.4%** |

This is observed data the organisers handed you, not D4–D10 labels. It tells
you whether a cluster was even open on the target date and what demand looked
like.

**Honest result: I could not demonstrate a gain from it** (best variant
−0.0021; cruder variants *hurt* by fragmenting cells). Two reasons, and only
one is fixable:

1. My evaluator is a median lookup table. Binning a continuous factor into a
   categorical key fragments cells badly; a GBM would not have this problem.
2. More fundamentally — a *national* date factor is almost collinear with
   day-of-week, which the model already has. Its unique information is
   **calendar anomalies**, and train (Apr–Sep 2025) has almost none to learn
   from. The train-side experiment is structurally unable to show the value.

So treat §8 as a hypothesis with strong structural support but no measured
gain. Test it against §1, not against overall OOF.

---

## What I tested that did NOT work

Recording these so you don't spend time on them:

| idea | result |
|---|---|
| Restrict training to lag-0 release windows only | **worse**: 0.4024 vs 0.3932 for lags 0–24. Your multi-anchor design is right; volume wins. |
| Apply the `s3 > 0` filter to training data | neutral (0.3932 → 0.3932) |
| Predict on run-rate basis instead of `scale` | slightly worse (0.3950 vs 0.3932) |
| National date factor from cross-film data, as key | worse (+0.0067) |
| Same, as a multiplier | worse (+0.0166) |

Explicit calendar features **did** work:

| variant | all rows | ordinary | anomaly |
|---|---|---|---|
| baseline | 0.3932 | 0.3587 | 0.8485 |
| + `is_holiday(target)` + holiday count in D1–D3 | **0.3835** | 0.3516 | **0.8035** |
| + `nonwork(target)` + nonwork count in D1–D3 | **0.3831** | 0.3509 | 0.8064 |

−0.0097 overall but **−0.045 on anomaly rows**. Those rows are 7.1% of train
and 31.8% of test, so the same feature is worth roughly 4.5× more on the
leaderboard.

---

## Ranked action list

1. **Add Islamic-calendar features** — `days_since_eid` / `days_to_eid`,
   `in_ramadan`, `ramadan_day`, plus the D1–D3 composition counts. `days_since_eid`
   is genuinely learnable from April 2025. Addresses §1 and §2 (~32% of test rows).
2. **Guard the 2026-03-18 cohort** (6.08% of rows). Do not let its ratio profile
   decay to 0.13 by D10. Worth ~0.05–0.10 alone.
3. **Fix the release-date definition** (§5), then recompute the regime axis. You
   will find test is ~100% opening, so **drop the regime reweighting** and stop
   selecting features/blend weights by `MASE_tertimbang`.
4. **Condition decay and calibration on D1-dow**, not on `off` alone (§7).
5. **Give the `001`/`011` patterns dedicated treatment** (§6) — 8% of rows,
   25% of error.
6. **Try the cross-film target-date features** inside LightGBM (§8), evaluated
   specifically on anomaly rows.

Items 1–2 are the ones worth doing first: largest effect, and both are
supported by real training evidence rather than a prior.

---

## Reproducing

```bash
cd analysis
python3 verify.py        # the two proven structural rules
python3 d1_def.py        # sneak-preview corruption of anchor_lag
python3 calendar.py      # train/test calendar regime overlap
python3 straddle.py      # window-straddle exposure
python3 crossfilm.py     # cross-film coverage of target dates
python3 eid.py           # the 2026-03-18 cohort
python3 harness.py       # construction comparison          (~45 s)
python3 exp2.py          # basis + pattern breakdown        (~90 s)
python3 exp5.py          # calendar features + gap arithmetic (~60 s)
```

Pure standard library — no pandas/numpy needed, since this sandbox has no
network access to install them.

---

## The corrected notebook

`corrected_pipeline.ipynb` (32 cells) implements fixes 1–6 and writes two
submissions. `pipeline.py` is the same code as a flat script.

```bash
pip install numpy lightgbm      # pandas is not required
jupyter lab corrected_pipeline.ipynb
```

Run it from the folder holding the competition CSVs, or set `DATA_DIR`.

### What it writes

| file | contents |
|---|---|
| `submission.csv` | all fixes, including the Idulfitri guard |
| `submission_no_eid_guard.csv` | all fixes except the guard |

Submit both. The difference isolates the one bet in the pipeline that cannot be
validated offline (see below), and 4,417 rows differ between them.

### Design choice: standard library only

Every data-handling and feature-engineering step uses only the Python standard
library. `numpy` and `lightgbm` are touched in two cells. This was deliberate:
the sandbox this was developed in had no network access, so numpy/pandas could
not be installed, and a pandas pipeline would have shipped **untested**.

Instead the notebook carries a hierarchical-median fallback model, which let the
entire pipeline — windows, features, CV, calibration, diagnostics, the guard and
submission writing — be executed and verified end-to-end. The fallback scores
**0.3931** raw / **0.3896** calibrated on the proxy population, exactly
reproducing the independent measurement in §7, which confirms the notebook and
the analysis scripts agree.

So if LightGBM is missing the notebook still produces a valid submission, just a
weaker one. With LightGBM it swaps in a boosted L1 model on the same features.

**Untested surface:** the LightGBM calls themselves (`lgb.Dataset`, `lgb.train`,
`predict`) and the numpy matrix fill. Roughly 15 lines, written conventionally
with a fallback for the older `early_stopping_rounds` API. Everything else ran.

### Honest limitation on fix 2

`train.csv` starts 2025-04-01, which is already Eid+1, so **no training window
can have its observation days before Eid and its target days after it.** The
straddle the 2026-03-18 cohort needs is structurally absent — not fixable with
more features or a bigger model. Cell 14 asserts this and prints `0`.

`days_since_eid` still helps the model recognise an Eid day, but every training
example of one has its `scale` measured inside the Eid window too, so the model
never sees a pre-Eid denominator. The guard is therefore an explicit **prior**.

Cell 14 quantifies the bet instead of hiding it, scoring nine guard strategies
against seven truth scenarios. The structure it exposes:

* `uplift`, built only from train, says those target days should run at
  **1.95** (Eid Saturday) down to **0.64–0.93** (the following weekdays)
* the model predicts **1.09 → 0.11**, so horizon decay is overwhelming the
  calendar signal at D8–D10
* a floor proportional to `uplift` lifts D8–D10 from 0.10→0.55 and barely
  touches D4–D7

`calendar 0.5` came out with **negative worst-case regret** — better than no
guard under every scenario tested. The default is `calendar 0.6`, whose worst
case is +0.0031 with up to −0.0295 upside. Set `EID_GUARD_MODE = "none"` to
disable, or `"flat"` for a constant floor.

### Reading the output

Cell 11 prints MASE by calendar regime, activity pattern, horizon and D1
day-of-week. Those breakdowns, not the overall number, are what the original
notebook was missing — compare the regime rows against §1 and the pattern rows
against §6.

The proxy MASE is **not** a leaderboard estimate. The proxy population contains
no Ramadan, because train contains none, so the leaderboard will read higher.
§1 gives the reweighting arithmetic.
