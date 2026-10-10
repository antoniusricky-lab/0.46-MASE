# External data — provenance

The competition permits external data **only if publicly available on or before 2025-09-30**, and
the committee may require source, publisher, publication or archive date, access time, data
version and how each source was used. Everything here is recorded for that purpose.

Obtained from the public repository `github.com/LeonArif/JOINTS` (`external/`), which itself
records a source URL and publication date per row. **Verify independently before submitting a
finalist notebook** — the rule tests when a source became *available*, which is not the same as
the dates the data *describes*.

## Included

| file | rows | publication dates | post-cutoff |
|------|------|-------------------|-------------|
| `cinepoint/cinepoint_daily_top_prev.csv` | 5,211 | data 2023-09-20 → 2025-03-31 | **0%** |
| `cinepoint/cinepoint_daily_top_train.csv` | 2,940 | data 2025-04-01 → 2025-09-30 | **0%** |
| `lebaran_admissions.csv` | 37 | 2024-04-12 → 2025-04-15 | 0% |
| `ramadan.csv` | 30 | 2025-06-25 | 0% |
| `filmindonesia_admissions.csv` | 285 | 2025-05-14 → 2025-09-14 | 0% |
| `audience_statistics.csv` | 8 | 2013-07-12 → 2025-06-19 | 0% |
| `cuti_bersama.csv` | 19 | 2024-10-14 → 2025-09-19 | 0% |
| `extra_national_holidays.csv` | 2 | 2024-10-14 | 0% |
| `school_calendar_regional.csv` | 125 | 2024-06-05 → 2025-08-26 | 0% |
| `school_holidays.csv` | 61 | 2024-01-01 | 0% |
| `season_statements.csv`, `season_trajectories.csv` | 17 / 193 | → 2025-04-26 | 0% |
| `wiki_pageviews_daily.csv` | 17,794 | 2025-01-02 → 2025-09-30 | 0% |
| `film_profile_test_part1.csv` | 285 | → 2025-09-14 | 0% |
| `film_profile.csv` | 392 | → 2025-09-30 | 0% (filtered, below) |
| `film_profile_train.csv` | 74 | → 2025-09-30 | 0% (filtered, below) |
| `film_profile_test_part2.csv` | 81 | → 2025-09-30 | 0% (filtered, below) |
| `city_province.csv` | 73 | reference table | — |

### Filtering applied

`external/sanitize_external.py` has been run over the film-profile tables. It drops a row when
`fact_source_date > 2025-09-30` (the fact was taken from an article published *during* the test
period — the date is usually visible in `fact_source_url`) or when the source repo's own
`info_public_before_cutoff` flag reads `no`. 16 rows removed:

| file | before | after | dropped |
|------|--------|-------|---------|
| `film_profile.csv` | 400 | 392 | 8 |
| `film_profile_train.csv` | 81 | 74 | 7 |
| `film_profile_test_part2.csv` | 82 | 81 | 1 |

Titles dropped: ALAS ROBAN (flagged `no`), MALAM 3 YASINAN (2025-11-16), OZORA (2025-10-27),
LIFT (2025-12-25), MENGEJAR RESTU (2025-11-17), RAJAH (2026-01-27), TANEUH KALAKNAT (2026-02),
MENUJU PELAMINAN (2025-10-15).

The script is idempotent; `python external/sanitize_external.py --check` exits non-zero if any
violation reappears, so it can gate CI or a pre-submission check.

**Two judgment calls, stated explicitly for the committee:**

1. **A future `world_premiere_date` is kept.** 82 rows in `film_profile.csv` carry premiere dates
   after the cutoff (up to 2026-03-18). A scheduled release date is not a leak — Indonesian
   distributors announce slates months ahead, so a March 2026 premiere was public well before
   2025-09-30. The rule tests when a fact was *published*, which is what `fact_source_date`
   records. Filtering on `world_premiere_date` would discard most of the release-schedule signal
   for no compliance gain.
2. **`info_public_before_cutoff == "unknown"` is kept** (66 rows) unless the row also fails the
   dated-source test. "unknown" means the compiler did not verify, not that the fact postdates
   the cutoff. If the committee requires affirmative proof rather than absence of contrary
   evidence, tighten the mask to `!= "yes"` — that would drop those 66 rows as well.

## Deliberately excluded

`cinepoint/cinepoint_daily_top.csv` — 3,094 rows covering **2025-10-01 → 2026-03-31**, i.e. daily
national admissions for the **test period**, 100% post-cutoff. That is the target variable
supplied by a third party and the external-data rule forbids it. The source repo's own scraper
docstring calls these dates "post cutoff (trial / assumption data only)" and its logs keep a
separate `main_legit.ipynb` pipeline without them.

`cinepoint_scrape.py` is kept only so the pre-cutoff portion can be re-scraped under your own
documented access, and so the provenance chain is inspectable. **Do not run it for dates after
2025-09-30.**

## Why the Cinepoint prior season matters

`cinepoint_daily_top_prev.csv` spans two complete **Oct–Mar** seasons — the same calendar window
as the test period, which `train.csv` (Apr–Sep 2025) structurally cannot contain. Day-of-week
adjusted national factors, baseline ordinary Oct–Nov 2024:

| period | factor |
|--------|--------|
| Dec 20 – Jan 4 (school break) | 1.335 |
| Jan 5 – Feb 14 (ordinary) | 0.828 |
| Ramadan wk1 / wk2 / wk3 / wk4 | 0.415 / 0.358 / 0.331 / 0.609 |
| pre-Eid (Mar 29–30) | 0.415 |
| Eid+0 … Eid+8 | 2.29, 3.39, 3.85, 3.12, 2.62, 2.19, 2.42, 5.26, 4.96 |
| **late-Ramadan → Eid-week** | **6.23×** |

These are the first measured priors for the three calendar cohorts that `train.csv` cannot teach,
and they replace the blind leaderboard probes used until now.
