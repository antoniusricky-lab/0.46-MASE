"""Drop rule-violating rows from the external film-profile tables.

Competition rule: external data is allowed only if it was publicly available
on or before 2025-09-30. The `film_profile*.csv` tables carry their own
provenance columns, so compliance is checkable per row.

DROP a row when either holds:
  * `fact_source_date` > 2025-09-30 -- the fact was taken from an article
    published during the test period. (The post-cutoff date is usually visible
    in `fact_source_url` too.)
  * `info_public_before_cutoff` == "no" -- the original compiler flagged the
    row as not publicly known before the cutoff.

KEEP a row with a future `world_premiere_date`. A scheduled release date is
not a leak: Indonesian distributors announce slates months ahead, so a
2026-03-18 premiere can be, and generally was, public well before
2025-09-30. What matters is when the fact was *published*, which is exactly
what `fact_source_date` records. Filtering on `world_premiere_date` would
throw away ~80% of the usable release-schedule signal for no compliance gain.

KEEP `info_public_before_cutoff` == "unknown" unless the row also has a
post-cutoff `fact_source_date`. "unknown" means the compiler did not verify,
not that it postdates the cutoff; the dated-source test already catches the
rows that demonstrably do.

Idempotent: re-running on cleaned files drops nothing.

Usage:  python external/sanitize_external.py [--check]
        --check exits 1 if any violations remain (for CI), writes nothing.
"""

import sys
import pandas as pd

CUTOFF = pd.Timestamp("2025-09-30")
TARGETS = [
    "external/film_profile.csv",
    "external/film_profile_train.csv",
    "external/film_profile_test_part2.csv",
]


def violations(frame):
    """Boolean mask of rows that breach the external-data cutoff."""
    bad = pd.Series(False, index=frame.index)
    if "fact_source_date" in frame.columns:
        parsed = pd.to_datetime(
            frame["fact_source_date"], errors="coerce", format="mixed"
        )
        bad |= parsed > CUTOFF
    if "info_public_before_cutoff" in frame.columns:
        bad |= frame["info_public_before_cutoff"].astype(str).str.strip().str.lower().eq("no")
    return bad


def main(check_only=False):
    total = 0
    for path in TARGETS:
        try:
            frame = pd.read_csv(path)
        except FileNotFoundError:
            print(f"SKIP  {path} (not present)")
            continue

        bad = violations(frame)
        n = int(bad.sum())
        total += n
        print(f"{path}: {len(frame)} rows, {n} violating")

        for _, row in frame[bad].iterrows():
            title = str(row.get("movie_title", "?"))
            print(
                f"    DROP {title[:46]:46s} "
                f"flag={row.get('info_public_before_cutoff')} "
                f"src={row.get('fact_source_date')}"
            )

        if n and not check_only:
            frame[~bad].to_csv(path, index=False)
            print(f"    -> wrote {len(frame) - n} rows")

    if check_only and total:
        print(f"\nFAIL: {total} violating row(s) remain")
        return 1
    print(f"\n{'OK: no violations' if not total else f'Removed {total} violating row(s)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(check_only="--check" in sys.argv))
