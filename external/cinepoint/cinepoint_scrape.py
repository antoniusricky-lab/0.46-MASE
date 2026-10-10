"""Scrape Cinepoint daily top box office (top films per day, with cumulative admissions).

Source: https://cinepoint.com/pages/tbo (public page, no login).
The table lists the #1 film per day; clicking a date opens a popup with that day's
ranking (rank, title, daily admissions, change, total admissions, showtimes).
The site API rejects direct calls, so this drives a real (visible) Chrome window.
A hidden or minimised window makes the page too slow, so keep the window on screen.

Dates after 2025-09-30 are post cutoff (trial / assumption data only).
Dates on or before 2025-09-30 (for example the previous season, Oct 2024 - Mar 2025) were public before the cutoff.

Restored from the compiled cinepoint_scrape.cpython-312.pyc (the .py was never committed).

Usage (from the repo root or leon/):
    python external/cinepoint/cinepoint_scrape.py --start 2025-10-01 --end 2026-03-31
    python external/cinepoint/cinepoint_scrape.py --start 2024-10-01 --end 2025-03-31 --out external/cinepoint/cinepoint_daily_top_prev.csv
Output: external/cinepoint/cinepoint_daily_top.csv by default (appended, resumable)
"""

import argparse
import calendar
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT_DEFAULT = Path(__file__).with_name("cinepoint_daily_top.csv")
FIELDS = ["date", "rank", "title", "daily_admission", "total_admission", "showtimes"]
URL = "https://cinepoint.com/pages/tbo?page=0&limit=100"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def pick(page, label_regex, option_text, settle=800):
    """Open the combobox whose aria-label matches and choose an option (retries if the list does not open)."""
    combos = page.get_by_role("combobox")
    for i in range(combos.count()):
        e = combos.nth(i)
        if not re.fullmatch(label_regex, e.get_attribute("aria-label") or ""):
            continue
        for _ in range(4):
            e.click()
            page.wait_for_timeout(900)
            opt = page.get_by_role("option").filter(has_text=re.compile(f"^\\s*{re.escape(option_text)}\\s*$"))
            if opt.count():
                opt.first.click()
                page.wait_for_timeout(settle)
                return True
            page.keyboard.press("Escape")
            page.wait_for_timeout(500)
        return False
    return False


def select_period(page, year, month):
    """Choose year then month in the filter and confirm via the 'Period:' header."""
    if not pick(page, r"\d{4}", str(year), settle=2500):
        return False
    if not pick(page, r"FULL_MONTH\.\d+", MONTHS[month - 1]):
        return False
    want = f"{MONTHS[month - 1][:3]} 1, {year}"
    for _ in range(40):
        page.wait_for_timeout(500)
        txt = page.locator("text=Period:").first.inner_text()
        if want not in txt:
            continue
        page.wait_for_timeout(1000)
        for size in ("100", "50", "40"):
            if pick(page, r"\d{2,3}", size, settle=1500):
                break
        return True
    return False


def to_int(s):
    digits = re.sub(r"[^0-9]", "", s or "")
    return int(digits) if digits else None


def months_between(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def load_done(path):
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {r["date"] for r in csv.DictReader(f)}


def close_popup(page):
    for _ in range(20):
        if not page.locator("div.p-dialog-mask").count():
            return
        btn = page.locator("div.p-dialog-mask button.p-dialog-header-close")
        try:
            if btn.count():
                btn.last.click(timeout=2000)
        except Exception:   # the dialog can detach while closing
            pass
        page.wait_for_timeout(250)
        if not page.locator("div.p-dialog-mask").count():
            continue
        page.keyboard.press("Escape")
        page.wait_for_timeout(250)


def popup_total(mask):
    """Total number of films of the day, printed after 'Rows per page' in the popup paginator (None if not found)."""
    m = re.search(r"Rows per page\s*\n\s*(\d+)", mask.inner_text())
    return int(m.group(1)) if m else None


def popup_show_all(page, mask):
    """Switch the popup's own 'Rows per page' dropdown (10 by default) to its largest option."""
    combo = mask.get_by_role("combobox")
    if not combo.count():
        return False
    for _ in range(3):
        combo.first.click()
        page.wait_for_timeout(700)
        sizes = [t.strip() for t in page.get_by_role("option").all_inner_texts() if t.strip().isdigit()]
        if sizes:
            biggest = max(sizes, key=int)
            page.get_by_role("option").filter(has_text=re.compile(rf"^\s*{biggest}\s*$")).first.click()
            page.wait_for_timeout(1000)
            return True
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
    return False


def read_page_rows(mask, iso):
    rows = []
    for i in range(mask.locator("tbody tr").count()):
        tds = mask.locator("tbody tr").nth(i).locator("td")
        if tds.count() < 6:
            continue
        cell = [tds.nth(k).inner_text().strip() for k in range(6)]
        rows.append({
            "date": iso,
            "rank": to_int(cell[0]),
            "title": cell[1].split("\n")[0].strip(),
            "daily_admission": to_int(cell[2]),
            "total_admission": to_int(cell[4]),
            "showtimes": to_int(cell[5]),
        })
    return rows


def read_popup_rows(page):
    """Rows of the open popup. Returns (iso_date, rows, total shown by the popup)."""
    mask = page.locator("div.p-dialog-mask").last
    header = mask.inner_text().strip().split("\n")[0].strip()
    iso = datetime.strptime(header, "%b %d, %Y").strftime("%Y-%m-%d")
    popup_show_all(page, mask)
    total = popup_total(mask)
    want = min(total, 100) if total else None
    # wait until the table holds every film of the day (or stops changing for 1.5 s)
    last, stable = -1, 0
    for _ in range(80):
        n = mask.locator("tbody tr").count()
        if want and n >= want:
            break
        stable = stable + 1 if n == last and n > 0 else 0
        if stable >= 15:
            break
        last = n
        page.wait_for_timeout(100)
    rows = read_page_rows(mask, iso)
    # fallback: further pages if the dropdown did not show everything (wait for the first rank to change after each click)
    for _ in range(10):
        nxt = mask.locator("button.p-paginator-next")
        if not nxt.count() or nxt.first.is_disabled():
            break
        first_rank = rows[-1]["rank"] if rows else None
        nxt.first.click()
        for _ in range(50):
            page.wait_for_timeout(100)
            cur = read_page_rows(mask, iso)
            if cur and cur[0]["rank"] != first_rank and cur[0]["rank"] not in {r["rank"] for r in rows}:
                break
        rows += [r for r in read_page_rows(mask, iso) if r["rank"] not in {x["rank"] for x in rows}]
    seen, uniq = set(), []
    for r in rows:
        if r["rank"] not in seen:
            seen.add(r["rank"])
            uniq.append(r)
    return iso, uniq, total


def scrape_cinepoint(start, end, out=OUT_DEFAULT):
    """Scrape every date from start to end (YYYY-MM-DD) into the CSV `out`; dates already in it are skipped."""
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    out = Path(out)
    done = load_done(out)
    new_file = not out.exists()

    with sync_playwright() as p, out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        browser = p.chromium.launch(channel="chrome", headless=False)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_selector("tbody a", timeout=30000)
        page.wait_for_timeout(1500)
        for y, m in months_between(start, end):
            ok = select_period(page, y, m)
            if not ok:   # the filter sometimes stops responding after a year change: reload the page and try once more
                page.goto(URL, wait_until="domcontentloaded")
                page.wait_for_selector("tbody a", timeout=30000)
                page.wait_for_timeout(1500)
                ok = select_period(page, y, m)
            if not ok:
                print(f"{y}-{m:02d}: could not select period, skipped", file=sys.stderr)
                continue
            # the table reloads at 100 rows per page after the period change: wait until every day of the month is listed
            days_in_month = calendar.monthrange(y, m)[1]
            for attempt in range(4):
                for _ in range(40):
                    if page.locator("tbody a").count() >= days_in_month:
                        break
                    page.wait_for_timeout(500)
                if page.locator("tbody a").count() >= days_in_month:
                    break
                pick(page, r"\d{2,3}", "100", settle=2500)
            if page.locator("tbody a").count() < days_in_month:
                print(f"{y}-{m:02d}: only {page.locator('tbody a').count()} of {days_in_month} dates listed", file=sys.stderr)
            labels = []
            for t in page.locator("tbody a").all_inner_texts():
                t = t.strip()
                if re.search(r"\d{4}", t) and t not in labels:
                    labels.append(t)
            for label in labels:
                d = datetime.strptime(label, "%b %d, %Y")
                iso = d.strftime("%Y-%m-%d")
                if not (start <= d <= end) or iso in done:
                    continue
                opened = False
                for _ in range(3):
                    close_popup(page)
                    page.locator("tbody a", has_text=label).first.click()
                    try:
                        page.wait_for_selector("div.p-dialog-mask tbody tr", timeout=8000)
                        opened = True
                        break
                    except Exception:
                        page.wait_for_timeout(1000)
                if not opened:
                    print(f"{iso}: popup did not open", file=sys.stderr)
                    continue
                got, rows, total = read_popup_rows(page)
                if got != iso:
                    print(f"{iso}: popup showed {got}, skipped", file=sys.stderr)
                    close_popup(page)
                    continue
                if total and len(rows) < min(total, 100):
                    print(f"{iso}: only {len(rows)} of {total} rows read, skipped (retry later)", file=sys.stderr)
                    close_popup(page)
                    continue
                writer.writerows(rows)
                f.flush()
                done.add(iso)
                print(f"{iso}: {len(rows)} rows, top = {rows[0]['title'] if rows else '-'}", flush=True)
                close_popup(page)
        browser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    scrape_cinepoint(args.start, args.end, args.out)


if __name__ == "__main__":
    main()
