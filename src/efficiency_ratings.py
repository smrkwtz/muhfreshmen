"""
Collects pre-tournament adjusted efficiency ratings from Barttorvik (T-Rank).

Barttorvik is a free site with ratings methodology nearly identical to KenPom.
If you have a KenPom subscription, see the KenPom section at the bottom for
how to swap in that data instead.

Output columns (per team per year):
  year, team, team_id_bart, adjEM, adjO, adjD, adjT

adjEM  = Adjusted Efficiency Margin (points per 100 possessions above average)
adjO   = Adjusted Offensive Efficiency (points scored per 100 possessions)
adjD   = Adjusted Defensive Efficiency (points allowed per 100 possessions)
adjT   = Adjusted Tempo (possessions per 40 minutes)
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from constants import TOURNAMENT_START_DATES, TOURNAMENT_YEARS

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research/academic project)"}
REQUEST_DELAY = 4

OUTPUT_PATH = Path("data/raw/efficiency_ratings.csv")

# Barttorvik table column order (as of 2024 - may drift slightly)
# The HTML table on the trank.php page has labelled <th> headers; we match by name.
BART_COL_MAP = {
    "team": "team",
    "conf": "conf",
    "adjoe": "adjO",     # Adjusted Offensive Efficiency
    "adjde": "adjD",     # Adjusted Defensive Efficiency
    "barthag": "barthag",
    "efg%": "eFG_pct",
    "adjt": "adjT",      # Adjusted Tempo
}


def _build_barttorvik_url(year: int) -> str:
    """
    Build URL for Barttorvik team ratings through the day before the tournament.
    Setting end= to the tournament start date excludes tournament games from
    the rating calculation, giving us a clean pre-tournament snapshot.
    """
    end_date = TOURNAMENT_START_DATES.get(year)
    if not end_date:
        raise ValueError(f"No tournament start date defined for {year}")

    # Season start: November 1 of the prior calendar year
    season_start = f"{year - 1}1101"

    params = (
        f"year={year}"
        f"&sort="
        f"&lastx=0"
        f"&hteam="
        f"&t2value="
        f"&conlimit=All"
        f"&state=All"
        f"&begin={season_start}"
        f"&end={end_date}"
        f"&top=0"
        f"&link=y"
        f"&q="
        f"&busted=0"
        f"&type=All"
        f"&mingames=0"
    )
    return f"https://barttorvik.com/trank.php?{params}#"


def _parse_barttorvik_table(html: str, year: int) -> list[dict]:
    """Parse the main ratings table from a Barttorvik trank.php page."""
    soup = BeautifulSoup(html, "lxml")

    # The ratings are in a <table id="t-rank-table"> or similar
    table = soup.find("table", id=re.compile(r"t.rank", re.I))
    if table is None:
        # Fallback: find the first large table on the page
        tables = soup.find_all("table")
        table = max(tables, key=lambda t: len(t.find_all("tr")), default=None)
    if table is None:
        return []

    # Parse header row to map column positions
    header_row = table.find("thead")
    if header_row is None:
        header_row = table.find("tr")
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]

    # Find column indices we care about
    def col_idx(name: str) -> int | None:
        try:
            return headers.index(name)
        except ValueError:
            return None

    team_col = col_idx("team")
    adjo_col = col_idx("adjoe")
    adjd_col = col_idx("adjde")
    adjt_col = col_idx("adjt")

    # adjEM is not always a direct column; compute as adjO - adjD
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        def cell(idx):
            if idx is None or idx >= len(cells):
                return None
            return cells[idx].get_text(strip=True)

        team_name = cell(team_col)
        if not team_name:
            continue

        # Extract school id from the team link if present
        team_a = cells[team_col].find("a") if team_col is not None else None
        school_id = None
        if team_a and team_a.get("href"):
            m = re.search(r"team=([^&]+)", team_a["href"])
            if m:
                school_id = m.group(1)

        try:
            adjo = float(cell(adjo_col)) if adjo_col is not None else None
            adjd = float(cell(adjd_col)) if adjd_col is not None else None
            adjt = float(cell(adjt_col)) if adjt_col is not None else None
            adjEM = round(adjo - adjd, 2) if (adjo is not None and adjd is not None) else None
        except (ValueError, TypeError):
            continue

        rows.append({
            "year": year,
            "team": team_name,
            "team_id_bart": school_id,
            "adjO": adjo,
            "adjD": adjd,
            "adjT": adjt,
            "adjEM": adjEM,
        })

    return rows


def fetch_ratings_for_year(year: int, session: requests.Session) -> list[dict]:
    url = _build_barttorvik_url(year)
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return _parse_barttorvik_table(resp.text, year)


def collect_all_efficiency_ratings(
    years: list[int] = TOURNAMENT_YEARS,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    session = requests.Session()

    for year in years:
        print(f"Fetching {year} efficiency ratings...", end=" ", flush=True)
        try:
            rows = fetch_ratings_for_year(year, session)
            all_rows.extend(rows)
            print(f"{len(rows)} teams")
        except requests.HTTPError as e:
            print(f"HTTP {e.response.status_code} - skipping")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} team-seasons → {output_path}")
    return df


# ---------------------------------------------------------------------------
# KenPom alternative
# ---------------------------------------------------------------------------
# If you have a KenPom subscription, you can export data from
#   https://kenpom.com/index.php  (use the "Export" button)
# for each year and save the CSVs as  data/raw/kenpom_{year}.csv
# Then call load_kenpom_exports() instead of collect_all_efficiency_ratings().
#
# KenPom CSV columns of interest:
#   TeamName, AdjEM, AdjO, AdjD, AdjT (column names may vary slightly)

def load_kenpom_exports(
    data_dir: Path = Path("data/raw"),
    years: list[int] = TOURNAMENT_YEARS,
) -> pd.DataFrame:
    """Load pre-downloaded KenPom CSV exports and return a unified DataFrame."""
    frames = []
    for year in years:
        path = data_dir / f"kenpom_{year}.csv"
        if not path.exists():
            print(f"  Missing {path} - skipping {year}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        # Normalise column names - KenPom export headers vary slightly
        rename = {}
        for col in df.columns:
            low = col.lower()
            if "team" in low:
                rename[col] = "team"
            elif low in ("adjem", "adj. em", "adj em"):
                rename[col] = "adjEM"
            elif low in ("adjo", "adj. o", "adj. oe", "adjoe"):
                rename[col] = "adjO"
            elif low in ("adjd", "adj. d", "adj. de", "adjde"):
                rename[col] = "adjD"
            elif low in ("adjt", "adj. t", "adj. tempo"):
                rename[col] = "adjT"
        df = df.rename(columns=rename)
        df["year"] = year
        frames.append(df[["year", "team", "adjEM", "adjO", "adjD", "adjT"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    collect_all_efficiency_ratings()
