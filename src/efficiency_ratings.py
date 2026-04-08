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


def _build_barttorvik_json_url(year: int) -> str:
    """
    Barttorvik loads its ratings table from a JSON endpoint, not from the
    HTML of trank.php (which is JS-rendered). The getteams.php endpoint
    accepts the same date-range parameters and returns a raw JavaScript array.
    """
    end_date = TOURNAMENT_START_DATES.get(year)
    if not end_date:
        raise ValueError(f"No tournament start date defined for {year}")
    season_start = f"{year - 1}1101"
    params = (
        f"year={year}"
        f"&conf=All"
        f"&state=All"
        f"&begin={season_start}"
        f"&end={end_date}"
        f"&top=0"
        f"&type=All"
        f"&links=y"
        f"&sortby="
        f"&mingames=0"
    )
    return f"https://barttorvik.com/getteams.php?{params}"


def _parse_barttorvik_json(data: list, year: int) -> list[dict]:
    """
    Parse the array returned by getteams.php.

    Each element is an array. The known column layout (may shift slightly):
      [0]  rank
      [1]  team name
      [2]  conference
      [3]  adjOE rank
      [4]  adjOE  ← Adjusted Offensive Efficiency
      [5]  adjDE rank
      [6]  adjDE  ← Adjusted Defensive Efficiency
      [7]  barthag (Pythagorean win %)
      [8]  record (string, e.g. "32-5")
      [9]  adjT rank
      [10] adjT   ← Adjusted Tempo
      ...

    We identify columns dynamically by scanning the first row for floats in
    the expected ranges (adjOE ~100-130, adjDE ~85-115, adjT ~60-80), which
    is more robust than relying on fixed indices.
    """
    if not data:
        return []

    # Detect column positions from the first data row
    first = data[0]

    def find_col(lo, hi, exclude=()) -> int | None:
        for i, v in enumerate(first):
            if i in exclude:
                continue
            try:
                f = float(v)
                if lo <= f <= hi:
                    return i
            except (TypeError, ValueError):
                pass
        return None

    team_col = 1   # always position 1
    adjo_col = find_col(85, 145)
    adjd_col = find_col(85, 130, exclude={adjo_col} if adjo_col is not None else set())
    adjt_col = find_col(55, 85,  exclude={adjo_col, adjd_col} - {None})

    rows = []
    for entry in data:
        try:
            team = str(entry[team_col]).strip()
            if not team:
                continue
            adjo = float(entry[adjo_col]) if adjo_col is not None else None
            adjd = float(entry[adjd_col]) if adjd_col is not None else None
            adjt = float(entry[adjt_col]) if adjt_col is not None else None
            adjEM = round(adjo - adjd, 2) if (adjo and adjd) else None
        except (IndexError, TypeError, ValueError):
            continue

        rows.append({
            "year": year,
            "team": team,
            "adjO": adjo,
            "adjD": adjd,
            "adjT": adjt,
            "adjEM": adjEM,
        })

    return rows


def fetch_ratings_for_year(year: int, session: requests.Session) -> list[dict]:
    import json

    url = _build_barttorvik_json_url(year)
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # Response is either JSON or a JS array literal — strip any variable prefix
    text = resp.text.strip()
    if text.startswith("var ") or text.startswith("let ") or text.startswith("const "):
        text = text.split("=", 1)[1].strip().rstrip(";")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [warn] JSON parse failed: {e}. First 200 chars: {text[:200]}")
        return []

    if not isinstance(data, list):
        print(f"  [warn] Unexpected response type: {type(data)}")
        return []

    return _parse_barttorvik_json(data, year)


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
