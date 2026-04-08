"""
Collects pre-tournament team efficiency ratings from sports-reference.com.

Uses the Advanced School Stats page for each season, which has:
  ORtg  - Offensive Rating (points scored per 100 possessions, raw)
  DRtg  - Defensive Rating (points allowed per 100 possessions, raw)
  Pace  - Possessions per 40 minutes

These are unadjusted for opponent quality (unlike KenPom's AdjO/AdjD), but
the same spread formula applies:
  expected_margin = (netRtg_A - netRtg_B) * avg_pace / 100

where netRtg = ORtg - DRtg.

If you have a KenPom subscription, call load_kenpom_exports() instead --
see the bottom of this file.

Output columns: year, team, team_id, ORtg, DRtg, Pace, netRtg
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from constants import TOURNAMENT_YEARS

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research/academic project)"}
REQUEST_DELAY = 4

OUTPUT_PATH = Path("data/raw/efficiency_ratings.csv")

URL_TEMPLATES = [
    "https://www.sports-reference.com/cbb/seasons/men/{year}-advanced-school-stats.html",
    "https://www.sports-reference.com/cbb/seasons/{year}-advanced-school-stats.html",
]


def _find_table(soup: BeautifulSoup, table_id: str) -> BeautifulSoup | None:
    """Find a table in the DOM or inside HTML comments."""
    table = soup.find("table", id=table_id)
    if table:
        return table
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id in comment:
            inner = BeautifulSoup(comment, "lxml")
            table = inner.find("table", id=table_id)
            if table:
                return table
    return None


def _parse_advanced_stats(soup: BeautifulSoup, year: int) -> list[dict]:
    """Parse ORtg, DRtg, Pace from the advanced school stats table."""
    # Table ids used across different years
    for table_id in ("adv_school_stats", "advanced-school-stats", "adv_stats",
                     "school_stats", "basic_school_stats"):
        table = _find_table(soup, table_id)
        if table:
            break

    if table is None:
        # Last resort: largest table on the page
        all_tables = soup.find_all("table")
        # Also check inside all comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            inner = BeautifulSoup(comment, "lxml")
            all_tables.extend(inner.find_all("table"))
        table = max(all_tables, key=lambda t: len(t.find_all("tr")), default=None)

    if table is None:
        return []

    # Map data-stat → column index from the last header row
    thead = table.find("thead")
    if not thead:
        return []
    header_rows = thead.find_all("tr")
    last_header = header_rows[-1]
    headers = [th.get("data-stat", th.get_text(strip=True)).lower()
               for th in last_header.find_all(["th", "td"])]

    def col(*names):
        for n in names:
            try:
                return headers.index(n)
            except ValueError:
                pass
        return None

    school_col = col("school_name", "school", "team")
    ortg_col   = col("off_rtg", "o_rtg", "ortg", "adj_oe", "adjoe")
    drtg_col   = col("def_rtg", "d_rtg", "drtg", "adj_de", "adjde")
    pace_col   = col("pace", "adj_tempo", "adjt", "tempo")

    if school_col is None or ortg_col is None or drtg_col is None:
        print(f"  [warn] Could not find rating columns. Headers: {headers[:20]}")
        return []

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = tr.find_all(["td", "th"])

        def val(idx):
            if idx is None or idx >= len(cells):
                return None
            return cells[idx].get_text(strip=True)

        school_raw = val(school_col)
        if not school_raw:
            continue

        # Extract school_id from link if available
        school_a = cells[school_col].find("a") if school_col < len(cells) else None
        school_id = None
        if school_a:
            m = re.search(r"/cbb/schools/([^/]+)/", school_a.get("href", ""))
            if m:
                school_id = m.group(1)

        # Strip trailing * (NCAA tournament participant marker)
        school = school_raw.rstrip("*").strip()

        try:
            ortg = float(val(ortg_col))
            drtg = float(val(drtg_col))
            pace = float(val(pace_col)) if pace_col is not None and val(pace_col) else None
        except (TypeError, ValueError):
            continue

        rows.append({
            "year": year,
            "team": school,
            "team_id": school_id,
            "ORtg": ortg,
            "DRtg": drtg,
            "Pace": pace,
            "netRtg": round(ortg - drtg, 2),
        })

    return rows


def fetch_ratings_for_year(year: int, session: requests.Session) -> list[dict]:
    for template in URL_TEMPLATES:
        url = template.format(year=year)
        resp = session.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            rows = _parse_advanced_stats(soup, year)
            if rows:
                return rows
    return []


BARTTORVIK_DIR = Path("data/manual/barttorvik")


def load_barttorvik_exports(
    data_dir: Path = BARTTORVIK_DIR,
    years: list[int] = TOURNAMENT_YEARS,
) -> pd.DataFrame:
    """
    Load manually exported Barttorvik CSVs from data/manual/barttorvik/.
    Each file should be named barttorvik_YYYY.csv and exported with the
    end-date filter set to the day before the tournament (see README.txt).

    Barttorvik CSV columns (typical export):
      Team, Conf, Rec, AdjOE, AdjDE, Barthag, EFG%, EFGD%, TOR, TORD,
      ORB, DRB, FTR, FTRD, 2PTM%, 2PTMD%, 3PTM%, 3PTMD%, AdjT, WAB
    """
    frames = []
    for year in years:
        path = data_dir / f"barttorvik_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        # Normalise column names — Barttorvik export headers are consistent
        # but may have slight variations
        rename = {}
        for col in df.columns:
            low = col.lower().strip()
            if low in ("team", "team name"):
                rename[col] = "team"
            elif low in ("adjoe", "adj oe", "adjo", "adj_oe"):
                rename[col] = "ORtg"
            elif low in ("adjde", "adj de", "adjd", "adj_de"):
                rename[col] = "DRtg"
            elif low in ("adjt", "adj t", "adj_t", "tempo"):
                rename[col] = "Pace"
        df = df.rename(columns=rename)
        df["year"] = year

        required = {"team", "ORtg", "DRtg"}
        if not required.issubset(df.columns):
            print(f"  [warn] {path.name} missing columns. Found: {list(df.columns)}")
            continue

        df["netRtg"] = df["ORtg"] - df["DRtg"]
        keep = ["year", "team", "ORtg", "DRtg", "netRtg"]
        if "Pace" in df.columns:
            keep.append("Pace")
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(result)} team-seasons from {len(frames)} Barttorvik export files")
    return result


def collect_all_efficiency_ratings(
    years: list[int] = TOURNAMENT_YEARS,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer manually exported Barttorvik data (better quality — opponent-adjusted)
    bart_df = load_barttorvik_exports(years=years)
    if not bart_df.empty:
        years_covered = set(bart_df["year"].unique())
        missing_years = [y for y in years if y not in years_covered]
        if missing_years:
            print(f"Barttorvik exports missing for {missing_years} — scraping sports-reference for those years")
        else:
            bart_df.to_csv(output_path, index=False)
            print(f"Saved {len(bart_df)} team-seasons (Barttorvik exports) → {output_path}")
            return bart_df
    else:
        missing_years = list(years)

    # Fall back to scraping sports-reference for years without Barttorvik exports
    all_rows = list(bart_df.to_dict("records")) if not bart_df.empty else []
    session = requests.Session()

    for year in missing_years:
        print(f"Fetching {year} efficiency ratings (sports-reference)...", end=" ", flush=True)
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
# Export CSVs from https://kenpom.com for each year, save as
# data/raw/kenpom_{year}.csv, then call load_kenpom_exports() in
# build_dataset.py instead of collect_all_efficiency_ratings().

def load_kenpom_exports(
    data_dir: Path = Path("data/raw"),
    years: list[int] = TOURNAMENT_YEARS,
) -> pd.DataFrame:
    frames = []
    for year in years:
        path = data_dir / f"kenpom_{year}.csv"
        if not path.exists():
            print(f"  Missing {path}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        rename = {}
        for col in df.columns:
            low = col.lower()
            if "team" in low:                        rename[col] = "team"
            elif low in ("adjem", "adj. em", "adj em"):  rename[col] = "netRtg"
            elif low in ("adjo", "adj. o", "adjoe"):     rename[col] = "ORtg"
            elif low in ("adjd", "adj. d", "adjde"):     rename[col] = "DRtg"
            elif low in ("adjt", "adj. t"):              rename[col] = "Pace"
        df = df.rename(columns=rename)
        df["year"] = year
        for c in ("ORtg", "DRtg", "Pace", "netRtg"):
            if c not in df.columns:
                df[c] = None
        if "netRtg" not in df.columns and "ORtg" in df.columns and "DRtg" in df.columns:
            df["netRtg"] = df["ORtg"] - df["DRtg"]
        frames.append(df[["year", "team", "ORtg", "DRtg", "Pace", "netRtg"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    collect_all_efficiency_ratings()
