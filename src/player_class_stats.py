"""
Scrapes per-player season stats (with class year) from sports-reference.com
for every team that appeared in a given year's NCAA tournament.

For each team-season we compute:
  - fr_min_share  : fraction of team minutes played by freshmen
  - fr_pts_share  : fraction of team points scored by freshmen

Output columns:
  year, team_id, team_name, fr_min_share, fr_pts_share,
  fr_min_raw, fr_pts_raw, total_min, total_pts, n_fr_players
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from constants import TOURNAMENT_YEARS

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research/academic project)"}
REQUEST_DELAY = 4  # seconds between requests

OUTPUT_PATH = Path("data/raw/player_class_stats.csv")

# Class labels used by sports-reference
FRESHMAN_LABELS = {"fr", "freshman"}
ALL_CLASS_LABELS = {"fr", "so", "jr", "sr", "freshman", "sophomore", "junior", "senior"}


def _get_per_game_table(soup: BeautifulSoup) -> BeautifulSoup | None:
    """
    Find the per-game stats table on a team season page.
    sports-reference sometimes hides tables inside HTML comments (due to ad
    injection); this function checks both visible DOM and commented-out HTML.
    """
    table = soup.find("table", id="per_game")
    if table:
        return table

    # Check inside HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "per_game" in comment:
            inner = BeautifulSoup(comment, "lxml")
            table = inner.find("table", id="per_game")
            if table:
                return table

    # Fallback: any table containing a "Class" column
    for tbl in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
        if "class" in headers and any(h in headers for h in ("mp", "g", "pts")):
            return tbl

    return None


def _parse_team_page(html: str, year: int, team_id: str) -> dict | None:
    """
    Parse a team season page and return the freshman-reliance metrics dict,
    or None if the page could not be parsed.
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract team name from page title
    title = soup.find("h1", {"itemprop": "name"})
    team_name = title.get_text(strip=True) if title else team_id

    table = _get_per_game_table(soup)
    if table is None:
        return None

    # Map header names → column indices
    header_cells = table.find("thead").find_all(["th", "td"])
    headers = [c.get("data-stat", c.get_text(strip=True)).lower() for c in header_cells]

    def col(name: str) -> int | None:
        # Try data-stat first, then text
        for attr in (name, name.replace("_", "")):
            try:
                return headers.index(attr)
            except ValueError:
                pass
        return None

    class_col = col("class") or col("yr") or col("class_")
    mp_col = col("mp")          # minutes per game
    g_col = col("g")            # games played
    pts_col = col("pts")        # points per game

    if class_col is None:
        return None

    # We need either (mp + g) to get total minutes, or just relative shares
    total_min = 0.0
    fr_min = 0.0
    total_pts = 0.0
    fr_pts = 0.0
    n_freshmen = 0

    for row in table.find("tbody").find_all("tr"):
        # Skip header/separator rows
        if "thead" in (row.get("class") or []):
            continue
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        def val(idx):
            if idx is None or idx >= len(cells):
                return ""
            return cells[idx].get_text(strip=True)

        class_val = val(class_col).lower().strip()
        if not class_val or class_val not in ALL_CLASS_LABELS:
            continue

        try:
            g = float(val(g_col)) if g_col is not None else 0
            mp = float(val(mp_col)) if mp_col is not None else 0
            pts = float(val(pts_col)) if pts_col is not None else 0
        except ValueError:
            continue

        # Total minutes for player across season = MP per game × games
        player_min = mp * g
        player_pts = pts * g

        total_min += player_min
        total_pts += player_pts

        if class_val in FRESHMAN_LABELS:
            fr_min += player_min
            fr_pts += player_pts
            n_freshmen += 1

    if total_min == 0 and total_pts == 0:
        return None

    return {
        "year": year,
        "team_id": team_id,
        "team_name": team_name,
        "fr_min_raw": round(fr_min, 1),
        "fr_pts_raw": round(fr_pts, 1),
        "total_min": round(total_min, 1),
        "total_pts": round(total_pts, 1),
        "fr_min_share": round(fr_min / total_min, 4) if total_min > 0 else None,
        "fr_pts_share": round(fr_pts / total_pts, 4) if total_pts > 0 else None,
        "n_fr_players": n_freshmen,
    }


def fetch_team_stats(
    team_id: str,
    year: int,
    session: requests.Session,
) -> dict | None:
    # Sports-reference added /men/ to school URLs; try new format first.
    for url in [
        f"https://www.sports-reference.com/cbb/schools/{team_id}/men/{year}.html",
        f"https://www.sports-reference.com/cbb/schools/{team_id}/{year}.html",
    ]:
        resp = session.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        result = _parse_team_page(resp.text, year, team_id)
        if result:
            return result
    return None


def collect_player_class_stats(
    tournament_results_path: Path = Path("data/raw/tournament_results.csv"),
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Load the tournament results to get the list of teams, then fetch each
    team's season page to extract freshman reliance metrics.
    """
    if not tournament_results_path.exists():
        raise FileNotFoundError(
            f"Run tournament_results.py first to create {tournament_results_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(tournament_results_path)

    # Build a deduplicated list of (year, team_id, team_name) for all tourney teams
    team_cols_1 = results[["year", "team1_id", "team1"]].rename(
        columns={"team1_id": "team_id", "team1": "team_name"}
    )
    team_cols_2 = results[["year", "team2_id", "team2"]].rename(
        columns={"team2_id": "team_id", "team2": "team_name"}
    )
    teams = (
        pd.concat([team_cols_1, team_cols_2])
        .dropna(subset=["team_id"])
        .drop_duplicates(subset=["year", "team_id"])
        .sort_values(["year", "team_id"])
        .reset_index(drop=True)
    )

    print(f"Fetching player class stats for {len(teams)} team-seasons...")
    session = requests.Session()
    all_rows = []

    for _, row in teams.iterrows():
        year, team_id, team_name = int(row.year), row.team_id, row.team_name
        print(f"  {year} {team_name} ({team_id})", end=" ... ", flush=True)
        try:
            data = fetch_team_stats(team_id, year, session)
            if data:
                all_rows.append(data)
                print(f"fr_min={data['fr_min_share']:.1%}  fr_pts={data['fr_pts_share']:.1%}")
            else:
                print("no data")
        except requests.HTTPError as e:
            print(f"HTTP {e.response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} team-seasons → {output_path}")
    return df


if __name__ == "__main__":
    collect_player_class_stats()
