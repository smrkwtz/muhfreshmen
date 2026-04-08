"""
Scrapes NCAA tournament game results from sports-reference.com for 2010-2026.

Uses the all_tourney summary table on each season's tournament page rather than
parsing the visual bracket HTML, which is fragile and uses per-region divs.

Output columns:
  year, round, round_order,
  team1, team1_id, team1_seed, team1_score,
  team2, team2_id, team2_seed, team2_score,
  winner

team1 is always the lower seed (favored). For equal seeds (First Four),
team1 is the winner.
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from constants import ROUND_ORDER, TOURNAMENT_YEARS

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research/academic project)"}
REQUEST_DELAY = 4

OUTPUT_PATH = Path("data/raw/tournament_results.csv")

# Sports-reference changed the URL format to include /men/ at some point.
# We try the new URL first, then fall back to the old one.
URL_TEMPLATES = [
    "https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html",
    "https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html",
]

ROUND_ALIASES = {
    "first four": "First Four",
    "opening round": "First Four",
    "first round": "First Round",
    "second round": "Second Round",
    "third round": "Second Round",
    "sweet 16": "Sweet 16",
    "sweet sixteen": "Sweet 16",
    "regional semifinal": "Sweet 16",
    "elite 8": "Elite Eight",
    "elite eight": "Elite Eight",
    "regional final": "Elite Eight",
    "final four": "Final Four",
    "national semifinal": "Final Four",
    "championship": "Championship",
    "national championship": "Championship",
    "semifinal": "Final Four",
    "final": "Championship",
}


def _normalize_round(label: str) -> str:
    return ROUND_ALIASES.get(label.lower().strip(), label.strip())


def _extract_school_id(href: str) -> str | None:
    m = re.search(r"/cbb/schools/([^/]+)/", href)
    return m.group(1) if m else None


def _find_table(soup: BeautifulSoup, table_id: str) -> BeautifulSoup | None:
    """
    Find a table by id in both the visible DOM and HTML comments.
    Sports-reference often wraps tables in <!-- --> comments.
    """
    table = soup.find("table", id=table_id)
    if table:
        return table

    wrapper_id = f"all_{table_id}"
    wrapper = soup.find("div", id=wrapper_id)
    if wrapper:
        for comment in wrapper.find_all(string=lambda t: isinstance(t, Comment)):
            inner = BeautifulSoup(comment, "lxml")
            table = inner.find("table", id=table_id)
            if table:
                return table

    # Broader comment search
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if f'id="{table_id}"' in comment:
            inner = BeautifulSoup(comment, "lxml")
            table = inner.find("table", id=table_id)
            if table:
                return table

    return None


def _parse_tourney_table(table: BeautifulSoup, year: int) -> list[dict]:
    """
    Parse the all_tourney summary table.

    Sports-reference columns (typical):
      Round | Date | Winner | Pts | Loser | Pts | OT | Location | Notes

    Seeds appear in parentheses after team names, e.g. "Connecticut (1)".
    """
    # Map header data-stat attributes → column indices.
    # Sports-reference sometimes uses data-stat, sometimes plain <th> text.
    thead = table.find("thead")
    if not thead:
        return []

    # Use the last header row (some tables have a group row above real headers)
    header_rows = thead.find_all("tr")
    header_row = header_rows[-1] if header_rows else thead
    headers = []
    for th in header_row.find_all(["th", "td"]):
        stat = th.get("data-stat", "").strip().lower()
        if not stat:
            stat = th.get_text(strip=True).lower()
        headers.append(stat)

    print(f"  [debug] headers: {headers}")
    print(f"  [debug] first row: {str(table.find('tbody').find('tr'))[:300]}")

    def col(*names: str) -> int | None:
        for name in names:
            try:
                return headers.index(name)
            except ValueError:
                pass
        return None

    round_col  = col("round", "rnd")
    winner_col = col("winner", "school_name", "winner_school", "team")
    wpts_col   = col("pts_winner", "pts", "winner_pts", "w pts", "wpts")
    loser_col  = col("loser", "loser_school", "opponent", "opp")
    lpts_col   = col("pts_loser", "loser_pts", "l pts", "lpts", "opp pts")

    if winner_col is None or loser_col is None:
        print(f"  [warn] Could not map winner/loser columns. Headers: {headers}")
        return []

    games = []
    for row in table.find("tbody").find_all("tr"):
        if row.get("class") and "thead" in row["class"]:
            continue
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        def cell_text(idx):
            if idx is None or idx >= len(cells):
                return ""
            return cells[idx].get_text(strip=True)

        def cell_link(idx):
            if idx is None or idx >= len(cells):
                return None
            return cells[idx].find("a")

        round_raw = cell_text(round_col)
        if not round_raw:
            continue
        round_name = _normalize_round(round_raw)
        round_order = ROUND_ORDER.get(round_name, 99)

        # Winner
        winner_a = cell_link(winner_col)
        winner_name_raw = cell_text(winner_col)
        winner_name, winner_seed = _parse_name_seed(winner_name_raw)
        winner_id = _extract_school_id(winner_a["href"]) if winner_a else None
        winner_pts_text = cell_text(wpts_col)
        winner_pts = int(winner_pts_text) if winner_pts_text.isdigit() else None

        # Loser
        loser_a = cell_link(loser_col)
        loser_name_raw = cell_text(loser_col)
        loser_name, loser_seed = _parse_name_seed(loser_name_raw)
        loser_id = _extract_school_id(loser_a["href"]) if loser_a else None
        loser_pts_text = cell_text(lpts_col)
        loser_pts = int(loser_pts_text) if loser_pts_text.isdigit() else None

        if not winner_name or not loser_name:
            continue

        # Canonically order so team1 is the lower seed (favored)
        if winner_seed is not None and loser_seed is not None and winner_seed > loser_seed:
            # Upset: loser was actually the favorite
            team1, t1_id, t1_seed, t1_score = loser_name, loser_id, loser_seed, loser_pts
            team2, t2_id, t2_seed, t2_score = winner_name, winner_id, winner_seed, winner_pts
        else:
            team1, t1_id, t1_seed, t1_score = winner_name, winner_id, winner_seed, winner_pts
            team2, t2_id, t2_seed, t2_score = loser_name, loser_id, loser_seed, loser_pts

        games.append({
            "year": year,
            "round": round_name,
            "round_order": round_order,
            "team1": team1,
            "team1_id": t1_id,
            "team1_seed": t1_seed,
            "team1_score": t1_score,
            "team2": team2,
            "team2_id": t2_id,
            "team2_seed": t2_seed,
            "team2_score": t2_score,
            "winner": winner_name,
        })

    return games


def _parse_name_seed(raw: str) -> tuple[str, int | None]:
    """
    Extract team name and seed from strings like 'Connecticut (1)' or
    'UConn (1) OT'. Returns (name, seed).
    """
    raw = re.sub(r"\s*(OT|2OT|3OT)\s*$", "", raw.strip())
    m = re.search(r"^(.*?)\s*\((\d{1,2})\)\s*$", raw)
    if m:
        return m.group(1).strip(), int(m.group(2))
    return raw.strip(), None


def _fetch_page(year: int, session: requests.Session) -> requests.Response | None:
    for template in URL_TEMPLATES:
        url = template.format(year=year)
        try:
            resp = session.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                return resp
        except requests.RequestException:
            continue
    return None


def scrape_tournament_year(year: int, session: requests.Session) -> list[dict]:
    resp = _fetch_page(year, session)
    if resp is None:
        print(f"  Could not fetch page")
        return []

    soup = BeautifulSoup(resp.text, "lxml")

    # Primary: all_tourney table (clean list of all games)
    table = _find_table(soup, "all_tourney")
    if table:
        games = _parse_tourney_table(table, year)
        if games:
            return games

    # Fallback: try alternate table ids used in older seasons
    for table_id in ("tourney_results", "ncaa_tourney", "tourney"):
        table = _find_table(soup, table_id)
        if table:
            games = _parse_tourney_table(table, year)
            if games:
                return games

    # Last resort: print available table ids so we can debug
    all_tables = soup.find_all("table")
    tids = [t.get("id") for t in all_tables if t.get("id")]
    print(f"  [warn] No tournament table found. Tables on page: {tids}")
    return []


def collect_all_tournament_results(
    years: list[int] = TOURNAMENT_YEARS,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_games = []
    session = requests.Session()

    for year in years:
        print(f"Scraping {year} tournament...", end=" ", flush=True)
        try:
            games = scrape_tournament_year(year, session)
            all_games.extend(games)
            print(f"{len(games)} games")
        except requests.HTTPError as e:
            print(f"HTTP {e.response.status_code} - skipping")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(REQUEST_DELAY)

    if not all_games:
        raise RuntimeError(
            "No tournament games were scraped. Check the [warn] output above "
            "to see what table IDs are available on the page."
        )

    df = pd.DataFrame(all_games)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} total games → {output_path}")
    return df


if __name__ == "__main__":
    collect_all_tournament_results()
