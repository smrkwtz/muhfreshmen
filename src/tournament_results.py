"""
Scrapes NCAA tournament game results from sports-reference.com for 2010-2026.

Output columns:
  year, round, round_order,
  team1, team1_id, team1_seed, team1_score,
  team2, team2_id, team2_seed, team2_score,
  winner

team1 is always the higher seed (lower seed number = favored).
If seeds are tied (First Four play-in), team1 is just whoever appears first.
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from constants import ROUND_ORDER, TOURNAMENT_YEARS

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research/academic project)"}
REQUEST_DELAY = 4  # seconds - sports-reference asks for respectful crawling

OUTPUT_PATH = Path("data/raw/tournament_results.csv")

# sports-reference sometimes spells round labels slightly differently across years
ROUND_ALIASES = {
    "first four": "First Four",
    "opening round": "First Four",
    "first round": "First Round",
    "second round": "Second Round",
    "third round": "Second Round",      # early 2010s naming
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
}


def _normalize_round(label: str) -> str:
    return ROUND_ALIASES.get(label.lower().strip(), label.strip())


def _extract_school_id(href: str) -> str | None:
    m = re.search(r"/cbb/schools/([^/]+)/", href)
    return m.group(1) if m else None


def _find_bracket(soup: BeautifulSoup) -> BeautifulSoup | None:
    """
    Find the bracket div in both the visible DOM and inside HTML comments.
    Sports-reference frequently wraps content in <!-- --> comments to
    defer rendering to JavaScript; we need to check both locations.
    """
    # 1. Try visible DOM first
    bracket = soup.find("div", id="bracket")
    if bracket:
        return bracket

    # 2. Search inside HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "bracket" not in comment:
            continue
        inner = BeautifulSoup(comment, "lxml")
        bracket = inner.find("div", id="bracket")
        if bracket:
            return bracket

    return None


def _debug_page_structure(soup: BeautifulSoup, year: int) -> None:
    """Print structural info to help diagnose parse failures."""
    print(f"\n  --- DEBUG for {year} ---")

    # List all div ids
    ids = [t.get("id") for t in soup.find_all("div") if t.get("id")]
    print(f"  Div IDs on page: {ids[:30]}")

    # List divs whose id or class contains "bracket"
    for tag in soup.find_all("div"):
        classes = " ".join(tag.get("class") or [])
        id_ = tag.get("id", "")
        if "bracket" in id_.lower() or "bracket" in classes.lower():
            print(f"  Bracket-related div: id='{id_}' class='{classes}'")

    # Print first 2000 chars of raw HTML for manual inspection
    print(f"  First 1000 chars of HTML:\n{str(soup)[:1000]}")
    print(f"  --- END DEBUG ---\n")


def scrape_tournament_year(year: int, session: requests.Session, debug: bool = False) -> list[dict]:
    """Parse one tournament bracket page and return a list of game dicts."""
    url = f"https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html"
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    games = []

    bracket = _find_bracket(soup)
    if not bracket:
        print(f"  [warn] No #bracket div found for {year}")
        if debug:
            _debug_page_structure(soup, year)
        return games

    # Each <div class="round"> holds one round's worth of games.
    # Within each round, game containers appear as direct child divs.
    for round_div in bracket.find_all("div", class_="round"):
        # Round label comes from an <h2> or <h3> heading inside the round div
        heading = round_div.find(["h2", "h3"])
        round_name = _normalize_round(heading.get_text()) if heading else "Unknown"
        round_order = ROUND_ORDER.get(round_name, 99)

        for game_div in round_div.find_all("div", recursive=False):
            # Each game div contains exactly two team divs
            team_rows = game_div.find_all("div", class_=re.compile(r"\bteam\b"))
            if len(team_rows) != 2:
                continue

            teams = []
            for row in team_rows:
                seed_el = row.find("span", class_="seed")
                name_a = row.find("a")
                score_el = row.find("span", class_="score")

                seed_text = seed_el.get_text(strip=True) if seed_el else ""
                seed = int(seed_text) if seed_text.isdigit() else None

                name = name_a.get_text(strip=True) if name_a else row.get_text(strip=True)
                school_id = _extract_school_id(name_a["href"]) if name_a and name_a.get("href") else None

                score_text = score_el.get_text(strip=True) if score_el else ""
                score = int(score_text) if score_text.isdigit() else None

                winner_class = "winner" in (row.get("class") or [])
                teams.append({
                    "name": name,
                    "school_id": school_id,
                    "seed": seed,
                    "score": score,
                    "is_winner": winner_class,
                })

            t1, t2 = teams

            # Canonically order so team1 is the lower seed (favored).
            # For equal seeds (First Four), preserve document order.
            if t1["seed"] is not None and t2["seed"] is not None and t1["seed"] > t2["seed"]:
                t1, t2 = t2, t1

            winner = t1["name"] if t1["is_winner"] else (t2["name"] if t2["is_winner"] else None)

            games.append({
                "year": year,
                "round": round_name,
                "round_order": round_order,
                "team1": t1["name"],
                "team1_id": t1["school_id"],
                "team1_seed": t1["seed"],
                "team1_score": t1["score"],
                "team2": t2["name"],
                "team2_id": t2["school_id"],
                "team2_seed": t2["seed"],
                "team2_score": t2["score"],
                "winner": winner,
            })

    return games


def collect_all_tournament_results(
    years: list[int] = TOURNAMENT_YEARS,
    output_path: Path = OUTPUT_PATH,
    debug: bool = False,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_games = []
    session = requests.Session()

    for year in years:
        print(f"Scraping {year} tournament...", end=" ", flush=True)
        try:
            games = scrape_tournament_year(year, session, debug=debug)
            all_games.extend(games)
            print(f"{len(games)} games")
        except requests.HTTPError as e:
            print(f"HTTP {e.response.status_code} - skipping")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(REQUEST_DELAY)

    if not all_games:
        raise RuntimeError(
            "No tournament games were scraped. The page structure may have changed "
            "or sports-reference is blocking the request. Re-run with debug=True "
            "(set DEBUG_SCRAPE=1 env var) to inspect the HTML."
        )

    df = pd.DataFrame(all_games)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} total games → {output_path}")
    return df


if __name__ == "__main__":
    import os
    debug = os.environ.get("DEBUG_SCRAPE", "0") == "1"
    collect_all_tournament_results(debug=debug)
