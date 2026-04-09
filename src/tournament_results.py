"""
Scrapes NCAA tournament game results from sports-reference.com for 2010-2026.

Confirmed HTML structure (from live page inspection):
  - 5 <div id="bracket"> elements: East, Midwest, South, West, National
  - Each contains <div class="round"> for each round
  - Each round contains plain <div> game containers
  - Each game has two team divs: winner has class="winner", loser has no class
  - Team div: <span>seed</span> <a href="/cbb/schools/...">Name</a> <a href="/cbb/boxscores/...">Score</a>

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
from bs4 import BeautifulSoup

from constants import ROUND_ORDER, TOURNAMENT_YEARS

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research/academic project)"}
REQUEST_DELAY = 4

OUTPUT_PATH = Path("data/raw/tournament_results.csv")

# Sports-reference added /men/ to the URL path at some point; try both.
URL_TEMPLATES = [
    "https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html",
    "https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html",
]

# Round names by position index within a regional bracket.
# If a regional has 5 round divs the first is First Four; if 4 it starts at First Round.
REGIONAL_ROUNDS = ["First Round", "Second Round", "Sweet 16", "Elite Eight"]
NATIONAL_ROUNDS = ["Final Four", "Championship"]
FIRST_FOUR     = "First Four"

# IDs of the parent divs that wrap the national bracket
NATIONAL_PARENT_IDS = {"national", "natl", "championship"}


def _extract_school_id(href: str) -> str | None:
    m = re.search(r"/cbb/schools/([^/]+)/", href)
    return m.group(1) if m else None


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


def _parse_team_div(div) -> dict | None:
    """Extract seed, name, school_id, score and winner flag from a team div."""
    is_winner = "winner" in (div.get("class") or [])

    seed_span = div.find("span")
    seed_text = seed_span.get_text(strip=True) if seed_span else ""
    seed = int(seed_text) if seed_text.isdigit() else None

    links = div.find_all("a")
    if not links:
        return None

    # First link → team page (name + school_id)
    name_a = links[0]
    name = name_a.get_text(strip=True)
    school_id = _extract_school_id(name_a.get("href", ""))

    # Second link → boxscore page (score is the link text)
    score = None
    if len(links) >= 2:
        score_text = links[1].get_text(strip=True)
        score = int(score_text) if score_text.isdigit() else None

    return {
        "name": name,
        "school_id": school_id,
        "seed": seed,
        "score": score,
        "is_winner": is_winner,
    }


def _parse_bracket(bracket_div, round_names: list[str], year: int) -> list[dict]:
    """Parse all rounds within one regional or national bracket div."""
    games = []
    round_divs = bracket_div.find_all("div", class_="round")

    # If a regional bracket has 5 rounds, the extra div is either:
    #   - a leading First Four div (older years): teams share the same seed
    #   - a trailing placeholder div (newer format): teams have different seeds
    # Peek at the first parseable game in round_divs[0] to tell them apart.
    if len(round_divs) == len(round_names) + 1:
        round_labels = round_names  # default: trailing placeholder
        for game_div in round_divs[0].find_all("div", recursive=False):
            child_divs = game_div.find_all("div", recursive=False)
            if len(child_divs) < 2:
                continue
            seeds = []
            for td in child_divs[:2]:
                span = td.find("span")
                txt = span.get_text(strip=True) if span else ""
                if txt.isdigit():
                    seeds.append(int(txt))
            if len(seeds) == 2:
                if seeds[0] == seeds[1]:
                    round_labels = [FIRST_FOUR] + round_names  # leading First Four
                break  # first game is enough
    else:
        round_labels = round_names

    for round_idx, round_div in enumerate(round_divs):
        round_name = round_labels[round_idx] if round_idx < len(round_labels) else f"Round {round_idx}"
        round_order = ROUND_ORDER.get(round_name, 99)

        # Each direct child div of the round div is a game container
        for game_div in round_div.find_all("div", recursive=False):
            child_divs = game_div.find_all("div", recursive=False)
            if len(child_divs) < 2:
                continue

            teams = []
            for td in child_divs[:2]:
                t = _parse_team_div(td)
                if t:
                    teams.append(t)

            if len(teams) < 2:
                continue

            t1, t2 = teams[0], teams[1]

            # Canonically order so team1 is the lower seed (favored)
            if (t1["seed"] is not None and t2["seed"] is not None
                    and t1["seed"] > t2["seed"]):
                t1, t2 = t2, t1

            winner = (t1["name"] if t1["is_winner"]
                      else t2["name"] if t2["is_winner"]
                      else None)

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


def scrape_tournament_year(year: int, session: requests.Session) -> list[dict]:
    resp = _fetch_page(year, session)
    if resp is None:
        print(f"  Could not fetch page")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    bracket_divs = soup.find_all("div", id="bracket")

    if not bracket_divs:
        print(f"  No #bracket divs found")
        return []

    all_games = []
    for bracket_div in bracket_divs:
        parent_id = (bracket_div.parent.get("id") or "").lower()
        is_national = parent_id in NATIONAL_PARENT_IDS
        round_names = NATIONAL_ROUNDS if is_national else REGIONAL_ROUNDS
        games = _parse_bracket(bracket_div, round_names, year)
        all_games.extend(games)

    return all_games


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
        raise RuntimeError("No tournament games were scraped.")

    df = pd.DataFrame(all_games)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} total games → {output_path}")
    return df


if __name__ == "__main__":
    collect_all_tournament_results()
