"""
Run this script locally (on your own machine) to scrape player class stats
from sports-reference. GitHub Actions IPs get blocked; your home/office IP
typically doesn't.

Usage (from the project root):
    pip install -r requirements.txt
    python collect_local.py

The script saves incrementally — you can Ctrl-C and restart and it will
skip teams already collected. Results go to data/raw/player_class_stats.csv.
"""

import sys
import time
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from constants import TOURNAMENT_YEARS
from player_class_stats import (
    HEADERS,
    FRESHMAN_LABELS,
    ALL_CLASS_LABELS,
    _get_per_game_table,
    _diagnose_response,
)

LOCAL_DELAY = 5       # seconds between requests (respectful for local use)
MAX_RETRIES = 3
OUTPUT_PATH = Path("data/raw/player_class_stats.csv")


def fetch_one(team_id: str, year: int, session: requests.Session) -> dict | None:
    urls = [
        f"https://www.sports-reference.com/cbb/schools/{team_id}/men/{year}.html",
        f"https://www.sports-reference.com/cbb/schools/{team_id}/{year}.html",
    ]
    for url in urls:
        for attempt in range(MAX_RETRIES):
            try:
                resp = session.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
            except Exception as e:
                print(f"    request error: {e}")
                break
            if resp.status_code == 404:
                break
            if resp.status_code == 429:
                wait = 30 * (2 ** attempt)
                print(f"    429 — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}")
                break

            soup = BeautifulSoup(resp.text, "lxml")
            title = soup.find("h1", {"itemprop": "name"})
            team_name = title.get_text(strip=True) if title else team_id

            table = _get_per_game_table(soup)
            if table is None:
                ids = [t.get("id", "?") for t in soup.find_all("table")]
                print(f"    table not found. ids={ids[:8]}")
                _diagnose_response(resp)
                break

            header_cells = table.find("thead").find_all(["th", "td"])
            headers_list = [c.get("data-stat", c.get_text(strip=True)).lower() for c in header_cells]

            def col(name):
                for attr in (name, name.replace("_", "")):
                    try:
                        return headers_list.index(attr)
                    except ValueError:
                        pass
                return None

            class_col = col("class_") or col("class") or col("yr")
            mp_col = col("mp")
            g_col = col("g")
            pts_col = col("pts")

            if class_col is None:
                print(f"    no class col. headers={headers_list[:15]}")
                break

            total_min = fr_min = total_pts = fr_pts = 0.0
            n_fr = 0

            for row in table.find("tbody").find_all("tr"):
                if "thead" in (row.get("class") or []):
                    continue
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue

                def val(idx):
                    if idx is None or idx >= len(cells):
                        return ""
                    return cells[idx].get_text(strip=True)

                cv = val(class_col).lower().strip()
                if not cv or cv not in ALL_CLASS_LABELS:
                    continue
                try:
                    g = float(val(g_col)) if g_col is not None else 0
                    mp = float(val(mp_col)) if mp_col is not None else 0
                    pts = float(val(pts_col)) if pts_col is not None else 0
                except ValueError:
                    continue

                pm = mp * g
                pp = pts * g
                total_min += pm
                total_pts += pp
                if cv in FRESHMAN_LABELS:
                    fr_min += pm
                    fr_pts += pp
                    n_fr += 1

            if total_min == 0 and total_pts == 0:
                print(f"    found table but no valid rows")
                break

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
                "n_fr_players": n_fr,
            }
    return None


def main():
    tournament_results = Path("data/raw/tournament_results.csv")
    if not tournament_results.exists():
        print("ERROR: data/raw/tournament_results.csv not found.")
        print("Run step 1 first: python run_pipeline.py --step 1")
        sys.exit(1)

    results = pd.read_csv(tournament_results)
    t1 = results[["year", "team1_id", "team1"]].rename(columns={"team1_id": "team_id", "team1": "team_name"})
    t2 = results[["year", "team2_id", "team2"]].rename(columns={"team2_id": "team_id", "team2": "team_name"})
    teams = (
        pd.concat([t1, t2])
        .dropna(subset=["team_id"])
        .drop_duplicates(subset=["year", "team_id"])
        .sort_values(["year", "team_id"])
        .reset_index(drop=True)
    )

    # Load already-scraped teams so we can resume
    already_done = set()
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        already_done = set(zip(existing["year"].astype(int), existing["team_id"]))
        print(f"Resuming — {len(already_done)} teams already scraped.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    remaining = [
        (int(r.year), r.team_id, r.team_name)
        for _, r in teams.iterrows()
        if (int(r.year), r.team_id) not in already_done
    ]

    print(f"{len(remaining)} team-seasons left to scrape (out of {len(teams)} total).")

    for i, (year, team_id, team_name) in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {year} {team_name} ({team_id})", end=" ... ", flush=True)
        data = fetch_one(team_id, year, session)
        if data:
            print(f"fr_min={data['fr_min_share']:.1%}  fr_pts={data['fr_pts_share']:.1%}")
            row_df = pd.DataFrame([data])
            row_df.to_csv(OUTPUT_PATH, mode="a", header=not OUTPUT_PATH.exists(), index=False)
        else:
            print("no data")
        time.sleep(LOCAL_DELAY)

    total = pd.read_csv(OUTPUT_PATH)
    print(f"\nDone. {len(total)} team-seasons saved to {OUTPUT_PATH}")
    print("Now commit the file and push:")
    print("  git add data/raw/player_class_stats.csv")
    print("  git commit -m 'Add scraped player class stats'")
    print("  git push")


if __name__ == "__main__":
    main()
