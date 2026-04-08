"""
Run this script locally (on your own machine) to scrape player class stats
from sports-reference. GitHub Actions IPs can be rate-limited; your home/
office IP is usually fine.

Usage (from the project root):
    pip install -r requirements.txt
    python run_pipeline.py --step 1   # if tournament_results.csv is missing
    python collect_local.py

Saves incrementally — safe to Ctrl-C and restart; already-scraped teams
are skipped. Results go to data/raw/player_class_stats.csv.
After finishing, commit and push the file so the workflow skips step 3:
    git add data/raw/player_class_stats.csv
    git commit -m "Add scraped player class stats"
    git push
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import requests

from player_class_stats import fetch_team_stats

LOCAL_DELAY = 5  # seconds between requests
OUTPUT_PATH = Path("data/raw/player_class_stats.csv")


def main():
    results_path = Path("data/raw/tournament_results.csv")
    if not results_path.exists():
        print("ERROR: data/raw/tournament_results.csv not found.")
        print("Run: python run_pipeline.py --step 1")
        sys.exit(1)

    results = pd.read_csv(results_path)
    t1 = results[["year", "team1_id", "team1"]].rename(columns={"team1_id": "team_id", "team1": "team_name"})
    t2 = results[["year", "team2_id", "team2"]].rename(columns={"team2_id": "team_id", "team2": "team_name"})
    teams = (
        pd.concat([t1, t2])
        .dropna(subset=["team_id"])
        .drop_duplicates(subset=["year", "team_id"])
        .sort_values(["year", "team_id"])
        .reset_index(drop=True)
    )

    already_done: set = set()
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        existing = pd.read_csv(OUTPUT_PATH)
        if not existing.empty:
            already_done = set(zip(existing["year"].astype(int), existing["team_id"]))
            print(f"Resuming — {len(already_done)} teams already scraped.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    remaining = [
        (int(r.year), r.team_id, r.team_name)
        for _, r in teams.iterrows()
        if (int(r.year), r.team_id) not in already_done
    ]
    print(f"{len(remaining)} team-seasons left (out of {len(teams)} total).")

    for i, (year, team_id, team_name) in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {year} {team_name} ({team_id})", end=" ... ", flush=True)
        try:
            data = fetch_team_stats(team_id, year, session)
            if data:
                print(f"fr_min={data['fr_min_share']:.1%}  fr_pts={data['fr_pts_share']:.1%}")
                write_header = not (OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0)
                pd.DataFrame([data]).to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
            else:
                print("no data")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(LOCAL_DELAY)

    if OUTPUT_PATH.exists():
        total = pd.read_csv(OUTPUT_PATH)
        print(f"\nDone. {len(total)} team-seasons saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
