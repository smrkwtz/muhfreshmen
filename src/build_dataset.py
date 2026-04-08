"""
Merges tournament results, efficiency ratings, and player class stats into a
single analysis-ready dataset.

For each game the output includes:
  - The expected spread (from pre-tournament efficiency ratings)
  - The actual margin
  - The spread delta (actual - expected), from team1's perspective
  - Freshman reliance metrics for both teams and their differential

team1 is always defined as the lower seed (favored team) in tournament_results.

Output: data/processed/analysis_dataset.csv
"""

from pathlib import Path

import pandas as pd

TOURNAMENT_RESULTS = Path("data/raw/tournament_results.csv")
EFFICIENCY_RATINGS = Path("data/raw/efficiency_ratings.csv")
PLAYER_CLASS_STATS = Path("data/raw/player_class_stats.csv")
OUTPUT_PATH = Path("data/processed/analysis_dataset.csv")


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------
# Sports-reference and Barttorvik use slightly different team name spellings.
# The mapping below covers the most common mismatches. Extend as needed after
# an initial merge audit (see check_unmatched_teams() below).

SREF_TO_BART_NAME = {
    "Connecticut": "UConn",
    "Miami (FL)": "Miami FL",
    "North Carolina": "North Carolina",
    "UNC Asheville": "NC Asheville",
    "USC": "Southern California",
    "UCSB": "UC Santa Barbara",
    "UTSA": "UT San Antonio",
    "UTEP": "Texas-El Paso",
    "Louisiana": "Louisiana Lafayette",
    "Louisiana-Lafayette": "Louisiana Lafayette",
    "Saint Mary's (CA)": "Saint Mary's",
    "Loyola Chicago": "Loyola-Chicago",
    "Army": "Army West Point",
    "LIU": "LIU Brooklyn",
    "Texas A&M-Corpus Christi": "TAMUCC",
    "Southeast Missouri St.": "SE Missouri State",
    "Detroit Mercy": "Detroit",
}


def normalise_name(name: str) -> str:
    """Apply name mapping; otherwise lowercase+strip for fuzzy matching."""
    if not isinstance(name, str):
        return ""
    mapped = SREF_TO_BART_NAME.get(name)
    if mapped:
        return mapped.lower().strip()
    return name.lower().strip()


# ---------------------------------------------------------------------------
# Spread calculation
# ---------------------------------------------------------------------------

def compute_expected_margin(row: pd.Series) -> float | None:
    """
    Expected margin for team1 (the favored team) over team2 on a neutral court:

        expected_margin = (adjEM_team1 - adjEM_team2) * avg_tempo / 100

    avg_tempo is the mean of both teams' adjusted tempos.
    Falls back to 70 possessions if tempo data is missing.
    """
    em1 = row.get("adjEM_team1")
    em2 = row.get("adjEM_team2")
    if pd.isna(em1) or pd.isna(em2):
        return None

    t1 = row.get("adjT_team1")
    t2 = row.get("adjT_team2")
    if pd.notna(t1) and pd.notna(t2):
        avg_tempo = (t1 + t2) / 2
    else:
        avg_tempo = 70.0

    return (em1 - em2) * avg_tempo / 100


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------

def build_analysis_dataset(
    results_path: Path = TOURNAMENT_RESULTS,
    ratings_path: Path = EFFICIENCY_RATINGS,
    class_path: Path = PLAYER_CLASS_STATS,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:

    # --- Load raw tables ---
    games = pd.read_csv(results_path)
    ratings = pd.read_csv(ratings_path)
    class_stats = pd.read_csv(class_path)

    # Normalise team names for matching
    games["team1_norm"] = games["team1"].apply(normalise_name)
    games["team2_norm"] = games["team2"].apply(normalise_name)
    ratings["team_norm"] = ratings["team"].apply(normalise_name)
    class_stats["team_norm"] = class_stats["team_name"].apply(normalise_name)

    # --- Join efficiency ratings ---
    rat_cols = ["year", "team_norm", "adjEM", "adjO", "adjD", "adjT"]
    games = games.merge(
        ratings[rat_cols].rename(columns={c: f"{c}_team1" if c not in ("year", "team_norm") else c
                                          for c in rat_cols}),
        left_on=["year", "team1_norm"],
        right_on=["year", "team_norm"],
        how="left",
    ).drop(columns="team_norm")

    games = games.merge(
        ratings[rat_cols].rename(columns={c: f"{c}_team2" if c not in ("year", "team_norm") else c
                                          for c in rat_cols}),
        left_on=["year", "team2_norm"],
        right_on=["year", "team_norm"],
        how="left",
    ).drop(columns="team_norm")

    # --- Join freshman class stats ---
    fr_cols = ["year", "team_norm", "fr_min_share", "fr_pts_share"]
    games = games.merge(
        class_stats[fr_cols].rename(columns={c: f"{c}_team1" if c not in ("year", "team_norm") else c
                                              for c in fr_cols}),
        left_on=["year", "team1_norm"],
        right_on=["year", "team_norm"],
        how="left",
    ).drop(columns="team_norm")

    games = games.merge(
        class_stats[fr_cols].rename(columns={c: f"{c}_team2" if c not in ("year", "team_norm") else c
                                              for c in fr_cols}),
        left_on=["year", "team2_norm"],
        right_on=["year", "team_norm"],
        how="left",
    ).drop(columns="team_norm")

    # --- Compute spreads and deltas ---
    games["expected_margin"] = games.apply(compute_expected_margin, axis=1)
    games["actual_margin"] = games["team1_score"] - games["team2_score"]

    # spread_delta > 0 means team1 (favored) outperformed expectations
    games["spread_delta"] = games["actual_margin"] - games["expected_margin"]

    # Freshman reliance differentials (favored minus underdog)
    games["fr_min_share_diff"] = games["fr_min_share_team1"] - games["fr_min_share_team2"]
    games["fr_pts_share_diff"] = games["fr_pts_share_team1"] - games["fr_pts_share_team2"]

    # Drop First Four (play-in games) from the main analysis - seeds are equal
    # and the games don't test the "upset" dynamic in the same way.
    # Leave them in but flag them for optional filtering.
    games["is_first_four"] = games["round"] == "First Four"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(output_path, index=False)
    print(f"Dataset: {len(games)} games, {games['spread_delta'].notna().sum()} with spread data")
    print(f"Saved → {output_path}")
    return games


def check_unmatched_teams(df: pd.DataFrame) -> None:
    """Print games where efficiency or freshman data is missing post-merge."""
    missing_eff = df[df["adjEM_team1"].isna() | df["adjEM_team2"].isna()]
    missing_fr = df[df["fr_min_share_team1"].isna() | df["fr_min_share_team2"].isna()]

    if not missing_eff.empty:
        print(f"\n{len(missing_eff)} games missing efficiency ratings:")
        for _, r in missing_eff.iterrows():
            missing = []
            if pd.isna(r.adjEM_team1):
                missing.append(r.team1)
            if pd.isna(r.adjEM_team2):
                missing.append(r.team2)
            print(f"  {r.year} {r.round}: {', '.join(missing)}")

    if not missing_fr.empty:
        print(f"\n{len(missing_fr)} games missing freshman data:")
        for _, r in missing_fr.iterrows():
            missing = []
            if pd.isna(r.fr_min_share_team1):
                missing.append(r.team1)
            if pd.isna(r.fr_min_share_team2):
                missing.append(r.team2)
            print(f"  {r.year} {r.round}: {', '.join(missing)}")


if __name__ == "__main__":
    df = build_analysis_dataset()
    check_unmatched_teams(df)
