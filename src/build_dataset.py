"""
Merges tournament results, efficiency ratings, and player class stats into an
analysis-ready dataset at the game-team level (two rows per game, one per team).

For each team-game row:
  - spread_delta    : actual margin from this team's perspective minus expected
                      (positive = beat expectations, negative = fell short)
  - netRtg          : team's pre-tournament efficiency margin (ORtg - DRtg)
  - fr_min_share    : fraction of regular-season minutes played by freshmen
  - fr_pts_share    : fraction of regular-season points scored by freshmen

The interaction netRtg × fr_min_share is the key predictor in the regression:
does being a highly-rated freshman-heavy team predict underperformance?

Output: data/processed/analysis_dataset.csv
"""

import re
from pathlib import Path

import pandas as pd

TOURNAMENT_RESULTS = Path("data/raw/tournament_results.csv")
EFFICIENCY_RATINGS = Path("data/raw/efficiency_ratings.csv")
PLAYER_CLASS_STATS = Path("data/raw/player_class_stats.csv")
OUTPUT_PATH = Path("data/processed/analysis_dataset.csv")


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

SREF_TO_BART_NAME = {
    # Abbreviations / acronyms sref uses that differ from Barttorvik
    "Connecticut": "UConn",
    "USC": "Southern California",
    "UCSB": "UC Santa Barbara",
    "UTSA": "UT San Antonio",
    "UTEP": "Texas-El Paso",
    "ETSU": "East Tennessee St.",
    "LIU": "LIU Brooklyn",
    "Army": "Army West Point",
    # Parenthetical disambiguators: sref appends "(State)" but Barttorvik varies
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Loyola (MD)": "Loyola MD",
    "Albany (NY)": "Albany",
    "Queens (NC)": "Queens",
    "St. John's (NY)": "St. John's",
    # Name shortening or hyphen/space differences
    "UNC Asheville": "NC Asheville",
    "College of Charleston": "Charleston",
    "Gardner-Webb": "Gardner Webb",
    "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
    "Southern Illinois-Edwardsville": "SIU Edwardsville",
    "SIU-Edwardsville": "SIU Edwardsville",
    "Southeast Missouri St.": "SE Missouri State",
    "Detroit Mercy": "Detroit",
    # State name / conference rebrands
    "Louisiana": "Louisiana Lafayette",
    "Louisiana-Lafayette": "Louisiana Lafayette",
    "Saint Mary's (CA)": "Saint Mary's",
    # Teams sref may display without 'State' suffix
    "Grambling": "Grambling St.",
    "Sam Houston": "Sam Houston St.",
    "Omaha": "Nebraska Omaha",       # old sref name before Nebraska-Omaha rebrand
    # Alternative formats / abbreviations sref uses in some years
    "NC State": "N.C. State",
    "North Carolina State": "N.C. State",
    "Ole Miss": "Mississippi",
    "California Baptist": "Cal Baptist",
    # Short names sref uses in brackets that differ from Barttorvik full names
    "Pitt": "Pittsburgh",
    "UNC": "North Carolina",
    "UMass": "Massachusetts",
    "FDU": "Fairleigh Dickinson",
    "Loyola (IL)": "Loyola Chicago",
    "McNeese": "McNeese St.",
}


def normalise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # Apply any explicit sref→Barttorvik remapping first
    name = SREF_TO_BART_NAME.get(name, name)
    name = name.lower().strip()
    # Barttorvik abbreviates "state" as "st." — apply after the dict lookup
    # so that dict values containing "State" (e.g. "N.C. State") are also normalised
    name = re.sub(r'\bstate\b', 'st.', name)
    return name


# ---------------------------------------------------------------------------
# Expected margin
# ---------------------------------------------------------------------------

def compute_expected_margin(row: pd.Series) -> float | None:
    """
    Expected margin for team1 over team2 on a neutral court:
        expected_margin = (netRtg_team1 - netRtg_team2) * avg_pace / 100
    """
    em1, em2 = row.get("netRtg_team1"), row.get("netRtg_team2")
    if pd.isna(em1) or pd.isna(em2):
        return None
    p1, p2 = row.get("Pace_team1"), row.get("Pace_team2")
    avg_pace = (p1 + p2) / 2 if pd.notna(p1) and pd.notna(p2) else 70.0
    return (em1 - em2) * avg_pace / 100


# ---------------------------------------------------------------------------
# Game-team level conversion
# ---------------------------------------------------------------------------

def to_team_game_level(games: pd.DataFrame) -> pd.DataFrame:
    """
    Explode one row per game into two rows — one per team.
    spread_delta is from that team's perspective (positive = beat expectations).
    game_id is used to cluster standard errors in the regression.
    """
    records = []
    for gid, row in enumerate(games.itertuples(index=False)):
        sd = row.spread_delta  # from team1 perspective; NaN if ratings missing
        base = dict(
            game_id=gid,
            year=row.year,
            round=row.round,
            is_first_four=getattr(row, "is_first_four", False),
        )
        em  = row.expected_margin   # positive = team1 favored
        am  = row.actual_margin     # positive = team1 won
        records.append({**base,
            "team_id":         row.team1_id,
            "team_name":       row.team1,
            "seed":            row.team1_seed,
            "opp_id":          row.team2_id,
            "opp_name":        row.team2,
            "opp_seed":        row.team2_seed,
            "team_score":      row.team1_score,
            "opp_score":       row.team2_score,
            "expected_margin": em,
            "actual_margin":   am,
            "spread_delta":    sd,
            "is_favorite":     True,
            "netRtg":          row.netRtg_team1,
            "fr_min_share":    row.fr_min_share_team1,
            "fr_pts_share":    row.fr_pts_share_team1,
        })
        records.append({**base,
            "team_id":         row.team2_id,
            "team_name":       row.team2,
            "seed":            row.team2_seed,
            "opp_id":          row.team1_id,
            "opp_name":        row.team1,
            "opp_seed":        row.team1_seed,
            "team_score":      row.team2_score,
            "opp_score":       row.team1_score,
            "expected_margin": -em if pd.notna(em) else None,
            "actual_margin":   -am if pd.notna(am) else None,
            "spread_delta":    -sd if pd.notna(sd) else None,
            "is_favorite":     False,
            "netRtg":          row.netRtg_team2,
            "fr_min_share":    row.fr_min_share_team2,
            "fr_pts_share":    row.fr_pts_share_team2,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_analysis_dataset(
    results_path: Path = TOURNAMENT_RESULTS,
    ratings_path: Path = EFFICIENCY_RATINGS,
    class_path: Path = PLAYER_CLASS_STATS,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:

    games    = pd.read_csv(results_path)
    ratings  = pd.read_csv(ratings_path)
    class_stats = pd.read_csv(class_path)

    games["team1_norm"]  = games["team1"].apply(normalise_name)
    games["team2_norm"]  = games["team2"].apply(normalise_name)
    ratings["team_norm"] = ratings["team"].apply(normalise_name)

    # Join efficiency ratings (team1 and team2) — matched on normalised team name
    rat_cols = [c for c in ["year", "team_norm", "netRtg", "ORtg", "DRtg", "Pace"]
                if c in ratings.columns or c in ("year", "team_norm")]
    for side in ("team1", "team2"):
        games = games.merge(
            ratings[rat_cols].rename(columns={
                c: f"{c}_{side}" for c in rat_cols if c not in ("year", "team_norm")
            }),
            left_on=["year", f"{side}_norm"],
            right_on=["year", "team_norm"],
            how="left",
        ).drop(columns="team_norm")

    # Join freshman stats (team1 and team2) — matched on team_id, which is
    # consistent between tournament_results and player_class_stats (both come
    # from the same sports-reference URL slug).
    fr_cols = ["year", "team_id", "fr_min_share", "fr_pts_share"]
    for side in ("team1", "team2"):
        games = games.merge(
            class_stats[fr_cols].rename(columns={
                c: f"{c}_{side}" for c in fr_cols if c not in ("year", "team_id")
            }),
            left_on=["year", f"{side}_id"],
            right_on=["year", "team_id"],
            how="left",
        ).drop(columns="team_id")

    # Compute spreads
    games["expected_margin"] = games.apply(compute_expected_margin, axis=1)
    games["actual_margin"]   = games["team1_score"] - games["team2_score"]
    games["spread_delta"]    = games["actual_margin"] - games["expected_margin"]
    games["is_first_four"]   = games["round"] == "First Four"

    print(f"Games: {len(games)} total, {games['spread_delta'].notna().sum()} with spread data")
    check_unmatched_teams(games)

    # Explode to game-team level
    team_games = to_team_game_level(games)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    team_games.to_csv(output_path, index=False)
    n_complete = team_games.dropna(subset=["spread_delta", "netRtg", "fr_min_share"]).shape[0]
    print(f"Team-game rows: {len(team_games)} ({n_complete} fully complete)")
    print(f"Saved → {output_path}")
    return team_games


def check_unmatched_teams(df: pd.DataFrame) -> None:
    """Print games where efficiency or freshman data is missing post-merge."""
    missing_eff = df[df["netRtg_team1"].isna() | df["netRtg_team2"].isna()]
    missing_fr  = df[df["fr_min_share_team1"].isna() | df["fr_min_share_team2"].isna()]

    if not missing_eff.empty:
        print(f"\n{len(missing_eff)} games missing efficiency ratings:")
        for _, r in missing_eff.head(10).iterrows():
            teams = [r.team1 if pd.isna(r.netRtg_team1) else None,
                     r.team2 if pd.isna(r.netRtg_team2) else None]
            print(f"  {r.year} {r['round']}: {', '.join(t for t in teams if t)}")

    if not missing_fr.empty:
        print(f"\n{len(missing_fr)} games missing freshman data:")
        for _, r in missing_fr.head(10).iterrows():
            teams = [r.team1 if pd.isna(r.fr_min_share_team1) else None,
                     r.team2 if pd.isna(r.fr_min_share_team2) else None]
            print(f"  {r.year} {r['round']}: {', '.join(t for t in teams if t)}")


if __name__ == "__main__":
    build_analysis_dataset()
