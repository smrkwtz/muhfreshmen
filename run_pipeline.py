"""
Main pipeline entry point.

Steps:
  1. Scrape NCAA tournament results from sports-reference.com
  2. Fetch pre-tournament efficiency ratings from Barttorvik
  3. Fetch player-by-class stats from sports-reference.com
  4. Merge into analysis dataset
  5. Run regression analysis

Each step caches its output to data/raw/ or data/processed/.
Re-running skips steps whose output already exists unless --force is passed.

Usage:
  python run_pipeline.py            # run all steps
  python run_pipeline.py --step 1   # run only step 1
  python run_pipeline.py --force    # re-run all steps even if output exists
  python run_pipeline.py --from 3   # re-run steps 3+ (useful after fixing name mapping)
"""

import argparse
import sys
from pathlib import Path

# Add src/ to path so step modules can import each other cleanly
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tournament_results import collect_all_tournament_results, OUTPUT_PATH as RESULTS_PATH
from efficiency_ratings import collect_all_efficiency_ratings, OUTPUT_PATH as RATINGS_PATH
from player_class_stats import collect_player_class_stats, OUTPUT_PATH as CLASS_PATH
from build_dataset import build_analysis_dataset

ANALYSIS_PATH = Path("data/processed/analysis_dataset.csv")


def step1_tournament_results(force: bool = False) -> None:
    if RESULTS_PATH.exists() and not force:
        print(f"[Step 1] Skipping - {RESULTS_PATH} already exists (use --force to re-run)")
        return
    print("[Step 1] Scraping tournament results from sports-reference.com...")
    collect_all_tournament_results()


def step2_efficiency_ratings(force: bool = False) -> None:
    if RATINGS_PATH.exists() and not force:
        print(f"[Step 2] Skipping - {RATINGS_PATH} already exists (use --force to re-run)")
        return
    print("[Step 2] Fetching pre-tournament efficiency ratings from Barttorvik...")
    collect_all_efficiency_ratings()


def step3_player_class_stats(force: bool = False) -> None:
    if CLASS_PATH.exists() and not force:
        print(f"[Step 3] Skipping - {CLASS_PATH} already exists (use --force to re-run)")
        return
    print("[Step 3] Fetching player class stats from sports-reference.com...")
    collect_player_class_stats()


def step4_build_dataset(force: bool = False) -> None:
    if ANALYSIS_PATH.exists() and not force:
        print(f"[Step 4] Skipping - {ANALYSIS_PATH} already exists (use --force to re-run)")
        return
    print("[Step 4] Building merged analysis dataset...")
    build_analysis_dataset()


def step5_analysis() -> None:
    print("[Step 5] Running regression analysis...")
    # Import here so matplotlib backend issues don't affect earlier steps
    sys.path.insert(0, str(Path(__file__).parent / "analysis"))
    from regression import run_all
    run_all()


STEPS = {
    1: step1_tournament_results,
    2: step2_efficiency_ratings,
    3: step3_player_class_stats,
    4: step4_build_dataset,
    5: step5_analysis,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freshman reliance / tournament outcomes pipeline")
    parser.add_argument("--step", type=int, choices=STEPS.keys(), help="Run only this step")
    parser.add_argument("--from", dest="from_step", type=int, help="Run from this step onwards")
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists")
    args = parser.parse_args()

    if args.step:
        fn = STEPS[args.step]
        fn(force=args.force) if args.step < 5 else fn()
    else:
        start = args.from_step or 1
        for step_num in range(start, 6):
            fn = STEPS[step_num]
            fn(force=args.force) if step_num < 5 else fn()
