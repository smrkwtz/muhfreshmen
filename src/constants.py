"""
Shared constants for the freshman reliance / tournament outcomes analysis.
"""

# 2020 tournament was cancelled (COVID)
TOURNAMENT_YEARS = [y for y in range(2010, 2027) if y != 2020]

# First-game date for each tournament year (First Four or First Round),
# used as the cutoff when pulling "pre-tournament" efficiency ratings.
TOURNAMENT_START_DATES = {
    2010: "20100318",
    2011: "20110315",
    2012: "20120313",
    2013: "20130319",
    2014: "20140318",
    2015: "20150317",
    2016: "20160315",
    2017: "20170314",
    2018: "20180313",
    2019: "20190319",
    2021: "20210318",
    2022: "20220315",
    2023: "20230314",
    2024: "20240319",
    2025: "20250318",
    2026: "20260318",  # approximate; verify once bracket is set
}

# Round display order (sports-reference bracket structure)
ROUND_ORDER = {
    "First Four": 0,
    "First Round": 1,
    "Second Round": 2,
    "Sweet 16": 3,
    "Elite Eight": 4,
    "Final Four": 5,
    "Championship": 6,
}

# Average possessions per 40 minutes - used as fallback when tempo data
# is unavailable. KenPom/Barttorvik adjusted tempo is used when present.
DEFAULT_POSSESSIONS = 70.0
