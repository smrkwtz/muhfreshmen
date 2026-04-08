"""
Cleans the Barttorvik Excel export and writes one CSV per year to
data/manual/barttorvik/barttorvik_YYYY.csv

Usage:
    python clean_barttorvik.py /path/to/your/file.xlsx

The Excel file should have one tab per year named "2010", "2011", etc.
Each tab has the raw Barttorvik copy-paste with:
  - Two rows per team (stats row + national-ranking comparison row)
  - Repeated header rows every ~25 teams
"""

import sys
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("data/manual/barttorvik")

# Columns we need (Barttorvik header names → our standard names)
KEEP_COLS = {
    "Team":   "team",
    "AdjOE":  "ORtg",
    "AdjDE":  "DRtg",
    "Adj T.": "Pace",
}


def clean_tab(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Clean one year's tab.
    - Drop repeated header rows (where Rk == 'Rk')
    - Drop rank-comparison rows (where Rk is empty or non-numeric)
    - Keep only the columns we need
    """
    # Rename columns from first row if the DataFrame has no header
    # (pandas may have read the first data row as header if Excel has no header)

    # Ensure column names match what we expect; if first row looks like a header
    # that pandas missed, promote it.
    if "Rk" not in df_raw.columns and "Team" not in df_raw.columns:
        df_raw.columns = df_raw.iloc[0]
        df_raw = df_raw.iloc[1:].reset_index(drop=True)

    # Drop rows where Rk is 'Rk' (repeated header rows)
    df = df_raw[df_raw["Rk"].astype(str).str.strip() != "Rk"].copy()

    # Keep only rows where Rk is a valid integer (drop rank-comparison rows)
    def is_int(val):
        try:
            int(str(val).strip())
            return True
        except (ValueError, TypeError):
            return False

    df = df[df["Rk"].apply(is_int)].copy()

    # Check for required columns
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        # Try case-insensitive match
        col_map = {c.lower(): c for c in df.columns}
        for needed in list(KEEP_COLS.keys()):
            if needed not in df.columns:
                match = col_map.get(needed.lower())
                if match:
                    df = df.rename(columns={match: needed})
                else:
                    print(f"  [warn] {year}: missing column '{needed}'. Available: {list(df.columns)}")

    # Select and rename
    available = {k: v for k, v in KEEP_COLS.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Cast to numeric
    for col in ("ORtg", "DRtg", "Pace"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where we couldn't parse the ratings
    df = df.dropna(subset=["ORtg", "DRtg"]).copy()

    # Compute netRtg
    df["netRtg"] = (df["ORtg"] - df["DRtg"]).round(2)
    df["year"] = year

    return df[["year", "team", "ORtg", "DRtg", "Pace", "netRtg"]].reset_index(drop=True)


def main(xlsx_path: str) -> None:
    path = Path(xlsx_path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {path.name}...")
    xl = pd.ExcelFile(path, engine="openpyxl")
    print(f"Found tabs: {xl.sheet_names}")

    for sheet in xl.sheet_names:
        try:
            year = int(str(sheet).strip())
        except ValueError:
            print(f"Skipping tab '{sheet}' (not a year)")
            continue

        df_raw = xl.parse(sheet, dtype=str, header=0)
        df_clean = clean_tab(df_raw, year)

        out_path = OUTPUT_DIR / f"barttorvik_{year}.csv"
        df_clean.to_csv(out_path, index=False)
        print(f"  {year}: {len(df_clean)} teams → {out_path}")

    print("\nDone. Run the pipeline to use these files.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_barttorvik.py /path/to/file.xlsx")
        sys.exit(1)
    main(sys.argv[1])
