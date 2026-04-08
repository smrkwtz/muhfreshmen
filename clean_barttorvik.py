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
    The Excel copy-paste has 2 preamble rows (D-I averages) before the
    real header row (Rk, Team, Conf, ...). Find that header row dynamically,
    promote it, then drop rank-comparison and repeated-header rows.
    """
    # Find the row containing "Rk" in the first column — that's the real header
    header_idx = None
    for i, val in enumerate(df_raw.iloc[:, 0]):
        if str(val).strip() == "Rk":
            header_idx = i
            break

    if header_idx is None:
        print(f"  [warn] {year}: could not find header row")
        return pd.DataFrame()

    # Promote that row to column names
    df = df_raw.iloc[header_idx + 1:].copy()
    df.columns = df_raw.iloc[header_idx].tolist()
    df = df.reset_index(drop=True)

    # Drop repeated header rows (where Rk == 'Rk')
    df = df[df["Rk"].astype(str).str.strip() != "Rk"].copy()

    # Keep only rows where Rk is a valid integer (drop rank-comparison rows)
    def is_int(val):
        try:
            int(str(val).strip())
            return True
        except (ValueError, TypeError):
            return False

    df = df[df["Rk"].apply(is_int)].copy()

    # Check for required columns (case-insensitive fallback)
    col_map = {str(c).lower(): c for c in df.columns}
    for needed, lower in [("Team", "team"), ("AdjOE", "adjoe"), ("AdjDE", "adjde")]:
        if needed not in df.columns and lower in col_map:
            df = df.rename(columns={col_map[lower]: needed})

    # Handle "Adj T." column — sometimes "Adj T." sometimes "AdjT"
    if "Adj T." not in df.columns:
        for variant in ("AdjT", "Adj T", "adj t.", "adjt"):
            if variant in col_map:
                df = df.rename(columns={col_map[variant]: "Adj T."})
                break

    required = {"Team", "AdjOE", "AdjDE"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [warn] {year}: missing columns {missing}. Available: {list(df.columns)}")
        return pd.DataFrame()

    # Select and rename
    keep = {"Team": "team", "AdjOE": "ORtg", "AdjDE": "DRtg"}
    if "Adj T." in df.columns:
        keep["Adj T."] = "Pace"
    df = df[list(keep.keys())].rename(columns=keep)

    # Cast to numeric
    for col in ("ORtg", "DRtg", "Pace"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["ORtg", "DRtg"]).copy()
    df["netRtg"] = (df["ORtg"] - df["DRtg"]).round(2)
    df["year"] = year

    cols = ["year", "team", "ORtg", "DRtg", "netRtg"]
    if "Pace" in df.columns:
        cols.insert(4, "Pace")
    return df[cols].reset_index(drop=True)


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

        df_raw = xl.parse(sheet, dtype=str, header=None)
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
