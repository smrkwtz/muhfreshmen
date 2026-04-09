"""
Scrapes per-player season stats (with class year) from sports-reference.com
for every team that appeared in a given year's NCAA tournament.

For each team-season we compute:
  - fr_min_share  : fraction of team minutes played by freshmen
  - fr_pts_share  : fraction of team points scored by freshmen

Output columns:
  year, team_id, team_name, fr_min_share, fr_pts_share,
  fr_min_raw, fr_pts_raw, total_min, total_pts, n_fr_players
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from constants import TOURNAMENT_YEARS

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research/academic project)"}
REQUEST_DELAY = 12  # seconds between requests — sports-reference rate-limits hard from CI IPs
MAX_RETRIES = 4     # retries on 429 with exponential backoff

OUTPUT_PATH = Path("data/raw/player_class_stats.csv")

# Class labels used by sports-reference
FRESHMAN_LABELS = {"fr", "freshman"}
ALL_CLASS_LABELS = {"fr", "so", "jr", "sr", "freshman", "sophomore", "junior", "senior"}


def _norm(name: str) -> str:
    """Normalize a player name for matching across tables.
    Removes Jr./Sr./II/III/IV suffixes, punctuation, and extra whitespace.
    'Mikel Brown Jr.' and 'Mikel Brown' both become 'mikel brown'.
    """
    name = name.lower()
    name = re.sub(r'\b(jr|sr|ii|iii|iv|v)\.?\b', '', name)
    name = re.sub(r'[^a-z0-9\s]', '', name)  # strip punctuation / apostrophes / hyphens
    return ' '.join(name.split())


def _idx(headers: list[str], names: tuple) -> int | None:
    """Return the first matching index from a list of candidate data-stat names."""
    for name in names:
        try:
            return headers.index(name)
        except ValueError:
            pass
    return None


def _cell(cells: list, idx: int | None) -> str:
    if idx is None or idx >= len(cells):
        return ""
    return cells[idx].get_text(strip=True)


def _table_headers(table) -> list[str]:
    return [c.get("data-stat", "") for c in table.find("thead").find_all(["th", "td"])]


def _find_table(soup: BeautifulSoup, *ids: str):
    """Find a table by ID, checking DOM first then HTML comments."""
    for tid in ids:
        tbl = soup.find("table", id=tid)
        if tbl:
            return tbl
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if not any(tid in comment for tid in ids):
            continue
        inner = BeautifulSoup(comment, "lxml")
        for tid in ids:
            tbl = inner.find("table", id=tid)
            if tbl:
                return tbl
    return None


def _parse_team_page(html: str, year: int, team_id: str) -> dict | None:
    """
    Parse a team season page and return the freshman-reliance metrics dict.

    Sports-reference structure (current):
      - `roster` table (DOM): has player name + class year (FR/SO/JR/SR)
      - `players_per_game` table (DOM): has player name + games + mp_per_g + pts_per_g
    We join them on player name to combine class and stats.
    Older page formats may still have class_ directly in the per-game table.
    """
    soup = BeautifulSoup(html, "lxml")

    title = soup.find("h1", {"itemprop": "name"})
    team_name = title.get_text(strip=True) if title else team_id

    # ── Step 1: build player → class year map from roster table ─────────────
    class_map: dict[str, str] = {}
    roster_tbl = _find_table(soup, "roster")
    if roster_tbl:
        h = _table_headers(roster_tbl)
        name_col = _idx(h, ("player", "name_display"))
        cls_col  = _idx(h, ("class_", "class", "yr"))
        if name_col is not None and cls_col is not None:
            for row in roster_tbl.find("tbody").find_all("tr"):
                cells = row.find_all(["td", "th"])
                name = _norm(_cell(cells, name_col))
                cls  = _cell(cells, cls_col).lower().strip()
                if name and cls:
                    class_map[name] = cls

    # ── Step 2: get per-game stats from players_per_game ────────────────────
    stats_tbl = _find_table(
        soup,
        "players_per_game", "per_game", "season-total_per_game",
    )
    if stats_tbl is None:
        ids = [t.get("id", "?") for t in soup.find_all("table")]
        print(f"    [debug] stats table not found. DOM ids={ids[:12]}")
        return None

    h = _table_headers(stats_tbl)
    s_name  = _idx(h, ("name_display", "player"))
    s_games = _idx(h, ("games", "g"))
    s_mp    = _idx(h, ("mp_per_g", "mp"))
    s_pts   = _idx(h, ("pts_per_g", "pts"))
    # Older formats may still carry class_ directly in the stats table
    s_cls   = _idx(h, ("class_", "class", "yr"))

    if s_games is None or s_mp is None or s_pts is None:
        print(f"    [debug] missing stat cols in {stats_tbl.get('id')}. headers={h[:20]}")
        return None

    total_min = fr_min = total_pts = fr_pts = 0.0
    n_fr = 0

    for row in stats_tbl.find("tbody").find_all("tr"):
        if "thead" in (row.get("class") or []):
            continue
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        # Class year: direct column (old format) or roster lookup (new format)
        if s_cls is not None:
            cls = _cell(cells, s_cls).lower().strip()
        else:
            player_name = _norm(_cell(cells, s_name)) if s_name is not None else ""
            cls = class_map.get(player_name, "").lower().strip()

        if not cls or cls not in ALL_CLASS_LABELS:
            # Warn if an active player can't be class-matched — indicates a name
            # mismatch between roster and per-game tables.
            if not cls and s_name is not None:
                raw_name = _cell(cells, s_name)
                try:
                    g_check = float(_cell(cells, s_games) or 0)
                    mp_check = float(_cell(cells, s_mp) or 0)
                except ValueError:
                    g_check = mp_check = 0
                if g_check >= 5 and mp_check >= 5:
                    print(f"    [warn] unmatched player '{raw_name}' ({g_check}g, {mp_check}mpg) — name mismatch?")
            continue

        try:
            g   = float(_cell(cells, s_games) or 0)
            mp  = float(_cell(cells, s_mp)    or 0)
            pts = float(_cell(cells, s_pts)   or 0)
        except ValueError:
            continue

        total_min += mp * g
        total_pts += pts * g
        if cls in FRESHMAN_LABELS:
            fr_min += mp * g
            fr_pts += pts * g
            n_fr += 1

    if total_min == 0 and total_pts == 0:
        print(f"    [debug] no valid rows. roster entries={len(class_map)}, s_cls={s_cls}")
        return None

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


def _diagnose_response(resp, label: str = "") -> None:
    """Print diagnostic info about an unexpected response body (bot page, etc.)."""
    body = resp.text
    snippet = body[:400].replace("\n", " ").strip()
    table_count_dom = body.count('<table')
    comment_count = body.count('<!--')
    has_cf = "cloudflare" in body.lower() or "cf-ray" in resp.headers.get("server", "").lower()
    has_sref = "sports-reference" in body.lower() or "sr-nav" in body.lower()
    print(
        f"    [diag{' ' + label if label else ''}] "
        f"status={resp.status_code} tables={table_count_dom} "
        f"comments={comment_count} cloudflare={has_cf} sref={has_sref}"
    )
    print(f"    [diag snippet] {snippet[:300]}")


def fetch_team_stats(
    team_id: str,
    year: int,
    session: requests.Session,
    verbose: bool = False,
) -> dict | None:
    urls = [
        f"https://www.sports-reference.com/cbb/schools/{team_id}/men/{year}.html",
        f"https://www.sports-reference.com/cbb/schools/{team_id}/{year}.html",
    ]
    for url in urls:
        for attempt in range(MAX_RETRIES):
            resp = session.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
            if resp.status_code == 404:
                break  # try next URL
            if resp.status_code == 429:
                wait = 60 * (2 ** attempt)  # 60s, 120s, 240s, 480s
                print(f"    429 rate-limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code not in (200,):
                print(f"    HTTP {resp.status_code} from {url}")
                resp.raise_for_status()
            result = _parse_team_page(resp.text, year, team_id)
            if result:
                return result
            # Parse returned None — log diagnostics so we can see what arrived
            _diagnose_response(resp, label=f"{team_id}/{year}")
            break  # try next URL
    return None


def collect_player_class_stats(
    tournament_results_path: Path = Path("data/raw/tournament_results.csv"),
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Load the tournament results to get the list of teams, then fetch each
    team's season page to extract freshman reliance metrics.
    """
    if not tournament_results_path.exists():
        raise FileNotFoundError(
            f"Run tournament_results.py first to create {tournament_results_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(tournament_results_path)

    # Build a deduplicated list of (year, team_id, team_name) for all tourney teams
    team_cols_1 = results[["year", "team1_id", "team1"]].rename(
        columns={"team1_id": "team_id", "team1": "team_name"}
    )
    team_cols_2 = results[["year", "team2_id", "team2"]].rename(
        columns={"team2_id": "team_id", "team2": "team_name"}
    )
    teams = (
        pd.concat([team_cols_1, team_cols_2])
        .dropna(subset=["team_id"])
        .drop_duplicates(subset=["year", "team_id"])
        .sort_values(["year", "team_id"])
        .reset_index(drop=True)
    )

    print(f"Fetching player class stats for {len(teams)} team-seasons...")
    session = requests.Session()
    all_rows = []

    for _, row in teams.iterrows():
        year, team_id, team_name = int(row.year), row.team_id, row.team_name
        print(f"  {year} {team_name} ({team_id})", end=" ... ", flush=True)
        try:
            data = fetch_team_stats(team_id, year, session)
            if data:
                all_rows.append(data)
                print(f"fr_min={data['fr_min_share']:.1%}  fr_pts={data['fr_pts_share']:.1%}")
            else:
                print("no data")
        except requests.HTTPError as e:
            print(f"HTTP {e.response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} team-seasons → {output_path}")
    return df


if __name__ == "__main__":
    collect_player_class_stats()
