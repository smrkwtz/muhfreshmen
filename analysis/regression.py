"""
Regression analysis: does freshman reliance predict tournament over/underperformance?

Unit of analysis: game-team (two observations per game, one per team).
Response variable: spread_delta from that team's perspective
  positive = beat pre-tournament expectations, negative = fell short.

Key model (per stats friend's suggestion):
  spread_delta ~ netRtg_c * fr_min_share

where netRtg_c is netRtg centered at its mean. The interaction term is the
main coefficient of interest: do highly-rated freshman-heavy teams systematically
underperform their efficiency rating in the tournament?

Standard errors are clustered by game_id because the two observations from the
same game are exact negatives of each other (not independent).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

DATA_PATH   = Path("data/processed/analysis_dataset.csv")
FIGURES_DIR = Path("analysis/figures")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(exclude_first_four: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if exclude_first_four:
        df = df[~df["is_first_four"].fillna(False)]

    df = df.dropna(subset=["spread_delta", "netRtg", "fr_min_share", "fr_pts_share"]).copy()

    # Center netRtg so the fr_min_share main effect is interpretable at
    # average team quality (not at netRtg = 0, which is a bad team)
    df["netRtg_c"] = df["netRtg"] - df["netRtg"].mean()

    print(f"Analysis sample: {len(df)} team-game observations")
    print(f"  {df['game_id'].nunique()} games, {df['year'].nunique()} seasons")
    print(f"  netRtg mean (all teams in sample): {df['netRtg'].mean():.1f}")
    return df


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def run_models(df: pd.DataFrame) -> dict:
    """
    Fit interaction models with game-clustered standard errors.
    Two variants: minutes share and points share.
    """
    kwargs = dict(cov_type="cluster", cov_kwds={"groups": df["game_id"]})
    return {
        "min": smf.ols("spread_delta ~ netRtg_c * fr_min_share", data=df).fit(**kwargs),
        "pts": smf.ols("spread_delta ~ netRtg_c * fr_pts_share", data=df).fit(**kwargs),
    }


def print_summaries(models: dict) -> None:
    labels = {"min": "Minutes share", "pts": "Points share"}
    for key, result in models.items():
        print(f"\n{'=' * 70}")
        print(f"Freshman metric: {labels[key]}")
        print(result.summary())


def print_summary_table(models: dict) -> None:
    fr_vars = {"min": "netRtg_c:fr_min_share", "pts": "netRtg_c:fr_pts_share"}
    print("\nKey interaction coefficients (clustered SEs):")
    print(f"{'Metric':<20} {'Interaction coef':>18} {'p-value':>10} {'R²':>8}")
    print("-" * 60)
    for key, result in models.items():
        iv = fr_vars[key]
        label = "netRtg × fr_min_share" if key == "min" else "netRtg × fr_pts_share"
        print(
            f"{label:<20} {result.params[iv]:>18.4f} "
            f"{result.pvalues[iv]:>10.4f} {result.rsquared:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_interaction(df: pd.DataFrame, fr_col: str, label: str, output_path: Path) -> None:
    """
    Visualise the interaction by splitting teams into high/low netRtg groups.
    If highly-rated freshman teams underperform, the high-netRtg group should
    show a steeper negative slope.
    """
    median_netRtg = df["netRtg"].median()
    df = df.copy()
    df["Quality"] = np.where(df["netRtg"] >= median_netRtg, "High netRtg", "Low netRtg")

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"High netRtg": "steelblue", "Low netRtg": "tomato"}
    for group, gdf in df.groupby("Quality"):
        color = palette[group]
        ax.scatter(gdf[fr_col], gdf["spread_delta"],
                   alpha=0.2, s=15, color=color, label="_nolegend_")
        # Fit line for this group
        m, b = np.polyfit(gdf[fr_col], gdf["spread_delta"], 1)
        x = np.linspace(gdf[fr_col].min(), gdf[fr_col].max(), 100)
        ax.plot(x, m * x + b, color=color, linewidth=2, label=f"{group} (slope={m:.1f})")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel("Spread delta (actual − expected, team's perspective)")
    ax.set_title(f"Tournament performance vs. {label}\nSplit by team quality (median netRtg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_interaction_coefs(models: dict, output_path: Path) -> None:
    """Coefficient forest plot for the interaction terms."""
    rows = []
    for key, result in models.items():
        iv = "netRtg_c:fr_min_share" if key == "min" else "netRtg_c:fr_pts_share"
        coef = result.params[iv]
        ci_low, ci_high = result.conf_int().loc[iv]
        rows.append({
            "label": "netRtg × fr_min_share" if key == "min" else "netRtg × fr_pts_share",
            "coef": coef, "ci_low": ci_low, "ci_high": ci_high,
            "p": result.pvalues[iv],
        })

    cdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, r in cdf.iterrows():
        color = "steelblue" if r.p < 0.05 else "gray"
        ax.plot([r.ci_low, r.ci_high], [i, i], color=color, linewidth=2.5)
        ax.scatter([r.coef], [i], color=color, zorder=3, s=80)
        ax.text(r.ci_high + 0.002, i, f"p={r.p:.3f}", va="center", fontsize=9)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(cdf)))
    ax.set_yticklabels(cdf["label"])
    ax.set_xlabel("Interaction coefficient (clustered SEs, 95% CI)")
    ax.set_title("Effect of team quality × freshman reliance on spread delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_round_breakdown(df: pd.DataFrame, fr_col: str, label: str, output_path: Path) -> None:
    round_order = ["First Round", "Second Round", "Sweet 16",
                   "Elite Eight", "Final Four", "Championship"]
    df_plot = df[df["round"].isin(round_order)].copy()
    df_plot["round"] = pd.Categorical(df_plot["round"], categories=round_order, ordered=True)
    g = sns.FacetGrid(df_plot, col="round", col_wrap=3, height=3, sharey=True)
    g.map_dataframe(sns.regplot, x=fr_col, y="spread_delta",
                    scatter_kws={"alpha": 0.2, "s": 10}, line_kws={"color": "crimson"})
    g.set_axis_labels(label, "Spread delta")
    g.figure.suptitle(f"Spread delta vs. {label} by round", y=1.02)
    g.figure.tight_layout()
    g.figure.savefig(output_path, dpi=150)
    plt.close(g.figure)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all(exclude_first_four: bool = True) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(exclude_first_four=exclude_first_four)
    models = run_models(df)
    print_summaries(models)
    print_summary_table(models)

    print("\nGenerating figures...")
    plot_interaction(df, "fr_min_share", "Fr. minutes share",
                     FIGURES_DIR / "interaction_min_share.png")
    plot_interaction(df, "fr_pts_share", "Fr. points share",
                     FIGURES_DIR / "interaction_pts_share.png")
    plot_interaction_coefs(models, FIGURES_DIR / "coef_plot.png")
    plot_round_breakdown(df, "fr_min_share", "Fr. minutes share",
                         FIGURES_DIR / "round_breakdown_min.png")
    plot_round_breakdown(df, "fr_pts_share", "Fr. points share",
                         FIGURES_DIR / "round_breakdown_pts.png")


if __name__ == "__main__":
    run_all()
