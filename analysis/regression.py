"""
Regression analysis: does freshman reliance explain tournament spread deltas?

We run four models:
  1. spread_delta ~ fr_min_share_diff          (differential only, minutes)
  2. spread_delta ~ fr_pts_share_diff          (differential only, points)
  3. spread_delta ~ fr_min_share_team1 + fr_min_share_team2  (separate effects, minutes)
  4. spread_delta ~ fr_pts_share_team1 + fr_pts_share_team2  (separate effects, points)

Models 3 & 4 let us distinguish whether the freshman effect is symmetric
(favored team underperforms AND underdog team outperforms) or asymmetric.

Outputs:
  - Printed regression summaries
  - analysis/figures/scatter_*.png  (scatter plots with fit line)
  - analysis/figures/coef_plot.png  (coefficient forest plot)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for CI / server environments
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

DATA_PATH = Path("data/processed/analysis_dataset.csv")
FIGURES_DIR = Path("analysis/figures")


def load_analysis_data(exclude_first_four: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    if exclude_first_four:
        df = df[~df["is_first_four"].fillna(False)]

    # Keep only games where we have all required data
    required = [
        "spread_delta", "expected_margin",
        "fr_min_share_team1", "fr_min_share_team2",
        "fr_pts_share_team1", "fr_pts_share_team2",
    ]
    df = df.dropna(subset=required).copy()
    print(f"Analysis sample: {len(df)} games across {df['year'].nunique()} seasons")
    return df


def run_regressions(df: pd.DataFrame) -> dict:
    """Fit OLS models and return results dict."""
    models = {
        "min_diff": smf.ols("spread_delta ~ fr_min_share_diff", data=df).fit(),
        "pts_diff": smf.ols("spread_delta ~ fr_pts_share_diff", data=df).fit(),
        "min_separate": smf.ols(
            "spread_delta ~ fr_min_share_team1 + fr_min_share_team2", data=df
        ).fit(),
        "pts_separate": smf.ols(
            "spread_delta ~ fr_pts_share_team1 + fr_pts_share_team2", data=df
        ).fit(),
    }
    return models


def print_summaries(models: dict) -> None:
    for name, result in models.items():
        print(f"\n{'=' * 70}")
        print(f"Model: {name}")
        print(result.summary())


def plot_scatter(df: pd.DataFrame, x_col: str, label: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=df,
        x=x_col,
        y="spread_delta",
        ax=ax,
        scatter_kws={"alpha": 0.3, "s": 20},
        line_kws={"color": "crimson"},
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel("Spread Delta (actual − expected, from favored team perspective)")
    ax.set_title(f"Tournament Spread Delta vs. {label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_coefficient_forest(models: dict, output_path: Path) -> None:
    """Simple coefficient plot for the differential models."""
    rows = []
    for name in ("min_diff", "pts_diff"):
        result = models[name]
        param_name = "fr_min_share_diff" if name == "min_diff" else "fr_pts_share_diff"
        coef = result.params[param_name]
        ci_low, ci_high = result.conf_int().loc[param_name]
        label = "Fr. Min-Share Diff" if name == "min_diff" else "Fr. Pts-Share Diff"
        rows.append({"label": label, "coef": coef, "ci_low": ci_low, "ci_high": ci_high})

    cdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, r in cdf.iterrows():
        ax.plot([r.ci_low, r.ci_high], [i, i], color="steelblue", linewidth=2)
        ax.scatter([r.coef], [i], color="steelblue", zorder=3, s=60)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(cdf)))
    ax.set_yticklabels(cdf["label"])
    ax.set_xlabel("OLS Coefficient (points)")
    ax.set_title("Effect of Freshman Reliance Differential on Spread Delta\n(95% CI)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_round_breakdown(df: pd.DataFrame, x_col: str, label: str, output_path: Path) -> None:
    """
    Faceted scatter by tournament round - freshman effects may be stronger
    in early rounds (larger seed gaps, more 1-seeds vs 16-seeds) or late
    rounds (higher pressure, larger stage).
    """
    round_order = [
        "First Round", "Second Round", "Sweet 16",
        "Elite Eight", "Final Four", "Championship",
    ]
    df_plot = df[df["round"].isin(round_order)].copy()
    df_plot["round"] = pd.Categorical(df_plot["round"], categories=round_order, ordered=True)

    g = sns.FacetGrid(df_plot, col="round", col_wrap=3, height=3, sharey=True)
    g.map_dataframe(sns.regplot, x=x_col, y="spread_delta",
                    scatter_kws={"alpha": 0.25, "s": 12}, line_kws={"color": "crimson"})
    g.set_axis_labels(label, "Spread Delta")
    g.figure.suptitle(f"Spread Delta vs. {label} by Round", y=1.02)
    g.figure.tight_layout()
    g.figure.savefig(output_path, dpi=150)
    plt.close(g.figure)
    print(f"  Saved {output_path}")


def run_all(exclude_first_four: bool = True) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_analysis_data(exclude_first_four=exclude_first_four)
    models = run_regressions(df)
    print_summaries(models)

    print("\nGenerating figures...")
    plot_scatter(df, "fr_min_share_diff",
                 "Fr. Min-Share Diff (favored − underdog)",
                 FIGURES_DIR / "scatter_min_share_diff.png")
    plot_scatter(df, "fr_pts_share_diff",
                 "Fr. Pts-Share Diff (favored − underdog)",
                 FIGURES_DIR / "scatter_pts_share_diff.png")
    plot_coefficient_forest(models, FIGURES_DIR / "coef_plot.png")
    plot_round_breakdown(df, "fr_min_share_diff",
                         "Fr. Min-Share Diff",
                         FIGURES_DIR / "round_breakdown_min.png")
    plot_round_breakdown(df, "fr_pts_share_diff",
                         "Fr. Pts-Share Diff",
                         FIGURES_DIR / "round_breakdown_pts.png")

    # Quick summary table
    print("\nSummary of differential model coefficients:")
    print(f"{'Metric':<25} {'Coef':>8} {'p-value':>10} {'R²':>8}")
    print("-" * 55)
    for name, label in [("min_diff", "Fr. Min-Share Diff"), ("pts_diff", "Fr. Pts-Share Diff")]:
        r = models[name]
        param = "fr_min_share_diff" if name == "min_diff" else "fr_pts_share_diff"
        print(
            f"{label:<25} {r.params[param]:>8.2f} {r.pvalues[param]:>10.4f} {r.rsquared:>8.4f}"
        )


if __name__ == "__main__":
    run_all()
