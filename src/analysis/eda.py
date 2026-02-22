"""
Exploratory Data Analysis — descriptive statistics, KPIs, and visualisations.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive — safe for scripts and servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, TARGET_COL, VIZ_DIR
from src.utils import get_logger

logger = get_logger(__name__)

VIZ_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, filename: str) -> None:
    path = VIZ_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def _save_csv(df: pd.DataFrame, filename: str) -> None:
    path = REPORTS_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved table: %s", path)


def summarise_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """count / mean / median / min / max of total fare per group, sorted by mean."""
    return (
        df.groupby(group_col)[TARGET_COL]
        .agg(count="count", mean="mean", median="median", min="min", max="max")
        .reset_index()
        .sort_values("mean", ascending=False)
    )


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix for all numeric columns."""
    return df.select_dtypes(include=[np.number]).corr()


# KPI tables
def kpi_avg_fare_per_airline(df: pd.DataFrame) -> pd.DataFrame:
    """Average total fare ranked by airline."""
    return (
        df.groupby("airline")[TARGET_COL]
        .mean()
        .reset_index(name="avg_fare_bdt")
        .sort_values("avg_fare_bdt", ascending=False)
    )


def kpi_popular_routes(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top N routes by flight count."""
    return (
        df.assign(
            route_label=df["source"].astype(str) + " → " + df["destination"].astype(str)
        )["route_label"]
        .value_counts()
        .head(top_n)
        .reset_index()
        .rename(columns={"route_label": "flight_count", "index": "route"})
    )


def kpi_seasonal_variation(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and median fare by seasonality label."""
    return (
        df.groupby("seasonality")[TARGET_COL]
        .agg(count="count", mean="mean", median="median")
        .reset_index()
        .sort_values("mean", ascending=False)
    )


def kpi_expensive_routes(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Top N routes by average fare."""
    return (
        df.assign(
            route_label=df["source"].astype(str) + " → " + df["destination"].astype(str)
        )
        .groupby("route_label")[TARGET_COL]
        .mean()
        .reset_index(name="avg_fare_bdt")
        .sort_values("avg_fare_bdt", ascending=False)
        .head(top_n)
    )


# Visualisations
def plot_fare_distributions(df: pd.DataFrame) -> None:
    """
    Separate histograms for total fare, base fare, and tax & surcharge.
    Requirement Step 3: 'Plot distributions of fares, base fares, and taxes.'
    """
    cols_labels = [
        ("total_fare_bdt", "Total Fare (BDT)", "fare_distribution_total.png"),
        ("base_fare_bdt", "Base Fare (BDT)", "fare_distribution_base.png"),
        ("tax_and_surcharge_bdt", "Tax & Surcharge (BDT)", "fare_distribution_tax.png"),
    ]
    for col, label, fname in cols_labels:
        if col not in df.columns:
            logger.warning("Skipping distribution plot: '%s' not found.", col)
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(
            df[col].dropna(), bins=60, edgecolor="white", alpha=0.85, color="steelblue"
        )
        ax.set_title(f"Distribution of {label}", fontsize=14)
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        _save_fig(fig, fname)


def plot_fare_by_airline(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Box-plot of fare spread per airline (top N by frequency).
    Requirement: 'Use boxplots to show fare variation across airlines.'
    """
    top = df["airline"].value_counts().head(top_n).index.tolist()
    data = [df[df["airline"] == a][TARGET_COL].dropna().values for a in top]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.boxplot(data, labels=top, showfliers=False, patch_artist=True)
    ax.set_xticklabels(top, rotation=40, ha="right")
    ax.set_title(f"Fare Distribution by Airline (Top {top_n})", fontsize=14)
    ax.set_ylabel("Total Fare (BDT)")
    plt.tight_layout()
    _save_fig(fig, "fare_by_airline.png")


def plot_fare_by_season_boxplot(df: pd.DataFrame) -> None:
    """
    Boxplot of fare variation across seasons.
    Requirement suggested viz: 'Fare Variation Across Seasons (boxplot)'.
    Uses the 'seasonality' column (commercial season: Eid, Hajj, etc.)
    """
    if "seasonality" not in df.columns:
        logger.warning("Skipping season boxplot: 'seasonality' column missing.")
        return

    seasons = df["seasonality"].unique().tolist()
    data = [df[df["seasonality"] == s][TARGET_COL].dropna().values for s in seasons]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, labels=seasons, showfliers=False, patch_artist=True)
    ax.set_title("Fare Variation Across Seasons (Boxplot)", fontsize=14)
    ax.set_ylabel("Total Fare (BDT)")
    ax.set_xlabel("Season")
    plt.tight_layout()
    _save_fig(fig, "fare_by_season_boxplot.png")


def plot_avg_fare_by_month(df: pd.DataFrame) -> None:
    """Bar chart of average fare per departure month."""
    if "departure_month" not in df.columns:
        return
    monthly = df.groupby("departure_month")[TARGET_COL].mean().sort_index()
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(monthly.index, monthly.values, color="steelblue")
    ax.set_xticks(monthly.index)
    ax.set_xticklabels([month_labels[m - 1] for m in monthly.index])
    ax.set_title("Average Total Fare by Departure Month", fontsize=14)
    ax.set_ylabel("Average Fare (BDT)")
    _save_fig(fig, "fare_by_month.png")


def plot_avg_fare_by_season(df: pd.DataFrame) -> None:
    """Horizontal bar chart of average fare per seasonality label."""
    if "seasonality" not in df.columns:
        return
    series = df.groupby("seasonality")[TARGET_COL].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(series.index.tolist(), series.values, color="steelblue")
    ax.set_title("Average Total Fare by Seasonality", fontsize=14)
    ax.set_xlabel("Average Fare (BDT)")
    _save_fig(fig, "fare_by_season.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Colour-coded heatmap of numeric feature correlations."""
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=14)
    plt.tight_layout()
    _save_fig(fig, "correlation_heatmap.png")


def plot_fare_components(df: pd.DataFrame) -> None:
    """
    Scatter of (base_fare + tax) vs total_fare — for business explanation only.
    These columns are NOT used as model predictors (leakage).
    """
    base_col, tax_col = "base_fare_bdt", "tax_and_surcharge_bdt"
    if not {base_col, tax_col, TARGET_COL}.issubset(df.columns):
        return
    sample = (
        df[[base_col, tax_col, TARGET_COL]]
        .dropna()
        .sample(n=min(2000, len(df)), random_state=42)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(sample[base_col] + sample[tax_col], sample[TARGET_COL], alpha=0.25, s=10)
    ax.set_xlabel("Base Fare + Tax & Surcharge (BDT)")
    ax.set_ylabel("Total Fare (BDT)")
    ax.set_title(
        "Fare Components vs Total Fare (EDA only — not used as predictors)", fontsize=12
    )
    _save_fig(fig, "fare_components_vs_total.png")


# Orchestrator
def run_eda(df: pd.DataFrame) -> None:
    """
    Run the full EDA suite and save all outputs.

    Tables → reports/
    Figures → visualizations/
    """
    logger.info("Running EDA...")

    # Descriptive tables
    for col in ["airline", "source", "destination", "seasonality"]:
        if col in df.columns:
            _save_csv(summarise_by_group(df, col), f"summary_by_{col}.csv")

    _save_csv(correlation_matrix(df).reset_index(), "correlation_matrix.csv")

    # KPI tables
    _save_csv(kpi_avg_fare_per_airline(df), "kpi_avg_fare_per_airline.csv")
    _save_csv(kpi_popular_routes(df), "kpi_popular_routes_top20.csv")
    _save_csv(kpi_seasonal_variation(df), "kpi_seasonal_variation.csv")
    _save_csv(kpi_expensive_routes(df), "kpi_expensive_routes_top5.csv")

    # Visualisations
    plot_fare_distributions(df)  # separate histograms for total, base, tax
    plot_fare_by_airline(df)  # boxplot by airline
    plot_fare_by_season_boxplot(df)  # boxplot by season (required viz)
    plot_avg_fare_by_month(df)  # bar chart by month
    plot_avg_fare_by_season(df)  # bar chart by season
    plot_correlation_heatmap(df)  # heatmap
    plot_fare_components(df)  # scatter EDA-only

    logger.info("EDA complete.")