"""
EDA (Exploratory Data Analysis) for Flight Fare Prediction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.utils import get_logger

logger = get_logger(__name__)


def _ensure_output_dirs() -> None:
    """Ensure output directories exist."""
    CONFIG.figures_dir.mkdir(exist_ok=True, parents=True)
    CONFIG.metrics_dir.mkdir(exist_ok=True, parents=True)


def _safe_filename(name: str) -> str:
    """Make a safe filename fragment."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _save_table(df: pd.DataFrame, filename: str) -> Path:
    """Save a table as CSV under outputs/metrics."""
    _ensure_output_dirs()
    path = CONFIG.metrics_dir / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved table: {path}")
    return path


def _save_figure(fig, filename: str) -> Path:
    """Save matplotlib figure under outputs/figures."""
    _ensure_output_dirs()
    path = CONFIG.figures_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")
    return path


# Descriptive Statistics
def summarize_fares_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Summarize total fare statistics by a categorical grouping column.

    Args:
        df: Input dataframe (expects 'total_fare_bdt').
        group_col: Column to group by (e.g., airline, Source, Destination, seasonality).

    Returns:
        Summary dataframe with count, mean, median, min, max.
    """
    summary = (
        df.groupby(group_col)["total_fare_bdt"]
        .agg(count="count", mean="mean", median="median", min="min", max="max")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return summary


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric features.

    Note: Correlation is used for multicollinearity diagnostics; it doesn't apply to
    categorical variables directly.

    Returns:
        Correlation matrix dataframe.
    """
    num_df = df.select_dtypes(include=["int64", "float64"]).copy()
    corr = num_df.corr(numeric_only=True)
    return corr


# Visual Analysis
def plot_distributions(df: pd.DataFrame) -> None:
    """
    Plot distributions for:
    - Total Fare
    - Base Fare
    - Tax & Surcharge

    Even though Base/Tax are leakage features for modeling Total Fare,
    they are still valid for EDA to understand price composition.
    """
    cols = ["total_fare_bdt", "base_fare_bdt", "tax_and_surcharge_bdt"]

    for col in cols:
        if col not in df.columns:
            logger.warning(f"Skipping distribution plot: missing column '{col}'")
            continue

        fig = plt.figure(figsize=(8, 5))
        plt.hist(df[col].dropna(), bins=50)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        _save_figure(fig, f"dist_{_safe_filename(col)}.png")


def boxplot_fare_by_airline(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Boxplot fare variation across airlines (top N airlines by frequency).

    Args:
        df: Input dataframe.
        top_n: Number of most frequent airlines to include for readability.
    """
    if "airline" not in df.columns:
        logger.warning("Skipping boxplot: 'airline' column missing")
        return

    top_airlines = df["airline"].value_counts().head(top_n).index
    subset = df[df["airline"].isin(top_airlines)].copy()

    fig = plt.figure(figsize=(12, 6))
    plt.boxplot(
        [subset[subset["airline"] == a]["total_fare_bdt"].values for a in top_airlines],
        tick_labels=top_airlines,
        showfliers=False,
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Fare Variation by Airline (Top {top_n} by Frequency)")
    plt.ylabel("total_fare_bdt")

    _save_figure(fig, f"boxplot_fare_by_airline_top_{top_n}.png")


def avg_fare_by_month(df: pd.DataFrame) -> None:
    """
    Plot average fare by departure month.

    Requires a parsed datetime column: 'departure_date_and_time'
    """
    if "departure_date_and_time" not in df.columns:
        logger.warning("Skipping avg fare by month: missing 'departure_date_and_time'")
        return

    if not np.issubdtype(df["departure_date_and_time"].dtype, np.datetime64):
        logger.warning(
            "Skipping avg fare by month: 'departure_date_and_time' not datetime"
        )
        return

    tmp = df.copy()
    tmp["dep_month"] = tmp["departure_date_and_time"].dt.month

    series = tmp.groupby("dep_month")["total_fare_bdt"].mean().sort_index()

    fig = plt.figure(figsize=(8, 5))
    plt.plot(series.index, series.values, marker="o")
    plt.title("Average Total Fare by Departure Month")
    plt.xlabel("Month")
    plt.ylabel("Average total_fare_bdt")
    plt.xticks(range(1, 13))

    _save_figure(fig, "avg_fare_by_month.png")


def avg_fare_by_season(df: pd.DataFrame) -> None:
    """
    Plot average fare by seasonality.
    """
    if "seasonality" not in df.columns:
        logger.warning("Skipping avg fare by season: missing 'seasonality'")
        return

    series = df.groupby("seasonality")["total_fare_bdt"].mean().sort_values()

    fig = plt.figure(figsize=(10, 5))
    plt.barh(series.index, series.values)
    plt.title("Average Total Fare by Seasonality")
    plt.xlabel("Average total_fare_bdt")

    _save_figure(fig, "avg_fare_by_seasonality.png")


def correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot correlation heatmap for numeric features.

    Uses matplotlib only (no seaborn dependency).
    """
    corr = correlation_table(df)

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()

    _save_figure(fig, "correlation_heatmap.png")


# KPI Exploration
def kpi_average_fare_per_airline(df: pd.DataFrame) -> pd.DataFrame:
    """
    KPI: Average fare per airline.
    """
    kpi = (
        df.groupby("airline")["total_fare_bdt"]
        .mean()
        .reset_index(name="avg_total_fare_bdt")
        .sort_values("avg_total_fare_bdt", ascending=False)
    )
    return kpi


def kpi_most_popular_route(df: pd.DataFrame) -> pd.DataFrame:
    """
    KPI: Most popular route by flight frequency.

    Route = Source -> Destination
    """
    tmp = df.copy()
    tmp["route"] = tmp["source"].astype(str) + "->" + tmp["destination"].astype(str)

    kpi = (
        tmp["route"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "route", "route": "flight_count"})
        .sort_values("flight_count", ascending=False)
    )
    return kpi


def kpi_seasonal_variation(df: pd.DataFrame) -> pd.DataFrame:
    """
    KPI: Seasonal fare variation (mean/median fare per season).
    """
    kpi = (
        df.groupby("seasonality")["total_fare_bdt"]
        .agg(count="count", mean="mean", median="median")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return kpi


def kpi_top_expensive_routes(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    KPI: Top N most expensive routes by average fare.
    """
    tmp = df.copy()
    tmp["route"] = tmp["source"].astype(str) + "->" + tmp["destination"].astype(str)

    kpi = (
        tmp.groupby("route")["total_fare_bdt"]
        .mean()
        .reset_index(name="avg_total_fare_bdt")
        .sort_values("avg_total_fare_bdt", ascending=False)
        .head(top_n)
    )
    return kpi


def eda_main(df: pd.DataFrame) -> None:
    """
    Run the full EDA suite and save artifacts.

    Outputs:
    - Tables (CSV) in outputs/metrics/
    - Figures (PNG) in outputs/figures/

    Args:
        df: Cleaned dataframe (ideally with parsed datetimes).
    """
    logger.info("Starting EDA...")

    # ---- Descriptive statistics tables ----
    for col in ["airline", "Source", "Destination", "seasonality"]:
        if col in df.columns:
            summary = summarize_fares_by_group(df, col)
            _save_table(summary, f"summary_fares_by_{_safe_filename(col)}.csv")
        else:
            logger.warning(f"Skipping group summary: missing '{col}'")

    # Correlations table
    corr = correlation_table(df)
    corr_path = CONFIG.metrics_dir / "correlation_matrix.csv"
    corr.to_csv(corr_path)
    logger.info(f"Saved table: {corr_path}")

    # Visual analysis
    plot_distributions(df)
    boxplot_fare_by_airline(df, top_n=15)
    avg_fare_by_month(df)
    avg_fare_by_season(df)
    correlation_heatmap(df)

    # KPI exploration tables
    if "airline" in df.columns:
        _save_table(kpi_average_fare_per_airline(df), "kpi_avg_fare_per_airline.csv")

    _save_table(
        kpi_most_popular_route(df).head(20), "kpi_most_popular_routes_top20.csv"
    )
    if "seasonality" in df.columns:
        _save_table(kpi_seasonal_variation(df), "kpi_seasonal_variation.csv")

    _save_table(kpi_top_expensive_routes(df, top_n=5), "kpi_top_5_expensive_routes.csv")

    logger.info("EDA completed successfully.")
