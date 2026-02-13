"""Exploratory Data Analysis module for the Flight Fare Prediction project."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import TARGET_COL, VISUALIZATIONS_PATH
from src.utils import get_logger

logger = get_logger(__name__)



def descriptive_statistics(df):
    logger.info("Performing descriptive statistics...")
    logger.info("Average fare by airline:\n%s", df.groupby("airline")[TARGET_COL].mean().sort_values(ascending=False))
    logger.info("Average fare by source:\n%s", df.groupby("source")[TARGET_COL].mean().sort_values(ascending=False))
    logger.info("Average fare by destination:\n%s", df.groupby("destination")[TARGET_COL].mean().sort_values(ascending=False))
    logger.info("Average fare by seasonality:\n%s", df.groupby("seasonality")[TARGET_COL].mean().sort_values(ascending=False))



def create_visualizations(df):
    logger.info("Creating EDA visualizations...")
    VISUALIZATIONS_PATH.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(df[TARGET_COL], bins=50, kde=True)
    plt.title("Distribution of Total Fare")
    plt.savefig(VISUALIZATIONS_PATH / "fare_distribution.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.boxplot(x="airline", y=TARGET_COL, data=df)
    plt.xticks(rotation=45, ha="right")
    plt.title("Fare Variation by Airline")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_PATH / "fare_by_airline.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    monthly_fare = df.groupby("departure_month")[TARGET_COL].mean()
    monthly_fare.plot(kind="bar")
    plt.title("Average Fare by Departure Month")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_PATH / "fare_by_month.png")
    plt.close()

    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_PATH / "correlation_heatmap.png")
    plt.close()

    # Component view is acceptable for business explanation, not modeling.
    if {"base_fare_bdt", "tax_and_surcharge_bdt", TARGET_COL}.issubset(df.columns):
        sample = df[["base_fare_bdt", "tax_and_surcharge_bdt", TARGET_COL]].sample(
            n=min(1200, len(df)), random_state=42
        )
        plt.figure(figsize=(10, 6))
        plt.scatter(sample["base_fare_bdt"] + sample["tax_and_surcharge_bdt"], sample[TARGET_COL], alpha=0.3)
        plt.xlabel("Base + Tax (component view)")
        plt.ylabel("Total Fare")
        plt.title("Fare Components vs Total Fare (EDA only)")
        plt.tight_layout()
        plt.savefig(VISUALIZATIONS_PATH / "fare_components_vs_total.png")
        plt.close()

    logger.info("Visualizations created and saved.")



def kpi_exploration(df):
    logger.info("Exploring KPIs...")
    avg_fare_airline = df.groupby("airline")[TARGET_COL].mean().sort_values(ascending=False)
    popular_routes = df.groupby(["source", "destination"]).size().sort_values(ascending=False).head(5)
    seasonal_fare = df.groupby("seasonality")[TARGET_COL].mean().sort_values(ascending=False)
    expensive_routes = (
        df.groupby(["source", "destination"])[TARGET_COL].mean().sort_values(ascending=False).head(5)
    )

    logger.info("Average fare per airline:\n%s", avg_fare_airline)
    logger.info("Top 5 popular routes:\n%s", popular_routes)
    logger.info("Seasonal fare variation:\n%s", seasonal_fare)
    logger.info("Top 5 expensive routes:\n%s", expensive_routes)
