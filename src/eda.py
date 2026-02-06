"""
Exploratory Data Analysis module for the Flight Fare Prediction project.
Performs descriptive statistics and visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import VISUALIZATIONS_PATH
from src.utils import get_logger

logger = get_logger(__name__)


def descriptive_statistics(df):
    """
    Performs descriptive statistics on the dataset.

    Args:
        df (pd.DataFrame): The dataset.
    """
    logger.info("Performing descriptive statistics...")

    # Summary by airline
    logger.info("Average fare by airline:")
    logger.info(
        df.groupby("airline")["total_fare_bdt"].mean().sort_values(ascending=False)
    )

    # Summary by source and destination
    logger.info("Average fare by source:")
    logger.info(
        df.groupby("source")["total_fare_bdt"].mean().sort_values(ascending=False)
    )

    logger.info("Average fare by destination:")
    logger.info(
        df.groupby("destination")["total_fare_bdt"].mean().sort_values(ascending=False)
    )

    # Summary by season
    logger.info("Average fare by seasonality:")
    logger.info(
        df.groupby("seasonality")["total_fare_bdt"].mean().sort_values(ascending=False)
    )


def create_visualizations(df):
    """
    Creates visualizations for EDA.

    Args:
        df (pd.DataFrame): The dataset.
    """
    logger.info("Creating visualizations...")

    # Set style
    sns.set_style("whitegrid")

    # Distribution of total fare
    plt.figure(figsize=(10, 6))
    sns.histplot(df["total_fare_bdt"], bins=50, kde=True)
    plt.title("Distribution of Total Fare")
    plt.savefig(f"{VISUALIZATIONS_PATH}/fare_distribution.png")
    plt.close()

    # Boxplot of fare by airline
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="airline", y="total_fare_bdt", data=df)
    plt.xticks(rotation=45)
    plt.title("Fare Variation by Airline")
    plt.savefig(f"{VISUALIZATIONS_PATH}/fare_by_airline.png")
    plt.close()

    # Average fare by month
    plt.figure(figsize=(10, 6))
    monthly_fare = df.groupby("departure_month")["total_fare_bdt"].mean()
    monthly_fare.plot(kind="bar")
    plt.title("Average Fare by Month")
    plt.savefig(f"{VISUALIZATIONS_PATH}/fare_by_month.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{VISUALIZATIONS_PATH}/correlation_heatmap.png")
    plt.close()

    logger.info("Visualizations created and saved.")


def kpi_exploration(df):
    """
    Explores key performance indicators.

    Args:
        df (pd.DataFrame): The dataset.
    """
    logger.info("Exploring KPIs...")

    # Average fare per airline
    avg_fare_airline = (
        df.groupby("airline")["total_fare_bdt"].mean().sort_values(ascending=False)
    )
    logger.info(f"Average fare per airline:\n{avg_fare_airline}")

    # Most popular routes
    popular_routes = (
        df.groupby(["source", "destination"])
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
    logger.info(f"Top 5 popular routes:\n{popular_routes}")

    # Seasonal fare variation
    seasonal_fare = (
        df.groupby("seasonality")["total_fare_bdt"].mean().sort_values(ascending=False)
    )
    logger.info(f"Seasonal fare variation:\n{seasonal_fare}")

    # Top 5 most expensive routes
    expensive_routes = (
        df.groupby(["source", "destination"])["total_fare_bdt"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    logger.info(f"Top 5 most expensive routes:\n{expensive_routes}")
    logger.info(f"Top 5 most expensive routes:\n{expensive_routes}")
    logger.info(f"Top 5 most expensive routes:\n{expensive_routes}")
