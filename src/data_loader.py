"""
Data loading module for the Flight Fare Prediction project.
Handles loading and initial inspection of the dataset.
"""

import pandas as pd

from src.config import DATA_PATH
from src.utils import get_logger

logger = get_logger(__name__)


def load_data():
    """
    Loads the flight price dataset from CSV.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        # Clean column names
        df = clean_column_names(df)
        # Validate required columns
        required_columns = [
            "airline",
            "source",
            "destination",
            "departure_date_and_time",
            "arrival_date_and_time",
            "duration_hrs",
            "stopovers",
            "aircraft_type",
            "class",
            "booking_source",
            "base_fare_bdt",
            "tax_and_surcharge_bdt",
            "total_fare_bdt",
            "seasonality",
            "days_before_departure",
        ]
        validate_required_columns(df, required_columns)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def clean_column_names(df):
    """
    Cleans column names by removing spaces and special characters.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: Dataset with cleaned column names.
    """
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("&", "and")
        .str.lower()
    )
    logger.info("Column names cleaned.")
    return df


def validate_required_columns(df, required_columns):
    """
    Validates that required columns are present in the dataset.

    Args:
        df (pd.DataFrame): The dataset.
        required_columns (list): List of required column names.

    Raises:
        ValueError: If any required column is missing.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    logger.info("All required columns are present.")


def inspect_data(df):
    """
    Performs initial inspection of the dataset.

    Args:
        df (pd.DataFrame): The dataset to inspect.
    """
    logger.info("Dataset Info:")
    logger.info(df.info())
    logger.info("\nDataset Description:")
    logger.info(df.describe())
    logger.info("\nFirst 5 rows:")
    logger.info(df.head())
    logger.info(f"\nMissing values:\n{df.isnull().sum()}")
    logger.info(f"\nData types:\n{df.dtypes}")
