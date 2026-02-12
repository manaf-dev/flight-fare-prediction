"""
Data preprocessing module for the Flight Fare Prediction project.
Handles data cleaning, feature engineering, and encoding.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.utils import get_logger

logger = get_logger(__name__)


def clean_data(df):
    """
    Cleans the dataset by handling missing values and correcting invalid entries.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    logger.info("Starting data cleaning...")

    # Remove duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")

    # Handle missing values
    # For numerical columns, impute with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(
                f"Imputed {df[col].isnull().sum()} missing values in {col} with median: {median_val}"
            )

    # For categorical columns, impute with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(
                f"Imputed {df[col].isnull().sum()} missing values in {col} with mode: {mode_val}"
            )

    # Correct invalid entries
    # Ensure fares are positive
    fare_cols = ["base_fare_bdt", "tax_and_surcharge_bdt", "total_fare_bdt"]
    for col in fare_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                df = df[df[col] >= 0]
                logger.info(f"Removed {negative_count} rows with negative {col}")

    # Convert date columns to datetime
    date_cols = ["departure_date_and_time", "arrival_date_and_time"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop rows with invalid dates
    invalid_dates = df[date_cols].isnull().any(axis=1).sum()
    if invalid_dates > 0:
        df = df.dropna(subset=date_cols)
        logger.info(f"Removed {invalid_dates} rows with invalid dates")

    logger.info(f"Data cleaning completed. Final shape: {df.shape}")
    return df


def feature_engineering(df):
    """
    Performs feature engineering on the cleaned dataset.

    Args:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        pd.DataFrame: Dataset with engineered features.
    """
    logger.info("Starting feature engineering...")

    # Extract date features
    df["departure_date_and_time"] = pd.to_datetime(df["departure_date_and_time"])
    df["arrival_date_and_time"] = pd.to_datetime(df["arrival_date_and_time"])

    df["departure_month"] = df["departure_date_and_time"].dt.month
    df["departure_day"] = df["departure_date_and_time"].dt.day
    df["departure_hour"] = df["departure_date_and_time"].dt.hour
    df["departure_weekday"] = df["departure_date_and_time"].dt.weekday

    # Create season based on month
    def get_season(month):
        if month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"
        else:
            return "Winter"

    df["season"] = df["departure_month"].apply(get_season)

    # Calculate duration in hours if not present or recalculate
    if "duration_hrs" not in df.columns or df["duration_hrs"].isnull().any():
        df["duration_hrs"] = (
            df["arrival_date_and_time"] - df["departure_date_and_time"]
        ).dt.total_seconds() / 3600

    # Ensure Total Fare is correct
    df["calculated_total_fare"] = df["base_fare_bdt"] + df["tax_and_surcharge_bdt"]
    fare_diff = (df["total_fare_bdt"] - df["calculated_total_fare"]).abs()
    if fare_diff.max() > 1:  # Allow small differences due to rounding
        df["total_fare_bdt"] = df["calculated_total_fare"]
        logger.warning(
            "Total Fare doesn't match Base Fare + Tax in some rows. Replaced them with calculated_total_fare"
        )

    # Drop unnecessary columns
    cols_to_drop = [
        "source_name",
        "destination_name",
        "departure_date_and_time",
        "arrival_date_and_time",
        "calculated_total_fare",
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    logger.info(f"Feature engineering completed. New shape: {df.shape}")
    return df


def encode_and_scale(df):
    """
    Encodes categorical variables and scales numerical features.

    Args:
        df (pd.DataFrame): Dataset with engineered features.

    Returns:
        tuple: (processed_df, scaler, encoders)
    """
    logger.info("Starting encoding and scaling...")

    # Separate features and target
    target = "total_fare_bdt"
    X = df.drop(columns=[target])
    y = df[target]

    # Define categorical and numerical features
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    numerical_features = [col for col in NUMERICAL_FEATURES if col in X.columns]

    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features: {numerical_features}")

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    # Get feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = []
    for i, col in enumerate(categorical_features):
        categories = cat_encoder.categories_[i][1:]  # Drop first category
        cat_feature_names.extend([f"{col}_{cat}" for cat in categories])

    feature_names = numerical_features + cat_feature_names

    # Create processed DataFrame
    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    processed_df[target] = y.values

    logger.info(
        f"Encoding and scaling completed. Processed shape: {processed_df.shape}"
    )
    # describe df
    logger.info(f"Processed DataFrame:\n{processed_df.describe()}")
    return (
        processed_df,
        preprocessor.named_transformers_["num"],
        preprocessor.named_transformers_["cat"],
    )
