"""Data loading module for the Flight Fare Prediction project."""

import re

import pandas as pd

from src.config import DATA_PATH, DATE_COLS
from src.data_preprocessing import preprocess_dataframe
from src.utils import get_logger

logger = get_logger(__name__)


def _to_snake_case(value: str) -> str:
    cleaned = value.strip().lower().replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return re.sub(r"_+", "_", cleaned).strip("_")


def load_data(path=DATA_PATH):
    """Load raw CSV robustly, normalize headers, and log basic diagnostics."""
    try:
        df = pd.read_csv(path)
        df.columns = [_to_snake_case(col) for col in df.columns]

        for date_col in DATE_COLS + ["departure_date_and_time", "arrival_date_and_time"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        logger.info("Dataset loaded from %s", path)
        logger.info("Raw shape: %s", df.shape)
        logger.info("Null counts:\n%s", df.isnull().sum().sort_values(ascending=False))
        return df
    except Exception:
        logger.exception("Error loading dataset from %s", path)
        raise


def get_dataframe(path=DATA_PATH):
    """Return cleaned dataframe with canonical names/features for downstream tasks."""
    raw_df = load_data(path)
    clean_df = preprocess_dataframe(raw_df, inference_mode=False)
    logger.info("Canonical dataframe shape: %s", clean_df.shape)
    return clean_df


def inspect_data(df):
    """Log quick inspection statistics."""
    logger.info("Dataset info:\n%s", df.info())
    logger.info("Head:\n%s", df.head().to_string())
    logger.info("Describe:\n%s", df.describe(include="all").transpose().head(30).to_string())
