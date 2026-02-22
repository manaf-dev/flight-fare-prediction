"""
Data cleaning logic.

Includes:
- type conversions (dates)
- normalization of categorical fields (stopovers)
- basic sanity checks (non-negative duration/fare)
"""

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "airline",
    "source",
    "destination",
    "departure_date_and_time",
    "stopovers",
    "aircraft_type",
    "seasonality",
    "days_before_departure",
]

NUMERIC_COLUMNS = [
    "duration_hrs",
    "base_fare_bdt",
    "tax_and_surcharge_bdt",
    "total_fare_bdt",
    "days_before_departure",
]


def enforce_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric columns safely.

    Invalid entries become NaN, which we then handle.

    Why:
    CSV imports can silently convert numeric columns to object.
    """
    out = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            before_invalid = out[col].isna().sum()

            out[col] = pd.to_numeric(out[col], errors="coerce")

            after_invalid = out[col].isna().sum()

            if after_invalid > before_invalid:
                logger.warning(
                    f"{col}: introduced {after_invalid - before_invalid} NaNs during numeric coercion"
                )

    return out


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using defensible strategies.

    Numeric: median (robust to outliers)
    Categorical: mode
    """
    out = df.copy()

    missing_summary = out.isna().sum()
    missing_summary = missing_summary[missing_summary > 0]

    if not missing_summary.empty:
        logger.info(f"Missing values detected:\n{missing_summary}")

    # Numeric imputation
    numeric_cols = out.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        if out[col].isna().any():
            median_val = out[col].median()
            out[col] = out[col].fillna(median_val)
            logger.info(f"Imputed numeric column {col} with median={median_val}")

    # Categorical imputation
    cat_cols = out.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        if out[col].isna().any():
            mode_val = out[col].mode()[0]
            out[col] = out[col].fillna(mode_val)
            logger.info(f"Imputed categorical column {col} with mode='{mode_val}'")

    return out


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove logically invalid rows.

    Conditions removed:
    - negative fares
    - zero or negative duration
    """
    out = df.copy()

    before = len(out)

    conditions = (
        (out["total_fare_bdt"] > 0)
        & (out["duration_hrs"] > 0)
        & (out["base_fare_bdt"] >= 0)
        & (out["tax_and_surcharge_bdt"] >= 0)
    )

    out = out[conditions]

    removed = before - len(out)

    if removed > 0:
        logger.warning(f"Removed {removed} invalid rows")

    return out


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert datetime columns safely.
    """
    out = df.copy()

    out["departure_date_and_time"] = pd.to_datetime(
        out["departure_date_and_time"], errors="coerce"
    )

    out["arrival_date_and_time"] = pd.to_datetime(
        out["arrival_date_and_time"], errors="coerce"
    )

    return out


def normalize_stopovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert stopovers text to numeric count.
    """
    out = df.copy()

    mapping = {
        "Direct": 0,
        "1 Stop": 1,
        "2 Stops": 2,
    }

    out["stopovers_count"] = out["stopovers"].map(mapping)

    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master cleaning function.

    Order matters:
    1. Type enforcement
    2. Missing handling
    3. Invalid row removal
    4. Datetime parsing
    5. Feature normalization
    """
    logger.info(f"Starting cleaning: {len(df)} rows")

    out = enforce_numeric_types(df)
    out = handle_missing_values(out)
    out = remove_invalid_rows(out)
    out = parse_datetimes(out)
    out = normalize_stopovers(out)

    logger.info(f"Finished cleaning: {len(out)} rows")

    return out
