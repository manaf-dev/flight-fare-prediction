"""Data preprocessing and feature engineering utilities."""

import re
from typing import Dict

import numpy as np
import pandas as pd

from src.config import DATE_COLS, LEAKAGE_COLS, TARGET_COL
from src.utils import get_logger

logger = get_logger(__name__)

CANONICAL_COLS = [
    "airline",
    "source",
    "destination",
    "stopovers",
    "aircraft_type",
    "class",
    "booking_source",
    "seasonality",
    "departure_datetime",
    "arrival_datetime",
    "duration_hrs",
    "days_before_departure",
    TARGET_COL,
]

COLUMN_ALIASES: Dict[str, str] = {
    "airline": "airline",
    "source": "source",
    "destination": "destination",
    "stopovers": "stopovers",
    "aircraft_type": "aircraft_type",
    "class": "class",
    "booking_source": "booking_source",
    "seasonality": "seasonality",
    "departure_date_and_time": "departure_datetime",
    "departure_datetime": "departure_datetime",
    "arrival_date_and_time": "arrival_datetime",
    "arrival_datetime": "arrival_datetime",
    "duration_hrs": "duration_hrs",
    "duration_hrs_": "duration_hrs",
    "duration": "duration_hrs",
    "days_before_departure": "days_before_departure",
    "total_fare_bdt": TARGET_COL,
    "total_fare": TARGET_COL,
    "base_fare_bdt": "base_fare_bdt",
    "tax_and_surcharge_bdt": "tax_and_surcharge_bdt",
}

CATEGORICAL_BASE = [
    "airline",
    "source",
    "destination",
    "stopovers",
    "aircraft_type",
    "class",
    "booking_source",
    "seasonality",
]
NUMERICAL_BASE = ["duration_hrs", "days_before_departure"]



def _snake_case(name: str) -> str:
    cleaned = name.strip().lower().replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return re.sub(r"_+", "_", cleaned).strip("_")



def _to_numeric(series: pd.Series) -> pd.Series:
    as_str = series.astype(str)
    as_str = as_str.str.replace(",", "", regex=False)
    as_str = as_str.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(as_str, errors="coerce")



def _drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    junk = [col for col in df.columns if col.startswith("unnamed") or col in {"index", "level_0"}]
    if junk:
        logger.info("Dropping junk/index columns: %s", junk)
        df = df.drop(columns=junk, errors="ignore")
    return df



def _rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: COLUMN_ALIASES[col] for col in df.columns if col in COLUMN_ALIASES}
    if renamed:
        df = df.rename(columns=renamed)
    return df



def preprocess_dataframe(df: pd.DataFrame, inference_mode: bool = False) -> pd.DataFrame:
    """Clean and engineer features in a training/inference-compatible way."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    out = df.copy()
    out.columns = [_snake_case(col) for col in out.columns]
    out = _drop_junk_columns(out)
    out = _rename_to_canonical(out)

    # Ensure expected columns exist for a stable schema.
    for col in CANONICAL_COLS:
        if col not in out.columns:
            out[col] = np.nan

    # Parse datetime columns.
    for col in DATE_COLS:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    # Numeric conversions.
    numeric_cols = ["duration_hrs", "days_before_departure", TARGET_COL] + LEAKAGE_COLS
    for col in numeric_cols:
        if col in out.columns:
            out[col] = _to_numeric(out[col])

    # Fill missing target values ONLY if target is missing and both components exist.
    if TARGET_COL in out.columns and all(col in out.columns for col in LEAKAGE_COLS):
        missing_target = out[TARGET_COL].isna()
        if missing_target.any():
            derived_target = out[LEAKAGE_COLS[0]] + out[LEAKAGE_COLS[1]]
            out.loc[missing_target, TARGET_COL] = derived_target.loc[missing_target]
            logger.info("Filled %s missing target values using base+tax components", int(missing_target.sum()))

    # Feature engineering from departure datetime.
    out["departure_month"] = out["departure_datetime"].dt.month
    out["departure_day"] = out["departure_datetime"].dt.day
    out["departure_hour"] = out["departure_datetime"].dt.hour
    out["departure_weekday"] = out["departure_datetime"].dt.weekday
    out["is_weekend"] = out["departure_weekday"].isin([5, 6]).astype(float)
    out["is_peak_hour"] = out["departure_hour"].isin([6, 7, 8, 9, 17, 18, 19, 20, 21]).astype(float)

    out["source"] = out["source"].astype(str)
    out["destination"] = out["destination"].astype(str)
    out["route"] = out["source"].fillna("Unknown") + "_" + out["destination"].fillna("Unknown")
    out["route_frequency"] = out["route"].map(out["route"].value_counts(dropna=False)).astype(float)

    # Missing values.
    for col in CATEGORICAL_BASE + ["route"]:
        out[col] = out[col].replace({"nan": np.nan, "None": np.nan}).fillna("Unknown")

    num_fill_cols = [
        "duration_hrs",
        "days_before_departure",
        "departure_month",
        "departure_day",
        "departure_hour",
        "departure_weekday",
        "is_weekend",
        "is_peak_hour",
        "route_frequency",
    ]
    for col in num_fill_cols:
        if col in out.columns:
            if inference_mode:
                fallback = out[col].median() if out[col].notna().any() else 0.0
            else:
                fallback = out[col].median()
            out[col] = out[col].fillna(fallback)

    # Validate negatives.
    invalid_duration = (out["duration_hrs"] < 0).sum()
    if invalid_duration > 0:
        if inference_mode:
            out.loc[out["duration_hrs"] < 0, "duration_hrs"] = 0
            logger.info("Clipped %s negative duration values in inference mode", int(invalid_duration))
        else:
            out = out[out["duration_hrs"] >= 0]
            logger.info("Removed %s rows with negative duration", int(invalid_duration))

    if TARGET_COL in out.columns:
        invalid_target = (out[TARGET_COL] < 0).sum()
        if invalid_target > 0:
            if inference_mode:
                out.loc[out[TARGET_COL] < 0, TARGET_COL] = np.nan
            else:
                out = out[out[TARGET_COL] >= 0]
                logger.info("Removed %s rows with negative target fare", int(invalid_target))

    # Training mode: require valid datetime and target.
    if not inference_mode:
        before = len(out)
        out = out.dropna(subset=["departure_datetime", TARGET_COL])
        dropped = before - len(out)
        if dropped > 0:
            logger.info("Dropped %s rows with invalid departure datetime/target", dropped)

    return out.reset_index(drop=True)
