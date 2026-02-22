"""
Cleans the raw DataFrame before feature engineering.

Cleaning steps (in order):
1. Enforce numeric types on fare and duration columns.
2. Impute missing values (median for numerics, mode for categoricals).
3. Remove logically invalid rows (negative fares, zero duration).
4. Derive the target column if it is missing but components are present.

Design decisions:
- Each cleaning step is its own function so it can be tested independently.
- ``clean`` is the public entry point that orchestrates all steps.
- Cleaning is deterministic — no randomness, so results are reproducible.
"""

import pandas as pd

from src.config import LEAKAGE_COLS, TARGET_COL
from src.utils import get_logger

logger = get_logger(__name__)

# Columns that must be numeric for modelling.
NUMERIC_COLS = ["duration_hrs", "days_before_departure", TARGET_COL] + list(
    LEAKAGE_COLS
)


# Step helpers


def _enforce_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce fare and duration columns to float.

    CSV imports often leave these as object when the source file contains
    commas as thousands separators or stray text.
    """
    out = df.copy()
    for col in NUMERIC_COLS:
        if col not in out.columns:
            continue
        # Strip commas, currency symbols, whitespace before coercion.
        cleaned = (
            out[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^\d.\-]", "", regex=True)
        )
        coerced = pd.to_numeric(cleaned, errors="coerce")
        new_nulls = coerced.isna().sum() - out[col].isna().sum()
        if new_nulls > 0:
            logger.warning(
                "Column '%s': %d value(s) became NaN during numeric coercion.",
                col,
                new_nulls,
            )
        out[col] = coerced
    return out


def _derive_target_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    If ``total_fare_bdt`` is missing but both component columns are present,
    derive it as base_fare + tax_and_surcharge.

    This keeps rows we would otherwise have to drop.
    """
    out = df.copy()
    if TARGET_COL not in out.columns:
        return out

    base, tax = LEAKAGE_COLS[0], LEAKAGE_COLS[1]
    if base not in out.columns or tax not in out.columns:
        return out

    missing_mask = out[TARGET_COL].isna()
    if missing_mask.any():
        out.loc[missing_mask, TARGET_COL] = (
            out.loc[missing_mask, base] + out.loc[missing_mask, tax]
        )
        logger.info(
            "Derived %d missing target values from base_fare + tax.", missing_mask.sum()
        )
    return out


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using defensible strategies:
    - Numeric >> median  (robust to outliers)
    - Categorical >> mode
    """
    out = df.copy()

    missing = out.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("Missing values before imputation:\n%s", missing.to_string())

    for col in out.select_dtypes(include=["float64", "int64"]).columns:
        if out[col].isna().any():
            fill = out[col].median()
            out[col] = out[col].fillna(fill)

    for col in out.select_dtypes(include=["object"]).columns:
        if out[col].isna().any():
            fill = (
                out[col].mode(dropna=True)[0]
                if not out[col].mode(dropna=True).empty
                else "Unknown"
            )
            out[col] = out[col].fillna(fill)

    return out


def _remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that are logically impossible:
    - Negative or zero total fare.
    - Zero or negative flight duration.
    - Negative base fare or tax.
    """
    out = df.copy()
    before = len(out)

    mask = pd.Series(True, index=out.index)

    if TARGET_COL in out.columns:
        mask &= out[TARGET_COL] > 0

    if "duration_hrs" in out.columns:
        mask &= out["duration_hrs"] > 0

    for col in LEAKAGE_COLS:
        if col in out.columns:
            mask &= out[col] >= 0

    out = out[mask]
    removed = before - len(out)
    if removed > 0:
        logger.warning("Removed %d logically invalid rows.", removed)
    return out.reset_index(drop=True)


# Public entry point
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps in the correct order.

    Args:
        df: Raw DataFrame from :func:`src.data.load.load_raw`.

    Returns:
        Cleaned DataFrame — types are correct, no missing values,
        no invalid rows.
    """
    logger.info("Starting data cleaning — input shape: %s", df.shape)

    df = _enforce_numeric_types(df)
    df = _derive_target_if_missing(df)
    df = _impute_missing(df)
    df = _remove_invalid_rows(df)

    logger.info("Data cleaning complete — output shape: %s", df.shape)
    return df