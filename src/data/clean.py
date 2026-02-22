"""
Cleans the raw DataFrame before feature engineering.
"""

import pandas as pd

from src.config import LEAKAGE_COLS, TARGET_COL
from src.utils import get_logger

logger = get_logger(__name__)

NUMERIC_COLS = ["duration_hrs", "days_before_departure", TARGET_COL] + list(
    LEAKAGE_COLS
)

# City / airport name normalisation map
CITY_ALIAS_MAP = {
    "dacca": "DAC",
    "dhaka": "DAC",
    "dac": "DAC",
    "chittagong": "CGP",
    "chattogram": "CGP",
    "cgp": "CGP",
    "sylhet": "ZYL",
    "zyl": "ZYL",
    "cox's bazar": "CXB",
    "coxs bazar": "CXB",
    "cxb": "CXB",
    "jessore": "JSR",
    "jashore": "JSR",
    "jsr": "JSR",
    "rajshahi": "RJH",
    "rjh": "RJH",
    "barisal": "BZL",
    "bzl": "BZL",
    "saidpur": "SPD",
    "spd": "SPD",
}


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows. Keeps first occurrence."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        logger.info("Dropped %d duplicate rows.", removed)
    return df.reset_index(drop=True)


def _enforce_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce fare and duration columns to float, handling commas and stray text."""
    out = df.copy()
    for col in NUMERIC_COLS:
        if col not in out.columns:
            continue
        cleaned = (
            out[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^\d.\-]", "", regex=True)
        )
        coerced = pd.to_numeric(cleaned, errors="coerce")
        new_nulls = int(coerced.isna().sum()) - int(out[col].isna().sum())
        if new_nulls > 0:
            logger.warning(
                "Column '%s': %d value(s) became NaN during coercion.", col, new_nulls
            )
        out[col] = coerced
    return out


def _normalize_city_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise inconsistent city/airport name variants to IATA codes.
    Example: 'Dhaka', 'Dacca', 'dhaka' → 'DAC'
    """
    out = df.copy()
    for col in ["source", "destination"]:
        if col not in out.columns:
            continue
        out[col] = (
            out[col]
            .astype(str)
            .str.strip()
            .apply(lambda v: CITY_ALIAS_MAP.get(v.lower(), v))
        )
    return out


def _derive_target_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Derive total_fare_bdt = base_fare + tax where target is missing."""
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
            "Derived %d missing target values from base_fare + tax.",
            int(missing_mask.sum()),
        )
    return out


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numerics, mode for categoricals."""
    out = df.copy()
    missing = out.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("Missing values before imputation:\n%s", missing.to_string())

    for col in out.select_dtypes(include=["float64", "int64"]).columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    for col in out.select_dtypes(include=["object"]).columns:
        if out[col].isna().any():
            fill = out[col].mode(dropna=True)
            out[col] = out[col].fillna(fill[0] if not fill.empty else "Unknown")
    return out


def _remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with negative/zero fares or zero/negative duration."""
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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps in the correct order.

    Args:
        df: Raw DataFrame from load_raw().
    Returns:
        Cleaned DataFrame — correct types, no duplicates, no missing values,
        no invalid rows, city names normalised.
    """
    logger.info("Starting data cleaning — input shape: %s", df.shape)
    df = _drop_duplicates(df)
    df = _enforce_numeric_types(df)
    df = _normalize_city_names(df)
    df = _derive_target_if_missing(df)
    df = _impute_missing(df)
    df = _remove_invalid_rows(df)
    logger.info("Data cleaning complete — output shape: %s", df.shape)
    return df