"""
All feature engineering lives here.

Features created:
- Temporal: month, day, hour, weekday, is_weekend, is_peak_hour
  (extracted from ``departure_datetime``).
- Route: ``route`` string (source → destination) and ``route_frequency``
  (how many flights operate that route — a proxy for demand).
"""

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES
from src.utils import get_logger

logger = get_logger(__name__)

# Hours considered "peak" for air travel (morning rush + evening rush).
PEAK_HOURS = {6, 7, 8, 18, 19, 20, 21}


# Individual feature functions
def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar and clock features from ``departure_datetime``."""
    out = df.copy()

    dt = out.get("departure_datetime")
    if dt is None:
        logger.warning("'departure_datetime' missing — temporal features will be NaN.")
        for col in [
            "departure_month",
            "departure_day",
            "departure_hour",
            "departure_weekday",
            "is_weekend",
            "is_peak_hour",
        ]:
            out[col] = np.nan
        return out

    out["departure_month"] = dt.dt.month
    out["departure_day"] = dt.dt.day
    out["departure_hour"] = dt.dt.hour
    out["departure_weekday"] = dt.dt.weekday  # 0=Mon … 6=Sun
    out["is_weekend"] = out["departure_weekday"].isin([5, 6]).astype(float)
    out["is_peak_hour"] = out["departure_hour"].isin(PEAK_HOURS).astype(float)

    return out


def _add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ``route`` (source_destination) and ``route_frequency``.

    ``route_frequency`` is a count of how often that route appears in the
    *training* data — it acts as a demand proxy.  In inference mode this
    is filled from a pre-computed lookup (see ``build_inference``).
    """
    out = df.copy()

    src = out.get("source", pd.Series("Unknown", index=out.index)).astype(str)
    dest = out.get("destination", pd.Series("Unknown", index=out.index)).astype(str)

    out["route"] = src + "_" + dest
    out["route_frequency"] = (
        out["route"].map(out["route"].value_counts(dropna=False)).astype(float)
    )

    return out


def _fill_missing_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs introduced during feature engineering with sensible defaults."""
    out = df.copy()

    numeric_eng = [
        "departure_month",
        "departure_day",
        "departure_hour",
        "departure_weekday",
        "is_weekend",
        "is_peak_hour",
        "route_frequency",
    ]
    for col in numeric_eng:
        if col in out.columns and out[col].isna().any():
            fill = out[col].median() if out[col].notna().any() else 0.0
            out[col] = out[col].fillna(fill)

    for col in CATEGORICAL_FEATURES + ["route"]:
        if col in out.columns:
            out[col] = (
                out[col].replace({"nan": np.nan, "None": np.nan}).fillna("Unknown")
            )

    return out


# Public entry points


def build(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to a *training* DataFrame.

    Args:
        df: Cleaned DataFrame from :func:`src.data.clean.clean`.

    Returns:
        DataFrame with all engineered features appended.
    """
    logger.info("Building features — input shape: %s", df.shape)

    df = _add_temporal_features(df)
    df = _add_route_features(df)
    df = _fill_missing_engineered(df)

    logger.info("Feature engineering complete — output shape: %s", df.shape)
    return df


def build_inference(payload: dict, route_freq_map: dict | None = None) -> pd.DataFrame:
    """
    Convert a single prediction payload dict into a feature DataFrame.

    Identical transformations to ``build`` but:
    - Accepts a dict instead of a DataFrame.
    - Uses a pre-computed ``route_freq_map`` for ``route_frequency``
      instead of re-computing from data (which isn't available at inference time).

    Args:
        payload: Dict with at minimum the keys required by ``ALL_FEATURES``.
        route_freq_map: Dict mapping route strings to their training frequency.
                        If None, ``route_frequency`` defaults to 1.

    Returns:
        Single-row DataFrame ready for the trained pipeline.
    """
    row = dict(payload)

    # Parse departure_datetime if provided as string.
    dt_str = row.get("departure_datetime") or row.get("departure_date_and_time", "")
    try:
        dt = pd.to_datetime(dt_str)
    except Exception:
        dt = pd.NaT

    row["departure_datetime"] = dt

    row["departure_month"] = dt.month if pd.notna(dt) else 1
    row["departure_day"] = dt.day if pd.notna(dt) else 1
    row["departure_hour"] = dt.hour if pd.notna(dt) else 12
    row["departure_weekday"] = dt.weekday() if pd.notna(dt) else 0
    row["is_weekend"] = float(row["departure_weekday"] in [5, 6])
    row["is_peak_hour"] = float(row["departure_hour"] in PEAK_HOURS)

    src = str(row.get("source", "Unknown"))
    dest = str(row.get("destination", "Unknown"))
    route = f"{src}_{dest}"
    row["route"] = route
    row["route_frequency"] = float((route_freq_map or {}).get(route, 1))

    return pd.DataFrame([row])