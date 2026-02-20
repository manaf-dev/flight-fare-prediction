"""
Feature engineering for the flight fare dataset.

Implements the project-required engineered features:
- month, day, weekday, season from departure_date_and_time
- safe handling if datetime parsing fails
"""

import pandas as pd


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create month, day, weekday from departure_date_and_time.

    Args:
        df: Cleaned dataframe with 'departure_date_and_time' parsed as datetime.

    Returns:
        Dataframe with new columns:
        - dep_month
        - dep_day
        - dep_weekday
    """
    out = df.copy()

    if "departure_date_and_time" not in out.columns:
        return out

    dt = out["departure_date_and_time"]

    # If dt isn't datetime yet, attempt coercion
    if not pd.api.types.is_datetime64_any_dtype(dt):
        out["departure_date_and_time"] = pd.to_datetime(dt, errors="coerce")
        dt = out["departure_date_and_time"]

    out["dep_month"] = dt.dt.month
    out["dep_day"] = dt.dt.day
    out["dep_weekday"] = dt.dt.dayofweek  # 0=Mon ... 6=Sun

    return out


def map_month_to_season(month: int) -> str:
    """
    Map month -> season label.

    This is a generic, defensible season mapping:
    - Winter: Dec-Feb
    - Spring: Mar-May
    - Summer: Jun-Aug
    - Autumn: Sep-Nov
    """
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def add_season_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'season' from dep_month.

    Args:
        df: Dataframe with dep_month.

    Returns:
        Dataframe with 'season' categorical column.
    """
    out = df.copy()

    if "dep_month" not in out.columns:
        return out

    out["season"] = out["dep_month"].apply(
        lambda m: map_month_to_season(int(m)) if pd.notna(m) else "unknown"
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate feature engineering steps.

    Args:
        df: Cleaned dataframe.

    Returns:
        Feature-enriched dataframe.
    """
    out = df.copy()
    out = add_date_features(out)
    out = add_season_feature(out)
    return out
