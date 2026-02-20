"""
Training module.

Implements:
- train/test split (80/20)
- pipeline: preprocessing + model
- baseline model evaluation
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import CONFIG
from src.modeling.evaluate import regression_metrics
from src.modeling.preprocess import build_preprocessor


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X/y for modeling while preventing target leakage.

    Drops:
    - target column
    - leakage columns: base_fare_bdt, tax_surcharge_bdt
    - redundant columns in CONFIG.drop_cols
    - raw datetimes after feature extraction

    Args:
        df: Feature-engineered dataframe.

    Returns:
        X, y
    """
    y = df[CONFIG.target_col].copy()

    # already engineered date features, drop raw datetime fields
    cols_to_drop = [CONFIG.target_col, "departure_date_and_time"]
    cols_to_drop += [c for c in CONFIG.leakage_cols if c in df.columns]
    cols_to_drop += [c for c in CONFIG.drop_cols if c in df.columns]

    X = df.drop(columns=cols_to_drop, errors="ignore").copy()
    return X, y


def train_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train multiple models and return a comparison table.

    Models:
    - LinearRegression baseline
    - Ridge, Lasso (regularization)
    - RandomForest (non-linear)

    Returns:
        DataFrame with model metrics (R2, MAE, RMSE).
    """
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
    )

    preprocessor = build_preprocessor(X_train)

    candidates: Dict[str, object] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=CONFIG.random_state),
        "lasso": Lasso(alpha=0.001, random_state=CONFIG.random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=CONFIG.random_state, n_jobs=-1
        ),
    }

    rows = []
    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        predictions = pipe.predict(X_test)
        m = regression_metrics(y_test, predictions)

        rows.append({"model": name, "r2": m["r2"], "mae": m["mae"], "rmse": m["rmse"]})

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
