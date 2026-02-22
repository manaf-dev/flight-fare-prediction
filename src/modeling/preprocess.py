"""
Preprocessing for modeling.

Implements:
- One-hot encoding for categorical features
- Scaling for numeric features (StandardScaler)
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing transformer:
    - Numeric: median impute + standard scale
    - Categorical: mode impute + one-hot encode

    Args:
        X: Feature dataframe.

    Returns:
        ColumnTransformer ready for a sklearn Pipeline.
    """
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )
