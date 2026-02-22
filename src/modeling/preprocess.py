"""
Builds the sklearn ``ColumnTransformer`` that handles numeric scaling and
categorical one-hot encoding inside the model pipeline.

Kept separate from ``train.py`` so the same preprocessor can be reused in
cross-validation, tuning, and the saved inference artifact.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """
    Build a ``ColumnTransformer`` that:
    - Numeric columns: median-impute → StandardScaler
    - Categorical columns: mode-impute → OneHotEncoder (ignores unseen values)

    Only columns that are both in ``feature_cols`` AND in the project feature
    lists are included — this prevents silent failures when a column is absent.

    Args:
        feature_cols: List of columns available in the training DataFrame.

    Returns:
        Unfitted ``ColumnTransformer`` ready to be placed in a ``Pipeline``.
    """
    numerical = [f for f in NUMERICAL_FEATURES if f in feature_cols]
    categorical = [f for f in CATEGORICAL_FEATURES if f in feature_cols]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numerical),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",  # silently drop any unrecognised columns
    )

    return preprocessor