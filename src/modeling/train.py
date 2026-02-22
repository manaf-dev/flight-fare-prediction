"""
Model training, cross-validation, hyperparameter tuning, and selection.
"""

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    CV_RESULTS_PATH,
    FEAT_IMPORTANCE_PLOT,
    LEAKAGE_COLS,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    PIPELINE_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SIZE,
)
from src.modeling.evaluate import compute_metrics
from src.modeling.preprocess import build_preprocessor
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Candidate model definitions
# ---------------------------------------------------------------------------


def _get_baseline() -> dict:
    """
    Return the Step 4 baseline model: plain Linear Regression.

    LinearRegression has no hyperparameters to tune and serves as the
    lower-bound reference. Every other model should outperform it;
    if it doesn't, that's a signal to investigate data or features.
    """
    return {"LinearRegression": LinearRegression()}


def _get_advanced_candidates() -> dict:
    """
    Return Step 5 advanced models in order of increasing complexity.

    Order matters for logging clarity — simpler models first.
    """
    return {
        "Ridge": Ridge(),
        "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=20_000),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=10,  # constrained to avoid extreme overfitting
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=140,
            max_depth=16,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
    }


def _get_all_candidates() -> dict:
    """Return baseline + advanced candidates combined (baseline first)."""
    return {**_get_baseline(), **_get_advanced_candidates()}


def _tuning_space(model_name: str) -> dict:
    """
    Return the hyperparameter search space for RandomizedSearchCV.

    Only tree models are tuned because linear models have negligible gain
    from extensive tuning compared to the cost.
    """
    spaces = {
        "RandomForest": {
            "model__n_estimators": randint(180, 450),
            "model__max_depth": [None, 10, 14, 18, 24],
            "model__min_samples_split": randint(2, 14),
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": ["sqrt", "log2", None],
        },
        "HistGradientBoosting": {
            "model__max_depth": [None, 6, 8, 12],
            "model__learning_rate": uniform(0.01, 0.2),
            "model__max_iter": randint(120, 450),
            "model__l2_regularization": uniform(0.0, 1.5),
        },
    }
    if model_name not in spaces:
        raise ValueError(f"No tuning space defined for model '{model_name}'.")
    return spaces[model_name]


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------


def _chronological_split(df: pd.DataFrame, test_size: float):
    """
    Sort by departure_datetime and split the last ``test_size`` fraction as test.

    Raises:
        AssertionError: If any training timestamp is after the first test timestamp.
    """
    sorted_df = df.sort_values("departure_datetime").reset_index(drop=True)
    cut = int(len(sorted_df) * (1 - test_size))
    cut = max(1, min(cut, len(sorted_df) - 1))

    train = sorted_df.iloc[:cut].copy()
    test = sorted_df.iloc[cut:].copy()

    assert train["departure_datetime"].max() <= test["departure_datetime"].min(), (
        "Chronological split violated: training data contains timestamps after test start."
    )
    logger.info(
        "Chronological split — train: %d rows, test: %d rows. "
        "Train ends: %s | Test starts: %s",
        len(train),
        len(test),
        train["departure_datetime"].max().date(),
        test["departure_datetime"].min().date(),
    )
    return train, test


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------


def _plot_feature_importance(pipeline: Pipeline, top_n: int = 15) -> list:
    """
    Extract, plot, and save the top-N feature importances from a fitted pipeline.

    Strategy:
    - Tree models (RandomForest, DecisionTree, HistGradientBoosting):
        use native ``feature_importances_`` — reliable and fast.
    - Linear models (LinearRegression, Ridge, Lasso):
        use absolute coefficient values — magnitude = influence.

    The preprocessor is called with ``transform=False`` via
    ``get_feature_names_out()`` which works on an already-fitted
    ColumnTransformer without needing new data.

    Returns:
        List of dicts: [{"feature": name, "importance": value}, ...]
        Empty list if importances cannot be extracted.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # --- Extract feature names from the fitted preprocessor ---
    try:
        raw_names = list(preprocessor.get_feature_names_out())
    except Exception as exc:
        logger.warning("Could not get feature names from preprocessor: %s", exc)
        return []

    # Clean up sklearn's prefix format: "num__duration_hrs" → "duration_hrs"
    names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]

    # --- Extract importance values ---
    if hasattr(model, "feature_importances_"):
        # Tree-based models
        values = model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        # Linear models — use absolute coefficient magnitude
        coef = model.coef_
        values = np.abs(coef).tolist()
    else:
        logger.warning(
            "Model '%s' has neither feature_importances_ nor coef_.",
            type(model).__name__,
        )
        return []

    if len(names) != len(values):
        logger.warning(
            "Feature name/importance length mismatch: %d names vs %d values. Skipping plot.",
            len(names),
            len(values),
        )
        return []

    # --- Sort by importance descending, take top_n ---
    actual_top = min(top_n, len(values))
    order = np.argsort(values)[::-1][:actual_top]
    top_names = [names[i] for i in order]
    top_values = [values[i] for i in order]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, max(5, actual_top * 0.5)))
    y_pos = range(actual_top)
    ax.barh(list(y_pos), top_values[::-1], color="steelblue")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_names[::-1])
    ax.set_title(
        f"Top {actual_top} Feature Importances — {type(model).__name__}", fontsize=13
    )
    ax.set_xlabel("Importance")
    plt.tight_layout()

    FEAT_IMPORTANCE_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FEAT_IMPORTANCE_PLOT, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance plot: %s", FEAT_IMPORTANCE_PLOT)

    result = [
        {"feature": n, "importance": float(v)} for n, v in zip(top_names, top_values)
    ]
    logger.info(
        "Top 5 features: %s",
        [(r["feature"], round(r["importance"], 4)) for r in result[:5]],
    )
    return result


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_and_select(
    df: pd.DataFrame, test_size: float = TEST_SIZE, tuning: bool = True
) -> dict:
    """
    Run the full training pipeline and persist all artifacts.

    Steps:
    1. Chronological train/test split.
    2. Train LinearRegression as the Step 4 baseline and log its metrics.
    3. Train all Step 5 advanced models with TimeSeriesSplit CV.
    4. Tune the best tree model with RandomizedSearchCV.
    5. Evaluate all models on the held-out test set.
    6. Select the best model by holdout R².
    7. Save: model pkl, metrics CSV, CV results CSV, metadata JSON, feature plot.

    Args:
        df: Feature-engineered DataFrame.
        test_size: Fraction of data reserved for the final holdout test.
        tuning: Whether to run RandomizedSearchCV. Set False for quick runs.

    Returns:
        Dict with keys: best_model_name, best_pipeline, metrics_df, cv_df, metadata.
    """
    # --- Validate inputs ---
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing from the DataFrame.")
    if "departure_datetime" not in df.columns:
        raise ValueError(
            "'departure_datetime' column is required for chronological splitting."
        )

    # Guard against leakage — fail loudly before any training happens.
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    leaked = [c for c in LEAKAGE_COLS if c in feature_cols]
    if leaked:
        raise ValueError(f"Leakage columns detected in features: {leaked}")

    train_df, test_df = _chronological_split(df, test_size)

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    preprocessor = build_preprocessor(feature_cols)
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    cv_rows = []
    metric_rows = []
    fitted_pipes = {}

    all_candidates = _get_all_candidates()

    # --- Train and evaluate all models ---
    for name, estimator in all_candidates.items():
        is_baseline = name == "LinearRegression"
        label = "BASELINE" if is_baseline else "model"
        logger.info("Evaluating %s: %s", label, name)

        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=tscv,
            scoring={
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            n_jobs=-1,
            error_score="raise",
        )

        cv_rows.append(
            {
                "model": name,
                "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
                "cv_r2_std": float(np.std(cv_scores["test_r2"])),
                "cv_mae_mean": float(-np.mean(cv_scores["test_mae"])),
                "cv_rmse_mean": float(-np.mean(cv_scores["test_rmse"])),
                "tuned": False,
            }
        )

        pipe.fit(X_train, y_train)
        fitted_pipes[name] = pipe

        m = compute_metrics(y_test, pipe.predict(X_test))
        metric_rows.append({"model": name, **m, "is_tuned": False})

        logger.info(
            "%-25s CV R²=%.4f ± %.4f | Holdout R²=%.4f | MAE=%9.0f | RMSE=%9.0f%s",
            name,
            cv_rows[-1]["cv_r2_mean"],
            cv_rows[-1]["cv_r2_std"],
            m["r2"],
            m["mae"],
            m["rmse"],
            "  ← baseline" if is_baseline else "",
        )

    # --- Pick the best tree model (by CV R²) for tuning ---
    tree_names = ["DecisionTree", "RandomForest", "HistGradientBoosting"]
    best_tree = max(
        (r for r in cv_rows if r["model"] in tree_names),
        key=lambda r: r["cv_r2_mean"],
    )["model"]

    tuned_pipe, tuned_name = None, None

    if tuning:
        logger.info("Tuning '%s' with RandomizedSearchCV ...", best_tree)
        pipe_to_tune = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", _get_advanced_candidates()[best_tree]),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipe_to_tune,
            param_distributions=_tuning_space(best_tree),
            n_iter=12,
            scoring="r2",
            cv=tscv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        tuned_pipe = search.best_estimator_
        tuned_name = f"{best_tree}_tuned"

        m_tuned = compute_metrics(y_test, tuned_pipe.predict(X_test))
        cv_rows.append(
            {
                "model": tuned_name,
                "cv_r2_mean": float(search.best_score_),
                "cv_r2_std": float("nan"),
                "cv_mae_mean": float("nan"),
                "cv_rmse_mean": float("nan"),
                "tuned": True,
            }
        )
        metric_rows.append({"model": tuned_name, **m_tuned, "is_tuned": True})

        logger.info(
            "%-25s best CV R²=%.4f | Holdout R²=%.4f | MAE=%9.0f | RMSE=%9.0f",
            tuned_name,
            search.best_score_,
            m_tuned["r2"],
            m_tuned["mae"],
            m_tuned["rmse"],
        )
        logger.info("Best hyperparameters: %s", search.best_params_)

    # --- Select the best overall model by holdout R² ---
    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values("r2", ascending=False)
        .reset_index(drop=True)
    )
    cv_df = (
        pd.DataFrame(cv_rows)
        .sort_values("cv_r2_mean", ascending=False)
        .reset_index(drop=True)
    )

    best_name = metrics_df.iloc[0]["model"]
    if tuned_name and best_name == tuned_name:
        best_pipeline = tuned_pipe
    else:
        base = best_name.replace("_tuned", "")
        best_pipeline = fitted_pipes[base]

    # --- Feature importance plot (no X_test needed — uses fitted pipeline only) ---
    top_features = _plot_feature_importance(best_pipeline, top_n=15)

    # --- Build route frequency map for inference ---
    route_freq_map = (
        df["route"].value_counts(dropna=False).to_dict()
        if "route" in df.columns
        else {}
    )

    # --- Persist all artifacts ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, PIPELINE_PATH)
    metrics_df.to_csv(METRICS_PATH, index=False)
    cv_df.to_csv(CV_RESULTS_PATH, index=False)

    metadata = {
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "target_column": TARGET_COL,
        "leakage_cols_excluded": list(LEAKAGE_COLS),
        "feature_cols": feature_cols,
        "categorical_features": [f for f in CATEGORICAL_FEATURES if f in feature_cols],
        "numerical_features": [f for f in NUMERICAL_FEATURES if f in feature_cols],
        "test_size": test_size,
        "cv_folds": CV_FOLDS,
        "top_features": top_features,
        "route_freq_map": {k: int(v) for k, v in list(route_freq_map.items())[:500]},
    }
    MODEL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("=" * 55)
    logger.info(
        "Best model: '%s' | Holdout R²=%.4f", best_name, metrics_df.iloc[0]["r2"]
    )
    logger.info(
        "Baseline LinearRegression R²=%.4f (for comparison)",
        next(r["r2"] for r in metric_rows if r["model"] == "LinearRegression"),
    )
    logger.info("Artifacts saved → model: %s", PIPELINE_PATH)
    logger.info("=" * 55)

    return {
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "metrics_df": metrics_df,
        "cv_df": cv_df,
        "metadata": metadata,
    }
