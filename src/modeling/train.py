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
    VIZ_DIR,
)
from src.modeling.evaluate import compute_metrics
from src.modeling.preprocess import build_preprocessor
from src.utils import get_logger

logger = get_logger(__name__)


def _get_baseline() -> dict:
    """
    Step 4 baseline: plain LinearRegression.
    No regularisation, no hyperparameters — serves as the floor all other
    models must beat.
    """
    return {"LinearRegression": LinearRegression()}


def _get_advanced_candidates() -> dict:
    """Step 5 advanced models, ordered from simplest to most complex."""
    return {
        "Ridge": Ridge(),
        "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=20_000),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=10,  # constrained to prevent extreme overfitting
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
    """Baseline first, then advanced — order preserved in Python 3.7+ dicts."""
    return {**_get_baseline(), **_get_advanced_candidates()}


def _tuning_space(model_name: str) -> dict:
    """Hyperparameter search spaces for RandomizedSearchCV (tree models only)."""
    spaces = {
        "DecisionTree": {
            "model__max_depth": [5, 8, 10, 15, None],
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
        },
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
        raise ValueError(f"No tuning space defined for '{model_name}'.")
    return spaces[model_name]


def _chronological_split(df: pd.DataFrame, test_size: float):
    """
    Sort by departure_datetime; reserve the last test_size fraction as test.
    Asserts temporal ordering is respected.
    """
    sorted_df = df.sort_values("departure_datetime").reset_index(drop=True)
    cut = max(1, min(int(len(sorted_df) * (1 - test_size)), len(sorted_df) - 1))
    train, test = sorted_df.iloc[:cut].copy(), sorted_df.iloc[cut:].copy()

    assert train["departure_datetime"].max() <= test["departure_datetime"].min(), (
        "Chronological split violated: training data has timestamps after test start."
    )
    logger.info(
        "Chronological split — train: %d rows | test: %d rows | "
        "train ends: %s | test starts: %s",
        len(train),
        len(test),
        train["departure_datetime"].max().date(),
        test["departure_datetime"].min().date(),
    )
    return train, test


def _plot_actual_vs_predicted(y_test, y_pred, model_name: str) -> None:
    """
    Scatter plot of actual vs predicted fares for the best model.
    Step 4: 'Visualize actual vs. predicted values.'
    Perfect predictions lie on the diagonal line.
    """
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    sample_size = min(2000, len(y_test))
    idx = np.random.default_rng(42).choice(len(y_test), size=sample_size, replace=False)
    y_actual = np.array(y_test)[idx]
    y_hat = np.array(y_pred)[idx]

    ax.scatter(y_actual, y_hat, alpha=0.3, s=12, color="steelblue", label="Predictions")

    # Perfect-prediction diagonal
    lo, hi = min(y_actual.min(), y_hat.min()), max(y_actual.max(), y_hat.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual Total Fare (BDT)")
    ax.set_ylabel("Predicted Total Fare (BDT)")
    ax.set_title(f"Actual vs Predicted Fare — {model_name}", fontsize=13)
    ax.legend()
    plt.tight_layout()

    path = VIZ_DIR / "actual_vs_predicted.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved actual vs predicted plot: %s", path)


def _plot_residuals(y_test, y_pred, model_name: str) -> None:
    """
    Residual plot: predicted values on x-axis, residuals (actual - predicted) on y-axis.
    Step 4: 'Analyze residuals to detect underfitting or overfitting.'

    What to look for:
    - Random scatter around y=0 → well-fitted model
    - Funnel shape (heteroscedasticity) → variance increases with fare level
    - Curve pattern → model is missing non-linear structure
    """
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    residuals = np.array(y_test) - np.array(y_pred)
    sample_size = min(2000, len(residuals))
    idx = np.random.default_rng(42).choice(
        len(residuals), size=sample_size, replace=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Residuals vs Predicted
    axes[0].scatter(
        np.array(y_pred)[idx], residuals[idx], alpha=0.3, s=12, color="steelblue"
    )
    axes[0].axhline(0, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted Total Fare (BDT)")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    axes[0].set_title("Residuals vs Predicted")

    # Right: Residual distribution
    axes[1].hist(residuals, bins=60, edgecolor="white", alpha=0.85, color="steelblue")
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Residual (BDT)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=14)
    plt.tight_layout()

    path = VIZ_DIR / "residual_analysis.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved residual analysis plot: %s", path)


def _plot_bias_variance(cv_rows: list) -> None:
    """
    Bar chart showing CV R² mean ± std for each model.
    Step 5: 'Plot bias-variance tradeoff.'

    Interpretation:
    - High mean + low std → good generaliser (low bias, low variance)
    - Low mean → high bias (underfitting)
    - High std → high variance (overfitting / unstable)
    """
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    rows = [r for r in cv_rows if not r.get("tuned", False)]
    names = [r["model"] for r in rows]
    means = [r["cv_r2_mean"] for r in rows]
    stds = [r["cv_r2_std"] for r in rows]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color="steelblue",
        error_kw={"elinewidth": 2, "ecolor": "tomato"},
        alpha=0.85,
    )

    # Colour the baseline differently
    if "LinearRegression" in names:
        bars[names.index("LinearRegression")].set_color("grey")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Cross-Validated R²")
    ax.set_title(
        "Bias-Variance Tradeoff — CV R² Mean ± Std\n"
        "(Grey bar = LinearRegression baseline | Error bars show variance across CV folds)",
        fontsize=12,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()

    path = VIZ_DIR / "bias_variance_tradeoff.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved bias-variance tradeoff plot: %s", path)


# Feature importance / linear coefficient interpretation


def _plot_feature_importance(pipeline: Pipeline, top_n: int = 15) -> list:
    """
    Extract, plot, and save the top-N feature importances from the fitted pipeline.

    Tree models  → native feature_importances_ (mean decrease in impurity)
    Linear models → absolute coefficient magnitude (|coef|)

    Returns list of {"feature": name, "importance": value} dicts for metadata.
    """
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        raw_names = list(preprocessor.get_feature_names_out())
    except Exception as exc:
        logger.warning("Could not get feature names from preprocessor: %s", exc)
        return []

    # Strip sklearn prefix: "num__duration_hrs" → "duration_hrs"
    names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_.tolist()
        importance_label = "Feature Importance (mean decrease in impurity)"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).tolist()
        importance_label = "Absolute Coefficient Magnitude"
    else:
        logger.warning(
            "Model '%s' exposes neither feature_importances_ nor coef_.",
            type(model).__name__,
        )
        return []

    if len(names) != len(values):
        logger.warning(
            "Name/value length mismatch: %d names, %d values. Skipping plot.",
            len(names),
            len(values),
        )
        return []

    actual_top = min(top_n, len(values))
    order = np.argsort(values)[::-1][:actual_top]
    top_names = [names[i] for i in order]
    top_values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(5, actual_top * 0.5)))
    ax.barh(range(actual_top), top_values[::-1], color="steelblue")
    ax.set_yticks(range(actual_top))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel(importance_label)
    ax.set_title(
        f"Top {actual_top} Feature Importances — {type(model).__name__}", fontsize=13
    )
    plt.tight_layout()

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


def _save_linear_coefficients(fitted_pipes: dict, feature_cols: list) -> None:
    """
    Step 6: 'For linear models: Examine coefficients.'

    Save a CSV of LinearRegression, Ridge, and Lasso coefficients so they
    can be inspected and discussed. Coefficients are sorted by absolute magnitude.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    linear_models = ["LinearRegression", "Ridge", "Lasso"]

    for name in linear_models:
        if name not in fitted_pipes:
            continue
        pipe = fitted_pipes[name]
        model = pipe.named_steps["model"]
        preprocessor = pipe.named_steps["preprocessor"]

        if not hasattr(model, "coef_"):
            continue

        try:
            raw_names = list(preprocessor.get_feature_names_out())
            feat_names = [
                n.replace("num__", "").replace("cat__", "") for n in raw_names
            ]
            coefs = model.coef_.tolist()

            if len(feat_names) != len(coefs):
                continue

            coef_df = (
                pd.DataFrame({"feature": feat_names, "coefficient": coefs})
                .assign(abs_coef=lambda x: x["coefficient"].abs())
                .sort_values("abs_coef", ascending=False)
                .drop(columns="abs_coef")
            )
            path = REPORTS_DIR / f"linear_coefficients_{name.lower()}.csv"
            coef_df.to_csv(path, index=False)
            logger.info("Saved %s coefficients → %s", name, path)
        except Exception as exc:
            logger.warning("Could not extract coefficients for %s: %s", name, exc)


# Main training function
def train_and_select(
    df: pd.DataFrame, test_size: float = TEST_SIZE, tuning: bool = True
) -> dict:
    """
    Run the full training pipeline and persist all artifacts.

    Args:
        df: Feature-engineered DataFrame from src.features.build.build().
        test_size: Fraction of data for chronological holdout test.
        tuning: Run RandomizedSearchCV on best tree. False = faster dev run.

    Returns:
        Dict with: best_model_name, best_pipeline, metrics_df, cv_df, metadata.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from DataFrame.")
    if "departure_datetime" not in df.columns:
        raise ValueError("'departure_datetime' required for chronological split.")

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
    all_candidates = _get_all_candidates()

    cv_rows, metric_rows, fitted_pipes = [], [], {}

    # --- Train + evaluate all models ---
    for name, estimator in all_candidates.items():
        is_baseline = name == "LinearRegression"
        logger.info("Evaluating %s: %s", "BASELINE" if is_baseline else "model", name)

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
            "%-25s CV R²=%6.4f ±%5.4f | Holdout R²=%6.4f | MAE=%9.0f | RMSE=%9.0f%s",
            name,
            cv_rows[-1]["cv_r2_mean"],
            cv_rows[-1]["cv_r2_std"],
            m["r2"],
            m["mae"],
            m["rmse"],
            "  ← baseline" if is_baseline else "",
        )

    # --- Tune best tree model ---
    tree_names = ["DecisionTree", "RandomForest", "HistGradientBoosting"]
    best_tree = max(
        (r for r in cv_rows if r["model"] in tree_names),
        key=lambda r: r["cv_r2_mean"],
    )["model"]

    tuned_pipe, tuned_name = None, None

    if tuning:
        logger.info("Tuning '%s' with RandomizedSearchCV ...", best_tree)
        search = RandomizedSearchCV(
            estimator=Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", _get_advanced_candidates()[best_tree]),
                ]
            ),
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
            "%-25s best CV R²=%6.4f | Holdout R²=%6.4f | MAE=%9.0f | RMSE=%9.0f",
            tuned_name,
            search.best_score_,
            m_tuned["r2"],
            m_tuned["mae"],
            m_tuned["rmse"],
        )
        logger.info("Best hyperparameters: %s", search.best_params_)

    # --- Select winner ---
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
        best_pipeline = fitted_pipes[best_name.replace("_tuned", "")]

    # --- Diagnostic plots ---
    best_preds = best_pipeline.predict(X_test)
    _plot_actual_vs_predicted(y_test, best_preds, best_name)
    _plot_residuals(y_test, best_preds, best_name)
    _plot_bias_variance(cv_rows)

    # --- Feature importance + linear coefficients ---
    top_features = _plot_feature_importance(best_pipeline, top_n=15)
    _save_linear_coefficients(fitted_pipes, feature_cols)

    # --- Route frequency map for inference ---
    route_freq_map = (
        df["route"].value_counts(dropna=False).to_dict()
        if "route" in df.columns
        else {}
    )

    # --- Persist artifacts ---
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

    baseline_r2 = next(r["r2"] for r in metric_rows if r["model"] == "LinearRegression")
    logger.info("=" * 55)
    logger.info("Baseline LinearRegression  R² = %.4f", baseline_r2)
    logger.info(
        "Best model: %-20s R² = %.4f  (+%.4f vs baseline)",
        best_name,
        metrics_df.iloc[0]["r2"],
        metrics_df.iloc[0]["r2"] - baseline_r2,
    )
    logger.info("Artifacts saved → %s", PIPELINE_PATH)
    logger.info("=" * 55)

    return {
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "metrics_df": metrics_df,
        "cv_df": cv_df,
        "metadata": metadata,
    }