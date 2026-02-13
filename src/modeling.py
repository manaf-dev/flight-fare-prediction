"""Model training, tuning, evaluation, and artifact generation."""

import json
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    CV_RESULTS_PATH,
    LEAKAGE_COLS,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    NUMERICAL_FEATURES,
    PIPELINE_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    TOP_FEATURES_PLOT_PATH,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _metric_dict(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
    }


def _build_preprocessor(feature_cols):
    categorical = [f for f in CATEGORICAL_FEATURES if f in feature_cols]
    numerical = [f for f in NUMERICAL_FEATURES if f in feature_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, categorical, numerical


def _candidate_models():
    return {
        "Ridge": Ridge(),
        "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=20000),
        "RandomForest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            n_estimators=140,
            max_depth=16,
            max_features="sqrt",
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def _tuning_space(model_name):
    if model_name == "RandomForest":
        return {
            "model__n_estimators": randint(180, 450),
            "model__max_depth": [None, 10, 14, 18, 24],
            "model__min_samples_split": randint(2, 14),
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": ["sqrt", "log2", None],
        }
    if model_name == "HistGradientBoosting":
        return {
            "model__max_depth": [None, 6, 8, 12],
            "model__learning_rate": uniform(0.01, 0.2),
            "model__max_iter": randint(120, 450),
            "model__l2_regularization": uniform(0.0, 1.5),
        }
    raise ValueError(f"No tuning space for model: {model_name}")


def _plot_top_feature_importance(best_pipeline, X_ref, y_ref, top_n=15):
    model = best_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = model.feature_importances_
    else:
        perm = permutation_importance(
            best_pipeline,
            X_ref,
            y_ref,
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring="r2",
        )
        feature_names = np.array(X_ref.columns)
        importances = perm.importances_mean

    order = np.argsort(importances)[::-1][:top_n]

    top_names = [str(feature_names[i]) for i in order]
    top_values = [float(importances[i]) for i in order]

    plt.figure(figsize=(12, 7))
    plt.barh(range(len(order)), top_values[::-1])
    plt.yticks(range(len(order)), top_names[::-1])
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(TOP_FEATURES_PLOT_PATH)
    plt.close()
    logger.info("Saved top feature importance plot to %s", TOP_FEATURES_PLOT_PATH)

    return [{"feature": n, "importance": v} for n, v in zip(top_names, top_values)]


def _time_split(df, test_size=TEST_SIZE):
    sorted_df = df.sort_values("departure_datetime").reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1 - test_size))
    split_idx = max(1, min(split_idx, len(sorted_df) - 1))

    train_df = sorted_df.iloc[:split_idx].copy()
    test_df = sorted_df.iloc[split_idx:].copy()

    assert train_df["departure_datetime"].max() <= test_df["departure_datetime"].min(), (
        "Chronological split violated: train has timestamps after test start."
    )
    return train_df, test_df


def train_and_select_model(df, test_size=TEST_SIZE, tuning=True):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    if "departure_datetime" not in df.columns:
        raise ValueError("Missing required datetime column: departure_datetime")

    train_df, test_df = _time_split(df, test_size=test_size)

    feature_cols = [c for c in CATEGORICAL_FEATURES + NUMERICAL_FEATURES if c in df.columns]

    assert TARGET_COL not in feature_cols, "Target leaked into features"
    leakage_in_features = [c for c in LEAKAGE_COLS if c in feature_cols]
    assert not leakage_in_features, f"Leakage features found: {leakage_in_features}"

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    preprocessor, categorical_cols, numerical_cols = _build_preprocessor(feature_cols)
    logger.info("Training rows: %s | Test rows: %s", len(X_train), len(X_test))
    logger.info("Categorical features used: %s", categorical_cols)
    logger.info("Numerical features used: %s", numerical_cols)

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    models = _candidate_models()

    cv_rows = []
    metrics_rows = []
    fitted_pipelines = {}

    for name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

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

        cv_row = {
            "model": name,
            "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
            "cv_r2_std": float(np.std(cv_scores["test_r2"])),
            "cv_mae_mean": float(-np.mean(cv_scores["test_mae"])),
            "cv_rmse_mean": float(-np.mean(cv_scores["test_rmse"])),
            "tuned": False,
        }
        cv_rows.append(cv_row)

        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        holdout = _metric_dict(y_test, pipe.predict(X_test))
        metrics_rows.append(
            {
                "model": name,
                "r2": holdout["R2"],
                "mae": holdout["MAE"],
                "rmse": holdout["RMSE"],
                "is_tuned": False,
            }
        )

        logger.info(
            "%s | CV R2=%.4f, Holdout R2=%.4f, MAE=%.2f, RMSE=%.2f",
            name,
            cv_row["cv_r2_mean"],
            holdout["R2"],
            holdout["MAE"],
            holdout["RMSE"],
        )

    tree_models = ["RandomForest", "HistGradientBoosting"]
    best_tree = max(
        (row for row in cv_rows if row["model"] in tree_models),
        key=lambda row: row["cv_r2_mean"],
    )["model"]

    best_tuned_pipeline = None
    best_tuned_name = None

    if tuning:
        logger.info("Tuning best tree model selected by CV: %s", best_tree)
        tuned_pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", _candidate_models()[best_tree])]
        )

        random_search = RandomizedSearchCV(
            estimator=tuned_pipe,
            param_distributions=_tuning_space(best_tree),
            n_iter=12,
            scoring="r2",
            cv=tscv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
            error_score="raise",
        )
        random_search.fit(X_train, y_train)

        best_tuned_pipeline = random_search.best_estimator_
        best_tuned_name = f"{best_tree}_tuned"

        tuned_holdout = _metric_dict(y_test, best_tuned_pipeline.predict(X_test))

        cv_rows.append(
            {
                "model": best_tuned_name,
                "cv_r2_mean": float(random_search.best_score_),
                "cv_r2_std": np.nan,
                "cv_mae_mean": np.nan,
                "cv_rmse_mean": np.nan,
                "tuned": True,
            }
        )
        metrics_rows.append(
            {
                "model": best_tuned_name,
                "r2": tuned_holdout["R2"],
                "mae": tuned_holdout["MAE"],
                "rmse": tuned_holdout["RMSE"],
                "is_tuned": True,
            }
        )

        logger.info("Tuned %s best params: %s", best_tree, random_search.best_params_)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("r2", ascending=False).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)

    best_row = metrics_df.iloc[0]
    best_name = best_row["model"]

    if best_tuned_name and best_name == best_tuned_name:
        best_pipeline = best_tuned_pipeline
    else:
        base_name = best_name.replace("_tuned", "")
        best_pipeline = fitted_pipelines[base_name]

    top_features = _plot_top_feature_importance(best_pipeline, X_test, y_test, top_n=15)

    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CV_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, PIPELINE_PATH)
    metrics_df.to_csv(METRICS_PATH, index=False)
    cv_df.to_csv(CV_RESULTS_PATH, index=False)

    metadata = {
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
        "target_column": TARGET_COL,
        "expected_input_schema": feature_cols,
        "categorical_features": categorical_cols,
        "numerical_features": numerical_cols,
        "leakage_columns_excluded": LEAKAGE_COLS,
        "best_model": best_name,
        "test_size": test_size,
        "top_features": top_features,
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Saved best pipeline to %s", PIPELINE_PATH)
    logger.info("Saved metrics to %s and CV results to %s", METRICS_PATH, CV_RESULTS_PATH)

    return {
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "metrics_df": metrics_df,
        "cv_df": cv_df,
        "metadata": metadata,
    }


def write_model_report(metrics_df, metadata, output_path):
    """Write leakage-aware interpretation summary for reporting."""
    rows = []
    for _, row in metrics_df.iterrows():
        rows.append(f"| {row['model']} | {row['r2']:.4f} | {row['mae']:.2f} | {row['rmse']:.2f} |")

    top_lines = []
    for item in metadata.get("top_features", [])[:10]:
        pretty = item["feature"].replace("cat__", "").replace("num__", "")
        top_lines.append(f"- {pretty}: {item['importance']:.4f}")

    content = "\n".join(
        [
            "# Model Interpretation & Insights",
            "",
            "## Leakage Note",
            (
                "The target is `total_fare_bdt`. `base_fare_bdt` and `tax_and_surcharge_bdt` were "
                "excluded from model predictors to prevent target leakage, because total fare is mostly "
                "their arithmetic sum."
            ),
            "",
            "## Model Comparison (Holdout Test)",
            "| Model | R2 | MAE | RMSE |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            "## Best Model Rationale",
            f"Best model: `{metadata.get('best_model', 'N/A')}` selected by highest holdout R2 with leakage-free, time-aware validation.",
            "",
            "## Top Drivers",
            "Primary drivers include route, days_before_departure, seasonality, class, airline, stopovers, and departure-time features.",
            *top_lines,
            "",
            "## Operational Notes",
            "- Validation uses chronological split (first 80% train, last 20% test).",
            "- Cross-validation uses TimeSeriesSplit on the train period.",
            "- Inference uses a single saved pipeline artifact to avoid train/serve skew.",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info("Updated report: %s", output_path)
