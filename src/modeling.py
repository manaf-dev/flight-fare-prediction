"""
Modeling module for the Flight Fare Prediction project.
Handles model training, evaluation, and optimization.
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.config import (
    CV_FOLDS,
    MODELS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    VISUALIZATIONS_PATH,
)
from src.utils import get_logger

logger = get_logger(__name__)


def prepare_data_for_modeling(df):
    """
    Prepares data for modeling by splitting into features and target.

    Args:
        df (pd.DataFrame): The processed dataset.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    target = "total_fare_bdt"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(
        f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_baseline_model(X_train, y_train):
    """
    Trains a baseline Linear Regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        LinearRegression: Trained model.
    """
    logger.info("Training baseline Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Baseline model trained.")
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a model and logs metrics.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        model_name (str): Name of the model.

    Returns:
        dict: Evaluation metrics.
    """
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"{model_name} Evaluation Metrics:")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")

    return {"R2": r2, "MAE": mae, "RMSE": rmse}


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """
    Plots actual vs predicted values.

    Args:
        y_test (pd.Series): Actual values.
        y_pred (np.array): Predicted values.
        model_name (str): Name of the model.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Total Fare (BDT)")
    plt.ylabel("Predicted Total Fare (BDT)")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.savefig(
        f"{VISUALIZATIONS_PATH}/actual_vs_predicted_{model_name.replace(' ', '_')}.png"
    )
    plt.close()
    logger.info(f"Actual vs predicted plot saved for {model_name}")


def train_advanced_models(X_train, y_train):
    """
    Trains advanced regression models with Randomized Search for efficiency.
    """
    logger.info("Training advanced models using Randomized Search...")

    models = {}

    def fit_with_random_search(
        model_name, estimator, params, n_iter=20, suppress_conv_warning=False
    ):
        logger.info(f"Starting training for {model_name}...")

        # RandomizedSearchCV samples 'n_iter' combinations instead of all of them
        search = RandomizedSearchCV(
            estimator,
            param_distributions=params,
            n_iter=n_iter,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            error_score="raise",
            verbose=2,
        )

        try:
            if suppress_conv_warning:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    search.fit(X_train, y_train)
            else:
                search.fit(X_train, y_train)
        except Exception as exc:
            logger.exception(f"{model_name} training failed: {exc}")
            return

        models[model_name] = search.best_estimator_
        logger.info(f"{model_name} best params: {search.best_params_}")

    # Ridge & Lasso (Keep these small as they are fast anyway)
    fit_with_random_search(
        "Ridge Regression",
        Ridge(random_state=RANDOM_STATE),
        {"alpha": [0.1, 1.0, 10.0]},
        n_iter=3,
    )

    fit_with_random_search(
        "Lasso Regression",
        Lasso(random_state=RANDOM_STATE, max_iter=10000),
        {"alpha": [0.01, 0.1, 1.0]},
        n_iter=3,
        suppress_conv_warning=True,
    )

    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    fit_with_random_search(
        "Random Forest",
        RandomForestRegressor(random_state=RANDOM_STATE),
        rf_params,
        n_iter=15,
    )

    # Gradient Boosting
    gb_params = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5],
    }
    fit_with_random_search(
        "Gradient Boosting",
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        gb_params,
        n_iter=10,
    )

    if not models:
        raise RuntimeError("No advanced models were successfully trained.")

    return models


def compare_models(models, X_test, y_test):
    """
    Compares multiple models and returns a comparison DataFrame.

    Args:
        models (dict): Dictionary of trained models.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        pd.DataFrame: Model comparison results.
    """
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics["Model"] = name
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[["Model", "R2", "MAE", "RMSE"]]
    logger.info("Model comparison completed.")
    logger.info(comparison_df.to_string())
    return comparison_df


def plot_feature_importance(model, feature_names, model_name):
    """
    Plots feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.
        model_name (str): Name of the model.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importances - {model_name}")
        plt.bar(range(len(feature_names)), importances[indices], align="center")
        plt.xticks(
            range(len(feature_names)), [feature_names[i] for i in indices], rotation=90
        )
        plt.tight_layout()
        plt.savefig(
            f"{VISUALIZATIONS_PATH}/feature_importance_{model_name.replace(' ', '_')}.png"
        )
        plt.close()
        logger.info(f"Feature importance plot saved for {model_name}")
    else:
        logger.info(f"Model {model_name} does not have feature_importances_ attribute")


def save_model(model, model_name):
    """
    Saves the trained model to disk.

    Args:
        model: Trained model.
        model_name (str): Name of the model.
    """
    filename = f"{model_name.replace(' ', '_')}.pkl"
    filepath = f"{MODELS_PATH}/{filename}"
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
