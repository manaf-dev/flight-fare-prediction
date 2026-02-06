"""
Modeling module for the Flight Fare Prediction project.
Handles model training, evaluation, and optimization.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

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
    Trains advanced regression models with hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        dict: Dictionary of trained models.
    """
    logger.info("Training advanced models...")

    models = {}

    # Ridge Regression
    ridge_params = {"alpha": [0.1, 1.0, 10.0, 100.0]}
    ridge_grid = GridSearchCV(
        Ridge(random_state=RANDOM_STATE), ridge_params, cv=CV_FOLDS, scoring="r2"
    )
    ridge_grid.fit(X_train, y_train)
    models["Ridge Regression"] = ridge_grid.best_estimator_
    logger.info(f"Ridge best params: {ridge_grid.best_params_}")

    # Lasso Regression
    lasso_params = {"alpha": [0.001, 0.01, 0.1, 1.0]}
    lasso_grid = GridSearchCV(
        Lasso(random_state=RANDOM_STATE, max_iter=10000),
        lasso_params,
        cv=CV_FOLDS,
        scoring="r2",
    )
    lasso_grid.fit(X_train, y_train)
    models["Lasso Regression"] = lasso_grid.best_estimator_
    logger.info(f"Lasso best params: {lasso_grid.best_params_}")

    # Decision Tree
    dt_params = {"max_depth": [10, 20, None], "min_samples_split": [2, 5, 10]}
    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        dt_params,
        cv=CV_FOLDS,
        scoring="r2",
    )
    dt_grid.fit(X_train, y_train)
    models["Decision Tree"] = dt_grid.best_estimator_
    logger.info(f"Decision Tree best params: {dt_grid.best_params_}")

    # Random Forest
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        rf_params,
        cv=CV_FOLDS,
        scoring="r2",
    )
    rf_grid.fit(X_train, y_train)
    models["Random Forest"] = rf_grid.best_estimator_
    logger.info(f"Random Forest best params: {rf_grid.best_params_}")

    # Gradient Boosting
    gb_params = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    }
    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        gb_params,
        cv=CV_FOLDS,
        scoring="r2",
    )
    gb_grid.fit(X_train, y_train)
    models["Gradient Boosting"] = gb_grid.best_estimator_
    logger.info(f"Gradient Boosting best params: {gb_grid.best_params_}")

    logger.info("Advanced models trained.")
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
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
