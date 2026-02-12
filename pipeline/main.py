"""
Main pipeline script for the Flight Fare Prediction project.
Orchestrates the entire ML pipeline from data loading to model deployment.
"""

import sys

sys.path.append(".")

from src.data_loader import inspect_data, load_data
from src.data_preprocessing import clean_data, encode_and_scale, feature_engineering
from src.eda import create_visualizations, descriptive_statistics, kpi_exploration
from src.modeling import (
    compare_models,
    evaluate_model,
    plot_actual_vs_predicted,
    plot_feature_importance,
    prepare_data_for_modeling,
    save_model,
    train_advanced_models,
    train_baseline_model,
)
from src.utils import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    """
    Main function to run the entire ML pipeline.
    """
    logger.info("Starting Flight Fare Prediction Pipeline")
    setup_logging()  # Set up logging configuration

    # Step 1: Load and inspect data
    logger.info("Step 1: Loading and inspecting data")
    df = load_data()

    inspect_data(df)

    # Step 2: Clean and preprocess data
    logger.info("Step 2: Cleaning and preprocessing data")
    df_cleaned = clean_data(df)
    df_featured = feature_engineering(df_cleaned)

    # Step 3: Exploratory Data Analysis (before encoding)
    logger.info("Step 3: Performing Exploratory Data Analysis")
    descriptive_statistics(df_featured)
    create_visualizations(df_featured)
    kpi_exploration(df_featured)

    df_processed, scaler, encoders = encode_and_scale(df_featured)

    # Step 4: Model Development (Baseline)
    logger.info("Step 4: Training baseline model")
    X_train, X_test, y_train, y_test, feature_names = prepare_data_for_modeling(
        df_processed
    )
    baseline_model = train_baseline_model(X_train, y_train)
    baseline_metrics = evaluate_model(
        baseline_model, X_test, y_test, "Baseline Linear Regression"
    )
    plot_actual_vs_predicted(
        y_test, baseline_model.predict(X_test), "Baseline Linear Regression"
    )

    # Step 5: Advanced Modeling
    logger.info("Step 5: Training advanced models")
    advanced_models = train_advanced_models(X_train, y_train)
    comparison_df = compare_models(advanced_models, X_test, y_test)

    # Select best model (based on R2 score)
    best_model_name = comparison_df.loc[comparison_df["R2"].idxmax(), "Model"]
    best_model = advanced_models[best_model_name]

    logger.info(f"Best model: {best_model_name}")

    # Step 6: Model Interpretation
    logger.info("Step 6: Model interpretation and insights")
    plot_feature_importance(best_model, feature_names, best_model_name)

    # Save all trained models
    logger.info("Saving all trained models...")
    save_model(baseline_model, "Baseline Linear Regression")
    for model_name, model in advanced_models.items():
        save_model(model, model_name)

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
