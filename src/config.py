"""Configuration file for the Flight Fare Prediction project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "Flight_Price_Dataset_of_Bangladesh.csv"
LOGS_PATH = PROJECT_ROOT / "logs"
MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"
VISUALIZATIONS_PATH = PROJECT_ROOT / "visualizations"

# Core modeling constants
TARGET_COL = "total_fare_bdt"
LEAKAGE_COLS = ["base_fare_bdt", "tax_and_surcharge_bdt"]
DATE_COLS = ["departure_datetime", "arrival_datetime"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Canonical feature groups (leakage removed)
CATEGORICAL_FEATURES = [
    "airline",
    "source",
    "destination",
    "stopovers",
    "aircraft_type",
    "class",
    "booking_source",
    "seasonality",
    "route",
]

NUMERICAL_FEATURES = [
    "duration_hrs",
    "days_before_departure",
    "departure_month",
    "departure_day",
    "departure_hour",
    "departure_weekday",
    "is_weekend",
    "is_peak_hour",
    "route_frequency",
]

# Artifacts
PIPELINE_PATH = MODELS_PATH / "best_model_pipeline.pkl"
METRICS_PATH = REPORTS_PATH / "model_metrics.csv"
CV_RESULTS_PATH = REPORTS_PATH / "cv_results.csv"
MODEL_METADATA_PATH = MODELS_PATH / "model_metadata.json"
TOP_FEATURES_PLOT_PATH = VISUALIZATIONS_PATH / "feature_importance_top15.png"
