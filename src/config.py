"""
Single source of truth for all project-wide settings: paths, column names, feature lists, and modelling parameters.
"""

from pathlib import Path

# Root paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VIZ_DIR = PROJECT_ROOT / "visualizations"

# Input dataset
RAW_DATA_PATH = DATA_DIR / "Flight_Price_Dataset_of_Bangladesh.csv"

# Saved artifacts
PIPELINE_PATH = MODELS_DIR / "best_model_pipeline.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.csv"
INTERPRETATION_PATH = REPORTS_DIR / "model_interpretation.md"
FEAT_IMPORTANCE_PLOT = VIZ_DIR / "feature_importance_top15.png"


# Column names

TARGET_COL = "total_fare_bdt"

# These columns directly determine the target (base + tax = total), so they
# must NEVER be used as model inputs — that would be target leakage.
LEAKAGE_COLS = ["base_fare_bdt", "tax_and_surcharge_bdt"]

# Datetime columns that need parsing
DATETIME_COLS = ["departure_datetime", "arrival_datetime"]


# Feature lists

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

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# Modelling parameters


RANDOM_STATE = 42
TEST_SIZE = 0.20  # chronological holdout fraction
CV_FOLDS = 5  # folds for TimeSeriesSplit
