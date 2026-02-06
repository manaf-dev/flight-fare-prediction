"""
Configuration file for the Flight Fare Prediction project.
Contains paths, constants, and hyperparameters.
"""

import os

# Paths
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "Flight_Price_Dataset_of_Bangladesh.csv",
)
LOGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REPORTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
VISUALIZATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "visualizations"
)

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering
CATEGORICAL_FEATURES = [
    "airline",
    "source",
    "destination",
    "stopovers",
    "aircraft_type",
    "class",
    "booking_source",
    "seasonality",
]
NUMERICAL_FEATURES = [
    "duration_hrs",
    "base_fare_bdt",
    "tax_and_surcharge_bdt",
    "days_before_departure",
]
