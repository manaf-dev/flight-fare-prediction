"""
Inference module for Flight Fare Prediction.
Handles loading preprocessors and making predictions with the trained model.
"""

import os

import joblib
import pandas as pd

from src.config import MODELS_PATH

def load_inference_artifacts():
    """
    Load trained model and preprocessors used for inference.

    Returns:
        tuple: (model, scaler)
    """
    try:
        model_path = os.path.join(MODELS_PATH, "Gradient_Boosting.pkl")
        scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        raise Exception(f"Error loading inference artifacts: {e}")


def preprocess_input(input_data, model, scaler):
    """
    Preprocess input data into the exact feature schema expected by the trained model.

    Args:
        input_data (dict): Raw input data from user
        model: Trained model with feature_names_in_
        scaler: Trained scaler for numerical features

    Returns:
        pd.DataFrame: Processed single-row feature frame ready for prediction
    """
    try:
        # Convert to DataFrame and engineer fields used during training
        df = pd.DataFrame([input_data])
        df = feature_engineering(df)

        if not hasattr(model, "feature_names_in_"):
            raise ValueError("Model does not expose feature_names_in_.")

        expected_columns = list(model.feature_names_in_)
        processed = pd.DataFrame(0.0, index=[0], columns=expected_columns)

        # Scale numeric features with the trained scaler
        numeric_columns = list(getattr(scaler, "feature_names_in_", []))
        if numeric_columns:
            processed.loc[:, numeric_columns] = scaler.transform(df[numeric_columns])

        # Populate categorical one-hot features directly from expected model columns
        categorical_prefixes = [
            "airline",
            "source",
            "destination",
            "stopovers",
            "aircraft_type",
            "class",
            "booking_source",
            "seasonality",
            "season",
        ]

        for prefix in categorical_prefixes:
            if prefix not in df.columns:
                continue
            value = str(df.iloc[0][prefix])
            one_hot_col = f"{prefix}_{value}"
            if one_hot_col in processed.columns:
                processed.at[0, one_hot_col] = 1.0

        return processed

    except Exception as e:
        raise Exception(f"Error preprocessing input: {e}")

def predict_fare(input_data):
    """
    Make fare prediction for given input data.

    Args:
        input_data (dict): Flight details

    Returns:
        float: Predicted fare
    """
    try:
        # Load model and preprocessors
        model, scaler = load_inference_artifacts()

        # Preprocess input
        processed_input = preprocess_input(input_data, model, scaler)

        # Final safety alignment: enforce exact feature schema expected by model.
        expected_columns = list(getattr(model, "feature_names_in_", []))
        if expected_columns:
            if "total_fare_bdt" in processed_input.columns and "total_fare_bdt" not in expected_columns:
                processed_input = processed_input.drop(columns=["total_fare_bdt"])
            processed_input = processed_input.reindex(columns=expected_columns, fill_value=0.0)

        # Make prediction
        prediction = model.predict(processed_input)[0]

        return prediction

    except Exception as e:
        raise Exception(f"Error making prediction: {e}")

def feature_engineering(df):
    """
    Apply feature engineering to input data.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    df = df.copy()

    # Convert date to datetime
    df['departure_date_and_time'] = pd.to_datetime(df['departure_date_and_time'])

    # Extract date features
    df['departure_month'] = df['departure_date_and_time'].dt.month
    df['departure_day'] = df['departure_date_and_time'].dt.day
    df['departure_hour'] = df['departure_date_and_time'].dt.hour
    df['departure_weekday'] = df['departure_date_and_time'].dt.weekday

    # Create season feature
    def get_season(month):
        if month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"
        else:
            return "Winter"

    df["season"] = df["departure_month"].apply(get_season)

    # Recreate seasonality values used by training data
    def get_seasonality(month):
        if month in [11, 12, 1]:
            return "Winter Holidays"
        elif month in [4, 5]:
            return "Eid"
        elif month in [6, 7, 8]:
            return "Hajj"
        else:
            return "Regular"

    df["seasonality"] = df["departure_month"].apply(get_seasonality)

    # Ensure numerical fields are numeric for scaler.transform
    numeric_cols = [
        "duration_hrs",
        "base_fare_bdt",
        "tax_and_surcharge_bdt",
        "days_before_departure",
        "departure_month",
        "departure_day",
        "departure_hour",
        "departure_weekday",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
